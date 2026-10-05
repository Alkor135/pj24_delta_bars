r"""Выполняет исследование связи дневного объёма и переключений SuperTrend/SMA.

Примеры из корня проекта:
    .\.venv\Scripts\python.exe -m research.volume_trend --stage audit
    .\.venv\Scripts\python.exe -m research.volume_trend --stage analyze --repetitions 5000
    .\.venv\Scripts\python.exe -m research.volume_trend --stage report
    .\.venv\Scripts\python.exe -m research.volume_trend --stage all --symbols RTS MIX

Этап audit сверяет исходники и фиксирует параметры до расчёта индикаторов.
Этап analyze считает точные объёмы в общем окне, события и блочную статистику.
Этап report создаёт Markdown-отчёт и автономные графики Plotly.
ZIP/SQLite только читаются; кэш и результаты записываются в --output.
"""

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import json
import platform
import sys

import numpy as np
import pandas as pd

from research.volume_trend_data import NS, audit_symbol, common_window, time_text, window_volume
from research.volume_trend_metrics import bootstrap_effect, daily_metrics, group_effect, holm, indicators


def save_json(path, value):
    """Записывает JSON value в path с русским текстом и отступами."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def arguments():
    """Читает параметры запуска и возвращает проверенный Namespace."""
    parser = argparse.ArgumentParser(description="Проверка объёма и трендовости RTS/MIX")
    parser.add_argument("--stage", choices=("audit", "analyze", "report", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("C:/data_quote"))
    parser.add_argument("--output", type=Path, default=Path("results/volume_trend"))
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-10-02")
    parser.add_argument("--symbols", choices=("RTS", "MIX"), nargs="+", default=["RTS", "MIX"])
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args()
    args.data_dir, args.output = args.data_dir.resolve(), args.output.resolve()
    if pd.Timestamp(args.start) > pd.Timestamp(args.end) or args.repetitions < 100:
        parser.error("Неверный диапазон дат или меньше 100 bootstrap-повторов")
    return args


def audit(args):
    """Выполняет аудит args, фиксирует конфигурацию и окно по границам без индикаторов."""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cache").mkdir(exist_ok=True)
    config = {"created": datetime.now().isoformat(timespec="seconds"), "data_dir": str(args.data_dir), "start": args.start, "end": args.end, "symbols": args.symbols,
              "lookback": 20, "atr_period": 10, "multiplier": 3, "sma": [3, 34], "warmup_bars": 100, "weekends": "исключены", "bounds_inclusive": True,
              "clearing": "не исключается", "repetitions": args.repetitions, "block": 20, "block_checks": [40, 60], "seed": 20261005,
              "fixed_window_rule": "95%-квантиль первых и 5%-квантиль последних сделок, округление внутрь до 5 минут; общее для инструментов",
              "gap_reset_calendar_days": 10, "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    save_json(args.output / "config.json", config)
    parts = [audit_symbol(args.data_dir, args.output, symbol, args.start, args.end) for symbol in args.symbols]
    all_days = pd.concat(parts, ignore_index=True)
    firsts, lasts = [], []
    missing = []
    for symbol, part in zip(args.symbols, parts):
        good = part[part.valid & part.weekday.lt(5) & part.day.ge(args.start)]
        firsts.append(float(good.first_ns.quantile(.95)))
        lasts.append(float(good.last_ns.quantile(.05)))
        present = set(part.day)
        for day in pd.bdate_range(args.start, args.end):
            if day.date().isoformat() not in present:
                missing.append({"symbol": symbol, "day": day.date().isoformat(), "status": "нет архива; праздник/пропуск не различены"})
    width = 300 * NS
    fixed_start = int(np.ceil(max(firsts)/width)*width)
    fixed_end = int(np.floor(min(lasts)/width)*width)
    config.update(fixed_start_ns=fixed_start, fixed_end_ns=fixed_end, fixed_start=time_text(fixed_start), fixed_end=time_text(fixed_end))
    if fixed_start >= fixed_end:
        raise ValueError("Фиксированное окно не имеет положительной длительности")
    save_json(args.output / "config.json", config)
    all_days.to_csv(args.output / "audit.csv", index=False)
    pd.DataFrame(missing, columns=["symbol", "day", "status"]).to_csv(args.output / "missing_weekdays.csv", index=False)
    print(f"Аудит завершён: {len(all_days)} архивов, ошибок {sum(~all_days.valid)}; фиксированное окно {config['fixed_start']}–{config['fixed_end']}", flush=True)


def segment_map(audit_days):
    """Возвращает сегменты дат; подтверждённая ошибка или разрыв >10 дней сбрасывает историю."""
    result, previous, label, broken = {}, None, 0, False
    for row in audit_days.itertuples():
        day = pd.Timestamp(row.day)
        if broken or (previous is not None and (day-previous).days>10):
            label += 1
        result[row.day] = label
        broken = not row.valid
        previous = day
    return result


def decorated_bars(raw, segments, include_weekends=False):
    """Прогревает индикаторы raw отдельно в сегментах; по умолчанию исключает выходные."""
    raw = raw.copy()
    raw["segment"] = raw.day.map(segments)
    raw = raw[raw.segment.notna()]
    pieces = [indicators(group, include_weekends=include_weekends) for _, group in raw.groupby("segment", sort=False)]
    return pd.concat(pieces, ignore_index=True) if pieces else raw


def volume_features(args, symbol, audit_days, config):
    """Формирует дневную таблицу относительных объёмов из точных префиксных сумм 21 дня."""
    rows, history = [], deque(maxlen=20)
    for row in audit_days.itertuples():
        if row.weekday >= 5:
            continue
        current = {"day": row.day, "valid": row.valid}
        if row.valid:
            with np.load(args.output / "cache" / symbol / f"{row.day.replace('-', '')}.npz") as stored:
                times, volumes = stored["times"], stored["volumes"]
            current.update(times=times, prefix=np.r_[np.int64(0), volumes.cumsum()], first=int(times[0]), last=int(times[-1]))
        if row.day >= config["start"]:
            result = {"symbol": symbol, "day": row.day, "year": int(row.day[:4]), "period": "2026" if row.day[:4]=="2026" else "2022–2025", "valid_observation": False,
                      "history_days": json.dumps([h["day"] for h in history]), "issue": ""}
            if not row.valid:
                result["issue"] = "ошибка исходных данных"
            elif len(history)<20:
                result["issue"] = "меньше 20 предыдущих будних дат"
            elif not all(h["valid"] for h in history):
                result["issue"] = "ошибка в 20-дневной истории"
            else:
                all_dates = list(history)+[current]
                bounds = common_window([(h["first"], h["last"]) for h in all_dates])
                if bounds is None:
                    result["issue"] = "нет окна положительной длительности"
                else:
                    start, end = bounds
                    previous = [window_volume(h["times"], h["prefix"], start, end) for h in history]
                    volume = window_volume(times, current["prefix"], start, end)
                    mean, median = float(np.mean(previous)), float(np.median(previous))
                    result.update(start_ns=start, end_ns=end, start=time_text(start), end=time_text(end), hours=(end-start)/3.6e12,
                                  volume=volume, volume_mean=mean, volume_median=median, r_mean=volume/mean if mean else np.nan, r_median=volume/median if median else np.nan,
                                  full_volume=int(row.volume), observed_first=int(row.first_ns), observed_last=int(row.last_ns),
                                  history_volumes=json.dumps(previous), history_bounds=json.dumps([{ "day": h["day"], "first": time_text(h["first"]), "last": time_text(h["last"])} for h in all_dates]),
                                  valid_observation=bool(mean>0 and median>0))
                    fs, fe = config["fixed_start_ns"], config["fixed_end_ns"]
                    result["fixed_valid"] = all(h["first"]<=fs and h["last"]>=fe for h in all_dates)
                    if result["fixed_valid"]:
                        baseline = [window_volume(h["times"], h["prefix"], fs, fe) for h in history]
                        fv = window_volume(times, current["prefix"], fs, fe)
                        result.update(fixed_volume=fv, fixed_mean=float(np.mean(baseline)), fixed_median=float(np.median(baseline)),
                                      r_mean_fixed=fv/np.mean(baseline) if np.mean(baseline)>0 else np.nan,
                                      r_median_fixed=fv/np.median(baseline) if np.median(baseline)>0 else np.nan)
            rows.append(result)
        history.append(current)
    return pd.DataFrame(rows)


def analyze_symbol(args, symbol, config):
    """Рассчитывает признаки symbol на дельта-барах и свечах 1/5/15 минут, возвращает таблицу."""
    audit_days = pd.read_csv(args.output / f"{symbol}_audit.csv")
    features = volume_features(args, symbol, audit_days, config)
    segments = segment_map(audit_days)
    features["segment"] = features.day.map(segments)
    raw = pd.read_pickle(args.output / "cache" / f"{symbol}_bars.pkl")
    delta = decorated_bars(raw, segments)
    weekend = decorated_bars(raw, segments, include_weekends=True)
    frames = {"delta": delta, "delta_weekends": weekend}
    for minutes in (1, 5, 15):
        pieces = []
        for day in audit_days[audit_days.valid].itertuples():
            piece = pd.read_pickle(args.output / "cache" / symbol / f"{day.day.replace('-', '')}_{minutes}m.pkl")
            piece["day"], piece["weekday"] = day.day, day.weekday
            pieces.append(piece)
        frames[f"time{minutes}"] = decorated_bars(pd.concat(pieces, ignore_index=True), segments)
    groups = {name: {day: group for day, group in frame.groupby("day", sort=False)} for name, frame in frames.items()}
    output_rows = []
    for index, row in features.iterrows():
        item = row.to_dict()
        if row.valid_observation:
            start, end = int(row.start_ns), int(row.end_ns)
            for name, daily in groups.items():
                if row.day not in daily:
                    continue
                bars = daily[row.day]
                metrics = daily_metrics(bars, start, end)
                item.update({f"{name}_{key}": value for key, value in metrics.items()})
                if name=="delta":
                    item["threshold"] = float(bars.threshold.iloc[0])
                    item["delta_volume_coverage"] = metrics["included_volume"]/row.volume if row.volume else np.nan
                    partial = daily_metrics(bars, start, end, drop_partial=True)
                    item.update({f"delta_complete_{key}": value for key, value in partial.items()})
                if row.get("fixed_valid", False) is True or row.get("fixed_valid", False)==True:
                    metrics = daily_metrics(bars, config["fixed_start_ns"], config["fixed_end_ns"])
                    item.update({f"fixed_{name}_{key}": value for key, value in metrics.items()})
        output_rows.append(item)
        if index%200==0:
            print(f"Показатели {symbol}: {index+1}/{len(features)}", flush=True)
    result = pd.DataFrame(output_rows)
    result.to_csv(args.output / f"{symbol}_daily.csv", index=False)
    return result


def comparisons(data, repetitions):
    """Возвращает блочные сравнения основных/контрольных показателей и поправку Holm."""
    metrics = [f"{frame}_{metric}" for frame in ("delta", "time5", "time1", "time15") for metric in ("st", "sma", "st_hour", "sma_hour", "st_100", "sma_100")]
    metrics += ["time5_er", "time5_range", "delta_bars", "delta_complete_st", "delta_complete_sma", "delta_weekends_st", "delta_weekends_sma"]
    metrics += [f"fixed_{frame}_{metric}" for frame in ("delta", "time5") for metric in ("st", "sma", "st_100", "sma_100")]
    rows = []
    for symbol, part in data.groupby("symbol", sort=False):
        for period in ("все", "2022–2025", "2026"):
            sample = part if period=="все" else part[part.period.eq(period)]
            sample = sample.sort_values("day")
            for basis in ("mean", "median"):
                for metric in metrics:
                    if metric not in sample:
                        continue
                    ratio = f"r_{basis}_fixed" if metric.startswith("fixed_") else f"r_{basis}"
                    segments = sample.segment.to_numpy()
                    effect = bootstrap_effect(sample[metric].to_numpy(), sample[ratio].to_numpy(), repetitions, 20, segments=segments)
                    rows.append({"symbol": symbol, "period": period, "basis": basis, "metric": metric, "block": 20, **effect})
            print(f"Статистика {symbol}, {period}: готово", flush=True)
        sample = part[part.period.eq("2026")].sort_values("day")
        for block in (40, 60):
            for basis in ("mean", "median"):
                for metric in ("delta_st", "delta_sma"):
                    effect = bootstrap_effect(sample[metric].to_numpy(), sample[f"r_{basis}"].to_numpy(), repetitions, block, segments=sample.segment.to_numpy())
                    rows.append({"symbol": symbol, "period": "2026", "basis": basis, "metric": metric, "block": block, **effect})
    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    for block in (20, 40, 60):
        main = result.period.eq("2026") & result.metric.isin(["delta_st", "delta_sma"]) & result.block.eq(block)
        result.loc[main, "p_holm"] = holm(result.loc[main, "p"])
    return result


def descriptive_tables(data, output):
    """Сохраняет годовые/режимные сводки, диапазоны R и заранее заданные исключения."""
    rows, bins, sensitivities = [], [], []
    for symbol, part in data.groupby("symbol"):
        part = part.copy()
        part["observed_open_hour"] = (part.observed_first/3.6e12).apply(np.floor)
        part["observed_close_hour"] = (part.observed_last/3.6e12).apply(np.floor)
        for basis in ("mean", "median"):
            ratio = f"r_{basis}"
            for label, group in part.groupby("year"):
                for metric in ("delta_st", "delta_sma", "time5_st", "time5_sma", "time5_er"):
                    rows.append({"symbol": symbol, "basis": basis, "group": "год", "label": str(label), "metric": metric, **group_effect(group[metric], group[ratio])})
            for label, group in part.groupby(["observed_open_hour", "observed_close_hour"]):
                for metric in ("delta_st", "delta_sma", "time5_st", "time5_sma"):
                    rows.append({"symbol": symbol, "basis": basis, "group": "границы", "label": str(label), "metric": metric, **group_effect(group[metric], group[ratio])})
            labels = ["≤0.75", "0.75–1", "1–1.25", "1.25–1.5", ">1.5"]
            categories = pd.cut(part[ratio], [-np.inf, .75, 1, 1.25, 1.5, np.inf], labels=labels, right=True)
            for label in labels:
                group = part[categories.eq(label)]
                item = {"symbol": symbol, "basis": basis, "bin": label, "n": len(group)}
                for metric in ("delta_st", "delta_sma", "delta_st_100", "delta_sma_100", "time5_st", "time5_sma", "time5_er", "delta_bars"):
                    item[metric] = group[metric].mean()
                bins.append(item)
            previous_first, previous_last = part.observed_first.shift(), part.observed_last.shift()
            changed = (abs(part.observed_first-previous_first)>1800*NS) | (abs(part.observed_last-previous_last)>1800*NS)
            dates = pd.to_datetime(part.day)
            # Календарные диапазоны заданы независимо от итогового эффекта.
            calendar_holidays = dates.dt.strftime("%m-%d").isin(["01-01", "01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-08", "02-23", "03-08", "05-01", "05-09", "06-12", "11-04"])
            possible_roll = dates.dt.month.isin([3, 6, 9, 12]) & dates.dt.day.between(10, 25)
            checks = {"без_верхнего_1процента": part.volume.le(part.volume.quantile(.99)), "без_резких_границ": ~changed,
                      "без_календарных_праздников": ~calendar_holidays, "без_предполагаемого_ролловера": ~possible_roll}
            for name, mask in checks.items():
                for period in ("все", "2026"):
                    group = part[mask & (True if period=="все" else part.period.eq(period))]
                    for metric in ("delta_st", "delta_sma", "time5_st", "time5_sma"):
                        sensitivities.append({"symbol": symbol, "basis": basis, "period": period, "variant": name, "metric": metric, **group_effect(group[metric], group[ratio])})
    pd.DataFrame(rows).to_csv(output / "year_and_bounds.csv", index=False)
    pd.DataFrame(bins).to_csv(output / "ratio_bins.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(output / "sensitivities.csv", index=False)


def analyze(args):
    """Выполняет расчёт по сохранённой конфигурации аудита, сохраняя дневные таблицы и сравнения."""
    config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
    parts = [analyze_symbol(args, symbol, config) for symbol in config["symbols"]]
    data = pd.concat(parts, ignore_index=True)
    data.to_csv(args.output / "daily.csv", index=False)
    good = data[data.valid_observation].copy()
    result = comparisons(good, args.repetitions)
    result.to_csv(args.output / "comparisons.csv", index=False)
    descriptive_tables(good, args.output)
    save_json(args.output / "analysis_run.json", {"repetitions": args.repetitions, "seed": config["seed"], "command": " ".join(sys.argv), "completed": datetime.now().isoformat(timespec="seconds")})


def report(args):
    """Создаёт итоговый отчёт и автономные графики из сохранённых результатов args.output."""
    from research.volume_trend_report import make_report
    make_report(args.output)


def main():
    """Выполняет выбранные этапы и завершает запуск с сохранёнными результатами."""
    args = arguments()
    if args.stage in ("audit", "all"):
        audit(args)
    if args.stage in ("analyze", "all"):
        analyze(args)
    if args.stage in ("report", "all"):
        report(args)


if __name__ == "__main__":
    main()
