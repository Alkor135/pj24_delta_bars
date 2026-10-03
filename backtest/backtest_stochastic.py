"""Проверка Stochastic (14, 3, 3) с ALF на дельта-барах и исходных тиках.

Примеры запуска из корня проекта:
    python backtest/backtest_stochastic.py
    python backtest/backtest_stochastic.py --symbols RTS --start 2026-09-01 --end 2026-09-30
    python backtest/backtest_stochastic.py --symbols RTS MIX --costs 0,2,4,8 --reward-risk 2
    python backtest/backtest_stochastic.py --period 14 --smooth-k 3 --smooth-d 3 --alf-alpha 0.4
    python -m backtest.backtest_stochastic --help
    python -m unittest -v tests.test_backtest_stochastic

Вход после пересечения %K уровня 20 вверх либо 80 вниз, подтверждённого %D,
наклоном ALF и стороной закрытия относительно ALF. Индикаторы используют только
прошлые бары; стохастик исключает неполные остатки, ALF соответствует просмотрщику.
Стоп — за экстремумом последних трёх полных баров одного дня с запасом один шаг.
Цель — 2 исходных риска. Касание стопа/цели исполняется следующим реальным тиком.
Одна позиция на инструмент, входы 10:00–18:30, закрытие первым тиком с 18:40 МСК.
База сравнения снимает только условие стохастика. Подбора параметров нет.
Сценарии затрат задаются в шагах за круг; PnL — пункты одного контракта.
Базы и ZIP открываются для чтения; OHLC, число баров и покрытие строк сверяются
с исходными тиками и журналом. Прогрев виден в покрытии. Отчёты сохраняются
в results/stochastic; каждый календарный год с 2026 показывается отдельно.
Пути исходников для SHA256 определяются от корня проекта, относительный
--output — от текущей рабочей папки, в том числе при запуске по абсолютному пути.
"""

import argparse
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    # Пакет source доступен при запуске файла из любой рабочей папки.
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from source.chart_data import laguerre
from source.duration_data import Session, build_day, read_tick_zip


MODES = {"stochastic": "ALF + Stochastic", "alf": "ALF без стохастика"}
TRADE_COLUMNS = ["day", "symbol", "mode", "side", "signal_bar", "signal_time",
    "signal_close", "signal_alf", "signal_k", "signal_d", "entry_row", "entry_time",
    "entry_price", "stop_price", "target_price", "risk_points", "trigger_row",
    "trigger_time", "exit_row", "exit_time", "exit_price", "reason", "gross_pnl",
    "cost_points", "net_pnl"]
LIMITATIONS = (
    "PnL выражен в пунктах на один контракт, не в рублях и не в доходности счёта. "
    "Затраты — сценарии спреда, проскальзывания и комиссии, не тариф брокера. "
    "Стоп и цель являются рыночными сигналами: касание последней ценой, исполнение "
    "на следующем тике. Стакан, очередь и реальная задержка неизвестны. "
    "Доли секунды искусственные; порядок строк сохраняется. "
    "Источник — непрерывная склейка без кода контракта. Перерывы и аномальные "
    "дни нельзя считать обычным торговым режимом. Дни без тика дневного закрытия "
    "исключены и перечислены в покрытии; их риск не смоделирован. "
    "2026 год уже просматривался в предыдущем анализе, поэтому это отдельный "
    "период сравнения, а не новый независимый контроль. Параметры не оптимизировались."
)


def stochastic_frame(bars, period=14, smooth_k=3, smooth_d=3, alpha=0.4):
    """Возвращает копию баров с причинными %K, %D и ALF по указанным окнам.

    bars содержит OHLC и is_complete. period задаёт диапазон, smooth_k и
    smooth_d — окна SMA, alpha — коэффициент ALF. Нулевой диапазон и прогрев
    сохраняются как NaN. Неполные бары участвуют только в ALF.
    """
    if any(int(x) != x or x < 1 for x in (period, smooth_k, smooth_d)):
        raise ValueError("Окна стохастика должны быть положительными целыми")
    result = bars.copy().reset_index(drop=True)
    result["alf"] = laguerre(result.close.to_numpy(), alpha)
    complete = result.is_complete == 1
    full = result.loc[complete]
    lowest = full.low.rolling(period, min_periods=period).min()
    highest = full.high.rolling(period, min_periods=period).max()
    span = (highest - lowest).replace(0, np.nan)
    raw = 100 * (full.close - lowest) / span
    k = raw.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    result["k"], result["d"] = np.nan, np.nan
    result.loc[complete, "k"] = k
    result.loc[complete, "d"] = d
    return result


def entry_signals(frame, mode="stochastic", stop_lookback=3, session=None, tick_size=10):
    """Возвращает упорядоченные сигналы с номером следующего тика и уровнем стопа.

    frame содержит индикаторы и строки тиков. mode выбирает стратегию или
    базу сравнения; stop_lookback — число полных баров одного дня, session —
    московские часы, tick_size — запас стопа. Будущая цена входа не используется.
    """
    if mode not in MODES or stop_lookback < 1 or int(stop_lookback) != stop_lookback:
        raise ValueError("Некорректный режим или окно стопа")
    if not np.isfinite(tick_size) or tick_size <= 0:
        raise ValueError("Шаг цены должен быть положительным")
    session = session or Session()
    full = frame.loc[frame.is_complete == 1].copy()
    previous_k = full.k.shift(1)
    rising = (full.alf > full.alf.shift(1)) & (full.close > full.alf)
    falling = (full.alf < full.alf.shift(1)) & (full.close < full.alf)
    if mode == "stochastic":
        rising &= (previous_k <= 20) & (full.k > 20) & (full.k > full.d)
        falling &= (previous_k >= 80) & (full.k < 80) & (full.k < full.d)
    earliest = full.start_time.astype(str).shift(stop_lookback - 1).str.slice(11, 19)
    end_time = full.end_time.astype(str).str.slice(11, 19)
    valid = full.day.eq(full.day.shift(stop_lookback - 1))
    valid &= (earliest >= session.start) & (end_time < session.entry_end)
    low = full.low.rolling(stop_lookback).min() - tick_size
    high = full.high.rolling(stop_lookback).max() + tick_size
    signals = []
    for index, bar in full.loc[valid & (rising | falling)].iterrows():
        is_long = bool(rising.loc[index])
        signals.append(dict(day=str(bar.day), entry_row=int(bar.end_row) + 1,
            signal_bar=int(bar.bar_index), signal_time=str(bar.end_time),
            side="Long" if is_long else "Short",
            stop_price=float(low.loc[index] if is_long else high.loc[index]),
            signal_close=float(bar.close), signal_alf=float(bar.alf),
            signal_k=None if pd.isna(bar.k) else float(bar.k),
            signal_d=None if pd.isna(bar.d) else float(bar.d)))
    return signals


def _path_metrics(values, previous_peak):
    """Возвращает пик, минимум и просадку ценового пути с учётом предыдущего пика."""
    values = np.asarray(values, dtype=float)
    peaks = np.maximum.accumulate(np.maximum(values, previous_peak))
    return float(peaks[-1]), float(values.min()), float((peaks - values).max())


def simulate_day(day, signals, prices, stamps, close_row, tick_size,
                 reward_risk=2, cost_ticks=4, entry_end="18:30:00"):
    """Возвращает сделки и дневную тиковую статистику одного инструмента.

    signals — сигналы в порядке строк, prices/stamps — исходные тики,
    close_row — строка обязательного выхода (с единицы), tick_size — шаг цены,
    reward_risk — цель в исходных рисках, cost_ticks — затраты за круг,
    entry_end — предельное московское время фактического входа.
    Касания исполняются следующим тиком; сигналы в позиции пропускаются.
    """
    prices = np.asarray(prices, dtype=float)
    if (len(prices) != len(stamps) or not 1 <= close_row <= len(prices)
            or not np.isfinite(prices).all() or (prices <= 0).any()):
        raise ValueError("Некорректные тики или строка дневного выхода")
    if (not np.isfinite([tick_size, reward_risk, cost_ticks]).all()
            or tick_size <= 0 or reward_risk <= 0 or cost_ticks < 0):
        raise ValueError("Некорректная цель, шаг или затраты")
    if any(int(x["entry_row"]) != x["entry_row"] or x["entry_row"] < 1 for x in signals):
        raise ValueError("Строки входа должны быть положительными целыми")
    rows = [x["entry_row"] for x in signals]
    if rows != sorted(rows):
        raise ValueError("Сигналы должны быть упорядочены по строкам")
    trades = []
    gross, peak, minimum, max_dd = 0.0, 0.0, 0.0, 0.0
    cost = float(cost_ticks * tick_size)
    previous_exit, invalid, busy = 0, 0, 0
    close_index = close_row - 1
    for signal in signals:
        entry_row = int(signal["entry_row"])
        if entry_row <= previous_exit:
            busy += 1
            continue
        if entry_row >= close_row:
            continue
        if signal["side"] not in ("Long", "Short"):
            raise ValueError("Неизвестная сторона сделки")
        entry_index = entry_row - 1
        if str(stamps[entry_index])[11:19] >= entry_end:
            continue
        direction = 1 if signal["side"] == "Long" else -1
        entry, stop = float(prices[entry_index]), float(signal["stop_price"])
        risk = direction * (entry - stop)
        if not np.isfinite(risk) or risk <= 0:
            invalid += 1
            continue
        target = entry + direction * reward_risk * risk
        # Закрытие по времени имеет приоритет на своей строке. Касание до неё
        # исполняется на следующей строке, в том числе на строке дневного выхода.
        available = prices[entry_index + 1:close_index]
        stop_hit = direction * (available - stop) <= 0
        target_hit = direction * (available - target) >= 0
        hits = np.flatnonzero(stop_hit | target_hit)
        trigger_index, exit_index, reason = close_index, close_index, "time"
        if len(hits):
            offset = int(hits[0])
            trigger_index = entry_index + 1 + offset
            exit_index = trigger_index + 1
            reason = "stop" if stop_hit[offset] else "target"
        exit_price = float(prices[exit_index])
        pnl = direction * (exit_price - entry)
        path = gross + direction * (prices[entry_index:exit_index + 1] - entry)
        path = path - len(trades) * cost - cost / 2
        peak, path_min, path_dd = _path_metrics(path, peak)
        minimum, max_dd = min(minimum, path_min), max(max_dd, path_dd)
        gross += pnl
        closed_net = gross - (len(trades) + 1) * cost
        peak, end_min, end_dd = _path_metrics([closed_net], peak)
        minimum, max_dd = min(minimum, end_min), max(max_dd, end_dd)
        trade = dict(signal, day=day, entry_time=str(stamps[entry_index]),
            exit_time=str(stamps[exit_index]), trigger_time=str(stamps[trigger_index]),
            trigger_row=trigger_index + 1, exit_row=exit_index + 1,
            entry_price=entry, exit_price=exit_price, target_price=target,
            risk_points=risk, reason=reason, gross_pnl=float(pnl),
            cost_points=cost, net_pnl=float(pnl - cost))
        trades.append(trade)
        previous_exit = exit_index + 1
    daily = dict(day=day, trades=len(trades), gross_pnl=gross,
        net_pnl=gross - len(trades) * cost, equity_peak=peak, equity_min=minimum,
        max_drawdown=max_dd, signals=len(signals), invalid_stops=invalid,
        signals_in_position=busy, cost_ticks=cost_ticks)
    return dict(trades=trades, daily=daily)


def _summarize(trades, days, cost_ticks, tick_size):
    """Возвращает статистику сделок и полную тиковую просадку заданного периода."""
    values = np.asarray([t["gross_pnl"] - cost_ticks * tick_size for t in trades])
    cumulative, peak, max_dd = 0.0, 0.0, 0.0
    for row in days:
        max_dd = max(max_dd, row["max_drawdown"], peak - cumulative - row["equity_min"])
        peak = max(peak, cumulative + row["equity_peak"])
        cumulative += row["net_pnl"]
    wins = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return dict(days=len(days), trades=len(values), gross_pnl=float(sum(t["gross_pnl"] for t in trades)),
        net_pnl=float(values.sum()), average_trade=float(values.mean()) if len(values) else None,
        profit_factor=wins / losses if losses else None,
        win_rate=float((values > 0).mean()) if len(values) else None,
        max_drawdown=float(max_dd), long_pnl=float(sum(t["gross_pnl"] - cost_ticks * tick_size
            for t in trades if t["side"] == "Long")),
        short_pnl=float(sum(t["gross_pnl"] - cost_ticks * tick_size
            for t in trades if t["side"] == "Short")))


def validate_extremes(bars, prices, expected_count=None):
    """Сверяет high/low и полное покрытие тиков; возвращает None или вызывает ошибку.

    bars содержит границы строк и экстремумы, prices — все сделки дня,
    expected_count — число баров по журналу. Пропуски, перекрытия и изменённые
    внутрибара экстремумы запрещены, включая последний неполный остаток.
    """
    if expected_count is not None and len(bars) != expected_count:
        raise ValueError("Не совпало число баров с журналом")
    starts = bars.start_row.to_numpy(dtype=int) - 1
    ends = bars.end_row.to_numpy(dtype=int) - 1
    if (not len(starts) or starts[0] != 0 or ends[-1] != len(prices) - 1
            or (starts < 0).any() or (ends < starts).any()
            or not np.array_equal(starts[1:], ends[:-1] + 1)):
        raise ValueError("Нарушено полное покрытие исходных строк дня")
    prices = np.asarray(prices, dtype=float)
    high = np.maximum.reduceat(prices, starts)
    low = np.minimum.reduceat(prices, starts)
    if not np.array_equal(high, bars.high.to_numpy()) or not np.array_equal(low, bars.low.to_numpy()):
        raise ValueError("Экстремумы high/low не соответствуют исходным тикам")


def report_periods(first_day, last_day):
    """Возвращает точные календарные диапазоны сводки по границам доступных дат."""
    first_year = min(2022, date.fromisoformat(first_day).year)
    last_year = date.fromisoformat(last_day).year
    periods = [(f"{first_year}–2025", f"{first_year}-01-01", "2026-01-01")]
    periods.extend((str(year), f"{year}-01-01", f"{year + 1}-01-01")
                   for year in range(2026, max(2026, last_year) + 1))
    return periods


def _load_database(path, symbol, start, end, dataset_id, settings):
    """Читает бары до end и все статусы журнала в start–end для symbol/dataset_id.

    settings задаёт индикаторы. Возвращает бары с прогревом, журнал и метаданные;
    записи без баров остаются в журнале для явного описания покрытия.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Нет базы: {path}")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        sets = db.execute("SELECT DISTINCT dataset_id FROM bars WHERE symbol=?", (symbol,)).fetchall()
        if dataset_id is None:
            if len(sets) != 1:
                raise ValueError(f"{symbol}: необходимо указать --dataset-id")
            dataset_id = sets[0][0]
        config = db.execute("SELECT config_json FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if config is None:
            raise ValueError("Набор отсутствует")
        bars = pd.read_sql_query("SELECT * FROM bars WHERE dataset_id=? AND symbol=? AND day<=? "
                                 "ORDER BY day,bar_index", db, params=(dataset_id, symbol, end))
        sources = pd.read_sql_query("SELECT * FROM days WHERE dataset_id=? AND symbol=? "
                                    "AND day>=? AND day<=? ORDER BY day", db,
                                    params=(dataset_id, symbol, start, end))
    if bars.empty or sources.empty:
        raise ValueError(f"{symbol}: нет данных в заданном диапазоне")
    if bars.duplicated(["day", "bar_index"]).any():
        raise ValueError("Повторные номера баров")
    if not np.isfinite(bars[["open", "high", "low", "close"]].to_numpy()).all():
        raise ValueError("Некорректные цены баров")
    frame = stochastic_frame(bars, settings.period, settings.smooth_k, settings.smooth_d, settings.alf_alpha)
    ready = sources.loc[sources.status == "ready"]
    metadata = dict(db=str(path), symbol=symbol, dataset_id=dataset_id, bar_config=json.loads(config[0]),
        first_day=str(sources.day.min()), last_day=str(sources.day.max()),
        first_ready_day=str(ready.day.min()), last_ready_day=str(ready.day.max()), ready_days=len(ready),
        source_days=len(sources), warmup_bars=int((bars.day < start).sum()),
        bars_sha256=sha256(bars.to_json(orient="split", double_precision=15).encode()).hexdigest(),
        source_manifest_sha256=sha256(sources[["day", "sha256"]].to_json(orient="records").encode()).hexdigest())
    return frame, sources, metadata


def _table(frame):
    """Возвращает безопасную HTML-таблицу сводных показателей с русскими заголовками."""
    labels = dict(symbol="Инструмент", strategy="Стратегия", period="Период",
        cost_ticks="Затраты, шагов", days="Дней", trades="Сделок", gross_pnl="До затрат",
        net_pnl="После затрат", average_trade="Средняя сделка", profit_factor="Фактор прибыли",
        win_rate="Доля прибыльных", max_drawdown="Тиковая просадка", long_pnl="Покупки", short_pnl="Продажи")
    return frame.rename(columns=labels).to_html(index=False, border=0, na_rep="—",
                                              float_format="%.3f", escape=True)


def _write_report(folder, summary, daily, metadata):
    """Создаёт автономный интерактивный отчёт с PnL, периодом 2026 и издержками."""
    parts = []
    for symbol, frame in daily.groupby("symbol", sort=False):
        base = frame[frame.cost_ticks == metadata["settings"]["base_cost"]]
        only_2026 = base[(base.day >= "2026-01-01") & (base.day < "2027-01-01")]
        for title, selected in (("Вся история", base), ("Только 2026 год", only_2026)):
            if selected.empty:
                continue
            figure = go.Figure()
            for mode, series in selected.groupby("mode", sort=False):
                figure.add_trace(go.Scatter(x=series.day, y=series.net_pnl.cumsum(),
                                            name=MODES[mode], mode="lines"))
            figure.update_layout(template="plotly_white", title=f"{symbol}: {title}, после затрат",
                yaxis_title="Пункты на один контракт", height=430, hovermode="x unified",
                legend=dict(orientation="h", y=-0.2), margin=dict(b=90))
            parts.append(figure.to_html(full_html=False, include_plotlyjs=False))
    settings_text = json.dumps(metadata["settings"], ensure_ascii=False, indent=2)
    coverage_text = ''.join('<li>' + escape(
        f'{row["symbol"]}: журнал {row["first_day"]} — {row["last_day"]}; '
        f'готовые бары {row["first_ready_day"]} — {row["last_ready_day"]}; '
        f'включено {row["included_days"]} дней, исключено {row["excluded_days"]} '
        '(включая прогрев и отсутствие дневного исполнения).') + '</li>'
        for row in metadata["datasets"])
    html = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Тест стохастика</title>'
        '<style>body{font:16px Segoe UI,sans-serif;margin:30px auto;max-width:1400px;padding:20px;'
        'color:#24354b;background:#f5f7fb}table{border-collapse:collapse;background:white;font-size:14px}'
        'td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#e8eef6}'
        '.scroll{overflow-x:auto}pre{white-space:pre-wrap}a{color:#165ab6}</style>'
        '<h1>Стохастик на дельта-барах RTS и MIX</h1>'
        '<p>Вход: %K пересекает 20 вверх при %K &gt; %D, растущем ALF и close &gt; ALF; '
        'продажа: пересечение 80 вниз, %K &lt; %D, падающий ALF и close &lt; ALF. '
        'База сравнения снимает только условие стохастика.</p>'
        '<p>Стоп: за минимумом/максимумом последних трёх полных баров одного дня '
        'с запасом один шаг. Цель: 2R по умолчанию, от фактической цены входа. '
        'Касание стопа/цели исполняется следующей строкой исходных тиков.</p>'
        '<p>Параметры зафиксированы до расчёта. Обучение или выбор лучшего варианта не проводились.</p>'
        '<h2>Покрытие исходной истории</h2><ul>' + coverage_text + '</ul>'
        '<h2>Сводка по периодам и затратам</h2><div class="scroll">' + _table(summary) + '</div>'
        '<h2>Графики при опорных затратах</h2><script>' + get_plotlyjs() + '</script>'
        + ''.join(parts) + '<h2>Правила и ограничения</h2><p>' + escape(LIMITATIONS) + '</p>'
        '<p><a href="summary.csv">Сводка CSV</a> · <a href="daily.csv">Дневные результаты</a> · '
        '<a href="trades.csv">Сделки при опорных затратах</a> · '
        '<a href="coverage.csv">Покрытие и исключённые дни</a> · '
        '<a href="metadata.json">Настройки и контрольные суммы</a></p><pre>'
        + escape(settings_text) + '</pre></html>')
    path = folder / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def parse_args(argv=None):
    """Разбирает и проверяет параметры CLI; возвращает настройки фиксированного теста."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("C:/data_quote/delta_bars.sqlite3"))
    parser.add_argument("--symbols", nargs="+", choices=["RTS", "MIX"], default=["RTS", "MIX"])
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="9999-12-31")
    parser.add_argument("--dataset-id")
    parser.add_argument("--period", type=int, default=14)
    parser.add_argument("--smooth-k", type=int, default=3)
    parser.add_argument("--smooth-d", type=int, default=3)
    parser.add_argument("--alf-alpha", type=float, default=0.4)
    parser.add_argument("--reward-risk", type=float, default=2)
    parser.add_argument("--costs", default="0,2,4,8")
    parser.add_argument("--base-cost", type=float, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/stochastic"))
    args = parser.parse_args(argv)
    if date.fromisoformat(args.start) > date.fromisoformat(args.end):
        parser.error("Начало должно быть не позже конца")
    args.costs = sorted(set(float(x) for x in args.costs.split(",")))
    if (not args.costs or not np.isfinite(args.costs).all() or min(args.costs) < 0
            or args.base_cost not in args.costs):
        parser.error("Затраты должны быть неотрицательными; --base-cost должен входить в --costs")
    if (min(args.period, args.smooth_k, args.smooth_d) < 1
            or not np.isfinite([args.alf_alpha, args.reward_risk]).all()
            or not 0 < args.alf_alpha <= 1 or args.reward_risk <= 0):
        parser.error("Некорректные окна, ALF или отношение цели к риску")
    return args


def main(argv=None):
    """Выполняет тест по argv, сохраняет таблицы и возвращает путь HTML-отчёта.

    argv — список аргументов либо None для CLI. SHA256 исходников читаются
    от PROJECT_ROOT; относительный --output считается от рабочей папки.
    """
    args = parse_args(argv)
    session = Session()
    stamp = datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S_%f")
    folder = args.output.resolve() / ("run_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    metadata = dict(settings={key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()}, session=vars(session),
        stop_lookback=3, stop_buffer_ticks=1, units="пункты на один контракт",
        limitations=LIMITATIONS, datasets=[],
        implementation_sha256={name: sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in
            ("backtest/backtest_stochastic.py", "source/chart_data.py", "source/duration_data.py")})
    all_trades, all_daily, coverage, summary = [], [], [], []
    for symbol in dict.fromkeys(args.symbols):
        tick_size = {"RTS": 10, "MIX": 25}[symbol]
        frame, sources, source_meta = _load_database(args.db, symbol, args.start, args.end,
                                                     args.dataset_id, args)
        source_meta["tick_size"] = tick_size
        metadata["datasets"].append(source_meta)
        bars_by_day = dict(tuple(frame.loc[frame.day >= args.start].groupby("day", sort=False)))
        signals_by_mode = {}
        for mode in MODES:
            grouped = {}
            for signal in entry_signals(frame, mode=mode, tick_size=tick_size, session=session):
                if signal["day"] >= args.start:
                    grouped.setdefault(signal["day"], []).append(signal)
            signals_by_mode[mode] = grouped
        print(f"{symbol}: журнал {len(sources)} дней, {source_meta['ready_days']} готовых архивов; "
              "параметры зафиксированы", flush=True)
        for count, source in enumerate(sources.itertuples(), 1):
            if source.status != "ready":
                reason = "Прогрев построения дельта-баров" if source.status == "warmup" else f"Статус дня: {source.status}"
                coverage.append(dict(date=source.day, status="excluded", reason=reason,
                                     bars=source.bar_count, ticks=source.tick_count, symbol=symbol))
                continue
            if source.day not in bars_by_day:
                raise ValueError(f"{source.day}: готовый день без баров")
            bars = bars_by_day[source.day]
            ticks = read_tick_zip(Path(source.file_path), source.sha256)
            if len(ticks) != source.tick_count:
                raise ValueError(f"{source.day}: число тиков не совпало")
            validate_extremes(bars, ticks["last"].to_numpy(), expected_count=source.bar_count)
            prepared, audit = build_day(source.day, bars, ticks, session)
            coverage.append(dict(audit, symbol=symbol))
            if prepared is not None:
                prices, stamps = ticks["last"].to_numpy(), ticks.datetime.astype(str).to_numpy()
                if not np.allclose(prices / tick_size, np.round(prices / tick_size), rtol=0, atol=1e-7):
                    raise ValueError(f"{symbol} {source.day}: цены вне решётки шага")
                for mode in MODES:
                    signatures = None
                    for cost in args.costs:
                        result = simulate_day(source.day, signals_by_mode[mode].get(source.day, []),
                            prices, stamps, audit["close_row"], tick_size, args.reward_risk, cost)
                        signature = [(t["entry_row"], t["exit_row"], t["side"]) for t in result["trades"]]
                        if signatures is not None and signatures != signature:
                            raise AssertionError("Издержки изменили моменты сделок")
                        signatures = signature
                        all_daily.append(dict(result["daily"], symbol=symbol, mode=mode))
                        if cost == args.base_cost:
                            all_trades.extend(dict(t, symbol=symbol, mode=mode) for t in result["trades"])
            if count % 100 == 0 or count == len(sources):
                print(f"{symbol}: проверено {count}/{len(sources)} дней", flush=True)
        source_meta["included_days"] = sum(x["symbol"] == symbol and x["status"] == "included" for x in coverage)
        source_meta["excluded_days"] = sum(x["symbol"] == symbol and x["status"] != "included" for x in coverage)
        for mode in MODES:
            trades = [t for t in all_trades if t["symbol"] == symbol and t["mode"] == mode]
            for cost in args.costs:
                days = [d for d in all_daily if d["symbol"] == symbol and d["mode"] == mode
                        and d["cost_ticks"] == cost]
                for period, left, right in report_periods(source_meta["first_day"], source_meta["last_day"]):
                    period_trades = [t for t in trades if left <= t["day"] < right]
                    period_days = [d for d in days if left <= d["day"] < right]
                    stats = _summarize(period_trades, period_days, cost, tick_size)
                    if not np.isclose(stats["net_pnl"], sum(d["net_pnl"] for d in period_days)):
                        raise AssertionError("Журнал сделок не совпал с дневным результатом")
                    summary.append(dict(symbol=symbol, strategy=MODES[mode], period=period,
                                        cost_ticks=cost, **stats))
    if not all_daily:
        raise ValueError("Нет дней с исполнимым закрытием")
    summaries, days = pd.DataFrame(summary), pd.DataFrame(all_daily)
    trades = pd.DataFrame(all_trades, columns=TRADE_COLUMNS)
    summaries.to_csv(folder / "summary.csv", index=False, encoding="utf-8-sig")
    days.to_csv(folder / "daily.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(folder / "trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(folder / "coverage.csv", index=False, encoding="utf-8-sig")
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _write_report(folder, summaries, days, metadata)
    print(summaries[summaries.cost_ticks == args.base_cost].to_string(index=False), flush=True)
    print(f"Отчёт: {report}", flush=True)
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f"Тест остановлен: {error}", file=sys.stderr)
        sys.exit(1)
