"""Исторический тест цвета Heikin Ashi на дельта-барах без других индикаторов.

Примеры запуска из корня проекта:
    python backtest/backtest_heikin_ashi.py
    python backtest/backtest_heikin_ashi.py --symbols RTS --start 2026-09-01 --end 2026-09-30
    python backtest/backtest_heikin_ashi.py --confirmations 1,2 --costs 0,2,4,8 --base-cost 4
    python -m backtest.backtest_heikin_ashi --help
    python -m unittest -v tests.test_backtest_heikin_ashi

HA-close = (O+H+L+C)/4, HA-open = (предыдущие HA-open+HA-close)/2.
Первый HA-open = (O+C)/2. Только полные бары участвуют в рекурсии, которая
продолжается через границы дней. Зелёная свеча означает покупку, красная —
продажу; доджи не даёт сигнала и сбрасывает серию подтверждения. Проверяются
фиксированные варианты с одной и двумя свечами одного цвета подряд.
Торговая серия начинается заново каждый день и использует бары, начатые с 10:00.
Вход/переворот — следующий исходный тик после закрытия подтверждающего бара;
цены HA служат только сигналом. Повторный цвет не добавляет позицию. После
18:30 разрешён только выход; в 18:40 закрытие первым доступным тиком. Стопов,
целей, ALF, стохастика и иных индикаторов нет; перенос позиции ночью запрещён.
PnL — пункты одного контракта, затраты — шаги за круг, поровну на вход/выход.
База и ZIP читаются без изменения, OHLC и покрытие сверяются с тиками,
включая всю историю прогрева до --start; сделки возможны только в start–end.
Таблицы, контрольные суммы и автономный HTML сохраняются в results/heikin_ashi.
Подбора параметров нет; история с 2026 года разбивается на календарные годы.
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
    # Соседний бэктест и source импортируются независимо от рабочей папки.
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from backtest.backtest_stochastic import _path_metrics, _summarize, _table, report_periods, validate_extremes
from source.duration_data import Session, build_day, read_tick_zip


TRADE_COLUMNS = ["day", "symbol", "confirmation", "side", "signal_bar", "signal_time",
    "trigger_row", "ha_open", "ha_close", "entry_row", "entry_time", "entry_price",
    "exit_bar", "exit_signal_time", "exit_row", "exit_time", "exit_price", "reason",
    "gross_pnl", "cost_points", "net_pnl"]
LIMITATIONS = (
    "Результат в пунктах на один контракт, не в рублях и не в процентах счёта. "
    "HA — синтетические цены; сделки исполняются только по следующему исходному тику. "
    "Затраты являются сценариями комиссии, спреда и проскальзывания за круг, не тарифом брокера. "
    "Стакан, очередь, реальная задержка неизвестны; доли секунды искусственные, порядок строк сохранён. "
    "Стопов и целей нет. Просадка рассчитана по каждому тику открытой позиции, включая обе половины затрат. "
    "Источник — непрерывная склейка без кода контракта. Дни без тика закрытия сессии исключены; "
    "их риск не смоделирован. Неполные остатки не участвуют в HA. "
    "2026 год уже просматривался ранее: это отдельный период сравнения, а не новый независимый контроль. "
    "Правила и варианты зафиксированы до расчёта; оптимизация не проводилась."
)


def heikin_ashi_frame(bars):
    """Возвращает копию хронологических bars с HA-OHLC и знаком цвета.

    bars содержит исходные OHLC и is_complete. Рекурсия использует только
    полные бары и не сбрасывается на новой дате; неполные строки получают NaN.
    Цвет равен знаку HA-close минус HA-open: 1, -1 либо 0 для доджи.
    """
    frame = bars.reset_index(drop=True).copy()
    values = np.full((len(frame), 5), np.nan)
    previous = None
    for i, row in enumerate(frame.itertuples()):
        if row.is_complete != 1:
            continue
        ha_close = (row.open + row.high + row.low + row.close) / 4
        ha_open = (row.open + row.close) / 2 if previous is None else sum(previous) / 2
        values[i] = [ha_open, max(row.high, ha_open, ha_close),
                     min(row.low, ha_open, ha_close), ha_close, np.sign(ha_close - ha_open)]
        previous = ha_open, ha_close
    frame[["ha_open", "ha_high", "ha_low", "ha_close", "ha_color"]] = values
    return frame


def color_targets(frame, confirmation=1, session=None):
    """Возвращает целевые направления после confirmation закрытых свечей цвета.

    frame содержит HA и границы исходных строк; session задаёт торговое время.
    Подтверждение сбрасывается на новой дате, доджи и недопустимом баре.
    Неполные бары не дают сигналов. Повторные направления оставлены в списке:
    движок удерживает позицию, не совершая дополнительных сделок.
    """
    if int(confirmation) != confirmation or confirmation < 1:
        raise ValueError("Число подтверждений должно быть положительным целым")
    session = session or Session()
    targets, previous_day, previous_color, streak = [], None, 0, 0
    for row in frame.itertuples():
        if row.day != previous_day:
            previous_day, previous_color, streak = row.day, 0, 0
        if (row.is_complete != 1 or str(row.start_time)[:19] < f"{row.day} {session.start}"
                or str(row.end_time)[:19] >= f"{row.day} {session.close}"):
            previous_color, streak = 0, 0
            continue
        color = int(row.ha_color)
        if color == 0:
            previous_color, streak = 0, 0
            continue
        streak = streak + 1 if color == previous_color else 1
        previous_color = color
        if streak >= confirmation:
            targets.append(dict(day=row.day, signal_bar=int(row.bar_index),
                signal_time=str(row.end_time), trigger_row=int(row.end_row), side=color,
                ha_open=float(row.ha_open), ha_close=float(row.ha_close)))
    return targets


def simulate_day(day, targets, prices, stamps, close_row, tick_size, cost_ticks=4, session=None):
    """Моделирует цветовые targets в day по реальным prices/stamps; возвращает журнал и день.

    trigger_row и close_row — номера исходных строк с единицы. Сигнал исполняется
    строкой trigger_row+1, закрытие сессии — строкой close_row. tick_size переводит
    cost_ticks за круг в пункты. session ограничивает вход, но допускает поздний
    выход по цвету. Просадка учитывает все тики между сделками и плату за каждую
    сторону операции; при перевороте выход и новый вход имеют одну реальную цену.
    """
    prices = np.asarray(prices, dtype=float)
    session = session or Session()
    if (len(prices) != len(stamps) or not 1 <= close_row <= len(prices)
            or not np.isfinite(prices).all() or (prices <= 0).any()
            or not np.isfinite([tick_size, cost_ticks]).all() or tick_size <= 0 or cost_ticks < 0):
        raise ValueError("Некорректные тики, строка закрытия или затраты")
    close_index, fee = close_row - 1, cost_ticks * tick_size
    trades, position = [], None
    realized = peak = minimum = max_dd = 0.0
    mark_start, previous_row = 0, 0

    def record_path(values):
        """Добавляет значения equity к общей просадке; результат хранится в замыкании."""
        nonlocal peak, minimum, max_dd
        peak, path_min, path_dd = _path_metrics(values, peak)
        minimum, max_dd = min(minimum, path_min), max(max_dd, path_dd)

    def mark_to(index):
        """Оценивает открытую позицию по всем тикам до index включительно; возвращает None."""
        nonlocal mark_start
        if position is not None:
            path = realized - fee / 2 + position["direction"] * (
                prices[mark_start:index + 1] - position["entry_price"])
            record_path(path)
            mark_start = index + 1

    def close_position(index, reason, target=None):
        """Закрывает позицию в index по reason/target; обновляет журнал и возвращает None."""
        nonlocal position, realized
        pnl = position["direction"] * (float(prices[index]) - position["entry_price"])
        trade = {key: value for key, value in position.items() if key != "direction"}
        trade.update(exit_bar=target["signal_bar"] if target else None,
            exit_signal_time=target["signal_time"] if target else None,
            exit_row=index + 1, exit_time=str(stamps[index]), exit_price=float(prices[index]),
            reason=reason, gross_pnl=pnl, cost_points=fee, net_pnl=pnl - fee)
        trades.append(trade)
        realized += pnl - fee
        record_path([realized])
        position = None

    for target in targets:
        row, direction = target["trigger_row"], target["side"]
        if row <= previous_row or direction not in (-1, 1) or target["day"] != day:
            raise ValueError("Нарушен порядок или направление сигналов дня")
        previous_row = row
        index = row  # Следующая строка после trigger_row при индексации с нуля.
        if index >= close_index:
            break
        if position is not None and position["direction"] == direction:
            continue
        mark_to(index)
        if position is not None:
            close_position(index, "Смена цвета", target)
        actual_time = str(stamps[index])[:19]
        if not f"{day} {session.start}" <= actual_time < f"{day} {session.entry_end}":
            continue
        position = dict(target, side="Long" if direction == 1 else "Short", direction=direction,
            entry_row=index + 1, entry_time=str(stamps[index]), entry_price=float(prices[index]))
        mark_start = index
        record_path([realized - fee / 2])
    if position is not None:
        mark_to(close_index)
        close_position(close_index, "Закрытие сессии")
    gross = float(sum(t["gross_pnl"] for t in trades))
    return dict(trades=trades, daily=dict(day=day, trades=len(trades), gross_pnl=gross,
        net_pnl=realized, equity_peak=peak, equity_min=minimum, max_drawdown=max_dd,
        signals=len(targets), cost_ticks=cost_ticks))


def _load_database(path, symbol, start, end, dataset_id):
    """Читает OHLC и весь журнал до end для symbol/dataset_id из path.

    Возвращает HA-бары, журнал и метаданные. История до start используется
    только для причинного прогрева, но её источники тоже обязательно сверяются.
    Границы метаданных результата относятся к start–end. Других индикаторов нет.
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
            "AND day<=? ORDER BY day", db, params=(dataset_id, symbol, end))
    selected = sources.loc[sources.day >= start]
    if bars.empty or selected.empty:
        raise ValueError(f"{symbol}: нет данных в заданном диапазоне")
    if bars.duplicated(["day", "bar_index"]).any():
        raise ValueError("Повторные номера баров")
    if not np.isfinite(bars[["open", "high", "low", "close"]].to_numpy()).all():
        raise ValueError("Некорректные цены баров")
    if not set(bars.day).issubset(set(sources.loc[sources.status == "ready", "day"])):
        raise ValueError("Бары не имеют готового источника в журнале")
    ready = selected.loc[selected.status == "ready"]
    metadata = dict(db=str(path), symbol=symbol, dataset_id=dataset_id, bar_config=json.loads(config[0]),
        first_day=str(selected.day.min()), last_day=str(selected.day.max()),
        first_ready_day=str(ready.day.min()), last_ready_day=str(ready.day.max()), ready_days=len(ready),
        source_days=len(selected), history_source_days=len(sources),
        history_ready_days=int((sources.status == "ready").sum()),
        warmup_bars=int((bars.day < start).sum()),
        bars_sha256=sha256(bars.to_json(orient="split", double_precision=15).encode()).hexdigest(),
        source_manifest_sha256=sha256(sources[["day", "sha256"]].to_json(orient="records").encode()).hexdigest())
    return heikin_ashi_frame(bars), sources, metadata


def _write_report(folder, summary, daily, metadata):
    """Сохраняет в folder автономный HTML из summary/daily/metadata; возвращает путь."""
    parts = []
    for symbol, frame in daily.groupby("symbol", sort=False):
        base = frame[frame.cost_ticks == metadata["settings"]["base_cost"]]
        current = base[(base.day >= "2026-01-01") & (base.day < "2027-01-01")]
        for title, selected in (("Вся история", base), ("Только 2026 год", current)):
            if selected.empty:
                continue
            figure = go.Figure()
            for count, series in selected.groupby("confirmation", sort=False):
                figure.add_trace(go.Scatter(x=series.day, y=series.net_pnl.cumsum(),
                    name=f"Подтверждение: {count}", mode="lines"))
            figure.update_layout(template="plotly_white", title=f"{symbol}: {title}, после затрат",
                yaxis_title="Пункты на один контракт", height=430, hovermode="x unified",
                legend=dict(orientation="h", y=-0.2), margin=dict(b=90))
            parts.append(figure.to_html(full_html=False, include_plotlyjs=False))
    coverage = ''.join('<li>' + escape(f'{row["symbol"]}: {row["first_ready_day"]} — '
        f'{row["last_ready_day"]}; включено {row["included_days"]}, исключено '
        f'{row["excluded_days"]} дней журнала.') + '</li>' for row in metadata["datasets"])
    html = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Тест Heikin Ashi</title>'
        '<style>body{font:16px Segoe UI,sans-serif;margin:30px auto;max-width:1400px;padding:20px;'
        'color:#24354b;background:#f5f7fb}table{border-collapse:collapse;background:white;font-size:14px}'
        'td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#e8eef6}'
        '.scroll{overflow-x:auto}pre{white-space:pre-wrap}a{color:#165ab6}</style>'
        '<h1>Heikin Ashi на дельта-барах RTS и MIX</h1>'
        '<p>Только цвет HA: покупка после зелёной свечи, продажа после красной. '
        'Второй фиксированный вариант требует две свечи одного цвета подряд. '
        'Доджи удерживает позицию и сбрасывает серию. При противоположном подтверждении — переворот.</p>'
        '<p>Исполнение по следующему исходному тику после закрытия полного бара; '
        'HA-цены не используются для исполнения. Стопов, целей и других индикаторов нет. '
        'Входы 10:00–18:30, после 18:30 только выход; закрытие первым тиком с 18:40 МСК.</p>'
        '<p>HA рассчитывается по полной прошлой истории, включая вечерние и ночные бары. '
        'Неполные остатки пропущены, рекурсия через даты продолжается. '
        'Серия торгового подтверждения сбрасывается ежедневно.</p>'
        '<h2>Покрытие</h2><ul>' + coverage + '</ul><h2>Сводка и затраты</h2>'
        '<div class="scroll">' + _table(summary.rename(columns={"confirmation": "Свечей подтверждения"}))
        + '</div><h2>Графики при опорных затратах</h2>'
        '<script>' + get_plotlyjs() + '</script>' + ''.join(parts)
        + '<h2>Ограничения</h2><p>' + escape(LIMITATIONS) + '</p>'
        '<p>Формулы: <a href="https://www.tradingview.com/support/solutions/43000619436-understanding-heikin-ashi-charts/">'
        'TradingView</a>. <a href="https://www.tradingview.com/support/solutions/43000481029-strategy-produces-unrealistic-results-on-non-standard-chart-types-heikin-ashi-renko-etc/">'
        'Почему нельзя исполнять по синтетическим ценам</a>.</p>'
        '<p><a href="summary.csv">Сводка</a> · <a href="daily.csv">Дневные результаты</a> · '
        '<a href="trades.csv">Сделки при опорных затратах</a> · '
        '<a href="coverage.csv">Покрытие</a> · <a href="metadata.json">Настройки и контрольные суммы</a></p>'
        '<pre>' + escape(json.dumps(metadata["settings"], ensure_ascii=False, indent=2)) + '</pre></html>')
    path = folder / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def parse_args(argv=None):
    """Проверяет список параметров argv; возвращает настройки фиксированного теста."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("C:/data_quote/delta_bars.sqlite3"))
    parser.add_argument("--symbols", nargs="+", choices=["RTS", "MIX"], default=["RTS", "MIX"])
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="9999-12-31")
    parser.add_argument("--dataset-id")
    parser.add_argument("--confirmations", default="1,2")
    parser.add_argument("--costs", default="0,2,4,8")
    parser.add_argument("--base-cost", type=float, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/heikin_ashi"))
    args = parser.parse_args(argv)
    try:
        if date.fromisoformat(args.start) > date.fromisoformat(args.end):
            raise ValueError("Начало позже конца")
        args.confirmations = sorted(set(int(x) for x in args.confirmations.split(",")))
        args.costs = sorted(set(float(x) for x in args.costs.split(",")))
        if not args.confirmations or min(args.confirmations) < 1:
            raise ValueError("Число подтверждений должно быть положительным целым")
        if (not args.costs or not np.isfinite(args.costs).all() or min(args.costs) < 0
                or args.base_cost not in args.costs):
            raise ValueError("Затраты неотрицательны; --base-cost должен входить в --costs")
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    """Выполняет тест по argv, сохраняет результаты и возвращает путь HTML-отчёта.

    argv — список аргументов либо None для CLI. SHA256 исходников читаются
    от PROJECT_ROOT; относительный --output считается от рабочей папки.
    """
    args = parse_args(argv)
    session = Session()
    stamp = datetime.now(timezone(timedelta(hours=3))).strftime("%Y%m%d_%H%M%S_%f")
    folder = args.output.resolve() / ("run_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    metadata = dict(settings={key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()}, session=vars(session), units="пункты на один контракт",
        limitations=LIMITATIONS, datasets=[], implementation_sha256={name: sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in ("backtest/backtest_heikin_ashi.py", "backtest/backtest_stochastic.py", "source/duration_data.py")})
    all_trades, all_daily, coverage, summary = [], [], [], []
    for symbol in dict.fromkeys(args.symbols):
        tick_size = {"RTS": 10, "MIX": 25}[symbol]
        frame, sources, source_meta = _load_database(args.db, symbol, args.start, args.end, args.dataset_id)
        source_meta["tick_size"] = tick_size
        metadata["datasets"].append(source_meta)
        bars_by_day = dict(tuple(frame.groupby("day", sort=False)))
        grouped_targets = {}
        for confirmation in args.confirmations:
            grouped = {}
            for target in color_targets(frame, confirmation, session):
                if target["day"] >= args.start:
                    grouped.setdefault(target["day"], []).append(target)
            grouped_targets[confirmation] = grouped
        print(f"{symbol}: {source_meta['source_days']} дней теста; для сверки "
              f"{len(sources)} дней истории, {source_meta['history_ready_days']} готовых архивов", flush=True)
        for count, source in enumerate(sources.itertuples(), 1):
            warmup = source.day < args.start
            if source.status != "ready":
                reason = "Прогрев построения дельта-баров" if source.status == "warmup" else f"Статус: {source.status}"
                coverage.append(dict(date=source.day, status="warmup_unavailable" if warmup else "excluded", reason=reason,
                    bars=source.bar_count, ticks=source.tick_count, symbol=symbol))
                continue
            if source.day not in bars_by_day:
                raise ValueError(f"{source.day}: готовый день без баров")
            bars = bars_by_day[source.day]
            ticks = read_tick_zip(Path(source.file_path), source.sha256)
            if len(ticks) != source.tick_count:
                raise ValueError(f"{source.day}: число тиков не совпало")
            validate_extremes(bars, ticks["last"].to_numpy(), source.bar_count)
            prepared, audit = build_day(source.day, bars, ticks, session)
            if warmup:
                audit.update(status="validated_warmup", reason="Сверена история прогрева HA, без сделок")
            coverage.append(dict(audit, symbol=symbol))
            prices, stamps = ticks["last"].to_numpy(), ticks.datetime.astype(str).to_numpy()
            if not np.allclose(prices / tick_size, np.round(prices / tick_size), rtol=0, atol=1e-7):
                raise ValueError(f"{symbol} {source.day}: цены вне решётки шага")
            if prepared is not None and not warmup:
                for confirmation in args.confirmations:
                    signature = None
                    for cost in args.costs:
                        result = simulate_day(source.day, grouped_targets[confirmation].get(source.day, []),
                            prices, stamps, audit["close_row"], tick_size, cost, session)
                        current = [(t["entry_row"], t["exit_row"], t["side"]) for t in result["trades"]]
                        if signature is not None and signature != current:
                            raise AssertionError("Издержки изменили моменты сделок")
                        signature = current
                        all_daily.append(dict(result["daily"], symbol=symbol, confirmation=confirmation))
                        if cost == args.base_cost:
                            all_trades.extend(dict(t, symbol=symbol, confirmation=confirmation) for t in result["trades"])
            if count % 100 == 0 or count == len(sources):
                print(f"{symbol}: проверено {count}/{len(sources)} дней", flush=True)
        source_meta["included_days"] = sum(x["symbol"] == symbol and x["status"] == "included" for x in coverage)
        source_meta["excluded_days"] = sum(x["symbol"] == symbol and x["status"] == "excluded" for x in coverage)
        source_meta["validated_warmup_days"] = sum(
            x["symbol"] == symbol and x["status"] == "validated_warmup" for x in coverage)
        for confirmation in args.confirmations:
            trades = [t for t in all_trades if t["symbol"] == symbol and t["confirmation"] == confirmation]
            for cost in args.costs:
                days = [d for d in all_daily if d["symbol"] == symbol and d["confirmation"] == confirmation
                        and d["cost_ticks"] == cost]
                for period, left, right in report_periods(source_meta["first_day"], source_meta["last_day"]):
                    period_trades = [t for t in trades if left <= t["day"] < right]
                    period_days = [d for d in days if left <= d["day"] < right]
                    stats = _summarize(period_trades, period_days, cost, tick_size)
                    if not np.isclose(stats["net_pnl"], sum(d["net_pnl"] for d in period_days)):
                        raise AssertionError("Журнал сделок не совпал с дневным результатом")
                    summary.append(dict(symbol=symbol, strategy=f"HA: {confirmation}",
                        confirmation=confirmation, period=period, cost_ticks=cost, **stats))
    if not all_daily:
        raise ValueError("Нет дней с исполнимым закрытием")
    summary_frame, daily_frame = pd.DataFrame(summary), pd.DataFrame(all_daily)
    for name, frame in (("summary", summary_frame), ("daily", daily_frame),
            ("trades", pd.DataFrame(all_trades).reindex(columns=TRADE_COLUMNS)),
            ("coverage", pd.DataFrame(coverage))):
        frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig")
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _write_report(folder, summary_frame, daily_frame, metadata)
    print(summary_frame.to_string(index=False), flush=True)
    print(f"Отчёт: {report}", flush=True)
    return report


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f"Ошибка теста: {error}", file=sys.stderr)
        raise SystemExit(1)
