"""Автономный интерактивный отчёт исследования длительности дельта-баров.

Отчёт содержит встроенный Plotly и открывается без подключения к интернету.
Результаты передаются как обычный словарь; финансовые расчёты здесь не меняются.

Примеры запуска из корня проекта:
    python -m unittest tests.test_duration_report -v
    python backtest_duration.py --symbols RTS MIX
"""

from datetime import date, datetime
from html import escape
import json
import math
from numbers import Real
from pathlib import Path
from urllib.parse import urlsplit

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


COLORS = ["#2367b7", "#dc7637", "#288b7d", "#9864ad", "#bc5477", "#62884a",
          "#988033", "#5b8095", "#b86653", "#7273b6", "#388d9b", "#72777f"]
LABELS = {
    "label": "Вход / выход, с", "entry_seconds": "Вход, с", "exit_seconds": "Выход, с",
    "selection_basis": "Основание сравнения", "robust_score": "Качество соседней области",
    "training_eligible": "Прошла отбор на обучении", "parameter_index": "Номер пары",
    "test_pnl": "PnL проверки", "test_trades": "Сделок на проверке", "train_pnl": "PnL обучения",
    "close_time": "Тик дневного выхода", "close_delay_seconds": "Задержка выхода, с", "close_row": "Строка выхода",
    "development_pnl": "PnL разработки", "holdout_pnl": "PnL контроля",
    "development_trades": "Сделок: разработка", "holdout_trades": "Сделок: контроль",
    "holdout_max_dd": "Макс. просадка: контроль", "holdout_profit_factor": "Profit factor: контроль",
    "holdout_win_rate": "Доля прибыльных: контроль", "holdout_long_pnl": "PnL покупок: контроль",
    "holdout_short_pnl": "PnL продаж: контроль", "date": "Дата", "status": "Статус",
    "reason": "Причина", "year": "Год", "trades": "Сделок", "net_pnl": "Чистый PnL",
    "gross_pnl": "PnL до затрат", "pnl": "PnL", "net": "Чистый PnL", "gross": "PnL до затрат",
    "entry_time": "Время входа", "exit_time": "Время выхода", "entry_price": "Цена входа",
    "exit_price": "Цена выхода", "direction": "Направление", "side": "Сторона",
    "entry_signal_close": "Закрытие сигнального бара", "entry_signal_alf": "ALF сигнального бара",
    "entry_filter": "Фильтр входа", "entry_rule": "Условия входа",
    "position_direction": "Направление входов", "direction_multiplier": "Множитель направления",
    "entry_script": "Скрипт запуска",
    "alf_history_start": "Начало истории ALF", "alf_warmup_bars": "Баров прогрева до начала теста",
    "alf_history_sha256": "Контрольная сумма истории ALF",
    "symbol": "Инструмент", "contract": "Контракт", "entry_date": "Дата входа",
    "exit_date": "Дата выхода", "exit_reason": "Причина выхода", "duration": "Длительность",
    "units": "Единицы результата", "tick_size": "Шаг цены", "session": "Торговая сессия",
    "base_cost_ticks": "Базовые затраты, тиков", "dates": "Периоды данных",
    "development_start": "Начало разработки", "development_end": "Конец разработки",
    "holdout_start": "Начало контроля", "holdout_end": "Конец контроля",
    "disclaimer": "Ограничения исследования", "train_start": "Начало обучения",
    "train_end": "Конец обучения", "test_start": "Начало проверки", "test_end": "Конец проверки",
    "selected_label": "Выбранная пара", "selected_entry": "Выбранный вход, с",
    "selected_exit": "Выбранный выход, с", "cost_ticks": "Затраты, тиков",
    "rows": "Строк", "bars": "Баров", "ticks": "Тиков", "files": "Файлов",
    "fold": "Окно", "no_trade": "Без торговли", "selection_reason": "Причина выбора",
    "max_dd": "Максимальная просадка", "profit_factor": "Profit factor", "win_rate": "Доля прибыльных",
}
SUMMARY_COLUMNS = ["label", "selection_basis", "entry_seconds", "exit_seconds", "development_pnl", "holdout_pnl",
                   "development_trades", "holdout_trades", "holdout_max_dd", "holdout_profit_factor",
                   "holdout_win_rate", "holdout_long_pnl", "holdout_short_pnl"]


def _clean(value):
    """Переводит данные в обычный JSON, заменяя неопределённые числа на null."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real):
        if not math.isfinite(value):
            return None
        return int(value) if isinstance(value, int) else float(value)
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _json(value):
    """Сериализует данные без возможности завершить встроенный тег script."""
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _display(value):
    """Форматирует одно значение для безопасного вывода в таблице."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, Real):
        return f"{value:,.2f}".rstrip("0").rstrip(".").replace(",", "\u202f").replace(".", ",")
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return escape(str(value))


def _table(rows, columns=None):
    """Создаёт безопасную прокручиваемую таблицу с сортировкой в браузере."""
    if not rows:
        return '<div class="empty">Нет данных для этой таблицы.</div>'
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    headers = "".join(f'<th scope="col"><button type="button" data-sort="{index}">{escape(LABELS.get(key, key))}<span aria-hidden="true"> ↕</span></button></th>'
                      for index, key in enumerate(columns))
    body = []
    for row in rows:
        cells = []
        for key in columns:
            value = row.get(key)
            numeric = isinstance(value, Real) and not isinstance(value, bool)
            tone = " negative" if numeric and value < 0 else ""
            sort_value = str(value) if numeric else str(value or "")
            cells.append(f'<td class="{"number" if numeric else "text"}{tone}" data-value="{escape(sort_value, quote=True)}" data-numeric="{str(numeric).lower()}">{_display(value)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table-scroll" tabindex="0"><table><thead><tr>' + headers + '</tr></thead><tbody>' + "".join(body) + "</tbody></table></div>"


def _layout(title, unit, holdout_start=None, height=430):
    """Задаёт общий светлый стиль графиков и границу контрольного периода."""
    layout = dict(template="none", title=dict(text=title, font=dict(size=16)), height=height,
                  paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=70, r=25, t=65, b=85),
                  font=dict(family="Segoe UI, Arial, sans-serif", color="#23364b", size=12),
                  hovermode="x unified", dragmode="zoom", colorway=COLORS,
                  legend=dict(orientation="h", y=-0.2, x=0, groupclick="togglegroup"),
                  xaxis=dict(showgrid=False, zeroline=False, title=None),
                  yaxis=dict(title=unit, gridcolor="#e7edf3", zeroline=True,
                             zerolinecolor="#73869b", zerolinewidth=1, rangemode="tozero"))
    if holdout_start:
        layout["shapes"] = [dict(type="line", x0=holdout_start, x1=holdout_start, xref="x",
                                 y0=0, y1=1, yref="paper", line=dict(color="#8897a7", width=1.5, dash="dot"))]
        layout["annotations"] = [dict(x=holdout_start, y=1, xref="x", yref="paper", text="Контроль с "+escape(holdout_start),
                                      showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11, color="#67798c"))]
    return layout


def _curve_figure(rows, value_key, title, unit, holdout_start, x_key="dates"):
    """Строит линейные серии в порядке разработки с пятью видимыми вариантами."""
    figure = go.Figure(layout=_layout(title, unit, holdout_start))
    for index, row in enumerate(rows):
        figure.add_trace(go.Scatter(x=list(row.get(x_key, [])), y=list(row.get(value_key, [])),
                                    name=escape(str(row.get("label", f"Пара {index + 1}"))),
                                    mode="lines", visible=True if index < 5 else "legendonly",
                                    line=dict(color=COLORS[index % len(COLORS)], width=1.9),
                                    connectgaps=False,
                                    hovertemplate="%{x}<br>%{fullData.name}: %{y:,.2f}<extra></extra>"))
    return figure


def _build_figures(payload):
    """Готовит графики из дневных агрегатов без расчёта или отбора стратегий."""
    figures = {}
    meta = payload.get("meta") or {}
    unit = escape(str(meta.get("units", "пункты котировки на 1 контракт")))
    holdout_start = str(meta.get("holdout_start") or "2026-01-01")
    curves = payload.get("curves") or []
    if curves:
        equity = _curve_figure(curves, "net", "Накопленный PnL после затрат", unit, holdout_start)
        equity.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0, y=1.17,
                                               buttons=[dict(label="После затрат", method="update",
                                                             args=[{"y": [list(row.get("net", [])) for row in curves]},
                                                                   {"title.text": "Накопленный PnL после затрат"}]),
                                                        dict(label="До затрат", method="update",
                                                             args=[{"y": [list(row.get("gross", [])) for row in curves]},
                                                                   {"title.text": "Накопленный PnL до затрат"}])])])
        equity.update_layout(margin=dict(t=100))
        figures["equity"] = equity
        control=[]
        for row in curves:
            indexes=[i for i,stamp in enumerate(row.get('dates',[])) if stamp>=holdout_start]
            if not indexes:
                continue
            first=indexes[0]
            baseline=row['net'][first-1] if first else 0
            values=[row['net'][i]-baseline for i in indexes]
            control.append(dict(label=row.get('label',''),dates=[holdout_start]+[row['dates'][i] for i in indexes],net=[0]+values))
        if control:
            figures['holdout-equity']=_curve_figure(control,'net','Только контроль: PnL после затрат от нуля',unit,None)
        figures["drawdown"] = _curve_figure(curves, "drawdown", "Просадка по итогам дня", unit, holdout_start)
        intraday = [dict(row, _color_index=index) for index, row in enumerate(curves)
                    if row.get("intraday_drawdown")]
        if intraday:
            figures["intraday-drawdown"] = _curve_figure(intraday, "intraday_drawdown", "Минимум просадки внутри дня", unit, holdout_start)
            for trace, row in zip(figures["intraday-drawdown"].data, intraday):
                trace.line.color = COLORS[row["_color_index"] % len(COLORS)]
    monthly = payload.get("monthly") or []
    if monthly:
        fig = go.Figure(layout=_layout("Результат каждого месяца после затрат", unit, holdout_start))
        fig.update_layout(barmode="group")
        fig.update_xaxes(type="date", tickformat="%m.%Y")
        for index, row in enumerate(monthly):
            months = [month + "-01" if len(str(month)) == 7 else month for month in row.get("months", [])]
            fig.add_trace(go.Bar(x=months, y=list(row.get("pnl", [])), name=escape(str(row.get("label", ""))),
                                 marker_color=COLORS[index % len(COLORS)],
                                 visible=True if index < 5 else "legendonly",
                                 hovertemplate="%{x|%m.%Y}<br>%{fullData.name}: %{y:,.2f}<extra></extra>"))
        figures["monthly"] = fig
    for index, row in enumerate(payload.get("heatmaps") or []):
        fig = go.Figure(layout=_layout(escape(str(row.get("title", "Разработка 2022–2025"))), unit, height=400))
        value_label=escape(str(row.get('value_label','PnL')))
        kind=row.get('kind','diverging')
        scale=[[0,"#f4f6f9"],[1,"#b74950" if kind=='risk' else '#276fad']] if kind!='diverging' else [[0,"#b74950"],[0.5,"#f4f6f9"],[1,"#276fad"]]
        fig.add_trace(go.Heatmap(x=list(row.get("entries", [])), y=list(row.get("exits", [])),
                                 z=[list(values) for values in row.get("z", [])], zmid=0 if kind=='diverging' else None,
                                 colorscale=scale,
                                 colorbar=dict(thickness=12, title=dict(text=value_label)), hoverongaps=False,
                                 hovertemplate="Вход: %{x} с<br>Выход: %{y} с<br>"+value_label+": %{z:,.2f}<extra></extra>"))
        fig.update_xaxes(title="Порог входа, с", type="category", zeroline=False)
        fig.update_yaxes(title="Порог выхода, с", type="category", zeroline=False, rangemode="normal")
        figures[f"heatmap-{index}"] = fig
    costs = payload.get("costs") or []
    if costs:
        fig = go.Figure(layout=_layout("Чувствительность к затратам исполнения", unit))
        fig.update_xaxes(title="Затраты, тиков (согласно модели исполнения)")
        for index, row in enumerate(costs):
            for key, suffix, dash in (("development_pnl", "разработка", "solid"), ("holdout_pnl", "контроль", "dash")):
                fig.add_trace(go.Scatter(x=list(row.get("cost_ticks", [])), y=list(row.get(key, [])),
                                         name=escape(str(row.get("label", ""))) + " · " + suffix,
                                         legendgroup=str(index), mode="lines+markers", connectgaps=False,
                                         visible=True if index < 5 else "legendonly",
                                         line=dict(color=COLORS[index % len(COLORS)], dash=dash),
                                         hovertemplate="%{x} тиков<br>%{fullData.name}: %{y:,.2f}<extra></extra>"))
        if meta.get("base_cost_ticks") is not None:
            fig.add_vline(x=meta["base_cost_ticks"], line_dash="dot", line_color="#73869b")
        figures["costs"] = fig
    wf = payload.get("walk_forward") or {}
    if wf.get("dates"):
        fig = go.Figure(layout=_layout("Последовательная проверка: PnL", unit, holdout_start))
        for key, label, dash in (("net", "После затрат", "solid"), ("gross", "До затрат", "dot")):
            fig.add_trace(go.Scatter(x=list(wf.get("dates", [])), y=list(wf.get(key, [])), name=label,
                                     mode="lines", line=dict(color="#2367b7" if key == "net" else "#a0b6cf", dash=dash),
                                     hovertemplate="%{x}<br>%{fullData.name}: %{y:,.2f}<extra></extra>"))
        figures["walk-forward"] = fig
        figures["walk-forward-dd"] = _curve_figure([dict(wf, label="Последовательная проверка")], "drawdown",
                                                   "Последовательная проверка: просадка", unit, holdout_start)
    return {key: fig.to_plotly_json() for key, fig in figures.items()}


def _chart(figures, key, extra_class=""):
    """Создаёт контейнер интерактивного графика или видимое пустое состояние."""
    if key not in figures:
        return '<div class="empty">Нет данных для этого графика.</div>'
    return f'<div class="chart {extra_class}" id="{key}" role="img" aria-label="Интерактивный график"></div>'


def _links(items):
    """Добавляет ссылки на результаты, отбрасывая исполняемые схемы адресов."""
    links = []
    for item in items:
        href = str(item.get("href", "")).strip()
        if not href or any(ord(char) < 32 for char in href):
            continue
        if urlsplit(href).scheme.lower() not in ("", "http", "https"):
            continue
        links.append(f'<a href="{escape(href, quote=True)}">{escape(str(item.get("title", href)))}</a>')
    return '<div class="related">' + " · ".join(links) + "</div>" if links else ""


def write_report(output_dir, payload):
    """Сохраняет автономный report.html и возвращает путь к созданному файлу.

    payload содержит symbol, meta и списки summary, curves, heatmaps, monthly,
    costs, coverage, trades; walk_forward и links необязательны. Серии остаются
    в порядке вызывающей стороны. Первые пять пар видимы сразу, остальные
    включаются щелчком по легенде. Недоступные метрики отображаются как тире.
    """
    payload = _clean(payload)
    meta = payload.get("meta") or {}
    figures = _build_figures(payload)
    summary = payload.get("summary") or []
    first = summary[0] if summary else {}
    symbol = escape(str(payload.get("symbol", "Инструмент")))
    units = escape(str(meta.get("units", "пункты котировки на 1 контракт")))
    development_label=escape(str(meta.get('development_start','2022'))+' — '+str(meta.get('development_end','2025'))) if meta.get('development_start') else '2022–2025'
    control_label=escape(str(meta.get('holdout_start','2026-01-01')))
    base_label=_display(meta.get('base_cost_ticks'))
    direction_label=_display(meta.get('position_direction','прямое'))
    cards = [("Сравниваемых пар", len(summary)),
             ("Первая пара по порядку разработки", first.get("label")),
             ("Её PnL · разработка", first.get("development_pnl")),
             ("Её PnL · контроль", first.get("holdout_pnl"))]
    card_html = "".join(f'<div class="stat"><div class="stat-label">{label}</div><strong class="{"negative" if isinstance(value, Real) and value < 0 else ""}">{_display(value)}</strong></div>'
                        for label, value in cards)
    metadata = "".join(f'<dt>{escape(LABELS.get(key, key))}</dt><dd>{_display(value)}</dd>' for key, value in meta.items())
    heatmaps = "".join(_chart(figures, f"heatmap-{index}") for index in range(len(payload.get("heatmaps") or [])))
    if not heatmaps:
        heatmaps = '<div class="empty">Нет данных для тепловых карт.</div>'
    coverage = payload.get("coverage") or []
    status_counts = {}
    for row in coverage:
        status = str(row.get("status", "Не указан"))
        status_counts[status] = status_counts.get(status, 0) + 1
    coverage_badges = " ".join(f'<span class="badge">{escape(status)}: {count}</span>' for status, count in status_counts.items())
    wf = payload.get("walk_forward") or {}
    disclaimer = _display(meta.get("disclaimer") or "Исторический результат не гарантирует прибыль в будущем. Выводы ограничены доступными данными и принятой моделью исполнения.")
    entry_rule = ('<p><strong>Условия входа</strong>: ' + _display(meta['entry_rule']) + '</p>') if meta.get('entry_rule') else ''
    page = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{symbol} · Длительность дельта-баров</title><style>{_CSS}</style></head>
<body><div class="page">
<header><div class="eyebrow">ИССЛЕДОВАНИЕ ДЕЛЬТА-БАРОВ</div><h1>{symbol}<span>Длительность → торговый результат</span></h1>
<p class="lead">Сравнение пар порогов входа и выхода, устойчивости результата и затрат исполнения.</p>
<div class="badges"><span class="badge">Направление входов · {direction_label}</span><span class="badge development">Разработка · {development_label}</span><span class="badge holdout">Контроль · с {control_label}</span><span class="badge">Опорные издержки · {base_label} шагов за круг</span><span class="badge">Автономный HTML · без интернета</span></div>
{_links(payload.get("links") or [])}</header>
<nav aria-label="Разделы отчёта"><a href="#comparison">PnL</a><a href="#risk">Просадки</a><a href="#months">Месяцы</a><a href="#maps">Тепловые карты</a><a href="#execution">Затраты</a><a href="#results">Таблица</a><a href="#forward">Последовательная проверка</a><a href="#data">Данные</a></nav>
<div class="stat-grid">{card_html}</div>
<aside class="method"><strong>Как читать результат</strong><p>Фиксированные пары отбираются только на периоде разработки {development_label}. Контроль с {control_label} не участвует в выборе порогов. Порядок кривых и таблицы задан отбором на разработке; значение контрольного PnL не определяет порядок. Пар, прошедших условия отбора на разработке: <strong>{_display(meta.get('training_eligible_pairs'))}</strong>.</p>
{entry_rule}<p>Единицы PnL: <strong>{units}</strong>; это не рубли. PnL после затрат включает принятую модель исполнения. {disclaimer}</p></aside>
<section id="comparison"><div class="section-heading"><span class="index">01</span><div><h2>Накопленный результат</h2><p>Каждая линия — фиксированная пара порогов. Пунктир отмечает начало контроля.</p></div></div>
<p class="hint">Щелчок по легенде включает или скрывает линию; двойной щелчок оставляет одну. Выделение мышью приближает участок, двойной щелчок по графику возвращает масштаб. Первые пять пар включены по умолчанию.</p>
{_chart(figures, "equity")}<h3>Контрольный участок отдельно</h3><p class="hint">Накопленная прибыль периода подбора обнулена. Пары и их порядок сохранены.</p>{_chart(figures, "holdout-equity")}</section>
<section id="risk"><div class="section-heading"><span class="index">02</span><div><h2>Просадки и риск открытой позиции</h2><p>Просадка — снижение от предыдущего максимума результата; значения ниже нуля.</p></div></div>
{_chart(figures, "drawdown")}<div class="note">Дневная кривая может скрывать риск открытой позиции. Ниже показан минимум просадки за день по переоценке на открытиях/закрытиях дельта-баров и тике дневного выхода. Экстремумы между этими точками не измерены; это не непрерывная тиковая оценка риска.</div>
{_chart(figures, "intraday-drawdown")}</section>
<section id="months"><div class="section-heading"><span class="index">03</span><div><h2>Месячный PnL</h2><p>Распределение результата после затрат по месяцам: устойчивость важнее одного удачного участка.</p></div></div>{_chart(figures, "monthly")}</section>
<section id="maps"><div class="section-heading"><span class="index">04</span><div><h2>Области параметров на разработке</h2><p>Только {development_label}. По горизонтали — порог входа, по вертикали — порог выхода. Пустая ячейка означает отсутствие значения.</p></div></div><div class="heatmap-grid">{heatmaps}</div></section>
<section id="execution"><div class="section-heading"><span class="index">05</span><div><h2>Чувствительность к затратам</h2><p>Сплошные линии — разработка, штриховые — контроль. Вертикаль отмечает базовые затраты, если они заданы.</p></div></div>{_chart(figures, "costs")}</section>
<section id="results"><div class="section-heading"><span class="index">06</span><div><h2>Сравнение фиксированных пар</h2><p>Все показатели относятся к указанному периоду. «—» означает недоступную или неопределённую метрику. Щелчок по заголовку сортирует столбец.</p></div></div>{_table(summary, SUMMARY_COLUMNS)}</section>
<section id="forward"><div class="section-heading"><span class="index">07</span><div><h2>Последовательная проверка во времени</h2><p>Пара выбирается на прошлом окне и применяется к следующему. Таблица показывает выбор в каждом окне; окна без сделок или без выбранной пары сохраняются.</p></div></div>
{_chart(figures, "walk-forward")}{_chart(figures, "walk-forward-dd")}<h3>Окна отбора и проверки</h3>{_table(wf.get("folds") or [])}</section>
<section id="data"><div class="section-heading"><span class="index">08</span><div><h2>Данные и воспроизводимость</h2><p>Покрытие дат, настройки расчёта и примеры сделок для проверки результата.</p></div></div>
<details><summary>Настройки и ограничения</summary><dl class="metadata">{metadata or '<dt>Метаданные</dt><dd>Нет данных</dd>'}</dl></details>
<details><summary>Покрытие данных · {len(coverage)} записей</summary><div class="badges">{coverage_badges}</div>{_table(coverage)}</details>
<details><summary>Примеры сделок · до 200 записей</summary><p class="hint">Первые записи переданной выборки представительной стратегии; это не полный журнал.</p>{_table((payload.get("trades") or [])[:200])}</details></section>
<footer>{symbol} · Исследование исторических данных · Все графики и данные встроены в этот файл.</footer>
</div><script id="plotly-library">{get_plotlyjs()}</script>
<script id="report-data" type="application/json">{_json(payload)}</script>
<script id="figure-data" type="application/json">{_json(figures)}</script>
<script>{_JS}</script></body></html>'''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path


_CSS = r'''
:root{--ink:#203449;--muted:#607487;--line:#e0e7ef;--blue:#2367b7;--bg:#f1f5f9}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:74px}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.55 "Segoe UI",Arial,sans-serif}.page{max-width:1500px;margin:auto;padding:38px 32px 24px}header{padding:8px 4px 24px}.eyebrow{font-size:11px;letter-spacing:2px;font-weight:700;color:var(--blue)}h1{font-size:43px;line-height:1.12;letter-spacing:-1px;margin:12px 0 14px}h1 span{display:block;font-size:27px;font-weight:500;letter-spacing:-.4px;margin-top:7px}h2{font-size:23px;letter-spacing:-.4px;line-height:1.25;margin:0 0 7px}h3{font-size:17px;margin:24px 0 12px}p{margin:8px 0}.lead{font-size:17px;color:var(--muted);max-width:850px}.badges{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 2px}.badge{display:inline-block;background:#e7edf4;border:1px solid #d8e1eb;border-radius:6px;padding:5px 10px;font-size:12px;color:#52697f}.development{background:#e3edf9;color:#205894;border-color:#ccdef1}.holdout{background:#e4f2ee;color:#277662;border-color:#c9e5dc}.related{font-size:13px;margin-top:16px}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}nav{position:sticky;top:0;z-index:5;display:flex;gap:22px;padding:14px 17px;background:#fffffff2;backdrop-filter:blur(8px);border:1px solid var(--line);border-radius:9px;margin:0 0 22px;overflow:auto;white-space:nowrap;font-size:13px;box-shadow:0 3px 14px #20344907}.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}.stat{background:white;border:1px solid var(--line);border-radius:10px;padding:18px 20px;min-width:0}.stat-label{font-size:12px;color:var(--muted);min-height:20px}.stat strong{display:block;font-size:26px;margin-top:6px;font-weight:650;overflow-wrap:anywhere}.negative{color:#a94e53}.method{background:#eaf0f7;border-left:3px solid #6489b5;border-radius:0 9px 9px 0;padding:17px 21px;margin:0 0 24px;font-size:13px}.method p{max-width:1250px}.method strong{font-weight:650}section{background:#fff;border:1px solid var(--line);border-radius:12px;padding:25px 27px;margin-bottom:23px;overflow:hidden}.section-heading{display:flex;gap:16px;margin-bottom:14px}.section-heading p{font-size:13px;color:var(--muted);margin:0}.index{display:flex;align-items:center;justify-content:center;background:#edf2f8;color:#547798;border-radius:8px;font-size:12px;font-weight:700;width:36px;height:36px;flex-shrink:0}.hint{font-size:12px;color:var(--muted);margin:8px 0 16px}.chart{width:100%;min-height:420px}.heatmap-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.heatmap-grid .chart{min-height:390px}.note{background:#f6f8fb;border:1px solid #e7ecf2;border-radius:7px;padding:12px 16px;font-size:13px;color:#617388;margin:10px 0 18px}.empty{background:#f7f9fc;border:1px dashed #d5dee8;border-radius:8px;padding:24px;color:#738296;font-size:14px;margin:12px 0}.table-scroll{overflow:auto;max-height:670px;border:1px solid var(--line);border-radius:7px;margin:15px 0}table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th{position:sticky;top:0;z-index:1;background:#f1f5f9;text-align:left;border-bottom:1px solid #d6e0ea;min-width:95px}th button{font:inherit;color:var(--muted);font-weight:600;border:0;background:none;text-align:left;padding:12px 11px;width:100%;cursor:pointer}th button:hover{color:var(--blue)}td{padding:10px 12px;border-bottom:1px solid #edf1f5;max-width:370px;overflow-wrap:anywhere}td.number{text-align:right;white-space:nowrap}td.text{min-width:90px}tr:hover td{background:#f5f8fb}tbody tr:last-child td{border-bottom:0}details{border-top:1px solid var(--line);padding:15px 0}details:last-child{padding-bottom:0}summary{font-weight:600;font-size:14px;cursor:pointer;color:#41627f}summary:hover{color:var(--blue)}.metadata{display:grid;grid-template-columns:minmax(160px,1fr) 3fr;gap:1px 0;background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden;font-size:12px}.metadata dt,.metadata dd{margin:0;padding:10px 12px;background:white;overflow-wrap:anywhere}.metadata dt{font-weight:600;color:var(--muted)}footer{font-size:11px;color:#738497;padding:6px 4px 16px}@media(max-width:1000px){.stat-grid{grid-template-columns:repeat(2,1fr)}.heatmap-grid{grid-template-columns:1fr}.page{padding:24px 17px}section{padding:22px 16px}}@media(max-width:600px){.page{padding:19px 10px}h1{font-size:34px}h1 span{font-size:23px}.stat{padding:13px}.stat strong{font-size:22px}.stat-label{min-height:36px}.method{padding:14px}.section-heading{gap:10px}h2{font-size:20px}nav{gap:18px;margin-bottom:15px}.metadata{grid-template-columns:1fr}.metadata dd{padding-top:0}.chart{min-height:400px}section{padding:19px 8px}footer{font-size:10px}}@media print{nav{position:static}.page{padding:0}section{break-inside:avoid}.stat-grid{grid-template-columns:repeat(4,1fr)}.modebar{display:none!important}}
'''

_JS = r'''
// Русские подписи встроенной панели Plotly.
Plotly.register({moduleType:"locale",name:"ru",dictionary:{
"Download plot as a PNG":"Скачать график PNG","Zoom":"Масштаб","Pan":"Перемещение",
"Zoom in":"Приблизить","Zoom out":"Отдалить","Autoscale":"Подобрать масштаб",
"Reset axes":"Сбросить оси","Toggle Spike Lines":"Линии к курсору",
"Show closest data on hover":"Ближайшая точка","Compare data on hover":"Сравнить точки",
"Double-click on legend to isolate one trace":"Двойной щелчок оставляет одну линию",
"Double-click to zoom back out":"Двойной щелчок сбрасывает масштаб",
"Produced with Plotly.js":"Создано с Plotly.js"},format:{decimal:",",thousands:" ",
days:["Воскресенье","Понедельник","Вторник","Среда","Четверг","Пятница","Суббота"],
shortDays:["Вс","Пн","Вт","Ср","Чт","Пт","Сб"],
months:["Январь","Февраль","Март","Апрель","Май","Июнь","Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"],
shortMonths:["Янв","Фев","Мар","Апр","Май","Июн","Июл","Авг","Сен","Окт","Ноя","Дек"],date:"%d.%m.%Y"}});
const figures=JSON.parse(document.getElementById("figure-data").textContent);
// Отрисовка использует только данные и библиотеку внутри текущего файла.
for(const [id,figure] of Object.entries(figures)){
  Plotly.newPlot(id,figure.data,figure.layout,{responsive:true,displaylogo:false,scrollZoom:true,locale:"ru",
    modeBarButtonsToRemove:["select2d","lasso2d"],toImageButtonOptions:{format:"png",scale:2,filename:id}});
}
// Сортировка меняет только видимую таблицу, не исходные расчёты или порядок графиков.
document.querySelectorAll("button[data-sort]").forEach(button=>button.addEventListener("click",()=>{
  const table=button.closest("table"),body=table.tBodies[0],column=Number(button.dataset.sort);
  const direction=button.dataset.direction==="asc"?-1:1;
  table.querySelectorAll("button[data-sort]").forEach(item=>{delete item.dataset.direction;item.parentElement.removeAttribute("aria-sort");});
  button.dataset.direction=direction===1?"asc":"desc";
  button.parentElement.setAttribute("aria-sort",direction===1?"ascending":"descending");
  const rows=Array.from(body.rows);
  rows.sort((left,right)=>{
    const a=left.cells[column],b=right.cells[column];
    if(a.dataset.value==="")return b.dataset.value===""?0:1;
    if(b.dataset.value==="")return -1;
    return direction*(a.dataset.numeric==="true"&&b.dataset.numeric==="true"
      ? Number(a.dataset.value)-Number(b.dataset.value)
      : a.dataset.value.localeCompare(b.dataset.value,"ru",{numeric:true}));
  });
  rows.forEach(row=>body.appendChild(row));
}));
'''
