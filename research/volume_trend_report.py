r"""Создаёт отчёт и автономные графики по сохранённому исследованию объёма.

Запуск из корня проекта:
    .\.venv\Scripts\python.exe -m research.volume_trend --stage report

Таблицы читаются из --output управляющего скрипта; вычисления не переобучаются.
Графики Plotly сохраняются в HTML с библиотекой внутри файла для работы офлайн.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.volume_trend_data import time_text


def number(value, digits=2):
    """Форматирует числовое value для русской таблицы либо возвращает прочерк."""
    return f"{value:.{digits}f}".replace(".", ",") if pd.notna(value) else "—"


def effect_table(data):
    """Возвращает Markdown-таблицу эффекта и индивидуального интервала для строк data."""
    lines = ["| Инструмент | База объёма | Показатель | Дней высокий/обычный | Обычный | Высокий | Изменение | 95% ДИ разницы | p Holm |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |"]
    for row in data.itertuples():
        lines.append(f"| {row.symbol} | {'среднее' if row.basis=='mean' else 'медиана'} | {row.metric} | {row.n_high}/{row.n_low} | {number(row.low_mean)} | {number(row.high_mean)} | {number(row.change_percent, 1)}% | [{number(row.ci_low)}; {number(row.ci_high)}] | {number(row.p_holm, 4)} |")
    return "\n".join(lines)


def primary_conclusion(main):
    """Возвращает вывод по таблице main; не смешивает отсутствующие p и отсутствие эффекта."""
    evaluated = int(main.p_holm.notna().sum())
    if not len(main) or evaluated < len(main):
        return f"Данных недостаточно для полного вывода по основным сравнениям: статистическая неопределённость оценена для {evaluated} из {len(main)} проверок. Неоценённые эффекты нельзя считать незначимыми."
    negative = main[main.difference.lt(0) & main.p_holm.lt(.05)]
    positive = main[main.difference.gt(0) & main.p_holm.lt(.05)]
    if len(negative)==len(main):
        return "Основные сравнения 2026 года поддерживают гипотезу о меньшем числе переключений при высоком объёме. Устойчивость на других типах баров оценивается ниже."
    if len(negative)==0:
        return f"Убедительного подтверждения гипотезы о меньшем числе переключений при высоком объёме на дельта-барах не получено. В {len(positive)} из {len(main)} основных сравнений 2026 года обнаружено статистически значимое увеличение числа событий после поправки Holm."
    return f"Гипотеза поддерживается частично: снижение числа переключений значимо в {len(negative)} из {len(main)} основных сравнений 2026 года. Общий вывод требует учитывать расхождения индикаторов и инструментов."


def create_charts(output, comparisons, bins, daily):
    """Сохраняет HTML сравнения, зависимость от R и распределение длительности окна."""
    panels = [("delta_st", "SuperTrend: переключений за окно"), ("delta_sma", "SMA 3/34: пересечений за окно"),
              ("delta_st_100", "SuperTrend: на 100 переходов"), ("delta_sma_100", "SMA: на 100 переходов"),
              ("time5_st", "SuperTrend на 5 мин: за окно"), ("time5_sma", "SMA на 5 мин: за окно")]
    figure = make_subplots(rows=3, cols=2, subplot_titles=[label for _, label in panels])
    for index, (metric, _) in enumerate(panels):
        sample = comparisons[comparisons.period.eq("2026") & comparisons.basis.eq("median") & comparisons.metric.eq(metric) & comparisons.block.eq(20)]
        for label, column, color in (("Объём ≤ медианы", "low_mean", "#718096"), ("Объём > медианы", "high_mean", "#1677b8")):
            figure.add_trace(go.Bar(x=sample.symbol, y=sample[column], name=label, marker_color=color, legendgroup=label, showlegend=index==0), row=index//2+1, col=index%2+1)
    figure.update_layout(title="Объём и переключения индикаторов — проверочный период 2026", height=1050, width=1100, template="plotly_white", barmode="group")
    figure.write_html(output / "comparison_2026.html", include_plotlyjs=True)
    graph = make_subplots(rows=2, cols=2, subplot_titles=["Число переключений SuperTrend", "Число пересечений SMA", "Число дельта-баров", "Эффективность пути на 5 мин"])
    for symbol, group in bins[bins.basis.eq("median")].groupby("symbol"):
        for index, column in enumerate(("delta_st", "delta_sma", "delta_bars", "time5_er")):
            graph.add_trace(go.Scatter(x=group.bin, y=group[column], mode="lines+markers", name=symbol, legendgroup=symbol, showlegend=index==0), row=index//2+1, col=index%2+1)
    graph.update_layout(title="Зависимость от объёма относительно прошлой медианы — вся история", height=760, width=1100, template="plotly_white")
    graph.write_html(output / "relative_volume.html", include_plotlyjs=True)
    distribution = go.Figure()
    for symbol, group in daily.groupby("symbol"):
        distribution.add_trace(go.Histogram(x=group.hours, name=symbol, opacity=.65, nbinsx=40))
    distribution.update_layout(title="Длительность общего окна 21 дней, клиринги включены", xaxis_title="Часы", yaxis_title="Число дней", template="plotly_white", barmode="overlay")
    distribution.write_html(output / "window_hours.html", include_plotlyjs=True)


def make_report(output):
    """Создаёт report.md и графики в output; выводы основывает на готовых сравнениях."""
    output = Path(output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    run = json.loads((output / "analysis_run.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(output / "audit.csv")
    daily = pd.read_csv(output / "daily.csv")
    comparisons = pd.read_csv(output / "comparisons.csv")
    bins = pd.read_csv(output / "ratio_bins.csv")
    good = daily[daily.valid_observation]
    main = comparisons[comparisons.period.eq("2026") & comparisons.metric.isin(["delta_st", "delta_sma"]) & comparisons.block.eq(20)]
    conclusion = primary_conclusion(main)
    explanation = []
    for symbol in config["symbols"]:
        selected = comparisons[comparisons.symbol.eq(symbol) & comparisons.period.eq("2026") & comparisons.basis.eq("median") & comparisons.block.eq(20)].set_index("metric")
        explanation.append(f"{symbol}: число дельта-баров при высоком объёме изменилось на {number(selected.loc['delta_bars', 'change_percent'], 1)}%, число переключений SuperTrend — на {number(selected.loc['delta_st', 'change_percent'], 1)}%, пересечений SMA — на {number(selected.loc['delta_sma', 'change_percent'], 1)}%. На одинаковой пятиминутной сетке число переключений SuperTrend изменилось на {number(selected.loc['time5_st', 'change_percent'], 1)}%, пересечений SMA — на {number(selected.loc['time5_sma', 'change_percent'], 1)}%.")
    lines = ["# Объём и дневная трендовость RTS/MIX", "", conclusion, "",
             "Рост числа событий на дельта-барах нельзя трактовать как самостоятельное доказательство меньшей трендовости: при высоком объёме формируется больше баров. Контроль на одном временном масштабе показывает частичную поддержку идеи более направленного движения, особенно по SMA; сила связи зависит от инструмента, индикатора и торгового окна.", "",
             *explanation, "",
             "Контрольные сравнения являются исследовательскими: для их p-значений не проведена поправка на все просмотренные варианты. Отдельный удачный контроль не заменяет заранее выбранные основные проверки.", "",
             "## Выборка и аудит", "",
             f"Проверено {len(audit)} архивов, включая предысторию и будние праздники. Ошибок сверки: {int((~audit.valid).sum())}. Исходники не изменялись.", "",
             "| Инструмент | Архивы аудита | Будних наблюдений с базой | Даты наблюдений | Дней 2026 |", "| --- | ---: | ---: | --- | ---: |"]
    for symbol, group in good.groupby("symbol"):
        lines.append(f"| {symbol} | {int(audit.symbol.eq(symbol).sum())} | {len(group)} | {group.day.min()}–{group.day.max()} | {int(group.period.eq('2026').sum())} |")
    lines += ["", "Число пригодных исходов после прогрева и удаления граничных баров:", "",
              "| Инструмент | Дельта SuperTrend | Дельта SMA | 5-мин SuperTrend | 5-мин SMA |", "| --- | ---: | ---: | ---: | ---: |"]
    for symbol, group in good.groupby("symbol"):
        lines.append(f"| {symbol} | {int(group.delta_st.notna().sum())} | {int(group.delta_sma.notna().sum())} | {int(group.time5_st.notna().sum())} | {int(group.time5_sma.notna().sum())} |")
    excluded = daily[~daily.valid_observation]
    lines += ["", "Причины пропусков дневной оценки:", ""]
    for (symbol, reason), group in excluded.groupby(["symbol", "issue"]):
        lines.append(f"- {symbol}: {reason} — {len(group)} дат.")
    missing = pd.read_csv(output / "missing_weekdays.csv")
    lines += ["", f"Будних календарных дат без архива: {len(missing)} записей по инструментам. Они перечислены в `missing_weekdays.csv`: биржевые выходные и неизвестные пропуски без внешнего календаря не различены.", "",
              "Суммы поля `volume`, число тиков, SHA256 ZIP и последовательное покрытие исходных строк сверены с БД. Это проверяет обработку, но не независимую полноту или единицы объёма поставщика. Объём измеряется суммой исходного поля `volume`; денежный оборот не используется.", "",
              "## Методика", "",
              "Для каждой даты взяты 20 предыдущих доступных будних дат с тиками. Выходные исключены из базы и основного прогрева. Общее начало — самая поздняя первая сделка, общее окончание — самая ранняя последняя сделка среди всех 21 дней; обе границы включены. Клиринги не вычитаются. Исторические объёмы пересчитываются в том же окне, текущий день в среднее/медиану не входит.", "",
              "Высокий объём: отношение к предыдущему среднему или медиане строго больше 1. SuperTrend: ATR Уайлдера 10, множитель 3; SMA: close, периоды 3/34. Прогрев — минимум 100 предыдущих баров. Состояния не сбрасываются ежедневно; ночной переход не считается внутридневным событием. После подтверждённых ошибок и календарного разрыва более 10 дней прогрев возобновляется заново.", "",
              "Основная оценка использует дельта-бары, целиком лежащие внутри окна. События не соединяются через исключённые бары. Дневной остаток включён, отдельный вариант его исключает. Полные временные свечи строятся по московской сетке; паузы получают предыдущую цену и нулевой объём.", "",
              f"Фиксированный контроль: {config['fixed_start']}–{config['fixed_end']}. Часы выбраны до расчёта индикаторов по 95%-квантилю первых и 5%-квантилю последних сделок с округлением внутрь до пяти минут. Для включения все 21 дня должны охватывать это окно.", "",
              "## Основные сравнения: 2026 год", "", effect_table(main), "",
              "`delta_st` — переключения SuperTrend; `delta_sma` — пересечения SMA. Отрицательное изменение поддерживает исходную гипотезу. ДИ относятся к разнице средних в числе событий и являются индивидуальными; поправка Holm применяется к восьми p-значениям.", "",
              f"Блочный bootstrap: {run['repetitions']} повторов, последовательные блоки по 20 торговых наблюдений, seed {config['seed']}. Блоки не пересекают выделенные разрывы. Проверки с блоками 40 и 60 сохранены в `comparisons.csv`. Малое количество независимых блоков в 2026 году ограничивает точность оценки. Если все сегменты не длиннее блока, ДИ/p не вычисляются; короткие сегменты при наличии длинного остаются фиксированными частями, их неопределённость не учтена.", "",
              "## Проверки устойчивости", "",
              "Ниже использована медианная база и блок 20. Для контрольных показателей p Holm не вычисляется: они не входят в восемь основных проверок.", ""]
    control_names = ["delta_st_100", "delta_sma_100", "time5_st", "time5_sma", "time5_er", "fixed_delta_st", "fixed_delta_sma", "fixed_time5_st", "fixed_time5_sma"]
    controls = comparisons[comparisons.period.eq("2026") & comparisons.basis.eq("median") & comparisons.metric.isin(control_names) & comparisons.block.eq(20)]
    lines += [effect_table(controls), "", "`*_100` — события на 100 пригодных переходов; `time5_*` — пятиминутные свечи; `*_er` — эффективность ценового пути от 0 до 1; `fixed_*` — фиксированное окно и объём, пересчитанный в нём.", "",
              "Частота на 100 баров и абсолютное число событий отвечают на разные вопросы. Снижение частоты при увеличении числа баров не означает уменьшения числа переключений за день.", "",
              "## Исторический период 2022–2025", "",
              effect_table(comparisons[comparisons.period.eq("2022–2025") & comparisons.metric.isin(["delta_st", "delta_sma"]) & comparisons.block.eq(20)]), "",
              "Годовые сводки и группы по границам дня: `year_and_bounds.csv`. Диапазоны относительного объёма: `ratio_bins.csv`. Контроли на 1/15 мин, без остатков и с выходными в прогреве: `comparisons.csv`. Исключение верхнего 1% объёмов, резких изменений границ, календарных праздников и предполагаемого квартального ролловера: `sensitivities.csv`. Последние два фильтра — календарные приближения, а не подтверждённая информация о режиме торгов и смене контракта.", "",
              "## Покрытие общего окна", "", "| Инструмент | Медиана часов | Минимум часов | Медиана доли объёма в пригодных дельта-барах | Медиана пустых 5-мин свечей | Фиксированное окно: дней |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for symbol, group in good.groupby("symbol"):
        lines.append(f"| {symbol} | {number(group.hours.median())} | {number(group.hours.min())} | {number(100*group.delta_volume_coverage.median(), 1)}% | {number(100*group.time5_empty_fraction.median(), 1)}% | {int(group.fixed_valid.eq(True).sum())} |")
    lines += ["", "## Ограничения вывода", "",
              "Первая/последняя сделки зависят от активности и полноты архива. Точность фактического биржевого расписания не предполагается. Низкая активность или обрезанный архив способны сузить окно. Подтверждённых ошибок не следует путать с невыясненной полнотой.", "",
              "Данные — непрерывная склейка без кода контракта. Ролловер, изменение спецификации и денежного масштаба объёма невозможно полностью проверить по трём столбцам CSV. Исследование показывает связь, но не причинность.", "",
              "Полный дневной объём известен только к концу дня. Результат не доказывает, что объём, известный утром, позволяет предсказать оставшееся движение.", "",
              "## Файлы и воспроизведение", "",
              "- `daily.csv` — все будние даты, базы из 20 дней, точные объёмы и показатели.",
              "- `audit.csv`, `missing_weekdays.csv` — сверка архивов и неизвестные отсутствующие даты.",
              "- `comparisons.csv` — размеры эффектов, индивидуальные ДИ и p-значения.",
              "- `config.json`, `analysis_run.json` — зафиксированные правила и команда запуска.",
              "- `comparison_2026.html`, `relative_volume.html`, `window_hours.html` — автономные графики.", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe -m research.volume_trend --stage audit", ".\\.venv\\Scripts\\python.exe -m research.volume_trend --stage analyze --repetitions 5000", ".\\.venv\\Scripts\\python.exe -m research.volume_trend --stage report", ".\\.venv\\Scripts\\python.exe -m unittest -v tests.test_volume_trend", "```", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    create_charts(output, comparisons, bins, good)
    print(f"Отчёт сохранён: {output / 'report.md'}", flush=True)
