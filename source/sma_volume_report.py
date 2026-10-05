r"""Создаёт автономный интерактивный отчёт бэктеста SMA 3/34 и объёма.

Запуск из корня:
    .\.venv\Scripts\python.exe -m backtest.backtest_sma_volume
    .\.venv\Scripts\python.exe -m backtest.backtest_sma_volume --costs 0,2,4,8 --base-cost 4

Получает готовые таблицы сделок, дневных исходов, парных тестов и покрытия.
Выводы вычисляются из результатов; сравниваются прибыль, тиковая просадка
и влияние затрат. Изменение PnL разлагается на результат до затрат и
экономию на меньшем числе сделок. Plotly, русская локаль, данные, графики и оформление
встроены в report.html, интернет и дополнительные файлы не требуются.
"""

from html import escape
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

MODES={"baseline":"Без фильтра объёма","volume":"С фильтром объёма"}
COLORS={"baseline":"#8896aa","volume":"#168a79"}
RU_LOCALE={"moduleType":"locale","name":"ru","dictionary":{
    "Download plot as a png":"Сохранить график в PNG","Zoom":"Масштабирование","Pan":"Перемещение",
    "Zoom in":"Приблизить","Zoom out":"Отдалить","Autoscale":"Автомасштаб","Reset axes":"Сбросить оси",
    "Box Select":"Выделение прямоугольником","Lasso Select":"Выделение лассо",
    "Toggle Spike Lines":"Линии к осям","Show closest data on hover":"Подсказка ближайшей точки",
    "Compare data on hover":"Сравнить значения"},"format":{"decimal":",","thousands":" "}}


def number(value, digits=0):
    """Возвращает русское форматирование value с digits знаками; отсутствующее — прочерк."""
    return f"{value:,.{digits}f}".replace(","," ").replace(".",",") if pd.notna(value) else "—"


def table(frame, columns):
    """Возвращает HTML таблицу frame; columns задаёт пары поле/русский заголовок."""
    if frame.empty:
        return "<p>Нет данных для этого периода.</p>"
    rows=['<div class="scroll"><table><thead><tr>'+''.join(f'<th>{escape(title)}</th>' for _,title in columns)+'</tr></thead><tbody>']
    for record in frame.to_dict("records"):
        cells=[]
        for key,_ in columns:
            value=record[key]
            if key=="mode":
                value=MODES[value]
            elif key in ("p","p_holm"):
                value=number(value,4)
            elif key=="win_rate":
                value=number(value*100,1)+"%"
            elif isinstance(value,(float,np.floating)):
                value=number(value,2 if key in ("profit_factor","mean_trade","mean_day","difference","ci_low","ci_high") else 0)
            cells.append(f"<td>{escape(str(value))}</td>")
        rows.append("<tr>"+''.join(cells)+"</tr>")
    return '\n'.join(rows)+"</tbody></table></div>"


def chart(figure, identifier):
    """Встраивает интерактивную figure в identifier с русской локалью и экспортом PNG."""
    figure.update_layout(template="plotly_white",font=dict(family="Arial, sans-serif",size=13,color="#26354c"),
                         margin=dict(l=70,r=35,t=85,b=85),paper_bgcolor="white")
    return figure.to_html(full_html=False,include_plotlyjs=False,include_mathjax=False,div_id=identifier,
        config={"responsive":True,"displaylogo":False,"locale":"ru","toImageButtonOptions":{"format":"png","scale":2}})


def daily_drawdown(frame):
    """Возвращает максимальную сквозную тиковую просадку каждого дня frame, в пунктах."""
    cumulative=peak=0.
    values=[]
    for row in frame.sort_values("day").itertuples():
        values.append(-max(row.max_drawdown,peak-cumulative-row.equity_min))
        peak=max(peak,cumulative+row.equity_peak)
        cumulative+=row.net_pnl
    return values


def equity_chart(daily, symbol, period, costs, base_cost, drawdown=False):
    """Строит equity или дневную тиковую просадку daily для symbol/period с выбором costs."""
    sample=daily[daily.symbol.eq(symbol)]
    if period=="2026":
        sample=sample[sample.day.str.startswith("2026")]
    if sample.empty:
        return ""
    figure=go.Figure()
    for cost in costs:
        for mode in MODES:
            group=sample[sample.cost_ticks.eq(cost)&sample["mode"].eq(mode)].sort_values("day")
            y=daily_drawdown(group) if drawdown else group.net_pnl.cumsum()
            figure.add_trace(go.Scatter(x=group.day,y=y,name=MODES[mode],line=dict(color=COLORS[mode]),
                visible=cost==base_cost,mode="lines",hovertemplate="%{x}<br>%{y:,.0f} пунктов<extra>%{fullData.name}</extra>"))
    buttons=[dict(label=f"{cost:g} шагов за круг",method="update",args=[{"visible":[c==cost for c in costs for _ in MODES]}]) for cost in costs]
    title="Максимальная просадка в течение дня" if drawdown else "Накопленный PnL"
    figure.update_layout(title=f"{symbol} · {period} · {title}",height=420,hovermode="x unified",
        legend=dict(orientation="h",y=-.2),updatemenus=[dict(buttons=buttons,active=costs.index(base_cost),x=1,y=1.18,xanchor="right")])
    figure.update_yaxes(title_text="Пункты одного контракта")
    return chart(figure,f"{'drawdown' if drawdown else 'equity'}-{symbol}-{period}")


def cost_chart(summary):
    """Показывает PnL за 2026 в summary при всех сценариях затрат, отдельно по инструментам."""
    current=summary[summary.period.eq("2026")]
    symbols=list(current.symbol.unique())
    if not symbols:
        return ""
    figure=make_subplots(rows=1,cols=len(symbols),subplot_titles=symbols)
    for col,symbol in enumerate(symbols,1):
        for mode in MODES:
            group=current[current.symbol.eq(symbol)&current["mode"].eq(mode)].sort_values("cost_ticks")
            figure.add_trace(go.Scatter(x=group.cost_ticks,y=group.net_pnl,mode="lines+markers",name=MODES[mode],
                legendgroup=mode,showlegend=col==1,line=dict(color=COLORS[mode]),
                hovertemplate="Затраты: %{x:g} шагов<br>PnL: %{y:,.0f} пунктов<extra>%{fullData.name}</extra>"),row=1,col=col)
        figure.add_hline(y=0,line_dash="dash",line_color="#6f7d90",row=1,col=col)
        figure.update_xaxes(title_text="Шагов цены за круг",row=1,col=col)
        figure.update_yaxes(title_text="Пункты одного контракта",row=1,col=col)
    figure.update_layout(height=430,legend=dict(orientation="h",y=-.2))
    return chart(figure,"cost-sensitivity")


def comparison_chart(comparisons):
    """Показывает среднюю парную разность PnL comparisons с индивидуальным 95%-ным ДИ."""
    sample=comparisons[comparisons.period.eq("2026")&comparisons.block.eq(20)]
    symbols=list(sample.symbol.unique())
    if not symbols:
        return ""
    figure=make_subplots(rows=1,cols=len(symbols),subplot_titles=symbols)
    for col,symbol in enumerate(symbols,1):
        group=sample[sample.symbol.eq(symbol)].sort_values("cost_ticks")
        figure.add_trace(go.Scatter(x=group.difference,y=[f"{v:g} шагов" for v in group.cost_ticks],mode="markers",
            marker=dict(color="#168a79",size=10),error_x=dict(array=group.ci_high-group.difference,arrayminus=group.difference-group.ci_low),
            customdata=group[["p_holm","total_difference"]],showlegend=False,
            hovertemplate="%{y}<br>Разность: %{x:.2f} пунктов/день<br>p Holm: %{customdata[0]:.4f}<br>Суммарно: %{customdata[1]:,.0f} пунктов<extra></extra>"),row=1,col=col)
        figure.add_vline(x=0,line_dash="dash",line_color="#6f7d90",row=1,col=col)
        figure.update_xaxes(title_text="С фильтром − без фильтра, пунктов/день",row=1,col=col)
    figure.update_layout(height=400)
    return chart(figure,"paired-effects")


def annual_chart(summary, base_cost):
    """Показывает годовые PnL summary при base_cost без новых статистических проверок."""
    sample=summary[summary.period.isin(["2022","2023","2024","2025","2026"])&summary.cost_ticks.eq(base_cost)]
    symbols=list(sample.symbol.unique())
    if not symbols:
        return ""
    figure=make_subplots(rows=1,cols=len(symbols),subplot_titles=symbols)
    for col,symbol in enumerate(symbols,1):
        for mode in MODES:
            group=sample[sample.symbol.eq(symbol)&sample["mode"].eq(mode)].sort_values("period")
            figure.add_trace(go.Bar(x=group.period,y=group.net_pnl,name=MODES[mode],legendgroup=mode,
                showlegend=col==1,marker_color=COLORS[mode],hovertemplate="%{x}<br>%{y:,.0f} пунктов<extra>%{fullData.name}</extra>"),row=1,col=col)
        figure.update_yaxes(title_text="Пункты одного контракта",row=1,col=col)
    figure.update_layout(height=450,barmode="group",legend=dict(orientation="h",y=-.2))
    return chart(figure,"annual-profit")


def conclusions(summary, comparisons, settings):
    """Возвращает выводы summary/comparisons при затратах settings и без затрат.

    Отделяет изменение PnL до затрат от экономии на числе сделок; отсутствие
    доступного p указывает как недостаточность данных, а не отрицательный тест.
    """
    sample=summary[summary.period.eq("2026")&summary.cost_ticks.eq(settings["base_cost"])]
    lines=[]
    for symbol in settings["symbols"]:
        group=sample[sample.symbol.eq(symbol)].set_index("mode")
        if not {"baseline","volume"}.issubset(group.index):
            lines.append(f"{symbol}: нет результатов 2026 года в выбранном диапазоне.")
            continue
        filtered,base=group.loc["volume"],group.loc["baseline"]
        gain=filtered.net_pnl-base.net_pnl
        test=comparisons[comparisons.period.eq("2026")&comparisons.block.eq(20)&comparisons.symbol.eq(symbol)&comparisons.cost_ticks.eq(settings["base_cost"])]
        p=test.p_holm.iloc[0] if len(test) else np.nan
        difference_text=f"результат с фильтром {'выше' if gain>0 else 'ниже'} на {number(abs(gain))} пунктов" if gain else "результаты равны"
        significance="Недостаточно данных для проверки значимости" if pd.isna(p) else "Различие статистически значимо" if p<.05 else "Значимое различие не установлено"
        profitability="Стратегия с фильтром прибыльна в этом сценарии затрат." if filtered.net_pnl>0 else "Стратегия с фильтром убыточна в этом сценарии затрат." if filtered.net_pnl<0 else "PnL стратегии с фильтром нулевой."
        gross_gain=filtered.gross_pnl-base.gross_pnl
        zero_test=comparisons[comparisons.period.eq("2026")&comparisons.block.eq(20)&comparisons.symbol.eq(symbol)&comparisons.cost_ticks.eq(0)]
        zero_p=zero_test.p_holm.iloc[0] if len(zero_test) else np.nan
        zero_text="Без затрат значимое различие не установлено" if pd.notna(zero_p) and zero_p>=.05 else "Без затрат различие статистически значимо" if pd.notna(zero_p) else "Проверка значимости без затрат отсутствует"
        lines.append(f"{symbol}: с фильтром {number(filtered.net_pnl)} пунктов, без фильтра {number(base.net_pnl)}; "
            f"{difference_text}. {significance} (p Holm = {number(p,4)}). {profitability} "
            f"Изменение PnL до затрат: {number(gross_gain)} пунктов; экономия затрат: {number(gain-gross_gain)} пунктов "
            f"за счёт {number(base.trades-filtered.trades)} исключённых сделок. {zero_text} (p Holm = {number(zero_p,4)}).")
    return lines


def write_report(folder, summary, daily, trades, comparisons, coverage, metadata):
    """Сохраняет автономный report.html в folder из таблиц и metadata; возвращает путь."""
    settings=metadata["settings"]
    cost=settings["base_cost"]
    main=summary[summary.period.eq("2026")&summary.cost_ticks.eq(cost)]
    comparison=comparisons[comparisons.period.eq("2026")&comparisons.block.eq(20)]
    base=summary[summary.cost_ticks.eq(cost)]
    columns=[("symbol","Инструмент"),("mode","Стратегия"),("days","Дней"),("active_days","Дней со сделками"),
        ("trades","Сделок"),("gross_pnl","PnL до затрат"),("net_pnl","PnL после затрат"),("mean_trade","Средняя сделка"),
        ("profit_factor","Profit factor"),("win_rate","Прибыльных сделок"),("max_drawdown","Тиковая просадка")]
    test_columns=[("symbol","Инструмент"),("cost_ticks","Затраты, шагов"),("days","Дней"),("total_difference","Δ суммарного PnL"),
        ("difference","Δ PnL/день"),("ci_low","95% ДИ: низ"),("ci_high","95% ДИ: верх"),("p","p"),("p_holm","p Holm"),("verdict","Вывод")]
    cover=coverage.groupby(["symbol","status","reason"],dropna=False).size().reset_index(name="days")
    figures=[]
    for symbol in settings["symbols"]:
        figures.append(equity_chart(daily,symbol,"2026",settings["costs"],cost))
        figures.append(equity_chart(daily,symbol,"все",settings["costs"],cost))
    drawdowns=''.join(equity_chart(daily,s,"2026",settings["costs"],cost,True) for s in settings["symbols"])
    text=''.join(f"<li>{escape(t)}</li>" for t in conclusions(summary,comparisons,settings))
    delays=coverage.loc[coverage.status.eq("included"),"close_delay_seconds"]
    actual_start,actual_end=daily.day.min(),daily.day.max()
    html=f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Бэктест SMA 3/34 и накопленного объёма — RTS/MIX</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#edf2f7;color:#26354c;font:16px/1.65 Arial,sans-serif}}
    main{{max-width:1340px;margin:auto;padding:28px}}header{{background:#183a55;color:white;border-radius:15px;padding:30px}}
    h1{{font-size:32px;line-height:1.25}}h2{{font-size:24px;margin-top:0}}h3{{font-size:18px}}section{{background:white;padding:26px;border-radius:12px;margin:20px 0;border:1px solid #dde5ef;overflow:hidden}}
    nav{{display:flex;gap:12px;flex-wrap:wrap;padding:18px 0}}a{{color:#245eb5}}nav a{{background:white;border-radius:18px;padding:5px 15px;text-decoration:none}}
    .scroll{{overflow:auto;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #dee5ee;white-space:nowrap;text-align:left}}th{{background:#eef3f9}}
    .note{{padding:16px 20px;background:#fff8e6;border-left:4px solid #bd913a;border-radius:5px}}.callout{{padding:16px 20px;background:#edf8f4;border-left:4px solid #168a79;border-radius:5px}}
    .small{{font-size:13px;color:#6a788d}}details{{margin:18px 0}}summary{{cursor:pointer;color:#245eb5;font-weight:bold}}pre{{white-space:pre-wrap;padding:18px;background:#f1f5fa;border-radius:8px}}li{{margin:9px 0}}
    @media(max-width:760px){{main{{padding:10px}}header,section{{padding:18px}}h1{{font-size:26px}}}}
    </style><script>{get_plotlyjs()}</script><script>Plotly.register({json.dumps(RU_LOCALE,ensure_ascii=False)});</script></head><body><main>
    <header><p>Тиковое исполнение · пятиминутные свечи · один контракт</p><h1>SMA 3/34: помогает ли фильтр объёма?</h1>
    <p>Объём текущего дня до сигнала сравнивается с медианой объёма до того же времени предыдущих 20 будних торговых дней.</p>
    <p>{actual_start} — {actual_end} · основной период: 2026 · 10:00–18:45 МСК</p></header>
    <nav><a href="#answer">Вывод</a><a href="#method">Правила</a><a href="#equity">PnL</a><a href="#costs">Затраты</a><a href="#tests">Значимость</a><a href="#history">История</a><a href="#coverage">Данные</a></nav>
    <section id="answer"><h2>Результат бэктеста</h2><p>Ниже сценарий <strong>{cost:g} шагов цены за круг</strong>, включая обе стороны сделки. Это иллюстрация чувствительности, не фактический тариф брокера. RTS: {cost*metadata['tick_sizes']['RTS']:g} пунктов за круг; MIX: {cost*metadata['tick_sizes']['MIX']:g}.</p>
    <div class="callout"><ul>{text}</ul></div>{table(main,columns)}
    <p>Все показатели в пунктах одного контракта, не в рублях и не в процентах капитала. Суммы двух инструментов не складываются. Просадка учитывает все тики открытой позиции. Profit factor — сумма прибылей / абсолютная сумма убытков после затрат.</p>
    <div class="note">Выбранный фильтр отличается от первоначальной группировки по итоговому дневному объёму: здесь он доступен в момент сигнала. Меньшее число пересечений само по себе не является проверкой доходности; этот бэктест проверяет конкретные сделки.</div></section>
    <section id="method"><h2>Точные правила</h2><ol>
    <li>Обычные OHLC-свечи по фиксированной пятиминутной сетке. SMA 3 и SMA 34 по close, без ежедневного сброса; прогрев 100 баров, разрыв более 10 календарных дней сбрасывает историю. Выходные не участвуют в индикаторах и объёмах.</li>
    <li>После 10:00 пересечение SMA 3 вверх через SMA 34 даёт Long, вниз — Short. Равенства не добавляют пересечений; переход между последней свечой прошлого дня и первой текущего не считается сигналом.</li>
    <li>На закрытии свечи t сравниваются объём [10:00,t) текущего дня и медиана [10:00,t) ровно 20 предыдущих доступных будних дат. Текущий день в медиану не входит, объём нового бара не участвует. Строгое превышение нужно только для первого входа дня.</li>
    <li>Сделка исполняется первым реальным тиком непосредственно следующей пятиминутной свечи. Если в ней нет сделки, старый сигнал пропускается. Фильтр, выполненный позднее без нового пересечения, сам по себе не открывает позицию.</li>
    <li>После первого входа обратные пересечения переворачивают позицию без повторной проверки объёма. Переворот оплачивает выход из старой и вход в новую позицию. Стопов, целей и увеличения позиции нет.</li>
    <li>В 18:45 закрытие первым доступным тиком имеет приоритет; сигнал, закрывшийся в 18:45, новую позицию не открывает. Переноса ночью нет. Параллельно исполняется та же стратегия без фильтра объёма.</li>
    <li>Для сопоставимости с фиксированным контролем гипотезы включены только даты, когда все 21 сессии охватывают 10:00–18:45 по первой/последней сделкам. Это ретроспективная проверка покрытия, а не условие входа по будущему объёму. Ограничение состава выборки нужно учитывать при переносе в реальную торговлю.</li>
    </ol><p>Затраты моделируются фиксированной суммой пунктов за каждую сторону: половина сценария на вход и половина на выход. Они объединяют комиссию, спред и проскальзывание. Рыночные ордера моделируются по ценам сделок, стакан и задержка системы неизвестны.</p></section>
    <section id="equity"><h2>Кривые результата и просадки</h2><p class="small">Графики интерактивные: подсказки, масштабирование, переключение легенды, сохранение PNG. В выпадающем списке меняется сценарий затрат. Для каждого периода накопление начинается с нуля.</p>
    {''.join(figures)}<h3>Тиковая просадка по дням 2026</h3><p class="small">Показана максимальная просадка внутри каждого дня относительно предшествовавшего пика капитала; это не только просадка по закрытиям дней.</p>{drawdowns}</section>
    <section id="costs"><h2>Чувствительность к затратам</h2>{cost_chart(summary)}
    {table(summary[summary.period.eq('2026')],[('cost_ticks','Шагов за круг')]+columns)}
    <p>Сценарии заданы заранее. Наиболее выгодный уровень затрат не выбирается. Сравнение учитывает и дни без входов: их PnL равен нулю.</p></section>
    <section id="tests"><h2>Повысил ли фильтр средний дневной PnL?</h2><p>Для каждой даты Δ = PnL с фильтром − PnL без фильтра. Положительная разность означает преимущество фильтра по PnL. Сравнение парное, включая нулевые дни: различия количества сделок и времени входа входят в эффект.</p>
    <p>Центрированный двусторонний блочный bootstrap: {settings['repetitions']} повторов, основной блок 20 последовательных пригодных дат, seed {settings['seed']}. В 2026 Holm охватывает {metadata['family_size']} проверок — все инструменты и сценарии затрат. Значимость: p Holm &lt;0,05.</p>
    <p>Индивидуальный 95%-ный ДИ симметричен по 95%-квантилю абсолютной центрированной bootstrap-ошибки. Он не скорректирован на множественные сравнения. Периоды/блоки имеют отдельные семьи; исторические оценки не добавляют независимых подтверждений к 2026.</p>
    {table(comparison,test_columns)}{comparison_chart(comparisons)}
    <details><summary>Проверка устойчивости: блоки 20/40/60</summary>{table(comparisons[comparisons.period.eq('2026')],[('block','Блок, дней')]+test_columns)}</details>
    <p>Неопределённость просадки и profit factor отдельно не оценивалась. Когда сегмент не длиннее блока, он фиксируется; без длинных сегментов ДИ/p отсутствуют. Это ограничение блочной оценки для коротких исторических частей.</p></section>
    <section id="history"><h2>Историческая устойчивость</h2><p>2026 уже рассматривался в предыдущем исследовании: это основной отдельный период, но не новая, ранее не изученная выборка. Все параметры зафиксированы до бэктеста, оптимизации не было.</p>
    {annual_chart(summary,cost)}
    <details><summary>Сводки всех периодов при {cost:g} шагах за круг</summary>{table(base,[('period','Период')]+columns)}</details>
    <details><summary>Парные сравнения 2022–2025 и всей выборки</summary>{table(comparisons[~comparisons.period.eq('2026')],[('period','Период')]+test_columns)}</details></section>
    <section id="coverage"><h2>Покрытие и воспроизводимость</h2>{table(cover,[('symbol','Инструмент'),('status','Статус'),('reason','Причина'),('days','Дней')])}
    <p>Заново прочитано и сверено {metadata['verification_days']} будних ZIP, включая прогрев. SHA256 исходных архивов совпал с аудитом. Время, число тиков, объём и все OHLCV пятиминутных свечей совпали.</p>
    <p>Фактическая задержка закрытия относительно 18:45: медиана {number(delays.median(),3)} с; 95%-квантиль {number(delays.quantile(.95),3)} с; максимум {number(delays.max(),3)} с.</p>
    <ul><li>Отсутствующие будние архивы не всегда отличимы от биржевых выходных. Первая/последняя сделки могут зависеть от активности и полноты источника. Историческое расписание и клиринги не восстанавливались.</li>
    <li>Источники — непрерывные склейки RTS/MIX без кода отдельного контракта. Ролловеры и изменение денежной стоимости пункта полностью не проверены. Доли секунды источника искусственные, порядок строк сохранён.</li>
    <li>Условие покрытия исключает короткие сессии и дни без тика закрытия. Риск таких дней не смоделирован. Сопоставление стратегий проводится на одной и той же включённой выборке.</li>
    <li>Для выбора стратегии нужно учитывать фактические затраты и риск. Данный результат описывает исторический тест выбранных правил, не проверку будущего исполнения.</li></ul>
    <details><summary>Файлы, параметры и команды запуска</summary><p>summary.csv — сводки; daily.csv — дневные PnL и тиковые экстремумы; trades.csv — сделки всех сценариев; signals.csv — объем на момент каждого сигнала; comparisons.csv — парные тесты; coverage.csv — все причины исключения; verification.csv — проверенные источники; metadata.json — параметры и хеши.</p>
    <pre>.\\.venv\\Scripts\\python.exe -m backtest.backtest_sma_volume --source "{escape(metadata['source'])}" --output "{escape(str(folder.parent))}" --symbols {' '.join(settings['symbols'])} --start {settings['start']} --end {settings['end']} --costs {','.join(f'{x:g}' for x in settings['costs'])} --base-cost {cost:g} --repetitions {settings['repetitions']}
.\\.venv\\Scripts\\python.exe -m unittest -v tests.test_backtest_sma_volume</pre>
    <pre>{escape(json.dumps(metadata,ensure_ascii=False,indent=2))}</pre></details></section>
    <footer class="small">Автономный HTML: графики, библиотека, данные и оформление встроены. Можно открывать без интернета и сервера.</footer></main></body></html>'''
    output=folder/"report.html"
    output.write_text(html,encoding="utf-8")
    return output
