r"""Создаёт автономный интерактивный HTML-отчёт по пятиминутным барам RTS/MIX.

Примеры запуска из корня проекта:
    .\.venv\Scripts\python.exe -m research.volume_trend_5m_report
    .\.venv\Scripts\python.exe -m research.volume_trend_5m_report --source results\volume_trend --repetitions 5000
    .\.venv\Scripts\python.exe -m research.volume_trend_5m_report --html results\volume_trend\five_minute_report.html
    .\.venv\Scripts\python.exe -m unittest -v tests.test_volume_trend_5m_report

Читает проверенную дневную таблицу, пересчитывает 5-минутные сравнения с общей
поправкой Holm на 16 тестов и проверками блоков 20/40/60. Симметричные ДИ
согласованы с центрированным двусторонним bootstrap-тестом. Plotly, данные,
таблицы и оформление встроены в единственный HTML; внешние ресурсы не нужны.
Все текстовые выводы формируются из пересчитанных результатов, включая
случаи отсутствия значимости и смешанных направлений связи.
Исходные тики, базы и предыдущий отчёт не изменяются.
"""

import argparse
from datetime import datetime
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from research.volume_trend_metrics import group_effect, holm

COLORS = {"RTS": "#245eb5", "MIX": "#ba6328", "low": "#7e8da1", "high": "#259c8c"}
WINDOWS = {"common": "Общее окно 21 дней", "fixed": "Фиксированное 10:00–18:45"}
BASIS = {"mean": "Среднее 20 дней", "median": "Медиана 20 дней"}
METRICS = {"st": "SuperTrend", "sma": "SMA 3/34"}


def number(value, digits=2):
    """Форматирует value с digits знаками для русской таблицы; NaN заменяет прочерком."""
    return f"{value:.{digits}f}".replace(".", ",") if pd.notna(value) else "—"


def p_text(value):
    """Форматирует p-значение value без округления малых значений до нуля."""
    return number(value, 4) if pd.notna(value) else "—"


def verdict(difference, p_holm):
    """Классифицирует знак difference при скорректированном p_holm и уровне 0,05."""
    if not np.isfinite(difference) or not np.isfinite(p_holm):
        return "Недостаточно данных"
    if p_holm >= .05:
        return "Не подтверждена"
    return "Поддерживается" if difference < 0 else "Обратная связь"


def overall_verdict(frame):
    """Возвращает общий вывод по frame, не смешивая частичную и универсальную поддержку."""
    if frame.empty or frame.p_holm.isna().any():
        return "Недостаточно данных для общего вывода по всем сравнениям."
    supported = int((frame.difference.lt(0) & frame.p_holm.lt(.05)).sum())
    opposite = int((frame.difference.gt(0) & frame.p_holm.lt(.05)).sum())
    if supported and opposite:
        return f"Смешанные результаты: значимое снижение в {supported}, рост в {opposite} из {len(frame)} сравнений; универсальная гипотеза не подтверждена."
    if supported==len(frame):
        return f"Гипотеза поддерживается во всех {len(frame)} сравнениях этого периода."
    if supported:
        return f"Частичная поддержка: статистически значимое снижение числа событий обнаружено в {supported} из {len(frame)} сравнений."
    if opposite:
        return f"Обнаружена обратная связь в {opposite} из {len(frame)} сравнений; универсальная гипотеза о снижении событий не поддерживается."
    return "Гипотеза не подтверждена: значимых различий не обнаружено. Это не доказывает отсутствие связи."


def indicator_conclusion(frame, indicator):
    """Возвращает вывод для indicator (st/sma) из frame по знаку эффекта и p Holm."""
    sample = frame[frame.indicator.eq(indicator)]
    name = METRICS[indicator]
    if sample.empty or sample.p_holm.isna().any():
        return f"По {name} недостаточно данных для вывода по всем сравнениям."
    supported = int((sample.difference.lt(0) & sample.p_holm.lt(.05)).sum())
    opposite = int((sample.difference.gt(0) & sample.p_holm.lt(.05)).sum())
    if supported or opposite:
        findings = []
        if supported:
            findings.append(f"статистически значимое снижение обнаружено в {supported} из {len(sample)} сравнений")
        if opposite:
            findings.append(f"статистически значимый рост обнаружен в {opposite} из {len(sample)} сравнений")
        return f"По {name} после общей поправки " + "; ".join(findings) + "."
    return f"По {name} после общей поправки значимых различий нет. Это означает отсутствие достаточного подтверждения, а не доказанное отсутствие эффекта и не опровержение гипотезы."


def bootstrap_summary(values, ratios, segments, repetitions=5000, block=20, seed=20261005):
    """Сравнивает values при ratios>1 и <=1 блочными повторами внутри segments.

    repetitions — число повторов; block — длина блока дат; seed — генератор.
    Возвращает средние, разницу, относительное изменение, симметричный 95%-ный
    ДИ и двустороннее p. ДИ строится из 95%-квантиля абсолютной центрированной
    bootstrap-ошибки; p использует ту же ошибку. Короткие сегменты фиксированы;
    если нет ни одного сегмента длиннее блока, ДИ/p не оцениваются.
    """
    values, ratios, segments = np.asarray(values, float), np.asarray(ratios, float), np.asarray(segments)
    result = group_effect(values, ratios)
    result.update(ci_low=np.nan, ci_high=np.nan, p=np.nan, bootstrap_valid=0)
    if min(result["n_high"], result["n_low"])<5:
        return result
    groups = [np.flatnonzero(segments==label) for label in np.unique(segments)]
    if not any(len(group)>block for group in groups):
        return result
    rng = np.random.default_rng(seed)
    valid = np.isfinite(values) & np.isfinite(ratios)
    high = valid & (ratios>1)
    low = valid & (ratios<=1)
    values_zero = np.where(np.isfinite(values), values, 0.)
    distribution = []
    # Векторизация не изменяет единицу статистики: индекс выбирает целые даты.
    for batch_start in range(0, repetitions, 250):
        count = min(250, repetitions-batch_start)
        pieces = []
        for group in groups:
            width = min(block, len(group))
            starts = rng.integers(0, len(group)-width+1, size=(count, int(np.ceil(len(group)/width))))
            positions = (starts[:, :, None]+np.arange(width)).reshape(count, -1)[:, :len(group)]
            pieces.append(group[positions])
        index = np.concatenate(pieces, axis=1)
        h, l = high[index], low[index]
        hn, ln = h.sum(axis=1), l.sum(axis=1)
        accepted = (hn>0) & (ln>0)
        with np.errstate(divide="ignore", invalid="ignore"):
            difference = (values_zero[index]*h).sum(axis=1)/hn - (values_zero[index]*l).sum(axis=1)/ln
        distribution.extend(difference[accepted].tolist())
    distribution = np.asarray(distribution)
    if len(distribution):
        errors = np.abs(distribution-result["difference"])
        radius = float(np.quantile(errors, .95, method="higher"))
        result["ci_low"], result["ci_high"] = result["difference"]-radius, result["difference"]+radius
        result["p"] = float((np.count_nonzero(errors>=abs(result["difference"]))+1)/(len(errors)+1))
        result["bootstrap_valid"] = len(distribution)
    return result


def adjust_families(data):
    """Добавляет p_holm к data отдельно для каждой пары период/блок, включая оба окна."""
    result = data.copy()
    result["p_holm"] = np.nan
    for _, group in result.groupby(["period", "block"], sort=False):
        result.loc[group.index, "p_holm"] = holm(group.p)
    result["verdict"] = [verdict(d, p) for d, p in zip(result.difference, result.p_holm)]
    return result


def statistics(daily, repetitions, seed):
    """Возвращает 5-минутные сравнения daily для трёх периодов и блоков 20/40/60."""
    rows = []
    for period in ("2026", "2022–2025", "все"):
        subset = daily if period=="все" else daily[daily.period.eq(period)]
        blocks = (20, 40, 60) if period=="2026" else (20,)
        for block in blocks:
            for symbol in ("RTS", "MIX"):
                sample = subset[subset.symbol.eq(symbol)].sort_values("day")
                for window in ("common", "fixed"):
                    for basis in ("mean", "median"):
                        ratio = f"r_{basis}"+ ("_fixed" if window=="fixed" else "")
                        for indicator in ("st", "sma"):
                            column = ("fixed_" if window=="fixed" else "")+f"time5_{indicator}"
                            effect = bootstrap_summary(sample[column], sample[ratio], sample.segment, repetitions, block, seed)
                            rows.append({"period": period, "block": block, "symbol": symbol, "window": window, "basis": basis, "indicator": indicator, **effect})
            print(f"Пятиминутная статистика: {period}, блок {block} — готово", flush=True)
    return adjust_families(pd.DataFrame(rows))


def table_html(data, sensitivity=False):
    """Возвращает HTML таблицу data с эффектом, p, ДИ и русскими названиями."""
    headings = ["Инструмент", "Окно", "База объёма", "Индикатор"]
    if sensitivity:
        headings.append("Блок, дней")
    headings += ["Дни: высокий / обычный", "Обычный объём", "Высокий объём", "Изменение", "Разница и 95% ДИ", "p", "p Holm", "Вывод"]
    lines = ['<div class="table-scroll"><table><thead><tr>'+"".join(f"<th>{escape(h)}</th>" for h in headings)+"</tr></thead><tbody>"]
    for row in data.itertuples():
        cells = [row.symbol, WINDOWS[row.window], BASIS[row.basis], METRICS[row.indicator]]
        if sensitivity:
            cells.append(str(row.block))
        cls = "support" if row.verdict=="Поддерживается" else "opposite" if row.verdict=="Обратная связь" else "neutral"
        cells += [f"{row.n_high} / {row.n_low}", number(row.low_mean), number(row.high_mean), number(row.change_percent, 1)+"%",
                  f"{number(row.difference)} [{number(row.ci_low)}; {number(row.ci_high)}]", p_text(row.p), p_text(row.p_holm), row.verdict]
        lines.append('<tr>'+"".join(f'<td class="{cls if i==len(cells)-1 else ""}">{escape(str(c))}</td>' for i,c in enumerate(cells))+"</tr>")
    return "\n".join(lines)+"</tbody></table></div>"


def chart_html(figure, identifier):
    """Встраивает интерактивный Plotly figure с identifier без повторной библиотеки."""
    figure.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif", size=13, color="#243044"), margin=dict(l=70,r=35,t=70,b=65), paper_bgcolor="#ffffff")
    return figure.to_html(full_html=False, include_plotlyjs=False, include_mathjax=False, div_id=identifier,
                          config={"responsive": True, "displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}})


def forest_chart(main):
    """Строит интерактивный график разницы событий main с симметричными 95%-ными ДИ."""
    figure = make_subplots(rows=1, cols=2, subplot_titles=list(WINDOWS.values()), horizontal_spacing=.18)
    for column, window in enumerate(WINDOWS, 1):
        for basis, marker in (("median", "circle"), ("mean", "diamond")):
            sample = main[main.window.eq(window) & main.basis.eq(basis)].copy()
            labels = [f"{r.symbol} · {METRICS[r.indicator]}" for r in sample.itertuples()]
            detail = [[BASIS[r.basis], r.low_mean, r.high_mean, r.change_percent, r.p_holm, r.verdict] for r in sample.itertuples()]
            figure.add_trace(go.Scatter(x=sample.difference, y=labels, mode="markers", marker=dict(size=10, symbol=marker, color="#259c8c" if basis=="median" else "#6b72b9"),
                                       error_x=dict(type="data", array=sample.ci_high-sample.difference, arrayminus=sample.difference-sample.ci_low),
                                       name=BASIS[basis], legendgroup=basis, showlegend=column==1, customdata=detail,
                                       hovertemplate="%{y}<br>%{customdata[0]}<br>Разница: %{x:.3f}<br>Обычный: %{customdata[1]:.3f}<br>Высокий: %{customdata[2]:.3f}<br>Изменение: %{customdata[3]:.1f}%<br>p Holm: %{customdata[4]:.4f}<br>%{customdata[5]}<extra></extra>"), row=1, col=column)
        figure.add_vline(x=0, line_dash="dash", line_color="#596477", row=1,col=column)
        figure.update_xaxes(title_text="Высокий − обычный, событий за окно", row=1,col=column)
    figure.update_layout(height=440, legend=dict(orientation="h", y=-.25), hovermode="closest")
    figure.update_yaxes(autorange="reversed")
    return chart_html(figure, "effect-forest")


def means_chart(main):
    """Показывает средние события main при медианной базе в четырёх интерактивных панелях."""
    titles = [f"{WINDOWS[w]} · {METRICS[m]}" for w in WINDOWS for m in METRICS]
    figure = make_subplots(rows=2, cols=2, subplot_titles=titles, vertical_spacing=.19)
    for index, (window, indicator) in enumerate((w,m) for w in WINDOWS for m in METRICS):
        sample = main[main.window.eq(window) & main.indicator.eq(indicator) & main.basis.eq("median")]
        for label, field, color in (("Объём ≤ медианы", "low_mean", COLORS["low"]), ("Объём > медианы", "high_mean", COLORS["high"])):
            figure.add_trace(go.Bar(x=sample.symbol,y=sample[field],name=label,legendgroup=label,showlegend=index==0,marker_color=color,
                                    text=sample[field].round(2),textposition="outside",hovertemplate="%{x}<br>Среднее число событий: %{y:.3f}<extra>%{fullData.name}</extra>"),row=index//2+1,col=index%2+1)
        figure.update_yaxes(title_text="Событий за окно", rangemode="tozero", row=index//2+1,col=index%2+1)
    figure.update_layout(height=700,barmode="group",legend=dict(orientation="h",y=-.12))
    return chart_html(figure,"group-means")


def distributions_chart(daily):
    """Показывает распределение пятиминутных событий daily за 2026 при медианной базе."""
    sample = daily[daily.period.eq("2026")]
    figure = make_subplots(rows=1,cols=2,subplot_titles=["SuperTrend · общее окно", "SMA 3/34 · общее окно"])
    for column, indicator in enumerate(METRICS,1):
        for high, label, color in ((False,"Объём ≤ медианы",COLORS["low"]),(True,"Объём > медианы",COLORS["high"])):
            group = sample[sample.r_median.gt(1).eq(high)]
            figure.add_trace(go.Box(x=group.symbol,y=group[f"time5_{indicator}"],name=label,legendgroup=label,showlegend=column==1,
                                    marker_color=color,boxpoints="all",jitter=.25,pointpos=0,marker_size=4),row=1,col=column)
        figure.update_yaxes(title_text="Событий за окно", row=1,col=column)
    figure.update_layout(height=480,boxmode="group",legend=dict(orientation="h",y=-.2))
    return chart_html(figure,"daily-distributions")


def annual_chart(daily):
    """Показывает годовые изменения средних daily при медианной базе без новых p-тестов."""
    figure=make_subplots(rows=1,cols=2,subplot_titles=["SuperTrend", "SMA 3/34"])
    for column, indicator in enumerate(METRICS,1):
        for symbol in ("RTS","MIX"):
            for window, dash in (("common","solid"),("fixed","dot")):
                values=[]
                for year, group in daily[daily.symbol.eq(symbol)].groupby("year"):
                    prefix="fixed_" if window=="fixed" else ""
                    ratio="r_median_fixed" if window=="fixed" else "r_median"
                    result=group_effect(group[f"{prefix}time5_{indicator}"],group[ratio])
                    values.append((int(year),result["change_percent"],result["n_high"],result["n_low"]))
                figure.add_trace(go.Scatter(x=[v[0] for v in values],y=[v[1] for v in values],mode="lines+markers",line=dict(color=COLORS[symbol],dash=dash),
                                           name=f"{symbol} · {WINDOWS[window]}",legendgroup=f"{symbol}-{window}",showlegend=column==1,
                                           customdata=[[v[2],v[3]] for v in values],hovertemplate="%{x}<br>Изменение: %{y:.1f}%<br>Дней высокий/обычный: %{customdata[0]}/%{customdata[1]}<extra>%{fullData.name}</extra>"),row=1,col=column)
        figure.add_hline(y=0,line_dash="dash",line_color="#596477",row=1,col=column)
        figure.update_yaxes(title_text="Изменение среднего, %", row=1,col=column)
        figure.update_xaxes(dtick=1,row=1,col=column)
    figure.update_layout(height=460,legend=dict(orientation="h",y=-.25))
    return chart_html(figure,"annual-effects")


def ratio_chart(daily):
    """Строит описательную зависимость 5-минутных событий и ER от относительного объёма daily."""
    figure=make_subplots(rows=1,cols=3,subplot_titles=["SuperTrend", "SMA 3/34", "Эффективность пути ER"])
    for symbol in ("RTS","MIX"):
        group=daily[daily.symbol.eq(symbol)&daily.period.eq("2026")].copy()
        group["bin"]=pd.cut(group.r_median,[-np.inf,.75,1,1.25,1.5,np.inf],labels=["≤0,75","0,75–1","1–1,25","1,25–1,5",">1,5"])
        summary=group.groupby("bin",observed=False).agg(st=("time5_st","mean"),sma=("time5_sma","mean"),er=("time5_er","mean"),n=("day","size"))
        for column, indicator in enumerate(("st","sma","er"),1):
            figure.add_trace(go.Scatter(x=summary.index.astype(str),y=summary[indicator],mode="lines+markers",name=symbol,legendgroup=symbol,showlegend=column==1,
                                       line=dict(color=COLORS[symbol]),customdata=summary.n,hovertemplate="R: %{x}<br>Среднее: %{y:.3f}<br>Дней: %{customdata}<extra>%{fullData.name}</extra>"),row=1,col=column)
        figure.update_xaxes(title_text="Объём / прошлая медиана")
    figure.update_layout(height=420,legend=dict(orientation="h",y=-.25))
    return chart_html(figure,"relative-volume")


def coverage_html(daily):
    """Возвращает таблицу покрытия daily за 2026 с числом дней, баров и длительностью."""
    lines=['<div class="table-scroll"><table><thead><tr><th>Инструмент</th><th>Окно</th><th>Пригодных дней</th><th>Медиана 5-мин баров</th><th>Медиана часов</th><th>Пустых свечей, медиана</th></tr></thead><tbody>']
    for symbol in ("RTS","MIX"):
        for window in WINDOWS:
            prefix="fixed_" if window=="fixed" else ""
            group=daily[daily.symbol.eq(symbol)&daily.period.eq("2026")&daily[f"{prefix}time5_st"].notna()]
            hours=8.75 if window=="fixed" else group.hours.median()
            cells=[symbol,WINDOWS[window],str(len(group)),number(group[f"{prefix}time5_bars"].median(),0),number(hours),number(100*group[f"{prefix}time5_empty_fraction"].median(),1)+"%"]
            lines.append("<tr>"+"".join(f"<td>{escape(c)}</td>" for c in cells)+"</tr>")
    return "\n".join(lines)+"</tbody></table></div>"


def build_html(daily, stats, config, repetitions, source_hash):
    """Возвращает автономный HTML по daily/stats, конфигурации config и хешу source_hash."""
    main=stats[stats.period.eq("2026")&stats.block.eq(20)]
    historical=stats[stats.period.eq("2022–2025")&stats.block.eq(20)]
    full=stats[stats.period.eq("все")&stats.block.eq(20)]
    sensitivity=stats[stats.period.eq("2026")].sort_values(["symbol","window","basis","indicator","block"])
    supported=main[main.difference.lt(0)&main.p_holm.lt(.05)]
    conclusion=overall_verdict(main)
    sma_conclusion=indicator_conclusion(main,"sma")
    st_conclusion=indicator_conclusion(main,"st")
    day_counts=[int(main[main.symbol.eq(symbol)&main.window.eq("common")&main.basis.eq("median")&main.indicator.eq("st")][["n_high","n_low"]].sum(axis=1).iloc[0]) for symbol in ("RTS","MIX")]
    descriptions=[]
    for row in supported.itertuples():
        descriptions.append(f"<li><strong>{row.symbol}, {METRICS[row.indicator]}</strong> · {WINDOWS[row.window]}, база — {BASIS[row.basis].lower()}: снижение на <strong>{number(-row.change_percent,1)}%</strong>, p Holm = {p_text(row.p_holm)}.</li>")
    stability=stats[stats.period.eq("2026")].groupby(["symbol","window","basis","indicator"]).apply(lambda g: bool((g.difference.lt(0)&g.p_holm.lt(.05)).all()),include_groups=False)
    stable_count=int(stability.sum())
    generated=datetime.now().strftime("%d.%m.%Y")
    css="""
    :root{--ink:#233044;--muted:#64718a;--line:#dfe6ef;--green:#117564;--bg:#eef2f7;--blue:#245eb5}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,sans-serif;line-height:1.6}
    main{max-width:1320px;margin:auto;padding:36px 28px 64px}header{padding:28px 32px;background:#173452;color:white;border-radius:16px}
    h1{font-size:34px;line-height:1.2;margin:10px 0 14px}h2{font-size:24px;margin:0 0 14px}h3{font-size:18px;margin:22px 0 8px}
    p{margin:10px 0}.eyebrow{font-size:12px;letter-spacing:1.3px;text-transform:uppercase;color:#b8cee3}
    .lead{font-size:18px;max-width:1000px}.meta{font-size:13px;color:#c6d6e6}nav{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}
    a{color:var(--blue)}nav a{text-decoration:none;background:white;padding:8px 14px;border:1px solid var(--line);border-radius:20px;font-size:14px}
    section{background:white;border:1px solid var(--line);border-radius:14px;padding:28px;margin:20px 0;overflow:hidden}
    .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}.card{background:#f5f8fc;padding:20px;border-radius:10px;border:1px solid var(--line)}
    .value{font-size:28px;font-weight:700;color:#173452}.small{font-size:13px;color:var(--muted)}.callout{border-left:4px solid var(--green);background:#edf8f4;padding:16px 20px;border-radius:6px}
    .note{border-left:4px solid #b38a2f;background:#fff9e9;padding:14px 18px;border-radius:6px}.table-scroll{overflow:auto;margin:20px 0}
    table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;background:#f0f4f9;position:sticky;top:0;white-space:nowrap}
    td,th{border-bottom:1px solid var(--line);padding:11px 10px;vertical-align:top}td{white-space:nowrap}tr:hover td{background:#f7faff}
    td.support{color:#08745a;font-weight:700}td.opposite{color:#aa332e;font-weight:700}td.neutral{color:var(--muted)}
    .chart{width:100%;min-height:300px}.formula{padding:12px 16px;background:#f5f7fb;font-family:Consolas,monospace;border-radius:8px;overflow:auto}
    details{margin:16px 0}summary{cursor:pointer;font-weight:700;color:var(--blue)}footer{padding:14px 0;color:var(--muted);font-size:13px}
    pre{white-space:pre-wrap;background:#f5f7fb;padding:16px;border-radius:8px;font-size:13px}li{margin:7px 0}
    @media(max-width:760px){main{padding:15px 10px}header,section{padding:20px 15px}h1{font-size:27px}.cards{grid-template-columns:1fr}}
    @media print{body{background:white}main{padding:0}section{break-inside:avoid;border:none}nav{display:none}.modebar{display:none!important}}
    """
    body=f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Объём и трендовость — пятиминутные бары RTS/MIX</title><style>{css}</style><script>{get_plotlyjs()}</script></head><body><main>
    <header><div class="eyebrow">Исследование тиковых данных · пятиминутные свечи</div><h1>Высокий объём делает день более трендовым?</h1>
    <p class="lead">{escape(overall_verdict(main))}</p><p class="meta">RTS и MIX · данные 2022–02.10.2026 · основная оценка: 2026 · отчёт {generated}</p></header>
    <nav><a href="#conclusion">Вывод</a><a href="#method">Методика</a><a href="#main-tests">Значимость</a><a href="#charts">Графики</a><a href="#stability">Устойчивость</a><a href="#history">История</a><a href="#limits">Ограничения</a></nav>
    <section id="conclusion"><h2>Что показали данные</h2><div class="cards"><div class="card"><div class="value">{len(supported)} / {len(main)}</div><div>значимых снижений после Holm</div></div><div class="card"><div class="value">{' + '.join(map(str,day_counts))}</div><div>дней общего окна в 2026</div></div><div class="card"><div class="value">{stable_count} / {len(main)}</div><div>снижений значимы при всех блоках 20/40/60</div></div></div>
    <div class="callout"><strong>{escape(conclusion)}</strong><p>{escape(sma_conclusion)}</p></div>
    <h3>Где снижение значимо при базовом блоке 20 дней</h3>{'<ul>'+''.join(descriptions)+'</ul>' if descriptions else '<p>Ни одно сравнение не показало значимого снижения после общей поправки.</p>'}
    <p>{escape(st_conclusion)}</p>
    <div class="note">Результаты пятиминутных контролей уже были просмотрены в предыдущем исследовании. Этот отчёт — последующий анализ той же выборки с явной поправкой на 16 проверок; 2026 нельзя считать новой, ранее не изученной независимой выборкой.</div></section>
    <section id="method"><h2>Как выполнено сравнение</h2><p>Для каждого дня берутся 20 предыдущих доступных будних дат с тиками. Субботы и воскресенья исключаются из базы и прогрева. Объём — сумма исходного поля <code>volume</code>; исследуемый день в базу не входит.</p>
    <div class="formula">R = объём исследуемого дня / среднее или медиана объёма предыдущих 20 дней<br>Высокий объём: R &gt; 1; обычный объём: R ≤ 1</div>
    <p><strong>Общее окно:</strong> самая поздняя первая сделка и самая ранняя последняя сделка среди всех 21 дней. Обе границы включены, клиринги не вычитаются. Исторические объёмы заново суммируются в тех же часах. Для событий используются только полные пятиминутные свечи внутри окна.</p>
    <p><strong>Фиксированное окно:</strong> 10:00–18:45. Оно было выбрано до расчёта индикаторов по распределению первых/последних сделок. Все 21 дня должны охватывать его; объём и принадлежность к группам пересчитываются именно для этого окна. Поэтому состав групп двух окон различается.</p>
    <p><strong>SuperTrend:</strong> ATR Уайлдера 10, множитель 3. <strong>SMA:</strong> периоды 3 и 34 по close. Событие — смена итогового направления или знака разности SMA; равенства SMA не добавляют лишних пересечений. Прогрев — 100 предыдущих баров, ежедневного сброса нет, ночной переход не считается. Внутридневные интервалы без сделок получают предыдущую цену и нулевой объём.</p>
    {coverage_html(daily)}
    <h3>Статистический критерий</h3><p>Единица наблюдения — день. Разница Δ = среднее число событий при высоком объёме − среднее при обычном. Отрицательная Δ поддерживает гипотезу. Двусторонний центрированный блочный bootstrap: {repetitions} повторов, блок 20 последовательных торговых наблюдений, seed {config['seed']}; блоки не переходят через выделенные разрывы.</p>
    <p>Основная семья — <strong>16 сравнений 2026 года</strong>: 2 инструмента × 2 индикатора × 2 базы объёма × 2 окна. Статистическая значимость определяется по <strong>p Holm &lt; 0,05</strong>. Исторический и полный периоды имеют отдельные семьи по 16 и служат дополнительным контекстом; они не добавляют независимые подтверждения к 2026.</p>
    <p>95%-ный интервал Δ симметричен: Δ ± 95%-квантиль абсолютной центрированной bootstrap-ошибки. Он использует ту же ошибку, что и p-тест. Интервалы <strong>индивидуальные</strong>, без поправки на 16 сравнений: интервал может исключать ноль, а скорректированное p всё ещё быть ≥ 0,05. В предыдущем отчёте были процентильные ДИ; здесь выбран согласованный с тестом способ построения интервала.</p></section>
    <section id="main-tests"><h2>Основные результаты 2026 года</h2><p>Средние — число событий за торговое окно. Процентное изменение рассчитано относительно группы обычного объёма. В последнем столбце решение принимается по скорректированному p, а не по одному знаку эффекта.</p>
    {table_html(main)}<h3>Разница событий и её неопределённость</h3><p class="small">Точка левее нуля означает меньше событий при высоком объёме. Наведите курсор для p Holm и итогового решения. Нажатие на легенду включает или скрывает базу объёма.</p>
    {forest_chart(main)}</section>
    <section id="charts"><h2>Средние и распределения</h2><p class="small">Графики интерактивные: масштабирование, подсказки, переключение серий в легенде и сохранение PNG через панель инструментов. Следующие графики используют медианную базу объёма.</p>
    {means_chart(main)}<h3>Разброс между днями</h3><p>Точки — отдельные дни. Ящики показывают медиану и межквартильный диапазон; различие средних не означает, что каждый день высокого объёма более трендовый.</p>
    {distributions_chart(daily)}<h3>Зависимость от величины относительного объёма</h3><p>Пять диапазонов R показывают форму связи. ER = |последний close − первый close| / сумма абсолютных изменений close. Больший ER соответствует более направленному пути. Это описательный дополнительный показатель, он не входит в 16 основных тестов; малые группы нельзя считать отдельным подтверждением.</p>
    {ratio_chart(daily)}</section>
    <section id="stability"><h2>Устойчивость к длине статистического блока</h2><p>Тот же эффект оценён с блоками 20, 40 и 60 дат. Это проверка устойчивости уже заданных сравнений; наиболее выгодный блок не выбирается. Значимое снижение при всех трёх блоках обнаружено в {stable_count} из 16 сочетаний.</p>
    <details><summary>Показать все 48 оценок 2026 года</summary>{table_html(sensitivity,True)}</details>
    <h3>Изменение эффекта по годам</h3><p class="small">Годовые проценты описательные, без дополнительных тестов значимости. Отрицательные значения — меньше событий при высоком объёме. Пунктир обозначает фиксированное окно.</p>
    {annual_chart(daily)}</section>
    <section id="history"><h2>Исторический период и вся выборка</h2><p><strong>2022–2025:</strong> {escape(overall_verdict(historical))}</p>
    <details><summary>Результаты 2022–2025 с поправкой на 16 сравнений</summary>{table_html(historical)}</details>
    <p><strong>Весь период 2022–02.10.2026:</strong> {escape(overall_verdict(full))}</p>
    <details><summary>Результаты всей выборки с поправкой на 16 сравнений</summary>{table_html(full)}</details>
    <p>Вся выборка включает оба периода и не является третьей независимой проверкой. Исторические оценки чувствительны к смене рыночных режимов и длины торгового окна.</p></section>
    <section id="limits"><h2>Что можно и нельзя заключить</h2><ul>
    <li>{escape(conclusion)} Отсутствие значимости не доказывает нулевой эффект, а связь не доказывает причинность.</li>
    <li>Первая/последняя сделки зависят от активности и полноты архива. Историческое расписание не восстанавливалось, клиринги не вычитались — это согласованные упрощения.</li>
    <li>Будние даты без архива не всегда можно отличить от биржевых выходных. В предыдущем аудите отмечены 50 одинаковых отсутствующих дат на инструмент; повреждений и расхождений сверки не найдено. Покрытие основной выборки приведено в таблице методики.</li>
    <li>В CSV нет кода отдельного контракта: полностью проверить ролловеры непрерывной склейки нельзя. Связь не означает причинность.</li>
    <li>Полный дневной объём известен к концу дня. Отчёт не проверяет прогноз трендовости оставшейся сессии по утреннему объёму.</li>
    <li>Если сегмент не длиннее блока, он фиксируется; при отсутствии длинных сегментов ДИ/p не оцениваются. В историческом периоде неопределённость фиксированных коротких частей не учтена; в 2026 каждый инструмент имеет один сегмент 194 дня.</li>
    </ul><details><summary>Происхождение данных и воспроизведение</summary><p>Источник: проверенная <code>daily.csv</code>. SHA256: <code>{source_hash}</code>. Статистические таблицы этого отчёта сохранены как <code>five_minute_statistics.csv</code>; параметры и хеши — <code>five_minute_report_manifest.json</code>.</p>
    <pre>.\\.venv\\Scripts\\python.exe -m research.volume_trend_5m_report --source results\\volume_trend --repetitions {repetitions}
.\\.venv\\Scripts\\python.exe -m unittest -v tests.test_volume_trend_5m_report</pre></details></section>
    <footer>Единый автономный HTML: Plotly, данные и оформление встроены. Для чтения не нужны интернет, сервер или дополнительные файлы.</footer>
    </main></body></html>"""
    return body


def main():
    """Читает параметры, пересчитывает статистику и сохраняет HTML/CSV/манифест отчёта."""
    parser=argparse.ArgumentParser(description="Автономный интерактивный отчёт по пятиминутным барам")
    parser.add_argument("--source",type=Path,default=Path("results/volume_trend"),help="Папка дневной таблицы и конфигурации")
    parser.add_argument("--html",type=Path,help="Путь итогового HTML; по умолчанию five_minute_report.html в --source")
    parser.add_argument("--repetitions",type=int,default=5000,help="Число bootstrap-повторов, минимум 100")
    args=parser.parse_args()
    if args.repetitions<100:
        parser.error("Требуется не меньше 100 повторов")
    source=args.source.resolve()
    destination=args.html.resolve() if args.html else source/"five_minute_report.html"
    data=pd.read_csv(source/"daily.csv")
    data=data[data.valid_observation].copy()
    config=json.loads((source/"config.json").read_text(encoding="utf-8"))
    stats=statistics(data,args.repetitions,config["seed"])
    stats.to_csv(source/"five_minute_statistics.csv",index=False)
    source_hash=sha256((source/"daily.csv").read_bytes()).hexdigest()
    html=build_html(data,stats,config,args.repetitions,source_hash)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(html,encoding="utf-8")
    manifest={"source_sha256":source_hash,"script_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),"generated":datetime.now().isoformat(timespec="seconds"),
              "repetitions":args.repetitions,"seed":config["seed"],"primary_period":"2026","family_size":16,"blocks":[20,40,60],"ci":"симметричный по абсолютной центрированной ошибке","html":str(destination)}
    (source/"five_minute_report_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Готово: {destination}",flush=True)


if __name__=="__main__":
    main()
