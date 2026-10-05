r"""Проверяет SMA 3/34 на пяти минутах с фильтром объёма того же времени 20 дней.

Примеры из корня проекта:
    .\.venv\Scripts\python.exe -m backtest.backtest_sma_volume
    .\.venv\Scripts\python.exe backtest/backtest_sma_volume.py --start 2026-01-01 --end 2026-10-02
    .\.venv\Scripts\python.exe -m backtest.backtest_sma_volume --costs 0,2,4,8 --base-cost 4 --repetitions 5000
    .\.venv\Scripts\python.exe -m unittest -v tests.test_backtest_sma_volume

Использует журнал проверенного исследования results/volume_trend. Читает
исходные ZIP заново, сверяет SHA256 и все OHLCV пяти минут с сохранёнными
свечами, включая прогрев. Субботы/воскресенья не участвуют. Нужны ровно 20
предыдущих доступных будних дат; все 21 сессии должны охватывать 10:00–18:45,
как в фиксированном контроле предыдущей гипотезы. Медиана накопленного
объёма [10:00,t) вычисляется без текущего дня. Первому входу нужен объём
выше медианы, последующим переворотам фильтр не нужен. Вход — первый тик
следующей свечи; выход — первый тик с 18:45. Стопов, целей, ночных позиций
и подбора параметров нет. Одновременно проверяется стратегия без фильтра.
PnL — пункты одного контракта; затраты — сценарии шагов цены за круг.
Дни без сделок сохраняются. Парный bootstrap/Holm оценивает изменение PnL.
CSV, метаданные, журнал покрытия и автономный интерактивный HTML сохраняются
в новой папке run_ внутри results/sma_volume; прежние результаты не меняются.
"""

import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):
    sys.path.insert(0,str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.volume_trend import segment_map
from research.volume_trend_data import read_ticks, time_bars, time_text
from research.volume_trend_metrics import holm
from source.sma_volume_engine import START,CLOSE,accumulated_volumes,sma_frame,make_signals,run_day,summarize,paired_bootstrap

BAR_FIELDS=["open","high","low","close","volume","start_ns","end_ns","is_complete"]
TICK_SIZES={"RTS":10.,"MIX":25.}


def load_verified_day(source, symbol, row, expected_bars=None):
    """Возвращает тики и свечи row из ZIP, сверяя SHA256 и снимок expected_bars.

    source — папка прошлого исследования, symbol — инструмент, row — запись
    аудита с датой, путём, хешем, границами и объёмом. Без expected_bars читает
    сохранённый 5m.pkl. Повреждение или расхождение останавливает бэктест.
    """
    payload=Path(row["file"]).read_bytes()
    if sha256(payload).hexdigest()!=row["sha256"]:
        raise ValueError(f"{symbol} {row['day']}: SHA256 исходного ZIP изменился")
    times,prices,volumes=read_ticks(BytesIO(payload),row["day"])
    if (len(times)!=row["tick_count"] or int(volumes.sum())!=row["volume"]
            or times[0]!=row["first_ns"] or times[-1]!=row["last_ns"]):
        raise ValueError(f"{symbol} {row['day']}: тики расходятся с аудитом")
    bars=expected_bars if expected_bars is not None else pd.read_pickle(source/"cache"/symbol/f"{row['day'].replace('-','')}_5m.pkl")
    actual=time_bars(times,prices,volumes,5)
    if len(actual)!=len(bars) or not np.array_equal(actual[BAR_FIELDS].to_numpy(),bars[BAR_FIELDS].to_numpy()):
        raise ValueError(f"{symbol} {row['day']}: пятиминутные свечи расходятся с исходными тиками")
    if not np.allclose(prices/TICK_SIZES[symbol],np.round(prices/TICK_SIZES[symbol]),rtol=0,atol=1e-7):
        raise ValueError(f"{symbol} {row['day']}: цены не кратны шагу {TICK_SIZES[symbol]}")
    return times,prices,volumes,bars


def run_symbol(args, symbol):
    """Возвращает сделки, дни, сигналы, покрытие и сверку одного symbol по args."""
    audit=pd.read_csv(args.source/f"{symbol}_audit.csv")
    audit=audit[audit.weekday.lt(5)&audit.day.le(args.end)].sort_values("day")
    # SMA имеет конечную память: для прогрева и объёма достаточно 20 прошлых дат.
    audit=pd.concat([audit[audit.day.lt(args.start)].tail(20),audit[audit.day.ge(args.start)]],ignore_index=True)
    if not audit.valid.all():
        raise ValueError(f"{symbol}: аудит содержит повреждённый будний день")
    segments=segment_map(audit)
    originals={}
    pieces=[]
    for row in audit.itertuples():
        bars=pd.read_pickle(args.source/"cache"/symbol/f"{row.day.replace('-','')}_5m.pkl")
        originals[row.day]=bars
        piece=bars.copy()
        piece["day"],piece["segment"],piece["bar_index"]=row.day,segments[row.day],np.arange(1,len(bars)+1)
        pieces.append(piece)
    if not pieces:
        raise ValueError(f"{symbol}: нет исходных дней")
    frame=sma_frame(pd.concat(pieces,ignore_index=True))
    grouped={day:group for day,group in frame.groupby("day",sort=False)}
    history=deque(maxlen=20)
    trades,days,signals,coverage,verification=[],[],[],[],[]
    for count,(_,row) in enumerate(audit.iterrows(),1):
        times,prices,volumes,bars=load_verified_day(args.source,symbol,row,originals[row.day])
        curve=accumulated_volumes(times,volumes)
        full=bool(times[0]<=START and times[-1]>=CLOSE)
        verification.append(dict(symbol=symbol,day=row.day,source_sha256=row.sha256,ticks=len(times),
                                 volume=int(volumes.sum()),bars=len(bars),verified=True))
        if args.start<=row.day<=args.end:
            reason=""
            if len(history)!=20:
                reason="Меньше 20 прошлых будних торговых дат"
            elif not full:
                reason="Текущая сессия не охватывает 10:00–18:45"
            elif not all(h["full"] for h in history):
                reason="Не все 20 прошлых сессий охватывают 10:00–18:45"
            day_bars=grouped[row.day]
            if not reason and not day_bars.warmed.any():
                reason="Недостаточный прогрев SMA"
            entry=dict(symbol=symbol,day=row.day,status="excluded" if reason else "included",reason=reason,
                       history_days=json.dumps([h["day"] for h in history]),first_ns=int(times[0]),last_ns=int(times[-1]),
                       segment=segments[row.day])
            if not reason:
                targets=make_signals(day_bars,times,curve,np.asarray([h["curve"] for h in history]))
                entry.update(signals=len(targets),passing_signals=sum(t["volume_pass"] for t in targets),
                             close_delay_seconds=float((times[np.searchsorted(times,CLOSE)]-CLOSE)/1e9))
                signals.extend(dict(t,symbol=symbol,day=row.day,signal_time=f"{row.day} {time_text(t['signal_ns'])}") for t in targets)
                for mode in ("baseline","volume"):
                    for cost in args.costs:
                        result=run_day(row.day,times,prices,targets,TICK_SIZES[symbol],cost,mode)
                        trades.extend(dict(t,symbol=symbol,cost_ticks=cost) for t in result["trades"])
                        days.append(dict(result["daily"],symbol=symbol,segment=segments[row.day]))
            coverage.append(entry)
        history.append(dict(day=row.day,curve=curve,full=full))
        if count%100==0 or count==len(audit):
            print(f"{symbol}: сверено {count}/{len(audit)} будних архивов, включено {sum(r['status']=='included' for r in coverage)} дней",flush=True)
    return trades,days,signals,coverage,verification


def result_tables(trades, daily, costs, repetitions):
    """Возвращает сводки и парные тесты trades/daily по периодам и costs.

    Главный период — 2026; исторический и общий дают контекст. В каждом
    периоде/блоке Holm охватывает инструменты и все сценарии затрат.
    """
    summaries,tests=[],[]
    periods=[("2026","2026-01-01","2026-12-31"),("2022–2025","2022-01-01","2025-12-31"),("все","0000","9999")]
    periods.extend((str(y),f"{y}-01-01",f"{y}-12-31") for y in range(2022,2026))
    for period,start,end in periods:
        selected=daily[daily.day.between(start,end)]
        for (symbol,mode,cost),group in selected.groupby(["symbol","mode","cost_ticks"],sort=False):
            chosen=trades[trades.symbol.eq(symbol)&trades["mode"].eq(mode)&trades.cost_ticks.eq(cost)&trades.day.between(start,end)]
            summaries.append(dict(period=period,symbol=symbol,mode=mode,cost_ticks=cost,**summarize(chosen,group)))
        if period not in ("2026","2022–2025","все"):
            continue
        for symbol,group in selected.groupby("symbol",sort=False):
            for cost in costs:
                sample=group[group.cost_ticks.eq(cost)]
                paired=sample.pivot(index="day",columns="mode",values="net_pnl").sort_index()
                if len(paired)==0:
                    continue
                segments=sample.drop_duplicates("day").set_index("day").loc[paired.index,"segment"]
                for block in ((20,40,60) if period=="2026" else (20,)):
                    effect=paired_bootstrap(paired.volume-paired.baseline,segments,repetitions,block)
                    tests.append(dict(period=period,symbol=symbol,cost_ticks=cost,block=block,days=len(paired),
                                      total_difference=float((paired.volume-paired.baseline).sum()),**effect))
    tests=pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"]=np.nan
        for _,group in tests.groupby(["period","block"],sort=False):
            tests.loc[group.index,"p_holm"]=holm(group.p)
        tests["verdict"]=["Недостаточно данных" if not np.isfinite(p) else "Нет значимого различия" if p>=.05 else
                          "Фильтр увеличил PnL" if d>0 else "Фильтр уменьшил PnL" for d,p in zip(tests.difference,tests.p_holm)]
    return pd.DataFrame(summaries),tests


def parse_args(argv=None):
    """Проверяет argv и возвращает пути, инструменты, даты, сценарии затрат и bootstrap."""
    parser=argparse.ArgumentParser(description="Бэктест SMA 3/34 с объёмом того же времени прошлых 20 дней")
    parser.add_argument("--source",type=Path,default=PROJECT_ROOT/"results/volume_trend")
    parser.add_argument("--output",type=Path,default=PROJECT_ROOT/"results/sma_volume")
    parser.add_argument("--symbols",nargs="+",choices=["RTS","MIX"],default=["RTS","MIX"])
    parser.add_argument("--start",default="2022-01-01")
    parser.add_argument("--end",default="2026-10-02")
    parser.add_argument("--costs",default="0,2,4,8",help="Затраты за круг в шагах цены")
    parser.add_argument("--base-cost",type=float,default=4)
    parser.add_argument("--repetitions",type=int,default=5000)
    args=parser.parse_args(argv)
    try:
        args.costs=sorted(set(float(x) for x in args.costs.split(",")))
        if (not args.costs or not np.isfinite(args.costs).all() or min(args.costs)<0
                or args.base_cost not in args.costs or args.repetitions<100):
            raise ValueError("Неотрицательные затраты, base-cost из costs и минимум 100 повторов")
        if pd.Timestamp(args.start)>pd.Timestamp(args.end):
            raise ValueError("Начало периода позже окончания")
        args.start=pd.Timestamp(args.start).date().isoformat()
        args.end=pd.Timestamp(args.end).date().isoformat()
        args.source,args.output=args.source.resolve(),args.output.resolve()
        args.symbols=list(dict.fromkeys(args.symbols))
    except (ValueError,TypeError) as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    """Выполняет проверку и бэктест по argv, сохраняет CSV/метаданные/HTML и возвращает папку."""
    args=parse_args(argv)
    created=datetime.now(timezone(timedelta(hours=3)))
    folder=args.output/f"run_{created.strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"progress.md").write_text("# Бэктест SMA 3/34 и накопленного объёма\n\nНачат; исходные ZIP проверяются заново.\n",encoding="utf-8")
    combined=[[],[],[],[],[]]
    for symbol in args.symbols:
        for target,part in zip(combined,run_symbol(args,symbol)):
            target.extend(part)
    trade_rows,daily_rows,signal_rows,coverage_rows,verification_rows=combined
    trades=pd.DataFrame(trade_rows)
    if trades.empty:
        trades=pd.DataFrame(columns=["day","symbol","mode","cost_ticks","side","gross_pnl","net_pnl"])
    daily=pd.DataFrame(daily_rows)
    if daily.empty:
        raise ValueError("Нет пригодных дней для бэктеста")
    coverage=pd.DataFrame(coverage_rows)
    summary,comparisons=result_tables(trades,daily,args.costs,args.repetitions)
    for name,frame in (("trades",trades),("daily",daily),("signals",pd.DataFrame(signal_rows)),("coverage",coverage),
                       ("verification",pd.DataFrame(verification_rows)),("summary",summary),("comparisons",comparisons)):
        frame.to_csv(folder/f"{name}.csv",index=False)
    sources=["backtest/backtest_sma_volume.py","source/sma_volume_engine.py","source/sma_volume_report.py",
             "research/volume_trend.py","research/volume_trend_data.py","research/volume_trend_metrics.py"]
    metadata=dict(created=created.isoformat(timespec="seconds"),settings=dict(start=args.start,end=args.end,symbols=args.symbols,
        costs=args.costs,base_cost=args.base_cost,repetitions=args.repetitions,seed=20261005,lookback=20,
        session_start="10:00",session_close="18:45",bar_minutes=5,sma=[3,34],warmup=100,
        volume_rule="Объём [10:00,t) > медиана [10:00,t) предыдущих 20 будних дат",reversal_filter=False),
        tick_sizes=TICK_SIZES,source=str(args.source),source_hashes={name:sha256((args.source/name).read_bytes()).hexdigest()
            for name in ["audit.csv","config.json",*[f"{s}_audit.csv" for s in args.symbols]]},
        scripts={name:sha256((PROJECT_ROOT/name).read_bytes()).hexdigest() for name in sources},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        family_size=len(args.symbols)*len(args.costs),verification_days=len(verification_rows))
    (folder/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    from source.sma_volume_report import write_report
    report=write_report(folder,summary,daily,trades,comparisons,coverage,metadata)
    (folder/"progress.md").write_text(f"# Бэктест завершён\n\nСверено {len(verification_rows)} будних ZIP.\n\nОтчёт: {report.name}.\n",encoding="utf-8")
    print(f"Готово: {report}",flush=True)
    print(summary[summary.period.eq("2026")&summary.cost_ticks.eq(args.base_cost)].to_string(index=False),flush=True)
    return folder


if __name__=="__main__":
    main()
