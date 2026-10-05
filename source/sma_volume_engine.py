r"""Моделирует SMA 3/34 с причинным фильтром накопленного объёма на пяти минутах.

Запуск из корня проекта:
    .\.venv\Scripts\python.exe -m backtest.backtest_sma_volume
    .\.venv\Scripts\python.exe -m unittest -v tests.test_backtest_sma_volume

Объём [10:00,t) сравнивается с медианой того же интервала 20 прошлых будних
дней. Сигнал появляется на закрытии свечи, сделка — первым тиком следующей
свечи. Фильтр нужен только для первого входа; затем обратные пересечения
переворачивают позицию. В 18:45 имеет приоритет дневной выход. Затраты за
круг делятся между входом/выходом, просадка учитывает все тики позиции.
Парный блочный bootstrap сравнивает дневные PnL с фильтром и без него.
"""

import numpy as np
import pandas as pd

from research.volume_trend_data import NS, time_text

START = 36000 * NS
CLOSE = 67500 * NS
WIDTH = 300 * NS
CUTOFFS = np.arange(START + WIDTH, CLOSE, WIDTH, dtype=np.int64)


def accumulated_volumes(times, volumes):
    """Возвращает 104 объёма [10:00,t) по times/volumes для закрытий до 18:45."""
    times, volumes = np.asarray(times), np.asarray(volumes)
    if len(times)!=len(volumes) or (np.diff(times)<0).any() or (volumes<0).any():
        raise ValueError("Неверные времена или объёмы тиков")
    prefix = np.r_[np.int64(0), volumes.cumsum(dtype=np.int64)]
    return prefix[np.searchsorted(times,CUTOFFS,side="left")] - prefix[np.searchsorted(times,START,side="left")]


def sma_frame(bars):
    """Возвращает bars с SMA 3/34, прогревом 100 и дневными пересечениями cross.

    bars упорядочены по дате/времени, содержат close/day/segment. Скользящие
    средние продолжаются между днями одного сегмента; смена знака через
    границу дня не считается. Равенства сохраняют последний ненулевой знак.
    """
    parts=[]
    for _, group in bars.groupby("segment",sort=False):
        group=group.copy()
        group["sma3"]=group.close.rolling(3).mean()
        group["sma34"]=group.close.rolling(34).mean()
        group["warmed"]=np.arange(len(group))>=100
        signs=np.sign(group.sma3-group.sma34).to_numpy()
        crosses=np.zeros(len(group),dtype=np.int8)
        previous,previous_day=0,None
        for i,(sign,day,warmed) in enumerate(zip(signs,group.day,group.warmed)):
            if day!=previous_day:
                previous=0
                previous_day=day
            if not warmed or not np.isfinite(sign):
                previous=0
                continue
            if sign:
                if previous and sign!=previous:
                    crosses[i]=int(sign)
                previous=sign
        group["cross"]=crosses
        parts.append(group)
    return pd.concat(parts,ignore_index=True) if parts else bars.copy()


def make_signals(bars, times, current_curve, history_curves):
    """Возвращает исполнимые пересечения bars и причинный фильтр на каждом сигнале.

    times — тики текущего дня; current_curve — 104 накопленных объёма;
    history_curves — матрица ровно 20×104 прошлых дней. Вход разрешён только
    внутри непосредственно следующей свечи и строго до 18:45.
    """
    history_curves=np.asarray(history_curves)
    if history_curves.shape!=(20,len(CUTOFFS)) or len(current_curve)!=len(CUTOFFS):
        raise ValueError("Для фильтра нужны ровно 20 прошлых дней и 104 отсечения")
    medians=np.median(history_curves,axis=0)
    targets=[]
    for bar in bars.itertuples():
        cutoff=int(bar.end_ns)+1
        if not bar.warmed or not bar.cross or bar.start_ns<START or cutoff>=CLOSE:
            continue
        offset=int((cutoff-START)//WIDTH-1)
        if offset<0 or offset>=len(CUTOFFS) or cutoff!=CUTOFFS[offset]:
            raise ValueError("Сигнал не соответствует пятиминутной сетке")
        index=int(np.searchsorted(times,cutoff,side="left"))
        if index==len(times) or times[index]>=min(cutoff+WIDTH,CLOSE):
            continue
        current,median=int(current_curve[offset]),float(medians[offset])
        targets.append(dict(entry_index=index,side=int(bar.cross),signal_bar=int(bar.bar_index),
            signal_ns=cutoff,sma3=float(bar.sma3),sma34=float(bar.sma34),current_volume=current,
            median_volume=median,volume_ratio=current/median if median>0 else np.nan,
            volume_pass=bool(current>median),entry_delay_seconds=float((times[index]-cutoff)/NS)))
    return targets


def run_day(day, times, prices, targets, tick_size, cost_ticks, mode):
    """Возвращает сделки и дневной исход по times/prices/targets для day.

    tick_size переводит cost_ticks за круг в пункты; mode равен volume или
    baseline. Первому входу volume нужен volume_pass, перевороты не повторяют
    фильтр. Выход — первый тик >=18:45, имеет приоритет над новым сигналом.
    Просадка рассчитывается по всем тикам открытой позиции с обеими платами.
    """
    times,prices=np.asarray(times),np.asarray(prices,float)
    if mode not in ("volume","baseline") or len(times)!=len(prices) or not len(times):
        raise ValueError("Неверный режим или массивы тиков")
    if ((np.diff(times)<0).any() or not np.isfinite(prices).all() or (prices<=0).any()
            or not np.isfinite([tick_size,cost_ticks]).all() or tick_size<=0 or cost_ticks<0):
        raise ValueError("Некорректные цены, порядок тиков или затраты")
    close_index=int(np.searchsorted(times,CLOSE,side="left"))
    if close_index==len(times):
        raise ValueError("Нет тика для закрытия с 18:45")
    fee=float(tick_size*cost_ticks)
    trades=[]
    position=None
    realized=peak=minimum=max_dd=0.
    mark_start=0
    previous_index=-1
    rejected=0

    def record_path(values):
        """Добавляет equity values к дневным максимуму, минимуму и полной просадке."""
        nonlocal peak,minimum,max_dd
        values=np.asarray(values,float)
        if len(values):
            running=np.maximum.accumulate(np.r_[peak,values])[1:]
            max_dd=max(max_dd,float(np.max(running-values)))
            peak=max(peak,float(values.max()))
            minimum=min(minimum,float(values.min()))

    def mark_to(index):
        """Оценивает открытую позицию по всем тикам до index включительно."""
        nonlocal mark_start
        if position is not None and mark_start<=index:
            record_path(realized-fee/2+position["direction"]*(prices[mark_start:index+1]-position["entry_price"]))
            mark_start=index+1

    def close_position(index, reason, signal=None):
        """Закрывает текущую позицию по тику index и reason; signal задаёт обратный сигнал."""
        nonlocal position,realized
        gross=position["direction"]*(float(prices[index])-position["entry_price"])
        trade={k:v for k,v in position.items() if k!="direction"}
        trade.update(exit_row=index+1,exit_time=f"{day} {time_text(times[index])}",exit_price=float(prices[index]),
                     exit_signal_time=f"{day} {time_text(signal['signal_ns'])}" if signal else None,
                     reason=reason,gross_pnl=gross,cost_points=fee,net_pnl=gross-fee)
        trades.append(trade)
        realized+=gross-fee
        record_path([realized])
        position=None

    for signal in targets:
        index,direction=int(signal["entry_index"]),int(signal["side"])
        if index<=previous_index or direction not in (-1,1) or index<0:
            raise ValueError("Нарушен порядок или направление сигналов")
        previous_index=index
        if index>=close_index:
            break
        if not START<=times[index]<CLOSE or times[index]<signal["signal_ns"]:
            raise ValueError("Непричинный вход или исполнение вне сессии")
        if position is None and mode=="volume" and not signal["volume_pass"]:
            rejected+=1
            continue
        if position is not None and position["direction"]==direction:
            continue
        mark_to(index)
        if position is not None:
            close_position(index,"Обратное пересечение",signal)
        position=dict(signal,day=day,mode=mode,side="Long" if direction==1 else "Short",direction=direction,
                      entry_row=index+1,entry_time=f"{day} {time_text(times[index])}",entry_price=float(prices[index]),
                      signal_time=f"{day} {time_text(signal['signal_ns'])}")
        mark_start=index
        record_path([realized-fee/2])
    if position is not None:
        mark_to(close_index)
        close_position(close_index,"Закрытие сессии")
    return dict(trades=trades,daily=dict(day=day,mode=mode,trades=len(trades),
        gross_pnl=float(sum(t["gross_pnl"] for t in trades)),net_pnl=realized,equity_peak=peak,
        equity_min=minimum,max_drawdown=max_dd,signals=len(targets),rejected_first_signals=rejected,
        cost_ticks=cost_ticks,close_delay_seconds=float((times[close_index]-CLOSE)/NS)))


def summarize(trades, days):
    """Возвращает PnL, сделки и сквозную тиковую просадку таблиц trades/days одного режима."""
    cumulative=peak=max_dd=0.
    for row in days.sort_values("day").itertuples():
        max_dd=max(max_dd,row.max_drawdown,peak-cumulative-row.equity_min)
        peak=max(peak,cumulative+row.equity_peak)
        cumulative+=row.net_pnl
    pnl=trades.net_pnl.to_numpy() if len(trades) else np.array([])
    profit=float(pnl[pnl>0].sum())
    loss=float(-pnl[pnl<0].sum())
    return dict(days=len(days),active_days=int(days.trades.gt(0).sum()),trades=len(pnl),
        gross_pnl=float(days.gross_pnl.sum()),net_pnl=float(days.net_pnl.sum()),
        mean_day=float(days.net_pnl.mean()),mean_trade=float(pnl.mean()) if len(pnl) else np.nan,
        win_rate=float((pnl>0).mean()) if len(pnl) else np.nan,
        profit_factor=profit/loss if loss else np.nan,max_drawdown=float(max_dd),
        long_pnl=float(trades[trades.side.eq("Long")].net_pnl.sum()) if len(trades) else 0.,
        short_pnl=float(trades[trades.side.eq("Short")].net_pnl.sum()) if len(trades) else 0.)


def paired_bootstrap(differences, segments, repetitions=5000, block=20, seed=20261005):
    """Возвращает среднюю дневную разность differences, симметричный ДИ и двустороннее p.

    Разности сохраняют парность двух стратегий по дате; segments запрещают
    переход блоков через разрыв. repetitions/block/seed задают число повторов,
    длину блока и генератор. Короткие сегменты фиксированы; без длинных ДИ/p нет.
    """
    values=np.asarray(differences,float)
    segments=np.asarray(segments)
    if len(values)!=len(segments) or not np.isfinite(values).all():
        raise ValueError("Для парного теста нужны конечные разности каждой даты")
    effect=float(values.mean()) if len(values) else np.nan
    result=dict(difference=effect,ci_low=np.nan,ci_high=np.nan,p=np.nan,bootstrap_valid=0)
    groups=[np.flatnonzero(segments==label) for label in np.unique(segments)]
    if not any(len(g)>block for g in groups):
        return result
    rng=np.random.default_rng(seed)
    distribution=[]
    for begin in range(0,repetitions,250):
        count=min(250,repetitions-begin)
        pieces=[]
        for group in groups:
            width=min(block,len(group))
            starts=rng.integers(0,len(group)-width+1,size=(count,int(np.ceil(len(group)/width))))
            indices=(starts[:,:,None]+np.arange(width)).reshape(count,-1)[:,:len(group)]
            pieces.append(group[indices])
        index=np.concatenate(pieces,axis=1)
        distribution.extend(values[index].mean(axis=1).tolist())
    errors=np.abs(np.asarray(distribution)-effect)
    radius=float(np.quantile(errors,.95,method="higher"))
    result.update(ci_low=effect-radius,ci_high=effect+radius,
                  p=float((np.count_nonzero(errors>=abs(effect))+1)/(len(errors)+1)),bootstrap_valid=len(errors))
    return result
