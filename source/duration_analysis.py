"""Оценка сетки порогов и последовательная проверка без подбора на будущем.

Запуск: python backtest_duration.py --symbols RTS MIX
Проверки: python -m unittest -v tests.test_duration_analysis
"""
import numpy as np
import pandas as pd


def drawdown(cumulative):
    """Возвращает отрицательную просадку с учётом начального капитала, равного нулю PnL."""
    values=np.asarray(cumulative,dtype=float)
    return values-np.maximum.accumulate(np.maximum(values,0),axis=0)


def grid_statistics(grid, mask, cost_points):
    """Считает показатели только выбранного временного участка по дневному PnL."""
    mask=np.asarray(mask,dtype=bool)
    gross=grid.gross[mask]
    counts=grid.count[mask]
    net=gross-cost_points*counts
    size=len(grid.params)
    total=net.sum(axis=0)
    count=counts.sum(axis=0)
    std=net.std(axis=0,ddof=1) if len(net)>1 else np.zeros(size)
    mean=net.mean(axis=0) if len(net) else np.zeros(size)
    quality=np.divide(mean,std,out=np.zeros(size),where=std>0)
    dd=-drawdown(np.cumsum(net,axis=0)).min(axis=0) if len(net) else np.zeros(size)
    return pd.DataFrame(dict(entry_seconds=grid.params[:,0],exit_seconds=grid.params[:,1],
        gross_pnl=gross.sum(axis=0),net_pnl=total,trades=count,
        average_trade=np.divide(total,count,out=np.zeros(size),where=count>0),
        max_drawdown=dd,daily_quality=quality,days=len(net)))


def rank_parameters(stats, params, min_trades=100):
    """Оценивает медиану ближайших соседей по двум осям, используя только обучение."""
    result=stats.copy()
    entries=sorted(set(params[:,0]))
    exits=sorted(set(params[:,1]))
    locations={(int(a),int(b)):i for i,(a,b) in enumerate(params)}
    scores=[]
    for a,b in params:
        ai,bi=entries.index(a),exits.index(b)
        neighbors=[locations[(int(x),int(y))] for x in entries[max(0,ai-1):ai+2]
                   for y in exits[max(0,bi-1):bi+2] if (int(x),int(y)) in locations]
        qualities=result.iloc[neighbors].daily_quality.to_numpy().copy()
        qualities[result.iloc[neighbors].trades.to_numpy()<min_trades]=-np.inf
        scores.append(float(np.median(qualities)))
    result['robust_score']=scores
    result['eligible']=(result.trades>=min_trades)&(result.net_pnl>0)&(result.robust_score>0)
    return result


def comparison_indices(ranked, params, limit=5):
    """Выбирает разные области по обучению, сохраняя убыточные примеры для диагностики."""
    ordered=ranked.sort_values(['robust_score','net_pnl','entry_seconds','exit_seconds'],
                              ascending=[False,False,True,True],kind='stable').index
    estep=min(np.diff(sorted(set(params[:,0]))),default=1)
    xstep=min(np.diff(sorted(set(params[:,1]))),default=1)
    selected=[]
    for index in ordered:
        if all(abs(params[index,0]-params[other,0])>=4*estep or
               abs(params[index,1]-params[other,1])>=4*xstep for other in selected):
            selected.append(int(index))
        if len(selected)>=limit:
            break
    return selected


def trade_metrics(rows):
    """Считает характеристики сделок без вымышленных значений при пустой выборке."""
    values=np.array([x['net_pnl'] for x in rows],dtype=float)
    wins=values[values>0].sum()
    losses=-values[values<0].sum()
    return dict(trades=len(values),net_pnl=float(values.sum()),
        average_trade=float(values.mean()) if len(values) else None,
        profit_factor=float(wins/losses) if losses else None,
        win_rate=float((values>0).mean()) if len(values) else None,
        long_pnl=float(sum(x['net_pnl'] for x in rows if x['side']=='Long')),
        short_pnl=float(sum(x['net_pnl'] for x in rows if x['side']=='Short')))


def walk_forward(grid, cost_points, holdout_start='2026-01-01', min_trades=100):
    """Подбирает на прошлых 12 месяцах и проверяет следующие 3, не затрагивая финальный участок."""
    dates=np.asarray(grid.dates)
    first=pd.Timestamp(grid.dates[0]).to_period('M').start_time
    boundary=first+pd.DateOffset(months=12)
    # Общие календарные кварталы, следующий после доступного года обучения.
    while boundary.month not in (1,4,7,10):
        boundary+=pd.DateOffset(months=1)
    stop=min(pd.Timestamp(holdout_start),pd.Timestamp(grid.dates[-1])+pd.Timedelta(days=1))
    folds=[]
    all_dates,all_net,all_gross,all_counts=[],[],[],[]
    while boundary<stop:
        right=min(boundary+pd.DateOffset(months=3),stop)
        left=boundary-pd.DateOffset(months=12)
        train=(dates>=left.strftime('%Y-%m-%d'))&(dates<boundary.strftime('%Y-%m-%d'))
        test=(dates>=boundary.strftime('%Y-%m-%d'))&(dates<right.strftime('%Y-%m-%d'))
        if not test.any():
            boundary=right
            continue
        ranked=rank_parameters(grid_statistics(grid,train,cost_points),grid.params,min_trades)
        candidates=ranked[ranked.eligible] if train.sum()>=60 else ranked.iloc[:0]
        chosen=None if candidates.empty else int(candidates.sort_values(
            ['robust_score','net_pnl','entry_seconds','exit_seconds'],ascending=[False,False,True,True],kind='stable').index[0])
        gross=grid.gross[test,chosen] if chosen is not None else np.zeros(test.sum())
        counts=grid.count[test,chosen] if chosen is not None else np.zeros(test.sum(),dtype=int)
        net=gross-cost_points*counts
        folds.append(dict(train_start=left.strftime('%Y-%m-%d'),
            train_end=(boundary-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            test_start=boundary.strftime('%Y-%m-%d'),test_end=(right-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            parameter_index=chosen,entry_seconds=int(grid.params[chosen,0]) if chosen is not None else None,
            exit_seconds=int(grid.params[chosen,1]) if chosen is not None else None,
            train_pnl=float(ranked.loc[chosen,'net_pnl']) if chosen is not None else None,
            robust_score=float(ranked.loc[chosen,'robust_score']) if chosen is not None else None,
            test_pnl=float(net.sum()),test_trades=int(counts.sum()),
            reason='Выбрано по прошлым 12 месяцам' if chosen is not None else 'Вне рынка: нет подходящих параметров'))
        all_dates.extend(dates[test].tolist())
        all_net.extend(net.tolist())
        all_gross.extend(gross.tolist())
        all_counts.extend(counts.tolist())
        boundary=right
    cumulative=np.cumsum(all_net)
    return dict(dates=all_dates,net=cumulative.tolist(),gross=np.cumsum(all_gross).tolist(),
        drawdown=drawdown(cumulative).tolist(),daily_net=all_net,daily_gross=all_gross,
        daily_trades=all_counts,folds=folds)
