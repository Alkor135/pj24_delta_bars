"""Проверка стратегии длительности дельта-баров RTS/MIX и сравнение графиков PnL.

Примеры запуска из папки проекта:
    python backtest_duration.py
    python backtest_duration.py --symbols RTS --entry-grid 1:45:1 --exit-grid 5:300:5
    python backtest_duration.py --symbols MIX --costs 0,2,4,8 --base-cost 4
    python backtest_duration.py --session-start 10:00 --entry-end 18:30 --close-time 18:40

Издержки задаются в шагах цены за полный круг, PnL — в пунктах на один контракт.
Базы и тиковые ZIP не изменяются. Каждый запуск создаёт отдельный каталог отчёта.
"""

import argparse
from contextlib import closing
from datetime import datetime,date
from hashlib import sha256
import html
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
import pandas as pd
from source.duration_engine import parameter_grid,simulate_grid,simulate_one
from source.duration_data import Session,prepare_database
from source.duration_analysis import grid_statistics,rank_parameters,comparison_indices,walk_forward,trade_metrics,drawdown


def parse_grid(value):
    """Разбирает включительный диапазон начало:конец:шаг в целых секундах."""
    start,stop,step=map(int,value.split(':'))
    if start<1 or stop<start or step<1:
        raise ValueError('Сетка должна иметь вид 1:45:1 с положительным шагом')
    return list(range(start,stop+1,step))


def matrix_map(params,values,title,value_label='PnL',kind='diverging'):
    """Преобразует значения допустимых пар в прямоугольную тепловую карту."""
    entries=sorted(set(int(x) for x in params[:,0]))
    exits=sorted(set(int(x) for x in params[:,1]))
    lookup={(int(a),int(b)):float(v) for (a,b),v in zip(params,values)}
    return dict(title=title,value_label=value_label,kind=kind,entries=entries,exits=exits,z=[[
        lookup.get((a,b)) if np.isfinite(lookup.get((a,b),np.nan)) else None for a in entries] for b in exits])


def build_payload(symbol,days,grid,coverage,meta,tick_size,costs,base_cost,holdout_start,min_trades=100,top=5):
    """Выбирает сравниваемые пары только по обучению и сверяет сетку с журналом сделок."""
    dates=np.asarray(grid.dates)
    development=dates<holdout_start
    holdout=~development
    if not development.any():
        raise ValueError('Нет периода подбора до начала финальной проверки')
    points=base_cost*tick_size
    ranked=rank_parameters(grid_statistics(grid,development,points),grid.params,min_trades)
    selected=comparison_indices(ranked,grid.params,top)
    gross_stats=grid_statistics(grid,development,0)
    gross_best=int(gross_stats.sort_values(['gross_pnl','entry_seconds','exit_seconds'],
                                         ascending=[False,True,True],kind='stable').index[0])
    if gross_best not in selected:
        selected.append(gross_best)
    summary,curves,monthly,sensitivity=[],[],[],[]
    journals,dailies,equities=[],[],[]
    for index in selected:
        entry,exit=map(int,grid.params[index])
        label=f'{entry} / {exit} с'
        detailed=simulate_one(days,entry,exit,points)
        daily=pd.DataFrame(detailed.daily)
        np.testing.assert_allclose(daily.gross_pnl,grid.gross[:,index],rtol=0,atol=1e-9)
        np.testing.assert_array_equal(daily.trades,grid.count[:,index])
        if abs(sum(x['net_pnl'] for x in detailed.trades)-daily.net_pnl.sum())>1e-7:
            raise ValueError('Журнал сделок не совпал с дневным PnL')
        trades=pd.DataFrame(detailed.trades)
        equity=pd.DataFrame(detailed.equity)
        net=daily.net_pnl.cumsum().to_numpy()
        gross=daily.gross_pnl.cumsum().to_numpy()
        intraday=equity.copy()
        intraday['drawdown']=drawdown(intraday.net_pnl.to_numpy())
        minima=intraday.groupby('day',sort=False).drawdown.min().reindex(grid.dates).to_numpy()
        hold_eq=equity[equity.day>=holdout_start].net_pnl.to_numpy()-daily.net_pnl.to_numpy()[development].sum()
        hold_dd=float(-drawdown(hold_eq).min()) if len(hold_eq) else None
        hold_trades=[x for x in detailed.trades if x['day']>=holdout_start]
        stats=trade_metrics(hold_trades)
        summary.append(dict(label=label,entry_seconds=entry,exit_seconds=exit,
            selection_basis='Лидер до издержек (диагностика)' if index==gross_best else 'Соседняя область по опорным издержкам',
            development_pnl=float(daily.net_pnl.to_numpy()[development].sum()),
            holdout_pnl=float(daily.net_pnl.to_numpy()[holdout].sum()),
            development_trades=int(daily.trades.to_numpy()[development].sum()),
            holdout_trades=stats['trades'],holdout_max_dd=hold_dd,
            holdout_profit_factor=stats['profit_factor'],holdout_win_rate=stats['win_rate'],
            holdout_long_pnl=stats['long_pnl'],holdout_short_pnl=stats['short_pnl'],
            robust_score=float(ranked.loc[index,'robust_score']) if np.isfinite(ranked.loc[index,'robust_score']) else None,
            training_eligible=bool(ranked.loc[index,'eligible'])))
        curves.append(dict(label=label,dates=grid.dates,net=net.tolist(),gross=gross.tolist(),
                           drawdown=drawdown(net).tolist(),intraday_drawdown=minima.tolist()))
        grouped=daily.assign(month=daily.day.str[:7]).groupby('month').net_pnl.sum()
        monthly.append(dict(label=label,months=grouped.index.tolist(),pnl=grouped.tolist()))
        sensitivity.append(dict(label=label,cost_ticks=costs,
            development_pnl=[float((grid.gross[development,index]-c*tick_size*grid.count[development,index]).sum()) for c in costs],
            holdout_pnl=[float((grid.gross[holdout,index]-c*tick_size*grid.count[holdout,index]).sum()) for c in costs]))
        for frame,target in ((trades,journals),(daily,dailies),(intraday,equities)):
            frame['label']=label
            target.append(frame)
    wf=walk_forward(grid,points,holdout_start,min_trades)
    metrics_frames=[]
    for cost in costs:
        frame=grid_statistics(grid,development,cost*tick_size)
        frame['cost_ticks']=cost
        metrics_frames.append(frame)
    metadata=dict(meta,units='пункты котировки на 1 контракт',holdout_start=holdout_start,
        development_start=str(dates[development][0]),development_end=str(dates[development][-1]),
        holdout_end=str(dates[holdout][-1]) if holdout.any() else None,
        base_cost_ticks=base_cost,cost_scenarios_ticks=costs,tick_size=tick_size,
        parameter_pairs=len(grid.params),min_train_trades=min_trades,
        training_eligible_pairs=int(ranked.eligible.sum()),
        selection='Медиана mean/std дневного PnL соседей 3×3, только обучение; финальный участок не участвует.',
        disclaimer='Исполнение по следующей сделке без стакана и задержки; сценарии издержек не являются тарифом брокера. '
        'Доли секунды искусственные. PnL не в рублях. Просадка открытой позиции измерена на открытиях/закрытиях баров '
        'и тике дневного выхода, поэтому может быть меньше внутритиковой. Исключённые дни видны в покрытии. '
        'Кривые сравнения выбраны по обучению и могут быть убыточными. Нулевой walk-forward означает отказ от торговли. '
        'После просмотра контроля его нельзя считать новым независимым тестом при изменении правил.')
    payload=dict(symbol=symbol,meta=metadata,summary=summary,curves=curves,monthly=monthly,costs=sensitivity,
        heatmaps=[matrix_map(grid.params,frame.net_pnl,f'PnL подбора: {cost:g} шагов за круг')
                  for cost,frame in zip(costs,metrics_frames)]+[
                  matrix_map(grid.params,ranked.max_drawdown,'Максимальная дневная просадка на периоде подбора','Просадка','risk'),
                  matrix_map(grid.params,ranked.robust_score,'Устойчивость: медиана качества соседей','Качество'),
                  matrix_map(grid.params,ranked.trades,'Количество сделок на периоде подбора','Сделки','count')],
        walk_forward=wf,coverage=coverage,trades=journals[0].head(200).to_dict('records') if journals else [],
        links=[dict(title='Таблица параметров периода подбора',href='grid_development.csv'),
               dict(title='Все сделки выбранных параметров',href='trades.csv'),
               dict(title='Результаты в SQLite',href='results.sqlite3'),
               dict(title='Окна последовательной проверки',href='walk_forward.csv'),
               dict(title='Покрытие исходных данных',href='coverage.csv')])
    tables=dict(grid_development=pd.concat(metrics_frames,ignore_index=True),
        ranking=ranked,summary=pd.DataFrame(summary),trades=pd.concat(journals,ignore_index=True),
        daily=pd.concat(dailies,ignore_index=True),equity=pd.concat(equities,ignore_index=True),
        coverage=pd.DataFrame(coverage),walk_forward=pd.DataFrame(wf['folds']),
        walk_forward_daily=pd.DataFrame(dict(day=wf['dates'],net_pnl=wf['daily_net'],
                                            gross_pnl=wf['daily_gross'],trades=wf['daily_trades'])))
    return payload,tables


def save_tables(folder,grid,payload,tables):
    """Сохраняет воспроизводимые таблицы, полный дневной массив сетки и описание запуска."""
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    database=folder/'results.sqlite3'
    if database.exists():
        raise ValueError('Каталог результатов уже содержит базу; используйте новый каталог')
    with closing(sqlite3.connect(database)) as db:
        for name,frame in tables.items():
            clean=frame.replace([np.inf,-np.inf],np.nan)
            if len(clean.columns):
                clean.to_sql(name,db,index=False)
            if name!='equity':
                clean.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
        db.execute('CREATE TABLE metadata (json TEXT NOT NULL)')
        db.execute('INSERT INTO metadata VALUES (?)',(json.dumps(payload['meta'],ensure_ascii=False),))
        db.commit()
    np.savez_compressed(folder/'grid_daily.npz',dates=np.asarray(grid.dates),params=grid.params,
                        gross=grid.gross,count=grid.count)
    (folder/'metadata.json').write_text(json.dumps(payload['meta'],ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'summary.json').write_text(json.dumps(payload['summary'],ensure_ascii=False,indent=2),encoding='utf-8')


def arguments():
    """Задаёт изменяемые диапазоны, часы и сценарии затрат исследования."""
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--symbols',nargs='+',choices=['RTS','MIX'],default=['RTS','MIX'])
    parser.add_argument('--data-dir',type=Path,default=Path('C:/data_quote'))
    parser.add_argument('--db',type=Path,help='Явная база: допускается с одним символом')
    parser.add_argument('--dataset-id')
    parser.add_argument('--start',default='2022-01-01')
    parser.add_argument('--end',default='9999-12-31')
    parser.add_argument('--entry-grid',default='1:45:1')
    parser.add_argument('--exit-grid',default='5:300:5')
    parser.add_argument('--costs',default='0,2,4,8',help='Шаги цены за вход и выход вместе')
    parser.add_argument('--base-cost',type=float,default=4)
    parser.add_argument('--tick-size',type=float,help='Переопределение шага цены для одного символа')
    parser.add_argument('--session-start',default='10:00')
    parser.add_argument('--entry-end',default='18:30')
    parser.add_argument('--close-time',default='18:40')
    parser.add_argument('--holdout-start',default='2026-01-01')
    parser.add_argument('--min-trades',type=int,default=100)
    parser.add_argument('--top',type=int,default=5)
    parser.add_argument('--output-dir',type=Path,default=Path('C:/data_quote/duration_backtests'))
    parser.add_argument('--cache-dir',type=Path,default=Path(__file__).resolve().parent/'.duration_cache')
    return parser.parse_args()


def main():
    """Запускает исследование, сохраняет отдельный отчёт каждого инструмента и общий индекс."""
    args=arguments()
    if len(args.symbols)!=len(set(args.symbols)):
        raise ValueError('Символы не должны повторяться')
    if len(args.symbols)!=1 and (args.db or args.tick_size is not None):
        raise ValueError('--db и --tick-size требуют одного символа')
    for value in (args.start,args.end,args.holdout_start):
        date.fromisoformat(value)
    if args.start>args.end or args.start>=args.holdout_start:
        raise ValueError('Нужен непустой период подбора до финальной проверки')
    costs=sorted(set(float(x) for x in args.costs.split(','))|{args.base_cost})
    if any(not np.isfinite(x) or x<0 for x in costs) or args.min_trades<1 or not 1<=args.top<=20:
        raise ValueError('Неверные издержки, число сделок или количество графиков')
    params=parameter_grid(parse_grid(args.entry_grid),parse_grid(args.exit_grid))
    if len(params)>100000:
        raise ValueError('Сетка слишком велика: максимум 100000 пар')
    session=Session(args.session_start,args.entry_end,args.close_time)
    run=args.output_dir/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True,exist_ok=False)
    report_links=[]
    from source.duration_report import write_report
    for symbol in args.symbols:
        db=args.db or args.data_dir/(symbol+'_delta_bars.sqlite3')
        days,coverage,meta=prepare_database(db,symbol,session,args.cache_dir,args.start,args.end,args.dataset_id)
        meta['implementation_sha256']={name:sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
            for name in ('backtest_duration.py','source/duration_engine.py','source/duration_data.py','source/duration_analysis.py','source/duration_report.py')}
        tick_size=args.tick_size if args.tick_size is not None else {'RTS':10.0,'MIX':25.0}[symbol]
        if not np.isfinite(tick_size) or tick_size<=0:
            raise ValueError('Шаг цены должен быть положительным')
        observed=np.array([event.price for day in days for event in day.events])
        if not np.allclose(observed/tick_size,np.round(observed/tick_size),rtol=0,atol=1e-7):
            raise ValueError('Цены не соответствуют шагу: уточните --tick-size')
        print(f'{symbol}: расчёт {len(params)} пар на {len(days)} днях...',flush=True)
        grid=simulate_grid(days,params)
        print(f'{symbol}: журнал сделок и проверка вне периода подбора...',flush=True)
        payload,tables=build_payload(symbol,days,grid,coverage,meta,tick_size,costs,args.base_cost,
                                    args.holdout_start,args.min_trades,args.top)
        folder=run/symbol
        save_tables(folder,grid,payload,tables)
        report=write_report(folder,payload)
        report_links.append((symbol,report.relative_to(run).as_posix()))
        print(f'{symbol}: отчёт {report}',flush=True)
        print(tables['summary'][['entry_seconds','exit_seconds','development_pnl','holdout_pnl','holdout_trades']].to_string(index=False),flush=True)
    links=''.join(f'<li><a href="{html.escape(link,quote=True)}">{html.escape(symbol)} — графики и результаты</a></li>' for symbol,link in report_links)
    (run/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Стратегия длительности</title>'
        '<style>body{font:18px Segoe UI,sans-serif;max-width:900px;margin:60px auto;background:#f4f7fb;color:#17283d}'
        'li{margin:24px 0}a{color:#165ab6}</style><h1>Стратегия длительности дельта-баров</h1>'
        '<p>Вход 1–45 с, выход 5–300 с по умолчанию. Точные настройки внутри отчёта каждого инструмента.</p>'
        '<p>PnL в пунктах на один контракт. Сценарии издержек — предположения для сравнения.</p><ul>'+links+'</ul></html>',encoding='utf-8')
    print(f'Готово: {run / "index.html"}',flush=True)


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,sqlite3.Error) as error:
        print(f'Исследование остановлено: {error}',file=sys.stderr)
        sys.exit(1)
