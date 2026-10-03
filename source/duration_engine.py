"""Модель исполнения стратегии по длительности завершённых дельта-баров.

Запуск исследования: python backtest/backtest_duration.py --symbols RTS MIX
Проверки: python -m unittest -v tests.test_duration_engine
"""

from dataclasses import dataclass, replace
import numpy as np


@dataclass
class Event:
    """Хранит доступный тик исполнения либо наблюдение цены для переоценки."""
    time: str
    row: int
    price: float
    duration: int = -1
    direction: int = 0
    enter: bool = False
    exit: bool = False
    force: bool = False
    signal_bar: int = -1
    signal_close: float | None = None
    signal_alf: float | None = None


@dataclass
class Day:
    """Объединяет события одного дня с обязательным закрытием по времени."""
    date: str
    events: list


@dataclass
class GridResult:
    """Содержит валовую дневную прибыль и число завершённых сделок всей сетки."""
    dates: list
    params: np.ndarray
    gross: np.ndarray
    count: np.ndarray


@dataclass
class SingleResult:
    """Хранит журнал сделок, дневной результат и переоценку открытых позиций."""
    trades: list
    daily: list
    equity: list


def reverse_directions(days):
    """Меняет сторону уже отобранных входов, сохраняя события и исходные объекты."""
    return [Day(day.date, [replace(event, direction=-event.direction) for event in day.events])
            for day in days]


def parameter_grid(entries, exits):
    """Создаёт упорядоченную сетку положительных целых порогов exit > entry."""
    entries, exits = list(entries), list(exits)
    if not entries or not exits or any(int(x) != x or x <= 0 for x in entries+exits):
        raise ValueError('Пороги должны быть положительными целыми секундами')
    pairs = [(a,b) for a in sorted(set(entries)) for b in sorted(set(exits)) if b > a]
    if not pairs:
        raise ValueError('Нет пар, где порог выхода больше порога входа')
    return np.asarray(pairs,dtype=np.int64)


def simulate_grid(days, params):
    """Одновременно считает все пары с исполнением только на последующих тиках."""
    gross = np.zeros((len(days),len(params)))
    count = np.zeros_like(gross,dtype=np.int32)
    entries, exits = params.T
    for di,day in enumerate(days):
        position = np.zeros(len(params),dtype=np.int8)
        entry_price = np.zeros(len(params))
        for event in day.events:
            if not (event.enter or event.exit or event.force):
                continue
            flat = position == 0
            closing = ~flat & (event.force | (event.exit & (event.duration > exits)))
            gross[di,closing] += position[closing]*(event.price-entry_price[closing])
            count[di,closing] += 1
            position[closing] = 0
            if event.enter and event.direction and not event.force:
                opening = flat & (event.duration < entries)
                position[opening] = event.direction
                entry_price[opening] = event.price
        if np.any(position):
            raise ValueError(f'{day.date}: нет исполнения закрытия по времени')
    return GridResult([x.date for x in days],params,gross,count)


def simulate_one(days, entry, exit, cost_points=0):
    """Считает подробный журнал одной пары и PnL открытой позиции на каждом событии."""
    if entry <= 0 or exit <= entry or not np.isfinite(cost_points) or cost_points < 0:
        raise ValueError('Нужны 0 < вход < выход и неотрицательные издержки')
    trades, daily, equity = [], [], []
    cumulative_gross, sides = 0.0, 0
    for day in days:
        position, opening = 0, None
        gross_day, count_day = 0.0, 0
        for event in day.events:
            was_flat = position == 0
            if position and (event.force or (event.exit and event.duration > exit)):
                pnl = position*(event.price-opening['price'])
                gross_day += pnl
                cumulative_gross += pnl
                count_day += 1
                sides += 1
                trades.append(dict(day=day.date,entry_seconds=entry,exit_seconds=exit,
                    side='Long' if position>0 else 'Short',entry_time=opening['time'],
                    exit_time=event.time,entry_row=opening['row'],exit_row=event.row,
                    entry_price=opening['price'],exit_price=event.price,
                    entry_duration=opening['duration'],exit_duration=event.duration,
                    entry_signal_bar=opening['signal_bar'],exit_signal_bar=event.signal_bar,
                    entry_signal_close=opening['signal_close'],entry_signal_alf=opening['signal_alf'],
                    reason='time' if event.force else 'duration',gross_pnl=pnl,
                    cost_points=cost_points,net_pnl=pnl-cost_points))
                position = 0
            if was_flat and event.enter and not event.force and event.direction and event.duration < entry:
                position = event.direction
                opening = dict(time=event.time,row=event.row,price=event.price,
                               duration=event.duration,signal_bar=event.signal_bar,
                               signal_close=event.signal_close,signal_alf=event.signal_alf)
                sides += 1
            unrealized = position*(event.price-opening['price']) if position else 0.0
            equity.append(dict(day=day.date,time=event.time,row=event.row,
                               gross_pnl=cumulative_gross+unrealized,
                               net_pnl=cumulative_gross+unrealized-sides*cost_points/2))
        if position:
            raise ValueError(f'{day.date}: осталась открытая позиция')
        daily.append(dict(day=day.date,gross_pnl=gross_day,trades=count_day,
                          net_pnl=gross_day-count_day*cost_points))
    return SingleResult(trades,daily,equity)
