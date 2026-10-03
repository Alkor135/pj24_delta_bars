"""Проверки обратного направления входов и неизменности условий исполнения.

Запуск из корня проекта: python -m unittest -v tests.test_duration_reversed
Используются собственные события; пользовательские котировки не изменяются.
Точка запуска импортируется из пакета backtest; модель исполнения — из source.
"""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from backtest import backtest_duration as runner
from source import duration_engine as engine
from source.duration_data import EntryFilter, apply_entry_filter


class ReversedTests(unittest.TestCase):
    """Сравнивает сделки исходной и обратной стратегии при одинаковых настройках."""

    def setUp(self):
        """Требует отдельного преобразования уже отобранных сигналов."""
        self.assertTrue(hasattr(engine, 'reverse_directions'), 'Нужно преобразование reverse_directions')

    def sample_day(self):
        """Создаёт два входа, дополнительный сигнал в позиции и разные причины выхода."""
        return engine.Day('2022-01-03', [
            engine.Event('10:00:01', 2, 100, 1, 1, True, True, signal_bar=0, signal_close=99, signal_alf=95),
            engine.Event('10:00:02', 4, 102, 1, -1, True, True, signal_bar=1, signal_close=101, signal_alf=105),
            engine.Event('10:01:10', 6, 90, 61, -1, True, True, signal_bar=2),
            engine.Event('10:01:12', 8, 110, 1, -1, True, True, signal_bar=3, signal_close=111, signal_alf=115),
            engine.Event('18:40:00', 10, 100, force=True)])

    def test_trade_times_prices_and_costs_stay_the_same(self):
        """Меняет сторону и валовый PnL, сохраняя исполнения, одну позицию и издержки."""
        days = [self.sample_day()]
        normal = engine.simulate_one(days, 2, 60, 4)
        reverse = engine.simulate_one(engine.reverse_directions(days), 2, 60, 4)
        self.assertEqual(len(normal.trades), 2)
        self.assertEqual(len(reverse.trades), 2)
        self.assertEqual([t['side'] for t in reverse.trades], ['Short', 'Long'])
        for first, second in zip(normal.trades, reverse.trades):
            for key in first:
                if key not in ('side', 'gross_pnl', 'net_pnl'):
                    self.assertEqual(first[key], second[key], key)
            self.assertEqual(second['gross_pnl'], -first['gross_pnl'])
            self.assertEqual(first['net_pnl'] + second['net_pnl'], -8)
        self.assertEqual(reverse.daily[0]['trades'], 2)
        for first, second in zip(normal.equity, reverse.equity):
            self.assertEqual(second['gross_pnl'], -first['gross_pnl'])
            self.assertEqual(first['gross_pnl'] - first['net_pnl'], second['gross_pnl'] - second['net_pnl'])

    def test_grid_inversion_and_scalar_agree(self):
        """Сетка сохраняет число сделок и инвертирует валовый результат каждой пары."""
        days = [self.sample_day()]
        flipped = engine.reverse_directions(days)
        params = engine.parameter_grid([1, 2, 5, 10], [20, 60, 90])
        normal = engine.simulate_grid(days, params)
        reverse = engine.simulate_grid(flipped, params)
        np.testing.assert_array_equal(normal.count, reverse.count)
        np.testing.assert_array_equal(-normal.gross, reverse.gross)
        for i, (entry, exit) in enumerate(params):
            detail = engine.simulate_one(flipped, int(entry), int(exit), 4)
            self.assertEqual(detail.daily[0]['gross_pnl'], reverse.gross[0, i])
            self.assertEqual(detail.daily[0]['trades'], reverse.count[0, i])

    def test_alf_is_checked_before_reversing_direction(self):
        """Продаёт по исходному разрешённому Long выше ALF и сохраняет запрет другого входа."""
        bars = pd.DataFrame({'bar_index': [0, 1, 2], 'close': [110., 90., 90.], 'alf': [100., 100., 100.]})
        day = engine.Day('2022-01-03', [
            engine.Event('10:00:01', 2, 115, 1, 1, True, True, signal_bar=0),
            engine.Event('10:00:02', 4, 95, 1, 1, True, True, signal_bar=1),
            engine.Event('10:00:03', 6, 95, 1, -1, True, True, signal_bar=2)])
        filtered = apply_entry_filter(day, bars, EntryFilter())
        reverse = engine.reverse_directions([filtered])[0]
        self.assertEqual([e.enter for e in reverse.events], [True, False, True])
        self.assertEqual([e.direction for e in reverse.events], [-1, -1, 1])
        self.assertEqual([e.exit for e in reverse.events], [True, True, True])
        self.assertEqual([e.signal_alf for e in reverse.events], [100., 100., 100.])

    def test_reverse_does_not_modify_input(self):
        """Не меняет исходные объекты; двойной разворот восстанавливает события."""
        days = [self.sample_day()]
        original_direction = days[0].events[0].direction
        flipped = engine.reverse_directions(days)
        self.assertEqual(days[0].events[0].direction, original_direction)
        self.assertEqual(engine.reverse_directions(flipped), days)
        self.assertIsNot(flipped[0], days[0])
        self.assertIsNot(flipped[0].events[0], days[0].events[0])
        self.assertEqual(engine.reverse_directions([]), [])

    def test_reversed_description_and_defaults(self):
        """Новый запуск имеет явные условия Short выше ALF и отдельную папку отчётов."""
        text = EntryFilter().describe(reverse=True)
        self.assertIn('Short при close > open', text)
        self.assertIn('Long при close < open', text)
        self.assertIn('Short только при close > ALF', text)
        self.assertIn('Long только при close < ALF', text)
        self.assertIn('отключён', EntryFilter('none').describe(reverse=True))
        normal = runner.arguments([])
        reverse = runner.arguments([], reverse=True, description='Обратная стратегия')
        self.assertEqual(normal.output_dir, Path('C:/data_quote/duration_backtests'))
        self.assertEqual(reverse.output_dir, Path('C:/data_quote/duration_backtests_reversed'))
        self.assertEqual(reverse.entry_filter, 'alf')
        self.assertEqual(reverse.symbols, ['RTS', 'MIX'])
        custom = runner.arguments(['--output-dir', 'custom', '--entry-filter', 'none'], reverse=True)
        self.assertEqual(custom.output_dir, Path('custom'))
        self.assertEqual(custom.entry_filter, 'none')


if __name__ == '__main__':
    unittest.main()
