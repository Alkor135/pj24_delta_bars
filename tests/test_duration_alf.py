"""Проверки фильтра ALF, прогрева истории и повторного использования кэша.

Запуск из корня проекта: python -m unittest -v tests.test_duration_alf
Все базы и тиковые архивы создаются во временной папке.
"""

from contextlib import closing
from hashlib import sha256
from pathlib import Path
import sqlite3
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd

from source import duration_data as data
from source.chart_data import load_bars
from source.duration_engine import Day, Event, parameter_grid, simulate_grid, simulate_one


class AlfTests(unittest.TestCase):
    """Проверяет согласованные условия входа на заранее известных значениях ALF."""

    def setUp(self):
        """Требует явных настроек и функции фильтрации готовых событий."""
        self.assertTrue(hasattr(data, 'EntryFilter'), 'Нужны настройки EntryFilter')
        self.assertTrue(hasattr(data, 'apply_entry_filter'), 'Нужен фильтр готовых событий')

    def test_direction_and_close_must_agree_with_alf(self):
        """Пропускает несовпадение направления, равенство ALF и doji."""
        directions = [1, -1, 1, -1, 1, -1, 0]
        closes = [110, 90, 90, 110, 100, 100, 110]
        bars = pd.DataFrame({'bar_index': range(7), 'close': closes, 'alf': [100.] * 7})
        day = Day('2022-01-03', [Event('10:00:01', i + 2, 105, 1, d, True, True,
                                     signal_bar=i) for i, d in enumerate(directions)])
        filtered = data.apply_entry_filter(day, bars, data.EntryFilter())
        self.assertEqual([e.enter for e in filtered.events], [True, True, False, False, False, False, False])
        self.assertTrue(all(e.exit for e in filtered.events))
        self.assertTrue(all(e.enter for e in day.events), 'Нельзя изменять исходные события кэша')
        self.assertEqual([e.signal_close for e in filtered.events], closes)
        self.assertEqual([e.signal_alf for e in filtered.events], [100.] * 7)

    def test_filter_does_not_enable_entry_outside_session(self):
        """Подходящая сторона ALF не отменяет запрет входа по времени."""
        day = Day('2022-01-03', [Event('18:35:00', 3, 120, 1, 1, False, True, signal_bar=0)])
        bars = pd.DataFrame({'bar_index': [0], 'close': [110.], 'alf': [100.]})
        filtered = data.apply_entry_filter(day, bars, data.EntryFilter())
        self.assertFalse(filtered.events[0].enter)
        self.assertTrue(filtered.events[0].exit)

    def test_exit_and_journal_use_signal_bar_values(self):
        """Фильтр не мешает выходу; журнал хранит close и ALF сигнального бара."""
        bars = pd.DataFrame({'bar_index': [0, 1], 'close': [110., 120.], 'alf': [100., 100.]})
        raw = Day('2022-01-03', [Event('10:00:02', 3, 80, 1, 1, True, True, signal_bar=0),
            Event('10:01:03', 5, 90, 61, -1, True, True, signal_bar=1),
            Event('18:40:00', 8, 95, force=True)])
        day = data.apply_entry_filter(raw, bars, data.EntryFilter())
        self.assertFalse(day.events[1].enter)
        result = simulate_one([day], 2, 60, 4)
        trade = result.trades[0]
        self.assertEqual((trade['entry_price'], trade['exit_price']), (80, 90))
        self.assertEqual((trade['entry_signal_close'], trade['entry_signal_alf']), (110, 100))
        self.assertEqual(trade['reason'], 'duration')
        grid = simulate_grid([day], parameter_grid([2], [60]))
        self.assertEqual(grid.gross[0, 0], trade['gross_pnl'])
        self.assertEqual(grid.count[0, 0], 1)

    def test_none_preserves_original_entry_flags(self):
        """Отключение ALF сохраняет прежние решения, включая вход против стороны ALF."""
        bars = pd.DataFrame({'bar_index': [0], 'close': [90.]})
        day = Day('2022-01-03', [Event('10:00:01', 3, 95, 1, 1, True, True, signal_bar=0)])
        filtered = data.apply_entry_filter(day, bars, data.EntryFilter('none'))
        self.assertTrue(filtered.events[0].enter)
        self.assertIsNone(filtered.events[0].signal_alf)

    def test_invalid_settings_and_missing_alf_rejected(self):
        """Отклоняет неверные настройки и отсутствующее конечное значение ALF."""
        for mode, alpha in [('other', .4), ('alf', 0), ('alf', 1.1), ('alf', float('nan'))]:
            with self.assertRaises(ValueError):
                data.EntryFilter(mode, alpha)
        day = Day('2022-01-03', [Event('10:00:01', 3, 95, 1, 1, True, True, signal_bar=0)])
        for frame in [pd.DataFrame({'bar_index': [0], 'close': [90.]}),
                      pd.DataFrame({'bar_index': [0], 'close': [90.], 'alf': [np.nan]})]:
            with self.assertRaises(ValueError):
                data.apply_entry_filter(day, frame, data.EntryFilter())


class AlfHistoryTests(unittest.TestCase):
    """Проверяет прогрев, причинность и безопасное применение ALF поверх общего кэша."""

    def setUp(self):
        """Создаёт три дня разных цен, архивы и SQLite с тем же набором полей."""
        self.assertTrue(hasattr(data, 'EntryFilter'), 'Нужны настройки EntryFilter')
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / 'bars.sqlite3'
        self.cache = self.root / 'cache'
        self.session = data.Session('10:00', '10:00:50', '10:01')
        frames, sources = [], []
        for day, offset in [('2022-01-03', 200), ('2022-01-04', 0), ('2022-01-05', 400)]:
            stamps = [f'{day} {t}' for t in ['10:00:00.001', '10:00:00.002',
                '10:00:01', '10:00:40', '10:00:41', '10:01:02']]
            prices = np.array([100, 110, 115, 90, 95, 105]) + offset
            ticks = pd.DataFrame({'datetime': stamps, 'last': prices, 'volume': 1})
            path = self.root / (day + '.zip')
            with ZipFile(path, 'w') as archive:
                archive.writestr('ticks.csv', ticks.to_csv(index=False))
            sources.append(dict(dataset_id='test', symbol='RTS', day=day, status='ready',
                                file_path=str(path), sha256=sha256(path.read_bytes()).hexdigest(), tick_count=6))
            frames.extend(dict(dataset_id='test', symbol='RTS', day=day, bar_index=i,
                start_row=a + 1, end_row=b + 1, start_time=stamps[a], end_time=stamps[b],
                open=float(prices[a]), close=float(prices[b]), high=float(max(prices[a:b + 1])),
                low=float(min(prices[a:b + 1])), volume=2, delta=1, threshold=1, tick_count=2,
                is_complete=int(i < 2), close_reason='threshold' if i < 2 else 'day_end')
                for i, (a, b) in enumerate([(0, 1), (2, 3), (4, 5)]))
        with closing(sqlite3.connect(self.db)) as db:
            pd.DataFrame(frames).to_sql('bars', db, index=False)
            pd.DataFrame(sources).to_sql('days', db, index=False)
            pd.DataFrame([dict(dataset_id='test', config_json='{}')]).to_sql('datasets', db, index=False)

    def prepare(self, mode='alf', alpha=.4, start='2022-01-03', end='2022-01-05', cache=None):
        """Подготавливает тестовую историю с выбранным фильтром и диапазоном."""
        return data.prepare_database(self.db, 'RTS', self.session, cache or self.cache,
            start, end, progress=None, entry_filter=data.EntryFilter(mode, alpha))

    def test_chart_parity_warmup_and_future_independence(self):
        """Дата начала и наличие будущих дней не меняют ALF уже завершённых баров."""
        full, _, _ = self.prepare()
        late, _, meta = self.prepare(start='2022-01-04')
        short, _, _ = self.prepare(end='2022-01-04')
        self.assertEqual(full[1:], late)
        self.assertEqual(full[:2], short)
        chart = load_bars(self.db, 'RTS', 'test', '2022-01-04', '2022-01-05').set_index(['day', 'bar_index'])
        for day in late:
            for event in day.events:
                if event.signal_bar >= 0:
                    self.assertEqual(event.signal_alf, chart.loc[(day.date, event.signal_bar), 'alf'])
        self.assertEqual(meta['entry_filter'], {'mode': 'alf', 'alpha': .4})
        self.assertEqual(meta['alf_warmup_bars'], 3)

    def test_cache_is_shared_but_rules_and_alpha_are_reapplied(self):
        """Один сырой кэш даёт корректные разные сигналы для режимов и значений α."""
        base, _, _ = self.prepare('none')
        paths = {p.name: sha256(p.read_bytes()).hexdigest() for p in self.cache.rglob('*.gz')}
        filtered, _, _ = self.prepare()
        changed, _, _ = self.prepare(alpha=1)
        self.assertNotEqual([e.enter for d in base for e in d.events], [e.enter for d in filtered for e in d.events])
        self.assertNotEqual([e.signal_alf for d in filtered for e in d.events],
                            [e.signal_alf for d in changed for e in d.events])
        self.assertEqual(paths, {p.name: sha256(p.read_bytes()).hexdigest() for p in self.cache.rglob('*.gz')})
        fresh, _, _ = self.prepare(cache=self.root / 'fresh')
        self.assertEqual(filtered, fresh)
        for day in base:
            self.assertTrue(all(e.signal_alf is None for e in day.events))

    def test_changed_warmup_invalidates_decision_without_changing_raw_cache(self):
        """Изменение старого закрытия влияет на ALF нового дня даже при готовом кэше."""
        before, _, _ = self.prepare(start='2022-01-04')
        with closing(sqlite3.connect(self.db)) as db:
            db.execute("UPDATE bars SET close=close+1000 WHERE day='2022-01-03'")
            db.commit()
        after, _, _ = self.prepare(start='2022-01-04')
        fresh, _, _ = self.prepare(start='2022-01-04', cache=self.root / 'fresh')
        self.assertNotEqual([e.signal_alf for e in before[0].events], [e.signal_alf for e in after[0].events])
        self.assertEqual(after, fresh)


if __name__ == '__main__':
    unittest.main()
