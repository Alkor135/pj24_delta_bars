"""Проверяет чтение баров и индикаторы будущего просмотрщика.

Запуск: python -m unittest -v tests.test_chart_data
Проверки работают на собственных временных базах и не меняют пользовательские данные.
"""

from contextlib import closing
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd


def frame(volumes, opens, closes, days=None, complete=None):
    """Создаёт небольшой набор свечей с управляемыми условиями объёмного сигнала."""
    count = len(volumes)
    return pd.DataFrame({"volume": volumes, "open": opens, "close": closes,
                         "high": np.maximum(opens, closes) + 1, "low": np.minimum(opens, closes) - 1,
                         "day": days or ["2026-01-05"] * count,
                         "is_complete": complete or [1] * count})


class IndicatorTests(unittest.TestCase):
    """Сверяет фильтр и сигналы с известными последовательностями свечей."""

    def test_module_available(self):
        """Требует наличия вычислительного модуля просмотрщика."""
        self.assertIsNotNone(importlib.util.find_spec("source.chart_data"))

    def test_laguerre_constant_and_invalid_alpha(self):
        """Фильтр постоянной цены не начинается с искусственного нуля."""
        from source.chart_data import laguerre
        np.testing.assert_allclose(laguerre(np.full(20, 100.0), 0.4), 100)
        self.assertEqual(len(laguerre([], 0.4)), 0)
        for alpha in (0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                laguerre([1, 2], alpha)

    def test_laguerre_matches_reference_and_is_causal(self):
        """Сравнивает с рекурсией образца, инициализированной первой ценой."""
        from source.chart_data import laguerre
        prices = np.array([100, 101, 98, 110, 112, 90, 93, 95], dtype=float)
        alpha, state, expected = 0.4, [prices[0]] * 4, []
        for price in prices:
            a, b, c, d = state
            a1 = alpha * price + (1 - alpha) * a
            b1 = -(1 - alpha) * a1 + a + (1 - alpha) * b
            c1 = -(1 - alpha) * b1 + b + (1 - alpha) * c
            d1 = -(1 - alpha) * c1 + c + (1 - alpha) * d
            state = [a1, b1, c1, d1]
            expected.append((a1 + 2 * b1 + 2 * c1 + d1) / 6)
        np.testing.assert_allclose(laguerre(prices, alpha), expected)
        np.testing.assert_allclose(laguerre(prices[:5], alpha), expected[:5])

    def test_four_volume_patterns(self):
        """Проверяет все четыре комбинации разворота и направления объёма."""
        from source.chart_data import volume_stops
        patterns = [("long_rising", [1, 2, 3], [3, 2, 1], [2, 1, 2]),
                    ("short_rising", [1, 2, 3], [1, 2, 3], [2, 3, 2]),
                    ("long_falling", [3, 2, 1], [3, 2, 1], [2, 1, 2]),
                    ("short_falling", [3, 2, 1], [1, 2, 3], [2, 3, 2])]
        for name, volumes, opens, closes in patterns:
            signals = volume_stops(frame(volumes, opens, closes))
            self.assertTrue(signals[name][-1], name)
            self.assertEqual(sum(np.count_nonzero(values) for values in signals.values()), 1)

    def test_no_signal_across_days_or_partial_bars(self):
        """Исключает тройки через полночь и с неполным баром, сохраняя сами свечи."""
        from source.chart_data import volume_stops
        for kwargs in ({"days": ["2026-01-05", "2026-01-05", "2026-01-06"]},
                       {"complete": [1, 0, 1]}, {"complete": [1, 1, 0]}):
            signals = volume_stops(frame([1, 2, 3], [3, 2, 1], [2, 1, 2], **kwargs))
            self.assertFalse(any(np.any(values) for values in signals.values()))


class DatabaseTests(unittest.TestCase):
    """Проверяет исключительно чтение SQLite, порядок баров и прогрев фильтра."""

    def setUp(self):
        """Готовит базу с двумя наборами и одинаковыми временными метками."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "sample.sqlite3"
        with closing(sqlite3.connect(self.db)) as connection:
            connection.executescript("""
                PRAGMA user_version=1;
                CREATE TABLE datasets(dataset_id TEXT PRIMARY KEY,config_json TEXT,created_at TEXT);
                CREATE TABLE bars(dataset_id TEXT,symbol TEXT,day TEXT,bar_index INTEGER,
                    start_time TEXT,end_time TEXT,open REAL,high REAL,low REAL,close REAL,
                    volume INTEGER,delta INTEGER,threshold INTEGER,tick_count INTEGER,is_complete INTEGER,
                    close_reason TEXT,PRIMARY KEY(dataset_id,symbol,day,bar_index));
                INSERT INTO datasets VALUES('a','{"lookback_weekdays_with_ticks":20,"target_minutes":5}','2026-01-01');
                INSERT INTO datasets VALUES('b','{"lookback_weekdays_with_ticks":10,"target_minutes":5}','2026-01-02');
            """)
            rows = []
            for dataset in ("a", "b"):
                for day in ("2026-01-05", "2026-01-06"):
                    for i in range(4):
                        value = 100 + i + (10 if day.endswith("06") else 0)
                        rows.append((dataset, "RTS", day, i, f"{day} 10:00:00.000000001",
                                     f"{day} 10:00:01.000000009", value, value+2, value-1,
                                     value+1, i+1, 2, 2, 5, 1, "threshold"))
            connection.executemany("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            connection.commit()

    def test_dataset_selection_and_duplicate_timestamps(self):
        """Сохраняет одинаковые времена отдельными барами и не смешивает наборы."""
        from source.chart_data import inspect_database, load_bars
        self.assertEqual(len(inspect_database(self.db, "RTS")), 2)
        data = load_bars(self.db, "RTS", "a", "2026-01-05", "2026-01-06", 0.4)
        self.assertEqual(len(data), 8)
        self.assertEqual(data.bar_index.tolist(), [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(data.x.tolist(), list(range(8)))
        self.assertEqual(data.start_time.iloc[0].nanosecond, 1)
        self.assertAlmostEqual(data.duration_seconds.iloc[0], 1.000000008)

    def test_filter_same_for_overlapping_ranges(self):
        """Начало видимого диапазона не меняет прогретый ALF на общих барах."""
        from source.chart_data import load_bars
        all_data = load_bars(self.db, "RTS", "a", "2026-01-05", "2026-01-06", 0.4)
        later = load_bars(self.db, "RTS", "a", "2026-01-06", "2026-01-06", 0.4)
        np.testing.assert_array_equal(later.alf, all_data.alf.iloc[4:])

    def test_missing_db_does_not_create_file_and_reads_are_readonly(self):
        """Не создаёт отсутствующую базу и запрещает запись через соединение просмотрщика."""
        from source.chart_data import readonly_connection, load_bars
        missing = self.db.with_name("missing.sqlite3")
        with self.assertRaises(ValueError):
            with readonly_connection(missing):
                pass
        self.assertFalse(missing.exists())
        before = self.db.read_bytes()
        load_bars(self.db, "RTS", "a", "2026-01-05", "2026-01-06", 0.4)
        self.assertEqual(before, self.db.read_bytes())
        with readonly_connection(self.db) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM bars")

    def test_empty_range_and_bad_dates(self):
        """Возвращает пустой диапазон без сбоя и сообщает о перепутанных датах."""
        from source.chart_data import load_bars
        self.assertTrue(load_bars(self.db, "RTS", "a", "2026-02-01", "2026-02-02", 0.4).empty)
        with self.assertRaises(ValueError):
            load_bars(self.db, "RTS", "a", "2026-02-02", "2026-02-01", 0.4)


if __name__ == "__main__":
    unittest.main()
