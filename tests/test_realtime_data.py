"""Проверяет дневной порог QUIK по прошлым ZIP и неизменность истории SQLite.

Запуск: .\\.venv\\Scripts\\python.exe -m unittest -v tests.test_realtime_data
Тесты создают небольшие архивы; текущий день специально отличается объёмом.
"""

from datetime import date, timedelta
import importlib.util
from pathlib import Path
import tempfile
import unittest

from tests.test_delta_bars import make_zip, sample_rows
from tick_to_delta_bars import convert


class RealtimeDataTests(unittest.TestCase):
    """Проверяет выбор окна, защиту от устаревшего порога и прогрев графика."""

    def setUp(self):
        """Создаёт собственную историю с коротким обучающим окном в два дня."""
        self.assertIsNotNone(importlib.util.find_spec("source.realtime_data"), "Источник реал-тайм ещё не реализован")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "RTS_delta_bars.sqlite3"
        first = date(2026, 9, 28)
        for offset in range(5):
            day = first + timedelta(days=offset)
            make_zip(self.root, "RTS", day, sample_rows(day, volume=99 if offset == 4 else 2))
        convert(self.root, self.db, ["RTS"], first, date(2026, 10, 2), lookback=2, log=lambda message: None)

    def test_threshold_uses_only_previous_days_without_writing_database(self):
        """Порог второго октября использует 30 сентября и 1 октября, база неизменна."""
        from source.realtime_data import prepare_threshold
        before = self.db.read_bytes()
        entry = prepare_threshold(self.db, "RTS", "2026-10-02")
        self.assertEqual(entry["window_start"], "2026-09-30")
        self.assertEqual(entry["window_end"], "2026-10-01")
        self.assertEqual(entry["window_days"], 2)
        self.assertGreater(entry["threshold"], 0)
        self.assertEqual(before, self.db.read_bytes())

    def test_cache_is_validated_and_missing_cache_recalculates(self):
        """Подготовленный порог проверяется по дате, набору и окну исходных данных."""
        from source.realtime_data import prepare_threshold, resolve_threshold, write_thresholds
        cache = self.root / "thresholds.json"
        entry = prepare_threshold(self.db, "RTS", "2026-10-02")
        write_thresholds(cache, [entry])
        resolved = resolve_threshold(self.db, "RTS", "2026-10-02", threshold_file=cache)
        self.assertEqual(resolved["threshold"], entry["threshold"])
        self.assertEqual(resolved["source"], "ночная подготовка")
        next_day = resolve_threshold(self.db, "RTS", "2026-10-03", threshold_file=cache)
        self.assertEqual(next_day["source"], "расчёт по прошлым тикам")
        self.assertEqual(next_day["window_end"], "2026-10-02")

    def test_manual_threshold_and_incompatible_cache(self):
        """Ручной порог обозначается явно; неверный подготовленный порог не применяется."""
        from source.realtime_data import prepare_threshold, resolve_threshold, write_thresholds
        cache = self.root / "thresholds.json"
        entry = prepare_threshold(self.db, "RTS", "2026-10-02")
        write_thresholds(cache, [dict(entry, fingerprint="неверный")])
        resolved = resolve_threshold(self.db, "RTS", "2026-10-02", threshold_file=cache)
        self.assertEqual(resolved["source"], "расчёт по прошлым тикам")
        self.assertEqual(resolve_threshold(self.db, "RTS", "2026-10-02", manual=7)["threshold"], 7)
        with self.assertRaises(ValueError):
            resolve_threshold(self.db, "RTS", "2026-10-02", manual=0)

    def test_history_excludes_today_and_keeps_warmup(self):
        """Загрузчик исключает сегодняшний день, сохраняя предысторию для индикаторов."""
        from source.realtime_data import load_history, prepare_threshold
        entry = prepare_threshold(self.db, "RTS", "2026-10-02")
        history = load_history(self.db, "RTS", entry["dataset_id"], "2026-10-02")
        self.assertEqual(history.day.max(), "2026-10-01")
        self.assertEqual(history.day.min(), "2026-09-30")
        self.assertNotIn("2026-10-02", history.day.tolist())


if __name__ == "__main__":
    unittest.main()
