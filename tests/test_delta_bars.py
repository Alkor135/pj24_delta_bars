"""Проверки адаптивных дельта-баров и безопасной дозаписи SQLite.

Запуск: python -m unittest -v tests.test_delta_bars
Тесты создают собственные небольшие ZIP во временной папке.
"""

from array import array
from contextlib import closing
from datetime import date, timedelta
import importlib.util
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile


def make_zip(root, symbol, day, rows):
    """Создаёт дневной ZIP из переданных строк тиков для интеграционной проверки."""
    folder = Path(root) / f"data_finam_{symbol}_tick_zip"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{day:%Y%m%d}.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(f"{day:%Y%m%d}.csv", "datetime,last,volume\n" + "\n".join(rows))
    return path


def sample_rows(day, volume=2):
    """Возвращает тики с двумя непустыми пятиминутными интервалами."""
    return [f"{day} 10:00:00.000000001,100,{volume}",
            f"{day} 10:00:00.000000002,101,{volume}",
            f"{day} 10:00:01,102,{volume}",
            f"{day} 10:05:00,101,{volume}",
            f"{day} 10:05:01,100,{volume}"]


def reference_ends(deltas, threshold):
    """Вычисляет эталонные границы прямым накоплением дельты каждой сделки."""
    ends, value = [], 0
    for i, delta in enumerate(deltas):
        value += delta
        if abs(value) >= threshold:
            ends.append(i)
            value = 0
    if deltas and (not ends or ends[-1] != len(deltas) - 1):
        ends.append(len(deltas) - 1)
    return ends


class CoreTests(unittest.TestCase):
    """Проверяет расчёт по независимому эталону и корректность входных тиков."""

    def test_modules_exist(self):
        """Требует наличия исполняемого конвертера и вычислительного ядра."""
        for name in ("source.delta_core", "source.delta_store", "tick_to_delta_bars"):
            self.assertIsNotNone(importlib.util.find_spec(name), name)

    def test_crossings_match_reference(self):
        """Сравнивает быстрый поиск с прямым расчётом, включая превышения и развороты."""
        from source.delta_core import DeltaPath
        rng = random.Random(73)
        cases = [[], [0] * 10, [0, 5, 5, -7, -7, 0], [0, 100, -1, -100, 1]]
        cases += [[rng.choice([-9, -3, -1, 0, 1, 4, 13]) for _ in range(300)] for _ in range(50)]
        for deltas in cases:
            values, total = array("q"), 0
            for value in deltas:
                total += value
                values.append(total)
            path = DeltaPath(values)
            for threshold in (1, 2, 5, 10, 30, 1000):
                expected = reference_ends(deltas, threshold)
                self.assertEqual(list(path.ends(threshold)), expected)
                self.assertEqual(path.count(threshold), len(expected))

    def test_tick_rule_neutral_and_duplicate_times(self):
        """Проверяет одинаковые цены, нейтральный начальный объём и сохранение дубликатов."""
        from source.delta_core import read_day, build_bars
        day = date(2024, 1, 2)
        with tempfile.TemporaryDirectory() as root:
            path = make_zip(root, "RTS", day, [f"{day} 10:00:00,100,2",
                f"{day} 10:00:00,100,3", f"{day} 10:00:01,101,4",
                f"{day} 10:00:02,101,5", f"{day} 10:05:00,100,7"])
            ticks = read_day(path, day, 5)
            self.assertEqual(list(ticks.path.values), [0, 0, 4, 9, 2])
            self.assertEqual(ticks.active_intervals, 2)
            bars = build_bars(ticks, 6)
            self.assertEqual([bar["delta"] for bar in bars], [9, -7])
            self.assertEqual([bar["volume"] for bar in bars], [14, 7])
            self.assertEqual(bars[0]["neutral_volume"], 5)
            self.assertEqual(bars[0]["tick_count"], 4)
            self.assertEqual(bars[0]["start_row"], 1)
            self.assertEqual(bars[1]["end_row"], 5)

    def test_partial_and_whole_trade(self):
        """Не дробит крупную сделку и сохраняет остаток дня отдельным баром."""
        from source.delta_core import read_day, build_bars
        day = date(2024, 1, 2)
        with tempfile.TemporaryDirectory() as root:
            path = make_zip(root, "RTS", day, [f"{day} 10:00:00,100,1",
                f"{day} 10:00:01,101,20", f"{day} 10:00:02,102,1"])
            bars = build_bars(read_day(path, day, 5), 5)
            self.assertEqual([b["delta"] for b in bars], [20, 1])
            self.assertEqual([b["close_reason"] for b in bars], ["threshold", "day_end"])
            self.assertEqual([b["is_complete"] for b in bars], [1, 0])

    def test_invalid_input_rejected(self):
        """Отклоняет плохую цену, объём, чужую дату и нарушенный порядок времени."""
        from source.delta_core import read_day
        day = date(2024, 1, 2)
        cases = [[f"{day} 10:00:00,nan,1"], [f"{day} 10:00:00,100,0"],
                 [f"{day} 10:00:00,100,1.5"], ["2024-01-03 10:00:00,100,1"],
                 [f"{day} 10:00:01,100,1", f"{day} 10:00:00,100,1"]]
        with tempfile.TemporaryDirectory() as root:
            for rows in cases:
                path = make_zip(root, "RTS", day, rows)
                with self.assertRaises(ValueError):
                    read_day(path, day, 5)

    def test_calibration_counts_real_bars(self):
        """Проверяет целевой счётчик и результат пробного построения при выбранном пороге."""
        from source.delta_core import read_day, calibrate
        with tempfile.TemporaryDirectory() as root:
            days = []
            for n in range(20):
                day = date(2024, 1, 1) + timedelta(days=n)
                ticks = read_day(make_zip(root, "RTS", day, sample_rows(day)), day, 5)
                days.append(ticks.history())
            result = calibrate(days)
            self.assertEqual(result.target_count, 40)
            self.assertEqual(result.actual_count, sum(d.path.count(result.threshold) for d in days))
            self.assertEqual(result.actual_count, 40)


class PipelineTests(unittest.TestCase):
    """Проверяет реальные ZIP и SQLite при повторных запусках конвертера."""

    def setUp(self):
        """Подготавливает независимую историю с выходными и отдельную базу."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "bars.sqlite3"
        self.first = date(2024, 1, 1)
        for offset in range(35):
            day = self.first + timedelta(days=offset)
            make_zip(self.root, "RTS", day, sample_rows(day))

    def run_converter(self, end, **kwargs):
        """Запускает конвертацию с фиксированными настройками тестового набора."""
        from tick_to_delta_bars import convert
        return convert(self.root, self.db, ["RTS"], self.first, end,
                       lookback=20, minutes=5, log=lambda message: None, **kwargs)

    def rows(self, sql):
        """Читает результат SQL и сразу закрывает соединение."""
        with closing(sqlite3.connect(self.db)) as connection:
            return connection.execute(sql).fetchall()

    def test_full_window_weekends_and_no_lookahead(self):
        """Использует строго прошлые 20 будних дней и пропускает выходные при обучении."""
        self.run_converter(date(2024, 1, 30))
        ready = self.rows("SELECT day, window_start, window_end, target_count, calibration_count FROM days WHERE status='ready' ORDER BY day")
        self.assertEqual(ready[0], ("2024-01-27", "2024-01-01", "2024-01-26", 40, 40))
        self.assertEqual(ready[2][0:3], ("2024-01-29", "2024-01-01", "2024-01-26"))
        self.assertEqual(ready[3][0:3], ("2024-01-30", "2024-01-02", "2024-01-29"))
        before = self.rows("SELECT day, threshold FROM days WHERE status='ready'")
        day = date(2024, 1, 30)
        make_zip(self.root, "RTS", day, sample_rows(day, volume=100000))
        self.run_converter(day)
        self.assertEqual(before, self.rows("SELECT day, threshold FROM days WHERE status='ready'"))

    def test_append_matches_one_pass_and_is_idempotent(self):
        """Дозапись даёт те же бары, что единый запуск, без повторных строк."""
        self.run_converter(date(2024, 1, 29))
        self.run_converter(date(2024, 2, 4))
        first = self.rows("SELECT * FROM bars ORDER BY symbol, day, bar_index")
        self.run_converter(date(2024, 2, 4))
        self.assertEqual(first, self.rows("SELECT * FROM bars ORDER BY symbol, day, bar_index"))
        old_db = self.db
        self.db = self.root / "fresh.sqlite3"
        self.run_converter(date(2024, 2, 4))
        self.assertEqual(first, self.rows("SELECT * FROM bars ORDER BY symbol, day, bar_index"))
        self.db = old_db

    def test_changed_and_inserted_history_rebuilds(self):
        """Пересчитывает затронутые дни при изменении и добавлении старых архивов."""
        removed = self.root / "data_finam_RTS_tick_zip" / "20240105.zip"
        removed.unlink()
        self.run_converter(date(2024, 2, 4))
        make_zip(self.root, "RTS", date(2024, 1, 5), sample_rows(date(2024, 1, 5), 20))
        make_zip(self.root, "RTS", date(2024, 1, 29), sample_rows(date(2024, 1, 29), 30))
        self.run_converter(date(2024, 2, 4))
        rebuilt = self.rows("SELECT * FROM bars ORDER BY day, bar_index")
        self.db = self.root / "fresh.sqlite3"
        self.run_converter(date(2024, 2, 4))
        self.assertEqual(rebuilt, self.rows("SELECT * FROM bars ORDER BY day, bar_index"))

    def test_volumes_ticks_and_daily_reset(self):
        """Сохраняет все тики и объёмы; начальный тик каждого дня остаётся нейтральным."""
        self.run_converter(date(2024, 2, 4))
        for row in self.rows("SELECT day, sum(tick_count), sum(volume), sum(up_volume+down_volume+neutral_volume),sum(neutral_volume) FROM bars GROUP BY day"):
            self.assertEqual(row[1:], (5, 10, 10, 2))

    def test_failed_day_not_recorded(self):
        """Повреждённый день не оставляет баров или отметки об успешной обработке."""
        day = date(2024, 1, 30)
        make_zip(self.root, "RTS", day, [f"{day} 10:00:00,100,-1"])
        with self.assertRaises(ValueError):
            self.run_converter(day)
        self.assertEqual(self.rows("SELECT count(*) FROM days WHERE day='2024-01-30'"), [(0,)])
        self.assertEqual(self.rows("SELECT count(*) FROM bars WHERE day='2024-01-30'"), [(0,)])
        self.assertEqual(self.rows("PRAGMA integrity_check"), [("ok",)])

    def test_sql_failure_rolls_back_day(self):
        """Имитирует ошибку вставки и проверяет атомарность дневной транзакции."""
        self.run_converter(date(2024, 1, 29))
        with closing(sqlite3.connect(self.db)) as connection:
            connection.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON bars WHEN NEW.day='2024-01-30' AND NEW.bar_index=1 BEGIN SELECT RAISE(ABORT,'test failure'); END")
            connection.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.run_converter(date(2024, 1, 30))
        self.assertEqual(self.rows("SELECT count(*) FROM days WHERE day='2024-01-30'"), [(0,)])
        self.assertEqual(self.rows("SELECT count(*) FROM bars WHERE day='2024-01-30'"), [(0,)])

    def test_deleted_history_rebuilds(self):
        """После удаления старого источника получает тот же результат, что чистый расчёт."""
        self.run_converter(date(2024, 2, 4))
        (self.root / "data_finam_RTS_tick_zip" / "20240105.zip").unlink()
        self.run_converter(date(2024, 2, 4))
        rebuilt = self.rows("SELECT * FROM bars ORDER BY day, bar_index")
        self.db = self.root / "fresh.sqlite3"
        self.run_converter(date(2024, 2, 4))
        self.assertEqual(rebuilt, self.rows("SELECT * FROM bars ORDER BY day, bar_index"))

    def test_empty_day_not_in_training(self):
        """Не считает пустой архив одним из двадцати обучающих дней."""
        make_zip(self.root, "RTS", date(2024, 1, 5), [])
        self.run_converter(date(2024, 2, 4))
        self.assertEqual(self.rows("SELECT min(day) FROM days WHERE status='ready'"), [("2024-01-30",)])

    def test_prestart_history_and_parameter_separation(self):
        """Прогревается до начала вывода и разделяет разные параметры в одной базе."""
        from tick_to_delta_bars import convert
        output_start = date(2024, 1, 29)
        for minutes in (5, 10):
            convert(self.root, self.db, ["RTS"], output_start, date(2024, 1, 30),
                    lookback=20, minutes=minutes, log=lambda message: None)
        self.assertEqual(self.rows("SELECT count(*) FROM datasets"), [(2,)])
        self.assertEqual(self.rows("SELECT min(day) FROM bars"), [("2024-01-29",)])
        self.assertEqual(self.rows("SELECT DISTINCT window_days FROM days WHERE status='ready'"), [(20,)])

    def test_concurrent_writer_rejected(self):
        """Запрещает второму процессу менять базу, пока действует системная блокировка."""
        from tick_to_delta_bars import RunLock
        with RunLock(self.db):
            with self.assertRaises(ValueError):
                self.run_converter(date(2024, 1, 30))


if __name__ == "__main__":
    unittest.main()
