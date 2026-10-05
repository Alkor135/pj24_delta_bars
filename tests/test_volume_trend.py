"""Проверяет точный объём, окна, события индикаторов и статистику исследования.

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m unittest -v tests.test_volume_trend

Примеры содержат известные вручную ответы; пользовательские данные не изменяются.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from types import SimpleNamespace

import numpy as np
import pandas as pd


class VolumeTrendTests(unittest.TestCase):
    """Проверяет ошибки, способные изменить классификацию дня или число событий."""

    def test_modules_exist(self):
        """Требует наличие модулей согласованного исследования."""
        self.assertIsNotNone(importlib.util.find_spec("research.volume_trend_data"))

    def test_window_includes_current_day_and_both_bounds(self):
        """Текущий день сужает окно; одинаковые сделки на границах учитываются все."""
        from research.volume_trend_data import common_window, window_volume
        self.assertEqual(common_window([(1, 10), (3, 9), (4, 8)]), (4, 8))
        times = np.array([1, 4, 4, 6, 8, 8, 9], dtype=np.int64)
        prefix = np.r_[0, np.cumsum([1, 2, 3, 4, 5, 6, 7])]
        self.assertEqual(window_volume(times, prefix, 4, 8), 20)
        self.assertIsNone(common_window([(2, 2), (1, 3)]))

    def test_weekends_and_invalid_history(self):
        """Выходные не входят в базу; повреждённый день не заменяется более старым."""
        from research.volume_trend_data import history_indices
        days = pd.DataFrame({"day": ["2026-09-30", "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"], "valid": [True, True, False, True, True, True]})
        self.assertEqual(history_indices(days, 5, 2), [1, 2])
        self.assertFalse(days.iloc[history_indices(days, 5, 2)].valid.all())

    def test_read_ticks_and_reject_wrong_day(self):
        """Проверяет сохранение повторяющихся отметок и отклонение чужой даты."""
        from research.volume_trend_data import read_ticks
        root = Path(__file__).resolve().parents[1] / ".cache"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(root))
            path = Path(directory) / "20261002.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("20261002.csv", "datetime,last,volume\n2026-10-02 10:00:00.000,100,2\n2026-10-02 10:00:00.000,100,3\n2026-10-02 10:00:01.000,101,4\n")
            times, prices, volumes = read_ticks(path, "2026-10-02")
            np.testing.assert_array_equal(volumes, [2, 3, 4])
            self.assertEqual(times[0], times[1])
            with self.assertRaises(ValueError):
                read_ticks(path, "2026-10-01")

    def test_sma_equalities_and_excluded_bars(self):
        """Равенства не создают лишних пересечений, исключённый бар разрывает связь."""
        from research.volume_trend_metrics import count_switches
        self.assertEqual(count_switches([1, 0, 0, -1, 0, -1, 1], [True]*7), (2, 6))
        self.assertEqual(count_switches([1, 0, 1], [True]*3), (0, 2))
        count, opportunities = count_switches([1, -1, -1], [True, False, True])
        self.assertTrue(np.isnan(count))
        self.assertEqual(opportunities, 0)
        self.assertEqual(count_switches([1, -1, 1], [True]*3), (2, 2))

    def test_clearing_gap_remains_in_time_grid(self):
        """Пауза внутри дня заполняется предыдущей ценой и нулевым объёмом."""
        from research.volume_trend_data import time_bars
        bars = time_bars(np.array([600, 1200])*10**9, np.array([100., 105.]), np.array([2, 3]), 5)
        np.testing.assert_array_equal(bars.close, [100, 100, 105])
        np.testing.assert_array_equal(bars.volume, [2, 0, 3])

    def test_supertrend_matches_existing_formula_and_is_causal(self):
        """Сверяет индикатор проекта и неизменность прошлых результатов при добавлении будущего."""
        from chart_delta_bars_supertrend import calculate_supertrend
        from research.volume_trend_metrics import indicators
        data = pd.DataFrame({"high": [11, 12, 13, 16, 17, 13, 10, 15], "low": [9, 10, 11, 14, 15, 11, 8, 13], "close": [10, 11, 12, 15, 16, 12, 9, 14]})
        result = indicators(data, atr_period=3, multiplier=1, warmup=0)
        np.testing.assert_array_equal(result.st_sign, calculate_supertrend(data, 3, 1).supertrend_direction)
        prefix = indicators(data.iloc[:5], atr_period=3, multiplier=1, warmup=0)
        np.testing.assert_array_equal(prefix.st_sign, result.st_sign.iloc[:5])

    def test_group_effect_and_holm(self):
        """Проверяет знак эффекта, строгую границу R>1 и монотонную поправку Holm."""
        from research.volume_trend_metrics import group_effect, holm
        effect = group_effect(np.array([1, 3, 4, 6]), np.array([1.1, 1.2, 1., .5]))
        self.assertEqual(effect["difference"], -3)
        self.assertEqual(effect["high_mean"], 2)
        self.assertEqual(effect["low_mean"], 5)
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])

    def test_holm_keeps_missing_members_of_the_planned_family(self):
        """Неоценённые сравнения не уменьшают заранее заданную семью из восьми тестов."""
        from research.volume_trend_metrics import holm
        values = holm([.01]+[np.nan]*7)
        self.assertEqual(values[0], .08)
        self.assertTrue(np.isnan(values[1:]).all())

    def test_bootstrap_does_not_claim_precision_without_resampleable_blocks(self):
        """Два сегмента длиной блока не дают независимой оценки неопределённости."""
        from research.volume_trend_metrics import bootstrap_effect
        values = np.r_[np.zeros(20), np.ones(20)*10]
        ratios = np.r_[np.ones(20)*2, np.ones(20)*.5]
        result = bootstrap_effect(values, ratios, repetitions=200, block=20, segments=np.repeat([0, 1], 20))
        self.assertTrue(np.isnan(result["p"]))
        self.assertTrue(np.isnan(result["ci_low"]))

    def test_damaged_zip_has_a_clear_data_error(self):
        """Повреждённый ZIP сообщает ошибку данных, пригодную для дневного аудита."""
        from research.volume_trend_data import read_ticks
        root = Path(__file__).resolve().parents[1] / ".cache"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(root))
            path = Path(directory) / "20261002.zip"
            path.write_bytes(b"damaged archive")
            with self.assertRaises(ValueError):
                read_ticks(path, "2026-10-02")

    def test_missing_primary_statistics_are_not_a_negative_hypothesis_result(self):
        """Отсутствие пригодных p сообщает нехватку данных, а не отсутствие связи."""
        from research import volume_trend_report
        self.assertTrue(hasattr(volume_trend_report, "primary_conclusion"))
        data = pd.DataFrame({"difference": [1.]*8, "p_holm": [np.nan]*8})
        self.assertIn("недостаточно", volume_trend_report.primary_conclusion(data))

    def test_block_bootstrap_reproducible_and_degenerate_groups(self):
        """Повторы воспроизводимы; выборка с одной группой не даёт ложного эффекта."""
        from research.volume_trend_metrics import bootstrap_effect
        y = np.tile([1., 2., 8., 9.], 30)
        ratio = np.tile([2., 2., .5, .5], 30)
        first = bootstrap_effect(y, ratio, repetitions=200, block=20, seed=123)
        self.assertEqual(first, bootstrap_effect(y, ratio, repetitions=200, block=20, seed=123))
        self.assertLess(first["ci_high"], 0)
        self.assertTrue(np.isnan(bootstrap_effect(y, np.ones(120), repetitions=20)["difference"]))

    def test_volume_features_are_causal_and_use_exactly_twenty_dates(self):
        """Будущий объём не меняет прошлые строки, текущий день не входит в базу."""
        from research.volume_trend import volume_features
        root = Path(__file__).resolve().parents[1] / ".cache"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            output = Path(directory).resolve()
            self.assertTrue(output.is_relative_to(root))
            cache = output / "cache" / "RTS"
            cache.mkdir(parents=True)
            dates = pd.bdate_range("2026-08-03", periods=25)
            times = np.array([36000, 36100, 40000], dtype=np.int64)*10**9
            rows = []
            for index, day in enumerate(dates):
                volumes = np.array([1, 1, 1]) if index!=20 else np.array([2, 2, 2])
                np.savez(cache/f"{day.strftime('%Y%m%d')}.npz", times=times, volumes=volumes)
                rows.append({"day": day.date().isoformat(), "weekday": day.weekday(), "valid": True, "volume": int(volumes.sum()), "first_ns": int(times[0]), "last_ns": int(times[-1])})
            audit = pd.DataFrame(rows)
            config = {"start": rows[0]["day"], "fixed_start_ns": int(times[0]), "fixed_end_ns": int(times[-1])}
            args = SimpleNamespace(output=output)
            before = volume_features(args, "RTS", audit, config)
            self.assertEqual(before.iloc[20].r_mean, 2)
            self.assertEqual(before.iloc[20].volume_mean, 3)
            import json
            self.assertEqual(len(json.loads(before.iloc[20].history_days)), 20)
            np.savez(cache/f"{dates[-1].strftime('%Y%m%d')}.npz", times=times, volumes=np.array([100, 100, 100]))
            audit.loc[24, "volume"] = 300
            after = volume_features(args, "RTS", audit, config)
            pd.testing.assert_frame_equal(before.iloc[:24], after.iloc[:24])

    def test_partial_and_boundary_bars_do_not_create_events_across_exclusion(self):
        """Граничные бары не дробятся, исключённый остаток не создаёт переход."""
        from research.volume_trend_metrics import daily_metrics
        data = pd.DataFrame({"start_ns": [1, 4, 7], "end_ns": [3, 6, 9], "warmed": [True]*3, "st_sign": [1, -1, 1],
                             "sma_sign": [1, -1, 1], "is_complete": [1, 1, 0], "close": [100, 99, 101], "high": [100, 99, 101],
                             "low": [100, 99, 101], "volume": [2, 3, 4]})
        result = daily_metrics(data, 2, 9)
        self.assertEqual(result["st"], 1)
        self.assertEqual(result["included_volume"], 7)
        self.assertEqual(result["excluded_bars"], 1)
        self.assertTrue(np.isnan(daily_metrics(data, 2, 9, drop_partial=True)["st"]))


if __name__ == "__main__":
    unittest.main()
