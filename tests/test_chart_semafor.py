"""Проверяет «Семафор», острова SMA и отдельное окно графика дельта-баров.

Запуск из корня: python -m unittest -v tests.test_chart_semafor
Числовые примеры рассчитаны вручную; SQLite создаётся во временной папке.
Qt работает offscreen. Проверяются перенос экстремума, прогрев без будущих
баров, отсутствие прежних индикаторов и изменение настроек через интерфейс.
"""

import importlib.util
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtTest, QtWidgets

from tests import test_chart_data


def turning_bars():
    """Возвращает десять OHLC-баров с двумя вершинами и одной впадиной."""
    return pd.DataFrame({"high": [11, 13, 12, 15, 14, 12, 10, 13, 16, 15],
                         "low": [9, 11, 10, 13, 12, 10, 8, 11, 14, 13],
                         "close": [10, 12, 11, 14, 13, 11, 9, 12, 15, 14]})


class SemaforTests(unittest.TestCase):
    """Сверяет экстремумы ZigZag, перерисовку и формулы цветных островов."""

    def test_script_available(self):
        """Требует самостоятельную точку входа с запрошенным именем."""
        self.assertIsNotNone(importlib.util.find_spec("chart_delta_bar_semafor"))

    def test_known_extrema_and_repainting(self):
        """Лучшая вершина заменяет предыдущую; постоянный rolling max не годится."""
        from chart_delta_bar_semafor import calculate_zigzag
        data = turning_bars()
        result = calculate_zigzag(data, depth=3, deviation=0, backstep=1, point=1)
        np.testing.assert_allclose(result.zz_high,
                                   [np.nan, np.nan, np.nan, 15, np.nan, np.nan,
                                    np.nan, np.nan, 16, np.nan])
        np.testing.assert_allclose(result.zz_low,
                                   [np.nan, np.nan, np.nan, np.nan, np.nan,
                                    np.nan, 8, np.nan, np.nan, np.nan])
        prefix = calculate_zigzag(data.iloc[:8], depth=3, deviation=0, backstep=1, point=1)
        self.assertEqual(prefix.zz_high.iloc[7], 13)
        self.assertTrue(np.isnan(result.zz_high.iloc[7]))

    def test_disabled_short_and_invalid_inputs(self):
        """Нулевой уровень скрыт, короткая история прогревается, ошибки запрещены."""
        from chart_delta_bar_semafor import calculate_zigzag, calculate_indicators
        data = turning_bars()
        for depth in (0, 30):
            self.assertTrue(calculate_zigzag(data, depth=depth).isna().all().all())
        self.assertTrue(calculate_zigzag(data.iloc[:0]).empty)
        for settings in ({"depth": -1}, {"depth": 2.5}, {"deviation": -1},
                         {"backstep": -1}, {"point": 0}, {"point": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                calculate_zigzag(data, **settings)
        broken = data.copy()
        broken.loc[3, "low"] = 100
        with self.assertRaises(ValueError):
            calculate_indicators(broken)

    def test_deviation_is_scaled_by_price_step(self):
        """Допуск измеряется в шагах цены: почти достигнутый минимум тоже отмечается."""
        from chart_delta_bar_semafor import calculate_zigzag
        data = pd.DataFrame({"high": [4, 5, 6, 7], "low": [0, 1, 2, 3], "close": [2, 3, 4, 5]})
        strict = calculate_zigzag(data, depth=3, deviation=1, backstep=0, point=1)
        loose = calculate_zigzag(data, depth=3, deviation=1, backstep=0, point=2)
        self.assertTrue(strict.zz_low.isna().all())
        self.assertEqual(loose.zz_low.iloc[3], 1)

    def test_sma_islands_and_independent_levels(self):
        """Простые MA дают правильный цвет, а три глубины рассчитываются независимо."""
        from chart_delta_bar_semafor import calculate_indicators
        data = turning_bars()
        result = calculate_indicators(data, depths=(3, 8, 0), deviation=0,
                                      backstep=1, point=1, ma_fast=2, ma_slow=3)
        np.testing.assert_allclose(result.ma_fast.iloc[:5], [np.nan, 11, 11.5, 12.5, 13.5])
        np.testing.assert_allclose(result.ma_slow.iloc[:5], [np.nan, np.nan, 11, 37/3, 38/3])
        self.assertEqual(result.island_direction.iloc[:5].tolist(), [0, 0, 1, 1, 1])
        self.assertTrue(result.sf3_high.isna().all())
        self.assertEqual(result.sf1_high.iloc[3], 15)
        self.assertTrue(np.isnan(result.sf2_high.iloc[3]))
        self.assertEqual(result.sf2_high.iloc[8], 16)
        self.assertEqual(data.columns.tolist(), ["high", "low", "close"])


class SemaforDatabaseTests(unittest.TestCase):
    """Проверяет прогрев, ограничение конечной даты и чтение без старых индикаторов."""

    def setUp(self):
        """Создаёт временную базу с двумя наборами и повторяющимся временем."""
        self.fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_history_warmup_and_readonly(self):
        """Срез дат сохраняет прогрев SMA/СФ; ALF и StopVolume не рассчитываются."""
        from chart_delta_bar_semafor import load_semafor_bars
        before = self.fixture.db.read_bytes()
        settings = dict(depths=(2, 3, 5), deviation=0, backstep=1, point=1, ma_fast=2, ma_slow=5)
        with patch("source.chart_data.laguerre", side_effect=AssertionError("ALF запрещён")), \
             patch("source.chart_data.volume_stops", side_effect=AssertionError("StopVolume запрещён")):
            whole = load_semafor_bars(self.fixture.db, "RTS", "a", "2026-01-05", "2026-01-06", **settings)
            later = load_semafor_bars(self.fixture.db, "RTS", "a", "2026-01-06", "2026-01-06", **settings)
            early = load_semafor_bars(self.fixture.db, "RTS", "a", "2026-01-05", "2026-01-05", **settings)
        self.assertEqual(len(whole), 8)
        self.assertEqual(len(early), 4)
        self.assertEqual(later.x.tolist(), [0, 1, 2, 3])
        for name in ["ma_fast", "ma_slow", "sf1_high", "sf2_low", "sf3_high"]:
            np.testing.assert_allclose(later[name], whole[name].iloc[4:])
        self.assertTrue(later.ma_slow.notna().all())
        self.assertEqual(later.dataset_id.unique().tolist(), ["a"])
        self.assertNotIn("alf", whole.columns)
        self.assertNotIn("long_rising", whole.columns)
        self.assertEqual(later.start_time.iloc[0].nanosecond, 1)
        self.assertEqual(before, self.fixture.db.read_bytes())
        with self.assertRaises(ValueError):
            load_semafor_bars(self.fixture.db, "RTS", "a", "2026-02-02", "2026-02-01")
        self.assertTrue(load_semafor_bars(self.fixture.db, "RTS", "a", "2026-02-01", "2026-02-02").empty)


class SemaforWindowTests(unittest.TestCase):
    """Проверяет реальные свечи, настройки, видимость и ошибочные диапазоны в Qt."""

    @classmethod
    def setUpClass(cls):
        """Сохраняет единственное QApplication для всех проверок окна."""
        from chart_delta_bars import create_application
        cls.app = create_application()

    def setUp(self):
        """Подготавливает независимую тестовую базу и её автоматическую очистку."""
        self.fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def wait_loaded(self, window):
        """Обрабатывает события до конца фоновой загрузки, максимум четыре секунды."""
        for _ in range(200):
            self.app.processEvents()
            if not window.loading:
                return
            QtTest.QTest.qWait(20)
        self.fail("Загрузка не завершилась за четыре секунды")

    def test_controls_visibility_reload_and_errors(self):
        """Настройки пересчитывают СФ, флажки скрывают элементы, ошибка сохраняет график."""
        from chart_delta_bar_semafor import ChartWindow
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-05", end="2026-01-06",
                             depths=(2, 3, 5), ma_fast=2, ma_slow=5, point=1)
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        self.assertEqual(len(window.data), 8)
        self.assertEqual(len(window.canvas.candles.datasrc.df), 8)
        self.assertIn("Семафор", window.windowTitle())
        controls = [w.text() for w in window.findChildren(QtWidgets.QCheckBox)]
        self.assertFalse(any("ALF" in text or "Stops" in text for text in controls))
        window.islands_check.setChecked(False)
        self.assertTrue(all(not item.isVisible() for item in window.canvas.island_items))
        window.semafor_checks[0].setChecked(False)
        self.assertTrue(all(not item.isVisible() for item in window.canvas.semafor_items[0]))
        window.duration_check.setChecked(True)
        self.assertTrue(window.canvas.duration_axis.isVisible())
        window.depth_spins[2].setValue(3)
        window.load_button.click()
        self.wait_loaded(window)
        self.assertEqual(window.data.attrs["depths"], (2, 3, 3))
        window.canvas.describe_bar(7)
        self.assertIn("Дельта", window.details.text())
        canvas = window.canvas
        window.start_edit.setDate(QtCore.QDate(2026, 2, 2))
        window.end_edit.setDate(QtCore.QDate(2026, 2, 1))
        window.load_button.click()
        self.wait_loaded(window)
        self.assertIs(window.canvas, canvas)
        self.assertIn("Ошибка", window.status_label.text())
        window.end_edit.setDate(QtCore.QDate(2026, 2, 3))
        window.load_button.click()
        self.wait_loaded(window)
        self.assertTrue(window.data.empty)
        self.assertIsNone(window.canvas)

    def test_zoom_includes_islands_and_offset_suns(self):
        """Автомасштаб включает SMA после гэпа и низины СФ с большим отступом."""
        from chart_delta_bar_semafor import ChartWindow, calculate_indicators, load_semafor_bars
        sample = load_semafor_bars(self.fixture.db, "RTS", "a", "2026-01-05", "2026-01-06")
        data = pd.concat([sample.iloc[:1]] * 85, ignore_index=True)
        prices = np.array([150000.] * 50 + [100000.] * 35)
        data["open"] = data["close"] = prices
        data["high"], data["low"] = prices + 10, prices - 10
        data["x"] = np.arange(len(data))
        data = calculate_indicators(data)
        window = ChartWindow(db_path=self.fixture.db, auto_load=False)
        self.addCleanup(window.close)
        window.offset_spin.setValue(5000)
        window.show()
        window.accept_data(data)
        self.app.processEvents()
        window.canvas.price_axis.vb.update_y_zoom(49.5, 74.5)
        bottom, top = window.canvas.price_axis.vb.viewRange()[1]
        self.assertLess(bottom, 94990)
        self.assertGreater(top, 148529)


if __name__ == "__main__":
    unittest.main()
