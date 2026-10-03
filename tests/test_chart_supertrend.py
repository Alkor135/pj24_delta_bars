"""Проверяет Supertrend, чтение с предысторией и отдельное окно дельта-баров.

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m unittest -v tests.test_chart_supertrend

Числовые примеры рассчитаны вручную. Базы создаются во временной папке;
окно Qt проверяется в режиме offscreen без изменения пользовательских данных.
Отдельно проверяется видимость линии, впервые рассчитанной на последнем баре.
Длинная история проверяет сохранение этой ступени при масштабировании.
"""

import importlib.util
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from contextlib import closing
import sqlite3

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtTest

from tests import test_chart_data


def example_bars():
    """Возвращает восемь OHLC-баров с разрывами цены и тремя разворотами тренда."""
    return pd.DataFrame({
        "high": [11, 12, 13, 16, 17, 13, 10, 15],
        "low": [9, 10, 11, 14, 15, 11, 8, 13],
        "close": [10, 11, 12, 15, 16, 12, 9, 14],
    })


class SupertrendTests(unittest.TestCase):
    """Проверяет числа, строгие пересечения, причинность и непригодные данные."""

    def test_script_available(self):
        """Требует отдельную точку входа для графика с индикатором Supertrend."""
        self.assertIsNotNone(importlib.util.find_spec("chart_delta_bars_supertrend"))

    def test_wilder_atr_and_trend_reversals(self):
        """Сверяет ATR с гэпами, подтягивание полос и оба направления разворота."""
        from chart_delta_bars_supertrend import calculate_supertrend
        data = example_bars()
        result = calculate_supertrend(data, period=3, multiplier=1)
        np.testing.assert_allclose(result.atr, [np.nan, np.nan, 2, 8/3, 22/9,
                                               89/27, 286/81, 1058/243])
        np.testing.assert_allclose(result.supertrend, [np.nan, np.nan, 14, 37/3,
                                                      122/9, 413/27, 1015/81, 2344/243])
        np.testing.assert_array_equal(result.supertrend_direction, [0, 0, -1, 1, 1, -1, -1, 1])
        np.testing.assert_allclose(result.supertrend_up, [np.nan, np.nan, np.nan, 37/3,
                                                         122/9, np.nan, np.nan, 2344/243])
        np.testing.assert_allclose(result.supertrend_down, [np.nan, np.nan, 14, np.nan,
                                                           np.nan, 413/27, 1015/81, np.nan])
        self.assertEqual(data.columns.tolist(), ["high", "low", "close"])

    def test_touching_band_does_not_reverse_trend(self):
        """Касание полосы не считается пересечением; период ATR 1 поддерживается."""
        from chart_delta_bars_supertrend import calculate_supertrend
        data = pd.DataFrame({"high": [11, 13, 14, 13, 11], "low": [9, 11, 12, 11, 9],
                             "close": [10, 12, 13, 11, 10]})
        result = calculate_supertrend(data, period=1, multiplier=1)
        np.testing.assert_allclose(result.atr, [2, 3, 2, 2, 2])
        np.testing.assert_allclose(result.supertrend, [12, 12, 11, 11, 12])
        np.testing.assert_array_equal(result.supertrend_direction, [-1, -1, 1, 1, -1])

    def test_warmup_empty_and_constant_prices(self):
        """До полного ATR линия отсутствует; пустая история и нулевой диапазон допустимы."""
        from chart_delta_bars_supertrend import calculate_supertrend
        data = example_bars().iloc[:2].copy()
        result = calculate_supertrend(data, period=3, multiplier=3)
        self.assertTrue(result.atr.isna().all())
        self.assertTrue(result.supertrend.isna().all())
        self.assertTrue((result.supertrend_direction == 0).all())
        self.assertTrue(calculate_supertrend(data.iloc[:0]).empty)
        constant = pd.DataFrame({"high": [100] * 5, "low": [100] * 5, "close": [100] * 5})
        result = calculate_supertrend(constant, period=3, multiplier=3)
        np.testing.assert_allclose(result.supertrend, [np.nan, np.nan, 100, 100, 100])

    def test_future_bars_do_not_change_past_values(self):
        """Добавление будущего разворота не меняет значения на уже рассчитанных барах."""
        from chart_delta_bars_supertrend import calculate_supertrend
        whole = calculate_supertrend(example_bars(), period=3, multiplier=1)
        prefix = calculate_supertrend(example_bars().iloc[:5], period=3, multiplier=1)
        pd.testing.assert_frame_equal(prefix, whole.iloc[:5])

    def test_rejects_invalid_parameters_and_prices(self):
        """Ошибочные настройки и невозможные OHLC останавливают расчёт понятной ошибкой."""
        from chart_delta_bars_supertrend import calculate_supertrend
        for period, multiplier in [(0, 3), (-1, 3), (2.5, 3), (True, 3),
                                   (3, 0), (3, -1), (3, np.nan), (3, np.inf)]:
            with self.subTest(period=period, multiplier=multiplier), self.assertRaises(ValueError):
                calculate_supertrend(example_bars(), period, multiplier)
        for column, value in [("high", np.nan), ("low", 12), ("close", 20)]:
            data = example_bars()
            data.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                calculate_supertrend(data)


class SupertrendDatabaseTests(unittest.TestCase):
    """Проверяет предысторию, сохранение баров и чтение базы без записи."""

    def setUp(self):
        """Создаёт временную базу с двумя наборами; регистрирует её удаление после теста."""
        self.fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_history_before_start_preserves_indicator_and_database(self):
        """Фильтр дат не сбрасывает ATR, ALF или тренд и не меняет байты SQLite."""
        from chart_delta_bars_supertrend import load_supertrend_bars
        before = self.fixture.db.read_bytes()
        whole = load_supertrend_bars(self.fixture.db, "RTS", "a", "2026-01-05", "2026-01-06",
                                    period=3, multiplier=1)
        later = load_supertrend_bars(self.fixture.db, "RTS", "a", "2026-01-06", "2026-01-06",
                                    period=3, multiplier=1)
        self.assertEqual(len(whole), 8)
        self.assertEqual(later.x.tolist(), [0, 1, 2, 3])
        for name in ["atr", "supertrend", "supertrend_direction", "alf"]:
            np.testing.assert_allclose(later[name], whole[name].iloc[4:])
        self.assertTrue(later.supertrend.notna().all())
        self.assertEqual(later.dataset_id.unique().tolist(), ["a"])
        self.assertEqual(before, self.fixture.db.read_bytes())
        self.assertEqual(later.start_time.iloc[0].nanosecond, 1)

    def test_empty_and_invalid_date_ranges(self):
        """Пустой период допустим, но перевёрнутые даты запрещены до загрузки истории."""
        from chart_delta_bars_supertrend import load_supertrend_bars
        self.assertTrue(load_supertrend_bars(self.fixture.db, "RTS", "a",
                                           "2026-02-01", "2026-02-02").empty)
        with self.assertRaises(ValueError):
            load_supertrend_bars(self.fixture.db, "RTS", "a", "2026-02-02", "2026-02-01")


class SupertrendWindowTests(unittest.TestCase):
    """Проверяет загрузку нового окна, параметры, видимость и прогрев индикатора."""

    @classmethod
    def setUpClass(cls):
        """Сохраняет единственное QApplication для всех проверок собственного окна."""
        from chart_delta_bars_supertrend import create_application
        cls.app = create_application()

    def setUp(self):
        """Создаёт временную базу и добавляет очистку после закрытия тестового окна."""
        self.fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def wait_loaded(self, window):
        """Обрабатывает события окна до завершения фоновой загрузки, не более 4 секунд."""
        for _ in range(200):
            self.app.processEvents()
            if not window.loading:
                return
            QtTest.QTest.qWait(20)
        self.fail("Загрузка окна не завершилась за 4 секунды")

    def test_controls_loading_and_visibility(self):
        """Новые параметры меняют расчёт, флажок скрывает обе линии, свечи сохраняются."""
        from chart_delta_bars_supertrend import ChartWindow
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-05", end="2026-01-06",
                             period=3, multiplier=1)
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        self.assertEqual(len(window.data), 8)
        self.assertEqual(len(window.canvas.candles.datasrc.df), 8)
        self.assertEqual(window.canvas.price_axis.getAxis("bottom").tickStrings([0, 4], 1, 4),
                         ["05.01\n10:00", "06.01\n10:00"])
        self.assertTrue(window.supertrend_check.isChecked())
        self.assertFalse(window.alf_check.isChecked())
        self.assertFalse(window.signals_check.isChecked())
        self.assertIn("Supertrend", window.windowTitle())
        window.supertrend_check.setChecked(False)
        self.assertTrue(all(not item.isVisible() for item in window.canvas.supertrend_items))
        window.supertrend_check.setChecked(True)
        self.assertTrue(all(item.isVisible() for item in window.canvas.supertrend_items))
        window.canvas.describe_bar(7)
        self.assertIn("Supertrend", window.details.text())
        self.assertIn("Дельта", window.details.text())
        previous = window.data.atr.copy()
        window.period_spin.setValue(1)
        window.multiplier_spin.setValue(2)
        window.load_button.click()
        self.wait_loaded(window)
        self.assertTrue(window.data.atr.notna().all())
        self.assertFalse(previous.equals(window.data.atr))
        self.assertEqual(window.data.attrs["supertrend_multiplier"], 2)
        window.duration_check.setChecked(True)
        self.assertTrue(window.canvas.duration_axis.isVisible())
        window.refresh_button.click()
        self.wait_loaded(window)
        self.assertEqual(len(window.data), 8)

    def test_warmup_error_and_empty_range(self):
        """Недостаток ATR не мешает свечам; ошибка сохраняет график, пустой период убирает его."""
        from chart_delta_bars_supertrend import ChartWindow
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-05", end="2026-01-06",
                             period=20)
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        self.assertEqual(len(window.data), 8)
        self.assertIsNotNone(window.canvas)
        self.assertTrue(window.data.supertrend.isna().all())
        self.assertIn("прогрев", window.status_label.text().lower())
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
        self.assertIn("Нет баров", window.status_label.text())

    def test_first_supertrend_on_last_bar_has_visible_segment(self):
        """Первое значение на последнем баре имеет горизонтальный отрезок, а не точку."""
        from chart_delta_bars_supertrend import ChartWindow
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-05", end="2026-01-06",
                             period=8)
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        self.assertEqual(int(window.data.supertrend.notna().sum()), 1)
        path = window.canvas.supertrend_items[1].curve.getPath()
        self.assertGreater(path.boundingRect().width(), 0)
        self.assertGreater(path.boundingRect().right(), 7)
        self.assertEqual(len(window.canvas.candles.datasrc.df), 8)

    def test_last_segment_survives_dense_history_and_zoom(self):
        """Последняя короткая ветвь не теряется из-за прореживания длинной истории."""
        from chart_delta_bars_supertrend import ChartWindow
        with closing(sqlite3.connect(self.fixture.db)) as connection:
            for dataset in ("a", "b"):
                template = list(connection.execute(
                    "SELECT * FROM bars WHERE dataset_id=? AND day='2026-01-06' LIMIT 1",
                    (dataset,)).fetchone())
                rows = []
                for index in range(4, 596):
                    template[3] = index
                    rows.append(tuple(template))
                connection.executemany("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            connection.commit()
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-05", end="2026-01-06",
                             period=600)
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        self.assertEqual(int(window.data.supertrend.notna().sum()), 1)
        path = window.canvas.supertrend_items[1].curve.getPath()
        self.assertGreater(path.boundingRect().width(), 0)
        window.all_button.click()
        self.app.processEvents()
        path = window.canvas.supertrend_items[1].curve.getPath()
        self.assertGreater(path.boundingRect().width(), 0)
        self.assertGreater(path.boundingRect().right(), 599)


if __name__ == "__main__":
    unittest.main()
