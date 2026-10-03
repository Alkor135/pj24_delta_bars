r"""Проверяет последовательный показ баров, причинность индикаторов и скорость.

Запуск из корня проекта:
    .\.venv\Scripts\python.exe -m unittest -v tests.test_chart_semafor_simulate

SQLite создаётся во временной папке, Qt работает offscreen. Числовые ожидания
заданы вручную: будущая вершина не должна менять уже показанный Семафор до
своего появления. Проверяются реальные графические объекты и таймер Qt.
"""

import importlib.util
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtTest

from tests import test_chart_data


def replay_bars():
    """Возвращает десять баров с будущей вершиной, границей дня и дневным остатком."""
    highs = np.array([11, 13, 12, 15, 14, 12, 10, 13, 16, 15], dtype=float)
    stamps = pd.date_range("2026-01-05 23:59:52", periods=10, freq="s")
    return pd.DataFrame(dict(dataset_id="a", symbol="RTS", day=stamps.strftime("%Y-%m-%d"),
                             bar_index=np.arange(10), start_time=stamps,
                             end_time=stamps + pd.Timedelta(milliseconds=500),
                             open=highs - 1, high=highs, low=highs - 2, close=highs - 1,
                             volume=np.arange(1, 11), delta=2, threshold=2, tick_count=5,
                             is_complete=[1] * 7 + [0, 1, 1], close_reason="threshold",
                             duration_seconds=0.5))


class ReplayTests(unittest.TestCase):
    """Ловит заглядывание вперёд, потерю прогрева и запись в исходную базу."""

    def setUp(self):
        """Требует новый модуль и сохраняет параметры проверяемого Семафора."""
        self.assertIsNotNone(importlib.util.find_spec("chart_delta_bar_semafor_simulate"),
                             "Отсутствует запрошенный скрипт симуляции")
        self.settings = dict(depths=(3, 0, 0), deviation=0, backstep=1,
                             point=1, ma_fast=2, ma_slow=3)

    def test_prefix_repaints_only_when_new_bar_arrives(self):
        """Будущая вершина 16 заменяет 13 только после девятого показанного бара."""
        from chart_delta_bar_semafor_simulate import ReplaySession
        session = ReplaySession(replay_bars(), "2026-01-05", 8, **self.settings)
        first = session.frame()
        self.assertEqual(len(first), 8)
        self.assertEqual(first.sf1_high.iloc[7], 13)
        self.assertEqual(first.ma_fast.iloc[-1], 10.5)
        next_frame = session.advance()
        self.assertEqual(next_frame.x.tolist(), list(range(9)))
        self.assertTrue(np.isnan(next_frame.sf1_high.iloc[7]))
        self.assertEqual(next_frame.sf1_high.iloc[8], 16)
        self.assertEqual(first.sf1_high.iloc[7], 13)
        session.advance()
        self.assertTrue(session.finished)
        self.assertEqual(len(session.advance()), 10)

    def test_warmup_precedes_visible_date(self):
        """Предыстория прогревает SMA, а будущие видимые бары ещё скрыты."""
        from chart_delta_bar_semafor_simulate import ReplaySession
        session = ReplaySession(replay_bars(), "2026-01-06", 1, **self.settings)
        data = session.frame()
        self.assertEqual(data.x.tolist(), [0])
        self.assertEqual(data.close.tolist(), [15])
        self.assertEqual(data.ma_fast.iloc[0], 13.5)
        self.assertEqual(data.ma_slow.iloc[0], 12)
        self.assertEqual(session.total, 2)
        self.assertFalse(session.finished)

    def test_short_empty_and_invalid_settings(self):
        """Короткий диапазон оставляет последний бар для шага; пустой завершается."""
        from chart_delta_bar_semafor_simulate import ReplaySession, validate_simulation
        short = ReplaySession(replay_bars().iloc[:3], "2026-01-05", 200, **self.settings)
        self.assertEqual(len(short.frame()), 2)
        self.assertFalse(short.finished)
        empty = ReplaySession(replay_bars(), "2026-02-01", 200, **self.settings)
        self.assertTrue(empty.frame().empty)
        self.assertTrue(empty.finished)
        for initial, interval in ((0, 1), (2, 0), (2, float("nan")), (2, 100)):
            with self.subTest(initial=initial, interval=interval), self.assertRaises(ValueError):
                validate_simulation(initial, interval)

    def test_database_snapshot_keeps_warmup_and_precision(self):
        """Снимок выбранного набора включает прогрев и не меняет файл SQLite."""
        from chart_delta_bar_semafor_simulate import load_replay_session
        fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        before = fixture.db.read_bytes()
        session = load_replay_session(fixture.db, "RTS", "a", "2026-01-06", "2026-01-06",
                                      initial_bars=1, **self.settings)
        data = session.frame()
        self.assertEqual(data.close.tolist(), [111])
        self.assertEqual(data.ma_fast.iloc[0], 107.5)
        self.assertEqual(data.start_time.iloc[0].nanosecond, 1)
        self.assertEqual(session.total, 4)
        self.assertEqual(before, fixture.db.read_bytes())
        with self.assertRaises(ValueError):
            load_replay_session(fixture.db, "RTS", "a", "2026-01-06", "2026-01-05")


class SimulationWindowTests(unittest.TestCase):
    """Ловит потерю графика при обновлении и ошибки управления воспроизведением."""

    @classmethod
    def setUpClass(cls):
        """Создаёт единое приложение Qt для интеграционных проверок."""
        from chart_delta_bars import create_application
        cls.app = create_application()

    def setUp(self):
        """Готовит временную SQLite и требует реализацию нового окна."""
        self.assertIsNotNone(importlib.util.find_spec("chart_delta_bar_semafor_simulate"),
                             "Отсутствует запрошенный скрипт симуляции")
        self.fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def wait_loaded(self, window):
        """Обрабатывает события до окончания расчёта window, максимум пять секунд."""
        for _ in range(250):
            self.app.processEvents()
            if not window.loading:
                return
            QtTest.QTest.qWait(20)
        self.fail("Фоновый расчёт не завершился за пять секунд")

    def make_window(self, **kwargs):
        """Открывает симуляцию с kwargs и регистрирует закрытие до очистки SQLite."""
        from chart_delta_bar_semafor_simulate import ChartWindow
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-05", end="2026-01-06",
                             initial_bars=2, interval=60, **kwargs)
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        return window

    def test_step_pause_speed_and_finish(self):
        """Пауза останавливает поток, шаг добавляет один бар, скорость действует сразу."""
        window = self.make_window()
        self.assertEqual(len(window.data), 2)
        self.assertTrue(window.timer.isActive())
        window.pause_button.click()
        self.assertFalse(window.timer.isActive())
        canvas, candles = window.canvas, window.canvas.candles
        window.step_button.click()
        self.wait_loaded(window)
        self.assertEqual(len(window.data), 3)
        self.assertIs(window.canvas, canvas)
        self.assertIs(window.canvas.candles, candles)
        self.assertEqual(len(candles.datasrc.df), 3)
        self.assertFalse(window.timer.isActive())
        window.interval_spin.setValue(0.05)
        self.assertEqual(window.timer.interval(), 50)
        window.pause_button.click()
        for _ in range(200):
            self.app.processEvents()
            if window.session.finished and not window.loading:
                break
            QtTest.QTest.qWait(20)
        self.assertEqual(len(window.data), 8)
        self.assertFalse(window.timer.isActive())
        self.assertFalse(window.step_button.isEnabled())
        window.restart_button.click()
        self.wait_loaded(window)
        window.pause_button.click()
        self.assertEqual(len(window.data), 2)

    def test_new_markers_repaint_and_day_boundaries(self):
        """Новые СФ и остаток дня появляются; старый экстремум исчезает на том же холсте."""
        from chart_delta_bar_semafor_simulate import ReplaySession
        window = self.make_window()
        window.pause_button.click()
        session = ReplaySession(replay_bars(), "2026-01-05", 2, depths=(3, 0, 0),
                                deviation=0, backstep=1, ma_fast=2, ma_slow=3)
        window.accept_session(session)
        window.pause_button.click()
        canvas = window.canvas
        window.semafor_checks[0].setChecked(False)
        for _ in range(6):
            window.step_button.click()
            self.wait_loaded(window)
        self.assertTrue(all(not item.isVisible() for item in canvas.semafor_items[0]))
        window.semafor_checks[0].setChecked(True)
        high = canvas.marker_items["sf1_high"][0]
        self.assertEqual(high.datasrc.y.iloc[7], 13 + window.offset_spin.value())
        self.assertIn(7, high.xData)
        self.assertIsNotNone(canvas.partial_item)
        self.assertFalse(canvas.day_lines)
        window.step_button.click()
        self.wait_loaded(window)
        self.assertIs(window.canvas, canvas)
        self.assertTrue(np.isnan(high.datasrc.y.iloc[7]))
        self.assertNotIn(7, high.xData)
        self.assertEqual(high.datasrc.y.iloc[8], 16 + window.offset_spin.value())
        self.assertEqual(len(canvas.day_lines), 2)
        self.assertEqual(canvas.price_axis.getAxis("bottom").data.x.tolist(), list(range(9)))

    def test_history_view_survives_update_and_failed_reload(self):
        """Отключённое слежение сохраняет X; неверные даты оставляют текущую симуляцию."""
        window = self.make_window()
        window.pause_button.click()
        window.follow_check.setChecked(False)
        canvas = window.canvas
        before = canvas.price_axis.vb.viewRange()[0]
        window.step_button.click()
        self.wait_loaded(window)
        np.testing.assert_allclose(canvas.price_axis.vb.viewRange()[0], before)
        window.start_edit.setDate(QtCore.QDate(2026, 2, 2))
        window.end_edit.setDate(QtCore.QDate(2026, 2, 1))
        window.load_button.click()
        self.wait_loaded(window)
        self.assertIs(window.canvas, canvas)
        self.assertEqual(len(window.data), 3)
        self.assertIn("Ошибка", window.status_label.text())

    def test_existing_markers_in_both_presets_and_reset_view(self):
        """Начальные СФ обоих пресетов обновляются; перезапуск возвращает видимые бары."""
        from chart_delta_bar_semafor_simulate import ReplaySession
        for preset in ("nuf", "fxi"):
            with self.subTest(preset=preset):
                window = self.make_window(preset=preset)
                window.pause_button.click()
                session = ReplaySession(replay_bars(), "2026-01-05", 8, depths=(3, 3, 3),
                                        deviation=0, backstep=1, ma_fast=2, ma_slow=3)
                window.accept_session(session)
                window.pause_button.click()
                high = window.canvas.marker_items["sf3_high"]
                self.assertEqual(len(high), 2 if preset == "fxi" else 1)
                window.step_button.click()
                self.wait_loaded(window)
                for item in high:
                    self.assertNotIn(7, item.xData)
                    self.assertIn(8, item.xData)
                window.canvas.price_axis.vb.setXRange(5, 9, padding=0)
                window.restart_button.click()
                self.wait_loaded(window)
                window.pause_button.click()
                left, right = window.canvas.price_axis.vb.viewRange()[0]
                self.assertLess(left, 1)
                self.assertGreater(right, 1)
                window.close()


if __name__ == "__main__":
    unittest.main()
