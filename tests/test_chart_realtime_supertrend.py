r"""Проверяет Supertrend на изменяемых барах QUIK и двух самостоятельных Qt-окнах.

Запуск из корня:
    .\.venv\Scripts\python.exe -m unittest -v tests.test_chart_realtime_supertrend
Временная история и сделки проверяют прогрев до видимого диапазона, обновление
текущей свечи и поздний тик. Qt offscreen проверяет прежние линии/масштаб,
видимость, компактную легенду, параметры ATR, карточку бара и первый короткий
отрезок Supertrend.
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from chart_delta_bars_supertrend import calculate_supertrend
from source.realtime_core import Trade
from tests import test_chart_data
from tests.test_chart_semafor_simulate import replay_bars


def make_session(symbol="RTS", start="2026-01-05", period=3):
    """Возвращает сессию symbol с полной предысторией, видимым start и ATR period."""
    from source.realtime_supertrend import SupertrendSession
    history = replay_bars()
    history["symbol"] = symbol
    return SupertrendSession(history, symbol, "a", start, "2026-01-07",
        lambda day: dict(threshold=5, source="ручной порог", day=day),
        dict(depths=(3, 0, 0), deviation=0, backstep=1, point=1,
             ma_fast=2, ma_slow=3, atr_period=period, multiplier=1))


class SupertrendSessionTests(unittest.TestCase):
    """Проверяет согласованность исторического Supertrend и обновляемых свечей."""

    def test_visible_start_keeps_full_atr_warmup(self):
        """Первый видимый бар использует ATR прошлых скрытых баров, как обычный Supertrend."""
        session = make_session(start="2026-01-06", period=5)
        reference = calculate_supertrend(replay_bars(), 5, 1).iloc[-2:].reset_index(drop=True)
        self.assertEqual(len(session.data), 2)
        self.assertTrue(session.data.atr.notna().all())
        pd.testing.assert_frame_equal(session.data[reference.columns], reference)
        self.assertIn("sf1_high", session.data)
        self.assertIn("ma_slow", session.data)
        self.assertEqual(session.data.attrs["supertrend_period"], 5)

    def test_live_and_late_ticks_recalculate_partial_bar_without_advancing_atr_twice(self):
        """Несколько тиков одного бара и поздняя сделка дают ATR пакетной последовательности OHLC."""
        session = make_session()
        original = session.data.copy()
        trades = [Trade(session.today, 100, "1", "SPBFUT", "RIH6", 100, 1),
                  Trade(session.today, 300, "3", "SPBFUT", "RIH6", 101, 3)]
        for trade in trades:
            session.ingest([trade])
        self.assertEqual(len(session.data), len(original) + 1)
        session.ingest([Trade(session.today, 200, "2", "SPBFUT", "RIH6", 104, 1)])
        expected = calculate_supertrend(session.data, 3, 1)
        pd.testing.assert_frame_equal(session.data[expected.columns], expected)
        pd.testing.assert_series_equal(session.data.atr.iloc[:len(original)], original.atr)
        session.ingest([Trade(session.today, 400, "4", "SPBFUT", "RIH6", 105, 10),
                        Trade(session.today, 500, "5", "SPBFUT", "RIH6", 102, 1)])
        expected = calculate_supertrend(session.data, 3, 1)
        pd.testing.assert_frame_equal(session.data[expected.columns], expected)
        self.assertEqual(session.data.is_complete.iloc[-2:].tolist(), [1, 0])


class SupertrendWindowTests(unittest.TestCase):
    """Проверяет дополнительные линии вместе с прежними свечами, Семафором и SMA."""

    @classmethod
    def setUpClass(cls):
        """Создаёт приложение Qt для независимых окон RTS и MIX."""
        from chart_delta_bars import create_application
        cls.app = create_application()

    def setUp(self):
        """Готовит временную SQLite и очистку после закрытия окон."""
        self.fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_two_windows_keep_lines_canvas_visibility_and_history_range(self):
        """Оба инструмента сохраняют холст, линии, видимость и X; легенда занимает меньше половины графика."""
        from source.realtime_supertrend import SupertrendWindow
        for symbol in ("RTS", "MIX"):
            window = SupertrendWindow(symbol, db_path=self.fixture.db, auto_load=False,
                                      atr_period=3, multiplier=1)
            self.addCleanup(window.close)
            session = make_session(symbol)
            window.accept_session(session)
            window.show()
            self.app.processEvents()
            canvas, lines = window.canvas, list(window.canvas.supertrend_items)
            self.assertLess(canvas.price_axis.legend.boundingRect().width(),
                            canvas.price_axis.vb.width() / 2)
            window.follow_check.setChecked(False)
            window.supertrend_check.setChecked(False)
            before = canvas.price_axis.vb.viewRange()[0]
            code = "RIH6" if symbol == "RTS" else "MXH6"
            session.ingest([Trade(session.today, 1, "1", "SPBFUT", code, 100, 1)])
            window.accept_session(session)
            self.assertIs(window.canvas, canvas)
            self.assertEqual(canvas.supertrend_items, lines)
            self.assertTrue(all(not item.isVisible() for item in lines))
            np.testing.assert_allclose(canvas.price_axis.vb.viewRange()[0], before)
            self.assertEqual(len(canvas.candles.datasrc.df), len(session.data))
            for item, column in zip(lines, ("supertrend_up", "supertrend_down")):
                np.testing.assert_allclose(item.yData[:-1], session.data[column], equal_nan=True)
            window.supertrend_check.setChecked(True)
            self.assertTrue(all(item.isVisible() for item in lines))
            window.duration_check.setChecked(True)
            self.assertTrue(canvas.duration_axis.isVisible())
            canvas.describe_bar(len(session.data) - 1)
            self.assertIn("Supertrend", window.details.text())
            self.assertIn("SMA", window.details.text())
            self.assertIn("Supertrend", window.windowTitle())
            self.assertTrue(window.islands_check.isChecked())
            self.assertEqual(window.current_request()["settings"]["atr_period"], 3)

    def test_first_live_supertrend_draws_short_last_segment(self):
        """ATR прогревается на первой живой свече, и единственная точка становится видимым отрезком."""
        from source.realtime_supertrend import SupertrendWindow
        period = len(replay_bars()) + 1
        window = SupertrendWindow("RTS", db_path=self.fixture.db, auto_load=False,
                                  atr_period=period, multiplier=1)
        self.addCleanup(window.close)
        session = make_session(period=period)
        window.accept_session(session)
        window.show()
        self.app.processEvents()
        self.assertTrue(session.data.supertrend.isna().all())
        self.assertIn("прогрев", window.status_label.text())
        session.ingest([Trade(session.today, 1, "1", "SPBFUT", "RIH6", 100, 1)])
        window.accept_session(session)
        path = window.canvas.supertrend_items[1].curve.getPath()
        self.assertGreater(path.boundingRect().width(), 0)
        self.assertGreater(path.boundingRect().right(), session.data.x.iloc[-1])

    def test_worker_uses_supertrend_session_and_controls_parameters(self):
        """Фоновая загрузка выбирает расширенный расчёт с текущими параметрами ATR окна."""
        from source.realtime_chart import RealtimeWorker
        from source.realtime_supertrend import SupertrendSession, SupertrendWindow
        window = SupertrendWindow("RTS", db_path=self.fixture.db, auto_load=False,
                                  atr_period=5, multiplier=2)
        self.addCleanup(window.close)
        packet = dict(status=dict(feed_connected=True, contracts={}, message="Подключено"),
                      trades=[], cursor=0, more=False)
        with patch("source.realtime_chart.moscow_day", return_value="2026-01-07"), \
             patch("source.realtime_chart.load_history", return_value=replay_bars()), \
             patch("source.realtime_chart.resolve_threshold", return_value=dict(threshold=5, source="ручной порог")), \
             patch("source.realtime_chart.ensure_collector"), \
             patch("source.realtime_chart.feed_request", return_value=packet):
            loaded, errors = [], []
            worker = RealtimeWorker(window.current_request())
            worker.loaded.connect(loaded.append)
            worker.failed.connect(errors.append)
            worker.run()
            self.assertEqual(errors, [])
            self.assertIsInstance(loaded[0], SupertrendSession)
            self.assertEqual(loaded[0].data.attrs["supertrend_period"], 5)
            self.assertEqual(loaded[0].data.attrs["supertrend_multiplier"], 2)


if __name__ == "__main__":
    unittest.main()
