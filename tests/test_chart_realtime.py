"""Проверяет текущую свечу, прогрев и два живых окна Qt без биржи.

Запуск: .\\.venv\\Scripts\\python.exe -m unittest -v tests.test_chart_realtime
Qt работает offscreen; используются реальные графические элементы finplot.
"""

import importlib.util
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np

from tests.test_chart_semafor_simulate import replay_bars
from tests import test_chart_data
from source.realtime_core import Trade


class LiveSessionTests(unittest.TestCase):
    """Проверяет текущий день вместе с предысторией и дневной границей."""

    def setUp(self):
        """Требует реал-тайм модуль до запуска расчётных тестов."""
        self.assertIsNotNone(importlib.util.find_spec("source.realtime_chart"), "Живое окно ещё не реализовано")

    def session(self, symbol="RTS"):
        """Возвращает сессию symbol с историей и заданным вручную порогом 5."""
        from source.realtime_chart import LiveSession
        history = replay_bars()
        history["symbol"] = symbol
        return LiveSession(history, symbol, "a", "2026-01-05", "2026-01-07",
                           lambda day: dict(threshold=5, source="ручной порог", day=day),
                           dict(depths=(3, 0, 0), deviation=0, backstep=1, point=1, ma_fast=2, ma_slow=3))

    def test_current_candle_changes_and_completed_bar_is_added(self):
        """Первая свеча меняется внутри бара; следующие сделки добавляют завершённую свечу."""
        session = self.session()
        def trade(number, price, qty):
            """Создаёт упорядоченную сделку текущего дня с параметрами цены и объёма."""
            return Trade("2026-01-07", number * 10**9, str(number), "SPBFUT", "RIH6", price, qty)
        session.ingest([trade(1, 100, 2)])
        first = session.data.copy()
        session.ingest([trade(2, 101, 3)])
        self.assertEqual(len(first), len(session.data))
        self.assertEqual(session.data.close.iloc[-1], 101)
        self.assertEqual(session.data.volume.iloc[-1], 5)
        self.assertEqual(session.data.ma_fast.iloc[-1], (14 + 101) / 2)
        session.ingest([trade(3, 102, 3), trade(4, 102, 1)])
        self.assertEqual(session.data.is_complete.iloc[-2:].tolist(), [1, 0])
        self.assertEqual(session.data.delta.iloc[-2:].tolist(), [6, 1])
        self.assertEqual(session.data.x.tolist(), list(range(len(session.data))))

    def test_midnight_finalizes_old_partial_and_resets_direction(self):
        """Полночь завершает старый остаток; новый день начинается с нейтральной дельты."""
        session = self.session()
        session.ingest([Trade("2026-01-07", 1, "1", "SPBFUT", "RIH6", 100, 2)])
        session.change_day("2026-01-08")
        session.ingest([Trade("2026-01-08", 1, "1", "SPBFUT", "RIH6", 120, 50)])
        self.assertEqual(session.data.close_reason.iloc[-2:].tolist(), ["day_end", "live"])
        self.assertEqual(session.data.delta.iloc[-1], 0)

    def test_nightly_update_is_loaded_without_restart(self):
        """Открытая сессия подхватывает позднее ночное обновление истории и порога до первой сделки."""
        from source.realtime_chart import RealtimeWorker
        with tempfile.TemporaryDirectory() as folder:
            database, thresholds = Path(folder) / "history.sqlite3", Path(folder) / "thresholds.json"
            database.write_bytes(b"before")
            thresholds.write_text("before", encoding="utf-8")
            template = self.session()
            request = dict(db=database, symbol="RTS", dataset_id="a", start=template.start,
                           settings=template.settings, config_path=Path(folder) / "quik.json",
                           threshold_file=thresholds, manual_threshold=None)
            packet = dict(status=dict(feed_connected=True, contracts={}, message="Подключено"),
                          trades=[], cursor=0, more=False)
            with patch("source.realtime_chart.moscow_day", return_value=template.today), \
                 patch("source.realtime_chart.load_config", return_value={}), \
                 patch("source.realtime_chart.select_dataset", return_value="a"), \
                 patch("source.realtime_chart.load_history", return_value=template.history.copy()) as history_loader, \
                 patch("source.realtime_chart.resolve_threshold", return_value=dict(threshold=100, source="расчёт")) as provider, \
                 patch("source.realtime_chart.ensure_collector"), \
                 patch("source.realtime_chart.feed_request", return_value=packet):
                loaded, errors = [], []
                worker = RealtimeWorker(request)
                worker.loaded.connect(loaded.append)
                worker.failed.connect(errors.append)
                worker.run()
                self.assertEqual(errors, [])
                session = loaded[-1]
                self.assertEqual(session.thresholds[template.today]["threshold"], 100)
                database.write_bytes(b"after-nightly-conversion")
                thresholds.write_text("after-nightly-calibration", encoding="utf-8")
                updated = template.history.copy()
                updated.loc[updated.index[-1], "close"] = 24
                updated.loc[updated.index[-1], "high"] = 24
                history_loader.return_value = updated
                provider.return_value = dict(threshold=200, source="ночная подготовка")
                worker = RealtimeWorker(request, session=session)
                worker.loaded.connect(loaded.append)
                worker.failed.connect(errors.append)
                worker.run()
                self.assertEqual(errors, [])
                self.assertEqual(history_loader.call_count, 2)
                self.assertEqual(session.thresholds[template.today]["threshold"], 200)
                self.assertEqual(session.data.close.iloc[-1], 24)
                worker = RealtimeWorker(request, session=session)
                worker.run()
                self.assertEqual(history_loader.call_count, 2)
                self.assertEqual(provider.call_count, 2)

    def test_threshold_stays_fixed_after_first_trade_and_reports_changed_source(self):
        """Новый ночной порог не меняет начатый день молча; сессия просит явную пересборку."""
        session = self.session()
        session.ingest([Trade(session.today, 1, "1", "SPBFUT", "RIH6", 100, 2)])
        session.threshold_provider = lambda day: dict(threshold=10, source="ночная подготовка")
        session.refresh_threshold(session.today)
        self.assertEqual(session.thresholds[session.today]["threshold"], 5)
        self.assertEqual(session.builders[session.today].threshold, 5)
        self.assertIn("Обновить", session.source_warning)


class LiveWindowTests(unittest.TestCase):
    """Проверяет сохранение холста и независимость окон двух инструментов."""

    @classmethod
    def setUpClass(cls):
        """Создаёт единственное приложение Qt для всех окон теста."""
        from chart_delta_bars import create_application
        cls.app = create_application()

    def test_two_windows_update_without_replacing_canvas_or_history_view(self):
        """Два окна имеют свои свечи, текущий бар и сохранённый диапазон истории."""
        self.assertIsNotNone(importlib.util.find_spec("source.realtime_chart"), "Живое окно ещё не реализовано")
        from source.realtime_chart import RealtimeWindow
        fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        sessions = LiveSessionTests()
        for symbol in ("RTS", "MIX"):
            window = RealtimeWindow(symbol, db_path=fixture.db, auto_load=False)
            self.addCleanup(window.close)
            session = sessions.session(symbol)
            window.accept_session(session)
            window.show()
            self.app.processEvents()
            canvas = window.canvas
            window.follow_check.setChecked(False)
            before = canvas.price_axis.vb.viewRange()[0]
            code = "RIH6" if symbol == "RTS" else "MXH6"
            session.ingest([Trade("2026-01-07", 1, "1", "SPBFUT", code, 100, 2)])
            window.accept_session(session)
            self.assertIs(window.canvas, canvas)
            self.assertEqual(len(canvas.candles.datasrc.df), len(session.data))
            np.testing.assert_allclose(canvas.price_axis.vb.viewRange()[0], before)
            self.assertFalse(window.symbol_combo.isEnabled())
            self.assertIn(symbol, window.windowTitle())

    def test_finished_workers_are_released(self):
        """Завершённые задачи не накапливают QThread в непрерывно работающем окне."""
        from PyQt6 import QtCore, QtTest
        from source.realtime_chart import RealtimeWindow, RealtimeWorker
        fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        window = RealtimeWindow("RTS", db_path=fixture.db, auto_load=False,
                                config_path=fixture.db.parent / "missing.json")
        self.addCleanup(window.close)
        window.start_load()
        for _ in range(100):
            self.app.processEvents()
            if not window.loading:
                break
            QtTest.QTest.qWait(10)
        self.assertFalse(window.loading)
        self.assertIsNone(window.worker)
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.assertEqual(window.findChildren(RealtimeWorker), [])


if __name__ == "__main__":
    unittest.main()
