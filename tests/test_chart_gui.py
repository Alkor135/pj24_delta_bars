"""Проверяет собственное окно просмотрщика без управления другими приложениями.

Запуск: python -m unittest -v tests.test_chart_gui
Qt работает в режиме offscreen; снимок сохраняется только при отдельной проверке.
"""

import importlib.util
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
import subprocess
import sys

from PyQt6 import QtCore, QtWidgets, QtTest

from tests import test_chart_data


class WindowTests(unittest.TestCase):
    """Проверяет выбор диапазона, видимость индикаторов и обработку пустых результатов."""

    @classmethod
    def setUpClass(cls):
        """Создаёт единственное приложение Qt для всех проверок окна."""
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        """Подготавливает небольшой независимый источник и окно без автозагрузки."""
        self.fixture = test_chart_data.DatabaseTests(methodName="test_dataset_selection_and_duplicate_timestamps")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def wait_loaded(self, window):
        """Ожидает окончания фоновой загрузки, обрабатывая события собственного окна."""
        for _ in range(200):
            self.app.processEvents()
            if not window.loading:
                return
            QtTest.QTest.qWait(20)
        self.fail("Загрузка окна не завершилась за 4 секунды")

    def test_gui_module_available(self):
        """Требует наличия точки входа графического просмотрщика."""
        self.assertIsNotNone(importlib.util.find_spec("chart_delta_bars"))

    def test_finplot_import_with_russian_locale(self):
        """Регрессия: импорт при ru_RU не должен завершать процесс через os._exit."""
        code = "import locale; locale.getdefaultlocale=lambda:('ru_RU','cp1251'); import finplot; print('import-ok')"
        process = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("import-ok", process.stdout)

    def test_controls_and_numeric_axis(self):
        """Сохраняет восемь свечей с дубликатами времени и переключает панели."""
        from chart_delta_bars import ChartWindow
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-05", end="2026-01-06")
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        self.assertEqual(len(window.data), 8)
        self.assertEqual(window.data.x.tolist(), list(range(8)))
        self.assertEqual(len(window.canvas.candles.datasrc.df), 8)
        self.assertEqual(window.canvas.price_axis.getAxis('bottom').tickStrings([0, 4], 1, 4),
                         ['05.01\n10:00', '06.01\n10:00'])
        window.alf_check.setChecked(False)
        self.assertFalse(window.canvas.alf_item.isVisible())
        window.duration_check.setChecked(True)
        self.assertTrue(window.canvas.duration_axis.isVisible())
        window.canvas.describe_bar(0)
        self.assertIn("Дельта", window.details.text())
        self.assertIn("10:00:00.000000001", window.details.text())
        window.refresh_button.click()
        self.wait_loaded(window)
        self.assertEqual(len(window.data), 8)

    def test_date_filter_and_empty_range(self):
        """Меняет даты и корректно показывает отсутствие баров без старого графика."""
        from chart_delta_bars import ChartWindow
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-06", end="2026-01-06")
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        self.assertEqual(len(window.data), 4)
        window.start_edit.setDate(QtCore.QDate(2026, 2, 1))
        window.end_edit.setDate(QtCore.QDate(2026, 2, 2))
        window.load_button.click()
        self.wait_loaded(window)
        self.assertTrue(window.data.empty)
        self.assertIn("Нет баров", window.status_label.text())

    def test_invalid_range_keeps_previous_chart(self):
        """Ошибка в датах не уничтожает предыдущую успешную выборку."""
        from chart_delta_bars import ChartWindow
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-05", end="2026-01-06")
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        canvas = window.canvas
        window.start_edit.setDate(QtCore.QDate(2026, 2, 2))
        window.end_edit.setDate(QtCore.QDate(2026, 2, 1))
        window.load_button.click()
        self.wait_loaded(window)
        self.assertIs(window.canvas, canvas)
        self.assertEqual(len(window.data), 8)
        self.assertIn("Ошибка", window.status_label.text())

    def test_show_all_includes_first_and_last_candle(self):
        """Кнопка всего диапазона оставляет обе крайние свечи внутри видимой области."""
        from chart_delta_bars import ChartWindow
        with closing(sqlite3.connect(self.fixture.db)) as connection:
            for dataset in ("a", "b"):
                template = list(connection.execute("SELECT * FROM bars WHERE dataset_id=? LIMIT 1", (dataset,)).fetchone())
                for index in range(4, 40):
                    template[3] = index
                    connection.execute("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", template)
            connection.commit()
        window = ChartWindow(db_path=self.fixture.db, start="2026-01-05", end="2026-01-06")
        self.addCleanup(window.close)
        window.show()
        self.wait_loaded(window)
        window.all_button.click()
        self.app.processEvents()
        left, right = window.canvas.price_axis.vb.viewRange()[0]
        self.assertLessEqual(left, -0.3)
        self.assertGreaterEqual(right, len(window.data) - 0.7)


if __name__ == "__main__":
    unittest.main()
