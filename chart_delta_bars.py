"""Интерактивный график дельта-баров RTS/MIX из SQLite на finplot и PyQt6.

Примеры запуска из папки проекта:
    .\.venv\Scripts\python.exe chart_delta_bars.py
    .\.venv\Scripts\python.exe chart_delta_bars.py --symbol MIX

    python chart_delta_bars.py
    python chart_delta_bars.py --symbol MIX --start 2026-09-01 --end 2026-09-28
    python chart_delta_bars.py --symbol RTS --db C:\\data_quote\\RTS_delta_bars.sqlite3
    python chart_delta_bars.py --symbol MIX --db C:\\data_quote\\MIX_delta_bars.sqlite3
Зависимости: python -m pip install -r requirements-chart.txt
Базы открываются только для чтения; индикаторы рассчитываются при просмотре.
"""

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import finplot as fplt

from chart_data import SIGNALS, inspect_database, load_bars

DEFAULT_DATA_DIR = Path(r"C:\data_quote")


def configure_plot():
    """Настраивает светлую схему, свечи и легенду в стиле исходного образца."""
    fplt.background = "#ffffff"
    fplt.foreground = "#334155"
    fplt.odd_plot_background = "#f8fafc"
    fplt.candle_bull_color = "#13a89e"
    fplt.candle_bull_body_color = "#ffffff"
    fplt.candle_bear_color = "#ef5350"
    fplt.candle_bear_body_color = "#ef5350"
    fplt.volume_bull_color = "#9bd7d1"
    fplt.volume_bull_body_color = "#9bd7d1"
    fplt.volume_bear_color = "#f6b0ae"
    fplt.legend_fill_color = "#ffffffe0"
    fplt.legend_border_color = "#cbd5e1"
    fplt.legend_text_color = "#334155"
    fplt.cross_hair_color = "#64748b"
    fplt.grid_alpha = 0.15
    fplt.key_esc_close = False
    fplt.autoviewrestore(False)
    pg.setConfigOptions(background=fplt.background, foreground=fplt.foreground)


class BarTimeAxis(pg.AxisItem):
    """Подписывает равномерную ось номеров баров реальным московским временем."""

    def __init__(self, data):
        """Сохраняет ссылку на отображаемые бары и задаёт высоту двухстрочной подписи."""
        super().__init__(orientation="bottom")
        self.data = data
        self.setHeight(46)

    def tickStrings(self, values, scale, spacing):
        """Преобразует номера баров в дату и время, не объединяя одинаковые метки."""
        labels = []
        for value in values:
            index = int(round(value))
            if 0 <= index < len(self.data):
                stamp = self.data.iloc[index].start_time
                labels.append(stamp.strftime("%d.%m\n%H:%M:%S" if spacing < 3 else "%d.%m\n%H:%M"))
            else:
                labels.append("")
        return labels


class ChartCanvas(pg.GraphicsLayoutWidget):
    """Встраивает свечи finplot, объёмы, сигналы и связанную панель длительности."""

    def __init__(self, owner, data, alpha, offset):
        """Создаёт оси и графические объекты для одной успешно загруженной выборки."""
        super().__init__(parent=owner)
        self.owner, self.data = owner, data
        self.title = "Дельта-бары"
        self.show_maximized = False
        self._axes = []
        self.price_axis, self.duration_axis = fplt.create_plot_widget(self, rows=2, init_zoom_periods=200)
        self._axes = [self.price_axis, self.duration_axis]
        self.addItem(self.price_axis, row=0, col=0)
        self.addItem(self.duration_axis, row=1, col=0)
        self.ci.layout.setRowStretchFactor(0, 5)
        self.ci.layout.setRowStretchFactor(1, 1)
        self.ci.setContentsMargins(0, 0, 0, 0)
        for axis in self._axes:
            axis.setAxisItems({"bottom": BarTimeAxis(data)})
            axis.set_visible(xgrid=True, ygrid=True, xaxis=True)
            axis.getAxis("right").setWidth(90)
            fplt.add_crosshair_info(self.crosshair_info, ax=axis)
        self.duration_axis.setLabel("right", "мин")
        self.candles = fplt.candlestick_ochl(data[["x", "open", "close", "high", "low"]], ax=self.price_axis)
        self.volume_axis = self.price_axis.overlay(scale=0.20)
        self.volume_item = fplt.volume_ocv(data[["x", "open", "close", "volume"]], ax=self.volume_axis)
        self.alf_item = fplt.plot(data.x, data.alf.rename("ALF"), color="#2586ba", width=1.2,
                                  ax=self.price_axis, legend=f"ALF · α={alpha:.2f}")
        self.signal_items = {}
        labels = {"long_rising": "Long · объём ↑", "short_rising": "Short · объём ↑",
                  "long_falling": "Long · объём ↓", "short_falling": "Short · объём ↓"}
        for name in SIGNALS:
            mask = data[name].to_numpy()
            item = None
            if mask.any():
                positions = data.low - offset if name.startswith("long") else data.high + offset
                values = positions.where(mask).rename(name)
                color = "#254bff" if name.endswith("rising") else "#087c29"
                item = fplt.plot(data.x, values, style="o", color=color, width=0.85,
                                 ax=self.price_axis, legend=labels[name])
            self.signal_items[name] = item
        self.partial_item = None
        partial = data.is_complete == 0
        if partial.any():
            self.partial_item = fplt.plot(data.x, (data.high + offset * 0.5).where(partial).rename("partial"),
                                          style="^", color="#d99519", width=0.8, ax=self.price_axis,
                                          legend="Неполный бар")
        self.day_lines = []
        day_starts = np.flatnonzero(data.day.to_numpy()[1:] != data.day.to_numpy()[:-1]) + 1
        for index in day_starts:
            for axis in self._axes:
                line = pg.InfiniteLine(pos=float(index) - 0.5, angle=90,
                                       pen=pg.mkPen("#94a3b8", width=1, style=QtCore.Qt.PenStyle.DashLine))
                axis.addItem(line, ignoreBounds=True)
                self.day_lines.append(line)
        minutes = (data.duration_seconds / 60).rename("duration_minutes")
        self.duration_item = fplt.plot(data.x, minutes, color="#64748b", width=1,
                                       ax=self.duration_axis, legend="Длительность бара, мин")
        fplt.refresh()

    @property
    def axs(self):
        """Предоставляет finplot список осей встроенного графика."""
        return self._axes

    def set_visibility(self, alf, signals, volume, duration, days, partial):
        """Показывает выбранные элементы без перечитывания или пересчёта истории."""
        self.alf_item.setVisible(alf)
        for item in self.signal_items.values():
            if item is not None:
                item.setVisible(signals)
        self.volume_axis.vb.setVisible(volume)
        self.duration_axis.setVisible(duration)
        self.duration_axis.setMinimumHeight(110 if duration else 0)
        self.duration_axis.setMaximumHeight(16777215 if duration else 0)
        self.ci.layout.setRowStretchFactor(1, 1 if duration else 0)
        self.ci.layout.setRowMinimumHeight(1, 110 if duration else 0)
        self.ci.layout.setRowMaximumHeight(1, 16777215 if duration else 0)
        self.price_axis.getAxis("bottom").setStyle(showValues=not duration)
        self.price_axis.getAxis("bottom").setHeight(12 if duration else 46)
        for line in self.day_lines:
            line.setVisible(days)
        if self.partial_item is not None:
            self.partial_item.setVisible(partial)
        self.ci.layout.invalidate()

    def describe_bar(self, index):
        """Выводит сведения о баре с исходной точностью времени и его длительностью."""
        if not 0 <= index < len(self.data):
            return
        bar = self.data.iloc[index]
        start = bar.start_time.isoformat(sep=" ")
        end = bar.end_time.isoformat(sep=" ")
        kind = "полный" if bar.is_complete else "неполный · конец дня"
        self.owner.details.setText(
            f"{bar.symbol} · {start} → {end} МСК · {kind}\n"
            f"O {bar.open:g}   H {bar.high:g}   L {bar.low:g}   C {bar.close:g}    "
            f"Объём {bar.volume:,}    Дельта {bar.delta:+,} / порог {bar.threshold:,}    "
            f"Сделок {bar.tick_count:,}    Длительность {bar.duration_seconds:.3f} с".replace(",", " "))

    def crosshair_info(self, x, y, xtext, ytext):
        """Заменяет числовую подпись курсора реальным временем и обновляет карточку бара."""
        index = int(round(x))
        if 0 <= index < len(self.data):
            self.describe_bar(index)
            xtext = self.data.iloc[index].start_time.strftime("%d.%m.%Y %H:%M:%S") + " МСК"
        return xtext, ytext

    def dispose(self):
        """Удаляет регистрации finplot перед заменой собственного встроенного графика."""
        for axis in list(fplt.overlay_axs):
            if axis.vb.win is self:
                fplt.overlay_axs.remove(axis)
        if self in fplt.windows:
            fplt.windows.remove(self)
        fplt.master_data.pop(self, None)
        if fplt.last_ax in self._axes:
            fplt.last_ax = None
        self.setParent(None)
        self.deleteLater()


class LoadWorker(QtCore.QThread):
    """Выполняет чтение SQLite и расчёт индикаторов вне потока интерфейса."""

    loaded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, request, parent):
        """Сохраняет неизменяемые параметры одной загрузки."""
        super().__init__(parent)
        self.request = request

    def run(self):
        """Возвращает рассчитанные бары или понятную причину неудачной загрузки."""
        try:
            data = load_bars(**self.request)
            self.loaded.emit(data)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ChartWindow(QtWidgets.QMainWindow):
    """Главное окно выбора базы, инструмента, диапазона и индикаторов."""

    def __init__(self, symbol="RTS", data_dir=DEFAULT_DATA_DIR, db_path=None, start=None, end=None, auto_load=True):
        """Создаёт элементы управления и запускает загрузку выбранного участка истории."""
        super().__init__()
        configure_plot()
        self.data_dir = Path(data_dir)
        self.fixed_db = Path(db_path) if db_path else None
        self.data = pd.DataFrame()
        self.canvas = None
        self.worker = None
        self.loading = False
        self._closing = False
        self.offsets = {"RTS": 40.0, "MIX": 100.0}
        self.last_symbol = symbol
        self.setWindowTitle("Дельта-бары · визуальный анализ")
        self.resize(1500, 920)
        self.setMinimumSize(1080, 650)
        self._build_controls(symbol)
        self._connect_controls()
        self.reload_catalog(initial=True, start=start, end=end)
        if auto_load and self.dataset_combo.count():
            self.start_load()

    def _build_controls(self, symbol):
        """Размещает выбор инструмента, дат, индикаторов и область графика."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 10, 12, 8)
        layout.setSpacing(8)
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Дельта-бары")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #16324f;")
        top.addWidget(title)
        self.symbol_combo = QtWidgets.QComboBox()
        self.symbol_combo.addItems(["RTS", "MIX"])
        self.symbol_combo.setCurrentText(symbol)
        self.symbol_combo.setMinimumWidth(75)
        top.addWidget(self.symbol_combo)
        self.start_edit, self.end_edit = QtWidgets.QDateEdit(), QtWidgets.QDateEdit()
        for label, edit in (("С", self.start_edit), ("По", self.end_edit)):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("dd.MM.yyyy")
            edit.setDate(QtCore.QDate.currentDate())
            top.addWidget(QtWidgets.QLabel(label))
            top.addWidget(edit)
        self.load_button = QtWidgets.QPushButton("Показать")
        self.load_button.setStyleSheet("QPushButton {background: #176b91; color: white; padding: 6px 16px; border-radius: 4px;} QPushButton:disabled {background: #94a3b8;}")
        self.refresh_button = QtWidgets.QPushButton("Обновить")
        self.refresh_button.setToolTip("Перечитать базу после дозаписи · Ctrl+R")
        self.all_button = QtWidgets.QPushButton("Весь диапазон")
        self.file_button = QtWidgets.QPushButton("База…")
        for button in (self.load_button, self.refresh_button, self.all_button, self.file_button):
            top.addWidget(button)
        top.addStretch()
        layout.addLayout(top)
        source = QtWidgets.QHBoxLayout()
        self.db_label = QtWidgets.QLabel()
        self.db_label.setStyleSheet("color: #64748b;")
        source.addWidget(self.db_label)
        source.addStretch()
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.setToolTip("Набор настроек построения баров")
        source.addWidget(self.dataset_combo)
        layout.addLayout(source)
        options = QtWidgets.QHBoxLayout()
        self.alf_check = QtWidgets.QCheckBox("ALF")
        self.alpha_spin = QtWidgets.QDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 1)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.4)
        self.alpha_spin.setPrefix("α ")
        self.signals_check = QtWidgets.QCheckBox("Volume Stops")
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(0, 100000)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setValue(self.offsets[symbol])
        self.offset_spin.setPrefix("Отступ ")
        self.offset_spin.setToolTip("Отступ сигнальных точек от high/low в пунктах цены")
        self.volume_check = QtWidgets.QCheckBox("Объём")
        self.days_check = QtWidgets.QCheckBox("Границы дней")
        self.partial_check = QtWidgets.QCheckBox("Неполные: метки")
        self.duration_check = QtWidgets.QCheckBox("Длительность")
        self.checkboxes = [self.alf_check, self.signals_check, self.volume_check,
                           self.days_check, self.partial_check, self.duration_check]
        for checkbox in self.checkboxes[:-1]:
            checkbox.setChecked(True)
        for widget in (self.alf_check, self.alpha_spin, self.signals_check, self.offset_spin,
                       self.volume_check, self.days_check, self.partial_check, self.duration_check):
            options.addWidget(widget)
        options.addStretch()
        layout.addLayout(options)
        self.chart_layout = QtWidgets.QVBoxLayout()
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.chart_layout, 1)
        self.placeholder = QtWidgets.QLabel("Выберите инструмент и период")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("background: white; color: #64748b; font-size: 16px;")
        self.chart_layout.addWidget(self.placeholder)
        self.details = QtWidgets.QLabel("Наведите курсор на свечу для просмотра параметров бара")
        self.details.setMinimumHeight(48)
        self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setStyleSheet("background: #eef3f8; color: #243c54; padding: 6px 10px; border-radius: 4px;")
        layout.addWidget(self.details)
        self.status_label = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(QtWidgets.QLabel("Время: МСК · Равный шаг баров"))

    def _connect_controls(self):
        """Подключает кнопки и переключатели, не меняя состояние исходной базы."""
        self.load_button.clicked.connect(self.start_load)
        self.refresh_button.clicked.connect(self.refresh_data)
        self.file_button.clicked.connect(self.choose_database)
        self.all_button.clicked.connect(self.show_all)
        self.symbol_combo.currentTextChanged.connect(self.change_symbol)
        self.dataset_combo.currentIndexChanged.connect(self.start_load)
        self.alpha_spin.editingFinished.connect(self.start_load)
        self.offset_spin.editingFinished.connect(self.redraw)
        for checkbox in self.checkboxes:
            checkbox.toggled.connect(self.apply_visibility)
        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        shortcut.activated.connect(self.refresh_data)

    def database_path(self):
        """Определяет базу выбранного тикера или явно указанный общий файл."""
        return self.fixed_db or self.data_dir / f"{self.symbol_combo.currentText()}_delta_bars.sqlite3"

    def reload_catalog(self, initial=False, start=None, end=None):
        """Обновляет доступные наборы, сохраняя выбор и даты при повторном чтении."""
        path = self.database_path()
        self.db_label.setText(str(path))
        self.db_label.setToolTip(str(path))
        previous_id = self.dataset_combo.currentData()
        previous_last = getattr(self, "last_available_day", None)
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        try:
            datasets = inspect_database(path, self.symbol_combo.currentText())
            if not datasets:
                raise ValueError("В этой базе нет баров выбранного инструмента")
            for info in datasets:
                self.dataset_combo.addItem(info.label, info.dataset_id)
            old_index = self.dataset_combo.findData(previous_id)
            if old_index >= 0:
                self.dataset_combo.setCurrentIndex(old_index)
            info = datasets[self.dataset_combo.currentIndex()]
            self.last_available_day = info.last_day
            if initial:
                start_day = start or max(date.fromisoformat(info.first_day), date.fromisoformat(info.last_day) - timedelta(days=29)).isoformat()
                self.start_edit.setDate(QtCore.QDate.fromString(start_day, "yyyy-MM-dd"))
                self.end_edit.setDate(QtCore.QDate.fromString(end or info.last_day, "yyyy-MM-dd"))
            elif previous_last and self.end_edit.date().toString("yyyy-MM-dd") == previous_last:
                self.end_edit.setDate(QtCore.QDate.fromString(info.last_day, "yyyy-MM-dd"))
            self.dataset_combo.setVisible(len(datasets) > 1)
            self.status_label.setText(f"Доступно {info.bar_count:,} баров · {info.first_day} — {info.last_day}".replace(",", " "))
            return True
        except Exception as exc:
            self.status_label.setText(f"Ошибка чтения базы: {exc}")
            return False
        finally:
            self.dataset_combo.blockSignals(False)

    def set_busy(self, busy):
        """Блокирует изменение параметров текущего запроса, оставляя навигацию графика доступной."""
        self.loading = busy
        for widget in (self.symbol_combo, self.dataset_combo, self.start_edit, self.end_edit,
                       self.load_button, self.refresh_button, self.file_button, self.alpha_spin, self.offset_spin):
            widget.setEnabled(not busy)

    def start_load(self, *args):
        """Запускает один фоновый запрос по текущим параметрам без повторного входа."""
        if self.loading or not self.dataset_combo.count():
            return
        request = {"path": self.database_path(), "symbol": self.symbol_combo.currentText(),
                   "dataset_id": self.dataset_combo.currentData(),
                   "start": self.start_edit.date().toString("yyyy-MM-dd"),
                   "end": self.end_edit.date().toString("yyyy-MM-dd"), "alpha": self.alpha_spin.value()}
        self.set_busy(True)
        self.status_label.setText("Загрузка баров и расчёт индикаторов с предысторией…")
        self.worker = LoadWorker(request, self)
        self.worker.loaded.connect(self.accept_data)
        self.worker.failed.connect(self.load_failed)
        self.worker.finished.connect(self.load_finished)
        self.worker.start()

    def load_finished(self):
        """Возвращает доступ к управлению после завершения фонового чтения."""
        self.set_busy(False)

    def load_failed(self, message):
        """Сообщает об ошибке, сохраняя последний успешно построенный график."""
        self.status_label.setText(f"Ошибка: {message}. Предыдущий график сохранён.")

    def accept_data(self, data):
        """Принимает результат фонового чтения и обновляет график и сведения о диапазоне."""
        if self._closing:
            return
        self.data = data
        try:
            self.redraw()
        except Exception as exc:
            self.status_label.setText(f"Не удалось построить график: {exc}")
            return
        if data.empty:
            self.status_label.setText("Нет баров в выбранном диапазоне")
            self.details.setText("Измените даты или выберите другой инструмент")
            return
        self.setWindowTitle(f"{data.symbol.iloc[0]} · Дельта-бары · ALF и Volume Stops")
        partial = int((data.is_complete == 0).sum())
        self.status_label.setText(f"{len(data):,} баров · {data.day.iloc[0]} — {data.day.iloc[-1]} · неполных {partial} · "
                                 "Колесо: масштаб · Перетаскивание: история".replace(",", " "))
        self.canvas.describe_bar(len(data) - 1)

    def redraw(self):
        """Перестраивает графические объекты по уже рассчитанным данным и отступу точек."""
        self.offsets[self.symbol_combo.currentText()] = self.offset_spin.value()
        if self.canvas is not None:
            self.chart_layout.removeWidget(self.canvas)
            self.canvas.dispose()
            self.canvas = None
        self.placeholder.setVisible(self.data.empty)
        if self.data.empty:
            self.placeholder.setText("Нет баров в выбранном диапазоне")
            return
        self.canvas = ChartCanvas(self, self.data, self.alpha_spin.value(), self.offset_spin.value())
        self.chart_layout.addWidget(self.canvas)
        self.apply_visibility()

    def apply_visibility(self, *args):
        """Применяет флажки индикаторов и вспомогательных отметок."""
        if self.canvas:
            self.canvas.set_visibility(self.alf_check.isChecked(), self.signals_check.isChecked(),
                                       self.volume_check.isChecked(), self.duration_check.isChecked(),
                                       self.days_check.isChecked(), self.partial_check.isChecked())

    def change_symbol(self, symbol):
        """Переключает тикер, сохраняя даты и индивидуальный отступ сигнальных точек."""
        self.offsets[self.last_symbol] = self.offset_spin.value()
        self.last_symbol = symbol
        self.offset_spin.setValue(self.offsets[symbol])
        if self.reload_catalog():
            self.start_load()

    def refresh_data(self):
        """Перечитывает каталог и бары после завершения очередной дозаписи конвертером."""
        if not self.loading and self.reload_catalog():
            self.start_load()

    def choose_database(self):
        """Позволяет пользователю указать SQLite вместо стандартных файлов инструментов."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать базу дельта-баров", str(self.database_path().parent),
                                                      "SQLite (*.sqlite3 *.sqlite *.db);;Все файлы (*)")
        if path:
            self.fixed_db = Path(path)
            if self.reload_catalog(initial=True):
                self.start_load()

    def show_all(self):
        """Масштабирует график на все загруженные бары выбранного диапазона."""
        if self.canvas is not None and not self.data.empty:
            fplt.set_x_pos(-0.5, len(self.data) - 0.5, ax=self.canvas.price_axis)

    def closeEvent(self, event):
        """Закрывает собственный поток и освобождает регистрацию графика перед выходом."""
        self._closing = True
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait()
        if self.canvas is not None:
            self.canvas.dispose()
            self.canvas = None
        super().closeEvent(event)


def create_application(argv=None):
    """Создаёт светлое приложение Qt с читаемыми русскими подписями."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or [])
    app.setStyle("Fusion")
    # Offscreen-платформа Qt в Windows не всегда перечисляет системные шрифты.
    if "Segoe UI" not in QtGui.QFontDatabase.families():
        font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf"
        if font_path.is_file():
            QtGui.QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QtGui.QFont("Segoe UI", 9))
    palette = app.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#f8fafc"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#25364b"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("white"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#25364b"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#f1f5f9"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#25364b"))
    app.setPalette(palette)
    return app


def valid_date(value):
    """Проверяет дату командной строки и возвращает канонический формат ISO."""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Ожидается дата YYYY-MM-DD") from exc


def main(argv=None):
    """Разбирает параметры и запускает интерактивное окно анализа дельта-баров."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", choices=("RTS", "MIX"), default="RTS", help="первоначальный инструмент")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="папка двух баз инструментов")
    parser.add_argument("--db", type=Path, help="явно выбранная база вместо стандартных файлов")
    parser.add_argument("--start", type=valid_date, help="начальная дата; по умолчанию последние 30 календарных дней")
    parser.add_argument("--end", type=valid_date, help="конечная дата; по умолчанию последний день базы")
    args = parser.parse_args(argv)
    app = create_application()
    window = ChartWindow(args.symbol, args.data_dir, args.db, args.start, args.end)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
