r"""Воспроизводит реальные дельта-бары SQLite с Семафором и островами SMA.

Примеры запуска из папки проекта:
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_simulate.py
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_simulate.py --symbol MIX
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_simulate.py --interval 0.1 --initial-bars 50
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_simulate.py --db C:\data_quote\delta_bars.sqlite3 --start 2026-09-01 --end 2026-09-30
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_simulate.py --preset fxi --depths 0 12 34
    .\.venv\Scripts\python.exe -m unittest -v tests.test_chart_semafor_simulate

Зависимости: python -m pip install -r requirements-chart.txt

Из выбранной базы один раз читается снимок истории до конечной даты. Сначала
показываются 200 баров выбранного диапазона, далее таймер добавляет один бар
каждую секунду. Если диапазон короче, последний бар оставляется для таймера.
История до начальной даты прогревает индикаторы. Будущие бары снимка не
участвуют в расчёте Семафора/SMA до своего появления на графике.

Скорость меняется на ходу полем «Интервал»: 0.05…60 секунд на бар. Пауза,
один шаг и перезапуск доступны в окне; --initial-bars задаёт стартовое число
видимых баров (1…1000000). Смена базы, дат, набора или настроек индикаторов
начинает воспроизведение заново. После последнего бара таймер останавливается.

Свечи и индикаторы обновляются через finplot.update_data/update_gfx на том же
холсте. «Следить за последним» сдвигает текущий масштаб к новому бару; отключите
флажок для просмотра истории. Расчёты идут в потоке Qt; если они занимают дольше
интервала, следующий бар ждёт окончания предыдущего расчёта и не пропускается.

Последние метки Семафора могут перемещаться и исчезать, как в исходном скрипте.
Это ускоренное воспроизведение готовых баров с исходным временем МСК и реальной
длительностью, без моделирования тиков внутри свечи. SQLite только читается.
"""

import argparse
from datetime import date
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import finplot as fplt

from chart_delta_bar_semafor import (
    DEFAULT_DATA_DIR, DEFAULT_DEPTHS, PRESETS, ChartCanvas as SemaforCanvas,
    ChartWindow as SemaforWindow, calculate_indicators, load_semafor_bars,
    sun_symbol, validate_settings,
)
from chart_delta_bars import LoadWorker as BaseLoadWorker, create_application, valid_date
from source.chart_data import SELECT_COLUMNS


def validate_simulation(initial_bars, interval):
    """Возвращает проверенные initial_bars и interval либо возбуждает ValueError.

    initial_bars — целое число стартовых баров 1…1000000; interval — секунды
    на бар 0.05…60 с точностью до сотых, как в поле управления скоростью.
    """
    if (isinstance(initial_bars, bool) or not isinstance(initial_bars, Integral)
            or not 1 <= initial_bars <= 1000000):
        raise ValueError("Начальное число баров должно быть целым от 1 до 1000000")
    try:
        interval = float(interval)
    except (TypeError, ValueError) as exc:
        raise ValueError("Интервал должен быть числом секунд") from exc
    if (not np.isfinite(interval) or not 0.05 <= interval <= 60
            or not np.isclose(interval, round(interval, 2), rtol=0, atol=1e-12)):
        raise ValueError("Интервал должен быть от 0.05 до 60 секунд, не более двух знаков после запятой")
    return int(initial_bars), interval


class ReplaySession:
    """Хранит снимок исходных баров и рассчитывает только уже поступивший префикс."""

    def __init__(self, history, start, initial_bars=200, depths=DEFAULT_DEPTHS,
                 deviation=1, backstep=1, point=1, ma_fast=3, ma_slow=34):
        """Задаёт history, дату start и начальное число баров; прогревает СФ/SMA.

        history — проверенные хронологические OHLC с day, включая предысторию;
        initial_bars — стартовое число видимых баров. Остальные параметры
        соответствуют calculate_indicators. Последний бар короткого диапазона
        оставляется для следующего шага; единственный бар показывается сразу.
        """
        initial_bars, _ = validate_simulation(initial_bars, 1)
        start = date.fromisoformat(str(start)).isoformat()
        self.history = history.copy().reset_index(drop=True)
        self.settings = validate_settings(depths, deviation, backstep, point, ma_fast, ma_slow)
        visible = np.flatnonzero(self.history.day.to_numpy() >= start)
        self.start_index = int(visible[0]) if len(visible) else len(self.history)
        self.total = len(self.history) - self.start_index
        self.shown = min(initial_bars, max(1, self.total - 1)) if self.total else 0
        self.data = self._calculate(self.shown)

    @property
    def finished(self):
        """Возвращает True, когда все бары выбранного диапазона уже показаны."""
        return self.shown >= self.total

    def _calculate(self, shown):
        """Возвращает видимый срез для shown баров, прогревая его только прошлым."""
        prefix = self.history.iloc[:self.start_index + shown]
        calculated = calculate_indicators(prefix, **self.settings)
        data = calculated.iloc[self.start_index:].copy().reset_index(drop=True)
        data["x"] = np.arange(len(data), dtype=np.int64)
        return data

    def frame(self):
        """Возвращает текущую рассчитанную таблицу без изменения позиции потока."""
        return self.data

    def advance(self):
        """Добавляет ровно один исходный бар и возвращает заново рассчитанный срез.

        Пересчёт всего поступившего префикса позволяет Семафору удалить прежние
        экстремумы. При ошибке позиция сохраняется; после конца новых баров нет.
        """
        if not self.finished:
            data = self._calculate(self.shown + 1)
            self.shown += 1
            self.data = data
        return self.data


def load_replay_session(path, symbol, dataset_id, start, end, initial_bars=200, **settings):
    """Возвращает ReplaySession для снимка SQLite с предысторией до start.

    path/symbol/dataset_id — источник; start/end — включительные даты ISO;
    initial_bars — стартовое число видимых баров; settings — параметры СФ/SMA.
    После end строки не читаются. Исходный загрузчик проверяет OHLC, времена
    и числовые поля; его индикаторы отключены и отброшены до создания потока.
    """
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if start > end:
        raise ValueError("Начальная дата должна быть не позже конечной")
    history = load_semafor_bars(path, symbol, dataset_id, date.min.isoformat(), end.isoformat(),
                                depths=(0, 0, 0), ma_fast=1, ma_slow=1)
    # В пустом результате загрузчика временные поля ещё не преобразованы.
    columns = SELECT_COLUMNS.split(",")
    if history.empty:
        history = pd.DataFrame(columns=columns + ["duration_seconds"])
    else:
        history = history[columns + ["duration_seconds"]].copy()
    history.attrs.clear()
    return ReplaySession(history, start.isoformat(), initial_bars, **settings)


class ChartCanvas(SemaforCanvas):
    """Обновляет исходный график Семафора без замены холста и потери навигации."""

    def __init__(self, owner, data, offset, preset):
        """Создаёт график data и регистрирует все уровни, в том числе пока пустые.

        owner — окно; offset — декоративный отступ; preset — стиль NUF/FXi.
        Пустые метки нужны для появления первых экстремумов в ходе симуляции.
        """
        super().__init__(owner, data, offset, preset)
        self.offset, self.preset = offset, preset
        self.marker_items = {}
        for level, depth in enumerate(data.attrs["depths"], 1):
            if not depth:
                continue
            for side in ("low", "high"):
                name = f"sf{level}_{side}"
                items = [item for item in self.semafor_items[level - 1]
                         if str(item.datasrc.df.columns[item.datasrc.col_data_offset]).startswith(name + "_")]
                self.marker_items[name] = items or self._create_markers(level, side)
        if self.partial_item is None:
            self.partial_item = fplt.plot(data.x, self._partial_values(data), style="^",
                                          color="#d99519", width=0.8, ax=self.price_axis,
                                          legend="Неполный бар")

    def _partial_values(self, data):
        """Возвращает цены декоративных меток неполных баров таблицы data."""
        return (data.high + self.offset * 0.5).where(data.is_complete == 0).rename("partial")

    def _create_markers(self, level, side):
        """Создаёт и возвращает элементы level=1…3, side=low/high по стилю пресета."""
        color = "#008820" if side == "low" else "#df2525"
        if self.preset == "fxi" and level == 2:
            color = "#e5b400"
        sign = -1 if side == "low" else 1
        name = f"sf{level}_{side}"
        positions = (self.data[name] + sign * self.offset).rename(name + "_display")
        item = fplt.plot(self.data.x, positions, style="o", color=color,
                         width=(7, 12, 21)[level - 1] / 7, ax=self.price_axis)
        item.setSymbol(sun_symbol() if level == 3 or (self.preset == "nuf" and level == 2) else "o")
        item.setSymbolPen(pg.mkPen(color, width=0.9))
        item.setDownsampling(auto=False)
        item.setZValue(19 + level)
        items = [item]
        if self.preset == "fxi" and level == 3:
            center = fplt.plot(self.data.x, positions.rename(name + "_center"), style="o",
                               color="#ffe33d", width=6 / 7, ax=self.price_axis, zoomscale=False)
            center.setDownsampling(auto=False)
            center.setZValue(24)
            items.append(center)
        self.semafor_items[level - 1].extend(items)
        return items

    def update_frame(self, data, follow=True):
        """Обновляет свечи/СФ/SMA/объём для data; follow сдвигает текущий масштаб.

        Сначала передаются все таблицы finplot, затем обновляется графика,
        включая удалённые экстремумы. Ось времени получает только видимые бары.
        """
        old_range = self.price_axis.vb.viewRange()[0]
        added = len(data) - len(self.data)
        self.data = data
        for axis in self._axes:
            axis.getAxis("bottom").data = data
            axis.getAxis("bottom").picture = None
            axis.getAxis("bottom").update()
        updates = [(self.candles, data[["x", "open", "close", "high", "low"]]),
                   (self.volume_item, data[["x", "open", "close", "volume"]]),
                   (self.duration_item, data[["x", "open", "close"]].assign(
                       duration_minutes=data.duration_seconds / 60))]
        for item, direction in zip(self.island_items[:2], (1, -1)):
            mask = data.island_direction.to_numpy() == direction
            item.setOpts(x=data.x.to_numpy(dtype=float)[mask],
                         y0=data.ma_slow.to_numpy()[mask], y1=data.ma_fast.to_numpy()[mask])
        for item, name in zip(self.island_items[2:], ("ma_fast", "ma_slow")):
            updates.append((item, data[["x", name]]))
        for name, items in self.marker_items.items():
            sign = -1 if name.endswith("low") else 1
            values = data[name] + sign * self.offset
            for item in items:
                column = item.datasrc.df.columns[item.datasrc.col_data_offset]
                updates.append((item, pd.DataFrame({"x": data.x, column: values})))
        updates.append((self.partial_item, pd.DataFrame({"x": data.x, "partial": self._partial_values(data)})))
        for item, values in updates:
            item.update_data(values, gfx=False)
        scatters = {self.partial_item, *(item for items in self.marker_items.values() for item in items)}
        for item, _ in updates:
            if item in scatters:
                # finplot при update_gfx передаёт NaN в ScatterPlotItem. Оставляем
                # полную таблицу для автомасштаба, рисуем только существующие метки.
                values = item.datasrc.y.to_numpy(dtype=float) / self.price_axis.vb.yscale.scalef
                mask = np.isfinite(values)
                item.setData(item.datasrc.index.to_numpy()[mask], values[mask])
            else:
                item.update_gfx()
        # Добавляем границы только поступивших дней, сохраняя прежние линии.
        indices = np.flatnonzero(data.day.to_numpy()[1:] != data.day.to_numpy()[:-1]) + 1
        for index in indices[len(self.day_lines) // len(self._axes):]:
            for axis in self._axes:
                line = pg.InfiniteLine(pos=float(index) - 0.5, angle=90,
                                       pen=pg.mkPen("#94a3b8", width=1, style=QtCore.Qt.PenStyle.DashLine))
                axis.addItem(line, ignoreBounds=True)
                self.day_lines.append(line)
        shift = added if follow else 0
        left, right = old_range[0] + shift, old_range[1] + shift
        self.price_axis.vb.update_y_zoom(left, right)
        # Ограничение минимального числа свечей finplot мешает восстановить
        # ширину короткой истории; диапазон X задаём также непосредственно Qt.
        self.price_axis.vb.setXRange(left, right, padding=0)
        self.owner.apply_visibility()
        self.describe_bar(len(data) - 1)


class ReplayWorker(BaseLoadWorker):
    """Читает снимок или рассчитывает очередной бар в отдельном потоке Qt."""

    def __init__(self, request, parent, session=None):
        """Сохраняет request загрузки либо session следующего шага; parent — окно."""
        super().__init__(request, parent)
        self.session = session

    def run(self):
        """Передаёт готовую сессию через loaded либо русское сообщение через failed."""
        try:
            if self.session is None:
                session = load_replay_session(**self.request)
            else:
                self.session.advance()
                session = self.session
            self.loaded.emit(session)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ChartWindow(SemaforWindow):
    """Сохраняет исходные настройки графика и добавляет управление потоком баров."""

    def __init__(self, symbol="RTS", data_dir=DEFAULT_DATA_DIR, db_path=None,
                 start=None, end=None, auto_load=True, initial_bars=200, interval=1, **settings):
        """Открывает симуляцию выбранной базы с interval секунд на новый бар.

        symbol/data_dir/db_path/start/end/auto_load — источник и даты;
        initial_bars — стартовое число баров; settings — настройки исходного окна.
        """
        self.initial_bars, self.initial_interval = validate_simulation(initial_bars, interval)
        self.session = None
        self.playing = False
        self._load_ok = False
        super().__init__(symbol, data_dir, db_path, start, end, auto_load, **settings)
        self.setWindowTitle(f"{symbol} · Семафор · Симуляция потока баров")

    def _build_controls(self, symbol):
        """Дополняет исходные элементы symbol кнопками и интервалом воспроизведения."""
        super()._build_controls(symbol)
        row = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton("Пауза")
        self.step_button = QtWidgets.QPushButton("Один бар")
        self.restart_button = QtWidgets.QPushButton("Сначала")
        for button in (self.pause_button, self.step_button, self.restart_button):
            row.addWidget(button)
        row.addWidget(QtWidgets.QLabel("Интервал"))
        self.interval_spin = QtWidgets.QDoubleSpinBox()
        self.interval_spin.setRange(0.05, 60)
        self.interval_spin.setDecimals(2)
        self.interval_spin.setSingleStep(0.05)
        self.interval_spin.setValue(self.initial_interval)
        self.interval_spin.setSuffix(" с/бар")
        self.interval_spin.setToolTip("Меньше интервал — быстрее поток. Меняется без перезапуска.")
        row.addWidget(self.interval_spin)
        row.addWidget(QtWidgets.QLabel("Начальных баров"))
        self.initial_spin = QtWidgets.QSpinBox()
        self.initial_spin.setRange(1, 1000000)
        self.initial_spin.setValue(self.initial_bars)
        self.initial_spin.setToolTip("Применяется при загрузке или нажатии «Сначала»")
        row.addWidget(self.initial_spin)
        self.follow_check = QtWidgets.QCheckBox("Следить за последним")
        self.follow_check.setChecked(True)
        row.addWidget(self.follow_check)
        row.addStretch()
        self.centralWidget().layout().insertLayout(4, row)
        self.load_button.setText("Начать")
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.setInterval(round(self.initial_interval * 1000))
        self.update_playback_controls()

    def _connect_controls(self):
        """Подключает существующую навигацию, таймер, паузу, шаг и смену скорости."""
        super()._connect_controls()
        self.timer.timeout.connect(self.next_bar)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.step_button.clicked.connect(self.step_once)
        self.restart_button.clicked.connect(self.start_load)
        self.interval_spin.valueChanged.connect(self.change_interval)

    def change_interval(self, seconds):
        """Меняет интервал таймера на seconds секунд, сохраняя позицию и паузу."""
        self.timer.setInterval(round(seconds * 1000))
        if self.session is not None and not self.loading:
            self.update_status()

    def update_playback_controls(self):
        """Согласует кнопки с загрузкой, наличием данных, паузой и концом потока."""
        available = self.session is not None and not self.session.finished
        self.pause_button.setText("Пауза" if self.playing else "Продолжить")
        self.pause_button.setEnabled(available)
        self.step_button.setEnabled(available and not self.loading)
        self.restart_button.setEnabled(self.session is not None and not self.loading)

    def set_busy(self, busy):
        """Блокирует исходные параметры при busy; смена скорости и пауза доступны."""
        super().set_busy(busy)
        self.initial_spin.setEnabled(not busy)
        self.update_playback_controls()

    def _start_worker(self, request, session=None):
        """Запускает один расчёт request/session и подключает обработчики окна."""
        self._load_ok = False
        self.set_busy(True)
        previous = self.worker
        self.worker = ReplayWorker(request, self, session)
        if previous is not None:
            previous.deleteLater()
        self.worker.loaded.connect(self.accept_session)
        self.worker.failed.connect(self.load_failed)
        self.worker.finished.connect(self.load_finished)
        self.worker.start()

    def start_load(self, *args):
        """Начинает новый снимок с текущими параметрами; args — сигнал Qt."""
        if self.loading or not self.dataset_combo.count():
            return
        self.timer.stop()
        self.playing = False
        request = dict(path=self.database_path(), symbol=self.symbol_combo.currentText(),
                       dataset_id=self.dataset_combo.currentData(),
                       start=self.start_edit.date().toString("yyyy-MM-dd"),
                       end=self.end_edit.date().toString("yyyy-MM-dd"),
                       initial_bars=self.initial_spin.value(),
                       depths=tuple(spin.value() for spin in self.depth_spins),
                       deviation=self.deviation_spin.value(), backstep=self.backstep_spin.value(),
                       point=self.point_spin.value(), ma_fast=self.ma_fast_spin.value(),
                       ma_slow=self.ma_slow_spin.value())
        self.status_label.setText("Чтение снимка баров и прогрев Семафора/SMA…")
        self._start_worker(request)

    def accept_session(self, session):
        """Принимает session, обновляя прежний холст либо создавая новый при загрузке."""
        if self._closing:
            return
        same_session = session is self.session
        try:
            self.data = session.frame()
            if same_session and self.canvas is not None:
                self.canvas.update_frame(self.data, self.follow_check.isChecked())
            else:
                self.redraw()
                self.playing = not session.finished
            self.session = session
            self._load_ok = True
            if session.finished:
                self.playing = False
                self.timer.stop()
            elif self.playing and not self.loading:
                self.timer.start()
            self.setWindowTitle(f"{self.symbol_combo.currentText()} · Семафор · Симуляция потока баров")
            self.update_status()
            self.update_playback_controls()
            if self.canvas is not None:
                self.canvas.describe_bar(len(self.data) - 1)
        except Exception as exc:
            self.load_failed(f"Не удалось обновить график: {exc}")

    def load_finished(self):
        """Освобождает параметры; запускает таймер, если он ещё не работает.

        Уже активный таймер не перезапускается, чтобы время расчёта не
        прибавлялось к каждому интервалу между очередными барами.
        """
        self.set_busy(False)
        if self._load_ok and self.playing and not self._closing and not self.timer.isActive():
            self.timer.start()

    def load_failed(self, message):
        """Останавливает поток при message, оставляя текущий график для просмотра."""
        self.playing = False
        self.timer.stop()
        super().load_failed(message)
        self.update_playback_controls()

    def update_status(self):
        """Показывает прогресс, состояние потока, интервал и время последнего бара."""
        if self.session is None:
            return
        if self.data.empty:
            self.status_label.setText("Нет баров в выбранном диапазоне")
            return
        state = "Завершено" if self.session.finished else ("Воспроизведение" if self.playing else "Пауза")
        count = sum(int(self.data[f"sf{level}_{side}"].notna().sum())
                    for level in (1, 2, 3) for side in ("low", "high"))
        stamp = self.data.end_time.iloc[-1].strftime("%d.%m.%Y %H:%M:%S")
        self.status_label.setText(
            f"{state} · {self.session.shown:,} / {self.session.total:,} баров · "
            f"{self.interval_spin.value():g} с/бар · меток СФ {count} · {stamp} МСК".replace(",", " "))

    def toggle_pause(self):
        """Переключает паузу без изменения текущего бара и состояния индикаторов."""
        if self.session is None or self.session.finished:
            return
        self.playing = not self.playing
        if self.playing and not self.loading:
            self.timer.start()
        else:
            self.timer.stop()
        self.update_playback_controls()
        self.update_status()

    def step_once(self):
        """Ставит поток на паузу и запрашивает ровно один очередной бар."""
        self.playing = False
        self.timer.stop()
        self.next_bar()

    def next_bar(self):
        """Запускает пересчёт следующего бара, исключая наложение фоновых задач.

        Активный таймер продолжает отсчитывать интервалы независимо от времени
        расчёта. Его события во время занятого потока не запускают второй шаг.
        """
        if self._closing or self.loading or self.session is None or self.session.finished:
            return
        self._start_worker({}, self.session)

    def redraw(self):
        """Пересоздаёт холст только при загрузке или изменении отступа меток."""
        self.offsets[self.symbol_combo.currentText()] = self.offset_spin.value()
        old_range = self.canvas.price_axis.vb.viewRange()[0] if self.canvas is not None else None
        if self.canvas is not None:
            self.chart_layout.removeWidget(self.canvas)
            self.canvas.dispose()
            self.canvas = None
        self.placeholder.setVisible(self.data.empty)
        if self.data.empty:
            self.placeholder.setText("Нет баров в выбранном диапазоне")
            return
        self.canvas = ChartCanvas(self, self.data, self.offset_spin.value(), self.preset_combo.currentData())
        self.chart_layout.addWidget(self.canvas)
        self.apply_visibility()
        if old_range is not None:
            fplt.set_x_pos(*old_range, ax=self.canvas.price_axis)

    def closeEvent(self, event):
        """Останавливает таймер и завершает поток перед закрытием окна event."""
        self.timer.stop()
        super().closeEvent(event)


def main(argv=None):
    """Разбирает argv, проверяет параметры и возвращает код завершения окна Qt."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", choices=("RTS", "MIX"), default="RTS", help="первоначальный инструмент")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="папка баз инструментов")
    parser.add_argument("--db", type=Path, help="явная общая или отдельная SQLite")
    parser.add_argument("--start", type=valid_date, help="начальная дата YYYY-MM-DD")
    parser.add_argument("--end", type=valid_date, help="конечная дата YYYY-MM-DD")
    parser.add_argument("--initial-bars", type=int, default=200, help="начальное число видимых баров, по умолчанию 200")
    parser.add_argument("--interval", type=float, default=1, help="секунд на бар: 0.05…60, по умолчанию 1")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="nuf", help="nuf — форум; fxi — Острова")
    parser.add_argument("--depths", type=int, nargs=3, default=DEFAULT_DEPTHS,
                        metavar=("МАЛ", "СРЕД", "СТАРШ"), help="три глубины СФ; 0 отключает уровень")
    parser.add_argument("--deviation", type=int, help="Deviation в единицах Point")
    parser.add_argument("--backstep", type=int, help="Backstep в барах")
    parser.add_argument("--point", type=float, default=1, help="единица цены для Deviation")
    parser.add_argument("--ma-fast", type=int, default=3, help="период быстрой SMA")
    parser.add_argument("--ma-slow", type=int, default=34, help="период медленной SMA")
    args = parser.parse_args(argv)
    default_dev, default_back = PRESETS[args.preset]
    try:
        initial_bars, interval = validate_simulation(args.initial_bars, args.interval)
        settings = validate_settings(args.depths, default_dev if args.deviation is None else args.deviation,
                                     default_back if args.backstep is None else args.backstep,
                                     args.point, args.ma_fast, args.ma_slow)
        if args.start and args.end and args.start > args.end:
            raise ValueError("Начальная дата должна быть не позже конечной")
    except ValueError as exc:
        parser.error(str(exc))
    app = create_application()
    window = ChartWindow(args.symbol, args.data_dir, args.db, args.start, args.end,
                         initial_bars=initial_bars, interval=interval, preset=args.preset, **settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
