r"""График дельта-баров RTS/MIX с «Семафором» MT4 и цветными островами SMA.

Примеры запуска из папки проекта:
    .\.venv\Scripts\python.exe chart_delta_bar_semafor.py
    .\.venv\Scripts\python.exe chart_delta_bar_semafor.py --symbol MIX
    .\.venv\Scripts\python.exe chart_delta_bar_semafor.py --db C:\data_quote\delta_bars.sqlite3
    .\.venv\Scripts\python.exe chart_delta_bar_semafor.py --preset fxi --depths 0 12 34
    .\.venv\Scripts\python.exe chart_delta_bar_semafor.py --depths 5 12 34 --deviation 1 --backstep 1
    .\.venv\Scripts\python.exe chart_delta_bar_semafor.py --ma-fast 3 --ma-slow 34 --point 1
    .\.venv\Scripts\python.exe chart_delta_bar_semafor.py --start 2026-09-01 --end 2026-09-30
    python -m unittest -v tests.test_chart_semafor

Зависимости: python -m pip install -r requirements-chart.txt
На основе интерфейса chart_delta_bars.py; ALF и StopVolume не рассчитываются.
SQLite открывается только для чтения. Свечи, объём, длительность, выбор дат,
наборов и инструмента сохранены. Время МСК, ось X равномерна по номерам баров.

Алгоритм CountZZ перенесён из вложенных FXi 3 Semafor.mq4 и NUF FXi 3 Semafor-1.mq4.
Три независимых ZigZag имеют Depth 5/12/34 баров (на скриншоте: СФ 34/12/5).
Depth=0 отключает уровень. Сначала выбираются экстремумы за Depth баров,
затем Deviation*Point отсекает кандидатов, Backstep удаляет менее сильные
близкие экстремумы, а финальный проход оставляет лучшие вершины/впадины.
Deviation — допуск кандидата относительно экстремума окна, НЕ процент
разворота. Point — единица цены MT4, по умолчанию 1 пункт котировки RTS/MIX;
это не биржевой шаг: для измерения допуска в тиках задайте 10 для RTS, 25 для MIX.
NUF (форум, по умолчанию): Deviation=1, Backstep=1, красные/зелёные метки.
FXi («Острова»): Deviation=5, Backstep=3, средние точки жёлтые, крупные солнца.
История до начальной даты прогревает индикаторы; после конечной даты не читается.
Неполные дневные бары включены, на границе дней расчёт не сбрасывается.
При добавлении баров последние точки МОГУТ ПЕРЕМЕЩАТЬСЯ/ИСЧЕЗАТЬ, как в MT4.
Показанная историческая метка не равна сигналу, известному в момент её бара.

«Острова» — область между простыми MA(close) 3/34, зелёная при быстрой MA
выше медленной, красная при обратном порядке. Периоды и видимость меняются
в окне. Автоматические сделки и треугольники 1-2-3 не строятся.
Вертикальный автомасштаб учитывает цены SMA и метки с декоративным отступом;
по краям оставляется место для крупных символов при приближении графика.
Первоисточник семейства MT4: https://www.mql5.com/en/code/7730
"""

import argparse
from datetime import date
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import finplot as fplt

from chart_delta_bars import (
    DEFAULT_DATA_DIR, BarTimeAxis, ChartCanvas as BaseChartCanvas,
    ChartWindow as BaseChartWindow, LoadWorker as BaseLoadWorker,
    create_application, valid_date,
)
from source.chart_data import SELECT_COLUMNS, readonly_connection

DEFAULT_DEPTHS = (5, 12, 34)
PRESETS = {"nuf": (1, 1), "fxi": (5, 3)}


def integer_parameter(value, name, minimum=0):
    """Возвращает целое значение от minimum до 10000 или возбуждает ValueError.

    value — проверяемое число; name — русское имя параметра для ошибки;
    minimum — нижняя допустимая граница. Дробные значения и bool запрещены.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= 10000:
        raise ValueError(f"{name}: требуется целое число от {minimum} до 10000")
    return int(value)


def validate_settings(depths, deviation, backstep, point, ma_fast, ma_slow):
    """Нормализует настройки СФ/SMA и возвращает словарь для расчёта и загрузки.

    depths — три Depth в барах; deviation — допуск в единицах point;
    backstep — число предыдущих баров для удаления слабого кандидата;
    point — положительная единица цены; ma_fast/ma_slow — периоды SMA.
    Некорректные настройки вызывают ValueError.
    """
    if len(depths) != 3:
        raise ValueError("Семафор требует ровно три глубины: малую, среднюю, старшую")
    depths = tuple(integer_parameter(v, "Глубина СФ") for v in depths)
    deviation = integer_parameter(deviation, "Deviation")
    backstep = integer_parameter(backstep, "Backstep")
    try:
        point = float(point)
    except (ValueError, TypeError) as exc:
        raise ValueError("Point должен быть числом") from exc
    if not np.isfinite(point) or not 0.01 <= point <= 100000 or not np.isclose(point, round(point, 2), rtol=0, atol=1e-12):
        raise ValueError("Point должен быть от 0.01 до 100000, не более двух знаков после запятой")
    return dict(depths=depths, deviation=deviation, backstep=backstep, point=point,
                ma_fast=integer_parameter(ma_fast, "Быстрая SMA", 1),
                ma_slow=integer_parameter(ma_slow, "Медленная SMA", 1))


def validate_prices(data):
    """Проверяет конечные high/low/close и их порядок; возвращает массивы high/low.

    data — DataFrame баров; close проверяется, если присутствует. Неверные
    цены вызывают ValueError; входная таблица не изменяется.
    """
    columns = ["high", "low"] + (["close"] if "close" in data else [])
    prices = data[columns].to_numpy(dtype=float)
    if not np.isfinite(prices).all():
        raise ValueError("В барах обнаружены некорректные цены")
    highs, lows = prices[:, 0], prices[:, 1]
    if np.any(highs < lows) or (len(columns) == 3 and np.any((prices[:, 2] < lows) | (prices[:, 2] > highs))):
        raise ValueError("Требуется low ≤ close ≤ high для каждого бара")
    return highs, lows


def calculate_zigzag(data, depth=12, deviation=1, backstep=1, point=1):
    """Возвращает разреженные zz_low/zz_high по алгоритму CountZZ семафора MT4.

    data — high/low и необязательный close в хронологическом порядке;
    depth — длина прошлого окна, 0 отключает уровень; deviation/point —
    ценовой допуск; backstep — удаление менее сильных близких кандидатов.
    NaN означает отсутствие метки. Финальный проход, как в исходном .mq4,
    может убрать ранее нарисованную точку при появлении лучшего экстремума.
    Первая низина прогрева скрывается согласно последнему циклу CountZZ.
    """
    settings = validate_settings((depth, 0, 0), deviation, backstep, point, 1, 1)
    depth, deviation, backstep, point = (settings[k] for k in ("depths", "deviation", "backstep", "point"))
    depth = depth[0]
    highs, lows = validate_prices(data)
    size = len(data)
    low_points = np.full(size, np.nan)
    high_points = np.full(size, np.nan)
    if depth == 0 or size < depth:
        return pd.DataFrame({"zz_low": low_points, "zz_high": high_points}, index=data.index)
    rolling_low = pd.Series(lows).rolling(depth).min().to_numpy()
    rolling_high = pd.Series(highs).rolling(depth).max().to_numpy()
    last_low = last_high = None
    tolerance = deviation * point
    for i in range(depth - 1, size):
        low, high = rolling_low[i], rolling_high[i]
        if low != last_low:
            last_low = low
            if lows[i] - low <= tolerance:
                for previous in range(max(0, i - backstep), i):
                    if low_points[previous] > low:
                        low_points[previous] = np.nan
                low_points[i] = low
        if high != last_high:
            last_high = high
            if high - highs[i] <= tolerance:
                for previous in range(max(0, i - backstep), i):
                    if high_points[previous] < high:
                        high_points[previous] = np.nan
                high_points[i] = high
    last_low = last_high = None
    low_position = high_position = None
    for i in range(depth - 1, size):
        low, high = low_points[i], high_points[i]
        if np.isfinite(high):
            if last_high is not None:
                if high > last_high:
                    high_points[high_position] = np.nan
                else:
                    high_points[i] = np.nan
            if last_high is None or high > last_high:
                last_high, high_position = high, i
            last_low = None
        if np.isfinite(low):
            if last_low is not None:
                if low < last_low:
                    low_points[low_position] = np.nan
                else:
                    low_points[i] = np.nan
            if last_low is None or low < last_low:
                last_low, low_position = low, i
            last_high = None
    low_points[depth - 1] = np.nan
    return pd.DataFrame({"zz_low": low_points, "zz_high": high_points}, index=data.index)


def calculate_indicators(data, depths=DEFAULT_DEPTHS, deviation=1, backstep=1,
                         point=1, ma_fast=3, ma_slow=34):
    """Возвращает копию OHLC с тремя уровнями СФ и островами простых SMA(close).

    data — хронологические бары; depths/deviation/backstep/point — настройки
    CountZZ; ma_fast/ma_slow — окна SMA. Атрибуты результата сохраняют настройки.
    Дни не разделяют расчёт. Начальные непрогретые значения равны NaN.
    """
    settings = validate_settings(depths, deviation, backstep, point, ma_fast, ma_slow)
    validate_prices(data)
    result = data.copy()
    for level, depth in enumerate(settings["depths"], 1):
        zigzag = calculate_zigzag(data, depth, deviation, backstep, point)
        result[f"sf{level}_low"] = zigzag.zz_low
        result[f"sf{level}_high"] = zigzag.zz_high
    result["ma_fast"] = result.close.rolling(settings["ma_fast"]).mean()
    result["ma_slow"] = result.close.rolling(settings["ma_slow"]).mean()
    result["island_direction"] = np.select(
        [result.ma_fast > result.ma_slow, result.ma_fast < result.ma_slow], [1, -1], default=0)
    result.attrs.update(settings)
    return result


def load_semafor_bars(path, symbol, dataset_id, start, end, **settings):
    """Читает SQLite без записи и возвращает видимый срез с прогретыми СФ/SMA.

    path — существующая база; symbol/dataset_id — выбранный инструмент/набор;
    start/end — включительные даты ISO; settings — параметры calculate_indicators.
    До start читается предыстория, после end бары не читаются. Сохраняются
    исходные времена, объёмы и неполные бары; x нумеруется с нуля.
    """
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if start > end:
        raise ValueError("Начальная дата должна быть не позже конечной")
    with readonly_connection(path) as connection:
        data = pd.read_sql_query(
            f"SELECT {SELECT_COLUMNS} FROM bars WHERE dataset_id=? AND symbol=? AND day<=? ORDER BY day,bar_index",
            connection, params=(dataset_id, symbol, end.isoformat()))
    if data.empty:
        return data
    if data.duplicated(["day", "bar_index"]).any():
        raise ValueError("В базе обнаружены повторяющиеся номера баров одного дня")
    numeric = data[["open", "high", "low", "close", "volume", "delta", "threshold", "tick_count"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("В базе обнаружены некорректные числовые значения")
    if ((data.open < data.low) | (data.open > data.high)).any():
        raise ValueError("Цена open должна находиться между low и high")
    for column in ("start_time", "end_time"):
        data[column] = pd.to_datetime(data[column], format="ISO8601", errors="raise")
    data["duration_seconds"] = (data.end_time - data.start_time).dt.total_seconds()
    if (data.duration_seconds < 0).any():
        raise ValueError("Конец бара предшествует его началу")
    data = calculate_indicators(data, **settings)
    visible = data.loc[data.day >= start.isoformat()].copy().reset_index(drop=True)
    visible["x"] = np.arange(len(visible), dtype=np.int64)
    return visible


def sun_symbol():
    """Возвращает восьмилучевой QPainterPath для «солнышка» фиксированного размера."""
    path = QtGui.QPainterPath()
    for i in range(16):
        angle = i * np.pi / 8
        radius = 0.5 if i % 2 == 0 else 0.25
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    path.closeSubpath()
    return path


class ChartCanvas(BaseChartCanvas):
    """Встраивает свечи, острова SMA, три уровня СФ, объём и длительность."""

    def __init__(self, owner, data, offset, preset):
        """Создаёт график data; offset отодвигает метки, preset задаёт стиль NUF/FXi.

        owner — главное окно; data — рассчитанные бары. Размер меток задан
        в пикселях, их цена в data сохраняется без визуального отступа.
        """
        pg.GraphicsLayoutWidget.__init__(self, parent=owner)
        self.owner, self.data = owner, data
        self.title, self.show_maximized = "Дельта-бары · Семафор", False
        self.price_axis, self.duration_axis = fplt.create_plot_widget(self, rows=2, init_zoom_periods=200)
        self.price_axis.vb.v_zoom_scale = 0.92
        self._axes = [self.price_axis, self.duration_axis]
        for row, axis in enumerate(self._axes):
            self.addItem(axis, row=row, col=0)
            axis.setAxisItems({"bottom": BarTimeAxis(data)})
            axis.set_visible(xgrid=True, ygrid=True, xaxis=True)
            axis.getAxis("right").setWidth(90)
            fplt.add_crosshair_info(self.crosshair_info, ax=axis)
        self.ci.layout.setRowStretchFactor(0, 5)
        self.ci.layout.setRowStretchFactor(1, 1)
        self.ci.setContentsMargins(0, 0, 0, 0)
        self.duration_axis.setLabel("right", "мин")
        self.island_items = []
        x = data.x.to_numpy(dtype=float)
        # Полосы одного бара воспроизводят DRAW_HISTOGRAM двух MA в MT4.
        for direction, color in ((1, "#16a34a"), (-1, "#ef4444")):
            mask = data.island_direction.to_numpy() == direction
            item = pg.BarGraphItem(x=x[mask], y0=data.ma_slow.to_numpy()[mask],
                                   y1=data.ma_fast.to_numpy()[mask], width=1,
                                   pen=None, brush=pg.mkBrush(color + "65"))
            item.setZValue(-10)
            self.price_axis.addItem(item)
            self.island_items.append(item)
        for name, color in (("ma_fast", "#16803d"), ("ma_slow", "#d54747")):
            item = fplt.plot(data.x, data[name].rename(name), color=color, width=1,
                             ax=self.price_axis)
            item.setDownsampling(auto=False)
            self.island_items.append(item)
        fplt.add_legend(f"Острова · SMA {data.attrs['ma_fast']}/{data.attrs['ma_slow']}", ax=self.price_axis)
        self.candles = fplt.candlestick_ochl(data[["x", "open", "close", "high", "low"]], ax=self.price_axis)
        self.volume_axis = self.price_axis.overlay(scale=0.20)
        self.volume_item = fplt.volume_ocv(data[["x", "open", "close", "volume"]], ax=self.volume_axis)
        self.semafor_items = [[], [], []]
        for level, depth in enumerate(data.attrs["depths"]):
            if not depth:
                continue
            for side, color, sign in (("low", "#008820", -1), ("high", "#df2525", 1)):
                if preset == "fxi" and level == 1:
                    color = "#e5b400"
                values = data[f"sf{level + 1}_{side}"]
                if not values.notna().any():
                    continue
                size = (7, 12, 21)[level]
                symbol = sun_symbol() if level == 2 or (preset == "nuf" and level == 1) else "o"
                # fplt.plot регистрирует цены меток в автомасштабе finplot.
                positions = (values + sign * offset).rename(f"sf{level + 1}_{side}_display")
                item = fplt.plot(data.x, positions, style="o", color=color,
                                 width=size / 7, ax=self.price_axis)
                item.setSymbol(symbol)
                item.setSymbolPen(pg.mkPen(color, width=0.9))
                item.setDownsampling(auto=False)
                item.setZValue(20 + level)
                self.semafor_items[level].append(item)
                if preset == "fxi" and level == 2:
                    center = fplt.plot(data.x, positions.rename(f"sf{level + 1}_{side}_center"),
                                       style="o", color="#ffe33d", width=6 / 7,
                                       ax=self.price_axis, zoomscale=False)
                    center.setDownsampling(auto=False)
                    center.setZValue(24)
                    self.semafor_items[level].append(center)
            fplt.add_legend(f"СФ {depth} · {('точки', 'светики', 'солнышки')[level]}", ax=self.price_axis)
        self.partial_item = None
        partial = data.is_complete == 0
        if partial.any():
            self.partial_item = fplt.plot(data.x, (data.high + offset * 0.5).where(partial).rename("partial"),
                                          style="^", color="#d99519", width=0.8,
                                          ax=self.price_axis, legend="Неполный бар")
        self.day_lines = []
        for index in np.flatnonzero(data.day.to_numpy()[1:] != data.day.to_numpy()[:-1]) + 1:
            for axis in self._axes:
                line = pg.InfiniteLine(pos=float(index) - 0.5, angle=90,
                                       pen=pg.mkPen("#94a3b8", width=1, style=QtCore.Qt.PenStyle.DashLine))
                axis.addItem(line, ignoreBounds=True)
                self.day_lines.append(line)
        duration = data[["x", "open", "close"]].assign(duration_minutes=data.duration_seconds / 60)
        self.duration_item = fplt.volume_ocv(duration, candle_width=0.8, ax=self.duration_axis)
        self.duration_item.colors.update(bull_frame="#64748b", bull_body="#64748b",
                                         bear_frame="#64748b", bear_body="#64748b")
        self.duration_item.resamp = None
        fplt.add_legend("Длительность бара, мин", ax=self.duration_axis)
        fplt.refresh()

    def set_visibility(self, islands, semafors, volume, duration, days, partial):
        """Применяет флажки островов, трёх СФ и вспомогательных панелей/меток.

        islands/volume/duration/days/partial — bool; semafors — три bool.
        Переключение видимости не запускает повторное чтение SQLite.
        """
        for item in self.island_items:
            item.setVisible(islands)
        for items, visible in zip(self.semafor_items, semafors):
            for item in items:
                item.setVisible(visible)
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
        """Добавляет к исходной карточке бара цены SMA и имеющиеся метки СФ.

        index — номер видимого бара; результат выводится в details окна.
        Цены СФ показываются без декоративного отступа.
        """
        super().describe_bar(index)
        if not 0 <= index < len(self.data):
            return
        bar = self.data.iloc[index]
        labels = []
        if np.isfinite(bar.ma_slow):
            labels.append(f"SMA {self.data.attrs['ma_fast']}/{self.data.attrs['ma_slow']}: {bar.ma_fast:g}/{bar.ma_slow:g}")
        for level, depth in enumerate(self.data.attrs["depths"], 1):
            for side, label in (("low", "низ"), ("high", "верх")):
                value = bar[f"sf{level}_{side}"]
                if np.isfinite(value):
                    labels.append(f"СФ {depth} {label}: {value:g}")
        if labels:
            self.owner.details.setText(self.owner.details.text() + "\n" + " · ".join(labels))


class LoadWorker(BaseLoadWorker):
    """Читает историю и рассчитывает СФ/SMA в отдельном потоке Qt."""

    def run(self):
        """Передаёт результат load_semafor_bars или причину ошибки через сигналы."""
        try:
            self.loaded.emit(load_semafor_bars(**self.request))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ChartWindow(BaseChartWindow):
    """Использует навигацию исходного окна с отдельными настройками СФ и островов."""

    def __init__(self, symbol="RTS", data_dir=DEFAULT_DATA_DIR, db_path=None,
                 start=None, end=None, auto_load=True, depths=DEFAULT_DEPTHS,
                 deviation=None, backstep=None, point=1, ma_fast=3, ma_slow=34, preset="nuf"):
        """Создаёт окно SQLite-графика с заданными периодами и пресетом NUF/FXi.

        symbol/data_dir/db_path/start/end/auto_load — выбор источника и дат;
        остальные параметры соответствуют validate_settings. None для
        deviation/backstep берёт значение выбранного preset.
        """
        if preset not in PRESETS:
            raise ValueError("Пресет должен быть nuf или fxi")
        default_dev, default_back = PRESETS[preset]
        self.initial_settings = validate_settings(depths, default_dev if deviation is None else deviation,
                                                  default_back if backstep is None else backstep,
                                                  point, ma_fast, ma_slow)
        self.initial_preset = preset
        super().__init__(symbol, data_dir, db_path, start, end, auto_load)
        self.setWindowTitle(f"{symbol} · Дельта-бары · Семафор и острова")

    def _build_controls(self, symbol):
        """Размещает источник, даты, настройки CountZZ/SMA, флажки и график symbol."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 10, 12, 8)
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Дельта-бары · Семафор")
        title.setStyleSheet("font-size: 17px; font-weight: 600; color: #16324f;")
        top.addWidget(title)
        self.symbol_combo = QtWidgets.QComboBox()
        self.symbol_combo.addItems(["RTS", "MIX"])
        self.symbol_combo.setCurrentText(symbol)
        top.addWidget(self.symbol_combo)
        self.start_edit, self.end_edit = QtWidgets.QDateEdit(), QtWidgets.QDateEdit()
        for label, edit in (("С", self.start_edit), ("По", self.end_edit)):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("dd.MM.yyyy")
            top.addWidget(QtWidgets.QLabel(label))
            top.addWidget(edit)
        self.load_button = QtWidgets.QPushButton("Показать")
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
        parameters = QtWidgets.QHBoxLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("NUF · форум", "nuf")
        self.preset_combo.addItem("FXi · Острова", "fxi")
        self.preset_combo.setCurrentIndex(self.preset_combo.findData(self.initial_preset))
        parameters.addWidget(self.preset_combo)
        self.depth_spins = []
        for label, depth in zip(("СФ мал.", "сред.", "старш."), self.initial_settings["depths"]):
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 10000)
            spin.setValue(depth)
            spin.setSpecialValueText("выкл.")
            spin.setToolTip("Depth ZigZag в дельта-барах; 0 отключает уровень")
            parameters.addWidget(QtWidgets.QLabel(label))
            parameters.addWidget(spin)
            self.depth_spins.append(spin)
        self.deviation_spin = QtWidgets.QSpinBox()
        self.backstep_spin = QtWidgets.QSpinBox()
        for label, spin, name in (("Dev", self.deviation_spin, "deviation"),
                                  ("Back", self.backstep_spin, "backstep")):
            spin.setRange(0, 10000)
            spin.setValue(self.initial_settings[name])
            parameters.addWidget(QtWidgets.QLabel(label))
            parameters.addWidget(spin)
        self.deviation_spin.setToolTip("Deviation: допуск к экстремуму окна в единицах Point")
        self.backstep_spin.setToolTip("Backstep: удаление менее сильных кандидатов в предыдущих барах")
        self.point_spin = QtWidgets.QDoubleSpinBox()
        self.point_spin.setRange(0.01, 100000)
        self.point_spin.setDecimals(2)
        self.point_spin.setValue(self.initial_settings["point"])
        self.point_spin.setToolTip("Единица цены для Deviation. 1 — пункт; 10 — тик RTS; 25 — тик MIX")
        parameters.addWidget(QtWidgets.QLabel("Point"))
        parameters.addWidget(self.point_spin)
        parameters.addStretch()
        layout.addLayout(parameters)
        options = QtWidgets.QHBoxLayout()
        self.islands_check = QtWidgets.QCheckBox("Острова SMA")
        self.ma_fast_spin, self.ma_slow_spin = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for spin, name in ((self.ma_fast_spin, "ma_fast"), (self.ma_slow_spin, "ma_slow")):
            spin.setRange(1, 10000)
            spin.setValue(self.initial_settings[name])
        options.addWidget(self.islands_check)
        options.addWidget(self.ma_fast_spin)
        options.addWidget(self.ma_slow_spin)
        self.semafor_checks = [QtWidgets.QCheckBox(label) for label in ("Точки", "Светики", "Солнышки")]
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(0, 100000)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setValue(self.offsets[symbol])
        self.offset_spin.setPrefix("Отступ ")
        self.offset_spin.setToolTip("Декоративный отступ от цены СФ; 0 помещает метку на экстремум")
        self.volume_check = QtWidgets.QCheckBox("Объём")
        self.days_check = QtWidgets.QCheckBox("Границы дней")
        self.partial_check = QtWidgets.QCheckBox("Неполные")
        self.duration_check = QtWidgets.QCheckBox("Длительность")
        self.checkboxes = [self.islands_check, *self.semafor_checks, self.volume_check,
                           self.days_check, self.partial_check, self.duration_check]
        for checkbox in self.checkboxes[:-1]:
            checkbox.setChecked(True)
        for widget in (*self.semafor_checks, self.offset_spin, self.volume_check,
                       self.days_check, self.partial_check, self.duration_check):
            options.addWidget(widget)
        options.addStretch()
        layout.addLayout(options)
        note = QtWidgets.QLabel("СФ 34/12/5 — три глубины ZigZag. Последние метки могут перемещаться при обновлении истории.")
        note.setStyleSheet("color: #64748b;")
        layout.addWidget(note)
        self.chart_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.chart_layout, 1)
        self.placeholder = QtWidgets.QLabel("Выберите инструмент и период")
        self.placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("background: white; color: #64748b; font-size: 16px;")
        self.chart_layout.addWidget(self.placeholder)
        self.details = QtWidgets.QLabel("Наведите курсор на свечу для просмотра параметров бара")
        self.details.setMinimumHeight(66)
        self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setStyleSheet("background: #eef3f8; color: #243c54; padding: 6px 10px;")
        layout.addWidget(self.details)
        self.status_label = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(QtWidgets.QLabel("Время: МСК · Равный шаг баров"))
        self.parameter_widgets = [*self.depth_spins, self.deviation_spin, self.backstep_spin,
                                  self.point_spin, self.ma_fast_spin, self.ma_slow_spin]

    def _connect_controls(self):
        """Подключает навигацию и пересчёт настроек без прежних индикаторов."""
        self.load_button.clicked.connect(self.start_load)
        self.refresh_button.clicked.connect(self.refresh_data)
        self.file_button.clicked.connect(self.choose_database)
        self.all_button.clicked.connect(self.show_all)
        self.symbol_combo.currentTextChanged.connect(self.change_symbol)
        self.dataset_combo.currentIndexChanged.connect(self.start_load)
        self.preset_combo.currentIndexChanged.connect(self.change_preset)
        for spin in self.parameter_widgets:
            spin.editingFinished.connect(self.start_load)
        self.offset_spin.editingFinished.connect(self.redraw)
        for checkbox in self.checkboxes:
            checkbox.toggled.connect(self.apply_visibility)
        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        shortcut.activated.connect(self.refresh_data)

    def change_preset(self, *args):
        """Выставляет Dev/Back выбранной версии и пересчитывает график; args — сигнал Qt."""
        deviation, backstep = PRESETS[self.preset_combo.currentData()]
        self.deviation_spin.setValue(deviation)
        self.backstep_spin.setValue(backstep)
        self.start_load()

    def set_busy(self, busy):
        """Блокирует параметры активной загрузки при busy=True; видимость доступна."""
        self.loading = busy
        for widget in (self.symbol_combo, self.dataset_combo, self.start_edit, self.end_edit,
                       self.load_button, self.refresh_button, self.file_button,
                       self.preset_combo, self.offset_spin, *self.parameter_widgets):
            widget.setEnabled(not busy)

    def start_load(self, *args):
        """Запускает одно чтение SQLite с текущими настройками СФ/SMA; args — сигнал Qt."""
        if self.loading or not self.dataset_combo.count():
            return
        request = dict(path=self.database_path(), symbol=self.symbol_combo.currentText(),
                       dataset_id=self.dataset_combo.currentData(),
                       start=self.start_edit.date().toString("yyyy-MM-dd"),
                       end=self.end_edit.date().toString("yyyy-MM-dd"),
                       depths=tuple(spin.value() for spin in self.depth_spins),
                       deviation=self.deviation_spin.value(), backstep=self.backstep_spin.value(),
                       point=self.point_spin.value(), ma_fast=self.ma_fast_spin.value(),
                       ma_slow=self.ma_slow_spin.value())
        self.set_busy(True)
        self.status_label.setText("Загрузка баров и расчёт Семафора/SMA с предысторией…")
        self.worker = LoadWorker(request, self)
        self.worker.loaded.connect(self.accept_data)
        self.worker.failed.connect(self.load_failed)
        self.worker.finished.connect(self.load_finished)
        self.worker.start()

    def accept_data(self, data):
        """Принимает рассчитанный data, обновляет свечи, заголовок и состояние прогрева."""
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
        self.setWindowTitle(f"{data.symbol.iloc[0]} · Дельта-бары · Семафор и острова")
        count = sum(int(data[f"sf{level}_{side}"].notna().sum())
                    for level in (1, 2, 3) for side in ("low", "high"))
        note = " · SMA: прогрев" if data.ma_slow.isna().any() else ""
        self.status_label.setText(f"{len(data):,} баров · меток СФ {count} · {data.day.iloc[0]} — {data.day.iloc[-1]}{note}".replace(",", " "))
        self.canvas.describe_bar(len(data) - 1)

    def redraw(self):
        """Перестраивает графические объекты с текущим отступом без чтения базы."""
        self.offsets[self.symbol_combo.currentText()] = self.offset_spin.value()
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

    def apply_visibility(self, *args):
        """Применяет видимость островов, уровней СФ и панелей; args — сигнал Qt."""
        if self.canvas:
            self.canvas.set_visibility(self.islands_check.isChecked(),
                                       [w.isChecked() for w in self.semafor_checks],
                                       self.volume_check.isChecked(), self.duration_check.isChecked(),
                                       self.days_check.isChecked(), self.partial_check.isChecked())


def main(argv=None):
    """Разбирает argv и запускает окно СФ/островов; возвращает код завершения Qt.

    argv — необязательный список аргументов командной строки. Ошибочные
    параметры выводятся argparse до открытия окна и чтения базы.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", choices=("RTS", "MIX"), default="RTS", help="первоначальный инструмент")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="папка баз инструментов")
    parser.add_argument("--db", type=Path, help="явная общая или отдельная SQLite")
    parser.add_argument("--start", type=valid_date, help="начальная дата YYYY-MM-DD")
    parser.add_argument("--end", type=valid_date, help="конечная дата YYYY-MM-DD")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="nuf", help="nuf — форум; fxi — Острова")
    parser.add_argument("--depths", type=int, nargs=3, default=DEFAULT_DEPTHS, metavar=("МАЛ", "СРЕД", "СТАРШ"), help="три глубины, по умолчанию 5 12 34; 0 отключает уровень")
    parser.add_argument("--deviation", type=int, help="Deviation в единицах Point; 1 у NUF, 5 у FXi")
    parser.add_argument("--backstep", type=int, help="Backstep в барах; 1 у NUF, 3 у FXi")
    parser.add_argument("--point", type=float, default=1, help="единица цены для Deviation, по умолчанию 1")
    parser.add_argument("--ma-fast", type=int, default=3, help="период быстрой SMA, по умолчанию 3")
    parser.add_argument("--ma-slow", type=int, default=34, help="период медленной SMA, по умолчанию 34")
    args = parser.parse_args(argv)
    default_dev, default_back = PRESETS[args.preset]
    try:
        settings = validate_settings(args.depths, default_dev if args.deviation is None else args.deviation,
                                     default_back if args.backstep is None else args.backstep,
                                     args.point, args.ma_fast, args.ma_slow)
        if args.start and args.end and args.start > args.end:
            raise ValueError("Начальная дата должна быть не позже конечной")
    except ValueError as exc:
        parser.error(str(exc))
    app = create_application()
    window = ChartWindow(args.symbol, args.data_dir, args.db, args.start, args.end,
                         preset=args.preset, **settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
