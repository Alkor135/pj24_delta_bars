r"""Интерактивный график дельта-баров RTS/MIX с индикатором Supertrend.

Примеры запуска из папки проекта:
    .\.venv\Scripts\python.exe chart_delta_bars_supertrend.py
    .\.venv\Scripts\python.exe chart_delta_bars_supertrend.py --symbol RTS --start 2022-09-01
    .\.venv\Scripts\python.exe chart_delta_bars_supertrend.py --symbol MIX
    .\.venv\Scripts\python.exe chart_delta_bars_supertrend.py --atr-period 10 --multiplier 3
    .\.venv\Scripts\python.exe chart_delta_bars_supertrend.py --symbol RTS --start 2026-09-01 --end 2026-09-28
    .\.venv\Scripts\python.exe chart_delta_bars_supertrend.py --db "C:\data_quote\delta_bars.sqlite3"
    python chart_delta_bars_supertrend.py --data-dir C:\data_quote --symbol MIX

Зависимости: python -m pip install -r requirements-chart.txt
Проверки: python -m unittest -v tests.test_chart_supertrend

Использует окно, свечи finplot и чтение SQLite исходного chart_delta_bars.py.
«Длительность» наследует нижнюю панель с серыми столбиками: высота каждого
столбика равна длительности бара в минутах, основание находится на нуле.
ATR Уайлдера строится по True Range; первый ATR — среднее первых N баров.
Supertrend использует (high + low) / 2 и подтягиваемые полосы ± множитель * ATR.
Пересечение полосы ценой close переключает тренд: рост — синяя линия,
снижение — красная. До N-го бара линия отсутствует, первый тренд — снижение.
Последняя ступень доходит до правого края свечи, даже при единственном значении.
Ветви рисуются без прореживания, чтобы короткие развороты сохранялись при масштабе.
История до начала видимого диапазона участвует в расчёте; будущие бары не
используются. Неполные дневные бары включены, на границе дня расчёт не сбрасывается.
Период ATR (1…10000) и множитель (0.01…1000, два десятичных знака) меняются
в окне или параметрами командной строки; по умолчанию 10 и 3 соответственно.
ALF и Volume Stops доступны отдельно, по умолчанию скрыты. База только читается.
"""

import argparse
from datetime import date
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtWidgets
import finplot as fplt

from chart_delta_bars import (
    DEFAULT_DATA_DIR,
    ChartCanvas as BaseChartCanvas,
    ChartWindow as BaseChartWindow,
    LoadWorker as BaseLoadWorker,
    create_application,
    valid_date,
)
from source.chart_data import load_bars

DEFAULT_ATR_PERIOD = 10
DEFAULT_MULTIPLIER = 3.0


def validate_parameters(period, multiplier):
    """Проверяет период ATR и множитель, возвращает их как int и float.

    period — целое число баров 1…10000; multiplier — конечное число 0.01…1000
    с точностью до двух десятичных знаков. При ошибке возбуждает ValueError.
    """
    if isinstance(period, bool) or not isinstance(period, Integral) or not 1 <= period <= 10000:
        raise ValueError("Период ATR должен быть целым числом от 1 до 10000")
    try:
        multiplier = float(multiplier)
    except (TypeError, ValueError) as exc:
        raise ValueError("Множитель Supertrend должен быть числом") from exc
    if not np.isfinite(multiplier) or not 0.01 <= multiplier <= 1000:
        raise ValueError("Множитель Supertrend должен быть от 0.01 до 1000")
    if not np.isclose(multiplier, round(multiplier, 2), rtol=0, atol=1e-12):
        raise ValueError("Множитель Supertrend должен иметь не более двух десятичных знаков")
    return int(period), multiplier


def calculate_supertrend(data, period=DEFAULT_ATR_PERIOD, multiplier=DEFAULT_MULTIPLIER):
    """Возвращает ATR, Supertrend, направление и две цветовые серии для OHLC-баров.

    data — DataFrame со столбцами high, low, close в порядке истории;
    period — период ATR в барах; multiplier — расстояние полос в единицах ATR.
    Результат сохраняет индекс data; направление равно +1 при росте, -1 при
    снижении, 0 при прогреве. Цены и параметры проверяются; data не изменяется.
    Первый True Range равен high-low, следующие учитывают предыдущий close.
    ATR и полосы рассчитываются последовательно без обращения к будущим барам.
    """
    period, multiplier = validate_parameters(period, multiplier)
    prices = data[["high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all():
        raise ValueError("Supertrend: цены должны быть конечными числами")
    high, low, close = prices.T
    if ((high < low) | (close < low) | (close > high)).any():
        raise ValueError("Supertrend: OHLC должны удовлетворять low ≤ close ≤ high")
    size = len(data)
    atr = np.full(size, np.nan)
    trend = np.full(size, np.nan)
    direction = np.zeros(size, dtype=np.int8)
    if size >= period:
        previous_close = np.concatenate((close[:1], close[:-1]))
        true_range = np.maximum.reduce((high - low, np.abs(high - previous_close),
                                        np.abs(low - previous_close)))
        first = period - 1
        atr[first] = true_range[:period].mean()
        midpoint = (high[first] + low[first]) / 2
        upper = midpoint + multiplier * atr[first]
        lower = midpoint - multiplier * atr[first]
        direction[first] = -1
        trend[first] = upper
        for index in range(period, size):
            atr[index] = atr[index - 1] + (true_range[index] - atr[index - 1]) / period
            midpoint = (high[index] + low[index]) / 2
            basic_upper = midpoint + multiplier * atr[index]
            basic_lower = midpoint - multiplier * atr[index]
            if basic_upper < upper or close[index - 1] > upper:
                upper = basic_upper
            if basic_lower > lower or close[index - 1] < lower:
                lower = basic_lower
            if direction[index - 1] == -1:
                direction[index] = 1 if close[index] > upper else -1
            else:
                direction[index] = -1 if close[index] < lower else 1
            trend[index] = lower if direction[index] == 1 else upper
    return pd.DataFrame({
        "atr": atr,
        "supertrend": trend,
        "supertrend_direction": direction,
        "supertrend_up": np.where(direction == 1, trend, np.nan),
        "supertrend_down": np.where(direction == -1, trend, np.nan),
    }, index=data.index)


def load_supertrend_bars(path, symbol, dataset_id, start, end, alpha=0.4,
                        period=DEFAULT_ATR_PERIOD, multiplier=DEFAULT_MULTIPLIER):
    """Читает бары и прогревает индикаторы до фильтрации видимого диапазона.

    path — SQLite; symbol и dataset_id выбирают один набор; start/end — даты ISO
    включительно; alpha задаёт ALF; period/multiplier задают Supertrend.
    Возвращает DataFrame исходного просмотрщика с дополнительными столбцами
    calculate_supertrend и новым равномерным x. Параметры сохраняет в attrs.
    База открывается только для чтения; бары после end не загружаются.
    """
    period, multiplier = validate_parameters(period, multiplier)
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if start > end:
        raise ValueError("Начальная дата должна быть не позже конечной")
    history = load_bars(path, symbol, dataset_id, date.min.isoformat(), end.isoformat(), alpha)
    if not history.empty:
        indicators = calculate_supertrend(history, period, multiplier)
        for name in indicators.columns:
            history[name] = indicators[name]
        history = history.loc[history.day >= start.isoformat()].copy().reset_index(drop=True)
        history["x"] = np.arange(len(history), dtype=np.int64)
    history.attrs["supertrend_period"] = period
    history.attrs["supertrend_multiplier"] = multiplier
    return history


class ChartCanvas(BaseChartCanvas):
    """Расширяет исходный график двумя цветными ветвями Supertrend."""

    def __init__(self, owner, data, alpha, offset):
        """Создаёт свечи, столбики длительности и линии Supertrend по data.

        owner — окно с карточкой бара; data содержит рассчитанные индикаторы.
        alpha — коэффициент ALF; offset — отступ меток; передаются исходному графику.
        Линии имеют разрывы при смене направления и ступени между барами.
        Последний отрезок продолжается на полбара вправо: единственное значение
        новой ветви остаётся видимым. Дополнительная точка не входит в данные.
        Прореживание отключено, чтобы не удалять короткие ветви и разрывы между ними.
        """
        super().__init__(owner, data, alpha, offset)
        self.title = "Дельта-бары · Supertrend"
        period = data.attrs["supertrend_period"]
        multiplier = data.attrs["supertrend_multiplier"]
        self.supertrend_items = []
        for column, color, label in (("supertrend_up", "#254bff", "рост"),
                                      ("supertrend_down", "#ef2525", "снижение")):
            item = fplt.plot(data.x, data[column], color=color, width=3, ax=self.price_axis,
                             legend=f"Supertrend · {period} × {multiplier:g} · {label}")
            x = np.append(item.xData, item.xData[-1] + 0.5)
            y = np.append(item.yData, item.yData[-1])
            item.setDownsampling(ds=1, auto=False)
            item.setData(x, y, stepMode="right", connect="finite")
            self.supertrend_items.append(item)
        fplt.refresh()

    def set_supertrend_visible(self, visible):
        """Показывает обе ветви при visible=True, иначе скрывает их без пересчёта."""
        for item in self.supertrend_items:
            item.setVisible(visible)

    def describe_bar(self, index):
        """Добавляет Supertrend, направление и ATR к карточке бара с номером index.

        Для индекса вне data ничего не меняет; во время прогрева сообщает об этом.
        """
        super().describe_bar(index)
        if not 0 <= index < len(self.data):
            return
        bar = self.data.iloc[index]
        period = self.data.attrs["supertrend_period"]
        if pd.isna(bar.supertrend):
            text = f"Supertrend: прогрев ATR ({period} баров)"
        else:
            direction = "рост" if bar.supertrend_direction == 1 else "снижение"
            text = f"Supertrend {bar.supertrend:g} · {direction} · ATR({period}) {bar.atr:g}"
        self.owner.details.setText(self.owner.details.text() + "\n" + text)


class LoadWorker(BaseLoadWorker):
    """Читает SQLite и рассчитывает Supertrend вне потока интерфейса."""

    def run(self):
        """Передаёт бары через loaded или текст ошибки через failed по self.request."""
        try:
            self.loaded.emit(load_supertrend_bars(**self.request))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ChartWindow(BaseChartWindow):
    """Добавляет параметры Supertrend к окну исходного просмотрщика."""

    def __init__(self, symbol="RTS", data_dir=DEFAULT_DATA_DIR, db_path=None, start=None,
                 end=None, auto_load=True, period=DEFAULT_ATR_PERIOD, multiplier=DEFAULT_MULTIPLIER):
        """Создаёт окно с параметрами базы/дат и начальной настройкой Supertrend.

        symbol — RTS/MIX; data_dir — папка баз; db_path — отдельная общая база;
        start/end — даты ISO либо None для последних 30 дней; auto_load включает
        первое чтение. period/multiplier проверяются до создания управления.
        """
        self.initial_period, self.initial_multiplier = validate_parameters(period, multiplier)
        super().__init__(symbol, data_dir, db_path, start, end, auto_load=False)
        self.setWindowTitle("Дельта-бары · Supertrend")
        if auto_load and self.dataset_combo.count():
            self.start_load()

    def _build_controls(self, symbol):
        """Добавляет к управлению инструмента symbol отдельную строку Supertrend.

        ALF и Volume Stops первоначально скрыты; исходные флажки доступны.
        """
        super()._build_controls(symbol)
        self.alf_check.setChecked(False)
        self.signals_check.setChecked(False)
        options = QtWidgets.QHBoxLayout()
        self.supertrend_check = QtWidgets.QCheckBox("Supertrend")
        self.supertrend_check.setChecked(True)
        options.addWidget(self.supertrend_check)
        options.addWidget(QtWidgets.QLabel("Период ATR"))
        self.period_spin = QtWidgets.QSpinBox()
        self.period_spin.setRange(1, 10000)
        self.period_spin.setValue(self.initial_period)
        self.period_spin.setToolTip("Число дельта-баров для ATR Уайлдера")
        options.addWidget(self.period_spin)
        options.addWidget(QtWidgets.QLabel("Множитель"))
        self.multiplier_spin = QtWidgets.QDoubleSpinBox()
        self.multiplier_spin.setRange(0.01, 1000)
        self.multiplier_spin.setDecimals(2)
        self.multiplier_spin.setSingleStep(0.5)
        self.multiplier_spin.setValue(self.initial_multiplier)
        self.multiplier_spin.setToolTip("Расстояние полос от (high + low) / 2 в единицах ATR")
        options.addWidget(self.multiplier_spin)
        colors = QtWidgets.QLabel("Рост — синий · Снижение — красный")
        colors.setStyleSheet("color: #64748b;")
        options.addWidget(colors)
        options.addStretch()
        self.centralWidget().layout().insertLayout(3, options)
        self.details.setMinimumHeight(70)

    def _connect_controls(self):
        """Подключает новые настройки к пересчёту, а флажок — только к видимости."""
        super()._connect_controls()
        self.period_spin.editingFinished.connect(self.start_load)
        self.multiplier_spin.editingFinished.connect(self.start_load)
        self.supertrend_check.toggled.connect(self.apply_visibility)

    def set_busy(self, busy):
        """Блокирует параметры Supertrend и исходные элементы на время загрузки busy."""
        super().set_busy(busy)
        self.period_spin.setEnabled(not busy)
        self.multiplier_spin.setEnabled(not busy)

    def start_load(self, *args):
        """Запускает чтение и расчёт с текущими настройками; args сигналов Qt игнорирует.

        Во время запроса и при отсутствии набора ничего не делает.
        """
        if self.loading or not self.dataset_combo.count():
            return
        request = {"path": self.database_path(), "symbol": self.symbol_combo.currentText(),
                   "dataset_id": self.dataset_combo.currentData(),
                   "start": self.start_edit.date().toString("yyyy-MM-dd"),
                   "end": self.end_edit.date().toString("yyyy-MM-dd"),
                   "alpha": self.alpha_spin.value(), "period": self.period_spin.value(),
                   "multiplier": self.multiplier_spin.value()}
        self.set_busy(True)
        self.status_label.setText("Загрузка баров и расчёт Supertrend с предысторией…")
        self.worker = LoadWorker(request, self)
        self.worker.loaded.connect(self.accept_data)
        self.worker.failed.connect(self.load_failed)
        self.worker.finished.connect(self.load_finished)
        self.worker.start()

    def redraw(self):
        """Перестраивает расширенный график по self.data, освобождая прежние объекты."""
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
        """Применяет исходные флажки и Supertrend; args сигналов Qt игнорирует."""
        super().apply_visibility()
        if self.canvas:
            self.canvas.set_supertrend_visible(self.supertrend_check.isChecked())

    def accept_data(self, data):
        """Принимает DataFrame data, обновляет заголовок и сообщает о прогреве ATR."""
        super().accept_data(data)
        if self._closing or data.empty or self.canvas is None:
            return
        period = data.attrs["supertrend_period"]
        multiplier = data.attrs["supertrend_multiplier"]
        self.setWindowTitle(f"{data.symbol.iloc[0]} · Дельта-бары · Supertrend ({period}, {multiplier:g})")
        if data.supertrend.isna().all():
            self.status_label.setText(self.status_label.text() + f" · Supertrend: прогрев ATR {period} баров")


def valid_period(value):
    """Преобразует строку value командной строки в допустимый целый период ATR."""
    try:
        period, _ = validate_parameters(int(value), DEFAULT_MULTIPLIER)
        return period
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def valid_multiplier(value):
    """Преобразует строку value командной строки в допустимый множитель Supertrend."""
    try:
        _, multiplier = validate_parameters(DEFAULT_ATR_PERIOD, value)
        return multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv=None):
    """Разбирает список аргументов argv и возвращает код завершения окна Qt.

    При argv=None читает командную строку процесса. --help и неправильные
    аргументы обрабатываются argparse до создания приложения Qt.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", choices=("RTS", "MIX"), default="RTS", help="первоначальный инструмент")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="папка двух баз инструментов")
    parser.add_argument("--db", type=Path, help="явно выбранная база вместо стандартных файлов")
    parser.add_argument("--start", type=valid_date, help="начальная дата; по умолчанию последние 30 календарных дней")
    parser.add_argument("--end", type=valid_date, help="конечная дата; по умолчанию последний день базы")
    parser.add_argument("--atr-period", type=valid_period, default=DEFAULT_ATR_PERIOD,
                        help="период ATR Уайлдера в дельта-барах, 1…10000 (по умолчанию 10)")
    parser.add_argument("--multiplier", type=valid_multiplier, default=DEFAULT_MULTIPLIER,
                        help="множитель ATR, 0.01…1000 (по умолчанию 3)")
    args = parser.parse_args(argv)
    app = create_application()
    window = ChartWindow(args.symbol, args.data_dir, args.db, args.start, args.end,
                         period=args.atr_period, multiplier=args.multiplier)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
