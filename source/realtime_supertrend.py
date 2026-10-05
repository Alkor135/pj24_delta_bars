r"""Добавляет Supertrend к двум реал-тайм графикам Семафора/SMA RTS и MIX.

Примеры из корня проекта:
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_supertrend_RTS.py
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_supertrend_MIX.py --atr-period 10 --multiplier 3
    .\.venv\Scripts\python.exe -m unittest -v tests.test_chart_realtime_supertrend
История и текущие дельта-бары поступают из обычной LiveSession и общего QUIK-
сборщика. Supertrend рассчитывается по всей доступной истории до фильтрации дат,
как в chart_delta_bars_supertrend.py. Изменение свечи пересчитывает её ATR,
сохраняя предыдущие бары. Обе ступенчатые ветви обновляются на прежнем холсте.
ATR и множитель меняются в GUI/CLI; флажок скрывает линии без остановки приёма.
Компактная общая легенда показывает параметры и цвета обеих ветвей.
"""

import numpy as np
import pandas as pd
from PyQt6 import QtWidgets
import finplot as fplt

from chart_delta_bars_supertrend import DEFAULT_ATR_PERIOD, DEFAULT_MULTIPLIER, calculate_supertrend, validate_parameters
from source.realtime_chart import ChartCanvas, LiveSession, RealtimeWindow


class SupertrendSession(LiveSession):
    """Добавляет Supertrend с полным прогревом к сессии Семафора и текущих свечей."""

    def __init__(self, history, symbol, dataset_id, start, today, threshold_provider, settings):
        """Создаёт обычную сессию с ATR settings.atr_period и множителем settings.multiplier.

        history/symbol/dataset_id/start/today/threshold_provider имеют смысл
        LiveSession. Оставшиеся settings задают Семафор/SMA. Параметры проверяются
        до расчёта; исходный словарь настроек не изменяется.
        """
        semafor_settings = dict(settings)
        self.atr_period, self.multiplier = validate_parameters(
            semafor_settings.pop("atr_period", DEFAULT_ATR_PERIOD),
            semafor_settings.pop("multiplier", DEFAULT_MULTIPLIER))
        super().__init__(history, symbol, dataset_id, start, today, threshold_provider, semafor_settings)

    def calculate_data(self, data):
        """Возвращает Семафор/SMA и Supertrend всей data; сохраняет параметры ATR в attrs."""
        calculated = super().calculate_data(data)
        indicators = calculate_supertrend(calculated, self.atr_period, self.multiplier)
        for column in indicators:
            calculated[column] = indicators[column]
        calculated.attrs["supertrend_period"] = self.atr_period
        calculated.attrs["supertrend_multiplier"] = self.multiplier
        return calculated


class SupertrendCanvas(ChartCanvas):
    """Обновляет свечи, Семафор/SMA и две ступенчатые ветви на прежних объектах."""

    def __init__(self, owner, data, offset, preset):
        """Добавляет к холсту owner/data/offset/preset синий и красный Supertrend."""
        super().__init__(owner, data, offset, preset)
        self.title = "Дельта-бары · Семафор · Supertrend"
        period, multiplier = data.attrs["supertrend_period"], data.attrs["supertrend_multiplier"]
        self.supertrend_items = []
        for column, color in (("supertrend_up", "#254bff"), ("supertrend_down", "#ef2525")):
            item = fplt.plot(data.x, data[column], color=color, width=3, ax=self.price_axis)
            item.setDownsampling(ds=1, auto=False)
            item.setZValue(18)
            self.supertrend_items.append(item)
            self._set_steps(item, data, column)
        # Одна строка согласована с существующими строками Семафора/SMA в finplot.
        fplt.add_legend(f'Supertrend {period} × {multiplier:g} · '
                        '<span style="color:#254bff">рост</span> / '
                        '<span style="color:#ef2525">снижение</span>', ax=self.price_axis)
        fplt.refresh()

    def _set_steps(self, item, data, column):
        """Рисует item по data[column] ступенями; продлевает последний отрезок на полбара."""
        if data.empty:
            return
        x = data.x.to_numpy(dtype=float)
        y = data[column].to_numpy(dtype=float) / self.price_axis.vb.yscale.scalef
        item.setData(np.append(x, x[-1] + 0.5), np.append(y, y[-1]),
                     stepMode="right", connect="finite")

    def update_frame(self, data, follow=True):
        """Обновляет две линии и обычный кадр data; follow сохраняет выбранное правило слежения."""
        for item, column in zip(self.supertrend_items, ("supertrend_up", "supertrend_down")):
            item.update_data(data[["x", column]], gfx=False)
            self._set_steps(item, data, column)
        super().update_frame(data, follow)

    def set_supertrend_visible(self, visible):
        """Показывает обе ветви при visible=True; расчёт и приём сделок продолжаются."""
        for item in self.supertrend_items:
            item.setVisible(visible)

    def describe_bar(self, index):
        """Дополняет карточку index значением Supertrend, направлением и ATR либо сообщением прогрева."""
        super().describe_bar(index)
        if not 0 <= index < len(self.data):
            return
        bar, period = self.data.iloc[index], self.data.attrs["supertrend_period"]
        if pd.isna(bar.supertrend):
            text = f"Supertrend: прогрев ATR ({period} баров)"
        else:
            direction = "рост" if bar.supertrend_direction == 1 else "снижение"
            text = f"Supertrend {bar.supertrend:g} · {direction} · ATR({period}) {bar.atr:g}"
        self.owner.details.setText(self.owner.details.text() + "\n" + text)


class SupertrendWindow(RealtimeWindow):
    """Сохраняет обычные реал-тайм параметры и добавляет Supertrend, ATR и множитель."""

    session_type, canvas_type = SupertrendSession, SupertrendCanvas

    def __init__(self, *args, atr_period=DEFAULT_ATR_PERIOD, multiplier=DEFAULT_MULTIPLIER, **kwargs):
        """Открывает обычный график args/kwargs с параметрами atr_period/multiplier Supertrend."""
        self.initial_period, self.initial_multiplier = validate_parameters(atr_period, multiplier)
        super().__init__(*args, **kwargs)
        self.setWindowTitle(self.windowTitle() + f" · Supertrend ({self.initial_period}, {self.initial_multiplier:g})")

    def _build_controls(self, symbol):
        """Добавляет строку настройки Supertrend к исходному окну symbol."""
        super()._build_controls(symbol)
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
        options.addWidget(self.multiplier_spin)
        options.addWidget(QtWidgets.QLabel("Рост — синий · Снижение — красный"))
        options.addStretch()
        self.centralWidget().layout().insertLayout(4, options)
        self.details.setMinimumHeight(88)

    def _connect_controls(self):
        """Подключает ATR/множитель к пересчёту, а флажок Supertrend только к видимости."""
        super()._connect_controls()
        self.period_spin.editingFinished.connect(self.start_load)
        self.multiplier_spin.editingFinished.connect(self.start_load)
        self.supertrend_check.toggled.connect(self.apply_visibility)

    def set_busy(self, busy):
        """Блокирует периоды при загрузке busy; флажок видимости доступен."""
        super().set_busy(busy)
        self.period_spin.setEnabled(not busy)
        self.multiplier_spin.setEnabled(not busy)

    def current_request(self):
        """Возвращает запрос общего worker с классом сессии и текущими параметрами Supertrend."""
        request = super().current_request()
        period, multiplier = validate_parameters(self.period_spin.value(), self.multiplier_spin.value())
        request["settings"].update(atr_period=period, multiplier=multiplier)
        return request

    def apply_visibility(self, *args):
        """Применяет прежние флажки и Supertrend; args являются параметрами сигналов Qt."""
        super().apply_visibility(*args)
        if self.canvas is not None:
            self.canvas.set_supertrend_visible(self.supertrend_check.isChecked())

    def accept_session(self, session):
        """Принимает session через обычное окно; дополняет заголовок и статус прогрева ATR."""
        super().accept_session(session)
        if self._closing or self.canvas is None:
            return
        period, multiplier = session.data.attrs["supertrend_period"], session.data.attrs["supertrend_multiplier"]
        self.setWindowTitle(self.windowTitle() + f" · Supertrend ({period}, {multiplier:g})")
        if not session.data.empty and session.data.supertrend.isna().all():
            self.status_label.setText(self.status_label.text() + f" · Supertrend: прогрев ATR {period} баров")
