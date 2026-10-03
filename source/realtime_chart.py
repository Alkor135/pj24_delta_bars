"""Общий реал-тайм график Семафора: история SQLite и сделки общего сборщика QUIK.

Примеры из корня проекта:
    python chart_delta_bar_semafor_realtime_RTS.py
    python chart_delta_bar_semafor_realtime_MIX.py --refresh-ms 200
    python chart_delta_bar_semafor_realtime_RTS.py --threshold 100 --start 2026-09-01
    python -m unittest -v tests.test_chart_realtime
Приём сделок независим от GUI. Фоновый worker читает журнал с курсором,
пересчитывает Семафор/SMA; GUI обновляет прежний холст и сохраняет масштаб.
Исторические БД только читаются. Текущий незавершённый бар имеет причину live.
Изменение ночных источников автоматически обновляет историю. До первой сделки
уточняется порог; после начала дня изменение порога требует кнопки «Обновить».
"""

import argparse
from datetime import date
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets
import finplot as fplt

from chart_delta_bar_semafor import DEFAULT_DEPTHS, PRESETS, ChartWindow as SemaforWindow, calculate_indicators, validate_settings
from chart_delta_bar_semafor_simulate import ChartCanvas
from chart_delta_bars import DEFAULT_DATA_DIR, LoadWorker as BaseLoadWorker, create_application, valid_date
from source.chart_data import SELECT_COLUMNS
from source.realtime_core import DayBuilder, Trade, positive_integer
from source.delta_core import format_time
from source.realtime_data import DEFAULT_THRESHOLD_FILE, load_history, resolve_threshold, select_dataset, source_revision
from source.realtime_feed import DEFAULT_CONFIG, ensure_collector, feed_request, load_config, moscow_day


class LiveSession:
    """Согласует исторические дни и изменяемые бары журнала, сохраняя прогрев индикаторов."""

    def __init__(self, history, symbol, dataset_id, start, today, threshold_provider, settings):
        """Создаёт сессию history/symbol; threshold_provider(day) возвращает порог и его источник.

        dataset_id/start — набор и начало видимой истории; today — текущая дата
        МСК; settings — параметры Семафора/SMA. История включает весь прогрев.
        """
        self.symbol, self.dataset_id = symbol, dataset_id
        self.start, self.today = date.fromisoformat(str(start)).isoformat(), date.fromisoformat(str(today)).isoformat()
        self.threshold_provider, self.settings = threshold_provider, settings
        self.history = history[SELECT_COLUMNS.split(",") + ["duration_seconds"]].copy()
        self.builders, self.thresholds = {}, {}
        self.source_revision, self.source_warning = None, ""
        self.cursor, self.version = 0, 0
        self.status = dict(message="Загрузка данных QUIK…", contracts={}, feed_connected=False)
        self._calculate()

    def ensure_threshold(self, day):
        """Возвращает и запоминает постоянный порог day; ошибки источника не подменяются."""
        if day not in self.thresholds:
            self.thresholds[day] = self.threshold_provider(day)
        return self.thresholds[day]

    def refresh_threshold(self, day):
        """Уточняет порог day до первой сделки; после начала сохраняет его и сообщает об изменении."""
        candidate = self.threshold_provider(day)
        previous, builder = self.thresholds.get(day), self.builders.get(day)
        if previous and builder and builder.trades and candidate["threshold"] != previous["threshold"]:
            self.source_warning = (f"Ночной порог изменился: {previous['threshold']} → {candidate['threshold']}. "
                                   "Нажмите «Обновить» для пересборки дня.")
        else:
            self.thresholds[day] = candidate
            self.source_warning = ""
        return self.thresholds[day]

    def _calculate(self):
        """Рассчитывает Семафор/SMA по истории и уже поступившим барам; обновляет data."""
        known_days = set(self.history.day)
        records = [dict(bar, dataset_id=self.dataset_id)
                   for day, builder in sorted(self.builders.items()) if day not in known_days
                   for bar in builder.snapshot(final=day < self.today)]
        live = pd.DataFrame(records)
        if not live.empty:
            for name in ("start_time", "end_time"):
                live[name] = pd.to_datetime(live[name], format="ISO8601")
            live["duration_seconds"] = (live.end_time - live.start_time).dt.total_seconds()
            raw = pd.concat([self.history, live[self.history.columns]], ignore_index=True)
        else:
            raw = self.history.copy()
        raw = raw.sort_values(["day", "bar_index"]).reset_index(drop=True)
        calculated = calculate_indicators(raw, **self.settings)
        self.data = calculated.loc[calculated.day >= self.start].copy().reset_index(drop=True)
        self.data["x"] = np.arange(len(self.data), dtype=np.int64)
        self.version += 1

    def ingest(self, trades):
        """Учитывает Trade из trades и пересчитывает график при изменениях; возвращает bool."""
        known_days, batches = set(self.history.day), {}
        for trade in trades:
            if trade.day in known_days:
                continue  # Дни ночной БД являются единственным источником исторических баров.
            if trade.day > self.today:
                raise ValueError("Журнал содержит сделки будущего календарного дня")
            batches.setdefault(trade.day, []).append(trade)
        changed = False
        for day, batch in sorted(batches.items()):
            if day not in self.builders:
                threshold = self.ensure_threshold(day)["threshold"]
                self.builders[day] = DayBuilder(day, threshold, self.symbol)
            changed = self.builders[day].ingest(batch) or changed
        if changed:
            self._calculate()
        return changed

    def change_day(self, day, history=None):
        """Переключает дату day, закрывает прошлые остатки; history заменяет ночную историю."""
        day = date.fromisoformat(str(day)).isoformat()
        if day < self.today:
            raise ValueError("Нельзя переводить текущий день назад")
        if day != self.today:
            self.source_warning = ""
        self.today = day
        if history is not None:
            self.history = history[self.history.columns].copy()
            for known in set(self.history.day):
                self.builders.pop(known, None)
        self._calculate()


class RealtimeWorker(BaseLoadWorker):
    """Читает историю/порог либо новые сделки и рассчитывает кадр вне GUI."""

    def __init__(self, request, parent=None, session=None):
        """Сохраняет request источника и прежнюю session для очередного обновления."""
        super().__init__(request, parent)
        self.session = session

    def run(self):
        """Передаёт готовую сессию через loaded; сетевые ошибки сохраняют предыдущий кадр."""
        try:
            request, today = self.request, moscow_day()
            config = load_config(request["config_path"])
            revision = source_revision(request["db"], request["threshold_file"])
            if self.session is None:
                selected = select_dataset(request["db"], request["symbol"], request["dataset_id"])
                history = load_history(request["db"], request["symbol"], selected, today)

                def threshold_provider(day):
                    """Возвращает проверенный порог day выбранного набора без записи истории."""
                    return resolve_threshold(request["db"], request["symbol"], day, selected,
                                             request["threshold_file"], request["manual_threshold"])

                session = LiveSession(history, request["symbol"], selected, request["start"], today,
                                      threshold_provider, request["settings"])
            else:
                session = self.session
                if session.today != today or session.source_revision != revision:
                    history = load_history(request["db"], session.symbol, session.dataset_id, today)
                    session.change_day(today, history)
            try:
                if session.source_revision != revision or today not in session.thresholds:
                    session.refresh_threshold(today)
                    session.source_revision = revision
                ensure_collector(config, request["config_path"])
                session.ensure_threshold(today)
                collected, cursor = [], session.cursor
                while not self.isInterruptionRequested():
                    packet = feed_request(config, "trades", symbol=session.symbol, after=cursor)
                    session.status = packet["status"]
                    collected.extend(Trade(**row) for row in packet["trades"])
                    cursor = packet["cursor"]
                    if not packet["more"]:
                        break
                if not self.isInterruptionRequested():
                    session.ingest(collected)
                    session.cursor = cursor
            except (OSError, ValueError, KeyError, TypeError) as exc:
                session.status = dict(session.status, feed_connected=False, message=f"Ожидание данных: {exc}")
            if not self.isInterruptionRequested():
                self.loaded.emit(session)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class RealtimeWindow(SemaforWindow):
    """Отдельное окно фиксированного инструмента с обновлением текущей свечи и слежением."""

    def __init__(self, symbol, data_dir=DEFAULT_DATA_DIR, db_path=None, start=None,
                 config_path=DEFAULT_CONFIG, threshold_file=DEFAULT_THRESHOLD_FILE,
                 manual_threshold=None, dataset_id=None, refresh_ms=200, auto_load=True, **settings):
        """Открывает symbol из db_path/data_dir; параметры задают источник, порог и частоту.

        config_path — общий JSON обоих окон; threshold_file — ночной кэш;
        manual_threshold — необязательный ручной порог; settings — настройки
        исходного Семафора. auto_load=False используется для автономной проверки GUI.
        """
        self._initialized, self._reload_requested = False, False
        self.session, self._drawn_version, self._retry_at = None, None, 0
        self.config_path, self.threshold_file = Path(config_path), Path(threshold_file)
        self.manual_threshold = manual_threshold
        self.refresh_ms = positive_integer(refresh_ms, "Интервал обновления")
        if not 50 <= self.refresh_ms <= 10000:
            raise ValueError("Интервал обновления должен быть 50…10000 мс")
        self.auto_running, self._canvas_preset = auto_load, None
        super().__init__(symbol, data_dir, db_path, start, moscow_day(), auto_load=False, **settings)
        if dataset_id is not None:
            index = self.dataset_combo.findData(dataset_id)
            if index < 0:
                raise ValueError(f"Не найден набор {dataset_id}")
            self.dataset_combo.setCurrentIndex(index)
        self.symbol_combo.setEnabled(False)
        self.end_edit.setDate(QtCore.QDate.fromString(moscow_day(), "yyyy-MM-dd"))
        self.end_edit.setEnabled(False)
        self.setWindowTitle(f"{symbol} · Семафор · Реальное время QUIK")
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.refresh_ms)
        self.timer.timeout.connect(self.next_frame)
        self._initialized = True
        if auto_load:
            self.start_load()
            self.timer.start()

    def _build_controls(self, symbol):
        """Дополняет исходный график symbol слежением и состоянием реального потока."""
        super()._build_controls(symbol)
        self.refresh_button.setToolTip("Перечитать историю и журнал · Ctrl+R")
        row = QtWidgets.QHBoxLayout()
        self.follow_check = QtWidgets.QCheckBox("Следить за последним")
        self.follow_check.setChecked(True)
        row.addWidget(self.follow_check)
        self.feed_label = QtWidgets.QLabel("QUIK: ожидание · Полнота текущего дня не подтверждена")
        self.feed_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.feed_label)
        row.addStretch()
        self.centralWidget().layout().insertLayout(4, row)

    def set_busy(self, busy):
        """Блокирует параметры при загрузке busy, сохраняя фиксированный symbol и дату конца."""
        super().set_busy(busy)
        self.symbol_combo.setEnabled(False)
        self.end_edit.setEnabled(False)

    def current_request(self):
        """Возвращает независимые параметры источника и индикаторов для фонового worker."""
        settings = validate_settings(tuple(spin.value() for spin in self.depth_spins), self.deviation_spin.value(),
                                     self.backstep_spin.value(), self.point_spin.value(),
                                     self.ma_fast_spin.value(), self.ma_slow_spin.value())
        return dict(db=self.database_path(), symbol=self.symbol_combo.currentText(),
                    dataset_id=self.dataset_combo.currentData(), start=self.start_edit.date().toString("yyyy-MM-dd"),
                    config_path=self.config_path, threshold_file=self.threshold_file,
                    manual_threshold=self.manual_threshold, settings=settings)

    def _start_worker(self, session=None):
        """Запускает одну загрузку либо обновление session; одновременно второй worker не создаётся."""
        request = self.current_request()
        if request["start"] > moscow_day():
            raise ValueError("Начальная дата позже текущего дня")
        if session is None:
            self.set_busy(True)
            self.status_label.setText("Загрузка истории, подготовка порога и подключение QUIK…")
        else:
            self.loading = True
        self.worker = RealtimeWorker(request, self, session)
        self.worker.loaded.connect(self.accept_session)
        self.worker.failed.connect(self.load_failed)
        self.worker.finished.connect(self.load_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def start_load(self, *args):
        """Перечитывает историю и журнал по текущим настройкам; args — параметры сигнала Qt."""
        if not self._initialized or self._closing:
            return
        if self.loading:
            self._reload_requested = True
            return
        if not self.dataset_combo.count():
            return
        try:
            self._start_worker()
        except ValueError as exc:
            self.load_failed(str(exc))

    def next_frame(self):
        """Запрашивает очередные сделки по таймеру, объединяя обновления занятого GUI."""
        if self._closing or self.loading or time.monotonic() < self._retry_at:
            return
        if self.session is None:
            self.start_load()
        else:
            try:
                self._start_worker(self.session)
            except ValueError as exc:
                self.load_failed(str(exc))

    def accept_session(self, session):
        """Принимает session, обновляет прежний холст при изменениях и сообщает состояние связи."""
        if self._closing:
            return
        previous = self.session
        self.session, self.data = session, session.data
        preset = self.preset_combo.currentData()
        try:
            if self.canvas is None or previous is not session or preset != self._canvas_preset:
                self.redraw()
            elif self._drawn_version != session.version and not self.data.empty:
                self.canvas.update_frame(self.data, self.follow_check.isChecked())
            self._drawn_version = session.version
        except Exception as exc:
            self.load_failed(f"Не удалось обновить график: {exc}")
            return
        status = session.status
        contract = status.get("contracts", {}).get(session.symbol, {}).get("sec_code", "не выбран")
        threshold = session.thresholds.get(session.today)
        threshold_text = (f"Порог {threshold['threshold']} · {threshold['source']}" if threshold else "Порог не подготовлен")
        last = status.get("last_trades", {}).get(session.symbol)
        last_text = f" · последняя сделка {format_time(last['day'], last['time_ns'])}" if last else ""
        quality = f" · ошибок сделок: {status.get('rejected_trades', 0)}" if status.get("rejected_trades") else ""
        self.feed_label.setText(f"{contract} · {threshold_text} · {status.get('message', 'Ожидание')}{last_text}{quality}")
        warning = f" · {session.source_warning}" if session.source_warning else ""
        self.status_label.setText((f"{len(self.data):,} баров · обновление {self.refresh_ms} мс · Полнота дня не подтверждена" + warning).replace(",", " "))
        self.status_label.setToolTip(session.source_warning)
        self.setWindowTitle(f"{session.symbol} · {contract} · Семафор · Реальное время QUIK")
        if not status.get("feed_connected"):
            self._retry_at = time.monotonic() + 3
        else:
            self._retry_at = 0

    def redraw(self):
        """Пересоздаёт холст при новых настройках; обычные сделки используют update_frame."""
        old_range = self.canvas.price_axis.vb.viewRange()[0] if self.canvas is not None else None
        if self.canvas is not None:
            self.chart_layout.removeWidget(self.canvas)
            self.canvas.dispose()
            self.canvas = None
        self.placeholder.setVisible(self.data.empty)
        if self.data.empty:
            self.placeholder.setText("Ожидание первых баров")
            return
        self.canvas = ChartCanvas(self, self.data, self.offset_spin.value(), self.preset_combo.currentData())
        self._canvas_preset = self.preset_combo.currentData()
        self.chart_layout.addWidget(self.canvas)
        self.apply_visibility()
        if old_range is not None:
            fplt.set_x_pos(*old_range, ax=self.canvas.price_axis)

    def load_failed(self, message):
        """Показывает message и сохраняет предыдущие бары; повтор разрешён через три секунды."""
        self.status_label.setText(f"Ошибка: {message}. Предыдущий график сохранён.")
        self._retry_at = time.monotonic() + 3

    def load_finished(self):
        """Освобождает завершённый worker и выполняет отложенное изменение настроек."""
        self.worker = None
        if self._closing:
            return
        self.set_busy(False)
        if self._reload_requested and not self._closing:
            self._reload_requested = False
            QtCore.QTimer.singleShot(0, self.start_load)

    def closeEvent(self, event):
        """Останавливает таймер/worker перед закрытием event; общий сборщик остаётся для другого окна."""
        self._closing = True
        self.timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        super().closeEvent(event)


def main(symbol, argv=None):
    """Разбирает argv и запускает фиксированный график symbol; возвращает код завершения Qt."""
    parser = argparse.ArgumentParser(description=f"{symbol}: дельта-бары и Семафор в реальном времени из QUIK")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="общая конфигурация QUIK")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="папка исторических баз")
    parser.add_argument("--db", type=Path, help="явная историческая SQLite")
    parser.add_argument("--dataset-id", help="явный набор вместо последнего")
    parser.add_argument("--start", type=valid_date, help="начало видимого диапазона YYYY-MM-DD")
    parser.add_argument("--threshold-file", type=Path, default=DEFAULT_THRESHOLD_FILE, help="ночной JSON порогов")
    parser.add_argument("--threshold", type=int, help="явный ручной порог вместо адаптивного")
    parser.add_argument("--refresh-ms", type=int, default=200, help="обновление графика 50…10000 мс")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="nuf", help="стиль NUF/FXi")
    parser.add_argument("--depths", type=int, nargs=3, default=DEFAULT_DEPTHS, help="три глубины Семафора")
    parser.add_argument("--deviation", type=int, help="допуск в единицах Point")
    parser.add_argument("--backstep", type=int, help="удаление близких экстремумов")
    parser.add_argument("--point", type=float, default=1, help="единица цены для допуска")
    parser.add_argument("--ma-fast", type=int, default=3, help="период быстрой SMA")
    parser.add_argument("--ma-slow", type=int, default=34, help="период медленной SMA")
    args = parser.parse_args(argv)
    try:
        load_config(args.config)
        if args.threshold is not None:
            positive_integer(args.threshold, "Ручной порог")
        dev, back = PRESETS[args.preset]
        settings = validate_settings(args.depths, dev if args.deviation is None else args.deviation,
                                     back if args.backstep is None else args.backstep, args.point, args.ma_fast, args.ma_slow)
        if not 50 <= args.refresh_ms <= 10000:
            raise ValueError("Интервал обновления должен быть 50…10000 мс")
        if args.start and args.start > moscow_day():
            raise ValueError("Начальная дата позже текущего дня")
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    app = create_application()
    window = RealtimeWindow(symbol, args.data_dir, args.db, args.start, args.config, args.threshold_file,
                            args.threshold, args.dataset_id, args.refresh_ms, preset=args.preset, **settings)
    window.show()
    return app.exec()
