"""Читает историю и точно подготавливает дневной порог реал-тайм.

Примеры из корня: python prepare_realtime_thresholds.py --date 2026-10-05
Проверки: python -m unittest -v tests.test_realtime_data
Текущий день исключён из калибровки. SQLite только читается; кэш проверяется
по дате, набору и метаданным обучающих ZIP из таблицы days.
Открытые окна отслеживают изменение SQLite/WAL и JSON без записи этих источников.
"""

from datetime import date, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from source.chart_data import inspect_database, readonly_connection
from source.delta_core import calibrate, read_day
from source.delta_store import ALGORITHM_VERSION
from source.realtime_core import positive_integer
from tick_to_delta_bars import fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD_FILE = PROJECT_ROOT / ".live_cache" / "thresholds.json"


def source_revision(db, threshold_file):
    """Возвращает метаданные db, её WAL и threshold_file для обнаружения ночного обновления."""
    revision = []
    for path in (Path(db), Path(str(db) + "-wal"), Path(threshold_file)):
        try:
            stat = path.stat()
            revision.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size, stat.st_ino))
        except FileNotFoundError:
            revision.append((str(path.resolve()), None))
    return tuple(revision)


def select_dataset(db, symbol, dataset_id=None):
    """Возвращает dataset_id для symbol в db; без явного значения выбирает последний набор."""
    datasets = inspect_database(db, symbol)
    if not datasets:
        raise ValueError(f"В базе нет истории {symbol}")
    selected = dataset_id or datasets[0].dataset_id
    if selected not in {item.dataset_id for item in datasets}:
        raise ValueError(f"В базе нет набора {selected} для {symbol}")
    return selected


def calibration_window(db, symbol, day, dataset_id=None):
    """Возвращает набор, конфигурацию и последние обучающие дни строго до day."""
    day = date.fromisoformat(str(day))
    selected = select_dataset(db, symbol, dataset_id)
    with readonly_connection(db) as connection:
        config = json.loads(connection.execute("SELECT config_json FROM datasets WHERE dataset_id=?", (selected,)).fetchone()[0])
        expected = dict(algorithm=ALGORITHM_VERSION, direction="tick_rule_carry_nonzero_daily_reset",
                        first_direction="neutral", overshoot="whole_trade_no_carry", day_end="save_partial_and_reset")
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Набор истории использует несовместимые правила дельта-баров")
        lookback = positive_integer(config["lookback_weekdays_with_ticks"], "Окно калибровки")
        rows = connection.execute("""SELECT day,file_path,sha256,tick_count,volume,active_intervals FROM days
            WHERE dataset_id=? AND symbol=? AND day<? AND tick_count>0
              AND strftime('%w',day) NOT IN ('0','6') ORDER BY day DESC LIMIT ?""",
            (selected, symbol, day.isoformat(), lookback)).fetchall()
    if len(rows) != lookback:
        raise ValueError(f"Нужно {lookback} прошлых будних дней с тиками; найдено {len(rows)}")
    columns = ("day", "file_path", "sha256", "tick_count", "volume", "active_intervals")
    return selected, config, [dict(zip(columns, row)) for row in reversed(rows)]


def window_fingerprint(dataset_id, config, rows):
    """Возвращает SHA-256 параметров dataset_id/config и метаданных обучающих rows."""
    payload = json.dumps([dataset_id, config, rows], sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(payload).hexdigest()


def prepare_threshold(db, symbol, day, dataset_id=None):
    """Возвращает порог и метаданные для day по прошлым тикам symbol, не записывая db."""
    day = date.fromisoformat(str(day)).isoformat()
    selected, config, rows = calibration_window(db, symbol, day, dataset_id)
    history = []
    for row in rows:
        path = Path(row["file_path"])
        if not path.is_file():
            raise ValueError(f"Нет архива для калибровки: {path}")
        if fingerprint(path) != row["sha256"]:
            raise ValueError(f"Архив изменён после построения БД: {path}; обновите историю ночью")
        ticks = read_day(path, date.fromisoformat(row["day"]), config["target_minutes"])
        if fingerprint(path) != row["sha256"]:
            raise ValueError(f"Архив изменяется во время расчёта: {path}")
        if len(ticks.times) != row["tick_count"] or ticks.volume != row["volume"]:
            raise ValueError(f"Статистика архива расходится с БД: {path}")
        history.append(ticks.history())
    result = calibrate(history)
    return dict(symbol=symbol, day=day, dataset_id=selected, threshold=result.threshold,
                window_start=rows[0]["day"], window_end=rows[-1]["day"], window_days=len(rows),
                target_count=result.target_count, actual_count=result.actual_count,
                fingerprint=window_fingerprint(selected, config, rows), source="расчёт по прошлым тикам")


def resolve_threshold(db, symbol, day, dataset_id=None, threshold_file=DEFAULT_THRESHOLD_FILE, manual=None):
    """Возвращает порог из threshold_file или расчёта; manual задаёт явный ручной порог."""
    day = date.fromisoformat(str(day)).isoformat()
    selected = select_dataset(db, symbol, dataset_id)
    if manual is not None:
        return dict(symbol=symbol, day=day, dataset_id=selected,
                    threshold=positive_integer(manual, "Ручной порог"), source="ручной порог")
    selected, config, rows = calibration_window(db, symbol, day, selected)
    path = Path(threshold_file)
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("version") == 1:
                for entry in cached.get("thresholds", []):
                    if (entry.get("symbol"), entry.get("day"), entry.get("dataset_id"), entry.get("fingerprint")) == (
                            symbol, day, selected, window_fingerprint(selected, config, rows)):
                        positive_integer(entry["threshold"], "Подготовленный порог")
                        return dict(entry, source="ночная подготовка")
        except (ValueError, KeyError, TypeError, AttributeError):
            pass  # Повреждённый кэш не заменяет точный расчёт прошлых тиков.
    return prepare_threshold(db, symbol, day, selected)


def write_thresholds(path, entries):
    """Атомарно сохраняет entries в path, сохраняя пороги других дат и наборов."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("Неизвестная версия файла порогов")
        existing = payload["thresholds"]
    combined = {(entry["symbol"], entry["day"], entry["dataset_id"]): entry for entry in existing}
    combined.update({(entry["symbol"], entry["day"], entry["dataset_id"]): entry for entry in entries})
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(version=1, thresholds=list(combined.values())), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_history(db, symbol, dataset_id, today):
    """Возвращает все исходные бары symbol до today для прогрева, исключая сегодняшний день."""
    from chart_delta_bar_semafor import load_semafor_bars
    end = date.fromisoformat(str(today)) - timedelta(days=1)
    return load_semafor_bars(db, symbol, dataset_id, date.min.isoformat(), end.isoformat(),
                             depths=(0, 0, 0), ma_fast=1, ma_slow=1)
