"""Хранилище дельта-баров и дневного прогресса в SQLite.

Используется конвертером: python tick_to_delta_bars.py --symbols RTS MIX
Проверки: python -m unittest -v tests.test_delta_bars.PipelineTests
Бары и прогресс одного дня фиксируются одной транзакцией.
"""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

SCHEMA_VERSION = 1
ALGORITHM_VERSION = "adaptive_tick_rule_v1"
BAR_COLUMNS = ("bar_index", "start_time", "end_time", "start_row", "end_row", "open", "high", "low",
               "close", "volume", "delta", "up_volume", "down_volume", "neutral_volume", "tick_count",
               "threshold", "is_complete", "close_reason")
DAY_COLUMNS = ("file_path", "sha256", "tick_count", "volume", "active_intervals", "status", "threshold",
               "window_start", "window_end", "window_days", "target_count", "calibration_count",
               "calibration_error", "calibration_evaluations", "bar_count")

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS days (
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    symbol TEXT NOT NULL,
    day TEXT NOT NULL,
    file_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    tick_count INTEGER NOT NULL,
    volume INTEGER NOT NULL,
    active_intervals INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('ready','warmup','prestart','empty')),
    threshold INTEGER,
    window_start TEXT,
    window_end TEXT,
    window_days INTEGER NOT NULL,
    target_count INTEGER,
    calibration_count INTEGER,
    calibration_error REAL,
    calibration_evaluations INTEGER,
    bar_count INTEGER NOT NULL,
    PRIMARY KEY(dataset_id, symbol, day)
);
CREATE TABLE IF NOT EXISTS bars (
    dataset_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    day TEXT NOT NULL,
    bar_index INTEGER NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    start_row INTEGER NOT NULL,
    end_row INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL CHECK(volume > 0),
    delta INTEGER NOT NULL,
    up_volume INTEGER NOT NULL,
    down_volume INTEGER NOT NULL,
    neutral_volume INTEGER NOT NULL,
    tick_count INTEGER NOT NULL CHECK(tick_count > 0),
    threshold INTEGER NOT NULL CHECK(threshold > 0),
    is_complete INTEGER NOT NULL CHECK(is_complete IN (0,1)),
    close_reason TEXT NOT NULL CHECK(close_reason IN ('threshold','day_end')),
    PRIMARY KEY(dataset_id, symbol, day, bar_index),
    FOREIGN KEY(dataset_id, symbol, day) REFERENCES days(dataset_id, symbol, day) ON DELETE CASCADE,
    CHECK(volume = up_volume + down_volume + neutral_volume),
    CHECK(delta = up_volume - down_volume)
);
CREATE INDEX IF NOT EXISTS bars_period ON bars(dataset_id, symbol, start_time);
"""


def open_database(path):
    """Открывает или создаёт базу, проверяя её версию и чужие таблицы до изменения."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if version not in (0, SCHEMA_VERSION) or (version == 0 and tables):
            raise ValueError("Файл SQLite имеет другую схему; укажите отдельный --db")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript("BEGIN IMMEDIATE;\n" + SCHEMA + f"\nPRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;")
        return connection
    except BaseException:
        connection.close()
        raise


@contextmanager
def transaction(connection):
    """Атомарно подтверждает изменения либо откатывает их при ошибке или Ctrl+C."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def register_dataset(connection, input_root, start, lookback, minutes):
    """Разделяет несовместимые настройки и сохраняет воспроизводимое описание алгоритма."""
    config = {"algorithm": ALGORITHM_VERSION, "source_root": str(Path(input_root).resolve()),
              "output_start": start.isoformat(), "lookback_weekdays_with_ticks": lookback,
              "target_minutes": minutes, "timezone": "Europe/Moscow", "utc_offset": "+03:00",
              "direction": "tick_rule_carry_nonzero_daily_reset", "first_direction": "neutral",
              "day_end": "save_partial_and_reset", "overshoot": "whole_trade_no_carry",
              "weekend_bars": True, "weekends_in_training": False,
              "calibration_count_includes_partial": True,
              "timestamp_precision": "source_synthetic_fraction_preserved"}
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False)
    dataset_id = sha256(encoded.encode()).hexdigest()[:16]
    with transaction(connection):
        connection.execute("INSERT OR IGNORE INTO datasets(dataset_id,config_json) VALUES (?,?)", (dataset_id, encoded))
    return dataset_id


def recorded_days(connection, dataset_id, symbol):
    """Возвращает дневной журнал конкретного инструмента и набора настроек."""
    return {row["day"]: dict(row) for row in connection.execute(
        "SELECT * FROM days WHERE dataset_id=? AND symbol=? ORDER BY day", (dataset_id, symbol))}


def rewind(connection, dataset_id, symbol, day):
    """Удаляет производные дни и бары начиная с изменившегося источника."""
    with transaction(connection):
        connection.execute("DELETE FROM days WHERE dataset_id=? AND symbol=? AND day>=?", (dataset_id, symbol, day))


def save_day(connection, dataset_id, symbol, day, metadata, bars):
    """Записывает статистику, прогресс и все бары дня одной транзакцией."""
    day_fields = ("dataset_id", "symbol", "day") + DAY_COLUMNS
    bar_fields = ("dataset_id", "symbol", "day") + BAR_COLUMNS
    with transaction(connection):
        connection.execute(f"INSERT INTO days ({','.join(day_fields)}) VALUES ({','.join('?' for _ in day_fields)})",
                           (dataset_id, symbol, day) + tuple(metadata[key] for key in DAY_COLUMNS))
        connection.executemany(f"INSERT INTO bars ({','.join(bar_fields)}) VALUES ({','.join('?' for _ in bar_fields)})",
                               [(dataset_id, symbol, day) + tuple(bar[key] for key in BAR_COLUMNS) for bar in bars])
