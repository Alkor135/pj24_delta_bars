"""Чтение дельта-баров SQLite и расчёт индикаторов для визуального анализа.

Запуск просмотрщика: python chart_delta_bars.py --symbol RTS
Проверки расчётов: python -m unittest -v tests.test_chart_data
База открывается только для чтения. Индикаторы в неё не записываются.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

SIGNALS = ("long_rising", "short_rising", "long_falling", "short_falling")
SELECT_COLUMNS = ("dataset_id,symbol,day,bar_index,start_time,end_time,open,high,low,close,"
                  "volume,delta,threshold,tick_count,is_complete,close_reason")


@dataclass(frozen=True)
class DatasetInfo:
    """Описывает доступный набор настроек и границы его баров."""

    dataset_id: str
    first_day: str
    last_day: str
    bar_count: int
    label: str


@contextmanager
def readonly_connection(path):
    """Открывает существующую SQLite для чтения и гарантированно закрывает соединение."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Не найдена база: {path}")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def inspect_database(path, symbol):
    """Возвращает наборы только выбранного инструмента, не смешивая разные настройки."""
    with readonly_connection(path) as connection:
        rows = connection.execute("""
            SELECT d.dataset_id,d.config_json,min(b.day),max(b.day),count(*)
            FROM datasets d JOIN bars b ON b.dataset_id=d.dataset_id
            WHERE b.symbol=? GROUP BY d.dataset_id ORDER BY d.created_at DESC,d.dataset_id
        """, (symbol,)).fetchall()
    result = []
    for dataset_id, config_json, first_day, last_day, count in rows:
        config = json.loads(config_json)
        lookback = config.get("lookback_weekdays_with_ticks", "?")
        minutes = config.get("target_minutes", "?")
        label = f"{lookback} дней → {minutes} мин · с {first_day} · {dataset_id[:6]}"
        result.append(DatasetInfo(dataset_id, first_day, last_day, count, label))
    return result


def laguerre(prices, alpha=0.4):
    """Вычисляет рекурсию Лагерра образца с инициализацией первой ценой вместо нулей."""
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Параметр ALF α должен быть больше 0 и не больше 1")
    prices = np.asarray(prices, dtype=float)
    result = np.empty(len(prices), dtype=float)
    if not len(prices):
        return result
    l0 = l1 = l2 = l3 = float(prices[0])
    gamma = 1 - alpha
    for i, price in enumerate(prices):
        old0, old1, old2 = l0, l1, l2
        l0 = alpha * float(price) + gamma * old0
        l1 = -gamma * l0 + old0 + gamma * l1
        l2 = -gamma * l1 + old1 + gamma * l2
        l3 = -gamma * l2 + old2 + gamma * l3
        result[i] = (l0 + 2 * l1 + 2 * l2 + l3) / 6
    return result


def volume_stops(data):
    """Находит четыре исходных объёмных шаблона среди трёх полных баров одного дня."""
    size = len(data)
    result = {name: np.zeros(size, dtype=bool) for name in SIGNALS}
    if size < 3:
        return result
    volume = data["volume"].to_numpy()
    opens, closes = data["open"].to_numpy(), data["close"].to_numpy()
    days = data["day"].to_numpy()
    complete = data["is_complete"].to_numpy() == 1
    valid = complete[:-2] & complete[1:-1] & complete[2:]
    valid &= (days[:-2] == days[1:-1]) & (days[1:-1] == days[2:])
    rising = (volume[:-2] < volume[1:-1]) & (volume[1:-1] < volume[2:])
    falling = (volume[:-2] > volume[1:-1]) & (volume[1:-1] > volume[2:])
    bullish, bearish = opens <= closes, opens >= closes
    long = bearish[:-2] & bearish[1:-1] & bullish[2:]
    short = bullish[:-2] & bullish[1:-1] & bearish[2:]
    result["long_rising"][2:] = valid & rising & long
    result["short_rising"][2:] = valid & rising & short
    result["long_falling"][2:] = valid & falling & long
    result["short_falling"][2:] = valid & falling & short
    return result


def load_bars(path, symbol, dataset_id, start, end, alpha=0.4):
    """Читает историю до конца диапазона, прогревает ALF и возвращает видимые бары."""
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
    data["start_time"] = pd.to_datetime(data["start_time"], format="ISO8601", errors="raise")
    data["end_time"] = pd.to_datetime(data["end_time"], format="ISO8601", errors="raise")
    data["duration_seconds"] = (data["end_time"] - data["start_time"]).dt.total_seconds()
    if (data["duration_seconds"] < 0).any():
        raise ValueError("Конец бара предшествует его началу")
    data["alf"] = laguerre(data["close"].to_numpy(), alpha)
    for name, values in volume_stops(data).items():
        data[name] = values
    visible = data.loc[data["day"] >= start.isoformat()].copy().reset_index(drop=True)
    visible["x"] = np.arange(len(visible), dtype=np.int64)
    return visible
