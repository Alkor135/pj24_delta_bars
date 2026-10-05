r"""Читает тики/дельта-бары и согласует дневные окна по первой/последней сделке.

Запуск из корня проекта:
    .\.venv\Scripts\python.exe -m research.volume_trend --stage audit
    .\.venv\Scripts\python.exe -m unittest -v tests.test_volume_trend

ZIP/SQLite только читаются. Повторы времени сохраняются. Клиринги остаются
внутри окна; объёмы вычисляются по исходным сделкам с включением обеих границ.
"""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import json
import sqlite3
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

NS = 1_000_000_000
DAY_NS = 86400 * NS


def stamp_ns(values):
    """Преобразует последовательность московских отметок в наносекунды суток."""
    return pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S.%f").astype("int64").to_numpy() % DAY_NS


def time_text(value):
    """Возвращает время суток для числа наносекунд value, сохраняя доли секунды."""
    seconds, fraction = divmod(int(value), NS)
    hour, remainder = divmod(seconds, 3600)
    minute, second = divmod(remainder, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}.{fraction:09d}"


def read_ticks(path, day):
    """Возвращает times/prices/volumes из ZIP path; проверяет день, числа и порядок."""
    try:
        with ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
                raise ValueError("Ожидается один CSV внутри ZIP")
            with archive.open(members[0]) as stream:
                data = pd.read_csv(stream, dtype={"datetime": str, "last": "float64", "volume": "int64"})
    except BadZipFile as exc:
        raise ValueError(f"Повреждённый ZIP: {path}") from exc
    if list(data.columns) != ["datetime", "last", "volume"] or data.empty:
        raise ValueError("Нет сделок или неправильные столбцы")
    if not data.datetime.str.slice(0, 10).eq(day).all():
        raise ValueError("Дата сделок отличается от даты архива")
    times = stamp_ns(data.datetime)
    prices = data["last"].to_numpy()
    volumes = data.volume.to_numpy()
    if (np.diff(times) < 0).any() or not np.isfinite(prices).all() or (prices <= 0).any() or (volumes <= 0).any():
        raise ValueError("Нарушен порядок времени или неположительные цена/объём")
    return times, prices, volumes


def common_window(bounds):
    """Возвращает включительное пересечение (start,end) границ bounds либо None."""
    start, end = max(item[0] for item in bounds), min(item[1] for item in bounds)
    return (int(start), int(end)) if start < end else None


def window_volume(times, prefix, start, end):
    """Суммирует объём по times и накопленной сумме prefix в закрытом окне start/end."""
    left = np.searchsorted(times, start, side="left")
    right = np.searchsorted(times, end, side="right")
    return int(prefix[right] - prefix[left])


def history_indices(days, current, lookback=20):
    """Выбирает индексы lookback предыдущих будних дат, не скрывая ошибочные записи."""
    dates = pd.to_datetime(days.day.iloc[:current])
    return days.index[:current][dates.dt.weekday.to_numpy() < 5].tolist()[-lookback:]


def time_bars(times, prices, volumes, minutes):
    """Строит OHLCV сетку minutes между первой/последней сделками, заполняя паузы."""
    width = minutes * 60 * NS
    bins = times // width
    edges = np.r_[0, np.flatnonzero(np.diff(bins)) + 1, len(times)]
    starts, ends = edges[:-1], edges[1:] - 1
    observed = pd.DataFrame({"slot": bins[starts], "open": prices[starts],
                             "high": np.maximum.reduceat(prices, starts),
                             "low": np.minimum.reduceat(prices, starts), "close": prices[ends],
                             "volume": np.add.reduceat(volumes, starts)})
    result = observed.set_index("slot").reindex(range(int(bins[0]), int(bins[-1])+1))
    close = result.close.ffill()
    for column in ("open", "high", "low", "close"):
        result[column] = result[column].fillna(close)
    result["volume"] = result.volume.fillna(0).astype("int64")
    result["start_ns"] = result.index.to_numpy() * width
    result["end_ns"] = result.start_ns + width - 1
    result["is_complete"] = 1
    return result.reset_index(drop=True)


@contextmanager
def snapshot_connection(path, output):
    """Читает согласованный снимок БД path в output с учётом WAL и стабильности."""
    path, output = Path(path), Path(output)
    wal = Path(str(path) + "-wal")
    if wal.exists() and wal.stat().st_size:
        # mode=ro использует существующий WAL; backup включает незачекпойнченные записи.
        source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        target = sqlite3.connect(output)
        try:
            source.backup(target)
        finally:
            source.close()
            target.close()
    else:
        before = path.stat()
        output.write_bytes(path.read_bytes())
        after = path.stat()
        if before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size or (wal.exists() and wal.stat().st_size):
            raise ValueError("База изменяется; повторите создание снимка")
    connection = sqlite3.connect(output.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def load_database(path, output, symbol):
    """Возвращает дневной журнал, бары и конфигурацию одного набора symbol из снимка."""
    with snapshot_connection(path, output) as connection:
        configs = connection.execute("SELECT dataset_id,config_json FROM datasets").fetchall()
        if len(configs) != 1:
            raise ValueError("Требуется ровно один набор данных")
        dataset_id, config = configs[0]
        days = pd.read_sql_query("SELECT * FROM days WHERE dataset_id=? AND symbol=? ORDER BY day", connection, params=(dataset_id, symbol))
        bars = pd.read_sql_query("SELECT * FROM bars WHERE dataset_id=? AND symbol=? ORDER BY day,bar_index", connection, params=(dataset_id, symbol))
    bars["start_ns"], bars["end_ns"] = stamp_ns(bars.start_time), stamp_ns(bars.end_time)
    bars["weekday"] = pd.to_datetime(bars.day).dt.weekday
    return days, bars, json.loads(config)


def audit_symbol(data_dir, output, symbol, start, end):
    """Проверяет архивы symbol и БД; сохраняет точный кэш и минутные свечи, возвращает аудит."""
    cache = output / "cache" / symbol
    cache.mkdir(parents=True, exist_ok=True)
    db_days, bars, config = load_database(data_dir / f"{symbol}_delta_bars.sqlite3", output / "cache" / f"{symbol}.sqlite3", symbol)
    db_days.to_csv(output / f"{symbol}_database_days.csv", index=False)
    bars.to_pickle(output / "cache" / f"{symbol}_bars.pkl")
    journal = db_days.set_index("day")
    totals = bars.groupby("day").agg(bar_volume=("volume", "sum"), bar_ticks=("tick_count", "sum"), bar_count=("bar_index", "size"))
    files = sorted((data_dir / f"data_finam_{symbol}_tick_zip").glob("*.zip"))
    before_start = [p for p in files if p.stem < start.replace("-", "") and pd.Timestamp(p.stem).weekday() < 5][-20:]
    selected = before_start + [p for p in files if start.replace("-", "") <= p.stem <= end.replace("-", "")]
    rows = []
    for index, path in enumerate(selected):
        day = pd.Timestamp(path.stem).date().isoformat()
        item = {"symbol": symbol, "day": day, "weekday": pd.Timestamp(day).weekday(), "file": str(path), "valid": False, "issue": "", "unknown_completeness": True}
        try:
            before = path.stat()
            digest = sha256(path.read_bytes()).hexdigest()
            cached = cache / f"{path.stem}.npz"
            if cached.exists():
                with np.load(cached) as stored:
                    if str(stored["sha256"]) != digest:
                        raise ValueError("Архив изменился после сохранения кэша")
                    times, prices, volumes = stored["times"], stored["prices"], stored["volumes"]
            else:
                times, prices, volumes = read_ticks(path, day)
                np.savez_compressed(cached, times=times, prices=prices, volumes=volumes, sha256=digest)
            after = path.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise ValueError("Архив изменился во время чтения")
            item.update({"sha256": digest, "first_ns": int(times[0]), "last_ns": int(times[-1]), "first": time_text(times[0]), "last": time_text(times[-1]), "tick_count": len(times), "volume": int(volumes.sum()), "max_gap_minutes": float(np.diff(times).max(initial=0) / NS / 60)})
            if day in journal.index:
                record = journal.loc[day]
                if digest != record.sha256 or len(times) != record.tick_count or int(volumes.sum()) != record.volume:
                    raise ValueError("Тики расходятся с дневным журналом БД")
                item["db_status"] = record.status
                if day in totals.index:
                    total = totals.loc[day]
                    if int(total.bar_volume) != item["volume"] or int(total.bar_ticks) != len(times) or int(total.bar_count) != record.bar_count:
                        raise ValueError("Бары не сохраняют весь объём/число тиков")
                    day_bars = bars[bars.day.eq(day)]
                    if not np.array_equal(day_bars.start_row.to_numpy(), np.r_[1, day_bars.end_row.to_numpy()[:-1]+1]) or int(day_bars.end_row.iloc[-1]) != len(times):
                        raise ValueError("Исходные строки баров не покрывают тики последовательно")
                    ends = day_bars.end_row.to_numpy(dtype=int)-1
                    begins = day_bars.start_row.to_numpy(dtype=int)-1
                    if not np.array_equal(times[begins], day_bars.start_ns) or not np.array_equal(times[ends], day_bars.end_ns):
                        raise ValueError("Время баров расходится с исходными строками")
                    item["bar_count"] = len(day_bars)
            else:
                item["db_status"] = "нет записи"
            for minutes in (1, 5, 15):
                time_bars(times, prices, volumes, minutes).to_pickle(cache / f"{path.stem}_{minutes}m.pkl")
            item["valid"] = True
        except (ValueError, OSError, KeyError) as exc:
            item["issue"] = str(exc)
        rows.append(item)
        if index % 100 == 0 or index == len(selected)-1:
            print(f"Аудит {symbol}: {index+1}/{len(selected)}, {day}, ошибок {sum(not r['valid'] for r in rows)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / f"{symbol}_audit.csv", index=False)
    (output / f"{symbol}_dataset_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
