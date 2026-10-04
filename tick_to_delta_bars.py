"""Конвертер дневных тиков Финама RTS/MIX в адаптивные дельта-бары SQLite.

Порог каждого дня подбирается по реальному числу баров за 20 предшествующих
будних дней с тиками. Цель — число непустых пятиминутных интервалов в этом окне.
Текущий день не входит в расчёт порога. Повторный запуск дозаписывает новые дни.

Примеры запуска из папки проекта:
    python tick_to_delta_bars.py
    python tick_to_delta_bars.py --symbols RTS --db C:\\data_quote\\RTS_delta_bars.sqlite3
    python tick_to_delta_bars.py --symbols MIX --db C:\\data_quote\\MIX_delta_bars.sqlite3

    python tick_to_delta_bars.py --symbols MIX --start 2022-01-01 --end 2022-03-01
    python tick_to_delta_bars.py --symbols RTS MIX --db C:\\data_quote\\delta_bars.sqlite3
    python tick_to_delta_bars.py --lookback 20 --target-minutes 5
Только стандартная библиотека Python 3.10+; исходные ZIP не изменяются.
"""

import argparse
from collections import deque
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import os
from pathlib import Path
import sqlite3
import sys
from zipfile import BadZipFile

from source.delta_core import read_day, calibrate, build_bars
from source.delta_store import open_database, register_dataset, recorded_days, rewind, save_day

DEFAULT_ROOT = Path(r"C:\data_quote")
DEFAULT_DB = DEFAULT_ROOT / "delta_bars.sqlite3"
DEFAULT_START = date(2022, 1, 1)
MOSCOW = timezone(timedelta(hours=3))


def fingerprint(path):
    """Вычисляет SHA-256 ZIP и выявляет изменение файла непосредственно во время чтения."""
    before = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Архив изменяется во время чтения: {path}; повторите запуск после загрузки")
    return digest.hexdigest()


class RunLock:
    """Предотвращает одновременную конвертацию одной базы двумя процессами."""

    def __init__(self, path):
        """Запоминает путь служебного файла блокировки рядом с базой."""
        self.path = Path(str(path) + ".lock")
        self.stream = None

    def __enter__(self):
        """Берёт системную блокировку, автоматически снимаемую при завершении процесса."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise ValueError(f"База уже обрабатывается другим процессом: {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, traceback):
        """Освобождает блокировку, не удаляя общий файл между конкурирующими запусками."""
        if self.stream is not None:
            self.stream.close()


def discover(input_root, symbol, end):
    """Находит дневные ZIP инструмента до указанной даты и проверяет имена дат."""
    folder = Path(input_root) / f"data_finam_{symbol}_tick_zip"
    if not folder.is_dir():
        raise ValueError(f"Не найдена папка исходных тиков: {folder}")
    result = {}
    for path in folder.glob("*.zip"):
        if len(path.stem) != 8 or not path.stem.isdigit():
            continue
        day = datetime.strptime(path.stem, "%Y%m%d").date()
        if day <= end:
            result[day] = path
    return dict(sorted(result.items()))


def process_symbol(connection, dataset_id, input_root, symbol, start, end, lookback, minutes, log):
    """Обрабатывает новые дни с прошлым окном и пересчитывает изменившийся участок истории."""
    sources = discover(input_root, symbol, end)
    records = recorded_days(connection, dataset_id, symbol)
    checksums, cached = {}, {}
    prestart = []
    count = 0
    # До начала выгрузки нужны ровно последние непустые будние дни, а не вся старая история.
    for day in reversed([day for day in sources if day < start and day.weekday() < 5]):
        path = sources[day]
        checksum = fingerprint(path)
        checksums[day] = checksum
        record = records.get(day.isoformat())
        if record is not None and record["sha256"] == checksum:
            nonempty = record["tick_count"] > 0
        else:
            log(f"{symbol} {day}: чтение истории для прогрева")
            ticks = read_day(path, day, minutes)
            cached[day] = ticks.history()
            nonempty = bool(ticks.times)
        prestart.append(day)
        count += int(nonempty)
        if count == lookback:
            break
    selected = {day: sources[day] for day in sorted(prestart + [day for day in sources if day >= start])}
    if not selected:
        raise ValueError(f"{symbol}: нет подходящих архивов до {end}")
    log(f"{symbol}: проверка {len(selected)} архивов по SHA-256")
    for index, (day, path) in enumerate(selected.items(), 1):
        if day not in checksums:
            checksums[day] = fingerprint(path)
        if index % 250 == 0:
            log(f"{symbol}: проверено {index}/{len(selected)} архивов")

    changes = []
    for day_string, record in records.items():
        day = date.fromisoformat(day_string)
        if day <= end and (day not in selected or checksums[day] != record["sha256"]):
            changes.append(day)
    last_recorded = max(records, default="")
    changes.extend(day for day in selected if day.isoformat() < last_recorded and day.isoformat() not in records)
    if changes:
        first_changed = min(changes)
        log(f"{symbol}: источник изменился; пересчёт начиная с {first_changed}")
        rewind(connection, dataset_id, symbol, first_changed.isoformat())
        records = {key: value for key, value in records.items() if key < first_changed.isoformat()}

    pending = [day for day in selected if day.isoformat() not in records]
    result = {"processed_days": 0, "skipped_days": len(selected) - len(pending), "bars_added": 0,
              "warmup_days": 0, "dataset_id": dataset_id}
    if not pending:
        log(f"{symbol}: новых данных нет; пропущено {len(selected)} дней")
        return result
    first_pending = pending[0]
    history = deque(maxlen=lookback)
    historical_days = [day for day in selected if day < first_pending and day.weekday() < 5
                       and records[day.isoformat()]["tick_count"] > 0][-lookback:]
    for day in historical_days:
        log(f"{symbol} {day}: восстановление обучающего окна")
        historical = cached.pop(day, None)
        if historical is None:
            historical = read_day(selected[day], day, minutes).history()
        if fingerprint(selected[day]) != checksums[day]:
            raise ValueError(f"Архив изменился во время расчёта: {selected[day]}; повторите запуск")
        history.append(historical)

    for day in pending:
        path = selected[day]
        log(f"{symbol} {day}: чтение тиков; в окне {len(history)}/{lookback} дней")
        historical = cached.pop(day, None)
        ticks = None
        if historical is None or day >= start:
            ticks = read_day(path, day, minutes)
            historical = ticks.history()
        if fingerprint(path) != checksums[day]:
            raise ValueError(f"Архив изменился во время расчёта: {path}; повторите запуск")
        tick_count = len(historical.path.values)
        calibration, bars = None, []
        if not tick_count:
            status = "empty"
        elif day < start:
            status = "prestart"
        elif len(history) < lookback:
            status = "warmup"
            result["warmup_days"] += 1
        else:
            status = "ready"
            calibration = calibrate(list(history))
            bars = build_bars(ticks, calibration.threshold)
        metadata = {"file_path": str(path.resolve()), "sha256": checksums[day], "tick_count": tick_count,
                    "volume": historical.volume, "active_intervals": historical.active_intervals,
                    "status": status, "threshold": calibration.threshold if calibration else None,
                    "window_start": history[0].day.isoformat() if history else None,
                    "window_end": history[-1].day.isoformat() if history else None,
                    "window_days": len(history), "target_count": calibration.target_count if calibration else None,
                    "calibration_count": calibration.actual_count if calibration else None,
                    "calibration_error": calibration.relative_error if calibration else None,
                    "calibration_evaluations": calibration.evaluations if calibration else None,
                    "bar_count": len(bars)}
        save_day(connection, dataset_id, symbol, day.isoformat(), metadata, bars)
        result["processed_days"] += 1
        result["bars_added"] += len(bars)
        if calibration:
            log(f"{symbol} {day}: порог {calibration.threshold}; калибровка {calibration.actual_count}/{calibration.target_count} "
                f"({calibration.relative_error:+.1%}); сохранено {len(bars)} баров; 5-мин интервалов сегодня {historical.active_intervals}")
        else:
            log(f"{symbol} {day}: {status}; тиков {tick_count}")
        if day.weekday() < 5 and tick_count:
            history.append(historical)
    return result


def convert(input_root, db, symbols, start, end, lookback=20, minutes=5, log=print):
    """Выполняет конвертацию выбранных инструментов с блокировкой и дневными транзакциями."""
    if start > end or lookback < 1 or minutes < 1 or minutes > 1440:
        raise ValueError("Неверный диапазон дат, длина окна или размер временного интервала")
    if end >= datetime.now(MOSCOW).date():
        raise ValueError("Текущий и будущие московские дни не обрабатываются: задайте --end не позднее вчера")
    if not symbols or any(symbol not in ("RTS", "MIX") for symbol in symbols):
        raise ValueError("Поддерживаются инструменты RTS и MIX")
    with RunLock(db):
        connection = open_database(db)
        try:
            dataset_id = register_dataset(connection, input_root, start, lookback, minutes)
            log(f"База: {Path(db).resolve()}; набор: {dataset_id}")
            results = {}
            for symbol in dict.fromkeys(symbols):
                results[symbol] = process_symbol(connection, dataset_id, input_root, symbol, start, end, lookback, minutes, log)
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return results
        finally:
            connection.close()


def parse_date(value):
    """Преобразует дату командной строки и выдаёт понятную ошибку формата."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Ожидается дата YYYY-MM-DD") from exc


def main(argv=None):
    """Разбирает параметры запуска и сообщает итоги либо причину безопасной остановки."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", choices=("RTS", "MIX"), default=["RTS", "MIX"], help="инструменты (оба по умолчанию)")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT, help="папка с каталогами data_finam_*_tick_zip")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="файл SQLite")
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START, help="начало сохранения баров (2022-01-01)")
    parser.add_argument("--end", type=parse_date, default=datetime.now(MOSCOW).date() - timedelta(days=1), help="конец включительно (вчера по Москве)")
    parser.add_argument("--lookback", type=int, default=20, help="число прошлых будних дней с тиками")
    parser.add_argument("--target-minutes", type=int, default=5, help="размер целевых временных интервалов в минутах")
    args = parser.parse_args(argv)
    try:
        results = convert(args.input_root, args.db, args.symbols, args.start, args.end,
                          lookback=args.lookback, minutes=args.target_minutes,
                          log=lambda message: print(message, flush=True))
    except KeyboardInterrupt:
        print("Остановлено пользователем. Завершённые дни сохранены; текущий день будет повторён при запуске.", file=sys.stderr)
        return 130
    except (ValueError, OSError, BadZipFile, sqlite3.Error) as exc:
        print(f"Конвертация остановлена: {exc}", file=sys.stderr)
        return 1
    for symbol, result in results.items():
        print(f"{symbol}: обработано дней {result['processed_days']}, пропущено {result['skipped_days']}, "
              f"добавлено баров {result['bars_added']}, дней прогрева {result['warmup_days']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
