r"""Подготавливает точные дневные пороги RTS/MIX для графиков QUIK.

Примеры запуска из папки проекта:
    .\.venv\Scripts\python.exe prepare_realtime_thresholds.py
    .\.venv\Scripts\python.exe prepare_realtime_thresholds.py --date 2026-10-05
    .\.venv\Scripts\python.exe prepare_realtime_thresholds.py --symbols RTS --data-dir C:\data_quote
    .\.venv\Scripts\python.exe prepare_realtime_thresholds.py --symbols MIX --db C:\data_quote\MIX_delta_bars.sqlite3
    .\.venv\Scripts\python.exe prepare_realtime_thresholds.py --output .live_cache\thresholds.json

По умолчанию дата — завтра по Москве. Запускайте после ночного обновления
истории. Из days берётся прошлое окно, проверяются SHA-256 ZIP, повторяется
calibrate конвертера. SQLite только читается, JSON обновляется атомарно.
--db предназначен для одного инструмента. QUIK и графики не требуются.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from source.realtime_data import DEFAULT_THRESHOLD_FILE, prepare_threshold, write_thresholds
from tick_to_delta_bars import RunLock


def main(argv=None):
    """Разбирает argv, рассчитывает пороги и записывает JSON; возвращает код 0/2."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", choices=("RTS", "MIX"), default=["RTS", "MIX"], help="инструменты")
    parser.add_argument("--date", type=date.fromisoformat,
                        default=datetime.now(timezone(timedelta(hours=3))).date() + timedelta(days=1), help="день порога YYYY-MM-DD")
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\data_quote"), help="папка исторических баз")
    parser.add_argument("--db", type=Path, help="явная база для одного инструмента")
    parser.add_argument("--dataset-id", help="явный набор вместо последнего")
    parser.add_argument("--output", type=Path, default=DEFAULT_THRESHOLD_FILE, help="файл подготовленных порогов")
    args = parser.parse_args(argv)
    if args.db and len(args.symbols) != 1:
        parser.error("С --db выберите один инструмент через --symbols")
    try:
        entries = []
        for symbol in dict.fromkeys(args.symbols):
            print(f"{symbol}: подготовка порога для {args.date} по прошлым тикам…", flush=True)
            entry = prepare_threshold(args.db or args.data_dir / f"{symbol}_delta_bars.sqlite3", symbol, args.date, args.dataset_id)
            entries.append(entry)
            print(f"{symbol}: порог {entry['threshold']}; окно {entry['window_start']} — {entry['window_end']}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with RunLock(args.output.with_suffix(args.output.suffix + ".lock")):
            write_thresholds(args.output, entries)
        print(f"Сохранено: {args.output.resolve()}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Ошибка подготовки порогов: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
