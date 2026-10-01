r"""Вывод трёх последних дат баров для каждой SQLite-базы в указанной папке.

Скрипт ищет файлы, оканчивающиеся на delta_bars.sqlite3, без обхода подпапок.
Каждая база открывается только для чтения. Из bars.day выбираются до трёх
различных дат по убыванию, общих для всех инструментов и наборов в этой базе.
Если баров меньше трёх дней, выводятся все доступные даты. Ошибка отдельной
базы выводится в отчёте и не мешает проверке остальных файлов.

Примеры запуска из папки проекта:
    python check_date_db_delta_bars.py
    python check_date_db_delta_bars.py --db-dir C:\data_quote
    python check_date_db_delta_bars.py --db-dir "D:\Котировки и базы"
    python check_date_db_delta_bars.py --help

По умолчанию используется C:\data_quote. Нужен Python 3.10+ без сторонних
библиотек. Код возврата: 0 — проверка без ошибок, включая отсутствие баз;
1 — недоступная папка или ошибка чтения хотя бы одной базы.
"""

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import sys

DEFAULT_DB_DIR = Path(r"C:\data_quote")


def latest_dates(path: Path) -> list[str]:
    """Возвращает до трёх последних различных дат баров из существующей базы.

    Параметр path — путь к SQLite-файлу с таблицей bars и столбцом day.
    Результат — список дат YYYY-MM-DD по убыванию; для пустой таблицы — [].
    База открывается только для чтения, соединение закрывается и при ошибке.
    Ошибки файловой системы и SQLite передаются вызывающему коду.
    """
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        rows = connection.execute(
            "SELECT DISTINCT day FROM bars ORDER BY day DESC LIMIT 3"
        ).fetchall()
    return [row[0] for row in rows]


def main(argv: list[str] | None = None) -> int:
    """Разбирает параметры, проверяет базы и печатает даты либо причину ошибки.

    Параметр argv — список аргументов без имени скрипта; None означает
    аргументы командной строки. Параметр --db-dir задаёт папку с базами.
    Результат — код возврата 0 при отсутствии ошибок или 1 при ошибке папки
    либо чтения базы. Ошибка одного файла не прерывает проверку остальных.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db-dir", type=Path, default=DEFAULT_DB_DIR,
        help=r"папка с базами (по умолчанию C:\data_quote)"
    )
    args = parser.parse_args(argv)
    try:
        if not args.db_dir.is_dir():
            raise ValueError(f"Не найдена папка: {args.db_dir}")
        databases = sorted(
            (path for path in args.db_dir.iterdir()
             if path.is_file() and path.name.lower().endswith("delta_bars.sqlite3")),
            key=lambda path: path.name.casefold(),
        )
    except (OSError, ValueError) as exc:
        print(f"Проверка остановлена: {exc}", file=sys.stderr)
        return 1

    if not databases:
        print(f"В папке {args.db_dir} не найдены базы *delta_bars.sqlite3.")
        return 0

    exit_code = 0
    for path in databases:
        try:
            dates = latest_dates(path)
        except (OSError, sqlite3.Error) as exc:
            print(f"{path.name}: ошибка чтения — {exc}")
            exit_code = 1
            continue
        print(f"{path.name}: {', '.join(dates) if dates else 'нет баров'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
