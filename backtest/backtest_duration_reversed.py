"""Бэктест стратегии длительности дельта-баров с обратным направлением входа.

Примеры запуска из корня проекта:
    python backtest/backtest_duration_reversed.py
    python backtest/backtest_duration_reversed.py --symbols RTS
    python backtest/backtest_duration_reversed.py --symbols MIX --alf-alpha 0.4
    python backtest/backtest_duration_reversed.py --entry-filter none
    python -m backtest.backtest_duration_reversed --help

Сигнал обычной покупки открывает Short, сигнал обычной продажи — Long.
При включённом ALF: продажа после быстрого растущего бара выше ALF,
покупка после быстрого падающего бара ниже ALF. Фильтры применяются до разворота.
В рынке не более одной позиции на инструмент; исполнение и выходы как в исходном тесте.
Издержки вычитаются заново. Отчёты: C:/data_quote/duration_backtests_reversed/run_...
Зависимости: python -m pip install -r requirements-backtest.txt
Кэш длительности остаётся в .duration_cache в корне проекта.
"""

from pathlib import Path
import sqlite3
import sys

if __package__ in (None, ""):
    # Прямой запуск из подпапки использует те же импорты, что и python -m.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.backtest_duration import main as run_backtest


def main():
    """Без параметров запускает обратную стратегию по CLI и возвращает None.

    Общая модель из backtest.backtest_duration создаёт отдельный отчёт,
    сохраняя правила исполнения, пути к кэшу и проверку SHA256 исходников.
    """
    run_backtest(reverse=True, description=__doc__)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f'Обратный бэктест остановлен: {error}', file=sys.stderr)
        sys.exit(1)
