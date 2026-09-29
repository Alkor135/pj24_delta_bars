"""Бэктест стратегии длительности дельта-баров с обратным направлением входа.

Примеры запуска из корня проекта:
    python backtest_duration_reversed.py
    python backtest_duration_reversed.py --symbols RTS
    python backtest_duration_reversed.py --symbols MIX --alf-alpha 0.4
    python backtest_duration_reversed.py --entry-filter none

Сигнал обычной покупки открывает Short, сигнал обычной продажи — Long.
При включённом ALF: продажа после быстрого растущего бара выше ALF,
покупка после быстрого падающего бара ниже ALF. Фильтры применяются до разворота.
В рынке не более одной позиции на инструмент; исполнение и выходы как в исходном тесте.
Издержки вычитаются заново. Отчёты: C:/data_quote/duration_backtests_reversed/run_...
Зависимости: python -m pip install -r requirements-backtest.txt
"""

import sqlite3
import sys

from backtest_duration import main as run_backtest


def main():
    """Запускает общую модель с фиксированным обратным направлением сделок."""
    run_backtest(reverse=True, description=__doc__)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f'Обратный бэктест остановлен: {error}', file=sys.stderr)
        sys.exit(1)
