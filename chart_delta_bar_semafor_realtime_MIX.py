r"""Открывает отдельный реал-тайм график дельта-баров MIX с Семафором и SMA.

Примеры запуска из папки проекта:
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_MIX.py
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_MIX.py --start 2026-09-01
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_MIX.py --threshold 450 --refresh-ms 200
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_MIX.py --config realtime_quik.json --help

История читается из C:\data_quote\MIX_delta_bars.sqlite3; сегодня строится
по обезличенным сделкам QUIK. Общий сборщик запускается автоматически и
параллельно обслуживает график RTS. БД истории не изменяется. Подключение и
контракт задаются в realtime_quik.json; инструкция — docs/realtime-quik.md.
Зависимости: python -m pip install -r requirements-chart.txt
"""

from source.realtime_chart import main


if __name__ == "__main__":
    raise SystemExit(main("MIX"))
