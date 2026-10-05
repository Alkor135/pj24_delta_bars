r"""Открывает отдельный реал-тайм график дельта-баров RTS с Семафором и SMA.

Примеры запуска из папки проекта:
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_RTS.py
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_RTS.py --start 2022-09-01
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_RTS.py --threshold 100 --refresh-ms 200
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_RTS.py --config realtime_quik.json --help

История читается из C:\data_quote\RTS_delta_bars.sqlite3; сегодня строится
по обезличенным сделкам QUIK. Общий сборщик запускается автоматически и
параллельно обслуживает график MIX. БД истории не изменяется. Подключение и
контракт задаются в realtime_quik.json; инструкция — docs/realtime-quik.md.
Зависимости: python -m pip install -r requirements-chart.txt
"""

from source.realtime_chart import main


if __name__ == "__main__":
    raise SystemExit(main("RTS"))
