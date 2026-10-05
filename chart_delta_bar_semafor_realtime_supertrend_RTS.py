r"""Открывает отдельный реал-тайм график RTS с Семафором, SMA и Supertrend.

Примеры запуска из папки проекта:
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_supertrend_RTS.py
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_supertrend_RTS.py --atr-period 10 --multiplier 3
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_supertrend_RTS.py --start 2022-09-01 --refresh-ms 200
    .\.venv\Scripts\python.exe chart_delta_bar_semafor_realtime_supertrend_RTS.py --config realtime_quik.json --help

История читается из C:\data_quote\RTS_delta_bars.sqlite3, текущий день строится
по обезличенным сделкам QUIK через общий сборщик всех реал-тайм окон проекта.
Семафор/SMA сохранены. Supertrend: ATR Уайлдера 10, множитель 3 по умолчанию,
синяя ветвь роста и красная снижения; текущая свеча и линии обновляются.
Параметры ATR меняются в GUI/CLI, история прогревает индикатор до --start.
Историческая БД только читается. Настройка QUIK: docs/realtime-quik.md.
Зависимости: python -m pip install -r requirements-chart.txt
"""

from source.realtime_chart import main


if __name__ == "__main__":
    raise SystemExit(main("RTS", supertrend=True))
