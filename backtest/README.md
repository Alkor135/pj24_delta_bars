# Бэктесты дельта-баров RTS/MIX

В этой папке находятся все четыре скрипта `backtest*.py`. Общие расчёты
используют пакет `source` из корня проекта. `__init__.py` позволяет импортировать
скрипты как `backtest.backtest_duration` и запускать их через `python -m`.

| Скрипт | Логика |
| --- | --- |
| `backtest_duration.py` | Перебирает пороги длительности входа/выхода, применяет фильтр ALF, выбирает параметры на обучении и сравнивает PnL вне периода подбора. |
| `backtest_duration_reversed.py` | Запускает ту же модель с противоположной стороной каждой разрешённой сделки, сохраняя правила фильтра, исполнения и затрат. |
| `backtest_stochastic.py` | Проверяет Stochastic (14, 3, 3) с ALF, стопом и целью 2R; сравнивает с вариантом ALF без стохастика. |
| `backtest_heikin_ashi.py` | Проверяет цвет Heikin Ashi с подтверждением одной/двумя свечами, переворотами и исполнением по исходным тикам. |

## Запуск

Из корня проекта с выбранным интерпретатором:

```powershell
python -m pip install -r requirements-backtest.txt
python backtest/backtest_duration.py --symbols RTS --entry-grid 1:45:1 --exit-grid 5:300:5
python backtest/backtest_duration_reversed.py --symbols MIX --entry-filter none
python backtest/backtest_stochastic.py --symbols RTS --start 2026-09-01 --end 2026-09-30
python backtest/backtest_heikin_ashi.py --confirmations 1,2 --costs 0,2,4,8 --base-cost 4
python -m backtest.backtest_duration --help
python -m backtest.backtest_duration_reversed --help
python -m backtest.backtest_stochastic --help
python -m backtest.backtest_heikin_ashi --help
```

Окружение проекта: `.venv/Scripts/python.exe`; при необходимости используйте
его вместо `python`. Из папки `backtest` доступен запуск
`../.venv/Scripts/python.exe backtest_duration.py --help`. Из другой папки
передавайте абсолютные пути к интерпретатору и скрипту; настройка `PYTHONPATH`
не требуется. В VS Code можно открыть файл из `backtest` и выбрать **Run Python File**.

Запускатели `start_backtest.cmd` и `start_backtest_reversed.cmd` остаются в корне
проекта, выбирают его как рабочую папку и передают аргументы новым путям:

```powershell
.\start_backtest.cmd --symbols RTS
.\start_backtest_reversed.cmd --symbols MIX --entry-filter none
```

## Параметры и результаты

Общие параметры: `--symbols RTS MIX`, включительные даты `--start`/`--end`,
`--db` и `--dataset-id`, затраты за круг в шагах цены `--costs 0,2,4,8` и
`--base-cost 4`. Исходные SQLite и ZIP открываются только для чтения.

**Длительность и обратная стратегия:**
`--data-dir` по умолчанию `C:/data_quote`; `--db` и `--tick-size` допустимы
для одного инструмента. `--entry-grid 1:45:1`, `--exit-grid 5:300:5`
задают включительные диапазоны секунд. `--entry-filter alf|none`
(по умолчанию `alf`), `--alf-alpha 0.4`, `--session-start 10:00`,
`--entry-end 18:30`, `--close-time 18:40`, `--holdout-start 2026-01-01`,
`--min-trades 100` и `--top 5` управляют исследованием. `--cache-dir`
по умолчанию остаётся в `.duration_cache` в корне проекта. `--output-dir`
по умолчанию `C:/data_quote/duration_backtests` либо
`C:/data_quote/duration_backtests_reversed`. Каждый запуск создаёт `run_...`
с `index.html`, отчётами инструментов, таблицами и SQLite результатов.

**Стохастик:** `--db` по умолчанию `C:/data_quote/delta_bars.sqlite3`;
`--period 14`, `--smooth-k 3`, `--smooth-d 3`, `--alf-alpha 0.4`,
`--reward-risk 2`; `--output` по умолчанию `results/stochastic`.

**Heikin Ashi:** та же стандартная база; `--confirmations 1,2` — числа полных
свечей для подтверждения цвета; `--output` по умолчанию `results/heikin_ashi`.
Оба скрипта создают в отдельном `run_...` файлы `report.html`, `summary.csv`,
`daily.csv`, `trades.csv`, `coverage.csv`, `metadata.json`.

Относительные пути, заданные пользователем, считаются от текущей рабочей папки.
Исходники для SHA256 в метаданных находятся от корня проекта независимо от
места запуска; ключи содержат пути вида `backtest/backtest_duration.py`.
Торговые алгоритмы и стандартные папки результатов при переносе сохранены.
Подробные правила — в [корневом README](../README.md) и
[инструкции стратегии длительности](../docs/duration-backtest.md).

## Проверки

```powershell
python -m unittest -v tests.test_backtest_launch
python -m unittest discover -s tests -v
```

Команды выполняются из корня проекта. Проверяются прямой и модульный запуск,
запуск из другой папки, создание четырёх отчётов на синтетических данных,
SHA256 исходников, расположение кэша и существующие расчёты стратегий.
