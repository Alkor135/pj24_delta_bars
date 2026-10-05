# Бэктесты дельта-баров RTS/MIX

В этой папке находятся пять скриптов `backtest*.py`. Общие расчёты
используют пакет `source` из корня проекта. `__init__.py` позволяет импортировать
скрипты как `backtest.backtest_duration` и запускать их через `python -m`.

| Скрипт | Логика |
| --- | --- |
| `backtest_sma_volume.py` | SMA 3/34 на пяти минутах, первый вход по накопленному объёму выше медианы тех же часов 20 будних дней, далее перевороты без повторного фильтра; сравнение с вариантом без фильтра. |
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

## SMA 3/34 и объём того же времени

```powershell
.\.venv\Scripts\python.exe -m backtest.backtest_sma_volume
.\.venv\Scripts\python.exe backtest/backtest_sma_volume.py --start 2026-01-01 --end 2026-10-02
.\.venv\Scripts\python.exe -m backtest.backtest_sma_volume --source results/volume_trend --costs 0,2,4,8 --base-cost 4 --repetitions 5000
.\.venv\Scripts\python.exe -m unittest -v tests.test_backtest_sma_volume
```

`--source` — предыдущий аудит с 5m.pkl; исходные ZIP читаются заново и
сверяются по SHA256, времени, объёму и полным OHLCV. `--symbols RTS MIX`,
`--start 2022-01-01`, `--end 2026-10-02` задают выборку. Все 21 сессии
должны охватывать 10:00–18:45; выходные исключены. SMA продолжаются между
днями, прогрев 100 баров, разрыв >10 дней сбрасывает индикатор. Сигнал —
закрытие полной пятиминутной свечи, вход — первый тик следующей свечи.
Если следующая свеча без сделок, запоздалого входа нет. В 18:45 приоритет
закрытия; входов по сигналу 18:45 нет. Long при пересечении вверх, Short вниз.
Фильтр проверяется только до первого входа, равенство объёма медиане не проходит.
Следующие перевороты выполняются независимо от фильтра. Стопов/целей нет.

`--costs` — сценарии затрат в шагах за круг; `--base-cost` — сценарий графиков,
не фактический тариф. RTS: шаг 10 пунктов, MIX: 25, кратность цен проверяется.
PnL — пункты одного контракта, не рубли. `--repetitions` — bootstrap-повторы.
Парные сравнения сохраняют нулевые дни, Holm охватывает оба инструмента и
все сценарии затрат. Периоды 2026, 2022–2025 и вся история рассматриваются
отдельно; 2026 уже изучался и не является новой независимой выборкой.

`--output results/sma_volume` создаёт новую папку `run_...` с автономным
интерактивным `report.html`, таблицами `summary.csv`, `daily.csv`, `trades.csv`,
`signals.csv`, `comparisons.csv`, `coverage.csv`, `verification.csv`,
хешами и параметрами в `metadata.json`. Источники и прежние отчёты не меняются.
Время сигнала в `signals.csv` имеет календарный формат `ГГГГ-ММ-ДД ЧЧ:ММ:СС`.
Отчёт отдельно показывает изменение PnL до затрат и экономию за счёт
исключённых сделок; команды в нём воспроизводят фактические параметры запуска.

## Проверки

```powershell
python -m unittest -v tests.test_backtest_launch
python -m unittest discover -s tests -v
```

Команды выполняются из корня проекта. Проверяются прямой и модульный запуск,
запуск из другой папки, создание четырёх отчётов на синтетических данных,
SHA256 исходников, расположение кэша и существующие расчёты стратегий.
