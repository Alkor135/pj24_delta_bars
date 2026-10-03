# Общие модули проекта

`source` — пакет Python с расчётами, хранением данных, индикаторами и отчётами.
Модули импортируются как `source.delta_core`, `source.duration_data` и т. д.
Точки запуска находятся в корне проекта и в папке `backtest`; вспомогательные
модули обычно не запускаются напрямую.

| Модуль | Назначение и логика |
| --- | --- |
| `__init__.py` | Объявляет пакет общих компонентов. |
| `delta_core.py` | Строит адаптивные дельта-бары из тиков. |
| `realtime_core.py` | Нормализует сделки QUIK, строит текущие дельта-бары по tick rule, пересчитывает поздние сделки и ведёт отдельный журнал SQLite с курсором. |
| `realtime_data.py` | Читает историю до текущего дня; подбирает порог по прошлым ZIP, проверяет дату/набор/окно кэша, атомарно сохраняет JSON порогов; отслеживает изменения SQLite/WAL и JSON ночной подготовки. |
| `quik_bridge.py` | Разрешённые рыночные запросы и независимый поток событий установленного QuikSharp; выбор серии RI/MX по справочнику погашений. |
| `realtime_feed.py` | Общий сборщик, отдельный журнал, повторная сверка дня, локальный HTTP и автоматический скрытый запуск процесса для двух окон; сохраняет хвост callback при ошибке снимка, разрыве и смене даты. |
| `realtime_chart.py` | Общая сессия исторических/живых баров, фоновое чтение журнала, текущая свеча, календарный переход и два самостоятельных окна фиксированных инструментов; подхватывает ночное обновление, фиксирует порог после первой сделки. |
| `delta_store.py` | Хранит наборы параметров, бары и журнал исходников в SQLite. |
| `chart_data.py` | Читает базы и рассчитывает индикаторы для просмотра. |
| `duration_data.py` | Проверяет SQLite и ZIP, создаёт события исполнения, применяет сессию и ALF, кэширует подготовленные данные. |
| `duration_engine.py` | Моделирует сделки для заданной пары длительностей или сетки пар; поддерживает обратные входы. |
| `duration_analysis.py` | Рассчитывает PnL, просадку, ранжирование параметров и последовательную проверку вне периода подбора. |
| `duration_report.py` | Создаёт автономный HTML с Plotly из готовых результатов исследования. |

## Запуск и параметры

Команды выполняются из корня проекта:

```powershell
python tick_to_delta_bars.py --help
python chart_delta_bars.py --help
python backtest/backtest_duration.py --symbols RTS MIX
python backtest/backtest_duration.py --symbols RTS --entry-grid 1:45:1 --exit-grid 5:300:5
python backtest/backtest_duration.py --symbols RTS --entry-filter alf --alf-alpha 0.4
python backtest/backtest_duration_reversed.py --symbols MIX --entry-filter none
python -m unittest -v tests.test_duration_data tests.test_duration_engine tests.test_duration_analysis tests.test_duration_report
python -m unittest -v tests.test_realtime_core
python -m unittest -v tests.test_realtime_data tests.test_realtime_feed tests.test_chart_realtime
```

Длительности задаются в секундах через `--entry-grid` и `--exit-grid`
(`начало:конец:шаг`). `--entry-filter alf|none` и `--alf-alpha` управляют
дополнительным фильтром; `--session-start`, `--entry-end`, `--close-time` задают
сессию. Источники задаются через `--data-dir` или `--db`, `--symbols`,
`--dataset-id`, `--start`/`--end`. Кэш выбирается через `--cache-dir` и по
умолчанию остаётся в `.duration_cache` в корне проекта. `--output-dir` задаёт
папку отчётов. Подробные настройки и параметры остальных точек запуска
описаны в [корневом README](../README.md), [README бэктестов](../backtest/README.md)
и [инструкции стратегии](../docs/duration-backtest.md).

Для живых графиков общие параметры подключения заданы в `realtime_quik.json`.
`realtime_chart.py` принимает историю и поток журнала, рассчитывает текущий кадр
в worker и сохраняет прежний холст при новых сделках. Кэш/порог, параметры
Семафора и CLI описаны в [инструкции QUIK](../docs/realtime-quik.md).
