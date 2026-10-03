@echo off
rem Назначение: запускает обратную стратегию длительности из папки backtest.
rem Логика: выбирает корень проекта, проверяет .venv и передаёт все аргументы Python.
rem Примеры: start_backtest_reversed.cmd --symbols RTS
rem           start_backtest_reversed.cmd --symbols MIX --entry-filter none
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Не найдено окружение .venv. Инструкция: docs\duration-backtest.md
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -B -u "backtest\backtest_duration_reversed.py" %*
pause
