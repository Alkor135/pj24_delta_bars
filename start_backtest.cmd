@echo off
rem Назначение: запускает исследование длительности из backtest в окружении проекта.
rem Логика: выбирает корень проекта, проверяет .venv и передаёт все аргументы Python.
rem Примеры: start_backtest.cmd --symbols RTS
rem           start_backtest.cmd --symbols MIX --entry-filter none
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Не найдено окружение .venv. Инструкция: docs\duration-backtest.md
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -B -u "backtest\backtest_duration.py" %*
pause
