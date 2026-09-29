@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Не найдено окружение .venv. Инструкция: docs\duration-backtest.md
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -B -u backtest_duration.py %*
pause
