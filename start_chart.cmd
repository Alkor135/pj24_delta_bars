@echo off
chcp 65001 >nul
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Не найдено окружение .venv. Инструкция установки: docs\chart.md
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0patch_finplot.py"
if errorlevel 1 (
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0chart_delta_bars.py" %*
if errorlevel 1 pause
