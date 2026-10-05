@echo off
rem Запускает реал-тайм MIX с Семафором/SMA и Supertrend из .venv.
rem Примеры: start_chart_realtime_supertrend_MIX.cmd
rem          start_chart_realtime_supertrend_MIX.cmd --atr-period 10 --multiplier 3
rem Общий сборщик читает QUIK; оба инструмента могут работать одновременно.
setlocal
chcp 65001 >nul
pushd "%~dp0"
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" "%~dp0chart_delta_bar_semafor_realtime_supertrend_MIX.py" %*
if errorlevel 1 pause
popd
