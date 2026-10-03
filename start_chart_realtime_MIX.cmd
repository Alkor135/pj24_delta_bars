@echo off
rem Запускает отдельный график MIX из окружения проекта; RTS работает одновременно.
rem Примеры: start_chart_realtime_MIX.cmd
rem          start_chart_realtime_MIX.cmd --threshold 450 --refresh-ms 200
rem Общий сборщик запускается автоматически; QUIK должен запускать QuikSharp.lua.
pushd "%~dp0"
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" "%~dp0chart_delta_bar_semafor_realtime_MIX.py" %*
if errorlevel 1 pause
popd
