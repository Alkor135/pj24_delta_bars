@echo off
rem Запускает отдельный график RTS из окружения проекта; MIX работает одновременно.
rem Примеры: start_chart_realtime_RTS.cmd
rem          start_chart_realtime_RTS.cmd --threshold 100 --refresh-ms 200
rem Общий сборщик запускается автоматически; QUIK должен запускать QuikSharp.lua.
pushd "%~dp0"
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" "%~dp0chart_delta_bar_semafor_realtime_RTS.py" %*
if errorlevel 1 pause
popd
