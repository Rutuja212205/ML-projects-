@echo off
setlocal
title ML Arena Diagnostic
cd /d "%~dp0"
echo Checking ML Algorithm Battle Arena...
echo.
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 diagnose.py
) else (
  python diagnose.py
)
if errorlevel 1 (
  echo.
  echo The diagnostic could not start. Make sure Python is installed.
  pause
)
endlocal
