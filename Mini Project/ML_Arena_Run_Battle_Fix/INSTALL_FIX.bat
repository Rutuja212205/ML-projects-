@echo off
setlocal
title ML Arena Run Battle Fix
set "PROJECT=%USERPROFILE%\OneDrive\Desktop\ML_Algorithm_Battle_Arena"
if not exist "%PROJECT%\app.py" set "PROJECT=%USERPROFILE%\Desktop\ML_Algorithm_Battle_Arena"
if not exist "%PROJECT%\app.py" (
  echo ERROR: Could not find ML_Algorithm_Battle_Arena on your Desktop.
  echo Please extract this package and place it on your Desktop, then run again.
  pause
  exit /b 1
)
if not exist "%PROJECT%\static\js" mkdir "%PROJECT%\static\js"
if exist "%PROJECT%\static\js\app.js" (
  copy /Y "%PROJECT%\static\js\app.js" "%PROJECT%\static\js\app.js.before_run_battle_fix.bak" >nul
)
copy /Y "%~dp0app.js" "%PROJECT%\static\js\app.js" >nul
if errorlevel 1 (
  echo ERROR: Could not replace app.js.
  echo Close any editor that may be locking the file and try again.
  pause
  exit /b 1
)
echo.
echo Installed updated app.js to:
echo %PROJECT%\static\js\app.js
echo.
echo Verification:
findstr /C:"battleInProgress" "%PROJECT%\static\js\app.js" >nul
if errorlevel 1 (
  echo FAILED: verification did not find the fix marker.
  pause
  exit /b 1
)
echo PASSED: updated file contains the Run Battle fix.
echo.
echo Now go to your browser and press Ctrl+F5.
start "" "http://127.0.0.1:5000/"
pause
endlocal
