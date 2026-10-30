@echo off
setlocal
set "PROJECT=%USERPROFILE%\OneDrive\Desktop\ML_Algorithm_Battle_Arena"
if not exist "%PROJECT%\static\js" set "PROJECT=%USERPROFILE%\Desktop\ML_Algorithm_Battle_Arena"
if not exist "%PROJECT%\static\js" (
  echo Could not find ML_Algorithm_Battle_Arena in your OneDrive Desktop or Desktop.
  echo If the project is somewhere else, move this repair package next to the project and run again.
  pause
  exit /b 1
)
if exist "%PROJECT%\static\js\app.js" (
  copy /Y "%PROJECT%\static\js\app.js" "%PROJECT%\static\js\app.js.backup" >nul
  if errorlevel 1 (
    echo Could not back up the current app.js. Close it in any editor and try again.
    pause
    exit /b 1
  )
)
copy /Y "%~dp0app.js" "%PROJECT%\static\js\app.js" >nul
if errorlevel 1 (
  echo Could not install the corrected app.js.
  pause
  exit /b 1
)
echo.
echo Repair file installed successfully.
echo Backup: %PROJECT%\static\js\app.js.backup
echo Now refresh the ML Arena browser tab with Ctrl+F5.
start "" "http://127.0.0.1:5000/"
pause
