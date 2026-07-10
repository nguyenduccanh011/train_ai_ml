@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PORT=8080"
set "SERVER_URL=http://localhost:%PORT%"

cd /d "%SCRIPT_DIR%"

echo Checking for existing server on port %PORT%...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT%" ^| findstr LISTENING') do (
    echo Stopping existing process %%P...
    taskkill /PID %%P /F >nul 2>&1
)

echo Starting visualization server on port %PORT%...
start "Stock ML Visualization Server" cmd /k "python serve.py --port %PORT%"

echo Waiting for server to warm up...
timeout /t 3 /nobreak >nul

echo Opening browser...
start "" "%SERVER_URL%/visualization/leaderboard.html"
start "" "%SERVER_URL%/visualization/dashboard.html"

endlocal
