@echo off
REM ======================================================================
REM  RobustStego - local launcher (double-click)
REM  Starts the real server (app + API on one address) and opens
REM  http://localhost:8000 when it is ready. Laptop only.
REM  To show it to others on the internet, use tools\start_public.ps1.
REM ======================================================================
title RobustStego (local) - close this window to stop
cd /d "%~dp0"

REM Warn if the public Tailscale address is still on: then this local server is public too.
where tailscale >nul 2>nul
if %errorlevel%==0 (
    tailscale funnel status 2>nul | findstr /C:"ts.net" >nul
    if not errorlevel 1 (
        echo.
        echo  WARNING: the PUBLIC Tailscale address is ON. Anyone with the link can reach this server.
        echo  To make it local only, close this window and run:  .\tools\start_public.ps1 -Stop
        echo.
    )
)

REM Open the browser once the server answers (the model takes a few seconds to load).
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 90;$i++){try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 http://127.0.0.1:8000/api/health | Out-Null; Start-Process 'http://localhost:8000'; break}catch{Start-Sleep 1}}"

echo Starting RobustStego on http://localhost:8000 ...
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --no-server-header
if errorlevel 1 (
    echo.
    echo  The server stopped with an error. Common causes:
    echo   - port 8000 is already in use: another server window is still open
    echo   - a package is missing: pip install -r requirements.txt -r requirements-auth.txt
    echo   - .env is missing or incomplete: see .env.example
)
pause
