@echo off
REM ======================================================================
REM  RobustStego - DEMO launcher (double-click)
REM  Opens straight onto the FYP: no sign-in, no accounts, no database,
REM  so it works without internet. THIS LAPTOP ONLY: the server refuses
REM  anything that does not come straight from this computer.
REM  For the full system use run_server.bat or tools\start_public.ps1.
REM ======================================================================
title RobustStego DEMO (this laptop only) - close this window to stop
cd /d "%~dp0"

REM Refuse to start while the public Tailscale address is on: the demo has no sign-in.
where tailscale >nul 2>nul
if %errorlevel%==0 (
    tailscale funnel status 2>nul | findstr /C:"ts.net" >nul
    if not errorlevel 1 (
        echo.
        echo  The PUBLIC Tailscale address is ON. The demo has no sign-in, so it will not start.
        echo  Turn the public address off first:  .\tools\start_public.ps1 -Stop
        echo.
        pause
        exit /b 1
    )
)

set DEMO_MODE=true

REM Open the browser once the server answers (the model takes a few seconds to load).
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 90;$i++){try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 http://127.0.0.1:8000/api/health | Out-Null; Start-Process 'http://localhost:8000'; break}catch{Start-Sleep 1}}"

echo Starting the RobustStego DEMO on http://localhost:8000 (this laptop only) ...
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --no-server-header
if errorlevel 1 (
    echo.
    echo  The demo stopped with an error. Common causes:
    echo   - port 8000 is already in use: another server window is still open
    echo   - a package is missing: pip install -r requirements.txt -r requirements-auth.txt
    echo   - .env is missing or incomplete: see .env.example
)
pause
