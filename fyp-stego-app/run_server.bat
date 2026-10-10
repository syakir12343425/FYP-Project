@echo off
title Steganography FYP Research UI Server
echo ==========================================================
echo   Starting Local Server: Steganography Simulation Suite
echo ==========================================================
echo.
echo [1/2] Opening your default web browser to http://localhost:8080...
timeout /t 2 /nobreak > nul
start "" "http://localhost:8080"

echo.
echo [2/2] Launching Python HTTP Server...
echo (To stop the server, close this window or press Ctrl+C)
echo.
python -m http.server 8080

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Python server failed to start.
    echo Trying Node.js http-server instead...
    echo.
    npx http-server -p 8080
)

pause
