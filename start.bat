@echo off
title AIOS-Local
echo ========================================
echo   AIOS-Local - Starting...
echo ========================================

cd /d "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"

:: Ensure venv exists
if not exist "%PYEXE%" (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        py -3 -m venv .venv
    ) else (
        where python >nul 2>&1
        if %errorlevel% equ 0 (
            python -m venv .venv
        ) else (
            echo [ERROR] No Python launcher found. Install Python 3.10+ and rerun.
            pause
            exit /b 1
        )
    )
)

:: Install/update dependencies
echo [1/4] Installing dependencies...
"%PYEXE%" -m pip install -q -r requirements.txt

:: Run healthcheck
echo [2/4] Running healthcheck...
"%PYEXE%" tools\healthcheck.py
if %errorlevel% neq 0 (
    echo [ERROR] Healthcheck failed. Fix the above issues and rerun.
    pause
    exit /b 1
)

:: Start Ollama if not running
echo [3/4] Checking Ollama...
curl -s http://localhost:11434 >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Ollama...
    start "" ollama serve
    timeout /t 3 /nobreak >nul
)

:: Start server
echo [4/4] Starting AIOS-Local server...
echo.
echo   URL: http://localhost:8765
echo   Press Ctrl+C to stop
echo.
start http://localhost:8765

"%PYEXE%" server.py
