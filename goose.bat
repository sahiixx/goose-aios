@echo off
REM ── Goose Local AI Agent ──────────────────────────────────────────
REM Runs Goose CLI with Ollama (fully local, no API keys, no tokens)
REM Prereqs: Ollama running + model pulled (ollama pull qwen2.5-coder:7b)

set "PATH=%USERPROFILE%\.local\bin;%PATH%"

REM Ensure Ollama is reachable
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [!] Ollama is not running. Start it first: ollama serve
    pause
    exit /b 1
)

echo.
echo  === Goose Local AI Agent ===
echo  Provider: Ollama (local)
echo  Model:    qwen2.5-coder:7b
echo  No API keys. No tokens. No cloud.
echo.

REM Launch interactive goose session in this directory
cd /d "%~dp0"
goose session %*
