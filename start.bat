@echo off
REM ===========================================================================
REM  HumanBot one-click launcher (Windows)
REM
REM    start.bat                  normal start
REM    start.bat --port 8899      force a port
REM    start.bat --no-bots        do not log in bots
REM    start.bat --reset-password regenerate console password
REM
REM  Everything below is plain ASCII on purpose, so the file runs correctly
REM  under any Windows code page.
REM ===========================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

cd /d "%~dp0"

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "PY_BIN="
set "VP="

echo ==============================================================
echo   HumanBot - starting up
echo ==============================================================

REM ---------------------------------------------------------------------------
REM 1. Locate Python
REM ---------------------------------------------------------------------------
echo [1/6] Looking for Python ...

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1
    if !errorlevel!==0 (
        set "PY_BIN=py -3"
        goto :pyfound
    )
)

for %%C in (python python3 python3.13 python3.12 python3.11 python3.10 python3.9) do (
    if not defined PY_BIN (
        where %%C >nul 2>&1
        if !errorlevel!==0 (
            %%C -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1
            if !errorlevel!==0 set "PY_BIN=%%C"
        )
    )
)

if not defined PY_BIN (
    echo [ERROR] Python 3.9+ not found. Please install Python from python.org
    echo         and tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:pyfound
echo       using: !PY_BIN!

REM ---------------------------------------------------------------------------
REM 2. Create / reuse virtualenv
REM ---------------------------------------------------------------------------
if not exist "%VENV_DIR%" (
    echo [2/6] Creating virtual environment ^(.venv^) ...
    %PY_BIN% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtualenv.
        pause
        exit /b 1
    )
) else (
    echo [2/6] Reusing virtual environment ^(.venv^)
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    set "VP=%VENV_DIR%\Scripts\python.exe"
) else if exist "%VENV_DIR%\bin\python" (
    set "VP=%VENV_DIR%\bin\python"
) else (
    echo [ERROR] virtualenv looks broken. Delete .venv and run again.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM 3. Install dependencies
REM ---------------------------------------------------------------------------
set "STAMP=%VENV_DIR%\.deps_ok"
set "NEED_INSTALL=0"
if not exist "%STAMP%" set "NEED_INSTALL=1"
if exist "%STAMP%" (
    for %%F in ("%SCRIPT_DIR%\requirements.txt") do (
        if %%~tF GTR %%~tSTAMP set "NEED_INSTALL=1"
    )
)

if "%NEED_INSTALL%"=="1" (
    echo [3/6] Installing dependencies ^(first run may take a few minutes^) ...
    "%VP%" -m pip install --upgrade pip -q 2>nul
    "%VP%" -m pip install -r "%SCRIPT_DIR%\requirements.txt" -q
    if errorlevel 1 (
        echo       quiet install failed, retrying with full output ...
        "%VP%" -m pip install -r "%SCRIPT_DIR%\requirements.txt"
        if errorlevel 1 (
            echo [ERROR] Dependency install failed.
            pause
            exit /b 1
        )
    )
    echo. > "%STAMP%"
    echo       dependencies installed.
) else (
    echo [3/6] Dependencies already installed ^(skip^)
)

REM ---------------------------------------------------------------------------
REM 4. Runtime folders
REM ---------------------------------------------------------------------------
if not exist "%SCRIPT_DIR%\config.yaml" (
    echo [ERROR] config.yaml not found. Please restore it before starting.
    pause
    exit /b 1
)
if not exist "%SCRIPT_DIR%\persona.yaml" (
    echo [WARN] persona.yaml not found, the built-in default persona will be used.
)
if not exist "%SCRIPT_DIR%\data"   mkdir "%SCRIPT_DIR%\data"
if not exist "%SCRIPT_DIR%\logs"   mkdir "%SCRIPT_DIR%\logs"
if not exist "%SCRIPT_DIR%\assets" mkdir "%SCRIPT_DIR%\assets"

echo [4/6] Runtime folders ready ^(data / logs / assets^)

REM ---------------------------------------------------------------------------
REM 5. Free-port detection
REM ---------------------------------------------------------------------------
set "REQUESTED_PORT="
set "ALLARGS="
:parseargs
if "%~1"=="" goto :parsedone
if /i "%~1"=="--port" (
    set "REQUESTED_PORT=%~2"
    shift
    shift
    goto :parseargs
)
set "ALLARGS=%ALLARGS% %~1"
shift
goto :parseargs
:parsedone

if defined REQUESTED_PORT (
    echo [5/6] Using requested port: !REQUESTED_PORT!
) else (
    for /f "delims=" %%P in ('"%VP%" -c "import socket;s=socket.socket();s.bind(('0.0.0.0',0));print(s.getsockname()[1]);s.close()"') do set "FREE_PORT=%%P"
    echo [5/6] Free port will be auto-detected by the application
)

REM ---------------------------------------------------------------------------
REM 6. Launch
REM ---------------------------------------------------------------------------
echo [6/6] Launching HumanBot ...
echo.

if defined REQUESTED_PORT (
    "%VP%" -m humanbot.main --port %REQUESTED_PORT% %ALLARGS%
) else (
    "%VP%" -m humanbot.main %ALLARGS%
)

echo.
echo HumanBot stopped.
pause
endlocal
