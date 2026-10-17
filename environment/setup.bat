@echo off
REM ============================================================
REM  LINGORA environment setup (offline, no network needed)
REM  Only dependency: jieba (installed from local wheel)
REM ============================================================
cd /d "%~dp0"

echo [1/3] Creating isolated venv .venv (does not touch global Python)...
py -3 -m venv .venv
if errorlevel 1 (
    echo [ERROR] Python 3 not found. Install Python 3.10+ first:
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [2/3] Installing jieba offline (local wheel)...
python -m pip install --no-index --find-links=. jieba==0.42.1
if errorlevel 1 (
    echo [ERROR] jieba install failed.
    pause
    exit /b 1
)

echo [3/3] Done.
echo.
echo Run LINGORA from the project root:
echo     environment\.venv\Scripts\python.exe main.py
echo.
pause
