@echo off
REM Starts MLperformance on Windows. Creates the virtual environment on first run.
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt || goto :error
)
.venv\Scripts\python -m streamlit run app.py
goto :eof

:error
echo Setup failed. Make sure Python 3.10+ is installed and on your PATH.
pause
