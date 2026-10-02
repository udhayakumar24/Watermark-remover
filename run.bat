@echo off
REM One-time setup + launch. Windows.
cd /d "%~dp0"

if not exist .venv (
  echo Creating virtualenv...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo Checking the neural model...
python scripts\fetch_model.py

echo.
python app.py %*
pause
