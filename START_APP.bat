@echo off
echo === AI Production Cycle Monitor ===
cd /d %~dp0

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
	call .venv\Scripts\activate.bat
) else (
	echo [ERROR] .venv not found. Run setup first:
	echo   python -m venv .venv
	echo   .venv\Scripts\pip install -r ai_production_monitor\requirements.txt
	pause
	exit /b 1
)

REM Run the application
cd ai_production_monitor
python run.py

pause
