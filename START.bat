@echo off
echo Starting Investment Cockpit...
cd /d "%~dp0"
python -m streamlit run app.py
pause


