@echo off
cd /d "%~dp0"
echo Starting Ninjoucon Mocap Studio...
call .\venv\Scripts\activate.bat
python .\web_app\run_server.py
pause

