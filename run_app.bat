@echo off
cd /d "%~dp0"
echo Starting MotionLearn Studio...
call .\venv\Scripts\activate.bat
python .\final-web-app\run_server.py
pause

