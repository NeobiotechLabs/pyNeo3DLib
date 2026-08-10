@echo off
rem pyNeo3DLib FastAPI 서버를 venv로 실행 (더블클릭용)
cd /d "%~dp0"
call venv\Scripts\activate.bat
python run_server.py
pause
