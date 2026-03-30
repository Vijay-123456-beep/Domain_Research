@echo off
setlocal

echo ========================================
echo    ScienceMiner Pro - Windows Setup
echo ========================================

REM 1. Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in path! Please install it first.
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PY_VER=%%i
echo Using: %PY_VER%

REM 2. Setup .env if it doesn't exist
if not exist "backend\.env" (
    echo Creating .env template from backend\.env.example...
    if exist "backend\.env.example" (
        copy backend\.env.example backend\.env
    else
        echo OPENROUTER_API_KEY= > backend\.env
        echo GROQ_API_KEY= >> backend\.env
        echo GEMINI_API_KEY= >> backend\.env
    fi
    echo [IMPORTANT] Please edit backend\.env with your API keys!
)

REM 3. Install dependencies
echo Installing dependencies...
python -m pip install -r backend\requirements.txt

REM 4. Run Diagnostics
echo Running Diagnostics...
python backend\diagnostics.py

echo Done! If all checks passed, start the server with:
echo cd backend ^&^& python api_server.py
pause
