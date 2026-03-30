#!/bin/bash

echo "========================================"
echo "   ScienceMiner Pro - Linux Setup"
echo "========================================"

# 1. Check for Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python was not found! Please install it first."
    exit 1
fi

echo "Using: $($PYTHON_CMD --version)"

# 2. Setup .env if it doesn't exist
if [ ! -f "backend/.env" ]; then
    echo "Creating .env template from backend/.env.example..."
    if [ -f "backend/.env.example" ]; then
        cp backend/.env.example backend/.env
    else
        touch backend/.env
        echo "OPENROUTER_API_KEY=" >> backend/.env
        echo "GROQ_API_KEY=" >> backend/.env
        echo "GEMINI_API_KEY=" >> backend/.env
    fi
    echo "[IMPORTANT] Please edit backend/.env with your API keys!"
fi

# 3. Install dependencies
echo "Installing dependencies..."
$PYTHON_CMD -m pip install -r backend/requirements.txt

# 4. Run Diagnostics
echo "Running Diagnostics..."
$PYTHON_CMD backend/diagnostics.py

echo "Done! If all checks passed, start the server with:"
echo "cd backend && $PYTHON_CMD api_server.py"
