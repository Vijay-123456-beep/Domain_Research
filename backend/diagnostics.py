import os
import sys
import socket
import json
import importlib.util
from pathlib import Path

def check_python():
    print(f"--- Python Environment ---")
    print(f"Version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print()

def check_dependencies():
    print(f"--- Dependencies Status ---")
    libs = [
        "fastapi", "uvicorn", "fitz", "pandas", "numpy", 
        "easyocr", "requests", "dotenv", "pydantic", "ultralytics"
    ]
    for lib in libs:
        spec = importlib.util.find_spec(lib)
        status = "INSTALLED" if spec is not None else "MISSING"
        print(f"{lib.ljust(15)}: {status}")
    print()

def check_env_file():
    print(f"--- Environment Configuration (.env) ---")
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print("[ERROR] .env file not found in backend directory.")
        print("Action: Create a .env file based on .env.example")
        return False
    
    print(".env file exists.")
    
    # Load and check keys (don't print full keys for security)
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)
    
    keys = ["OPENROUTER_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "TOGETHER_API_KEY"]
    for k in keys:
        val = os.getenv(k)
        if not val:
            print(f"[WARNING] {k} is missing or empty.")
        else:
            print(f"{k}: Found (Starts with {val[:6]}...)")
    print()
    return True

def check_port(port=8001):
    print(f"--- Port Availability ({port}) ---")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            print(f"Port {port} is AVAILABLE.")
        except socket.error:
            print(f"[ERROR] Port {port} is ALREADY IN USE or blocked.")
            print(f"Action: Close the application using port {port} or set a different PORT in .env")
    print()

def check_network():
    print(f"--- Network Connectivity ---")
    endpoints = [
        ("Google (General)", "https://www.google.com"),
        ("OpenRouter (API)", "https://openrouter.ai/api/v1/models"),
        ("HuggingFace (Models)", "https://huggingface.co")
    ]
    import requests
    for name, url in endpoints:
        try:
            r = requests.get(url, timeout=5)
            status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
            print(f"{name.ljust(20)}: {status}")
        except Exception as e:
            print(f"{name.ljust(20)}: FAILED ({type(e).__name__})")
    print()

def main():
    print("========================================")
    print("   ScienceMiner Pro - Diagnostics")
    print("========================================\n")
    
    check_python()
    check_dependencies()
    env_ok = check_env_file()
    check_port()
    check_network()
    
    print("========================================")
    if env_ok:
        print("Diagnostics complete. If all checks passed, run:")
        print("python api_server.py")
    else:
        print("Diagnostics found critical issues (see above).")
    print("========================================")

if __name__ == "__main__":
    main()
