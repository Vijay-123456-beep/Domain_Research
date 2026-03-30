import os
import subprocess
import asyncio
import pandas as pd
import uuid
import shutil
import zipfile
import io
import json

# Load .env file before any module that reads environment variables
from pathlib import Path
from dotenv import load_dotenv
_env_path = Path(__file__).parent / ".env"
_env_exists = _env_path.exists()
load_dotenv(dotenv_path=_env_path, override=False)

if not _env_exists:
    print("\n[WARNING] .env file not found in 'backend/' directory!")
    print("AI extraction may fail. Please run 'setup.sh' or create a .env file.\n")
elif not os.getenv("OPENROUTER_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    print("\n[WARNING] No LLM API keys found in .env!")
    print("AI extraction and validation will be disabled.\n")

import sys
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from keyword_extractor import extract_keywords
from cleanup_manager import start_cleanup_scheduler


app = FastAPI()

# Enable CORS for React frontend (Universal)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Start the background task cleanup scheduler
    start_cleanup_scheduler()

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Workspace root
SESSION_ROOT = os.path.join(PROJECT_ROOT, "sessions")
if not os.path.exists(SESSION_ROOT):
    os.makedirs(SESSION_ROOT)

class JobStatus:
    def __init__(self, job_id: str, workspace: str):
        self.job_id = job_id
        self.workspace = workspace
        self.is_running = False
        self.logs = []
        self.current_step = "Idle"
        self.result_file = os.path.join(workspace, "resultant_dataset.csv")
        self.process = None  # Track current active subprocess
        self.aborted = False # Track if manually stopped

# Global job store
jobs: Dict[str, JobStatus] = {}

class SearchQuery(BaseModel):
    query: str
    keywords: List[str]
    attributes: List[str] # New: SSA, Pore Size, etc.
    limit: Optional[int] = 200
    task_name: Optional[str] = None
    job_id: Optional[str] = None

async def run_pipeline(job_id: str, query: str, keywords: str, attributes: str, limit: int = 200):
    job = jobs[job_id]
    job.is_running = True
    job.logs = ["--- New Research Session Started ---"]
    
    workspace = job.workspace
    
    try:
        # Define the pipeline sequence
        # Note: We pass the workspace to each script via --workspace
        # Use the current Python interpreter environment
        python_cmd = sys.executable
        
        steps = [
            ("Downloading PDFs...", [python_cmd, "backend/api_paper_downloader.py", "--query", query, "--keywords", keywords, "--limit", str(limit), "--workspace", workspace]),
            ("NLP Screening...", [python_cmd, "backend/nlp_screening.py", "--keywords", keywords, "--workspace", workspace]),
            ("Moving Included Papers...", [python_cmd, "backend/copy_included.py", "--workspace", workspace]),
            ("Generating Aliases...", [python_cmd, "backend/alias_generator.py", "--attributes", attributes, "--workspace", workspace]),
            ("Analyzing Pages...", [python_cmd, "backend/page_analyzer.py", "--attributes", attributes, "--workspace", workspace]),
            ("Extracting Text...", [python_cmd, "backend/extract_text.py", "--attributes", attributes, "--workspace", workspace]),
            ("Extracting Tables...", [python_cmd, "backend/extract_table.py", "--attributes", attributes, "--workspace", workspace]),
            ("Processing Plots...", [python_cmd, "backend/extract_plots.py", "--attributes", attributes, "--workspace", workspace]),
            ("Merging Datasets...", [python_cmd, "backend/merge_datasets.py", "--attributes", attributes, "--workspace", workspace]),
        ]

        for step_name, cmd in steps:
            job.current_step = step_name
            job.logs.append(f"--- Starting: {step_name} ---")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=PROJECT_ROOT
            )
            job.process = process
            
            async def read_stream(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line_decoded = line.decode(errors='replace').strip()
                    if line_decoded:
                        job.logs.append(line_decoded)
                        # Use Bold Cyan for mirrored terminal logs to ensure visibility
                        print(f"\033[1m\033[96m[{job_id}][{step_name}] {line_decoded}\033[0m")

            await asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr)
            )
            
            await process.wait()
            job.process = None # Clear after step finishes
            
            if job.aborted:
                job.logs.append("--- Job Terminated by User ---")
                job.current_step = "Stopped"
                return

            if process.returncode != 0:
                job.logs.append(f"Error: {step_name} failed with exit code {process.returncode}")
                # Log if it was likely a script error or environment error
                if process.returncode == 1:
                    job.logs.append("Note: Exit code 1 often indicates a Python script error or missing dependency.")
                elif process.returncode == 127:
                    job.logs.append(f"Note: Exit code 127 indicates the command was not found. Check if Python is installed correctly.")
                job.current_step = "Failed"
                return

        job.logs.append("Please click active dataset button you can see the dataset and you can also download pdf .zip and .csv file")
        job.current_step = "Completed"
    except Exception as e:
        job.logs.append(f"Fatal Error: {str(e)}")
        job.current_step = "Failed"
    finally:
        job.is_running = False

@app.post("/start-mining")
async def start_mining(payload: SearchQuery, background_tasks: BackgroundTasks):
    # Use task_name with a unique suffix for collision prevention
    raw_task_name = payload.task_name or "run"
    sanitized_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in raw_task_name])
    # Append 4-char unique suffix
    job_id = f"{sanitized_name}_{str(uuid.uuid4())[:4]}"
    
    workspace = os.path.join(SESSION_ROOT, job_id)
    pdf_dir = os.path.join(PROJECT_ROOT, "PDFs", job_id)
    included_dir = os.path.join(PROJECT_ROOT, "Included", job_id)
    
    # Create necessary task folders
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(os.path.join(workspace), exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(included_dir, exist_ok=True)
    
    if job_id in jobs and jobs[job_id].is_running:
        raise HTTPException(status_code=400, detail="Job already running")
    
    jobs[job_id] = JobStatus(job_id, workspace)
    
    raw_keywords = payload.keywords
    if not raw_keywords or len(raw_keywords) == 0:
        # Automated LLM Extraction — returns structured {"primary": [...], "secondary": [...]}
        keywords_dict = extract_keywords(payload.query)
    else:
        # User-provided flat list — split evenly into primary/secondary
        mid = len(raw_keywords) // 2
        keywords_dict = {
            "primary": raw_keywords[:mid] or raw_keywords,
            "secondary": raw_keywords[mid:]
        }
    
    # Serialize to JSON string for CLI --keywords argument
    keywords_str = json.dumps(keywords_dict)
    
    # Refine Search Query: use top primary keywords to narrow API search
    primary_kw = keywords_dict.get("primary", [])
    refined_query = payload.query
    if primary_kw:
        refiners = " ".join(primary_kw[:2])
        refined_query = f"{payload.query} {refiners}"
        
    attributes_str = ",".join(payload.attributes)
    background_tasks.add_task(run_pipeline, job_id, refined_query, keywords_str, attributes_str, payload.limit)
    
    return {"message": "Pipeline started", "job_id": job_id}

@app.post("/stop-mining/{job_id}")
async def stop_mining(job_id: str):
    if job_id not in jobs:
        # Check if workspace exists even if job object is gone (server restart)
        workspace = os.path.join(SESSION_ROOT, job_id)
        if os.path.exists(workspace):
             return {"message": "Job is already inactive (Session exists on disk)"}
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if not job.is_running:
        return {"message": "Job is not running"}
    
    job.aborted = True
    if job.process:
        try:
            job.process.terminate()
            job.logs.append("!!! Manual Termination Signal Sent !!!")
        except Exception as e:
            job.logs.append(f"Termination Error: {str(e)}")
            
    return {"message": "Termination signal sent"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        # Check if workspace exists even if job object is gone (server restart)
        workspace = os.path.join(SESSION_ROOT, job_id)
        if os.path.exists(workspace):
             return {"is_running": False, "logs": ["Session exists on disk but inactive."], "current_step": "Idle"}
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "is_running": job.is_running,
        "logs": job.logs,
        "current_step": job.current_step
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    workspace = os.path.join(SESSION_ROOT, job_id)
    result_file = os.path.join(workspace, "resultant_dataset.csv")
    
    if not os.path.exists(result_file):
        return {"data": []}
    
    df = pd.read_csv(result_file)
    # Safely convert NaNs so they can be parsed by JSON cleanly
    df = df.fillna('')
    return {"data": df.head(1000).to_dict(orient="records")}

@app.get("/download-pdfs/{job_id}")
async def download_pdfs(job_id: str):
    workspace = os.path.join(SESSION_ROOT, job_id)
    result_file = os.path.join(workspace, "resultant_dataset.csv")
    pdf_dir_pdfs = os.path.join(PROJECT_ROOT, "PDFs", job_id)
    pdf_dir_included = os.path.join(PROJECT_ROOT, "Included", job_id)
    
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
        
    df = pd.read_csv(result_file)
    if 'File' not in df.columns:
        raise HTTPException(status_code=400, detail="No File column in dataset")
        
    unique_pdfs = df['File'].dropna().unique()
    
    # Create an in-memory zip file
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for pdf in unique_pdfs:
            pdf_path_inc = os.path.join(pdf_dir_included, pdf)
            pdf_path_pdf = os.path.join(pdf_dir_pdfs, pdf)
            if os.path.exists(pdf_path_inc):
                zf.write(pdf_path_inc, arcname=pdf)
            elif os.path.exists(pdf_path_pdf):
                zf.write(pdf_path_pdf, arcname=pdf)
                
    memory_file.seek(0)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        iter([memory_file.getvalue()]), 
        media_type="application/zip", 
        headers={"Content-Disposition": f"attachment; filename=extracted_papers_{job_id}.zip"}
    )

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    workspace = os.path.join(SESSION_ROOT, job_id)
    result_file = os.path.join(workspace, "resultant_dataset.csv")
    
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(result_file, media_type='text/csv', filename=f"mining_results_{job_id}.csv")

@app.get("/pdf/{job_id}/{filename}")
async def get_pdf(job_id: str, filename: str):
    pdf_path_inc = os.path.join(PROJECT_ROOT, "Included", job_id, filename)
    pdf_path_pdf = os.path.join(PROJECT_ROOT, "PDFs", job_id, filename)
    
    if os.path.exists(pdf_path_inc):
        return FileResponse(pdf_path_inc, media_type='application/pdf')
    elif os.path.exists(pdf_path_pdf):
        return FileResponse(pdf_path_pdf, media_type='application/pdf')
        
    raise HTTPException(status_code=404, detail="PDF not found")

if __name__ == "__main__":
    import uvicorn
    # Allow port to be set via PORT environment variable
    port = int(os.getenv("PORT", 8001))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
