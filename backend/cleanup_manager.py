import os
import shutil
import time
import threading
from pathlib import Path

# Directories to clean
TARGET_DIRS = ["sessions", "PDFs", "Included"]
# Cleanup threshold in days
CLEANUP_THRESHOLD_DAYS = 7
# Conversion from days to seconds
SECONDS_IN_DAY = 86400

def cleanup_old_tasks(days=CLEANUP_THRESHOLD_DAYS):
    """
    Scans target directories and recursively deletes folders older than 'days' days.
    """
    now = time.time()
    cutoff = now - (days * SECONDS_IN_DAY)
    
    # Get the project root (assumed to be parent of backend/)
    project_root = Path(__file__).parent.parent
    
    total_deleted = 0
    
    print(f"\n[CLEANUP] Starting task cleanup (threshold: {days} days)...")
    
    for dir_name in TARGET_DIRS:
        dir_path = project_root / dir_name
        
        if not dir_path.exists():
            continue
            
        print(f"[CLEANUP] Checking: {dir_path}")
        
        for item in dir_path.iterdir():
            if item.is_dir():
                # Check the last modification time of the folder
                mtime = item.stat().st_mtime
                if mtime < cutoff:
                    try:
                        folder_age_days = (now - mtime) / SECONDS_IN_DAY
                        print(f"  [DELETE] Removing {item.name} (Age: {folder_age_days:.1f} days)...")
                        shutil.rmtree(item)
                        total_deleted += 1
                    except Exception as e:
                        print(f"  [ERROR] Failed to delete {item.name}: {e}")
                        
    if total_deleted > 0:
        print(f"[CLEANUP] Finished. Deleted {total_deleted} old task folders.\n")
    else:
        print(f"[CLEANUP] Finished. No old folders found.\n")

def start_cleanup_scheduler():
    """
    Starts a background daemon thread that performs cleanup immediately, 
    then every 24 hours.
    """
    def run_periodically():
        while True:
            try:
                cleanup_old_tasks()
            except Exception as e:
                print(f"[CLEANUP_SCHEDULER] Error during cleanup: {e}")
            
            # Sleep for 24 hours (86400 seconds)
            time.sleep(SECONDS_IN_DAY)

    # Start as a daemon thread so it exits when the main process stops
    cleanup_thread = threading.Thread(target=run_periodically, daemon=True)
    cleanup_thread.start()
    print("[SYSTEM] Background Cleanup Scheduler started (24h period).")

if __name__ == "__main__":
    # For manual testing: run with 0 days to delete everything or provide a value
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else CLEANUP_THRESHOLD_DAYS
    cleanup_old_tasks(days)
