import json
import os
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))

def setup_paths(workspace=None):
    if workspace:
        project_root = os.path.dirname(os.path.dirname(workspace))
        task_id = os.path.basename(workspace)
        
        json_file = os.path.join(workspace, "2_Screening_Results", "screening_results.json")
        source_dir = os.path.join(project_root, "PDFs", task_id)
        target_dir = os.path.join(project_root, "Included", task_id)
    else:
        json_file = os.path.join(base_dir, "2_Screening_Results", "screening_results.json")
        source_dir = os.path.join(os.path.dirname(base_dir), "PDFs")
        target_dir = os.path.join(os.path.dirname(base_dir), "Included")
    return json_file, source_dir, target_dir

def main(workspace=None):
    json_file, source_dir, target_dir = setup_paths(workspace)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get list of all PDFs in source directory
    all_pdfs = []
    if os.path.exists(source_dir):
        all_pdfs = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]
    
    included_files = []
    
    # Try to load screening results
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            included_files = [item['file'] for item in results if item.get('include') is True]
        except Exception as e:
            print(f"Warning: Could not parse screening results: {e}")
    
    # Fallback: if no screening results or empty, include ALL PDFs
    if not included_files and all_pdfs:
        print(f"No screening results found. Falling back to including all {len(all_pdfs)} PDFs.")
        included_files = all_pdfs

    copied_count = 0
    for filename in included_files:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"Warning: File not found: {src_path}")

    print(f"Total papers marked 'include=true': {len(included_files)}")
    print(f"Total papers successfully copied: {copied_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Moving Included Papers")
    parser.add_argument("--workspace", type=str, help="Workspace directory for this session")
    args = parser.parse_args()
    
    main(args.workspace)
