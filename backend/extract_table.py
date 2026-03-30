import os
import json
import re
import csv
import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time

# Import cache manager
from cache_manager import get_cache_manager, cache_result, table_cache_key

from schema_loader import SchemaLoader, load_domain_schema
from unit_parser import UnitParser, get_unit_parser, ParsedValue
from validation_engine import ValidationEngine, get_validation_engine, ValidationResult, ValidationStatus
from llm_validator import validate_table_headers_llm, classify_table_cells_batch_llm, validate_table_row_llm

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Defaults - will be overridden by --workspace)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INCLUDED_DIR = os.path.join(os.path.dirname(BASE_DIR), "Included")
ANALYSIS_JSON = os.path.join(BASE_DIR, "page_analysis_results.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "table_extracted_data.csv")

COLUMNS = []
USER_ATTRIBUTES = []
SCHEMA_LOADER = None
UNIT_PARSER = None
VALIDATION_ENGINE = None

def is_false_positive(line, match_str, attr, expected_units=None):
    """
    Check if a match is a likely false positive (formula constant, figure ref, etc).
    """
    line_lower = line.lower()
    
    # 1. Figure/Table Reference Filtering
    ref_pattern = r'(?i)\b(?:fig(?:ure)?|table|ref|eq|step)\.?\s*\(?'+re.escape(match_str)+r'\b'
    if re.search(ref_pattern, line):
        return True, "Figure/Table Reference"

    # 2. Formula Detection
    if '=' in line:
        formula_vars = ['cs', 'csp', 'es', 'ps', 'td', 'v\'', 'v\"', '3600', '0.5']
        found_vars = [v for v in formula_vars if v in line_lower]
        if len(found_vars) >= 2 or '*' in line or '/' in line:
             return True, "Formula Constant"

    # 3. Unit-Aware Validation
    has_unit = False
    status_msg = "Missing required unit"
    valid_units = [u.lower().strip() for u in expected_units if u and u.strip()]
    
    if valid_units:
        # Check both after AND before the value for units
        p_after = re.escape(match_str) + r'.{0,15}(?:' + '|'.join([re.escape(u) for u in valid_units]) + r')(?:\b|(?![a-z]))'
        p_before = r'(?:' + '|'.join([re.escape(u) for u in valid_units]) + r').{0,5}' + re.escape(match_str)
        if re.search(p_after, line_lower) or re.search(p_before, line_lower):
            has_unit = True
        else:
            status_msg = f"Missing unit ({'/'.join(valid_units)})"
    else:
        # No units expected for this attribute
        has_unit = True

    # 4. Square Bracket Reference Filtering
    if re.search(r'\[\s*\d+[^\]]*' + re.escape(match_str) + r'[^\]]*\]', line):
        return True, "Square Bracket Reference"

    if not has_unit:
        return True, status_msg

    return False, None

def setup_paths(workspace_dir=None, attributes=None):
    global INCLUDED_DIR, ANALYSIS_JSON, OUTPUT_CSV, COLUMNS, USER_ATTRIBUTES
    global SCHEMA_LOADER, UNIT_PARSER, VALIDATION_ENGINE
    
    # Initialize new validation modules
    if workspace_dir:
        SCHEMA_LOADER = load_domain_schema(workspace_dir)
        UNIT_PARSER = get_unit_parser()
        VALIDATION_ENGINE = get_validation_engine(SCHEMA_LOADER)
        
    if SCHEMA_LOADER and hasattr(SCHEMA_LOADER, 'schema') and SCHEMA_LOADER.schema:
        USER_ATTRIBUTES = list(SCHEMA_LOADER.schema.keys())
        COLUMNS = ["File"] + USER_ATTRIBUTES + ["Proof"]
    elif attributes:
        USER_ATTRIBUTES = [a.strip() for a in attributes.split(",") if a.strip()]
        COLUMNS = ["File"] + USER_ATTRIBUTES + ["Proof"]
    else:
        USER_ATTRIBUTES = []
        COLUMNS = ["File", "Proof"]

    # Load aliases and units dynamically (legacy support)
    global DYNAMIC_ALIASES
    DYNAMIC_ALIASES = {}
    if workspace_dir:
        alias_file = os.path.join(workspace_dir, "aliases.json")
        if os.path.exists(alias_file):
            try:
                with open(alias_file, 'r', encoding='utf-8') as f:
                    raw_aliases = json.load(f)
                    DYNAMIC_ALIASES = {k.lower(): v for k, v in raw_aliases.items()}
            except Exception as e:
                print(f"Error loading aliases.json: {e}")

    if workspace_dir:
        project_root = os.path.dirname(os.path.dirname(workspace_dir))
        task_id = os.path.basename(workspace_dir)
        
        INCLUDED_DIR = os.path.join(project_root, "Included", task_id)
        ANALYSIS_JSON = os.path.join(workspace_dir, "page_analysis_results.json")
        OUTPUT_CSV = os.path.join(workspace_dir, "table_extracted_data.csv")
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


def reconstruct_table_headers_v2(table_data):
    """
    NEW: Multi-row header reconstruction with unit extraction.
    Returns list of column info with (attribute, unit, confidence).
    """
    if not table_data or len(table_data) < 2:
        return [], 0
    
    header_rows = []
    data_start_row = 0
    
    # Step 1: Detect header boundary
    for i, row in enumerate(table_data[:min(5, len(table_data))]):
        if not row:
            continue
            
        text_cells = sum(1 for cell in row if cell and not is_numeric_cell(cell))
        numeric_cells = sum(1 for cell in row if is_numeric_cell(cell))
        
        # Header indicators: mostly text, contains keywords, few numbers
        if text_cells > numeric_cells and text_cells > 0:
            header_rows.append(i)
        else:
            data_start_row = i
            break
    
    if not header_rows:
        data_start_row = 0
    
    # Step 2: Merge header rows and collect sample data for LLM
    num_cols = max(len(row) for row in table_data[:5]) if table_data else 0
    headers_with_samples = []
    
    for col_idx in range(num_cols):
        header_parts = []
        for h_row in header_rows:
            if col_idx < len(table_data[h_row]):
                cell = table_data[h_row][col_idx]
                if cell and str(cell).strip():
                    header_parts.append(str(cell).strip())
        
        full_header = " ".join(header_parts)
        
        # Collect top 3 sample values from this column
        samples = []
        for r_idx in range(data_start_row, min(data_start_row + 5, len(table_data))):
             if col_idx < len(table_data[r_idx]):
                 c_val = str(table_data[r_idx][col_idx]).strip()
                 if c_val: samples.append(c_val)
                 
        headers_with_samples.append((full_header, samples[:3]))
        
    # Step 3: Run LLM over all headers in ONE batch call
    print(f"Validating {len(headers_with_samples)} table headers via LLM...")
    llm_header_mapping = validate_table_headers_llm(headers_with_samples, USER_ATTRIBUTES, SCHEMA_LOADER)
    
    merged_headers = []
    for col_idx, (full_header, _) in enumerate(headers_with_samples):
        # Default to old parsing logic if LLM failed/skipped
        attribute, unit, confidence = parse_header_v2(full_header)
        
        if llm_header_mapping and full_header in llm_header_mapping:
             llm_data = llm_header_mapping[full_header]
             attribute = llm_data.get('attribute')
             unit = llm_data.get('unit')
             confidence = llm_data.get('confidence', 0.0)
        
        merged_headers.append({
            'col': col_idx,
            'raw_header': full_header,
            'attribute': attribute,
            'unit': unit,
            'confidence': confidence
        })
    
    return merged_headers, data_start_row


def is_numeric_cell(cell):
    """Check if cell contains a numeric value."""
    if not cell:
        return False
    cell_str = str(cell).strip()
    # Pattern for numeric values including scientific notation
    return bool(re.search(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', cell_str))


def parse_header_v2(header_text):
    """
    NEW: Parse header text to extract attribute name and unit.
    Uses schema matching and unit detection.
    """
    if not header_text or not SCHEMA_LOADER:
        return None, None, 0.0
    
    header_clean = re.sub(r'\s+', ' ', str(header_text)).strip()
    
    # Step 1: Try to find matching attribute from schema
    best_attr = None
    best_score = 0.0
    
    for attr_name in SCHEMA_LOADER.schema.keys():
        attr_data = SCHEMA_LOADER.schema[attr_name]
        aliases = [attr_name] + attr_data.get('aliases', [])
        
        for alias in aliases:
            alias_clean = alias.lower()
            header_lower = header_clean.lower()
            
            # Direct match
            if alias_clean in header_lower:
                score = len(alias_clean) / len(header_lower)
                if score > best_score:
                    best_score = score
                    best_attr = attr_name
            
            # Partial word matching
            alias_words = alias_clean.split()
            header_words = header_lower.split()
            matches = sum(1 for w in alias_words if w in header_words and len(w) > 3)
            if matches >= max(1, len(alias_words) // 2):
                score = matches / len(alias_words)
                if score > best_score:
                    best_score = score
                    best_attr = attr_name
    
    # Step 2: Extract unit from header
    unit = None
    if UNIT_PARSER:
        unit = UNIT_PARSER.extract_unit_from_header(header_clean)
    
    # If no unit in header, try to get from schema
    if not unit and best_attr:
        attr_data = SCHEMA_LOADER.schema.get(best_attr, {})
        units = attr_data.get('units', [])
        if units:
            unit = units[0]  # Use first expected unit as hint
    
    confidence = min(1.0, best_score + (0.2 if unit else 0))
    
    return best_attr, unit, confidence


# Cache compiled regex patterns for performance
DESCRIPTION_PATTERNS = [
    re.compile(r'\b(synthesis|preparation|fabrication|method|process|procedure)\b', re.IGNORECASE),
    re.compile(r'\b(carbonization|pyrolysis|annealing|hydrothermal|sol-gel)\b', re.IGNORECASE),
    re.compile(r'\b(°C|temperature|heated|cooled)\b.*\b(hour|hr|min|second)\b', re.IGNORECASE),
    re.compile(r'\b(material|sample|specimen|composite|substrate)\b', re.IGNORECASE),
    re.compile(r'\b(growth|treatment|reduction|dispersed|washed|prepared)\b', re.IGNORECASE),
]

# Pre-compiled number regex
NUM_REGEX = re.compile(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)')

from concurrent.futures import ThreadPoolExecutor

def extract_cell_value_v2(cell_value, header_info):
    """
    NEW: Extract value from cell with unit-aware validation.
    STRICT: Rejects any non-numeric or overly long content.
    """
    if not cell_value:
        return None
    
    cell_str = str(cell_value).strip()
    
    # STRICT: Reject long text immediately (likely descriptions, not values)
    if len(cell_str) > 50:
        return None
    
    # STRICT: Reject if contains newlines (multi-line content = description)
    if '\n' in cell_str or '\r' in cell_str:
        return None
    
    # STRICT: Reject if no numeric content at all
    if not NUM_REGEX.search(cell_str):
        return None
    
    # STRICT: Reject common description patterns (using pre-compiled regex)
    for pattern in DESCRIPTION_PATTERNS:
        if pattern.search(cell_str):
            return None
    
    # Use unit parser
    header_unit = header_info.get('unit')
    parsed = None
    
    if UNIT_PARSER:
        parsed = UNIT_PARSER.extract_from_table_cell(cell_str, header_unit)
    
    if not parsed:
        # Fallback: simple numeric extraction
        num_match = NUM_REGEX.search(cell_str)
        if num_match:
            try:
                val = num_match.group(1)
                parsed = ParsedValue(
                    value=val,
                    raw_value=cell_str,
                    unit=header_unit,
                    raw_unit=header_unit,
                    confidence=0.4,
                    value_type='extracted'
                )
            except ValueError:
                return None
    
    return parsed


def validate_and_map_cell_v2(parsed_value, header_info, filename, page_num, row_idx, col_idx):
    """
    NEW: Validate extracted value against schema and map to attribute.
    """
    if not parsed_value or not VALIDATION_ENGINE:
        return None
    
    header_attr = header_info.get('attribute')
    
    # Determine target attribute
    target_attr = header_attr
    if not target_attr:
        # Try to infer from unit dimension
        if parsed_value.unit:
            unit_dim = UNIT_PARSER.get_dimension(parsed_value.unit) if UNIT_PARSER else None
            if unit_dim and SCHEMA_LOADER:
                for attr_name, attr_data in SCHEMA_LOADER.schema.items():
                    if attr_data.get('dimension') == unit_dim:
                        target_attr = attr_name
                        break
    
    if not target_attr:
        return None
    
    # Build context for validation
    context = {
        'source_type': 'table_cell',
        'cell_content': parsed_value.raw_value,
        'page': page_num,
        'row': row_idx,
        'col': col_idx
    }
    
    # Run validation
    result = VALIDATION_ENGINE.validate_datapoint(
        value=parsed_value.value,
        unit=parsed_value.unit,
        attribute=target_attr,
        context=context
    )
    
    if not result.is_acceptable(0.5):
        return None
    
    return {
        'attribute': result.attribute,
        'value': result.value,
        'unit': result.unit,
        'confidence': result.confidence,
        'raw_value': parsed_value.raw_value
    }


def extract_table_data_v2(table, filename: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Extract data from table using schema-aware validation.
    Cached to avoid reprocessing identical tables.
    """
    cache_key = table_cache_key(tuple(tuple(row) for row in table.extract()))
    cache = get_cache_manager()
    
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    if not table:
        result = []
    else:
        table_data = table.extract()
        if not table_data or len(table_data) < 2:
            result = []
        else:
            # Reconstruct headers
            headers, data_start = reconstruct_table_headers_v2(table_data)
            
            # Batch processing for ambiguous cells
            ambiguous_cells_by_col = {} # col_idx -> list of {id, value_str}
            cell_objects = {} # Keep references to update later
            results = []
            
            # Process data rows (First Pass: Extract & Collect Ambiguous)
            for row_idx in range(data_start, len(table_data)):
                row = table_data[row_idx]
                row_result = {'_metadata': {'page': page_num, 'table_source': filename, 'row_idx': row_idx}}
                
                for col_idx, (header, cell_value) in enumerate(zip(headers, row)):
                    if not header or not cell_value:
                        continue
                        
                    header_conf = header.get('confidence', 0)
                    header_attr = header.get('attribute')
                    header_unit = header.get('unit')
                    
                    # If LLM header validation failed or returned UNKNOWN_NOISE, skip cell
                    if not header_attr or header_attr == "UNKNOWN_NOISE":
                        continue
                    
                    parsed = extract_cell_value_v2(cell_value, header)
                    if not parsed:
                        continue
                        
                    # GAP 2 Fix: Check Unit heuristics BEFORE LLM
                    unit_is_condition = ["a/g", "x", "c", "v", "mv/s", "ma/c", "k", "h"]
                    is_ambiguous = False
                    
                    if parsed.unit and parsed.unit.lower() in unit_is_condition:
                        is_ambiguous = True # Might be a condition disguised as a result
                        
                    if not parsed.unit and header_conf < 0.75:
                        is_ambiguous = True # Low confidence header + no cell unit = ambiguous
                        
                    # If ambiguous, queue it for the LLM
                    if is_ambiguous:
                        if col_idx not in ambiguous_cells_by_col:
                            ambiguous_cells_by_col[col_idx] = []
                        cell_id = f"{row_idx}_{col_idx}"
                        ambiguous_cells_by_col[col_idx].append({"id": cell_id, "value_str": parsed.raw_value})
                        # Store placeholder
                        cell_objects[cell_id] = {
                            "parsed": parsed,
                            "header": header,
                            "row_idx": row_idx,
                            "col_idx": col_idx,
                            "row_result_ref": row_result
                        }
                    else:
                        # Direct local validation
                        validated = validate_and_map_cell_v2(parsed, header, filename, page_num, row_idx, col_idx)
                        if validated:
                            row_result[validated['attribute']] = validated
                
                results.append(row_result)
            
            # Second Pass: Resolve Ambiguous Cells via LLM Batching
            for col_idx, cells in ambiguous_cells_by_col.items():
                if not cells: continue
                # We do 1 LLM call per ambiguous column to batch context logically
                header_info = headers[col_idx]
                print(f"Resolving {len(cells)} ambiguous cells for column '{header_info['attribute']}' via LLM...")
                valid_cell_ids = classify_table_cells_batch_llm(cells, header_info['attribute'], header_info['unit'] or "none")
                
                for cell_id in cells:
                    cid = cell_id["id"]
                    if cid in valid_cell_ids:
                        obj = cell_objects[cid]
                        validated = validate_and_map_cell_v2(obj["parsed"], obj["header"], filename, page_num, obj["row_idx"], obj["col_idx"])
                        if validated:
                            # Rule: One value per attribute in a row. Prevent overwrite.
                            if validated['attribute'] not in obj["row_result_ref"]:
                                obj["row_result_ref"][validated['attribute']] = validated
            
            # Final Pass: Row Validation (Cross-Column Checker)
            # NOTE: validate_table_row_llm is only called when LLM is reachable.
            # If LLM fails (rate-limited), we accept the row to avoid losing valid data.
            result = []
            for r in results:
                 if len(r) > 1: # Has actual data
                      row_valid = validate_table_row_llm(r, USER_ATTRIBUTES, SCHEMA_LOADER)
                      if row_valid is None or row_valid:
                           result.append(r)
            
            # result is now structurally validated by LLM
    
    # Cache the result
    cache.set(cache_key, result)
    return result


def clean_text(text):
    text = text.replace("−", "-").replace("·", ".").replace("μ", "u").replace("µ", "u")
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_features_from_text(text, filename, page_num):
    features = {attr: "" for attr in USER_ATTRIBUTES}
    features["Proof"] = ""
    matches_found = 0
    all_candidates = []
    text = clean_text(text)
    
    # Stop processing if we hit References
    is_references_section = False
    if "references" in text.lower() or "bibliography" in text.lower():
        is_references_section = True

    for attr in USER_ATTRIBUTES:
        attr_lower = attr.lower()
        metadata = DYNAMIC_ALIASES.get(attr) or DYNAMIC_ALIASES.get(attr_lower) or {}
        
        aliases = set()
        expected_units = []
        if isinstance(metadata, dict):
            d_aliases = metadata.get("aliases", [])
            expected_units = [u.lower() for u in metadata.get("units", [])]
            for da in d_aliases:
                aliases.add(da.lower())
        else:
            for da in metadata:
                aliases.add(da.lower())
        
        # 1. Match Keyword + Number
        alias_list = [re.escape(attr.lower())] + [re.escape(a.lower()) for a in aliases]
        combined_pattern = f"(?:{'|'.join(alias_list)})"
        
        # Forward Match: Keyword ... [0-50 chars] ... Number
        forward_pattern = fr"(?i)({combined_pattern}).{{0,50}}?\b([-+]?\d+(?:\.\d+)?)(\b|(?=[a-zA-Z]))"
        # Backward Match: Number ... [0-50 chars] ... Keyword
        backward_pattern = fr"(?i)\b([-+]?\d+(?:\.\d+)?)(\b|(?=[a-zA-Z])).{{0,50}}?({combined_pattern})"

        temp_matches = []
        for m in re.finditer(forward_pattern, text.lower()):
            temp_matches.append((m.group(2), m.start()))
        for m in re.finditer(backward_pattern, text.lower()):
            temp_matches.append((m.group(1), m.start()))

        for val, match_pos in temp_matches:
            # Validation
            fp, reason = is_false_positive(text, val, attr, expected_units)
            
            score = 0
            if not fp:
                score += 200 # Heavy weight for validated unit match
            elif "Missing required unit" in reason:
                score += 5   # Very weak without unit
            else:
                continue     # Hard skip (figures/formulas/citations)

            # Section penalty
            if is_references_section:
                score -= 1000 # Aggressive penalty for references

            # Contextual Scoring
            context_window = text.lower()[max(0, match_pos-150):min(len(text), match_pos+150)]
            
            if attr.lower() in context_window:
                score += 100
            else:
                words = [w for w in attr.lower().split() if len(w) > 3]
                matched_count = sum(1 for w in words if w in context_window)
                score += matched_count * 20
            
            all_candidates.append({
                "attr": attr,
                "val": val,
                "score": score,
                "pos": match_pos
            })

    # Sort all candidates by score
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    consumed_numbers = set() # (val, pos)
    assigned_attrs = set()
    
    for cand in all_candidates:
        if cand["score"] <= 0: continue
        num_key = (cand["val"], cand["pos"])
        if num_key not in consumed_numbers and cand["attr"] not in assigned_attrs:
            features[cand["attr"]] = cand["val"]
            consumed_numbers.add(num_key)
            assigned_attrs.add(cand["attr"])
            matches_found += 1
            proof_entry = f"Page {page_num}, Table → {cand['attr']}"
            # Append if there are already proofs
            if features["Proof"]:
                features["Proof"] += ", " + proof_entry
            else:
                features["Proof"] = proof_entry
            print(f"[PROOF] {cand['attr']}={cand['val']} | Source: {filename} | {proof_entry}")

    return features if matches_found > 0 else None

def process_page_parallel(args):
    """Process a single page - designed for parallel execution"""
    doc, filename, page_num, workspace = args
    page = doc[page_num]
    
    try:
        # Get page text once
        page_text = page.get_text()
        
        # Find tables on this page
        tables = page.find_tables()
        
        page_rows = []
        found_on_page = False
        
        if tables:
            # Use new schema-aware extraction
            for table in tables:
                table_results = extract_table_data_v2(table, filename, page_num)
                
                if table_results:
                    for row_data in table_results:
                        # Convert to CSV row format
                        row_rec = {col: "" for col in COLUMNS}
                        
                        # Get metadata
                        metadata = row_data.get('_metadata', {})
                        proof_parts = []
                        
                        # Extract validated attributes
                        for attr, data in row_data.items():
                            if attr.startswith('_'):
                                continue
                            
                            if isinstance(data, dict):
                                row_rec[attr] = str(data.get('value', ''))
                                unit = data.get('unit', '')
                                conf = data.get('confidence', 0)
                                proof_parts.append(
                                    f"{attr}={data.get('value', '')}{unit}(conf:{conf:.2f})"
                                )
                        
                        row_rec["File"] = filename
                        row_rec["Proof"] = f"Page {page_num}, Table → {attr}"
                        
                        page_rows.append(row_rec)
                        found_on_page = True
                    
                    print(f"[V2 TABLE] Extracted {len(table_results)} rows from page {page_num}")
        
        return page_rows, found_on_page
        
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        return [], False


def main():
    start_time = time.time()
    
    # Load schema and validation once
    global SCHEMA_LOADER, UNIT_PARSER, VALIDATION_ENGINE, DYNAMIC_ALIASES, USER_ATTRIBUTES, COLUMNS
    
    # Load dynamic aliases if available
    aliases_path = os.path.join(BASE_DIR, "aliases.json")
    if os.path.exists(aliases_path):
        with open(aliases_path, 'r') as f:
            DYNAMIC_ALIASES = json.load(f)
    
    # Load schema ONLY if not already loaded from the workspace
    if SCHEMA_LOADER is None:
        schema_path = os.path.join(BASE_DIR, "schema.json")
        if os.path.exists(schema_path):
            SCHEMA_LOADER = SchemaLoader(schema_path)
    
    # Initialize other components safely
    if UNIT_PARSER is None:
        UNIT_PARSER = get_unit_parser()
    if VALIDATION_ENGINE is None:
        VALIDATION_ENGINE = get_validation_engine(SCHEMA_LOADER)
    
    all_rows = []
    found_any_in_file = False
    
    try:
        # 1. Load Analysis JSON to find table pages
        analysis_data = []
        if os.path.exists(ANALYSIS_JSON):
            with open(ANALYSIS_JSON, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        else:
            print(f"    [WARNING] Analysis file not found. Falling back to first 3 pages.")
            if os.path.exists(INCLUDED_DIR):
                for file in os.listdir(INCLUDED_DIR):
                    if file.lower().endswith(".pdf"):
                        analysis_data.append({"file": file, "tables_pages": [1, 2, 3]})

        if not analysis_data:
            print("No analysis data or PDFs found to process.")
            return

        # Initialize output CSV with headers
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            
        for item in analysis_data:
            filename = item.get("file")
            if not filename: continue
            
            filepath = os.path.join(INCLUDED_DIR, filename)
            if not os.path.exists(filepath): continue
            
            tables_pages = item.get("tables_pages", [])
            if not tables_pages: continue
            
            print(f"\n=== Processing {filename} ({len(tables_pages)} table pages) ===")
            
            try:
                doc = fitz.open(filepath)
                
                # Prepare arguments for parallel processing just for the identified table pages
                page_args = [(doc, filename, i-1, BASE_DIR) for i in tables_pages if 0 <= i-1 < len(doc)]
                if not page_args:
                    doc.close()
                    continue
                    
                current_file_rows = []
                
                # Process pages sequentially to avoid PyMuPDF threading deadlocks and LLM rate limits
                for args in page_args:
                    page_num = args[2]
                    try:
                        page_rows, found_on_page = process_page_parallel(args)
                        if page_rows:
                            all_rows.extend(page_rows)
                            current_file_rows.extend(page_rows)
                            found_any_in_file = True
                    except Exception as e:
                        print(f"Page {page_num} processing failed: {e}")
                
                doc.close()
                
                # Progressively append to CSV to safeguard against crashes
                if current_file_rows:
                    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=COLUMNS)
                        writer.writerows(current_file_rows)
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        
        # Final explicit sort/save of everything
        if all_rows:
            all_rows.sort(key=lambda x: (x.get('File', ''), x.get('Proof', '')))
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(all_rows)
        
        elapsed = time.time() - start_time
        print(f"\n=== Extraction Complete ===")
        print(f"Total rows extracted: {len(all_rows)}")
        print(f"Processing time: {elapsed:.2f} seconds")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extracting Data")
    parser.add_argument("--workspace", type=str, help="Workspace directory for this session")
    parser.add_argument("--attributes", type=str, help="Comma-separated attributes to extract")
    args = parser.parse_args()
    
    setup_paths(args.workspace, args.attributes)
    main()
