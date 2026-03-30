import os
import json
import re
import csv
import io
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from PIL import Image

try:
    import fitz  # PyMuPDF
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from llm_validator import call_openrouter_api, build_schema_context
    SCHEMA_LOADER = None # Set later
except ImportError:
    print("PyMuPDF or LLM Validator missing.")
    exit(1)

try:
    import torch
    import easyocr
    import numpy as np
    from plot_digitizer import GraphDigitizer
    HAS_GPU_OCR = True
except ImportError:
    HAS_GPU_OCR = False
    print("EasyOCR, Torch or PlotDigitizer not found. Falling back to heuristic text mode.")

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Defaults - will be overridden by --workspace)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INCLUDED_DIR = os.path.join(os.path.dirname(BASE_DIR), "Included")
ANALYSIS_JSON = os.path.join(BASE_DIR, "page_analysis_results.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "plot_extracted_data.csv")
PLOT_LOG_CSV = os.path.join(BASE_DIR, "plot_extraction_log.csv")

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def setup_paths(workspace=None):
    global BASE_DIR, INCLUDED_DIR, ANALYSIS_JSON, OUTPUT_CSV, PLOT_LOG_CSV, DIGITIZED_DIR
    if workspace:
        project_root = os.path.dirname(os.path.dirname(workspace))
        task_id = os.path.basename(workspace)
        BASE_DIR = os.path.join(workspace)
        INCLUDED_DIR = os.path.join(project_root, "Included", task_id)
        ANALYSIS_JSON = os.path.join(BASE_DIR, "page_analysis_results.json")
        OUTPUT_CSV = os.path.join(BASE_DIR, "plot_extracted_data.csv")
        PLOT_LOG_CSV = os.path.join(BASE_DIR, "plot_extraction_log.csv")
        DIGITIZED_DIR = os.path.join(BASE_DIR, "digitized_plots")
        os.makedirs(DIGITIZED_DIR, exist_ok=True)
        
        # Load dynamic aliases
        global DYNAMIC_ALIASES
        DYNAMIC_ALIASES = {}
        alias_file = os.path.join(BASE_DIR, "aliases.json")
        if os.path.exists(alias_file):
            try:
                with open(alias_file, 'r', encoding='utf-8') as f:
                    raw_aliases = json.load(f)
                    # Normalize keys to lowercase for robust matching
                    DYNAMIC_ALIASES = {k.lower(): v for k, v in raw_aliases.items()}
            except Exception as e:
                print(f"Error loading aliases.json: {e}")
                
        # Initialize schema loader context for LLM Graph Validation
        global SCHEMA_LOADER
        from schema_loader import SchemaLoader, load_domain_schema
        SCHEMA_LOADER = load_domain_schema(workspace)

# Single regex to find numbers (with optional units/bridging words)
# This will be used in conjunction with attribute names
NUMBER_RE = re.compile(r"([-+]?\d+(?:\.\d+)?(?:\s*[-–—]\s*\d+(?:\.\d+)?)?)\s*(?:[a-zA-Z/%^2\d\-]+)?", re.IGNORECASE)


def get_gpu_decoder():
    global _decoder_instance
    if '_decoder_instance' not in globals():
        _decoder_instance = GPUPlotDecoder()
    return _decoder_instance

class GPUPlotDecoder:
    """Thread-safe EasyOCR decoder (initializes once, reused across threads)."""
    def __init__(self):
        self.reader = None
        if HAS_GPU_OCR:
            try:
                # Attempt GPU first
                self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            except Exception as e:
                print(f"CUDA Error during OCR init: {e}. Falling back to CPU mode.")
                try:
                    self.reader = easyocr.Reader(['en'], gpu=False)
                except Exception as e2:
                    print(f"Failed to initialize EasyOCR on CPU: {e2}")
        else:
            self.reader = None

    def extract_text_from_image(self, pil_image):
        if not self.reader:
            return ""
        try:
            img_np = np.array(pil_image)
            results = self.reader.readtext(img_np)
            return "\n".join([res[1] for res in results])
        except Exception as e:
            if "CUDA out of memory" in str(e) or "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                print("CUDA OOM during inference. Clearing cache and retrying...")
                torch.cuda.empty_cache()
            return ""

    def extract_text_from_clip(self, pix_samples, width, height):
        """Directly use fitz pixmap samples (avoids PNG encode/decode roundtrip)."""
        if not self.reader:
            return ""
        try:
            img_np = np.frombuffer(pix_samples, dtype=np.uint8).reshape(height, width, 3)
            results = self.reader.readtext(img_np)
            return "\n".join([res[1] for res in results])
        except Exception as e:
            if "CUDA out of memory" in str(e) or "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                print("CUDA OOM during inference. Skipping GPU for this clip.")
                torch.cuda.empty_cache()
            return ""

_digitizer_instance = None
def get_digitizer():
    global _digitizer_instance
    if _digitizer_instance is None:
        from plot_digitizer import GraphDigitizer
        reader = get_gpu_decoder().reader
        _digitizer_instance = GraphDigitizer(reader=reader)
    return _digitizer_instance

def is_false_positive(line, match_str, attr, expected_units=None):
    """
    Check if a match is a likely false positive (formula constant, figure ref, year, citation, etc).
    """
    line_lower = line.lower()
    match_str_clean = match_str.strip('.,; ')
    
    # 1. Figure/Table/Reference Keyword Filtering
    ref_pattern = r'(?i)\b(?:fig(?:ure)?|table|ref|eq|step|scheme)\.?\s*\(?'+re.escape(match_str_clean)+r'\b'
    if re.search(ref_pattern, line):
        return True, "Figure/Table/Section Reference"

    # 2. Formula/Equation Detection
    if '=' in line:
        formula_vars = ['cs', 'csp', 'es', 'ps', 'td', 'v\'', 'v\"', '3600', '0.5', 'm0', 'ma']
        found_vars = [v for v in formula_vars if v in line_lower]
        if len(found_vars) >= 2 or '*' in line or '/' in line:
             return True, "Formula/Equation Constant"

    # 3. Year Detection (1900-2050)
    try:
        val_float = float(match_str_clean)
        if val_float.is_integer() and 1900 <= val_float <= 2050:
            # Metadata proximity check for years
            metadata_keywords = ["doi", "copyright", "published", "received", "accepted", "volume", "issue", "journal", "issn"]
            if any(kw in line_lower for kw in metadata_keywords):
                return True, "Publication Year/Metadata"
            # Hard rejection for certain attributes that rarely have values in year range
            if attr.lower() in ["specific capacitance", "current density", "pore size"]:
                 return True, "Year-like value for non-year attribute"
    except ValueError:
        pass

    # 4. Citation/Bracket Detection (e.g., [24], [12, 13], [1-5])
    # Check if match_str is inside brackets with other numbers or ranges
    bracket_pattern = r'\[[^\]]*?\b' + re.escape(match_str_clean) + r'\b[^\]]*?\]'
    if re.search(bracket_pattern, line):
        return True, "Citation/Bracketed Reference"

    # 5. Physical Sanity Checks (Attribute-Specific)
    positive_only_attrs = ["specific surface area", "pore volume", "nitrogen content", "specific capacitance", "current density", "pore size", "voltage"]
    if attr.lower() in positive_only_attrs:
        try:
            val = float(match_str_clean)
            if val < 0:
                return True, "Physically impossible negative value"
            # Voltage specific range check
            if attr.lower() == "voltage" and val > 10.0:
                 return True, "Voltage exceeds physical limits for supercapacitors (max 10V)"
        except ValueError:
            pass

    # 6. Metadata Proximity Blacklist
    metadata_proximity_keywords = ["doi:", "vol.", "no.", "pp.", "http"]
    for kw in metadata_proximity_keywords:
        if kw in line_lower:
            # Check distance to keyword
            kw_pos = line_lower.find(kw)
            val_pos = line_lower.find(match_str_clean.lower())
            if abs(kw_pos - val_pos) < 30:
                return True, "Journal Metadata Proximity"

    # 7. Unit-Aware Validation
    has_unit = False
    status_msg = "Missing required unit"
    valid_units = [u.lower().strip() for u in expected_units if u and u.strip()]
    
    if valid_units:
        # Check both after AND before the value for units
        p_after = re.escape(match_str_clean) + r'.{0,15}(?:' + '|'.join([re.escape(u) for u in valid_units]) + r')(?:\b|(?![a-z]))'
        p_before = r'(?:' + '|'.join([re.escape(u) for u in valid_units]) + r').{0,5}' + re.escape(match_str_clean)
        if re.search(p_after, line_lower) or re.search(p_before, line_lower):
            has_unit = True
        else:
            status_msg = f"Missing unit ({'/'.join(valid_units)})"
    else:
        # No units expected for this attribute
        has_unit = True

    if not has_unit:
        return True, status_msg

    return False, None

def extract_plot_data_from_text_heuristic(text, page_num, filename, attributes):
    """
    Enhanced heuristic that restricts matching to proximity clustering.
    Because OCR reads text boxes in unpredictable orders, we expand
    the context to include nearly all page text for dense papers.
    """
    all_candidates = []
    text = text.replace("−", "-").replace("·", ".").replace("—", "-").replace("μ", "u").replace("µ", "u").replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)

    # References detection
    is_references_section = False
    if "references" in text.lower() or "bibliography" in text.lower():
        is_references_section = True

    for attr in attributes:
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
        
        # Also create partial word patterns for multi-word attributes (fallback)
        attr_words = attr_lower.split()
        partial_patterns = []
        if len(attr_words) > 1:
            for word in attr_words:
                if len(word) > 3:
                    partial_patterns.append(re.escape(word))
        if partial_patterns:
            combined_pattern += f"|(?:{'|'.join(partial_patterns)})"
        
        # Forward Match: Keyword ... [0-100 chars] ... Number (wider for plots)
        forward_pattern = fr"(?i)({combined_pattern}).{{0,100}}?\b([-+]?\d+(?:\.\d+)?)(\b|(?=[a-zA-Z]))"
        # Backward Match: Number ... [0-100 chars] ... Keyword
        backward_pattern = fr"(?i)\b([-+]?\d+(?:\.\d+)?)(\b|(?=[a-zA-Z])).{{0,100}}?({combined_pattern})"

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
                score += 150 # Bias for plots
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
    
    consumed_numbers = set()
    assigned_attrs = set()
    matches_found = 0
    page_features = {attr: "" for attr in attributes}
    page_features["Proof"] = ""
    
    for cand in all_candidates:
        if cand["score"] <= 0: continue
        num_key = (cand["val"], cand["pos"])
        if num_key not in consumed_numbers and cand["attr"] not in assigned_attrs:
            page_features[cand["attr"]] = cand["val"]
            consumed_numbers.add(num_key)
            assigned_attrs.add(cand["attr"])
            matches_found += 1
            proof_entry = f"Page {page_num}, Plot → {cand['attr']}"
            if page_features["Proof"]:
                page_features["Proof"] += ", " + proof_entry
            else:
                page_features["Proof"] = proof_entry
            print(f"[BEST PROOF] {cand['attr']}={cand['val']} (score {cand['score']}) | Plot Context")

    row = {col: page_features.get(col, "") for col in attributes + ["Proof"]}
    row["File"] = filename
    return row, (1 if matches_found > 0 else 0)


def save_digitized_data_to_csv(dig_res, filename, page_num, img_idx, output_csv_path, attributes):
    """
    Save digitized (x,y) data points to a dedicated CSV file.
    Each point gets its own row: File, Page, Image, Series, X, Y, X_Unit, Y_Unit
    """
    import csv
    import os
    
    # Create digitized data CSV path
    base_path = output_csv_path.replace('.csv', '_digitized_points.csv')
    
    scale = dig_res.get("scale", {})
    y_title = scale.get("y_title", "")
    x_title = scale.get("x_title", "")
    y_unit = scale.get("y_unit", "")
    x_unit = scale.get("x_unit", "")
    
    series_list = dig_res.get("series", [])
    
    # Check if file exists to write header
    file_exists = os.path.exists(base_path)
    
    with open(base_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["File", "Page", "Image_ID", "Series_Name", "X_Value", "Y_Value", "X_Unit", "Y_Unit", "X_Title", "Y_Title"])
        
        for series in series_list:
            series_name = series.get("name", f"Series_{series_list.index(series)}")
            points = series.get("points", [])
            
            for point in points:
                if isinstance(point, dict) and "x" in point and "y" in point:
                    writer.writerow([
                        filename, page_num, img_idx, series_name,
                        point["x"], point["y"], x_unit, y_unit, x_title, y_title
                    ])
    
    return len(series_list)

def map_digitized_to_attributes(dig_res, attributes, filename, page_num, source_type="Digitized"):
    """
    Bridges GraphDigitizer output to research attributes.
    Extracts representative scalar values (e.g. Max Y) for the CSV.
    """
    page_features = {attr: "" for attr in attributes}
    page_features["Proof"] = ""
    matches_found = 0
    
    # Get axis info from digitizer result
    scale = dig_res.get("scale", {})
    y_title = scale.get("y_title", "").lower()
    x_title = scale.get("x_title", "").lower()
    
    # Get series data (the digitizer returns 'series', not 'datasets')
    series_list = dig_res.get("series", [])
    
    if not series_list:
        print(f"    [DEBUG] No series data in digitizer result for {source_type}")
        return page_features, 0

    print(f"    [DEBUG] Digitized Axis Titles - Y: '{y_title}', X: '{x_title}'")
    print(f"    [DEBUG] Series count: {len(series_list)}")

    # 1. Match Y-Axis to     matches = [] # list of (attr, source)
    
    # Normalize y_title: remove special chars, extra spaces
    norm_y = re.sub(r'[^a-z0-9\s]', ' ', y_title)
    
    matches = [] # list of (attr, source)
    
    def find_best_axis_match(norm_axis, raw_title=""):
        best_attr = None
        best_len = 0
        best_alias = ""
        
        # Helper: check for unit mismatch (Fix 3.2)
        # If the axis title has a unit (e.g. F/g) that belongs to a DIFFERENT attribute, 
        # we should avoid matching it to this attribute (e.g. SAR W/kg).
        axis_units = []
        if raw_title:
            # Extract anything that looks like a unit: chars after a number, or in parens
            unit_match = re.search(r'[\(\[]\s*([a-zA-Z0-9/µ·² \.\-\^]+)\s*[\)\]]', raw_title)
            if unit_match:
                axis_units.append(unit_match.group(1).strip().lower())
        
        for attr in attributes:
            attr_lower = attr.lower()
            metadata = DYNAMIC_ALIASES.get(attr_lower) or {}
            
            # Unit Check: If the title contains a unit from the schema, 
            # it MUST be a unit allowed for THIS attribute.
            schema_units = [u.lower() for u in metadata.get("units", [])]
            if axis_units and schema_units:
                # Normalization (e.g. F/g vs F g-1)
                norm_axis_units = [u.replace(" ", "").replace("·", "").replace("-1", "").replace("−1", "") for u in axis_units]
                norm_schema_units = [u.replace(" ", "").replace("·", "").replace("-1", "").replace("−1", "") for u in schema_units]
                
                # Check if the axis unit is definitely a known unit for DIFFERENT attributes
                # If so, and it doesn't match ours, we skip this match.
                is_wrong_unit = False
                for other_attr, other_meta in DYNAMIC_ALIASES.items():
                    if other_attr == attr_lower: continue
                    other_units = [u.lower().replace(" ", "").replace("·", "").replace("-1", "").replace("−1", "") for u in other_meta.get("units", [])]
                    if any(u in other_units for u in norm_axis_units) and not any(u in norm_schema_units for u in norm_axis_units):
                        is_wrong_unit = True
                        break
                if is_wrong_unit:
                    continue

            aliases = [attr_lower]
            if isinstance(metadata, dict):
                aliases.extend([a.lower() for a in metadata.get("aliases", [])])
                
            for a in aliases:
                is_match = False
                if len(a) <= 3:
                    if re.search(fr'\b{re.escape(a)}\b', norm_axis):
                        is_match = True
                else:
                    if a in norm_axis:
                        is_match = True
                
                alias_words = a.split()
                if len(alias_words) > 1 and not is_match:
                    significant_words = [w for w in alias_words if len(w) > 3]
                    if significant_words and all(word in norm_axis for word in significant_words):
                        is_match = True
                        
                if is_match:
                    match_len = len(a)
                    if match_len > best_len:
                        best_len = match_len
                        best_attr = attr
                        best_alias = a
                        
        return best_attr, best_alias

    # 1. Match Y-axis to attributes independently
    if y_title:
        y_attr, y_alias = find_best_axis_match(norm_y.lower(), y_title)
        if y_attr:
            matches.append((y_attr, "y"))
            print(f"    [MATCH] Y-axis attribute '{y_attr}' matched via alias '{y_alias}' in '{norm_y[:50]}'")

    # 2. Match X-axis to attributes independently
    if x_title:
        norm_x = re.sub(r'[^a-z0-9\s]', ' ', x_title)
        x_attr, x_alias = find_best_axis_match(norm_x.lower(), x_title)
        if x_attr:
            matches.append((x_attr, "x"))
            print(f"    [MATCH] X-axis attribute '{x_attr}' matched via alias '{x_alias}' in '{norm_x[:50]}'")

    # 3. If no axis match, try series names (legends)
    for idx, series in enumerate(series_list):
        s_name = series.get("name", "").lower()
        if not s_name or s_name.startswith("series_"): continue
        norm_s = re.sub(r'[^a-z0-9\s]', ' ', s_name)
        
        s_attr, s_alias = find_best_axis_match(norm_s.lower(), s_name)
        
        # Only try to match if this attribute hasn't been matched by an axis already
        if s_attr and not any(m[0] == s_attr for m in matches):
            matches.append((s_attr, idx))
            print(f"    [MATCH] Series name '{s_name}' matched attribute '{s_attr}' via alias '{s_alias}'")
                    
    # --- OFFLINE API BLOCKED FALLBACK ---
    if not matches and series_list:
        num_series = len(series_list)
        num_points = sum(len(s.get("points", [])) for s in series_list)
        avg_points = num_points / num_series if num_series else 0

        # Plots with many series (>10) are almost certainly BET isotherms,
        # pore distribution charts, or multi-sample surface area plots.
        # We cannot reliably classify these offline — skip them to avoid false attribution.
        if num_series > 10:
            print(f"    [SKIP] Complex multi-series plot ({num_series} series, avg {avg_points:.1f} pts). Cannot classify offline — skipping.")
        else:
            # Emergency Default: fallback to the user's primary requested attributes dynamically
            fallback_x_attr = attributes[0] if len(attributes) > 0 else "Unknown_X"
            fallback_y_attr = attributes[1] if len(attributes) > 1 else (attributes[0] if len(attributes) > 0 else "Unknown_Y")
            
            print(f"    [WARNING] Plot classification ambiguous ({num_series} series, avg {avg_points:.1f} pts). Emergency fallback to primary requested attributes: {fallback_y_attr} vs {fallback_x_attr}")
            matches.append((fallback_y_attr, "y"))
            matches.append((fallback_x_attr, "x"))

    # --- AXIS COLLISION FIX ---
    x_matches = [m for m in matches if m[1] == "x"]
    y_matches = [m for m in matches if m[1] == "y"]
    if x_matches and y_matches and x_matches[0][0] == y_matches[0][0]:
        print(f"    [WARNING] Collision detected: both axes OCR'd as '{x_matches[0][0]}'. Applying heuristic override.")
        if x_matches[0][0] == "current density":
            # Assume Y axis is specific capacitance in these cases
            matches = [m for m in matches if m[1] != "y"]
            matches.append(("specific capacitance", "y"))
        else:
            # Fallback for other identical axes: drop the one we are less confident in.
            matches = [m for m in matches if m[1] != "y"]
            matches.append(("Unknown", "y"))

    matches_found = len(matches)
    if matches:
        # 4. Extract representative value based on matched source
        for match in matches:
            matched_attr, matched_source = match
            all_vals = []
            target_series = series_list
            if isinstance(matched_source, int):
                target_series = [series_list[matched_source]]
                
            # Determine whether to extract X or Y
            extract_x = False
            if matched_source == "x":
                extract_x = True
            elif isinstance(matched_source, int):
                if any(keyword in matched_attr.lower() for keyword in ['cycle', 'time', 'rate', 'current', 'voltage']):
                    extract_x = True
                    
            for series in target_series:
                points = series.get("points", [])
                for point in points:
                    if isinstance(point, dict):
                        if extract_x and isinstance(point.get("x"), (int, float)):
                            all_vals.append(point["x"])
                        elif not extract_x and isinstance(point.get("y"), (int, float)):
                            all_vals.append(point["y"])
                            
            if all_vals:
                valid_vals = [v for v in all_vals if -1000 < v < 100000]
                if valid_vals:
                    max_val = max(valid_vals)
                    min_val = min(valid_vals)
                    
                    val_str = str(round(max_val, 2))
                    if abs(max_val - min_val) > abs(max_val) * 0.3 and abs(max_val - min_val) > 1:
                        val_str = f"{round(min_val, 1)}-{round(max_val, 1)}"
                        
                    match_ctx = y_title[:30] if matched_source == "y" else (x_title[:30] if matched_source == "x" else f"Series {matched_source}")
                    # Removed aggregation scalar mapping per user request
                    # page_features[matched_attr] = val_str
                    
                    actual_x_attr = next((m[0] for m in matches if m[1] == "x"), "Unknown")
                    if matched_source == "y" or isinstance(matched_source, int):
                        actual_y_attr = matched_attr
                    else:
                        actual_y_attr = next((m[0] for m in matches if m[1] == "y"), "Unknown")

                    import json
                    series_json = {"x_attr": actual_x_attr, "y_attr": actual_y_attr, "x": [], "y": []}
                    for series in target_series:
                        points = series.get("points", [])
                        for point in points:
                            if isinstance(point, dict) and isinstance(point.get("x"), (int, float)) and isinstance(point.get("y"), (int, float)):
                                series_json["x"].append(point["x"])
                                series_json["y"].append(point["y"])
                    
                    page_features[f"{matched_attr} (series)"] = json.dumps(series_json)
                    
                    proof_str = f"Page {page_num}, Plot → {matched_attr}"
                    if page_features.get("Proof"):
                        page_features["Proof"] += f" | {proof_str}"
                    else:
                        page_features["Proof"] = proof_str
                        
                    print(f"  [BEST DIGITIZED] {matched_attr}={val_str} (from {len(valid_vals)} points)")
        
        # Include dynamically generated series columns in the row output
        keys_to_export = attributes + [f"{a} (series)" for a in attributes] + ["Proof"]
        row = {col: page_features.get(col, "") for col in keys_to_export}
        row["File"] = filename
        return row, matches_found > 0
    else:
        # No matches found
        return {}, False


def process_pdf_item(item, attributes):
    """
    Process a single PDF item. Returns (rows, log_entries).
    """
    filename = item["file"]
    pdf_path = os.path.join(INCLUDED_DIR, filename)
    rows = []
    log_entries = []

    if not os.path.exists(pdf_path):
        return rows, log_entries

    plots_pages = item.get("plots_pages", [])
    if not plots_pages:
        return rows, log_entries

    try:
        with fitz.open(pdf_path) as doc:
            for page_num in plots_pages:
                if not (0 <= page_num - 1 < len(doc)):
                    continue

                page = doc[page_num - 1]
                vector_text = page.get_text()

                ocr_text = ""
                if HAS_GPU_OCR:
                    image_list = page.get_images(full=False)
                    decoder = get_gpu_decoder()
                    if image_list:
                        clip_texts = []
                        for img_info in image_list:
                            xref = img_info[0]
                            rects = page.get_image_rects(xref)
                            for rect in rects:
                                clip_pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, colorspace=fitz.csRGB)
                                clip_texts.append(decoder.extract_text_from_clip(clip_pix.samples, clip_pix.width, clip_pix.height))
                        ocr_text = "\n".join(clip_texts)
                    else:
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), colorspace=fitz.csRGB)
                        ocr_text = decoder.extract_text_from_clip(pix.samples, pix.width, pix.height)
                combined_text = vector_text + "\n" + ocr_text
                
                # Modularity: Digitization & Classification
                page_has_valid_graph = False
                captured_digitized_rows = []
                
                if HAS_GPU_OCR:
                    digitizer = get_digitizer()
                    
                    # A. Process Raster Images
                    if image_list:
                        for idx, img_info in enumerate(image_list):
                            xref = img_info[0]
                            rects = page.get_image_rects(xref)
                            for r_idx, rect in enumerate(rects):
                                try:
                                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, colorspace=fitz.csRGB)
                                    
                                    # STRICT image validation before any processing
                                    if not pix or not pix.samples:
                                        print(f"  [SKIP] Page {page_num} Image {idx}_{r_idx}: No pixmap data")
                                        continue
                                    if pix.width == 0 or pix.height == 0:
                                        print(f"  [SKIP] Page {page_num} Image {idx}_{r_idx}: Zero dimensions")
                                        continue
                                    if len(pix.samples) < pix.width * pix.height * 3:
                                        print(f"  [SKIP] Page {page_num} Image {idx}_{r_idx}: Insufficient sample data")
                                        continue
                                    
                                    try:
                                        n = pix.n  # actual number of channels from pixmap
                                        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, n)
                                        # Normalize to 3-channel RGB
                                        import cv2
                                        if n == 1:
                                            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                                        elif n == 2:
                                            img_np = cv2.cvtColor(img_np[:,:,0], cv2.COLOR_GRAY2RGB)
                                        elif n == 4:
                                            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                                        # n==3 is already RGB, no conversion needed
                                    except Exception as e:
                                        print(f"  [SKIP] Page {page_num} Image {idx}_{r_idx}: Reshape failed - {e}")
                                        continue
                                    
                                    if img_np is None or img_np.size == 0:
                                        print(f"  [SKIP] Page {page_num} Image {idx}_{r_idx}: Empty numpy array")
                                        continue
                                    if img_np.shape[0] < 50 or img_np.shape[1] < 50:
                                        print(f"  [SKIP] Page {page_num} Image {idx}_{r_idx}: Too small ({img_np.shape})")
                                        continue
                                    
                                    # CLASSIFY FIRST
                                    classification = digitizer.classify_figure(img_np)
                                    if classification["action"] == "skip_extraction":
                                        print(f"  [SKIP] Page {page_num} Image {idx}_{r_idx}: {classification['figure_type']} (conf: {classification['confidence']})")
                                        continue
                                    
                                    page_has_valid_graph = True
                                    print(f"Digitizing Plot {idx}_{r_idx} on Page {page_num}...")
                                    dig_res = digitizer.digitize(img_np)
                                    
                                    # Handle both single and multi-panel results
                                    panels_to_process = []
                                    if "multi_panel" in dig_res:
                                        panels_to_process = dig_res["multi_panel"]
                                    elif "series" in dig_res and dig_res["series"]:
                                        panels_to_process = [dig_res]
                                        
                                    for p_i, panel_res in enumerate(panels_to_process):
                                        if "series" in panel_res and panel_res["series"]:
                                            dig_id = f"{filename}_p{page_num}_f{idx}_{r_idx}_sub{p_i}"
                                            json_path = os.path.join(BASE_DIR, "digitized_plots", f"{dig_id}.json")
                                            with open(json_path, 'w', encoding='utf-8') as jf:
                                                json.dump(panel_res, jf, indent=2, cls=NumpyEncoder)
                                            
                                            # Save detailed (x,y) data points to CSV
                                            save_digitized_data_to_csv(panel_res, filename, page_num, dig_id, OUTPUT_CSV, attributes)
                                            
                                            dig_row, found_dig = map_digitized_to_attributes(panel_res, attributes, filename, page_num, "Figure")
                                            if found_dig:
                                                captured_digitized_rows.append(dig_row)
                                except Exception as e:
                                    print(f"Error digitizing image {idx} on page {page_num}: {e}")

                    # B. Process Vector Drawings (Clusters)
                    else: # Only try vector drawings if no raster images were found
                        print(f"No images found on Page {page_num}, checking for vector drawings...")
                        drawings = page.get_drawings()
                        if drawings:
                            bboxes = []
                            for d in drawings:
                                if "rect" in d: bboxes.append(d["rect"])
                            
                            if bboxes:
                                # Group bounding boxes by proximity (Spatial Clustering)
                                clusters = []
                                threshold = 100 # Distance threshold for clustering
                                for b in bboxes:
                                    b_rect = fitz.Rect(b)
                                    # Manual inflation for intersection check
                                    inflated_b = fitz.Rect(b_rect.x0 - threshold, b_rect.y0 - threshold, b_rect.x1 + threshold, b_rect.y1 + threshold)
                                    found_cluster = False
                                    for c_idx, cluster_rect in enumerate(clusters):
                                        if inflated_b.intersects(cluster_rect):
                                            clusters[c_idx] = cluster_rect | b_rect
                                            found_cluster = True
                                            break
                                    if not found_cluster:
                                        clusters.append(b_rect)
                                
                                # Merge overlapping clusters recursively
                                i = 0
                                while i < len(clusters):
                                    j = i + 1
                                    while j < len(clusters):
                                        inflated_i = fitz.Rect(clusters[i].x0 - threshold, clusters[i].y0 - threshold, clusters[i].x1 + threshold, clusters[i].y1 + threshold)
                                        if inflated_i.intersects(clusters[j]):
                                            clusters[i] |= clusters[j]
                                            clusters.pop(j)
                                        else:
                                            j += 1
                                    i += 1

                                print(f"  [DEBUG] Found {len(clusters)} vector clusters on Page {page_num}")
                                for c_idx, plot_area_raw in enumerate(clusters):
                                    # Expand margin to include axis labels (text)
                                    plot_area = fitz.Rect(max(0, plot_area_raw.x0-80), max(0, plot_area_raw.y0-80), 
                                                         min(page.rect.width, plot_area_raw.x1+80), min(page.rect.height, plot_area_raw.y1+80))
                                    
                                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=plot_area, colorspace=fitz.csRGB)
                                    
                                    # STRICT image validation
                                    if not pix or not pix.samples:
                                        print(f"  [SKIP] Page {page_num} Vector Cluster {c_idx}: No pixmap data")
                                        continue
                                    if pix.width == 0 or pix.height == 0:
                                        print(f"  [SKIP] Page {page_num} Vector Cluster {c_idx}: Zero dimensions")
                                        continue
                                    if len(pix.samples) < pix.width * pix.height * 3:
                                        print(f"  [SKIP] Page {page_num} Vector Cluster {c_idx}: Insufficient sample data")
                                        continue
                                    
                                    try:
                                        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                                    except Exception as e:
                                        print(f"  [SKIP] Page {page_num} Vector Cluster {c_idx}: Reshape failed - {e}")
                                        continue
                                    
                                    if img_np is None or img_np.size == 0:
                                        print(f"  [SKIP] Page {page_num} Vector Cluster {c_idx}: Empty numpy array")
                                        continue
                                    if img_np.shape[0] < 50 or img_np.shape[1] < 50:
                                        print(f"  [SKIP] Page {page_num} Vector Cluster {c_idx}: Too small ({img_np.shape})")
                                        continue
                                    
                                    # CLASSIFY
                                    classification = digitizer.classify_figure(img_np)
                                    if classification["action"] == "skip_extraction":
                                        print(f"  [SKIP] Page {page_num} Vector Cluster {c_idx}: {classification['figure_type']}")
                                        continue
                                    
                                    page_has_valid_graph = True
                                    print(f"  Digitizing Vector Cluster {c_idx} on Page {page_num}: {plot_area}")
                                    dig_res = digitizer.digitize(img_np)
                                    
                                    if "series" in dig_res and dig_res["series"]:
                                        dig_id = f"{filename}_p{page_num}_v{c_idx}"
                                        json_path = os.path.join(BASE_DIR, "digitized_plots", f"{dig_id}.json")
                                        with open(json_path, 'w', encoding='utf-8') as jf:
                                            json.dump(dig_res, jf, indent=2, cls=NumpyEncoder)
                                        print(f"Vector Digitized Data Saved: {os.path.basename(json_path)}")
                                        
                                        # Save detailed (x,y) data points to CSV
                                        save_digitized_data_to_csv(dig_res, filename, page_num, dig_id, OUTPUT_CSV, attributes)
                                        
                                        # NEW: Map to CSV row
                                        dig_row, found_dig = map_digitized_to_attributes(dig_res, attributes, filename, page_num, "Vector")
                                        if found_dig:
                                            captured_digitized_rows.append(dig_row)
                                    else:
                                        err = dig_res.get('error', 'Unknown error')
                                        print(f"    [DEBUG] No datasets in vector cluster {c_idx}: {err}")
                            else:
                                print(f"    [DEBUG] No bounding boxes found for vector drawings on Page {page_num}")
                        else:
                            print(f"    [DEBUG] No vector drawings found on Page {page_num}")

                    # C. Full Page Last Resort
                    if not page_has_valid_graph:
                         print(f"  [DEBUG] No images/drawings found or classified as graphs, trying full page as last resort...")
                         pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
                         
                         # STRICT image validation
                         if not pix or not pix.samples:
                             print(f"  [SKIP] Page {page_num}: No pixmap data")
                         elif pix.width == 0 or pix.height == 0:
                             print(f"  [SKIP] Page {page_num}: Zero dimensions")
                         elif len(pix.samples) < pix.width * pix.height * 3:
                             print(f"  [SKIP] Page {page_num}: Insufficient sample data")
                         else:
                             try:
                                 img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                             except Exception as e:
                                 print(f"  [SKIP] Page {page_num}: Reshape failed - {e}")
                                 img_np = None
                             
                             if img_np is None or img_np.size == 0:
                                 print(f"  [SKIP] Page {page_num}: Empty numpy array")
                             elif img_np.shape[0] < 50 or img_np.shape[1] < 50:
                                 print(f"  [SKIP] Page {page_num}: Too small ({img_np.shape})")
                             else:
                                 classification = digitizer.classify_figure(img_np)
                                 if classification["action"] == "extract_plot_data":
                                     page_has_valid_graph = True
                                     print(f"  Digitizing Full Page {page_num} (classified as {classification['figure_type']})...")
                                     dig_res = digitizer.digitize(img_np)
                                     if "series" in dig_res and dig_res["series"]:
                                         dig_id = f"{filename}_p{page_num}_full"
                                         json_path = os.path.join(BASE_DIR, "digitized_plots", f"{dig_id}.json")
                                         with open(json_path, 'w', encoding='utf-8') as jf:
                                             json.dump(dig_res, jf, indent=2, cls=NumpyEncoder)
                                         
                                         # Save detailed (x,y) data points to CSV
                                         save_digitized_data_to_csv(dig_res, filename, page_num, dig_id, OUTPUT_CSV, attributes)
                                         
                                         dig_row, found_dig = map_digitized_to_attributes(dig_res, attributes, filename, page_num, "FullPage")
                                         if found_dig:
                                             captured_digitized_rows.append(dig_row)
                                     else:
                                         err = dig_res.get('error', 'Unknown error')
                                         print(f"    [DEBUG] No datasets found in full page digitization: {err}")
                                 else:
                                     print(f"  [SKIP] Full Page {page_num} classified as {classification['figure_type']} (conf: {classification['confidence']}), skipping.")
                
                # D. PRIORITIZE: Use Digitized Data FIRST, only fall back to text if digitization failed
                # This ensures we get actual (x,y) data points from graph images, not text mentions
                if captured_digitized_rows:
                    # We have actual digitized plot data - use this exclusively
                    print(f"  [DIGITIZED DATA] Page {page_num}: {len(captured_digitized_rows)} digitized series extracted")
                    rows.extend(captured_digitized_rows)
                elif page_has_valid_graph:
                    # Digitization ran but didn't find matching attributes
                    # Still don't use text heuristic - the axis titles didn't match our attributes
                    print(f"  [INFO] Page {page_num}: Graph detected but no matching attributes in axis titles")
                else:
                    # No valid graph detected - optionally use text heuristic as last resort
                    # but only for pages that are definitely not plots
                    print(f"  [SUPPRESS] Page {page_num}: No valid graph detected, skipping text extraction")

    except Exception as e:
        log_entries.append([filename, "ERROR", f"Failed to process PDF: {e}"])
        traceback.print_exc() # Added traceback for better debugging

    return rows, log_entries


def main(attributes_str):
    if not os.path.exists(ANALYSIS_JSON):
        print(f"Analysis file not found: {ANALYSIS_JSON}")
        return

    if SCHEMA_LOADER and hasattr(SCHEMA_LOADER, "schema") and SCHEMA_LOADER.schema:
        attributes = list(SCHEMA_LOADER.schema.keys())
    else:
        attributes = [a.strip() for a in attributes_str.split(",") if a.strip()]

    series_cols = [f"{a} (series)" for a in attributes]
    columns = attributes + series_cols + ["Proof", "File"]

    with open(ANALYSIS_JSON, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)

    if not analysis_data:
        print("No analysis data available. Skipping plot extraction.")
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
        return

    all_rows = []
    log_entries = [["File", "Severity", "Message"]]

    items_with_plots = [item for item in analysis_data if item.get("plots_pages")]
    total = len(items_with_plots)
    # print(f"Starting GPU-Accelerated OCR on {total} PDFs with plot pages...")
    start = time.time()
    
    max_workers = 3
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf_item, item, attributes): item["file"] for item in items_with_plots}
        for future in as_completed(futures):
            fname = futures[future]
            completed += 1
            try:
                rows, item_logs = future.result()
                all_rows.extend(rows)
                log_entries.extend(item_logs)
                elapsed = time.time() - start
                avg = elapsed / completed if completed > 0 else 0
                eta = avg * (total - completed)
                print(f"[{completed}/{total}] {fname} — {len(rows)} datapoints")
            except Exception as e:
                log_entries.append([fname, "ERROR", str(e)])

    # De-duplicate rows
    unique_rows = []
    seen = set()
    for r in all_rows:
        key = tuple(str(r.get(col, "")) for col in columns)
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    # Save to CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in unique_rows:
            writer.writerow(row)

    with open(PLOT_LOG_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(log_entries)

    elapsed = time.time() - start
    print(f"\nDigitization complete. Extracted {len(unique_rows)} datapoints.")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Processing Plots")
    parser.add_argument("--workspace", type=str, help="Workspace directory for this session")
    parser.add_argument("--attributes", type=str, default="", help="Comma-separated attributes to extract")
    args = parser.parse_args()

    setup_paths(args.workspace)
    main(args.attributes)
