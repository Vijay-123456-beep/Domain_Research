import pandas as pd
import os
import re
import math
import time
from concurrent.futures import ThreadPoolExecutor
from schema_loader import load_domain_schema

# ---------------------------------------------------------
# CONFIGURATION (Defaults - will be overridden by setup_paths)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_CSV  = os.path.join(BASE_DIR, "table_extracted_data.csv")
PLOT_CSV   = os.path.join(BASE_DIR, "plot_extracted_data.csv")
TEXT_CSV   = os.path.join(BASE_DIR, "text_extracted_data.csv")
OUTPUT_MASTER = os.path.join(BASE_DIR, "resultant_dataset.csv")
OUTPUT_PLOT_DATASET = os.path.join(BASE_DIR, "plot_dataset.csv")
COLUMNS = []

# Source priority: lower value = higher trust
SOURCE_PRIORITY = {"table": 1, "plot": 2, "text": 3}

# ---------------------------------------------------------
# MODULE-LEVEL SCHEMA LOADING
# Build per-attribute unit sets and optional physical ranges
# ---------------------------------------------------------
_ATTR_UNIT_MAP   = {}   # attr_name → set of valid canonical units
_ATTR_RANGE_MAP  = {}   # attr_name → (min, max) or None
_ALL_VALID_UNITS = set()

try:
    _schema_sys = load_domain_schema()
    if _schema_sys and hasattr(_schema_sys, 'schema'):
        for attr_name, attr_data in _schema_sys.schema.items():
            units = set(u.strip().lower() for u in attr_data.get("units", []))
            _ATTR_UNIT_MAP[attr_name.lower()] = units
            # Also map space/underscore variants so column names can be looser
            _ATTR_UNIT_MAP[attr_name.lower().replace("_", " ")] = units
            _ALL_VALID_UNITS.update(units)
            # Aliases also map back to this attribute's unit set
            for alias in attr_data.get("aliases", []):
                _ATTR_UNIT_MAP[alias.strip().lower()] = units
            # Optional physical range: schema may include {"range": [min, max]}
            rng = attr_data.get("range")
            if rng and len(rng) == 2:
                try:
                    _ATTR_RANGE_MAP[attr_name.lower()] = (float(rng[0]), float(rng[1]))
                except (ValueError, TypeError):
                    pass
except Exception as e:
    print(f"[Warning] Could not load schema for merge validation: {e}")

# ---- Unit Normalization Table (Fix 3.1) ----
# Maps any written variant → canonical schema form
_UNIT_NORMALIZE = {
    # F/g variants
    "f g-1":    "f/g",    "f·g-1":   "f/g",    "f g−1":  "f/g",
    "fg-1":     "f/g",    "f per g": "f/g",    "f/g":    "f/g",
    # mF variants
    "mf/g":     "mf/g",   "mf g-1":  "mf/g",
    # mF/cm² variants
    "mf/cm2":   "mf/cm²", "mf cm-2": "mf/cm²", "mfcm-2": "mf/cm²",
    # F/cm³ variants
    "f/cm3":    "f/cm³",  "f cm-3":  "f/cm³",
    # Wh/kg variants
    "wh kg-1":  "wh/kg",  "wh·kg-1": "wh/kg",  "wh kg−1": "wh/kg",
    "wh/kg":    "wh/kg",
    # W/kg variants
    "w kg-1":   "w/kg",   "w·kg-1":  "w/kg",   "w kg−1":  "w/kg",
    "w/kg":     "w/kg",
    # A/g variants
    "a g-1":    "a/g",    "a·g-1":   "a/g",    "a g−1":   "a/g",
    "a/g":      "a/g",
    # mAh/g variants
    "mah g-1":  "mah/g",  "mah·g-1": "mah/g",  "mah/g":  "mah/g",
    # m²/g variants
    "m2/g":     "m²/g",   "m2 g-1":  "m²/g",   "m²g-1":  "m²/g",
    # Ω variants
    "ohm":      "ω",      "ohm.cm":  "ω·cm",
}

def normalize_unit(raw_unit: str) -> str:
    """Return the canonical lowercase unit string, or the input lowercased if not found."""
    u = raw_unit.strip().lower()
    return _UNIT_NORMALIZE.get(u, u)

def normalize_attr_key(col_name: str) -> str:
    """Normalize an attribute column name to match schema keys (lowercased, underscores/spaces unified)."""
    return col_name.strip().lower().replace(" ", "_").replace("-", "_")

# Known junk patterns to reject regardless of content
_DESCRIPTION_PATTERNS = [
    re.compile(r'\b(?:synthesis|preparation|fabrication|method|process|procedure)\b', re.IGNORECASE),
    re.compile(r'\b(?:carbonization|pyrolysis|annealing|hydrothermal|sol-gel)\b', re.IGNORECASE),
    re.compile(r'\b(?:material|sample|specimen|composite|substrate)\b', re.IGNORECASE),
    re.compile(r'\b(?:growth|treatment|reduction|dispersed|washed|prepared)\b', re.IGNORECASE),
]

# Match a number followed by a unit-like token (preserving unit)
_VALUE_UNIT_RE = re.compile(
    r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s*±\s*\d+(?:\.\d+)?)?)\s*'
    r'([a-zA-Z%/µ·²]+(?:[·/][a-zA-Z²]+)*)'
)
# Matches bare numbers with NO following unit characters
_BARE_NUM_RE = re.compile(r'^\s*\d+(?:\.\d+)?\s*$')
# Matches ranges of bare numbers
_RANGE_BARE_NUM_RE = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$')
# Year-like numbers
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')


# ---------------------------------------------------------
# CELL VALIDATION (Fix 3.1, 3.4 — preserve + validate units)
# ---------------------------------------------------------
def validate_cell(value_str, attr_col):
    """
    Returns (cleaned_value, is_valid) for a raw cell string.
    - Extracts ALL value+unit pairs from the cell (not just the first)
    - Picks the pair whose unit best matches the attribute's schema
    - Normalizes unit variants before comparing
    - Preserves the full 'value + canonical_unit' string
    """
    if not value_str or str(value_str).strip() in ('', 'nan', 'None'):
        return '', False

    s = str(value_str).strip()

    # Reject pure descriptions (too long, no numeric signature)
    if len(s) > 60 and not _VALUE_UNIT_RE.search(s):
        return '', False

    if any(p.search(s) for p in _DESCRIPTION_PATTERNS):
        return '', False

    if _YEAR_RE.fullmatch(s.strip()):
        return '', False

    # Normalize attribute key for schema lookup (Fix 3.3)
    attr_key = normalize_attr_key(attr_col)
    # Also try underscore → space variant
    valid_units = _ATTR_UNIT_MAP.get(attr_key) or _ATTR_UNIT_MAP.get(attr_key.replace("_", " "))

    # Extract ALL (value, unit) pairs from the cell (Fix 3.2)
    all_matches = _VALUE_UNIT_RE.findall(s)
    if not all_matches:
        # Support bare numbers natively. Tables/Plots often have units in headers/axes.
        if _BARE_NUM_RE.match(s):
            # Check range if defined
            attr_base = attr_key
            if attr_base in _ATTR_RANGE_MAP:
                rmin, rmax = _ATTR_RANGE_MAP[attr_base]
                try:
                    num_val = float(s)
                    if not (rmin <= num_val <= rmax):
                        return '', False # Outside physical range
                except ValueError:
                    return '', False

            default_unit = list(valid_units)[0] if valid_units else ""
            return f"{s} {default_unit}".strip(), True
            
        range_match = _RANGE_BARE_NUM_RE.match(s)
        if range_match:
            attr_base = attr_key
            if attr_base in _ATTR_RANGE_MAP:
                rmin, rmax = _ATTR_RANGE_MAP[attr_base]
                try:
                    num_val_1 = float(range_match.group(1))
                    num_val_2 = float(range_match.group(2))
                    # both ends must be physically plausible
                    if not (rmin <= num_val_1 <= rmax) or not (rmin <= num_val_2 <= rmax):
                        return '', False
                except ValueError:
                    return '', False
                    
            default_unit = list(valid_units)[0] if valid_units else ""
            return f"{s} {default_unit}".strip(), True

        return '', False

    best_value = None
    best_unit = None

    for number_part, raw_unit in all_matches:
        unit_norm = normalize_unit(raw_unit)   # canonical form

        # Schema unit check (only if we know this attribute)
        if valid_units:
            if unit_norm not in valid_units:
                continue  # This pair's unit doesn't match → try next pair

        # Outlier range check
        attr_base = attr_key
        if attr_base in _ATTR_RANGE_MAP:
            rmin, rmax = _ATTR_RANGE_MAP[attr_base]
            try:
                num_val = float(re.sub(r'[^\d.\-eE]', '', number_part.split('±')[0]))
                if not (rmin <= num_val <= rmax):
                    continue  # Outside physical range
            except (ValueError, TypeError):
                pass

        # First matching pair wins (list is in order of appearance)
        best_value = number_part.strip()
        best_unit = unit_norm
        break

    if best_value is None:
        return '', False

    return f"{best_value} {best_unit}", True


# ---------------------------------------------------------
# PER-FILE CONSOLIDATION (Fix 3.3 — one row per paper)
# ---------------------------------------------------------
# Default confidence per source when no confidence column is provided (Fix 3.4)
_SOURCE_DEFAULT_CONFIDENCE = {
    "table": 0.9,
    "plot":  0.7,
    "text":  0.6,
}

# VERITAS MODE: Zero-tolerance for error (Fix 3.4)
VERITAS_MODE = True
VERITAS_THRESHOLD = 0.95


def consolidate_file_group(group_df, attr_cols):
    """
    Given all rows for a single file from all sources, produce ONE consolidated row.

    Resolution order per attribute column:
      1. Best source priority (table=1 > plot=2 > text=3)
      2. If tied: highest confidence score (if column exists)
      3. Collect all Proof entries with '|' separator
    """
    result = {"File": group_df["File"].iloc[0]}
    all_proofs = []
    source_summary_parts = []

    # Gather proofs from all source rows
    if "Proof" in group_df.columns:
        proof_vals = group_df["Proof"].dropna().astype(str)
        all_proofs = [p for p in proof_vals if p not in ('', 'nan')]

    for col in attr_cols:
        series_col = f"{col} (series)"
        if col not in group_df.columns:
            result[col] = ''
            result[series_col] = ''
            continue

        # Collect all valid candidates for this attribute, tagged with priority
        candidates = []
        for _, row in group_df.iterrows():
            raw = row.get(col, '')
            cleaned, valid = validate_cell(raw, col)
            if valid:
                priority = SOURCE_PRIORITY.get(str(row.get("_source", "text")).lower(), 3)
                # Use explicit confidence if available, else fall back to source default (Fix 3.4)
                raw_conf = row.get("confidence")
                if pd.isna(raw_conf) or raw_conf is None or str(raw_conf).strip() in ('', 'nan'):
                    source_tag = str(row.get("_source", "text")).lower()
                    confidence = _SOURCE_DEFAULT_CONFIDENCE.get(source_tag, 0.5)
                else:
                    confidence = float(raw_conf)
                
                # VERITAS: Table data is usually high-prio, but if it has no confidence, we must be cautious
                if VERITAS_MODE and priority == 1 and confidence < 0.8:
                     confidence = 0.8 # Give tables a slight boost but still subject to 0.95 threshold
                    
                series_data = row.get(series_col, "")
                if pd.isna(series_data): series_data = ""
                    
                candidates.append({
                    "value": cleaned,
                    "series_data": series_data,
                    "priority": priority,
                    "confidence": confidence,
                    "source": row.get("_source", "unknown")
                })

        if not candidates:
            result[col] = ''
            result[series_col] = ''
            continue

        # Sort: ascending priority (1=best), then descending confidence
        candidates.sort(key=lambda c: (c["priority"], -c["confidence"]))
        best = candidates[0]
        
        # VERITAS FILTER: Reject if confidence is too low (Fix 3.5)
        if VERITAS_MODE and best["confidence"] < VERITAS_THRESHOLD:
            # Check if there is a 'Table' source that is slightly below threshold but logically confirmed
            # Otherwise, reject.
            if best["priority"] > 1: # Not a table
                result[col] = ''
                result[series_col] = ''
                continue

        # VERITAS CONFLICT CHECK: If we have multiple high-priority sources that disagree (Fix 3.6)
        if VERITAS_MODE and len(candidates) > 1:
            top_two = candidates[:2]
            val1 = str(top_two[0]["value"]).lower()
            val2 = str(top_two[1]["value"]).lower()
            if val1 != val2:
                # Disagreement between best sources. Reject to be safe.
                print(f"    [VERITAS] Rejected {col} due to source conflict: '{val1}' vs '{val2}'")
                result[col] = ''
                result[series_col] = ''
                continue

        result[col] = best["value"]
        result[series_col] = best["series_data"]
        source_summary_parts.append(f"{col}←{best['source']}")

    result["Proof"] = " | ".join(all_proofs) if all_proofs else ""
    result["source_summary"] = ", ".join(source_summary_parts) if source_summary_parts else ""
    return result


# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
def setup_paths(workspace=None, attributes=None):
    global TABLE_CSV, PLOT_CSV, TEXT_CSV, OUTPUT_MASTER, COLUMNS, OUTPUT_PLOT_DATASET

    from schema_loader import load_domain_schema
    schema_loader = None
    if workspace:
        schema_loader = load_domain_schema(workspace)

    if schema_loader and hasattr(schema_loader, "schema") and schema_loader.schema:
        user_attrs = list(schema_loader.schema.keys())
    elif attributes:
        user_attrs = [a.strip() for a in attributes.split(",") if a.strip()]
    else:
        user_attrs = []
        
    COLUMNS = ["File"] + user_attrs + ["Proof", "source_summary"]

    if workspace:
        TABLE_CSV     = os.path.join(workspace, "table_extracted_data.csv")
        PLOT_CSV      = os.path.join(workspace, "plot_extracted_data.csv")
        TEXT_CSV      = os.path.join(workspace, "text_extracted_data.csv")
        OUTPUT_MASTER = os.path.join(workspace, "resultant_dataset.csv")
        OUTPUT_PLOT_DATASET = os.path.join(workspace, "plot_dataset.csv")

    os.makedirs(os.path.dirname(OUTPUT_MASTER) if os.path.dirname(OUTPUT_MASTER) else BASE_DIR, exist_ok=True)


def generate_plot_series_rows(plot_csv_path):
    """Explode JSON (series) arrays into a point-by-point row-wise structural array matching the final dataset schema."""
    if not os.path.exists(plot_csv_path): return []
    try:
        df = pd.read_csv(plot_csv_path)
    except Exception: return []
    if df.empty: return []
    
    records = []
    for _, row in df.iterrows():
        file_name = row.get("File", "Unknown")
        series_cols = [c for c in df.columns if str(c).endswith("(series)")]
        
        for scol in series_cols:
            val = str(row.get(scol, "")).strip()
            if val.startswith("{") and "x" in val and "y" in val:
                try:
                    import json
                    data = json.loads(val)
                    x_arr = data.get("x", [])
                    y_arr = data.get("y", [])
                    if not x_arr or not y_arr or len(x_arr) != len(y_arr): continue
                    x_attr = data.get("x_attr", "power density")
                    y_attr = data.get("y_attr", "energy density")
                    
                    if x_attr == "Unknown" or y_attr == "Unknown": continue

                    pairs = sorted(list(zip(x_arr, y_arr)), key=lambda p: p[0])
                    
                    # Remove duplicate X
                    unique_pairs = []
                    seen_x = set()
                    for x, y in pairs:
                        if x not in seen_x:
                            seen_x.add(x)
                            unique_pairs.append((x, y))
                    
                    # Remove curve outliers
                    if len(unique_pairs) > 5:
                        s_y = pd.Series([p[1] for p in unique_pairs])
                        med_y = s_y.rolling(window=5, center=True, min_periods=1).median()
                        diff = (s_y - med_y).abs()
                        std_diff = diff.std()
                        valid_mask = diff < (2 * std_diff + 5.0)
                        unique_pairs = [unique_pairs[i] for i in range(len(unique_pairs)) if valid_mask.iloc[i]]
                        
                    for x, y in unique_pairs:
                        # VERITAS PHYSICAL BOUNDS CHECK (Fix 3.7)
                        if VERITAS_MODE:
                            x_val, y_val = float(x), float(y)
                            # Check X range
                            if x_attr in _ATTR_RANGE_MAP:
                                xmin, xmax = _ATTR_RANGE_MAP[x_attr]
                                if not (xmin <= x_val <= xmax): continue
                            # Check Y range
                            if y_attr in _ATTR_RANGE_MAP:
                                ymin, ymax = _ATTR_RANGE_MAP[y_attr]
                                if not (ymin <= y_val <= ymax): continue

                        records.append({
                            "File": file_name,
                            x_attr: round(float(x), 4),
                            y_attr: round(float(y), 4),
                            "Proof": str(row.get("Proof", "")).split("|")[0].strip() if "Proof" in row and pd.notnull(row["Proof"]) else "Plot",
                            "source_summary": "plot_series_point"
                        })
                except Exception as e:
                    print(f"Error parsing series JSON: {e}")
                    
    return records

# ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------
def main():
    start_time = time.time()

    def load_csv_with_source(path, source_tag):
        """Load a CSV and tag it with its source name and priority."""
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    df["_source"] = source_tag
                    return df
            except Exception as e:
                print(f"Error loading {os.path.basename(path)}: {e}")
        return None

    sources = [
        (TABLE_CSV, "table"),
        (PLOT_CSV,  "plot"),
        (TEXT_CSV,  "text"),
    ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(load_csv_with_source, path, tag): tag for path, tag in sources}
        dfs = [f.result() for f in futures if f.result() is not None]

    if not dfs:
        print("No extraction data found to merge.")
        pd.DataFrame(columns=COLUMNS).to_csv(OUTPUT_MASTER, index=False)
        return

    print(f"Loaded sources: {[df['_source'].iloc[0] for df in dfs]}")

    # Concatenate all sources
    master_df = pd.concat(dfs, ignore_index=True, sort=False)

    # Determine attribute columns (exclude meta columns)
    meta_cols  = {"File", "Proof", "_source", "confidence", "source_summary"}
    # Base attributes (we don't want to process '(series)' columns directly through validation)
    attr_cols  = [c for c in COLUMNS if c not in meta_cols and not str(c).endswith("(series)")]
    
    # Also include any attribute columns present in the data but not explicitly requested
    extra_cols = [c for c in master_df.columns if c not in meta_cols and c not in attr_cols and not str(c).endswith("(series)")]
    all_attr_cols = attr_cols + extra_cols

    print(f"Attributes to consolidate: {all_attr_cols}")
    print(f"Raw rows from all sources: {len(master_df)}")

    # Ensure File column exists
    if "File" not in master_df.columns:
        print("[Error] No 'File' column found in extracted data. Aborting.")
        return

    # Group by File and consolidate to one row per paper
    print("Consolidating to one row per paper (source priority + confidence)...")
    consolidated_rows = []
    for file_name, group in master_df.groupby("File", sort=False):
        row = consolidate_file_group(group, all_attr_cols)
        consolidated_rows.append(row)

    final_df = pd.DataFrame(consolidated_rows)

    # Reorder columns: File, then attributes in requested order, then Proof, source_summary
    ordered_cols = (
        ["File"]
        + [c for c in attr_cols if c in final_df.columns]
        + [c for c in extra_cols if c in final_df.columns]
        + [c for c in ["Proof", "source_summary"] if c in final_df.columns]
    )
    final_df = final_df[ordered_cols]

    # Drop rows that have zero attribute data (all attrs empty)
    if all_attr_cols:
        attr_present = final_df[[c for c in all_attr_cols if c in final_df.columns]].ne('').any(axis=1)
        final_df = final_df[attr_present]

    # Append plot series arrays directly into the output dataset matrix
    series_rows = generate_plot_series_rows(PLOT_CSV)
    if series_rows:
        series_df = pd.DataFrame(series_rows)
        final_df = pd.concat([final_df, series_df], ignore_index=True, sort=False)
        final_df = final_df.fillna("")

    # Final cleanliness: Remove exact duplicate rows (Fix 3.3)
    if not final_df.empty:
        # Ensure all columns are compared for exact clones
        final_df = final_df.drop_duplicates().reset_index(drop=True)

    final_df.to_csv(OUTPUT_MASTER, index=False)

    elapsed = time.time() - start_time
    print(f"\n=== V2 Merge Complete ===")
    print(f"Input rows (all sources): {len(master_df)}")
    print(f"Output rows (1 per paper): {len(final_df)}")
    print(f"Schema attributes validated: {len(_ATTR_UNIT_MAP)}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Output: {OUTPUT_MASTER}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge Datasets V2 — Intelligent Consolidation Engine")
    parser.add_argument("--workspace",   type=str, help="Workspace directory for this session")
    parser.add_argument("--attributes",  type=str, help="Comma-separated attribute names")
    args = parser.parse_args()

    setup_paths(args.workspace, args.attributes)
    main()
