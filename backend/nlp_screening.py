import os
import json
import re
import argparse
from pathlib import Path
try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF (fitz) is not installed. Please install it using: pip install pymupdf")
    exit(1)
from schema_loader import load_domain_schema

# ---------------------------------------------------------
# MODULE-LEVEL SCHEMA LOADING
# Build alias → attribute_name map and valid unit set from schema
# ---------------------------------------------------------
_SCHEMA_UNITS = set()    # All valid measurement units across all attributes
_ALIAS_MAP = {}          # alias.lower() → attribute_name
_ATTR_UNITS = {}         # attribute_name → set of valid units
_PRIMARY_KW = []         # Injected via setup_paths
_SECONDARY_KW = []       # Injected via setup_paths

try:
    _schema_sys = load_domain_schema()
    if _schema_sys and hasattr(_schema_sys, 'schema'):
        for attr_name, attr_data in _schema_sys.schema.items():
            # Canonical unit set: store stripped/lowercased exact forms
            canonical = set(u.strip().lower() for u in attr_data.get("units", []))
            _ATTR_UNITS[attr_name] = canonical
            _SCHEMA_UNITS.update(canonical)
            # Build alias lookup
            for alias in attr_data.get("aliases", []):
                _ALIAS_MAP[alias.lower().strip()] = attr_name
            # Also map the attribute name itself
            _ALIAS_MAP[attr_name.lower().replace("_", " ")] = attr_name
            
    # CRITICAL: Always inject attribute names as primary keywords if not already present
    # This prevents 'Lithium' search queries from bypassing screening for 'SPION' tasks.
    for attr_name in _ATTR_UNITS.keys():
        clean_attr = attr_name.lower().replace("_", " ")
        if clean_attr not in _PRIMARY_KW:
            _PRIMARY_KW.append(clean_attr)
except Exception as e:
    print(f"[Warning] Could not load schema for screening: {e}")

# Canonical unit normalization table: handles common shorthand variants
# Maps variant → canonical form used in schema
_UNIT_NORMALIZE = {
    "f g-1": "f/g",    "f·g-1": "f/g",    "fg-1": "f/g",
    "wh kg-1": "wh/kg", "wh·kg-1": "wh/kg",
    "w kg-1": "w/kg",   "w·kg-1": "w/kg",
    "a g-1": "a/g",    "a·g-1": "a/g",
    "mah g-1": "mah/g",
    "m2 g-1": "m²/g",  "m2g-1": "m²/g",
}

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Will be overridden by setup_paths)
# ---------------------------------------------------------
INPUT_DIR = ""
OUTPUT_DIR = ""
OUTPUT_FILE = ""

# False positive patterns to reject explicitly before scoring
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')                         # Years: 1900–2099
_REF_RE = re.compile(r'\[\d+\]')                                    # References: [12]
_FIG_RE = re.compile(r'\b(fig(?:ure)?|table|equation|eq)\.?\s*\d+\b', re.IGNORECASE)  # Fig. 3
_DOI_RE = re.compile(r'10\.\d{4,}/\S+')                             # DOI strings

# Section boundary markers — strip everything after these in the text
_BACK_MATTER_RE = re.compile(
    r'\n\s*(?:references|bibliography|acknowledgements?|appendix|supporting information)\s*\n',
    re.IGNORECASE
)

# Review/survey paper markers
_REVIEW_MARKERS = {"review", "survey", "meta-analysis", "systematic review", "literature review", "overview", "perspective"}
_EXPERIMENTAL_MARKERS = {"experimental", "methodology", "characterization", "synthesis", "electrochemical", "results", "fabrication"}


def setup_paths(workspace=None, keywords=None):
    global INPUT_DIR, OUTPUT_DIR, OUTPUT_FILE, _PRIMARY_KW, _SECONDARY_KW
    
    if keywords:
        # Accept structured JSON dict or comma-separated legacy format
        try:
            kw_dict = json.loads(keywords)
            if isinstance(kw_dict, dict):
                _PRIMARY_KW = [k.strip() for k in kw_dict.get("primary", []) if k.strip()]
                _SECONDARY_KW = [k.strip() for k in kw_dict.get("secondary", []) if k.strip()]
            else:
                raise ValueError("Not a dict")
        except (json.JSONDecodeError, ValueError):
            # Legacy comma-separated string — split and divide
            raw = [k.strip() for k in keywords.split(",") if k.strip()]
            mid = len(raw) // 2
            _PRIMARY_KW = raw[:mid] or raw
            _SECONDARY_KW = raw[mid:]
    
    if workspace:
        # Override module-level schema with session-specific aliases.json if it exists
        local_aliases_path = os.path.join(workspace, "aliases.json")
        if os.path.exists(local_aliases_path):
            try:
                with open(local_aliases_path, 'r', encoding='utf-8') as f:
                    local_schema = json.load(f)
                    for attr_name, attr_data in local_schema.items():
                        # Update alias map and valid units from the session's actual targets
                        _ATTR_UNITS[attr_name] = set(u.strip().lower() for u in attr_data.get("units", []))
                        for alias in attr_data.get("aliases", []):
                            _ALIAS_MAP[alias.lower().strip()] = attr_name
                        _ALIAS_MAP[attr_name.lower().replace("_", " ")] = attr_name
                        # Add to primary keywords to ensure we gate for THESE attributes
                        _PRIMARY_KW.append(attr_name.lower().replace("_", " "))
            except Exception as e:
                print(f"[Warning] Failed to load local aliases for screening: {e}")

        project_root = os.path.dirname(os.path.dirname(workspace))
        task_id = os.path.basename(workspace)
        INPUT_DIR = os.path.join(project_root, "PDFs", task_id)
        OUTPUT_DIR = os.path.join(workspace, "2_Screening_Results")
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, "screening_results.json")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# SECTION FILTERING (Fix 3.5 / Improvement 2)
# ---------------------------------------------------------
def strip_back_matter(text):
    """
    Remove all content after section headers like 'References', 'Bibliography',
    'Acknowledgements', 'Appendix'. This avoids false positives from citation lists.
    """
    match = _BACK_MATTER_RE.search(text)
    if match:
        return text[:match.start()]
    return text


# ---------------------------------------------------------
# TEXT EXTRACTION: Smart Page Sampling
# ---------------------------------------------------------
def extract_sampled_text(pdf_path, max_head_pages=3, numeric_density_threshold=5):
    """
    Extract text from:
      1. First max_head_pages pages (abstract + intro)
      2. Any page that has high numeric density (more potential measurements)
    """
    text_parts = []
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                page_text = page.get_text()
                if not page_text:
                    continue
                # Always include first N pages
                if i < max_head_pages:
                    text_parts.append(page_text)
                else:
                    # Include high numeric density pages
                    numeric_hits = len(re.findall(r'\b\d+\.?\d*\b', page_text))
                    words = len(page_text.split())
                    density = numeric_hits / max(words, 1)
                    if density > 0.05 and numeric_hits >= numeric_density_threshold:
                        text_parts.append(page_text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None
    
    # Remove back matter (references, bibliography) before returning
    raw_text = "\n".join(text_parts)
    return strip_back_matter(raw_text)


# ---------------------------------------------------------
# VALIDATED NUMERIC DETECTION
# ---------------------------------------------------------
def is_false_positive_number(context_str, number_str):
    """Reject if the number is likely a year, reference, fig number, or metadata."""
    stripped = number_str.strip()
    if _YEAR_RE.fullmatch(stripped):
        return True
    if _REF_RE.search(context_str):
        return True
    if _FIG_RE.search(context_str):
        return True
    if _DOI_RE.search(context_str):
        return True
    return False

def find_valid_measurements(text):
    """
    Find all (value, unit, attribute) triples where the unit EXACTLY matches a schema attribute unit.
    Uses canonical normalization — no substring collisions like 'g' matching 'mg'.
    Returns list of dicts: {"value": str, "unit": str, "attribute": str, "pos": int}
    """
    measurements = []
    pattern = re.compile(r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s*±\s*\d+(?:\.\d+)?)?)\s*([a-zA-Z%/µ·²]+(?:[·/][a-zA-Z²]+)*)')
    
    for m in pattern.finditer(text):
        number_str = m.group(1)
        raw_unit = m.group(2).strip()
        unit_str = raw_unit.lower()
        context = text[max(0, m.start()-30):m.end()+30]
        
        if is_false_positive_number(context, number_str):
            continue
        
        # Normalize the unit via the canonicalization table
        unit_canonical = _UNIT_NORMALIZE.get(unit_str, unit_str)
        
        # STRICT exact match only — no substring matching
        matched_attr = None
        for attr_name, units in _ATTR_UNITS.items():
            if unit_canonical in units:
                matched_attr = attr_name
                break
        
        if matched_attr:
            measurements.append({
                "value": number_str.strip(),
                "unit": unit_canonical,
                "attribute": matched_attr,
                "pos": m.start()
            })
    
    return measurements


# ---------------------------------------------------------
# ALIAS-AWARE KEYWORD MATCHING
# ---------------------------------------------------------
def find_keyword_hits(text, primary_kw, secondary_kw):
    """
    Match primary + secondary keywords against text, then expand using only the aliases
    for the attributes already triggered by the user's keywords (pre-filtered).
    Returns list of dicts: {"term": str, "attribute": str or None, "strength": str, "pos": int}
    """
    hits = []
    text_lower = text.lower()
    triggered_attributes = set()
    
    def search_term(term, strength):
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        for m in re.finditer(pattern, text_lower):
            attr = _ALIAS_MAP.get(term.lower())
            if attr:
                triggered_attributes.add(attr)
            hits.append({"term": term, "attribute": attr, "strength": strength, "pos": m.start()})
    
    for kw in primary_kw:
        search_term(kw, "primary")
    for kw in secondary_kw:
        search_term(kw, "secondary")
    
    # Pre-filter alias scan: only expand aliases for triggered attributes
    # This avoids O(N × all_aliases) — now O(N × relevant_aliases)
    relevant_aliases = {
        alias: attr for alias, attr in _ALIAS_MAP.items()
        if attr in triggered_attributes and len(alias) >= 3
    }
    
    # Extend with fallback: if no attrs triggered yet, scan only short curated alias list
    if not triggered_attributes:
        relevant_aliases = {k: v for k, v in _ALIAS_MAP.items() if len(k) >= 5}
    
    existing_positions = {h["pos"] for h in hits}
    for alias, attr_name in relevant_aliases.items():
        pattern = r'\b' + re.escape(alias) + r'\b'
        for m in re.finditer(pattern, text_lower):
            if m.start() not in existing_positions:
                hits.append({"term": alias, "attribute": attr_name, "strength": "alias", "pos": m.start()})
                existing_positions.add(m.start())
    
    return hits


# ---------------------------------------------------------
# PROXIMITY SCORING
# ---------------------------------------------------------
def compute_proximity_score(keyword_hits, measurements, text, proximity_words=15):
    """
    Score how many (keyword, measurement) pairs appear within proximity_words of each other.
    Returns:
      - proximity_score: float 0-1
      - matched_pairs: list of (keyword_term, measurement_value+unit) pairs
    """
    if not keyword_hits or not measurements:
        return 0.0, []
    
    words = text.split()
    word_positions = {}
    idx = 0
    char_pos = 0
    for word in words:
        word_positions[char_pos] = idx
        char_pos += len(word) + 1
        idx += 1
    
    def char_to_word_idx(char_pos):
        # Find nearest word index before or at this char position
        candidates = [k for k in word_positions if k <= char_pos]
        if not candidates: return 0
        return word_positions[max(candidates)]
    
    strong_pairs = []
    weak_pairs = []
    
    for hit in keyword_hits:
        for meas in measurements:
            kw_word_idx = char_to_word_idx(hit["pos"])
            meas_word_idx = char_to_word_idx(meas["pos"])
            distance = abs(kw_word_idx - meas_word_idx)
            
            if distance <= proximity_words:
                strong_pairs.append((hit["term"], f"{meas['value']} {meas['unit']}"))
            elif distance <= proximity_words * 3:
                weak_pairs.append((hit["term"], f"{meas['value']} {meas['unit']}"))
    
    # Score: strong pairs worth more than weak
    strong_score = min(1.0, len(strong_pairs) * 0.4)
    weak_score = min(0.5, len(weak_pairs) * 0.1)
    
    return min(1.0, strong_score + weak_score * 0.3), strong_pairs + weak_pairs


# ---------------------------------------------------------
# MAIN SCREENING FUNCTION
# ---------------------------------------------------------
def screen_paper(pdf_path, score_threshold=0.40):
    filename = os.path.basename(pdf_path)
    text = extract_sampled_text(pdf_path)
    
    if text is None or len(text.strip()) < 50:
        return {
            "file": filename,
            "include": False,
            "score": 0.0,
            "paper_type": "unknown",
            "reason": "Failed to extract sufficient text from PDF.",
            "keyword_hits": [],
            "measurements": []
        }
    
    text_lower = text.lower()
    
    # ---- Paper Type Detection ----
    has_review_marker = any(marker in text_lower for marker in _REVIEW_MARKERS)
    has_experimental = any(marker in text_lower for marker in _EXPERIMENTAL_MARKERS)
    
    if has_review_marker and not has_experimental:
        paper_type = "review"
        review_penalty = 0.7
    elif has_review_marker:
        paper_type = "review_with_data"
        review_penalty = 0.85
    else:
        paper_type = "research_article"
        review_penalty = 1.0
    
    # ---- Keyword Matching ----
    kw_hits = find_keyword_hits(text, _PRIMARY_KW, _SECONDARY_KW)
    
    primary_hits = [h for h in kw_hits if h["strength"] == "primary"]
    secondary_hits = [h for h in kw_hits if h["strength"] == "secondary"]
    alias_hits = [h for h in kw_hits if h["strength"] == "alias"]
    
    # Keyword score (0 - 0.40)
    # Primary hits are mandatory — zero primary hits → domain gate fails
    if not primary_hits and not alias_hits:
        return {
            "file": filename,
            "include": False,
            "score": 0.0,
            "paper_type": paper_type,
            "reason": "No primary keyword or alias match found — domain gate failed.",
            "keyword_hits": [],
            "measurements": []
        }
    
    # Score: weighted combination of hit quality
    kw_score = min(0.40,
        len(primary_hits) * 0.12 +
        len(alias_hits) * 0.08 +
        len(secondary_hits) * 0.04
    )

    # ---- Validated Numeric Detection ----
    measurements = find_valid_measurements(text)
    
    # Unique attributes from measurements
    meas_attributes = set(m["attribute"] for m in measurements)
    
    # Numeric validity score (0 - 0.30): weighted by unique attribute coverage
    # Rather than raw count: reward diversity not volume
    unique_attr_count = len(meas_attributes)
    raw_meas_count = len(measurements)
    
    # Score favors papers that show measurements across MULTIPLE attributes
    attr_diversity_score = min(0.20, unique_attr_count * 0.07)
    # Also reward raw count but with diminishing returns (log scale)
    import math
    density_bonus = min(0.10, math.log1p(raw_meas_count) * 0.03)
    numeric_score = attr_diversity_score + density_bonus
    
    # ---- Attribute Match Score ----
    # Check if any keyword maps to an attribute AND there's a measurement for the same attribute
    hit_attributes = set(h["attribute"] for h in kw_hits if h["attribute"])
    meas_attributes = set(m["attribute"] for m in measurements)
    shared_attributes = hit_attributes & meas_attributes
    
    # Attribute match score (0 - 0.30)
    attr_score = min(0.30, len(shared_attributes) * 0.10)
    
    # ---- Proximity Bonus ----
    proximity_score, matched_pairs = compute_proximity_score(kw_hits, measurements, text)
    # Proximity acts as a score amplifier — up to +0.10 bonus
    proximity_bonus = min(0.10, proximity_score * 0.10)
    
    # ---- Final Score Assembly ----
    raw_score = kw_score + numeric_score + attr_score + proximity_bonus
    final_score = round(min(1.0, raw_score) * review_penalty, 3)
    
    include = final_score >= score_threshold
    
    reason_parts = []
    if not include:
        reason_parts.append(f"Score {final_score:.2f} below threshold {score_threshold:.2f}")
    if paper_type in ("review", "review_with_data"):
        reason_parts.append(f"Review paper detected (penalty x{review_penalty})")
    if not reason_parts:
        reason_parts.append("All criteria met")

    return {
        "file": filename,
        "include": include,
        "score": final_score,
        "paper_type": paper_type,
        "reason": " | ".join(reason_parts),
        "keyword_hits": [{"term": h["term"], "attribute": h["attribute"], "strength": h["strength"]} for h in kw_hits[:15]],
        "measurements": [{"value": m["value"], "unit": m["unit"], "attribute": m["attribute"]} for m in measurements[:10]],
        "matched_pairs": [f"{pair[0]} → {pair[1]}" for pair in matched_pairs[:5]],
        "attributes_matched": list(shared_attributes)
    }


# ---------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------
def main():
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found. Skipping screening.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return

    print(f"Found {len(pdf_files)} PDF files. Starting V2 attribute-aware screening...")
    print(f"  Primary Keywords  [{len(_PRIMARY_KW)}]: {', '.join(_PRIMARY_KW)}")
    print(f"  Secondary Keywords[{len(_SECONDARY_KW)}]: {', '.join(_SECONDARY_KW)}")
    print(f"  Schema Units loaded: {len(_SCHEMA_UNITS)}")
    print(f"  Schema Aliases loaded: {len(_ALIAS_MAP)}")
    
    results = []
    included_count = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        print(f"Processing [{i}/{len(pdf_files)}]: {pdf_file}")
        
        result = screen_paper(pdf_path)
        results.append(result)
        
        status = "INCLUDED" if result["include"] else "EXCLUDED"
        print(f"  -> {status} | Score: {result['score']:.3f} | {result['reason'][:60]}")
        if result["include"]:
            included_count += 1
            
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- V2 Screening Complete ---")
    print(f"Total Processed : {len(pdf_files)}")
    print(f"Total Included  : {included_count}")
    print(f"Total Excluded  : {len(pdf_files) - included_count}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Screening V2 — Attribute-Aware Engine")
    parser.add_argument("--workspace", type=str, help="Workspace directory for this session")
    parser.add_argument("--keywords", type=str, 
        help='Keywords as JSON dict {"primary": [...], "secondary": [...]} or comma-separated string')
    args = parser.parse_args()
    
    setup_paths(args.workspace, args.keywords)
    main()
