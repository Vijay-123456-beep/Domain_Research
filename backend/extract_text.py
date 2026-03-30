import os
import json
import re
import csv
import fitz
from typing import Dict, List, Optional, Any

from schema_loader import SchemaLoader, load_domain_schema
from unit_parser import UnitParser, get_unit_parser, ParsedValue
from validation_engine import ValidationEngine, get_validation_engine, ValidationResult, ValidationStatus
from llm_validator import extract_measurements_batch

# ---------------------------------------------------------
# GLOBAL CONFIGURATION (Defaults - will be overridden by --workspace)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INCLUDED_DIR = os.path.join(os.path.dirname(BASE_DIR), "Included")
ANALYSIS_JSON = os.path.join(BASE_DIR, "page_analysis_results.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "text_extracted_data.csv")
LOG_CSV = os.path.join(BASE_DIR, "text_extraction_log.csv")

COLUMNS = []
USER_ATTRIBUTES = []
SCHEMA_LOADER = None
UNIT_PARSER = None
VALIDATION_ENGINE = None

def setup_paths(workspace=None, attributes=None):
    global INCLUDED_DIR, ANALYSIS_JSON, OUTPUT_CSV, LOG_CSV, COLUMNS, USER_ATTRIBUTES
    global SCHEMA_LOADER, UNIT_PARSER, VALIDATION_ENGINE
    
    # Initialize new validation modules
    if workspace:
        SCHEMA_LOADER = load_domain_schema(workspace)
        UNIT_PARSER = get_unit_parser()
        VALIDATION_ENGINE = get_validation_engine(SCHEMA_LOADER)
        
    if SCHEMA_LOADER and hasattr(SCHEMA_LOADER, 'schema') and SCHEMA_LOADER.schema:
        USER_ATTRIBUTES = list(SCHEMA_LOADER.schema.keys())
        COLUMNS = USER_ATTRIBUTES + ["Proof", "File"]
    elif attributes:
        USER_ATTRIBUTES = [a.strip() for a in attributes.split(",") if a.strip()]
        COLUMNS = USER_ATTRIBUTES + ["Proof", "File"]
    else:
        # No default fallback with hardcoded attributes
        USER_ATTRIBUTES = []
        COLUMNS = ["Proof", "File"]

    # Load aliases and units dynamically (legacy support)
    global DYNAMIC_ALIASES
    DYNAMIC_ALIASES = {}
    if workspace:
        alias_file = os.path.join(workspace, "aliases.json")
        if os.path.exists(alias_file):
            try:
                with open(alias_file, 'r', encoding='utf-8') as f:
                    raw_aliases = json.load(f)
                    DYNAMIC_ALIASES = {k.lower(): v for k, v in raw_aliases.items()}
            except Exception as e:
                print(f"Error loading aliases.json: {e}")

    if workspace:
        project_root = os.path.dirname(os.path.dirname(workspace))
        task_id = os.path.basename(workspace)
        
        INCLUDED_DIR = os.path.join(project_root, "Included", task_id)
        ANALYSIS_JSON = os.path.join(workspace, "page_analysis_results.json")
        OUTPUT_CSV = os.path.join(workspace, "text_extracted_data.csv")
        LOG_CSV = os.path.join(workspace, "text_extraction_log.csv")
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# PHYSICAL VALIDATION RANGES - Add physics-based sanity checks
PHYSICAL_RANGES = {
    # ── Magnetic nanoparticle attributes (legacy) ──────────────────────────────
    "AMF frequency": (1, 1e12, True),
    "AMF amplitude": (0.001, 1000, True),
    "saturation magnetization": (0.1, 500, True),
    "specific absorption rate": (0.1, 5000, True),
    "core diameter": (0.1, 10000, True),
    "diameter standard deviation": (0.001, 1000, True),
    "temperature": (0.1, 2000, True),
    "coercivity": (0.0, 1e6, True),
    "remanence": (0.0, 100, True),
    # ── Biomass carbon supercapacitor attributes ────────────────────────────────
    # Specific capacitance: published range 1–3000 F/g; above 3000 is not credible
    "specific capacitance": (1.0, 3000.0, True),
    # Current density: 0.05–100 A/g is the experimental playground
    "current density": (0.01, 100.0, True),
    # BET specific surface area: activated carbons 0.1–3500 m²/g
    "specific surface area": (0.1, 3500.0, True),
    # Micropore/total pore volume: 0.001–5 cm³/g
    "micropore volume": (0.001, 5.0, True),
    "pore volume": (0.001, 5.0, True),
    # Nitrogen/oxygen/carbon content: 0–100 wt%
    "nitrogen content": (0.01, 60.0, True),
    "oxygen content": (0.01, 60.0, True),
    "carbon content": (10.0, 100.0, True),
    # ID/IG ratio: physically bounded [0.3, 2.5] for disordered carbons
    "id ig ratio": (0.3, 2.5, True),
    # Pore size (diameter): 0.1–100 nm for micro/mesoporous carbons
    "pore size": (0.1, 200.0, True),
    # Carbon-to-oxygen ratio: 1–500 (atomic or mass ratio)
    "carbon oxygen ratio": (1.0, 500.0, True),
    # Scan rate: 1–1000 mV/s typical CV experiments
    "scan rate": (0.1, 1000.0, True),
    # Potential window: -5 to 5 V; absolute value 0.1–5 V
    "potential window": (-5.0, 5.0, False),
    # Micropore surface area: 0–3500 m²/g (subset of BET)
    "micropore surface area": (0.1, 3500.0, True),
    # Energy density: 0.01–1000 Wh/kg
    "energy density": (0.01, 1000.0, True),
    # Power density: 0.01–100000 W/kg
    "power density": (0.01, 100000.0, True),
}

# Attributes that should NEVER be negative
POSITIVE_ONLY_ATTRS = [
    "AMF frequency", "AMF amplitude", "saturation magnetization",
    "specific absorption rate", "core diameter", "diameter standard deviation",
    "temperature", "coercivity", "remanence", "SPION concentration",
    "core surface area", "core volume",
    # Supercapacitor positives
    "specific capacitance", "current density", "specific surface area",
    "micropore volume", "pore volume", "nitrogen content", "oxygen content",
    "carbon content", "id ig ratio", "pore size", "carbon oxygen ratio",
    "scan rate", "micropore surface area", "energy density", "power density",
]

# Axis label patterns to reject - these are chart labels, not data
AXIS_LABEL_PATTERNS = [
    r'frequency\s*\([^)]+\)\s*$',  # "Frequency (MHz)" at end of line
    r'\bfrequency\s*\(?\s*(?:mhz|khz|hz|ghz)\s*\)?\s*$',  # Frequency with unit at EOL
    r'\d+\s*(?:mhz|khz|hz|ghz)\s*$',  # Just a number with frequency unit at EOL
    r'\b(?:x|y)\s*-?axis\b',  # X-axis, Y-axis mentions
    r'^\s*(?:0|50|100|150|200|250|300|350|400|450|500)\s*$',  # Isolated tick marks
]

# Strict Extraction Rejection Patterns (Rule 2)
THEORETICAL_PATTERNS = [
    r'\b(?:calculated|simulated|theoretical|predicted|expected|estimated|modeled)\b'
]
TREND_PATTERNS = [
    r'\b(?:increased|decreased|higher|lower|better|worse|improved|reduced|dropped|rose|trend)\b'
]
METHOD_PATTERNS = [
    r'\b(?:prepared|synthesized|washed|heated|cooled|stirred|mixed|added|dropwise|annealed)\b'
]

def is_axis_label(line, val):
    """Check if this is likely a chart axis label rather than actual data."""
    line_lower = line.lower().strip()
    
    # Check axis label patterns
    for pattern in AXIS_LABEL_PATTERNS:
        if re.search(pattern, line_lower):
            return True
    
    # If line ONLY contains the number and whitespace, it's likely a tick
    if re.match(r'^\s*[-+]?\d+(?:\.\d+)?\s*$', line.strip()):
        return True
    
    # If the number appears with just unit and no context words, likely axis
    context_words = ['measured', 'value', 'was', 'is', 'found', 'obtained', 'determined', 'calculated', 'gave', 'yielded']
    has_context = any(w in line_lower for w in context_words)
    if not has_context:
        # Check if it's just number + unit pattern
        unit_pattern = r'[-+]?\d+(?:\.\d+)?\s*(?:mhz|khz|hz|ghz|nm|mm|cm|µm|um|kg|g|mg|w|mw|kw|t|mt|°c|k|a/m|ka/m)\s*$'
        if re.search(unit_pattern, line_lower):
            return True
    
    return False

def validate_physical_range(attr, val):
    """
    Validate if extracted value is within physically reasonable range.
    Returns (is_valid, reason) tuple.
    """
    try:
        # Handle range strings like "10-20" by taking first value
        val_str = str(val).split('-')[0].strip()
        val_float = float(val_str)
    except (ValueError, TypeError):
        return True, None  # Can't validate, assume ok
    
    attr_lower = attr.lower()
    
    # Check positive-only attributes
    for pos_attr in POSITIVE_ONLY_ATTRS:
        if pos_attr.lower() in attr_lower:
            if val_float < 0:
                return False, f"Negative value for positive-definite quantity {attr}"
            # Also reject near-zero for things that can't be zero
            if pos_attr.lower() in ["saturation magnetization", "specific absorption rate"]:
                if val_float == 0 or val_float < 0.01:
                    return False, f"Zero/invalid {attr} - magnetic materials must have non-zero values"
            break
    
    # Check defined ranges
    for range_attr, (min_val, max_val, must_be_positive) in PHYSICAL_RANGES.items():
        if range_attr.lower() in attr_lower:
            if val_float < min_val or val_float > max_val:
                return False, f"{attr}={val_float} outside physical range [{min_val}, {max_val}]"
            if must_be_positive and val_float <= 0:
                return False, f"{attr} must be positive, got {val_float}"
            break
    
    return True, None

def clean_text_line(text):
    # Normalize dashes and special chars
    text = text.replace("−", "-").replace("–", "-").replace("—", "-").replace("·", ".").replace("μ", "u").replace("µ", "u")
    text = re.sub(r'\s+', ' ', text)
    return text


def is_false_positive(line, match_str, attr, expected_units=None):
    """
    Check if a match is a likely false positive (formula constant, figure ref, year, citation, etc).
    """
    line_lower = line.lower()
    match_str_clean = match_str.strip('.,; ')
    
    # 1. Figure/Table/Reference Keyword Filtering
    ref_pattern = r'(?i)\b(?:fig(?:ure)?|table|ref|eq|step|scheme)\.?\s*\(?'+re.escape(match_str_clean)+r'\b'
    if re.search(ref_pattern, line):
        return True, "Figure/Table Reference"

    # 2. Year Detection (1900-2050)
    try:
        val_float = float(match_str_clean.split('-')[0]) # Handle ranges
        if 1900 <= val_float <= 2050:
            metadata_keywords = ["doi", "copyright", "published", "received", "accepted", "volume", "issue", "journal", "issn"]
            if any(kw in line_lower for kw in metadata_keywords):
                return True, "Publication Year/Metadata"
            if attr.lower() in ["specific capacitance", "current density", "pore size"]:
                 return True, "Year-like value for non-year attribute"
    except ValueError:
        pass

    # 3. Citation/Bracket Detection (e.g., [24], [12, 13], [1-5])
    bracket_pattern = r'\[[^\]]*?\b' + re.escape(match_str_clean) + r'\b[^\]]*?\]'
    if re.search(bracket_pattern, line):
        return True, "Citation/Bracketed Reference"

    # 4. Metadata Proximity Rejection
    metadata_blacklist = ["doi:", "vol.", "pp.", "issue", "issn"]
    for kw in metadata_blacklist:
        if kw in line_lower:
            pos = line_lower.find(kw)
            match_pos = line_lower.find(match_str_clean.lower())
            if abs(pos - match_pos) < 30:
                return True, "Metadata noise proximity"

    # 4a. Electrode Preparation / Binder Context Rejection
    binder_context_kws = [
        "acetylene black", "pvdf", "polyvinylidene", "nmp", "n-methylpyrrolidone",
        "binder", "nickel foam", "ni foam", "current collector",
        "slurry", "painted onto", "coated onto", "wt%",
    ]
    if attr.lower() in ("carbon content", "nitrogen content", "oxygen content"):
        if any(kw in line_lower for kw in binder_context_kws):
            return True, "Electrode binder/preparation context — not material property"

    # 4b. Comparative Upper / Lower Bound Rejection
    bound_phrases = [
        "smaller than", "less than", "below", "no more than",
        "lower than", "at most", "not exceed", "<",
        "larger than", "more than", "greater than", "above", ">",
    ]
    physical_attrs = {
        "pore size", "specific surface area", "pore volume",
        "micropore volume", "micropore surface area",
    }
    if attr.lower() in physical_attrs:
        pos_val = line_lower.find(match_str_clean.lower())
        surrounding = line_lower[max(0, pos_val - 40): pos_val + 40]
        if any(bp in surrounding for bp in bound_phrases):
            return True, "Comparative bound phrasing — not a measured value"

    # 5. Axis Label Detection - Reject chart tick marks and axis labels
    if is_axis_label(line, match_str_clean):
        return True, "Likely chart axis label/tick mark"

    # 6. Physical Sanity Checks
    positive_only_attrs = ["specific surface area", "pore volume", "nitrogen content", "specific capacitance", "current density", "pore size", "voltage"]
    if attr.lower() in positive_only_attrs:
        try:
            val = float(match_str_clean.split('-')[0])
            if val < 0:
                return True, "Physically impossible negative value"
            if attr.lower() == "voltage" and val > 10.0:
                 return True, "Voltage out of range"
        except ValueError:
            pass

    # 7. Physical Range Validation
    is_valid, reason = validate_physical_range(attr, match_str_clean)
    if not is_valid:
        return True, reason

    # 8. Unit-Aware Validation
    valid_units = [u.lower().strip() for u in expected_units if u and u.strip()]
    if valid_units:
        p_after = re.escape(match_str_clean) + r'.{0,15}(?:' + '|'.join([re.escape(u) for u in valid_units]) + r')(?:\b|(?![a-z]))'
        if not re.search(p_after, line_lower):
            return True, f"Missing unit ({'/'.join(valid_units)})"

    return False, None


def extract_value_unit_from_text_v2(text: str, expected_attr: str = None) -> Optional[ParsedValue]:
    """
    NEW: Extract value and unit from text using unit parser.
    Looks for patterns like 'value unit', 'value (unit)', 'value [unit]'.
    """
    if not text or not UNIT_PARSER:
        return None
    
    # Use unit parser to extract value-unit pairs
    pairs = UNIT_PARSER.extract_units_from_text(text)
    
    if pairs:
        # Return first valid pair
        value_str, unit_str = pairs[0]
        parsed = UNIT_PARSER.parse_value_string(value_str, unit_str)
        return parsed
    
    # Try to extract just numeric value if no unit found
    num_match = re.search(r'([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)', text)
    if num_match:
        return UNIT_PARSER.parse_value_string(num_match.group(1), None)
    
    return None


def validate_text_extraction_v2(value: Any, unit: Optional[str], 
                                   attribute: str, context: Dict) -> Optional[Dict]:
    """
    NEW: Validate text extraction using validation engine.
    Returns validated data or None if rejected.
    """
    if not VALIDATION_ENGINE:
        return None
    
    # Run validation
    result = VALIDATION_ENGINE.validate_datapoint(
        value=value,
        unit=unit,
        attribute=attribute,
        context=context
    )
    
    if not result.is_acceptable(0.4):  # Lower threshold for text
        return None
    
    return {
        'attribute': result.attribute,
        'value': result.value,
        'unit': result.unit,
        'confidence': result.confidence,
        'raw_value': str(value)
    }


def score_attribute_match_v2(text: str, parsed_value: ParsedValue, 
                              attribute: str, context_window: str = None) -> float:
    """
    NEW: Score how well a value matches an attribute using unit + context.
    Returns confidence score 0.0-1.0.
    """
    if not SCHEMA_LOADER:
        return 0.0
    
    # Get attribute data from schema
    attr_data = SCHEMA_LOADER.schema.get(attribute)
    if not attr_data:
        # Try to find by alias
        found = SCHEMA_LOADER.find_attribute_by_name(attribute)
        if found:
            attr_data = found[1]
            attribute = found[0]
        else:
            return 0.0
    
    scores = {
        'unit_match': 0.0,
        'context_match': 0.0,
        'range_fit': 0.0
    }
    
    # 1. Unit Match Score (Critical - Rule 1 & Rule 3)
    if parsed_value.unit:
        expected_units = attr_data.get('units', [])
        if UNIT_PARSER and UNIT_PARSER.is_valid_unit_for_attribute(parsed_value.unit, expected_units):
            scores['unit_match'] = 1.0
        elif expected_units and SCHEMA_LOADER.are_units_compatible(parsed_value.unit, expected_units[0]):
            scores['unit_match'] = 0.8
        else:
            return 0.0 # Strict fail: Unit present but incompatible
    else:
        # No unit - absolute failure unless attribute is dimensionless (Rule 1)
        if attr_data.get('dimension') == 'dimensionless':
            scores['unit_match'] = 0.5
        else:
            return 0.0 # Strict fail: Mandatory unit missing
    
    # 2. Context Match Score (30% weight)
    if context_window:
        context_lower = context_window.lower()
        aliases = [attribute] + attr_data.get('aliases', [])
        
        for alias in aliases:
            if alias.lower() in context_lower:
                scores['context_match'] = 1.0
                break
            # Partial word matching
            alias_words = alias.lower().split()
            matches = sum(1 for w in alias_words if len(w) > 3 and w in context_lower)
            if matches >= max(1, len(alias_words) // 2):
                scores['context_match'] = max(scores['context_match'], 0.7)
    
    # 3. Range Fit Score (20% weight)
    expected_range = attr_data.get('expected_range')
    if expected_range:
        min_val = expected_range.get('min')
        max_val = expected_range.get('max')
        val = parsed_value.value
        
        if min_val is not None and max_val is not None:
            if min_val <= val <= max_val:
                # Perfect fit - bonus
                center = (min_val + max_val) / 2
                distance_from_center = abs(val - center) / (max_val - min_val)
                scores['range_fit'] = 1.0 - (distance_from_center * 0.3)
            else:
                # Outside range - penalty
                if val < min_val:
                    scores['range_fit'] = max(0, 1.0 - (min_val - val) / min_val) if min_val > 0 else 0
                else:
                    scores['range_fit'] = max(0, 1.0 - (val - max_val) / max_val)
    
    # Weighted sum: Prioritize unit over context (Rule 3)
    total_score = (
        0.6 * scores['unit_match'] +
        0.3 * scores['context_match'] +
        0.1 * scores['range_fit']
    )
    
    if total_score < 0:
        return 0.0
    
    return round(total_score, 3)


def extract_text_data_v2(page_text: str, page_num: int, filename: str) -> List[Dict]:
    """
    NEW: Hybrid extraction pipeline (Regex Pre-filter -> LLM Batch -> Schema Validation).
    """
    if not SCHEMA_LOADER or not UNIT_PARSER:
        return []
    
    results = []
    lines = [clean_text_line(l) for l in page_text.split("\n")]
    
    # 1. Detect references section
    is_references = False
    valid_lines = []
    for i, line in enumerate(lines):
        if i > len(lines) * 0.5:
            if re.match(r'^\s*references?\s*$', line.lower()) or \
               re.match(r'^\s*bibliography\s*$', line.lower()):
                is_references = True
                break
                
    # 2. Heuristic Pre-Filtering (Collect Batch)
    candidate_batch = []
    line_map = {} # map ID back to actual line number and text
    
    for line_idx, line in enumerate(lines, 1):
        if not line.strip():
            continue
        
        if is_references and line_idx > len(lines) * 0.7:
            break
            
        line_lower = line.lower()
        if any(re.search(p, line_lower) for p in THEORETICAL_PATTERNS):
            continue
        if any(re.search(p, line_lower) for p in TREND_PATTERNS):
            continue
        if any(re.search(p, line_lower) for p in METHOD_PATTERNS):
            continue
            
        # VERY STRICT Phase 1: Only sentences with numbers AND units pass
        pairs = UNIT_PARSER.extract_units_from_text(line)
        if not pairs:
            continue
            
        # Add to candidate batch for LLM WITH CONTEXT (Gap 4)
        prev_line = lines[line_idx - 2].strip() if line_idx > 1 else ""
        next_line = lines[line_idx].strip() if line_idx < len(lines) else ""
        
        # Build contextual block but flag the primary line
        context_text = f"{prev_line} {line} {next_line}".strip()
        
        candidate_id = len(candidate_batch) + 1
        candidate_batch.append({
            "id": candidate_id,
            "text": context_text
        })
        line_map[candidate_id] = {"idx": line_idx, "text": context_text}

    # 3. LLM Batch Extraction
    if not candidate_batch:
        return []
        
    print(f"[PAGE {page_num}] Sending {len(candidate_batch)} pre-filtered sentences to LLM...")
    llm_results = extract_measurements_batch(candidate_batch, USER_ATTRIBUTES, SCHEMA_LOADER)
    
    # 4. Local Schema Validation on LLM output
    for item in llm_results:
        cid = item.get("id")
        if cid not in line_map:
            continue
            
        line_idx = line_map[cid]["idx"]
        line_text = line_map[cid]["text"]
        
        attr = item.get("attribute")
        val = item.get("value")
        unit = item.get("unit")
        llm_conf = item.get("confidence", 0)
        
        if not attr or val is None or llm_conf < 0.70:
            continue
            
        # GAP 1: Secondary Trend Guard AFTER LLM
        line_lower_ctx = line_text.lower()
        if any(re.search(p, line_lower_ctx) for p in TREND_PATTERNS):
            continue
            
        # GAP 2: Condition vs Measurement Filtering
        # "A/g", "°C", "V", "mA", "mV/s" are commonly conditions, not outcomes unless explicitly mapped
        condition_units = ["a/g", "ma/g", "°c", "c", "v", "mv/s", "a", "ma", "k"]
        if unit and unit.lower().strip() in condition_units:
            # If the unit is literally a condition unit, make sure the LLM actually mapped it to a matching attribute
            # For instance, if the LLM mapped "1 A/g" to "Specific Capacitance", that's WRONG, it's a condition.
            expected = SCHEMA_LOADER.schema.get(attr, {}).get('units', [])
            if unit.lower() not in [u.lower() for u in expected]:
                 continue # This is a condition hijacked as a result, drop it
            
        # Ensure LLM didn't hallucinate an attribute
        if attr not in USER_ATTRIBUTES and not SCHEMA_LOADER.schema.get(attr):
             # Try alias match fallback
             found = SCHEMA_LOADER.find_attribute_by_name(attr)
             if found: attr = found[0]
             else: continue
        
        context = {
            'source_type': 'text',
            'page': page_num,
            'line': line_idx,
            'surrounding_text': line_text,
            'reason': item.get("reason", "")
        }
        
        validated = validate_text_extraction_v2(val, unit, attr, context)
        
        if validated:
            validated['score'] = llm_conf
            validated['proof'] = f"[Page {page_num}, Line ~{line_idx}, {attr}, {val} {unit}] (LLM Conf: {llm_conf})"
            results.append(validated)
    
    return results


def extract_text_data(page_text, page_num, filename):
    """
    Extracts features from the text of a single page, tracking provenance.
    Returns a list of candidates: [(attr, val, score, proof_entry)]
    """
    # Try new V2 extraction first
    v2_results = extract_text_data_v2(page_text, page_num, filename)
    if v2_results:
        # Convert to legacy format
        candidates = []
        for result in v2_results:
            candidates.append((
                result['attribute'],
                str(result['value']),
                int(result.get('score', 0.5) * 200),  # Scale to legacy score range
                result.get('proof', '')
            ))
        return candidates
    
    # Fall back to legacy extraction
    lines = [clean_text_line(l) for l in page_text.split("\n")]
    page_candidates = []
    # Stop processing if we hit References
    is_references_section = False

    for line_idx, line in enumerate(lines, 1):
        line_lower = line.lower()
        # Detect start of References
        if line_idx > len(lines) * 0.5 and ("references" in line_lower or "bibliography" in line_lower):
            if len(line.strip()) < 20: # Likely a header
                is_references_section = True

        for attr in USER_ATTRIBUTES:
            attr_lower = attr.lower()
            metadata = DYNAMIC_ALIASES.get(attr) or DYNAMIC_ALIASES.get(attr_lower) or {}
            
            # ... (metadata loading remains the same)
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
            
            # Match Keyword + Number in ±8 words window
            alias_list_clean = [attr.lower()] + [a.lower() for a in aliases]
            val_pattern = r"([-+]?\d+(?:\.\d+)?(?:\s*[-,–—]\s*\d+(?:\.\d+)?|(?:\s*,\s*\d+(?:\.\d+)?)+)?)"
            
            words = line.split()
            all_matches = []
            for i, word in enumerate(words):
                word_clean = word.lower().strip(".,()[]")
                # Check for exact alias match OR partial match for multi-word attributes
                is_match = False
                if word_clean in alias_list_clean:
                    is_match = True
                else:
                    # Try partial matching for multi-word attributes (e.g., "lithium" matching "lithium ion conductivity")
                    attr_words = attr_lower.split()
                    if len(attr_words) > 1 and word_clean in attr_words and len(word_clean) > 3:
                        is_match = True
                
                if is_match:
                    # Look ±8 words for a number
                    start = max(0, i - 8)
                    end = min(len(words), i + 9)
                    for j in range(start, end):
                        if i == j: continue
                        val_match = re.search(val_pattern, words[j])
                        if val_match:
                            val = val_match.group(1)
                            all_matches.append((val, line_lower.find(val.lower())))

            for val, match_pos in all_matches:
                # Validation
                fp, reason = is_false_positive(line, val, attr, expected_units)
                
                score = 0
                if not fp:
                    score += 200 # Heavy weight for validated unit match
                elif "Missing required unit" in reason:
                    score += 5   # Very weak without unit
                    # print(f"    [DEBUG] {attr} matched '{val}' but {reason}")
                else:
                    # print(f"    [DEBUG] {attr} matched '{val}' but FILTERED: {reason}")
                    continue     # Hard skip (figures/formulas/citations)

                # Section penalty
                if is_references_section:
                    score -= 1000 # Aggressive penalty for references

                # Contextual Scoring
                context_window = line_lower[max(0, match_pos-150):min(len(line_lower), match_pos+150)]
                
                # Check for exact attribute name match (High confidence)
                if attr.lower() in context_window:
                    score += 100
                else:
                    # Partial word match score
                    words = [w for w in attr.lower().split() if len(w) > 3]
                    matched_count = sum(1 for w in words if w in context_window)
                    score += matched_count * 20

                proof_entry = f"Page {page_num}, Line {line_idx} → {attr}"
                page_candidates.append((attr, val, score, proof_entry))

    return page_candidates

def main():
    all_rows = []
    log_entries = [["File", "Severity", "Message"]]
    analysis_data = []

    if os.path.exists(ANALYSIS_JSON):
        with open(ANALYSIS_JSON, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
    else:
        print(f"    [WARNING] Analysis file not found (API offline/blocked). Falling back to regex-only scanning for first 3 pages of all PDFs.")
        if os.path.exists(INCLUDED_DIR):
            for file in os.listdir(INCLUDED_DIR):
                if file.lower().endswith(".pdf"):
                    analysis_data.append({"file": file, "text_pages": [1, 2, 3]})

    if not analysis_data:
        print("No analysis data or PDFs found to process.")
        return

    for item in analysis_data:
        filename = item["file"]
        pdf_path = os.path.join(INCLUDED_DIR, filename)
        
        if not os.path.exists(pdf_path):
            continue
            
        text_pages = item.get("text_pages", [])
        if not text_pages:
            continue
            
        try:
            with fitz.open(pdf_path) as doc:
                # Store all candidates for this PDF
                global_candidates = {attr: [] for attr in USER_ATTRIBUTES}
                
                for page_num in text_pages:
                    if 0 <= page_num - 1 < len(doc):
                        page = doc[page_num - 1]
                        page_text = page.get_text()
                        if page_text.strip():
                            p_candidates = extract_text_data(page_text, page_num, filename)
                            for attr, val, score, proof in p_candidates:
                                matched_attr = next((a for a in USER_ATTRIBUTES if a.lower() == attr.lower()), None)
                                if matched_attr:
                                    global_candidates[matched_attr].append((val, score, proof))
                
                # Pick unique candidates for each attribute
                row = {col: "" for col in COLUMNS}
                row["File"] = filename
                found_any = False
                
                # Combine all candidates from all pages into a single pool
                pool = []
                for attr, matches in global_candidates.items():
                    for val, score, proof in matches:
                        pool.append({"attr": attr, "val": val, "score": score, "proof": proof})
                
                # Sort pool by score desc
                pool.sort(key=lambda x: x["score"], reverse=True)
                
                consumed_vals = set() # (val, proof) pair as unique ID for a data point
                assigned_attrs = set()
                proofs = []

                # Replace global greedy assignment (Gap 3)
                # Group by attribute instead of sorting all (Case-insensitive match to prevent KeyErrors)
                attr_pools = {attr: [] for attr in USER_ATTRIBUTES}
                for cand in pool:
                    if cand["score"] > 0:
                        matched_attr = next((a for a in USER_ATTRIBUTES if a.lower() == cand["attr"].lower()), None)
                        if matched_attr:
                            cand["attr"] = matched_attr
                            attr_pools[matched_attr].append(cand)
                
                consumed_vals = set() 
                assigned_attrs = set()
                proofs = []

                # Performance metrics where the best sample has the HIGHEST value
                MAX_VALUE_ATTRS = {
                    "specific capacitance", "energy density",
                    "specific surface area", "pore volume", "micropore volume",
                    "micropore surface area",
                }

                # Find the absolute best candidate for *each* attribute independently
                for attr, candidates in attr_pools.items():
                    if not candidates: continue
                    
                    # Sort within this attribute group by score first
                    candidates.sort(key=lambda x: x["score"], reverse=True)
                    
                    # For performance metrics: among candidates with score within 20% of top,
                    # pick the one with the highest numeric value (best sample result)
                    if attr.lower() in MAX_VALUE_ATTRS:
                        top_score = candidates[0]["score"]
                        score_threshold = max(top_score * 0.8, top_score - 50)
                        competitive = [c for c in candidates if c["score"] >= score_threshold]
                        def _numeric(c):
                            try: return float(str(c["val"]).split("-")[0].split(",")[0].strip())
                            except: return -1
                        competitive.sort(key=_numeric, reverse=True)
                        best_cand = competitive[0]
                        print(f"  [MAX-VALUE] {attr}: selected {best_cand['val']} (from {len(competitive)} competitive candidates)")
                    else:
                        best_cand = candidates[0]
                    
                    # Strip the attribute name from the proof to use the pure line location as the uniqueness key
                    pure_location = best_cand["proof"].split(" → ")[0] if " → " in best_cand["proof"] else best_cand["proof"]
                    val_id = (best_cand["val"], pure_location)
                    
                    # Rule 4: One value = One attribute binding
                    # If this value was already claimed by an attribute, skip it
                    if val_id not in consumed_vals:
                         row[best_cand["attr"]] = best_cand["val"]
                         consumed_vals.add(val_id)
                         assigned_attrs.add(best_cand["attr"])
                         proofs.append(best_cand["proof"])
                         found_any = True
                         print(f"[BEST PROOF] {filename} | {best_cand['attr']}={best_cand['val']} (score {best_cand['score']})")

                if found_any:
                    row["Proof"] = ", ".join(list(dict.fromkeys(proofs)))
                    all_rows.append(row)
                    
        except Exception as e:
            log_entries.append([filename, "ERROR", f"Failed to read PDF: {e}"])
            continue

    # Deduplicate
    unique_rows = []
    seen = set()
    for r in all_rows:
        row_tuple = tuple(str(r.get(col, "")) for col in COLUMNS)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(r)

    # Save CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(unique_rows)
        
    with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(log_entries)
        
    print(f"Text mining complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extracting Text")
    parser.add_argument("--workspace", type=str, help="Workspace directory for this session")
    parser.add_argument("--attributes", type=str, help="Comma-separated attributes to extract")
    args = parser.parse_args()
    
    setup_paths(args.workspace, args.attributes)
    main()
