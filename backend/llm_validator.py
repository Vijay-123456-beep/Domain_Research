import os
import json
import requests
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
import time

from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=False)

from llm_client import call_llm

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.3-70b-instruct:free"
MAX_TOKENS = 3500  # Leave buffer for system prompt and output

try:
    # Use standard cl100k_base which is compatible with GPT-4 models
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tokenizer = None


def count_tokens(text: str) -> int:
    """Fallback token counting if tiktoken fails."""
    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text.split()) * 1.3  # Rough heuristic


def build_schema_context(attributes: List[str], schema_loader) -> str:
    """Build a precise context block so the LLM knows what each attribute means."""
    if not schema_loader:
        return ", ".join(attributes)

    context_lines = []
    for attr in attributes:
        attr_data = schema_loader.schema.get(attr)
        if not attr_data:
            # Try lowercase lookup if necessary
            for key, val in schema_loader.schema.items():
                if key.lower() == attr.lower():
                    attr_data = val
                    break
        
        if attr_data:
            units = ", ".join(attr_data.get('units', []))
            aliases = ", ".join(attr_data.get('aliases', []))
            context_lines.append(f"Attribute: {attr}")
            context_lines.append(f"  - Units: {units}")
            context_lines.append(f"  - Aliases: {aliases}")
            context_lines.append("")
    
    return "\n".join(context_lines)

# Global throttle: minimum seconds between LLM API calls to respect rate limits
_LAST_LLM_CALL_TIME = 0.0
_LLM_MIN_INTERVAL = 3.0  # seconds between calls

def _throttle_llm():
    """Sleep if necessary to respect global rate limit interval."""
    global _LAST_LLM_CALL_TIME
    now = time.time()
    elapsed = now - _LAST_LLM_CALL_TIME
    if elapsed < _LLM_MIN_INTERVAL:
        time.sleep(_LLM_MIN_INTERVAL - elapsed)
    _LAST_LLM_CALL_TIME = time.time()

def call_openrouter_api(messages: List[Dict], retries=3, model_override: str=None) -> Optional[List[Dict]]:
    """Legacy wrapper for centralized LLM client."""
    return call_llm(messages, model_override=model_override)



def extract_measurements_batch(candidate_sentences: List[Dict[str, Any]], 
                               attributes: List[str], 
                               schema_loader) -> List[Dict[str, Any]]:
    """
    Sends a token-aware batch of sentences to the LLM for strict validation.
    Candidate sentence dict should look like:
    { "id": 1, "text": "The capacitance reached 245 F/g." }
    """
    if not candidate_sentences:
        return []

    schema_context = build_schema_context(attributes, schema_loader)
    
    system_prompt = f"""You are a strict scientific data extraction and validation engine. 
Your objective is to identify TRUE experimental measurements and map them to the correct predefined attribute.

[AVAILABLE ATTRIBUTES & SCHEMA]
{schema_context}

[STRICT INSTRUCTIONS]
1. EXTRACT ONLY TRUE MEASUREMENTS: A valid measurement is explicitly reported as a RESULT (e.g., "is", "was", "obtained").
2. DISAMBIGUATE CONDITIONS vs RESULTS (CRITICAL): Ignore experimental conditions (e.g., "measured at 1 A/g", "heated to 500 C"). If a unit describes a testing parameter (like Voltage, Current Density, Temperature) DO NOT extract it as the main measurement result unless explicitly requested by the schema.
3. REJECT NOISE: Reject sentences detailing trends ("increased to"), theoretical models ("calculated"), methods ("synthesized"), citations, years, or generic descriptions.
4. ZERO HALLUCINATION: You MUST ONLY extract numbers and units that appear EXACTLY in the provided sentence block. Do not imply or convert values.
5. CROSS-SENTENCE CONTEXT: You will receive 3-sentence blocks. The measurement is likely in the middle sentence, but its meaning/attribute might be defined in the previous or next sentence. READ ALL 3 to decide the attribute.
6. ONE VALUE, ONE ATTRIBUTE: Map the measurement to the single most appropriate attribute from the provided schema using Unit compatibility and Context.
7. MANDATORY OUTPUT FORMAT: You must return a JSON object with a "measurements" array.

[INPUT FORMAT]
You will receive context blocks formatted as:
[ID: X] Previous Sentence. Target Sentence with Number. Next Sentence.

[OUTPUT FORMAT]
Return a JSON object:
Return your response in JSON format.
{{
  "measurements": [
    {{
      "id": X, // MUST match the input ID
      "is_valid": true,
      "value": 245.0, // numeric value
      "unit": "F/g", // string unit
      "attribute": "Specific Capacitance", // mapped attribute from schema
      "confidence": 0.95, // 0.0 to 1.0
      "reason": "Explicit result statement with matching unit."
    }}
  ]
}}
* If a sentence contains MULTIPLE valid measurements, return multiple objects in the array with the SAME ID.
* If a sentence contains NO valid measurements (or is rejected), do NOT include it in the array.
"""

    results = []
    
    # Batch constraints
    current_batch = []
    current_tokens = count_tokens(system_prompt)
    
    def process_current_batch():
        if not current_batch: return []
        
        user_prompt = "Process the following sentences:\n\n"
        for item in current_batch:
            user_prompt += f"[ID: {item['id']}] {item['text']}\n"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[LLM_BATCH] Sending {len(current_batch)} sentences to LLM...")
        llm_output = call_openrouter_api(messages)
        
        if llm_output is None:
            # Fallback triggered: Return an empty list to signify reliance on legacy heuristic
             print(f"[LLM_WARN] Fallback triggered. Dropping LLM verification for batch.")
             return []
             
        valid_extracted = []
        
        # Handle dict wrapping vs direct list
        items_to_process = []
        if isinstance(llm_output, dict) and "measurements" in llm_output:
            items_to_process = llm_output["measurements"]
        elif isinstance(llm_output, list):
            items_to_process = llm_output
            
        for item in items_to_process:
            if isinstance(item, dict) and item.get("is_valid") == True and item.get("confidence", 0) >= 0.70:
                valid_extracted.append(item)
                
        return valid_extracted

    # Token-aware batching loop
    for sent_item in candidate_sentences:
        sent_cost = count_tokens(f"[ID: {sent_item['id']}] {sent_item['text']}\n")
        
        if current_tokens + sent_cost > MAX_TOKENS:
            # Flush batch
            results.extend(process_current_batch())
            current_batch = [sent_item]
            current_tokens = count_tokens(system_prompt) + sent_cost
        else:
            current_batch.append(sent_item)
            current_tokens += sent_cost

    # Flush remaining
    results.extend(process_current_batch())
    
    return results

    return mapping_results


def validate_table_headers_llm(headers_with_samples: List[Tuple[str, List[str]]], attributes: List[str], schema_loader) -> Dict[str, Dict]:
    """
    Analyzes table headers (and sample cell values) and maps them to schema attributes using OpenRouter.
    Returns: Dict mapping the original header string to {attribute, unit, confidence}
    """
    if not headers_with_samples or not attributes:
        return {}
        
    schema_context = build_schema_context(attributes, schema_loader)
    
    system_prompt = f"""You are a scientific table understanding system.
Your task is to interpret table column headers and determine their TRUE scientific meaning.

[AVAILABLE ATTRIBUTES & SCHEMA]
{schema_context}

[RULES]
1. Map each header to the single most appropriate attribute from the schema.
2. Be careful with derived or ambiguous terms (e.g. "retention" ≠ capacitance).
3. If a header is ambiguous (like "Value" or "Result"), infer the true attribute by looking at the provided Sample Data Values and their implied units.
4. Extract the expected scientific unit from the header (or sample data) if present.
5. If a header + sample data does not clearly map to an attribute, map it to "UNKNOWN_NOISE".
6. ZERO HALLUCINATION: You must process the exact header string provided, do not invent names.

[OUTPUT FORMAT]
Return a JSON object with a "headers" array. Return your response in JSON format.
{{
  "headers": [
    {{
      "original_header": "Capacitance (Csp) [F/g]",
      "mapped_attribute": "Specific Capacitance",
      "extracted_unit": "F/g",
      "confidence": 0.95,
      "reason": "Direct alias match and matching unit."
    }}
  ]
}}
"""
    
    user_prompt = "Interpret the following table headers (with sample data from their columns):\n"
    for h, samples in headers_with_samples:
        samples_str = ", ".join(samples[:3]) if samples else "None"
        user_prompt += f'- Header: "{h}" | Sample Data: [{samples_str}]\n'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"[LLM_TABLE] Sending {len(headers_with_samples)} headers to LLM for classification...")
    llm_output = call_openrouter_api(messages)
    
    mapping_results = {}
    if llm_output and isinstance(llm_output, list): # handle "headers" array
        for item in llm_output:
            orig = item.get("original_header")
            attr = item.get("mapped_attribute")
            if orig and attr and attr != "UNKNOWN_NOISE":
                mapping_results[orig] = {
                    "attribute": attr,
                    "unit": item.get("extracted_unit"),
                    "confidence": item.get("confidence", 0.0)
                }
    elif llm_output and isinstance(llm_output, dict) and "headers" in llm_output:
        for item in llm_output["headers"]:
            orig = item.get("original_header")
            attr = item.get("mapped_attribute")
            if orig and attr and attr != "UNKNOWN_NOISE":
                mapping_results[orig] = {
                    "attribute": attr,
                    "unit": item.get("extracted_unit"),
                    "confidence": item.get("confidence", 0.0)
                }
                
    return mapping_results


def classify_table_cells_batch_llm(cells: List[Dict], attribute: str, header_unit: str) -> List[Dict]:
    """
    Classifies a batch of ambiguous table cells as MEASUREMENT, CONDITION, or NOISE.
    Input cells dict format: {"id": 1, "value_str": "1 A/g"}
    """
    if not cells:
        return []

    system_prompt = f"""You are a scientific table cell validator.
You are evaluating ambiguous cells found under the column for attribute '{attribute}'.
The column header implied the unit: '{header_unit}'.

[TASK]
Classify each cell value strictly as:
1. MEASUREMENT → actual experimental result mapping exactly to '{attribute}' (KEEP)
2. CONDITION → experimental testing parameter like current density, temperature, scan rate (REJECT)
3. NOISE → general text or metadata description (REJECT)

[RULES]
1. ZERO HALLUCINATION: DO NOT modify the value.
2. Check if the cell's unit conflicts with the expected attribute. (e.g., "1 A/g" in a Capacitance column is a CONDITION).
3. Return JSON format.

[OUTPUT FORMAT]
Return your response in JSON format.
{{
  "classifications": [
    {{
      "id": X,
      "type": "CONDITION",
      "confidence": 0.99,
      "reason": "A/g designates a current density testing condition, not {attribute}."
    }}
  ]
}}
"""
    
    user_prompt = "Classify these cells:\n"
    for c in cells:
        user_prompt += f'[ID: {c["id"]}] Value: "{c["value_str"]}"\n'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    llm_output = call_openrouter_api(messages)
    
    valid_cells = []
    if llm_output and isinstance(llm_output, dict) and "classifications" in llm_output:
        for item in llm_output["classifications"]:
             # We only care if it's explicitly confirmed as a MEASUREMENT with high confidence
             if item.get("type") == "MEASUREMENT" and item.get("confidence", 0) >= 0.70:
                 valid_cells.append(item.get("id"))
    
    return valid_cells


def validate_table_row_llm(row_data: Dict, attributes: List[str], schema_loader) -> bool:
    """
    Validates a completely extracted table row to catch cross-column contradictions
    (e.g., Energy Density column actually holds Current Density conditions).
    """
    if not row_data:
         return False
         
    schema_context = build_schema_context(attributes, schema_loader)
    
    system_prompt = f"""You are validating extracted scientific table data row by row.

[AVAILABLE SCHEMA]
{schema_context}

Your task is to ensure CROSS-COLUMN CONSISTENCY.
Look at the combination of values extracted for this single row. Do they make logical sense?

[RULES]
1. Verify if the unit of a mapped value contradicts the mapped attribute (e.g., "1 A/g" mapped to "Energy Density" is a contradiction, it's a testing condition).
2. If ANY mapped attribute is fundamentally incorrect due to a unit contradiction, OR if the row contains values that are not the actual experimental results, you must mark the row as INVALID.
3. If the row contains a logical, compatible set of measurements for the given attributes, mark it VALID.
4. Return your response in JSON format.

[OUTPUT FORMAT]
{{
  "is_valid": true/false,
  "confidence": 0.95,
  "reason": "Detailed explanation of why row is accepted or rejected based on cross-column compatibility."
}}
"""
    
    summary_parts = []
    for k, v in row_data.items():
        if k.startswith("_"): continue
        if isinstance(v, dict):
            summary_parts.append(f"{k}: {v.get('value')} {v.get('unit', '')}")
            
    user_prompt = "Validate this extracted row mapping:\n" + "\n".join(summary_parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    llm_output = call_openrouter_api(messages)
    
    if llm_output and isinstance(llm_output, dict):
         if llm_output.get("is_valid") == True and llm_output.get("confidence", 0) >= 0.70:
             return True
         print(f"[LLM_TABLE_ROW_REJECT] Row rejected: {llm_output.get('reason', 'No reason given')}")
         return False
    # LLM unavailable (rate-limited/API failure) - return None so caller can fail-open
    print(f"[LLM_TABLE_ROW_SKIP] LLM unavailable, accepting row by default")
    return None

def evaluate_graph_image_llm(base64_image: str) -> Dict[str, Any]:
    """
    Evaluates a base64 encoded image to determine if it is a scientific graph
    containing plotted data and numeric tick labels, rejecting diagrams/SEM.
    """
    system_prompt = """You are analyzing a PDF page image crop.

Determine if the image contains a robust scientific graph. Return your response in JSON format.

Rules:
- Graph must have axes (X and Y)
- Must contain numeric tick labels
- Must contain plotted data (lines, curves, markers)

Reject:
- SEM/TEM images
- diagrams
- flowcharts
- chemical structures

Output Format (JSON):
{{
  "is_graph": true/false,
  "confidence": 0.0 to 1.0,
  "reason": "Brief explanation of identification features",
  "x_axis_label": "The extracted text of the X axis label if present (e.g. 'Power Density (W/kg)')",
  "y_axis_label": "The extracted text of the Y axis label if present (e.g. 'Energy Density (Wh/kg)')"
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the features from this image and return the response in JSON format."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    llm_output = call_openrouter_api(messages, model_override="nvidia/nemotron-nano-12b-v2-vl:free")
    
    if llm_output and isinstance(llm_output, dict):
        return {
            "is_graph": bool(llm_output.get("is_graph", False)),
            "confidence": float(llm_output.get("confidence", 0.0)),
            "x_axis_label": llm_output.get("x_axis_label", ""),
            "y_axis_label": llm_output.get("y_axis_label", "")
        }
    
    return {"is_graph": False, "confidence": 0.0, "x_axis_label": "", "y_axis_label": ""}
