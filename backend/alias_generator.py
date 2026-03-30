import re
import os
import json
import time
from llm_client import call_llm


def generate_aliases(attributes_str, workspace_path):
    attributes = [a.strip() for a in attributes_str.split(",") if a.strip()]
    if not attributes:
        return {}

    prompt = f"""You are a scientific text mining expert.

For each attribute, generate a COMPLETE alias set for detecting it in research papers.

For each attribute, include ALL of the following:

1. Exact name (unchanged)
2. Underscore format (snake_case)
3. Core simplified term (e.g., "capacitance", "volume")
4. Scientific variants (e.g., "gravimetric capacitance")
5. Common abbreviations (e.g., SC, PV)
6. Symbolic forms (e.g., C, Cs, V)
7. Domain-specific variations used in literature

Rules:
- DO NOT exclude the original attribute name
- DO NOT be minimal — maximize recall
- Include both strict and relaxed matches
- Avoid generic unrelated words
- Output ONLY JSON

INPUT ATTRIBUTES:
{", ".join(attributes)}

Format:
{{
  "attribute_name": {{
    "aliases": ["...", "..."],
    "units": ["...", "..."],
    "range": [min_value, max_value]
  }}
}}

CRITICAL: For the 'range' field, research the typical scientific bounds for this attribute in the given domain. Use [null, null] only if the attribute is purely categorical (like 'material name'). Otherwise, provide plausible numeric limits to prevent data extraction errors.
"""

    messages = [{"role": "user", "content": prompt}]
    
    print(f"Generating aliases for {len(attributes)} attributes...")
    alias_map = call_llm(messages)
    
    if alias_map and isinstance(alias_map, dict):
        # Validate and remap structure
        # The LLM sometimes changes casing of attribute names — do a case-insensitive remap
        # to align the LLM-returned keys back onto the original user-provided attribute names.
        remapped = {}
        def normalize_key(k):
            return re.sub(r'[^a-z0-9]', '', k.lower())
            
        alias_map_norm = {normalize_key(k): k for k in alias_map.keys()}
        
        for attr in attributes:
            attr_norm = normalize_key(attr)
            llm_key = alias_map_norm.get(attr_norm)
            
            entry = alias_map.get(llm_key, {}) if llm_key else {}
            if not isinstance(entry, dict):
                entry = {}
            
            attr_lower = attr.lower()
            raw_aliases = [a for a in entry.get("aliases", []) if isinstance(a, str) and a.strip()]
            raw_units = [u for u in entry.get("units", []) if isinstance(u, str) and u.strip()]

            # Deduplicate (case-insensitive) and filter out the attribute name itself
            seen_aliases = set()
            clean_aliases = []
            for a in raw_aliases:
                a_lower = a.strip().lower()
                if a_lower != attr_lower and a_lower not in seen_aliases:
                    seen_aliases.add(a_lower)
                    clean_aliases.append(a.strip())
            
            seen_units = set()
            clean_units = []
            for u in raw_units:
                u_lower = u.strip().lower()
                if u_lower not in seen_units:
                    seen_units.add(u_lower)
                    clean_units.append(u.strip())

            raw_range = entry.get("range", [])
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raw_range = None

            remapped[attr] = {
                "aliases": clean_aliases, 
                "units": clean_units,
                "range": raw_range
            }
        
        alias_map = remapped
        
        # Save to workspace
        os.makedirs(workspace_path, exist_ok=True)
        output_path = os.path.join(workspace_path, "aliases.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alias_map, f, indent=4)
        
        print(f"Generated aliases for {len(attributes)} attributes.")
        return alias_map

    
    # Fallback: create empty aliases structure
    print("All retries failed or LLM returned invalid format. Creating fallback empty aliases.")
    fallback = {}
    for attr in attributes:
        fallback[attr] = {"aliases": [], "units": [], "range": None}
    
    os.makedirs(workspace_path, exist_ok=True)
    output_path = os.path.join(workspace_path, "aliases.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fallback, f, indent=4)
    
    return fallback


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True)
    args = parser.parse_args()
    
    generate_aliases(args.attributes, args.workspace)