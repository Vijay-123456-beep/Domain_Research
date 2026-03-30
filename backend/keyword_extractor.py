import sys
import os
import json
import re
from schema_loader import load_domain_schema
from llm_client import call_llm

# Build domain alias vocabulary at module load time (schema-grounded, not hardcoded)
_DOMAIN_ALIASES = set()
try:
    _schema = load_domain_schema()
    if _schema and hasattr(_schema, 'schema'):
        for v in _schema.schema.values():
            for alias in v.get("aliases", []):
                _DOMAIN_ALIASES.add(alias.lower())
except Exception:
    pass

def extract_keywords(query):
    """
    Extracts structured technical keyword phrases from a research query 
    using OpenRouter API with a fallback to local N-Grams.
    Returns: {"primary": [str], "secondary": [str]}
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not found in environment. Falling back to local N-Gram heuristic.", file=sys.stderr)
        return fallback_extraction(query)

    try:
        prompt = f"""
        Extract highly relevant technical keywords or short phrases from the following research query for use in an academic paper screening engine.

        CRITICAL RULES:
        1. PHRASE-FIRST STRATEGY: Prioritize bigrams/trigrams (e.g. "carbon supercapacitor", "specific capacitance") over single words (e.g. "carbon").
        2. UNIT-AWARE KEYWORDS: Include heavily used quantitative units associated with the domain natively if applicable (e.g. "F/g", "Wh/kg", "A/g", "mAh/g").
        3. DOMAIN ENFORCEMENT: Reject any keywords not explicitly related to the scientific domain. Avoid generic CS terms like "machine learning model" or "performance analysis" unless strongly anchored (e.g. "biomass ML model").
        4. Do NOT output variations of the exact same phrase multiple times. 

        OUTPUT FORMAT:
        You must output ONLY a valid JSON object matching this exact structure:
        {{
            "primary": ["must-match phrase 1", "must-match phrase 2"...],
            "secondary": ["optional phrase 1", "unit phrase 2"...]
        }}
        
        Example Query: "solid-state electrolytes for Li-ion batteries"
        Example Output: {{"primary": ["solid state electrolyte", "lithium ion battery", "superionic conductor"], "secondary": ["Li ion conductivity", "all solid state battery", "ionic transport"]}}
        
        Query: {query}
        """
        
        messages = [{"role": "user", "content": prompt}]
        raw_data = call_llm(messages)
        
        if not raw_data:
            print("Warning: LLM Extraction failed or returned no data. Falling back to local N-Gram heuristic.", file=sys.stderr)
            return fallback_extraction(query)

        # Handle the case where the LLM might return a dictionary directly (parsed by call_llm)
        if isinstance(raw_data, dict):
            keywords_dict = raw_data
        else:
            # Fallback if call_llm returned a string instead of parsed JSON
            try:
                keywords_dict = json.loads(raw_data)
            except:
                keywords_dict = {"primary": [], "secondary": []}


        # Post-Processing: Clean, Deduplicate (Case Insensitive), and Domain Verify
        def clean_and_dedup(lst, query_words):
             seen = set()
             res = []
             has_domain = False
             
             if not isinstance(lst, list): return res, False
             
             for kw in lst:
                  if not isinstance(kw, str): continue
                  k = kw.strip()
                  if len(k) < 3: continue
                  if any(bad in k.lower() for bad in ['let me know', 'here ', 'output', 'note:', 'as follow']):
                      continue
                  
                  if k.lower() not in seen:
                       seen.add(k.lower())
                       res.append(k)
                       # Domain check 1: substring match with query words
                       q_match = any(qw in k.lower() for qw in query_words) or any(k.lower() in qw for qw in query_words)
                       # Domain check 2: alias vocabulary match (schema-grounded)
                       a_match = any(alias in k.lower() or k.lower() in alias for alias in _DOMAIN_ALIASES)
                       if q_match or a_match:
                            has_domain = True
             return res, has_domain

        stop_words = {"for", "the", "and", "with", "from", "using", "based", "that", "this", "model", "analysis", "study"}
        query_words = [w.lower() for w in query.split() if len(w) > 3 and w.lower() not in stop_words]
        
        primary_clean, p_domain = clean_and_dedup(keywords_dict.get("primary", []), query_words)
        secondary_clean, s_domain = clean_and_dedup(keywords_dict.get("secondary", []), query_words)
        
        # Domain Safety Enforcement (Fix 3.4)
        if not (p_domain or s_domain):
             print("Warning: LLM Keywords failed domain verification. Triggering Fallback logic.", file=sys.stderr)
             return fallback_extraction(query)
            
        final_dict = {
            "primary": primary_clean[:6],   # Capped at 4-6 phrases
            "secondary": secondary_clean[:5] # Capped at 3-5 phrases
        }
        
        if not final_dict["primary"] and not final_dict["secondary"]:
            return fallback_extraction(query)
            
        return final_dict
        
    except Exception as e:
        print(f"OpenRouter Extraction Error: {e}", file=sys.stderr)
        return fallback_extraction(query)

def generate_ngrams(words, n):
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

def fallback_extraction(query):
    """
    Offline local generation of primary/secondary phrases using N-Grams (Fix 3.5).
    """
    stop_words = {"for", "the", "and", "with", "from", "using", "based", "that", "this", "model", "analysis", "study", "system", "data", "method"}
    words = [w.strip() for w in query.split() if w.strip().lower() not in stop_words and len(w) > 2]
    
    primary = []
    secondary = []
    
    if len(words) >= 3:
        primary.extend(generate_ngrams(words, 3))
    if len(words) >= 2:
        primary.extend(generate_ngrams(words, 2))
        
    secondary.extend(words)
    
    # Case-Insensitive Deduplication
    def dedup(lst):
        seen = set()
        res = []
        for item in lst:
            if item.lower() not in seen:
                seen.add(item.lower())
                res.append(item)
        return res
        
    return {
        "primary": dedup(primary)[:8],
        "secondary": dedup(secondary)[:8]
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python keyword_extractor.py '<query>'")
        sys.exit(1)
        
    query = sys.argv[1]
    # Return structured JSON serialization to the CLI standard output
    print(json.dumps(extract_keywords(query), indent=2))
