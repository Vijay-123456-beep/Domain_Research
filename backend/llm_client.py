import os
import json
import time
import requests
import re
from typing import List, Dict, Any, Optional

# Load .env manually if needed (some scripts run directly)
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=False)

# ── Global rate-limit tracking ──────────────────────────────────────────────
# Per-provider cooldown end-times. If time.time() < _PROVIDER_COOLDOWN[name],
# we skip that provider until it recovers.
_PROVIDER_COOLDOWN: Dict[str, float] = {}

# Minimum guaranteed gap between *any* two API calls (free-tier safe)
_MIN_API_INTERVAL = 4.0
_LAST_CALL_TIME = 0.0
# ─────────────────────────────────────────────────────────────────────────────


def clean_json_response(content: str) -> str:
    """Clean and extract valid JSON from LLM response that may have garbage or markdown."""
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    content = content.strip()

    brace_count = 0
    bracket_count = 0
    start_idx = -1
    end_idx = -1

    for i, char in enumerate(content):
        if char == '{':
            if brace_count == 0 and bracket_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and bracket_count == 0 and start_idx != -1:
                end_idx = i + 1
                break
        elif char == '[':
            if brace_count == 0 and bracket_count == 0:
                start_idx = i
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if brace_count == 0 and bracket_count == 0 and start_idx != -1:
                end_idx = i + 1
                break

    if start_idx != -1 and end_idx != -1:
        return content[start_idx:end_idx]
    return content


def call_llm(messages: List[Dict], max_tokens: int = 1024, temperature: float = 0.1, model_override: str = None) -> Optional[Any]:
    """
    Centralized LLM caller with automatic provider fallback/rotation.
    Rotates through Groq -> Gemini -> Together AI -> OpenRouter.

    Features:
      - Per-provider exponential cooldown on 429 / 402 errors
      - Global minimum gap between requests (_MIN_API_INTERVAL)
      - Smart provider skipping when still in cooldown window
      - Correct Gemini model for v1beta/openai endpoint (gemini-2.0-flash)
    """
    global _LAST_CALL_TIME, _PROVIDER_COOLDOWN

    providers = [
        {
            "name": "Groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key": os.getenv("GROQ_API_KEY", ""),
            "model": "llama-3.3-70b-versatile",
            "json_mode": True,
            "base_cooldown": 12,   # seconds to cool after first 429
        },
        {
            "name": "Gemini",
            "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "key": os.getenv("GEMINI_API_KEY", ""),
            # gemini-2.0-flash: confirmed working on v1beta/openai endpoint, generous free quota
            "model": "gemini-2.0-flash",
            "json_mode": False,
            "base_cooldown": 8,
        },
        {
            "name": "Together AI",
            "url": "https://api.together.xyz/v1/chat/completions",
            "key": os.getenv("TOGETHER_API_KEY", ""),
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "json_mode": False,
            "base_cooldown": 20,
        },
        {
            "name": "OpenRouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key": os.getenv("OPENROUTER_API_KEY", ""),
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "json_mode": False,
            "base_cooldown": 15,
        }
    ]

    # Filter providers with a configured API key
    active_providers = [p for p in providers if p["key"] and p["key"].strip() != ""]

    if not active_providers:
        print("[LLM_CLIENT] Warning: NO API KEYS FOUND in environment.")
        return None

    # Handle model_override (e.g. for vision models)
    if model_override:
        if "/" in model_override:
            or_provider = next((p for p in active_providers if p["name"] == "OpenRouter"), None)
            if or_provider:
                active_providers = [{**or_provider, "model": model_override}]
        else:
            for p in active_providers:
                p["model"] = model_override

    for attempt, provider in enumerate(active_providers):
        pname = provider["name"]

        # ── Skip provider if it's still in its cooldown window ────────────────
        cooldown_until = _PROVIDER_COOLDOWN.get(pname, 0)
        remaining_cooldown = cooldown_until - time.time()
        if remaining_cooldown > 0:
            print(f"[LLM_CLIENT] Skipping {pname} – cooling down for {remaining_cooldown:.0f}s more.")
            continue

        # ── Enforce global minimum gap between any two API calls ───────────────
        now = time.time()
        elapsed = now - _LAST_CALL_TIME
        if elapsed < _MIN_API_INTERVAL:
            time.sleep(_MIN_API_INTERVAL - elapsed)
        _LAST_CALL_TIME = time.time()

        try:
            payload = {
                "model": provider["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            if provider.get("json_mode"):
                payload["response_format"] = {"type": "json_object"}

            hdrs = {
                "Authorization": f"Bearer {provider['key']}",
                "Content-Type": "application/json"
            }

            print(f"\033[1m\033[93m[LLM_CLIENT] Attempt {attempt+1}: Calling {pname} ({provider['model']})...\033[0m")

            response = requests.post(provider["url"], headers=hdrs, json=payload, timeout=60)

            # ── Rate limited or out of credits → exponential backoff ───────────
            if response.status_code in [429, 402]:
                reason = "rate limited" if response.status_code == 429 else "credit limit exceeded"
                prev_remaining = max(0, _PROVIDER_COOLDOWN.get(pname, time.time()) - time.time())
                # Double the cooldown each successive hit, cap at 5 minutes
                new_cooldown = min(300, max(provider["base_cooldown"], prev_remaining * 2))
                _PROVIDER_COOLDOWN[pname] = time.time() + new_cooldown
                print(f"[LLM_CLIENT] {pname} {reason} ({response.status_code}). Cooling down for {new_cooldown:.0f}s.")
                continue

            if response.status_code != 200:
                print(f"[LLM_CLIENT] {pname} error {response.status_code}: {response.text[:200]}")
                continue

            # ── Success → clear this provider's cooldown ───────────────────────
            _PROVIDER_COOLDOWN.pop(pname, None)

            res_data = response.json()
            raw_content = res_data['choices'][0]['message'].get('content')
            if not raw_content:
                print(f"[LLM_CLIENT] {pname} returned empty content. Trying next provider...")
                continue

            content = raw_content.strip()

            try:
                cleaned_content = clean_json_response(content)
                parsed_json = json.loads(cleaned_content)
                print(f"\033[92m[LLM_CLIENT] {pname} responded successfully.\033[0m")
                return parsed_json
            except (json.JSONDecodeError, ValueError):
                # Non-JSON response (e.g. plain text answer) – return as-is
                return content

        except requests.exceptions.Timeout:
            print(f"[LLM_CLIENT] {pname} timed out after 60s.")
        except Exception as e:
            print(f"[LLM_CLIENT] Exception with {pname}: {e}")
            time.sleep(2 + attempt)

    print("[LLM_CLIENT] All providers exhausted or cooling down. Returning None (heuristic fallback will activate).")
    return None


if __name__ == "__main__":
    res = call_llm([{"role": "user", "content": "Return 'Hello' in JSON format: {\"msg\": \"Hello\"}"}])
    print(f"Test Result: {res}")
