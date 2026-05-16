# models/report_generator.py
import os
import requests


def generate_impression(findings_text: str) -> str:
    """
    Calls an external LLM endpoint to generate Impression from FindingsText.
    Set REPORT_LLM_URL in environment to enable.
    If not set, returns a placeholder string (so the website keeps working).
    """
    url = os.getenv("REPORT_LLM_URL", "").strip()
    if not url:
        return "Impression generation is not configured yet (REPORT_LLM_URL is missing)."

    timeout = float(os.getenv("REPORT_LLM_TIMEOUT", "60"))

    payload = {"findings_text": findings_text}

    # Optional: if you deploy an API with simple token auth
    token = os.getenv("REPORT_LLM_TOKEN", "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()

    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    # expected: {"impression": "..."}
    impression = (data.get("impression") or "").strip()
    mode = data.get("mode", "unknown")
    return impression or f"No impression returned from LLM (mode={mode})."
