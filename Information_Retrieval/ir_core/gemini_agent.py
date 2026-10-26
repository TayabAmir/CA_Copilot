"""Optional Gemini agent for extracting source metadata.

The agent is deliberately isolated from ingestion: its output is advisory
metadata, while the local parser remains the source of truth for indexing.
"""
import json
import os
import urllib.error
import urllib.request

from .documents import _pdf_text, from_json, parse_file


MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _local_metadata(filename, text):
    name = filename.rsplit("/", 1)[-1]
    lower = f"{name} {text[:4000]}".lower()
    return {
        "title": "Income Tax Ordinance, 2001" if "income" in lower and "ordinance" in lower else name,
        "law": "Income Tax Ordinance, 2001" if "income tax ordinance" in lower else None,
        "source_type": "Ordinance" if "ordinance" in lower else None,
        "tax_year": None,
        "effective_date": None,
        "amended_up_to": None,
        "jurisdiction": "Pakistan" if "pakistan" in lower or "fbr" in lower else None,
        "confidence": 0.35,
        "source": "local fallback",
    }


def _source_text(filename, data):
    if filename.lower().endswith(".pdf"):
        return _pdf_text(data)
    return data.decode("utf-8", "replace")


def analyze_source(filename, data):
    """Extract metadata with Gemini, falling back explicitly to local heuristics."""
    text = _source_text(filename, data)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        result = _local_metadata(filename, text)
        result["warning"] = "GEMINI_API_KEY is not configured; local metadata fallback was used."
        return result

    prompt = """You are a tax-document metadata extraction agent.
Return ONLY valid JSON with exactly these keys:
title, law, source_type, tax_year, effective_date, amended_up_to, jurisdiction, confidence.
Use null when a value is not explicitly supported by the source. Do not invent dates.
confidence must be a number from 0 to 1.

SOURCE FILENAME:
{filename}

SOURCE TEXT:
{text}
""".format(filename=filename, text=text[:60000])
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    request = urllib.request.Request(
        API_URL.format(model=os.getenv("GEMINI_MODEL", MODEL)) + "?key=" + api_key,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode())
        raw = body["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(raw)
        result["source"] = "Gemini"
        return result
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError,
            TypeError, ValueError, TimeoutError) as exc:
        result = _local_metadata(filename, text)
        result["warning"] = f"Gemini metadata extraction failed; local fallback was used: {exc}"
        return result


def structure_source(filename, data):
    """Return parsed Documents, using Gemini hierarchy when available."""
    text = _source_text(filename, data)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return parse_file(filename, data), "local fallback: GEMINI_API_KEY is not configured"

    prompt = """You are a tax-document structuring agent.
Convert the source into a faithful hierarchical JSON document. Preserve the source
wording; do not summarize, invent, or omit legal text.
Return ONLY valid JSON with this shape:
{
  "title": "document title",
  "id": "short stable id",
  "meta": {"Law": null, "Source Type": null, "Tax Year": null,
           "Effective Date": null, "Amended Up To": null, "Jurisdiction": null},
  "sections": [
    {"heading": "Chapter or Section heading", "text": "...",
     "children": [
       {"heading": "Subsection or Paragraph heading", "text": "...", "children": []}
     ]}
  ]
}
Use empty strings or arrays when a value is unavailable. Keep the hierarchy
to levels supported by the source: chapter, section, subsection, paragraph,
proviso, and explanation. Do not invent headings.

SOURCE FILENAME:
""" + filename + """

SOURCE TEXT:
""" + text[:120000]
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    request = urllib.request.Request(
        API_URL.format(model=os.getenv("GEMINI_MODEL", MODEL)) + "?key=" + api_key,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode())
        raw = body["candidates"][0]["content"]["parts"][0]["text"]
        structured = json.loads(raw)
        if not isinstance(structured, dict) or not isinstance(structured.get("sections"), list):
            raise ValueError("Gemini returned no valid sections array")
        structured.setdefault("meta", {})
        structured.setdefault("title", filename)
        structured.setdefault("id", "")
        docs = from_json(structured, filename)
        if not docs or not docs[0].nodes:
            raise ValueError("Gemini returned an empty document structure")
        return docs, "Gemini structured ingestion"
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError,
            TypeError, ValueError, TimeoutError, json.JSONDecodeError) as exc:
        return parse_file(filename, data), f"local fallback: Gemini structuring failed ({exc})"
