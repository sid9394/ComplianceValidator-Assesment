"""
check_tools.py
These are the 10 compliance check functions that the validator agent uses.
Each one is wrapped with @tool so CrewAI can give it to the agent as a callable tool.

A1, A2 - document authenticity (format check and duplicate detection)
B1, B7 - GST compliance (GSTIN validation and CGST/SGST/IGST consistency)
C1, C2 - arithmetic (line item maths and subtotal check)
D1, D2 - TDS compliance (these two actually use the LLM, everything else is pure logic)
E1, E3 - policy checks (PO tolerance and approved vendor list)

A few things I kept consistent across all of them:
- inputs are always strings because of how CrewAI passes tool arguments
- they always return a JSON string and never crash - always return something even on errors
- all config values come from config/config.py
"""

import json
import os
import re
from datetime import datetime
import time
import requests
import litellm
from crewai.tools import tool
from difflib import SequenceMatcher
from dotenv import load_dotenv

from config.config import (
    TDS_SECTIONS_FILE,
    MOCK_API_BASE_URL,
    MOCK_API_HEADERS,
    MOCK_API_TIMEOUT,
    MOCK_API_ENDPOINTS,
    ACTIVE_MODEL,
    ACTIVE_MAX_TOKENS,
    ACTIVE_TEMPERATURE,
    LLM_PROVIDER,
    BUYER_TURNOVER_PREVIOUS_FY,
    ARITHMETIC_TOLERANCE_RS,
    PO_TOLERANCE_PERCENT,
    RETRY_WAIT_SECONDS,
    REQUEST_DELAY_SECONDS,
    ENABLE_RESPONSE_CACHE,
    ENABLE_GSTIN_CACHE,
    ENABLE_VENDOR_TDS_CACHE,
)

load_dotenv()

litellm.suppress_debug_info = True

# turn on litellm's in-memory cache so identical prompts don't make duplicate API calls
# the agent sometimes calls the same tool twice when it's figuring out what to do next
if ENABLE_RESPONSE_CACHE:
    litellm.cache = litellm.Cache(type="local")

# ============================================================
# setup
# ============================================================

GSTIN_REGEX_PATTERN = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"

# D1 stores the LLM result here, D2 reads from it so we only make one LLM call per invoice
_tds_cache: dict = {}

# if the same vendor appears in multiple invoices we can skip the LLM call and reuse the result
_vendor_tds_cache: dict = {}

# cache GSTIN lookups so we don't hit the mock API twice for the same GSTIN in one run
_gstin_cache: dict = {}


# ============================================================
# helpers
# ============================================================

_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y",
    "%d %B %Y", "%B %d, %Y", "%d/%m/%y", "%d-%m-%y",
]

def _parse_date(date_str: str):
    """Try to parse a date string in the formats Indian invoices commonly use."""
    if not date_str:
        return None
    s = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def fix_gstin_ocr(gstin: str) -> str:
    """
    Fix common OCR mistakes in GSTIN at the positions that should be digits.
    OCR often reads O as 0, I as 1, l as 1, etc. at positions 0, 1 and 9-12.
    Example: O7AABCG5678H1Z9 becomes 07AABCG5678H1Z9.
    """
    if not gstin:
        return ""
    chars = list(gstin.upper().strip())
    for pos in [0, 1, 9, 10, 11, 12]:
        if pos < len(chars):
            if chars[pos] == "O": chars[pos] = "0"
            if chars[pos] == "I": chars[pos] = "1"
            if chars[pos] == "l": chars[pos] = "1"
    return "".join(chars)


def find_vendor(gstin: str, registry: dict) -> dict:
    """Look up a vendor by GSTIN in the registry. Returns an empty dict if not found."""
    if not gstin:
        return {}
    for vendor in registry.get("vendors", []):
        if (vendor.get("gstin") or "").upper().strip() == gstin.upper().strip():
            return vendor
    return {}


def _field_similarity(a: str, b: str) -> float:
    """How similar two strings are, from 0.0 to 1.0."""
    return SequenceMatcher(None, str(a), str(b)).ratio()


def _invoice_similarity(gstin1: str, inv_num1: str, amount1: str,
                        gstin2: str, inv_num2: str, amount2: str) -> float:
    """Average similarity across the 3 fields we use to detect duplicate invoices."""
    scores = [
        _field_similarity(gstin1.upper().strip(), gstin2.upper().strip()),
        _field_similarity(inv_num1.strip(), inv_num2.strip()),
        _field_similarity(amount1, amount2),
    ]
    return sum(scores) / 3


def call_mock_api(endpoint_key: str, payload: dict) -> dict:
    """
    Call the mock GSTIN API with the auth header from config.
    Returns the response dict or an error dict if the server isn't running.
    """
    url = f"{MOCK_API_BASE_URL}{MOCK_API_ENDPOINTS[endpoint_key]}"
    try:
        response = requests.post(url, headers=MOCK_API_HEADERS, json=payload, timeout=MOCK_API_TIMEOUT)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "MOCK_API_UNAVAILABLE", "message": f"Start server: python mock_api/server.py", "available": False}
    except Exception as e:
        return {"error": str(e), "available": False}


def call_llm(prompt: str) -> str:
    """
    Makes an LLM call through LiteLLM. Retries automatically if we hit a rate limit.
    Which model it uses depends on LLM_PROVIDER in the .env file.
    """
    # small delay before each groq call so we don't go over 30 RPM
    if REQUEST_DELAY_SECONDS > 0:
        time.sleep(REQUEST_DELAY_SECONDS)

    for attempt, wait in enumerate(RETRY_WAIT_SECONDS, start=1):
        try:
            response = litellm.completion(
                model=ACTIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ACTIVE_MAX_TOKENS,
                temperature=ACTIVE_TEMPERATURE,
                caching=ENABLE_RESPONSE_CACHE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "429" in err or "rate limit" in err or "resource exhausted" in err
            if is_rate_limit:
                if attempt < len(RETRY_WAIT_SECONDS):
                    print(f"  [{LLM_PROVIDER}] Rate limit hit. Waiting {wait}s (attempt {attempt}/{len(RETRY_WAIT_SECONDS)})...")
                    time.sleep(wait)
                else:
                    raise Exception(f"Rate limit exceeded after {len(RETRY_WAIT_SECONDS)} retries: {e}")
            else:
                raise e
    raise Exception("LLM call failed")


# maps each vendor type to which TDS sections could apply to them
_VENDOR_TYPE_TO_SECTIONS: dict = {
    "IT_SERVICES":           {"194J"},
    "PROFESSIONAL_SERVICES": {"194J"},
    "CONTRACTOR":            {"194C"},
    "INDIVIDUAL_CONTRACTOR": {"194C"},
    "TRANSPORT":             {"194C"},
    "COMMISSION_AGENT":      {"194H"},
    "BROKER":                {"194H"},
    "RENT":                  {"194I"},
    "GOODS_SUPPLIER":        {"194Q"},
    "FOREIGN_VENDOR":        {"195"},
}

_DOMESTIC_COUNTRIES = {"", "india", "in"}


def _filter_tds_sections(all_sections: list, vendor_record: dict, invoice: dict) -> list:
    """
    Figures out which TDS sections are actually relevant for this vendor and invoice,
    so I don't have to send the entire tds_sections.json to the LLM every time.

    How it picks:
    1. map the vendor type to its likely sections
    2. always include whatever section is explicitly set in the registry in case the vendor type is wrong
    3. if it's a foreign vendor, only send section 195
    4. always include 206AB because it's a rate modifier that can apply on top of any section
    5. if the vendor type is unrecognised, send everything just to be safe
    """
    vendor_type      = (vendor_record.get("vendor_type") or "").upper().strip()
    explicit_section = (vendor_record.get("tds_section") or "").upper().strip()
    vendor_country   = (
        vendor_record.get("country") or
        invoice.get("vendor", {}).get("country") or ""
    ).lower().strip()

    needed: set = set()

    if vendor_type in _VENDOR_TYPE_TO_SECTIONS:
        needed |= _VENDOR_TYPE_TO_SECTIONS[vendor_type]

    # always honour the registry's explicit section - catches misclassified vendors
    if explicit_section and explicit_section not in ("N/A", "NONE"):
        needed.add(explicit_section)

    # foreign vendor: 195 only - domestic sections don't apply to non-residents
    is_foreign = (
        vendor_type == "FOREIGN_VENDOR" or
        (vendor_country and vendor_country not in _DOMESTIC_COUNTRIES)
    )
    if is_foreign:
        needed = {"195"}

    # unknown vendor type with no explicit section - send everything so no rules get missed
    if not needed:
        return all_sections

    # 206AB modifies the rate on any section, always include it
    needed.add("206AB")

    available = {s["section"] for s in all_sections}
    to_include = needed & available
    return [s for s in all_sections if s["section"] in to_include]


def _vendor_tds_key(vendor_record: dict, invoice: dict) -> str:
    """
    Makes a cache key for reusing TDS results across invoices from the same vendor.
    I include the amount bracket because some TDS thresholds are amount-dependent.
    Brackets are: low (under 30k), mid (under 1L), high (over 1L).
    """
    amount = float(invoice.get("total_amount", 0))
    bracket = "low" if amount <= 30_000 else "mid" if amount <= 100_000 else "high"
    return "|".join([
        (vendor_record.get("vendor_id") or vendor_record.get("vendor_type") or "UNKNOWN"),
        str(vendor_record.get("section_206ab_applicable", False)),
        str(bool(vendor_record.get("lower_deduction_cert"))),
        (vendor_record.get("country") or ""),
        bracket,
    ])


def get_tds_from_llm(invoice: dict, vendor_record: dict) -> dict:
    """
    This is the single LLM call that powers both D1 and D2.
    D1 calls this and the result gets cached. D2 reads from that cache so there's no
    second LLM call. If the same vendor appeared in a previous invoice, the cross-invoice
    cache skips the LLM call entirely.

    I only send the TDS sections relevant to this vendor type, not the whole file.
    """
    invoice_id = invoice.get("invoice_id", "unknown")

    # D2 will just read this from the cache
    if invoice_id in _tds_cache:
        return _tds_cache[invoice_id]

    # same vendor appeared in a previous invoice, reuse that result
    if ENABLE_VENDOR_TDS_CACHE:
        vkey = _vendor_tds_key(vendor_record, invoice)
        if vkey in _vendor_tds_cache:
            cached = _vendor_tds_cache[vkey]
            _tds_cache[invoice_id] = cached
            print(f"  [TDS cache] Reusing vendor TDS result for key: {vkey}")
            return cached

    # Build trimmed TDS rules context - only fields the LLM needs
    tds_context = {}
    if TDS_SECTIONS_FILE.exists():
        with open(TDS_SECTIONS_FILE) as f:
            raw = json.load(f)

        all_trimmed_sections = []
        keep_rate_fields = ["rate", "rate_individual", "rate_company",
                            "rate_professional", "rate_technical",
                            "rate_building", "rate_machinery", "rate_no_pan"]
        for s in raw.get("tds_sections", []):
            entry = {k: s[k] for k in ["section", "applicable_to", "tds_on_gst",
                                        "notes", "exceptions", "classification_rules",
                                        "sub_sections", "dtaa_countries"]
                     if s.get(k) is not None}
            entry["threshold"] = s.get("threshold") or s.get("single_payment_threshold")
            for rf in keep_rate_fields:
                if rf in s:
                    entry[rf] = s[rf]
            all_trimmed_sections.append(entry)

        tds_context = {
            "tds_sections": _filter_tds_sections(all_trimmed_sections, vendor_record, invoice),
            "special_rules": raw.get("special_rules", {})
        }

    # Build trimmed invoice and vendor - only TDS-relevant fields
    v = invoice.get("vendor", {})
    trimmed_invoice = {
        "invoice_id":    invoice.get("invoice_id"),
        "invoice_date":  invoice.get("invoice_date"),
        "total_amount":  invoice.get("total_amount", 0),
        "subtotal":      invoice.get("subtotal", 0),
        "gst_under_rcm": invoice.get("gst_under_rcm", False),
        "igst_amount":   invoice.get("igst_amount", 0),
        "cgst_amount":   invoice.get("cgst_amount", 0),
        "vendor":        {"name": v.get("name"), "gstin": v.get("gstin"),
                          "pan": v.get("pan"), "country": v.get("country")},
        "line_items":    [{"description": i.get("description"), "amount": i.get("amount")}
                          for i in invoice.get("line_items", [])]
    }

    trimmed_vendor = {k: vendor_record.get(k) for k in [
        "vendor_id", "legal_name", "vendor_type", "tds_section", "pan", "status",
        "lower_deduction_cert", "section_206ab_applicable",
        "withholding_tax_rate", "tax_treaty_country", "form_10f_available", "country"
    ] if vendor_record.get(k) is not None}

    prompt = f"""You are an Indian TDS compliance expert.

TDS RULES:
{json.dumps(tds_context, separators=(',', ': '))}

BUYER TURNOVER: Rs {BUYER_TURNOVER_PREVIOUS_FY:,} (above Rs 10 Crore, so 194Q applies)

VENDOR:
{json.dumps(trimmed_vendor, indent=1)}

INVOICE:
{json.dumps(trimmed_invoice, indent=1)}

Determine TDS for this invoice. Check all of these:
- Which section applies for this vendor type and service?
- Company=2% or individual=1% for 194C
- Technical=2% or professional=10% for 194J
- Lower deduction certificate in vendor record reduces the rate
- section_206ab_applicable=true in vendor record doubles rate, minimum 5%
- GTA transport vendor charging GST forward = 194C exempt
- 194I rent uses GROSS amount including GST for TDS base, all others use subtotal
- Foreign vendor = section 195, use DTAA rate if country available
- No PAN = 20% rate
- LDC valid for 194C, 194J, 194I, 194H only

Respond ONLY with this JSON, no other text:
{{
  "d1": {{
    "tds_applicable": true or false,
    "tds_section": "194C/194J/194H/194I/194Q/195/N/A",
    "confidence": 0.0 to 1.0,
    "reasoning": "one sentence"
  }},
  "d2": {{
    "tds_section": "same as d1",
    "tds_rate": 0.0,
    "tds_base_amount": 0.0,
    "tds_amount": 0.0,
    "base_amount_note": "subtotal or gross and why",
    "special_rules_applied": [],
    "vendor_classification": "company or individual",
    "service_classification": "technical or professional or N/A",
    "confidence": 0.0 to 1.0,
    "reasoning": "step by step"
  }}
}}"""

    # Default fallback if LLM fails
    fallback = lambda reason, code: {
        "d1": {"tds_applicable": False, "tds_section": code, "confidence": 0.10, "reasoning": reason},
        "d2": {"tds_section": code, "tds_rate": 0.0, "tds_base_amount": 0.0, "tds_amount": 0.0,
               "base_amount_note": "", "special_rules_applied": [], "vendor_classification": "",
               "service_classification": "", "confidence": 0.10, "reasoning": reason}
    }

    try:
        raw = call_llm(prompt).replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        result = fallback(f"JSON parse failed: {e}", "UNKNOWN")
    except Exception as e:
        result = fallback(f"LLM call failed: {e}", "ERROR")

    _tds_cache[invoice_id] = result
    if ENABLE_VENDOR_TDS_CACHE:
        _vendor_tds_cache[_vendor_tds_key(vendor_record, invoice)] = result
    return result


def clear_tds_cache(invoice_id: str):
    """Call this after each invoice is done so the per-invoice cache doesn't grow forever."""
    _tds_cache.pop(invoice_id, None)


# ============================================================
# A - document authenticity
# ============================================================

@tool("Check Invoice Number Format")
def check_invoice_format(invoice_number: str) -> str:
    """Use this tool to check if the invoice number looks like a valid Indian invoice number (A1 check).
    Input: invoice_number as a plain string.
    Returns: JSON with check_id A1, passed, confidence, finding."""
    try:
        if not invoice_number:
            return json.dumps({"check_id": "A1", "passed": False, "confidence": 0.99,
                               "finding": "Invoice number is missing or null"})
        num = invoice_number.strip()

        # Indian invoice numbers don't follow one standard format so I check against a few patterns
        patterns = [
            r"^[A-Z0-9]{2,10}[-/][A-Z0-9]{2,10}[-/][0-9]{4}[-/][0-9]{3,8}$",
            r"^[A-Z]{2,5}[-/][0-9]{4}[-/][0-9]{4,6}$",
            r"^[A-Z0-9/-]{5,30}$",
        ]
        is_valid = any(re.match(p, num.upper()) for p in patterns)

        return json.dumps({
            "check_id":   "A1",
            "passed":     is_valid,
            "confidence": 0.90,
            "finding":    f"Invoice number '{num}' format {'valid' if is_valid else 'INVALID'}",
            "evidence":   {"invoice_number": num}
        })
    except Exception as e:
        return json.dumps({"check_id": "A1", "passed": False, "confidence": 0.20,
                           "finding": f"Error: {e}"})


@tool("Check for Duplicate Invoice")
def check_duplicate_invoice(invoice_number: str, vendor_gstin: str, seen_invoices_json: str,
                            invoice_amount: str = "0", invoice_date: str = "") -> str:
    """Use this tool to check if this invoice has been seen before in this batch (A2 check).
    Stores each seen invoice as GSTIN::invoice_number::amount::date.
    Only compares entries within a 365-day window. Flags exact duplicates and near-duplicates above 95% similarity.
    Input: invoice_number, vendor_gstin, seen_invoices_json as a JSON array of those key strings,
    invoice_amount as a plain string (total_amount from invoice),
    invoice_date as YYYY-MM-DD string (invoice_date from invoice).
    Returns: JSON with check_id A2, passed, confidence, finding."""
    try:
        if not vendor_gstin or not invoice_number:
            return json.dumps({"check_id": "A2", "passed": False, "confidence": 0.50,
                               "finding": "invoice_number or vendor_gstin is missing - cannot check duplicates"})

        seen       = json.loads(seen_invoices_json) if seen_invoices_json else []
        gstin_c    = (vendor_gstin or "").strip().upper()
        inv_c      = (invoice_number or "").strip()
        amount_c   = str(invoice_amount or "0").strip()

        current_date = _parse_date(invoice_date)

        is_exact   = False
        near_match = None   # {"similarity": float, "entry": str, "seen_inv_num": str}

        for entry in seen:
            parts = entry.split("::")
            if len(parts) < 2:
                continue
            s_gstin   = parts[0]
            s_inv_num = parts[1]
            s_amount  = parts[2] if len(parts) >= 3 else "0"
            s_date    = parts[3].strip() if len(parts) >= 4 else ""

            # 365-day window gate: skip entries outside the window when both dates are known
            if current_date and s_date:
                seen_date = _parse_date(s_date)
                if seen_date and abs((current_date - seen_date).days) > 365:
                    continue

            # Exact match: all 3 fields identical
            if s_gstin == gstin_c and s_inv_num == inv_c and s_amount == amount_c:
                is_exact = True
                break

            # Near-duplicate: average similarity across all 3 fields >= 0.95
            sim = _invoice_similarity(gstin_c, inv_c, amount_c, s_gstin, s_inv_num, s_amount)
            if sim >= 0.95 and (near_match is None or sim > near_match["similarity"]):
                near_match = {"similarity": round(sim, 3), "entry": entry, "seen_inv_num": s_inv_num}

        if is_exact:
            return json.dumps({
                "check_id":   "A2",
                "passed":     False,
                "confidence": 1.00,
                "finding":    "EXACT DUPLICATE - same vendor, invoice number, and amount already seen in this batch",
                "evidence":   {"type": "exact", "gstin": gstin_c,
                               "invoice_number": inv_c, "total_seen": len(seen)}
            })

        if near_match:
            return json.dumps({
                "check_id":   "A2",
                "passed":     False,
                "confidence": 0.90,
                "finding":    (f"NEAR-DUPLICATE - {near_match['similarity']*100:.1f}% similar to "
                               f"previously seen invoice {near_match['seen_inv_num']} "
                               f"(threshold 95% across vendor_gstin, invoice_number, invoice_amount)"),
                "evidence":   {"type": "near_duplicate", "similarity": near_match["similarity"],
                               "matched_entry": near_match["entry"], "total_seen": len(seen)}
            })

        return json.dumps({
            "check_id":   "A2",
            "passed":     True,
            "confidence": 1.00,
            "finding":    "No duplicate found",
            "evidence":   {"gstin": gstin_c, "invoice_number": inv_c, "total_seen": len(seen)}
        })
    except Exception as e:
        return json.dumps({"check_id": "A2", "passed": False, "confidence": 0.20,
                           "finding": f"Error: {e}"})


# ============================================================
# B - GST compliance
# ============================================================

@tool("Validate GSTIN Format and Active Status")
def check_gstin_format(vendor_gstin: str) -> str:
    """Use this tool to validate the vendor GSTIN (B1 check).
    Fixes OCR errors first, then checks the format, then checks the mock API for active status.
    Input: vendor_gstin as a plain string.
    Returns: JSON with check_id B1, passed, confidence, finding."""
    try:
        if not vendor_gstin:
            return json.dumps({"check_id": "B1", "passed": False, "confidence": 0.99,
                               "finding": "vendor_gstin is missing or null"})
        gstin = fix_gstin_ocr(vendor_gstin)

        # format check
        if not re.match(GSTIN_REGEX_PATTERN, gstin):
            return json.dumps({
                "check_id":   "B1",
                "passed":     False,
                "confidence": 0.98,
                "finding":    f"GSTIN '{gstin}' does not match the 15-char format",
                "evidence":   {"original": vendor_gstin, "normalized": gstin}
            })

        # check active status via the mock API, use the cache if we already looked this one up
        if ENABLE_GSTIN_CACHE and gstin in _gstin_cache:
            return _gstin_cache[gstin]

        api    = call_mock_api("validate_gstin", {"gstin": gstin})
        api_ok = "available" not in api or api.get("available", True)
        status = api.get("status", "UNKNOWN" if not api_ok else "ACTIVE")
        active = api.get("valid", True) and status not in ["SUSPENDED", "CANCELLED", "INACTIVE"]
        passed = active

        result = json.dumps({
            "check_id":   "B1",
            "passed":     passed,
            "confidence": 0.95 if api_ok else 0.70,
            "finding":    f"GSTIN '{gstin}' is {'valid, status: ' + status if passed else 'FAILED - status: ' + status}",
            "evidence":   {"original": vendor_gstin, "normalized": gstin,
                           "api_status": status, "api_available": api_ok,
                           "legal_name": api.get("legal_name", "")}
        })

        if ENABLE_GSTIN_CACHE and api_ok:
            _gstin_cache[gstin] = result

        return result
    except Exception as e:
        return json.dumps({"check_id": "B1", "passed": False, "confidence": 0.20,
                           "finding": f"Error: {e}"})


@tool("Check GST Rate Consistency CGST SGST IGST")
def check_gst_rate_consistency(invoice_json: str) -> str:
    """Use this tool to check whether the right GST type was used on this invoice (B7 check).
    Inter-state transactions should use IGST only. Intra-state should use CGST plus SGST. Never both together.
    Input: full invoice as a JSON string.
    Returns: JSON with check_id B7, passed, confidence, finding."""
    try:
        if not invoice_json or not invoice_json.strip():
            return json.dumps({"check_id": "B7", "passed": False, "confidence": 0.20,
                               "finding": "invoice_json is empty - pass the full invoice JSON string from the extractor output"})
        inv  = json.loads(invoice_json)
        cgst = float(inv.get("cgst_rate", 0))
        sgst = float(inv.get("sgst_rate", 0))
        igst = float(inv.get("igst_rate", 0))
        cgst_amt = float(inv.get("cgst_amount", 0))
        sgst_amt = float(inv.get("sgst_amount", 0))

        has_cgst_sgst = cgst > 0 or sgst > 0
        has_igst      = igst > 0

        # RCM means the buyer pays the GST, so the rate check doesn't apply here
        if inv.get("gst_under_rcm") or inv.get("rcm_applicable"):
            return json.dumps({"check_id": "B7", "passed": True, "confidence": 0.95,
                               "finding": "RCM invoice - buyer pays GST, rate check skipped",
                               "evidence": {"rcm": True}})

        # foreign vendors don't charge Indian GST
        vendor = inv.get("vendor", {})
        if not vendor.get("gstin") and vendor.get("country"):
            return json.dumps({"check_id": "B7", "passed": True, "confidence": 0.95,
                               "finding": "Foreign vendor - Indian GST not applicable",
                               "evidence": {"foreign_vendor": True}})

        # mixing CGST/SGST and IGST on the same invoice is always wrong
        if has_cgst_sgst and has_igst:
            return json.dumps({"check_id": "B7", "passed": False, "confidence": 0.99,
                               "finding": "INVALID - invoice mixes CGST/SGST with IGST",
                               "evidence": {"cgst_rate": cgst, "sgst_rate": sgst, "igst_rate": igst}})

        # intra-state: CGST and SGST must be equal
        if has_cgst_sgst:
            passed = abs(cgst - sgst) < 0.01 and abs(cgst_amt - sgst_amt) < 1.0
            return json.dumps({"check_id": "B7", "passed": passed, "confidence": 0.97,
                               "finding": f"Intra-state: CGST {cgst}% {'equals' if passed else 'does NOT equal'} SGST {sgst}%",
                               "evidence": {"type": "intra-state", "cgst_rate": cgst, "sgst_rate": sgst}})

        # inter-state: IGST only
        if has_igst:
            return json.dumps({"check_id": "B7", "passed": True, "confidence": 0.97,
                               "finding": f"Inter-state: IGST at {igst}% applied correctly",
                               "evidence": {"type": "inter-state", "igst_rate": igst}})

        # no GST at all - could be an exempt supply
        return json.dumps({"check_id": "B7", "passed": True, "confidence": 0.70,
                           "finding": "No GST on invoice - verify if exempt",
                           "evidence": {"cgst": cgst, "sgst": sgst, "igst": igst}})
    except Exception as e:
        return json.dumps({"check_id": "B7", "passed": False, "confidence": 0.20,
                           "finding": f"Error: {e}"})


# ============================================================
# C - arithmetic
# ============================================================

@tool("Check Line Item Arithmetic")
def check_line_item_arithmetic(line_items_json: str) -> str:
    """Use this tool to verify that quantity x rate = amount for every line item (C1 check).
    Allows Rs 1 rounding tolerance.
    Input: line_items array as a JSON string.
    Returns: JSON with check_id C1, passed, confidence, errors list, finding."""
    try:
        if not line_items_json or not line_items_json.strip():
            return json.dumps({"check_id": "C1", "passed": False, "confidence": 0.20,
                               "finding": "line_items_json is empty - pass the line_items array as a JSON string"})
        items = json.loads(line_items_json)
        if not items:
            return json.dumps({"check_id": "C1", "passed": False, "confidence": 0.50,
                               "finding": "No line items found"})

        errors = []
        for i, item in enumerate(items, start=1):
            try:
                qty  = float(item.get("quantity", 0))
                rate = float(item.get("rate", item.get("unit_price", 0)))
                amt  = float(item.get("amount", 0))
                exp  = round(qty * rate, 2)
                if abs(exp - amt) > ARITHMETIC_TOLERANCE_RS:
                    errors.append({"line": i, "desc": str(item.get("description", ""))[:50],
                                   "expected": exp, "actual": amt, "diff": round(amt - exp, 2)})
            except (ValueError, TypeError) as e:
                errors.append({"line": i, "error": str(e)})

        passed = len(errors) == 0
        return json.dumps({
            "check_id":   "C1",
            "passed":     passed,
            "confidence": 1.00,
            "finding":    f"All {len(items)} line items correct" if passed else f"{len(errors)} arithmetic error(s)",
            "evidence":   {"total_items": len(items), "errors": errors}
        })
    except Exception as e:
        return json.dumps({"check_id": "C1", "passed": False, "confidence": 0.20,
                           "finding": f"Error: {e}"})


@tool("Check Subtotal Matches Sum of Line Items")
def check_subtotal_matches_line_items(invoice_json: str) -> str:
    """Use this tool to check that the subtotal equals the sum of all line item amounts (C2 check).
    Allows Rs 1 rounding tolerance.
    Input: full invoice as a JSON string.
    Returns: JSON with check_id C2, passed, confidence, finding."""
    try:
        if not invoice_json or not invoice_json.strip():
            return json.dumps({"check_id": "C2", "passed": False, "confidence": 0.20,
                               "finding": "invoice_json is empty - pass the full invoice JSON string from the extractor output"})
        inv        = json.loads(invoice_json)
        items      = inv.get("line_items", [])
        stated     = float(inv.get("subtotal", 0))
        calculated = round(sum(float(i.get("amount", 0)) for i in items), 2)
        diff       = round(abs(calculated - stated), 2)
        passed     = diff <= ARITHMETIC_TOLERANCE_RS

        return json.dumps({
            "check_id":   "C2",
            "passed":     passed,
            "confidence": 1.00,
            "finding":    f"Subtotal {'matches' if passed else 'MISMATCH'}: calculated Rs {calculated:,.2f} vs stated Rs {stated:,.2f}",
            "evidence":   {"calculated": calculated, "stated": stated, "diff": diff, "items": len(items)}
        })
    except Exception as e:
        return json.dumps({"check_id": "C2", "passed": False, "confidence": 0.20,
                           "finding": f"Error: {e}"})


# ============================================================
# D - TDS compliance
# D1 and D2 share one LLM call. D1 runs it and caches the result. D2 reads from the cache.
# ============================================================

@tool("Check TDS Applicability D1")
def check_tds_applicability(invoice_json: str, vendor_registry_json: str) -> str:
    """Use this tool to check whether TDS applies to this invoice (D1 check).
    Calls the LLM with the relevant TDS rules and vendor info. Result is cached for D2 to reuse.
    Input: invoice as a JSON string, vendor_registry as a JSON string.
    Returns: JSON with check_id D1, passed, tds_applicable, tds_section, confidence, finding."""
    try:
        if not invoice_json or not invoice_json.strip():
            return json.dumps({"check_id": "D1", "passed": False, "confidence": 0.20,
                               "tds_applicable": False, "finding": "invoice_json is empty - pass the full invoice JSON string from the extractor output"})
        invoice  = json.loads(invoice_json)
        registry = json.loads(vendor_registry_json)

        vendor_gstin = invoice.get("vendor", {}).get("gstin", "") or invoice.get("vendor_gstin", "")
        vendor       = find_vendor(vendor_gstin, registry)

        # the result of this gets cached so D2 doesn't have to call the LLM again
        llm = get_tds_from_llm(invoice, vendor)
        d1  = llm.get("d1", {})

        applicable = d1.get("tds_applicable", False)
        section    = d1.get("tds_section", "UNKNOWN")
        confidence = float(d1.get("confidence", 0.50))
        reasoning  = d1.get("reasoning", "")

        return json.dumps({
            "check_id":       "D1",
            "passed":         True,
            "confidence":     confidence,
            "tds_applicable": applicable,
            "tds_section":    section,
            "finding":        f"TDS {'APPLICABLE' if applicable else 'not applicable'} | Section: {section} | {reasoning}",
            "evidence":       {"tds_applicable": applicable, "tds_section": section, "reasoning": reasoning}
        })
    except Exception as e:
        return json.dumps({"check_id": "D1", "passed": False, "confidence": 0.10,
                           "tds_applicable": False, "finding": f"Error: {e}"})


@tool("Determine TDS Section and Calculate TDS Amount D2")
def check_tds_section_and_rate(invoice_json: str, vendor_registry_json: str) -> str:
    """Use this tool to figure out the TDS section, rate and amount for this invoice (D2 check).
    If D1 already ran, this reads from the cache and makes no extra LLM call.
    Also does a sanity check on the LLM's arithmetic because it sometimes gets the maths wrong.
    Input: invoice as a JSON string, vendor_registry as a JSON string.
    Returns: JSON with check_id D2, tds_section, tds_rate, tds_amount, confidence, finding."""
    try:
        if not invoice_json or not invoice_json.strip():
            return json.dumps({"check_id": "D2", "passed": False, "confidence": 0.20,
                               "tds_section": "UNKNOWN", "finding": "invoice_json is empty - pass the full invoice JSON string from the extractor output",
                               "requires_human_review": True})
        invoice  = json.loads(invoice_json)
        registry = json.loads(vendor_registry_json)

        vendor_gstin = invoice.get("vendor", {}).get("gstin", "") or invoice.get("vendor_gstin", "")
        vendor       = find_vendor(vendor_gstin, registry)

        # reads from cache if D1 already ran
        llm = get_tds_from_llm(invoice, vendor)
        d2  = llm.get("d2", {})

        section        = d2.get("tds_section", "UNKNOWN")
        rate           = float(d2.get("tds_rate", 0.0))
        base           = float(d2.get("tds_base_amount", 0.0))
        llm_amount     = float(d2.get("tds_amount", 0.0))
        base_note      = d2.get("base_amount_note", "")
        special_rules  = d2.get("special_rules_applied", [])
        confidence     = float(d2.get("confidence", 0.50))
        reasoning      = d2.get("reasoning", "")
        vendor_class   = d2.get("vendor_classification", "")
        service_class  = d2.get("service_classification", "")

        # double checking the LLM's maths since it can make arithmetic mistakes
        expected_amount = round(base * rate / 100, 2)
        math_is_correct = abs(expected_amount - llm_amount) <= ARITHMETIC_TOLERANCE_RS
        final_amount    = llm_amount if math_is_correct else expected_amount

        if not math_is_correct:
            special_rules.append(f"Math corrected: Rs {llm_amount:,.2f} -> Rs {expected_amount:,.2f}")
            confidence = min(confidence, 0.80)

        # Build finding from all parts
        parts = [f"Section {section}", f"Rate {rate}%", f"TDS Rs {final_amount:,.2f}", base_note]
        parts += special_rules
        if vendor_class:  parts.append(f"Vendor: {vendor_class}")
        if service_class: parts.append(f"Service: {service_class}")

        return json.dumps({
            "check_id":             "D2",
            "passed":               True,
            "confidence":           round(confidence, 3),
            "tds_section":          section,
            "tds_rate":             rate,
            "tds_base_amount":      base,
            "tds_amount":           final_amount,
            "arithmetic_verified":  math_is_correct,
            "finding":              " | ".join(p for p in parts if p),
            "evidence": {
                "tds_section":           section,
                "tds_rate":              rate,
                "tds_base_amount":       base,
                "tds_amount":            final_amount,
                "base_amount_note":      base_note,
                "special_rules_applied": special_rules,
                "vendor_classification": vendor_class,
                "service_classification": service_class,
                "reasoning":             reasoning,
                "arithmetic_verified":   math_is_correct
            }
        })
    except Exception as e:
        return json.dumps({"check_id": "D2", "passed": False, "confidence": 0.00,
                           "tds_section": "ERROR", "finding": f"Error: {e}",
                           "requires_human_review": True})


# ============================================================
# E - policy and business rules
# ============================================================

@tool("Check Invoice Amount Within PO Tolerance")
def check_po_amount_tolerance(invoice_json: str) -> str:
    """Use this tool to check if the invoice total is within the allowed tolerance of the PO amount (E1 check).
    Tolerance is 5% by default, set in config. If there's no PO amount on the invoice this check is skipped.
    Input: full invoice as a JSON string.
    Returns: JSON with check_id E1, passed, confidence, variance_percent, finding."""
    try:
        if not invoice_json or not invoice_json.strip():
            return json.dumps({"check_id": "E1", "passed": False, "confidence": 0.20,
                               "finding": "invoice_json is empty - pass the full invoice JSON string from the extractor output"})
        inv          = json.loads(invoice_json)
        total        = float(inv.get("total_amount", 0))
        po_amount = float(inv.get("po_amount") or 0)
        po_reference = inv.get("po_reference", inv.get("po_number", ""))

        # no PO amount means this check doesn't apply - skip it rather than failing
        if po_amount == 0:
            return json.dumps({
                "check_id":   "E1",
                "passed":     True,
                "confidence": 1.00,
                "skipped":    True,
                "finding":    "No PO amount on invoice - E1 skipped",
                "evidence":   {"po_reference": po_reference or "none"}
            })

        tolerance = po_amount * (PO_TOLERANCE_PERCENT / 100)
        diff      = abs(total - po_amount)
        variance  = round((diff / po_amount) * 100, 2)
        passed    = diff <= tolerance

        return json.dumps({
            "check_id":   "E1",
            "passed":     passed,
            "confidence": 0.99,
            "skipped":    False,
            "finding":    f"{'Within' if passed else 'EXCEEDS'} PO tolerance - variance {variance}% (limit {PO_TOLERANCE_PERCENT}%)",
            "evidence":   {"invoice_total": total, "po_amount": po_amount,
                           "variance_percent": variance, "tolerance_percent": PO_TOLERANCE_PERCENT,
                           "po_reference": po_reference}
        })
    except Exception as e:
        return json.dumps({"check_id": "E1", "passed": False, "confidence": 0.20,
                           "finding": f"Error: {e}"})


@tool("Check Vendor is on Approved Vendor List")
def check_approved_vendor(vendor_gstin: str, vendor_registry_json: str) -> str:
    """Use this tool to check if the vendor is in the approved vendor list and is currently ACTIVE (E3 check).
    Being in the registry is not enough - they also have to be ACTIVE, not SUSPENDED or CANCELLED.
    Input: vendor_gstin as a plain string, vendor_registry as a JSON string.
    Returns: JSON with check_id E3, passed, confidence, finding."""
    try:
        if not vendor_gstin:
            return json.dumps({"check_id": "E3", "passed": False, "confidence": 0.99,
                               "finding": "vendor_gstin is missing or null - cannot verify approved vendor list"})
        registry = json.loads(vendor_registry_json)
        gstin    = fix_gstin_ocr(vendor_gstin)
        vendor   = find_vendor(gstin, registry)

        if not vendor:
            return json.dumps({"check_id": "E3", "passed": False, "confidence": 0.99,
                               "finding": f"Vendor GSTIN '{gstin}' not in approved registry",
                               "evidence": {"gstin": gstin, "found": False}})

        status = vendor.get("status", "UNKNOWN")
        active = status == "ACTIVE"

        if not active:
            return json.dumps({"check_id": "E3", "passed": False, "confidence": 0.99,
                               "finding": f"Vendor '{vendor.get('legal_name', '')}' status is '{status}'",
                               "evidence": {"gstin": gstin, "status": status,
                                            "suspension_date": vendor.get("suspension_date", "N/A")}})

        return json.dumps({
            "check_id":   "E3",
            "passed":     True,
            "confidence": 1.00,
            "finding":    f"Vendor '{vendor.get('legal_name', '')}' is approved and ACTIVE",
            "evidence":   {"gstin": gstin, "vendor_id": vendor.get("vendor_id", ""),
                           "vendor_name": vendor.get("legal_name", ""),
                           "vendor_type": vendor.get("vendor_type", ""), "status": status}
        })
    except Exception as e:
        return json.dumps({"check_id": "E3", "passed": False, "confidence": 0.20,
                           "finding": f"Error: {e}"})
