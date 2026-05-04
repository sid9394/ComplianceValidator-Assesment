"""
check_tools.py
All 10 compliance checks as @tool functions for the Validator Agent.
A1, A2, B1, B7, C1, C2, E1, E3 are rule-based (regex, math, lookups).
D1 and D2 are fully LLM-powered using tds_sections.json as context.
Every function returns a JSON string and never crashes.
"""

import json
import os
import re
import requests
from pathlib import Path
from groq import Groq
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# SETUP
# ============================================================

# Setting up Groq client and loading config from .env
groq_client       = Groq(api_key=os.getenv("GROQ_API_KEY"))
MOCK_API_BASE_URL = os.getenv("MOCK_API_BASE_URL", "http://localhost:5000")
GROQ_MODEL        = "llama-3.3-70b-versatile"

# GSTIN is always 15 chars:
# 2 digit state code + 5 letter PAN name + 4 digit PAN serial
# + 1 letter entity type + 1 alphanumeric + Z + 1 checksum
GSTIN_REGEX_PATTERN = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fix_ocr_errors_in_gstin(raw_gstin: str) -> str:
    """
    GSTIN has specific digit-only positions (0,1 and 9-12).
    Fixing common OCR misreads O->0 and I->1 only at those positions
    so I don't accidentally change the letter parts of the GSTIN.
    Example: O7AABCG5678H1Z9 becomes 07AABCG5678H1Z9
    """
    cleaned_gstin = raw_gstin.upper().strip()
    gstin_chars   = list(cleaned_gstin)

    # Positions 0,1 are state code digits. Positions 9-12 are PAN serial digits.
    digit_only_positions = [0, 1, 9, 10, 11, 12]

    for position in digit_only_positions:
        if position < len(gstin_chars):
            if gstin_chars[position] == "O": gstin_chars[position] = "0"
            if gstin_chars[position] == "I": gstin_chars[position] = "1"
            if gstin_chars[position] == "l": gstin_chars[position] = "1"

    return "".join(gstin_chars)


def fix_ocr_errors_in_pan(raw_pan: str) -> str:
    """
    PAN format is AAAAA9999A - positions 5-8 must be digits.
    Same OCR fix as GSTIN but targeting PAN digit positions only.
    Example: AABCTI234F becomes AABCT1234F
    """
    cleaned_pan = raw_pan.upper().strip()
    pan_chars   = list(cleaned_pan)

    # Positions 5-8 are always digits in a PAN
    digit_only_positions = [5, 6, 7, 8]

    for position in digit_only_positions:
        if position < len(pan_chars):
            if pan_chars[position] == "O": pan_chars[position] = "0"
            if pan_chars[position] == "I": pan_chars[position] = "1"
            if pan_chars[position] == "l": pan_chars[position] = "1"

    return "".join(pan_chars)


def find_vendor_in_registry(vendor_gstin: str, vendor_registry: dict) -> dict:
    """
    Looping through vendor_registry.json vendors array
    to find a match by GSTIN. Returns empty dict if not found.
    """
    for vendor in vendor_registry.get("vendors", []):
        if vendor.get("gstin", "").upper().strip() == vendor_gstin.upper().strip():
            return vendor
    return {}


# ============================================================
# LLM HELPER FUNCTIONS - Used BY D1 AND D2 ONLY
# ============================================================

def load_file_as_text(file_path: str) -> str:
    """
    Loading a file and returning its contents as plain text.
    Returns empty string if file not found so the LLM call still works.
    """
    path = Path(file_path)
    if path.exists():
        with open(path) as f:
            return f.read()
    return ""


def run_tds_sanity_check(
    tds_base_amount: float,
    tds_rate: float,
    tds_amount_from_llm: float
) -> tuple:
    """
    Simple arithmetic check to catch LLM hallucinations on the TDS amount.
    Just verifying base * rate / 100 = amount. Allowing Rs 1 rounding difference.
    Returns (is_correct, expected_amount).
    """
    expected_amount = round(tds_base_amount * tds_rate / 100, 2)
    difference      = abs(expected_amount - tds_amount_from_llm)
    is_correct      = difference <= 1.0
    return is_correct, expected_amount


def call_llm_for_tds_determination(
    invoice: dict,
    vendor_record: dict,
    tds_rules_text: str,
    buyer_turnover: int
) -> dict:
    """
    Single LLM call that handles both D1 and D2 together.
    Passing the full tds_sections.json, vendor record, and invoice as context.
    The LLM reasons through all exceptions, special rules, and edge cases.
    Returns a structured dict with both D1 and D2 results.
    """
    vendor_context  = json.dumps(vendor_record, indent=2)
    invoice_context = json.dumps(invoice, indent=2)

    prompt = f"""You are an Indian TDS compliance expert working for FinanceGuard Solutions.

Your job is to analyze an invoice and determine:
1. Whether TDS applies (D1)
2. The correct TDS section, rate, base amount, and TDS amount (D2)

=== COMPLETE TDS RULES ===
{tds_rules_text}

=== BUYER DETAILS (FinanceGuard Solutions) ===
Previous FY Turnover: Rs {buyer_turnover:,}
This is important for Section 194Q which only applies if buyer turnover exceeds Rs 10 Crore.

=== VENDOR RECORD FROM REGISTRY ===
{vendor_context}

=== INVOICE TO ANALYZE ===
{invoice_context}

=== INSTRUCTIONS ===
Analyze this invoice carefully. Consider ALL of the following:

1. SECTION DETERMINATION: Which TDS section applies based on vendor type and service nature?

2. RATE DETERMINATION: What is the correct rate? Consider:
   - Is vendor a company or individual? (affects 194C rate)
   - Is service technical or professional? (affects 194J rate - technical=2%, professional=10%)
   - Does vendor have a valid Lower Deduction Certificate in their vendor record?
   - Is vendor PAN missing? (rate jumps to 20%)
   - Is vendor flagged under 206AB in their vendor record? (rate doubles, minimum 5%)

3. BASE AMOUNT: What amount should TDS be calculated on?
   - For 194I (rent): TDS on GROSS amount including GST
   - For all other sections: TDS on SUBTOTAL excluding GST

4. THRESHOLD CHECK: Is the invoice amount above the TDS threshold for this section?

5. EXCEPTIONS - Check for these carefully:
   - GTA (transport vendor) paying GST under forward charge means 194C TDS is exempt
   - Foreign vendor means Section 195, check DTAA country for reduced rate
   - 194Q only applies if buyer turnover exceeds Rs 10 Crore (it does here at Rs {buyer_turnover:,})
   - LDC only valid for sections 194C, 194J, 194I, 194H - not 194Q or 195
   - 206AB does not apply to sections 192, 192A, 194B, 194BB, 194LBC, 194N

6. COMPOSITE OR MIXED SUPPLIES: If invoice has multiple line items with different
   service types, determine the principal supply and apply that section.

7. RCM: If this is a Reverse Charge Mechanism invoice, the buyer pays GST.
   Note this in your reasoning.

Respond ONLY with valid JSON and nothing else. No explanation outside the JSON:
{{
  "d1": {{
    "tds_applicable": true or false,
    "tds_section": "194C or 194J or 194H or 194I or 194Q or 195 or N/A",
    "confidence": 0.0 to 1.0,
    "reasoning": "clear explanation of why TDS applies or does not"
  }},
  "d2": {{
    "tds_section": "same section as d1 or N/A",
    "tds_rate": 0.0,
    "tds_base_amount": 0.0,
    "tds_amount": 0.0,
    "base_amount_note": "explain whether subtotal or gross was used and why",
    "special_rules_applied": ["list any special rules like LDC, 206AB, GTA exception, no-PAN"],
    "vendor_classification": "company or individual - relevant for 194C",
    "service_classification": "technical or professional - relevant for 194J",
    "confidence": 0.0 to 1.0,
    "reasoning": "step by step explanation of how you arrived at this rate and amount"
  }}
}}"""

    try:
        llm_response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        raw_text = llm_response.choices[0].message.content.strip()
        # Stripping markdown fences if LLM wrapped the JSON
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_text)

    except json.JSONDecodeError as parse_error:
        # LLM returned something that is not valid JSON
        return {
            "error": f"LLM returned invalid JSON: {str(parse_error)}",
            "d1": {
                "tds_applicable": False,
                "tds_section":    "UNKNOWN",
                "confidence":     0.20,
                "reasoning":      "LLM response could not be parsed"
            },
            "d2": {
                "tds_section":           "UNKNOWN",
                "tds_rate":              0.0,
                "tds_base_amount":       0.0,
                "tds_amount":            0.0,
                "base_amount_note":      "Could not determine",
                "special_rules_applied": [],
                "vendor_classification": "",
                "service_classification": "",
                "confidence":            0.20,
                "reasoning":             "LLM response could not be parsed"
            }
        }

    except Exception as llm_error:
        # LLM call failed entirely
        return {
            "error": str(llm_error),
            "d1": {
                "tds_applicable": False,
                "tds_section":    "ERROR",
                "confidence":     0.10,
                "reasoning":      f"LLM call failed: {str(llm_error)}"
            },
            "d2": {
                "tds_section":           "ERROR",
                "tds_rate":              0.0,
                "tds_base_amount":       0.0,
                "tds_amount":            0.0,
                "base_amount_note":      "Could not determine",
                "special_rules_applied": [],
                "vendor_classification": "",
                "service_classification": "",
                "confidence":            0.10,
                "reasoning":             f"LLM call failed: {str(llm_error)}"
            }
        }


# ============================================================
# CATEGORY A - DOCUMENT AUTHENTICITY
# ============================================================

@tool("Check Invoice Number Format")
def check_invoice_format(invoice_number: str) -> str:
    """Im using this tool to validate the format of an invoice number.
    Input: invoice_number as a plain string.
    Returns: JSON string with check_id A1, passed, confidence, and finding."""
    try:
        cleaned_invoice_number = invoice_number.strip()

        # Indian invoice numbers come in many formats so checking against
        # multiple common patterns instead of one strict pattern
        valid_invoice_patterns = [
            r"^[A-Z0-9]{2,10}[-/][A-Z0-9]{2,10}[-/][0-9]{4}[-/][0-9]{3,8}$",
            r"^[A-Z]{2,5}[-/][0-9]{4}[-/][0-9]{4,6}$",
            r"^[A-Z0-9/-]{5,30}$",
        ]

        format_is_valid = any(
            re.match(pattern, cleaned_invoice_number.upper())
            for pattern in valid_invoice_patterns
        )

        return json.dumps({
            "check_id":   "A1",
            "passed":     format_is_valid,
            "confidence": 0.90,
            "finding": (
                f"Invoice number '{cleaned_invoice_number}' has a valid format"
                if format_is_valid
                else f"Invoice number '{cleaned_invoice_number}' does not match any known Indian invoice format"
            ),
            "evidence": {"invoice_number": cleaned_invoice_number}
        })

    except Exception as error:
        return json.dumps({
            "check_id": "A1", "passed": False, "confidence": 0.20,
            "finding":  f"Could not check invoice format: {str(error)}"
        })


@tool("Check for Duplicate Invoice")
def check_duplicate_invoice(
    invoice_number: str,
    vendor_gstin: str,
    already_seen_invoices_json: str
) -> str:
    """Im using this tool to detect if this invoice has already been processed in this batch.
    Building a unique key as GSTIN::invoice_number and checking against the seen list.
    Input: invoice_number string, vendor_gstin string,
           already_seen_invoices_json as a JSON array of GSTIN::invoice_number strings.
    Returns: JSON string with check_id A2, passed, confidence, and finding."""
    try:
        already_seen_list = json.loads(already_seen_invoices_json) if already_seen_invoices_json else []

        # Using GSTIN::invoice_number as unique key
        # Same invoice number from different vendors is allowed
        unique_invoice_key   = f"{vendor_gstin.upper().strip()}::{invoice_number.strip()}"
        invoice_is_duplicate = unique_invoice_key in already_seen_list

        return json.dumps({
            "check_id":   "A2",
            "passed":     not invoice_is_duplicate,
            "confidence": 1.00,
            "finding": (
                "DUPLICATE DETECTED - this invoice has already been processed"
                if invoice_is_duplicate
                else "No duplicate found - this is a new invoice"
            ),
            "evidence": {
                "unique_key":   unique_invoice_key,
                "is_duplicate": invoice_is_duplicate,
                "total_seen":   len(already_seen_list)
            }
        })

    except Exception as error:
        return json.dumps({
            "check_id": "A2", "passed": False, "confidence": 0.20,
            "finding":  f"Could not check for duplicates: {str(error)}"
        })


# ============================================================
# CATEGORY B - GST COMPLIANCE
# ============================================================

@tool("Validate GSTIN Format and Active Status")
def check_gstin_format(vendor_gstin: str) -> str:
    """Im using this tool to validate a vendor GSTIN.
    First fixing OCR errors, then checking 15-char format, then calling Mock API for active status.
    Input: vendor_gstin as a plain string.
    Returns: JSON string with check_id B1, passed, confidence, and finding."""
    try:
        # Fixing OCR errors before doing any validation
        gstin_after_ocr_fix   = fix_ocr_errors_in_gstin(vendor_gstin)
        gstin_format_is_valid = bool(re.match(GSTIN_REGEX_PATTERN, gstin_after_ocr_fix))

        if not gstin_format_is_valid:
            return json.dumps({
                "check_id":   "B1",
                "passed":     False,
                "confidence": 0.98,
                "finding":    f"GSTIN '{gstin_after_ocr_fix}' does not match the 15-char format",
                "evidence": {
                    "original_gstin":   vendor_gstin,
                    "normalized_gstin": gstin_after_ocr_fix,
                    "format_valid":     False
                }
            })

        # Calling Mock API to check if GSTIN is currently active
        try:
            api_response      = requests.post(
                f"{MOCK_API_BASE_URL}/api/gst/validate-gstin",
                json={"gstin": gstin_after_ocr_fix},
                timeout=5
            )
            api_result        = api_response.json()
            gstin_is_active   = api_result.get("valid", True)
            gstin_status_text = api_result.get("status", "UNKNOWN")
            api_was_available = True

        except Exception:
            # API unreachable - not failing the check but lowering confidence
            gstin_is_active   = True
            gstin_status_text = "API_UNAVAILABLE"
            api_was_available = False

        overall_passed = gstin_format_is_valid and gstin_is_active
        confidence     = 0.95 if api_was_available else 0.70

        return json.dumps({
            "check_id":   "B1",
            "passed":     overall_passed,
            "confidence": confidence,
            "finding": (
                f"GSTIN '{gstin_after_ocr_fix}' is valid and status is {gstin_status_text}"
                if overall_passed
                else f"GSTIN '{gstin_after_ocr_fix}' failed - status is {gstin_status_text}"
            ),
            "evidence": {
                "original_gstin":   vendor_gstin,
                "normalized_gstin": gstin_after_ocr_fix,
                "format_valid":     gstin_format_is_valid,
                "api_status":       gstin_status_text,
                "api_available":    api_was_available
            }
        })

    except Exception as error:
        return json.dumps({
            "check_id": "B1", "passed": False, "confidence": 0.20,
            "finding":  f"Could not validate GSTIN: {str(error)}"
        })


@tool("Check GST Rate Consistency CGST SGST IGST")
def check_gst_rate_consistency(invoice_json: str) -> str:
    """Im using this tool to check if the correct type of GST has been applied.
    Indian rule: inter-state uses IGST only, intra-state uses CGST plus SGST. Never both.
    Input: full invoice as a JSON string.
    Returns: JSON string with check_id B7, passed, confidence, and finding."""
    try:
        invoice = json.loads(invoice_json)

        cgst_rate   = float(invoice.get("cgst_rate",   0))
        sgst_rate   = float(invoice.get("sgst_rate",   0))
        igst_rate   = float(invoice.get("igst_rate",   0))
        cgst_amount = float(invoice.get("cgst_amount", 0))
        sgst_amount = float(invoice.get("sgst_amount", 0))

        invoice_has_cgst_or_sgst = (cgst_rate > 0 or sgst_rate > 0)
        invoice_has_igst         = (igst_rate > 0)
        rounding_tolerance       = 0.01

        # RCM invoices - buyer pays GST so rate check does not apply here
        if invoice.get("gst_under_rcm") or invoice.get("rcm_applicable"):
            return json.dumps({
                "check_id": "B7", "passed": True, "confidence": 0.95,
                "finding":  "RCM invoice - GST paid by buyer, rate check not applicable",
                "evidence": {"rcm": True}
            })

        # Foreign vendors do not charge Indian GST
        vendor_info = invoice.get("vendor", {})
        if not vendor_info.get("gstin") and vendor_info.get("country"):
            return json.dumps({
                "check_id": "B7", "passed": True, "confidence": 0.95,
                "finding":  "Foreign vendor - Indian GST not applicable",
                "evidence": {"foreign_vendor": True}
            })

        # Both CGST/SGST and IGST present at same time - this is invalid
        if invoice_has_cgst_or_sgst and invoice_has_igst:
            return json.dumps({
                "check_id":   "B7",
                "passed":     False,
                "confidence": 0.99,
                "finding":    "INVALID - invoice has both CGST/SGST and IGST which cannot be mixed",
                "evidence":   {"cgst_rate": cgst_rate, "sgst_rate": sgst_rate, "igst_rate": igst_rate}
            })

        # Intra-state: CGST must equal SGST in both rate and amount
        if invoice_has_cgst_or_sgst:
            rates_are_equal   = abs(cgst_rate - sgst_rate) < rounding_tolerance
            amounts_are_equal = abs(cgst_amount - sgst_amount) < 1.0
            check_passed      = rates_are_equal and amounts_are_equal

            return json.dumps({
                "check_id":   "B7",
                "passed":     check_passed,
                "confidence": 0.97,
                "finding": (
                    f"Intra-state: CGST {cgst_rate}% equals SGST {sgst_rate}% - correct"
                    if check_passed
                    else f"Intra-state: CGST {cgst_rate}% does not equal SGST {sgst_rate}%"
                ),
                "evidence": {
                    "invoice_type": "intra-state",
                    "cgst_rate":    cgst_rate,
                    "sgst_rate":    sgst_rate
                }
            })

        # Inter-state: only IGST present which is correct
        if invoice_has_igst:
            return json.dumps({
                "check_id":   "B7",
                "passed":     True,
                "confidence": 0.97,
                "finding":    f"Inter-state: IGST at {igst_rate}% applied correctly",
                "evidence":   {"invoice_type": "inter-state", "igst_rate": igst_rate}
            })

        # No GST at all - flagging for review, could be exempt
        return json.dumps({
            "check_id":   "B7",
            "passed":     True,
            "confidence": 0.70,
            "finding":    "No GST on this invoice - verify if exempt",
            "evidence":   {"cgst_rate": cgst_rate, "sgst_rate": sgst_rate, "igst_rate": igst_rate}
        })

    except Exception as error:
        return json.dumps({
            "check_id": "B7", "passed": False, "confidence": 0.20,
            "finding":  f"Could not check GST consistency: {str(error)}"
        })


# ============================================================
# CATEGORY C - ARITHMETIC CHECKS
# ============================================================

@tool("Check Line Item Arithmetic")
def check_line_item_arithmetic(line_items_json: str) -> str:
    """Im using this tool to verify quantity multiplied by rate equals amount for each line item.
    Input: line_items array as a JSON string.
    Returns: JSON string with check_id C1, passed, confidence, errors list, and finding."""
    try:
        line_items = json.loads(line_items_json)

        if not line_items:
            return json.dumps({
                "check_id": "C1", "passed": False, "confidence": 0.50,
                "finding":  "No line items found in invoice"
            })

        arithmetic_errors  = []
        rounding_tolerance = 1.0  # Allowing Rs 1 difference for rounding

        for line_number, line_item in enumerate(line_items, start=1):
            try:
                quantity        = float(line_item.get("quantity", 0))
                unit_rate       = float(line_item.get("rate", line_item.get("unit_price", 0)))
                stated_amount   = float(line_item.get("amount", 0))
                expected_amount = round(quantity * unit_rate, 2)
                difference      = abs(expected_amount - stated_amount)

                if difference > rounding_tolerance:
                    arithmetic_errors.append({
                        "line_number":     line_number,
                        "description":     str(line_item.get("description", ""))[:60],
                        "quantity":        quantity,
                        "unit_rate":       unit_rate,
                        "expected_amount": expected_amount,
                        "stated_amount":   stated_amount,
                        "difference":      round(stated_amount - expected_amount, 2)
                    })

            except (ValueError, TypeError) as parse_error:
                arithmetic_errors.append({
                    "line_number": line_number,
                    "error":       f"Could not parse line item: {str(parse_error)}"
                })

        check_passed = len(arithmetic_errors) == 0

        return json.dumps({
            "check_id":   "C1",
            "passed":     check_passed,
            "confidence": 1.00,
            "finding": (
                f"All {len(line_items)} line items have correct arithmetic"
                if check_passed
                else f"{len(arithmetic_errors)} line item(s) have arithmetic errors"
            ),
            "evidence": {
                "total_line_items":  len(line_items),
                "arithmetic_errors": arithmetic_errors
            }
        })

    except Exception as error:
        return json.dumps({
            "check_id": "C1", "passed": False, "confidence": 0.20,
            "finding":  f"Could not check line item arithmetic: {str(error)}"
        })


@tool("Check Subtotal Matches Sum of Line Items")
def check_subtotal_matches_line_items(invoice_json: str) -> str:
    """Im using this tool to verify the stated subtotal equals the sum of all line item amounts.
    Input: full invoice as a JSON string.
    Returns: JSON string with check_id C2, passed, confidence, and finding."""
    try:
        invoice            = json.loads(invoice_json)
        line_items         = invoice.get("line_items", [])
        stated_subtotal    = float(invoice.get("subtotal", 0))
        rounding_tolerance = 1.0

        # Summing all line item amounts and comparing to the stated subtotal
        calculated_subtotal = round(
            sum(float(item.get("amount", 0)) for item in line_items), 2
        )
        difference   = round(abs(calculated_subtotal - stated_subtotal), 2)
        check_passed = difference <= rounding_tolerance

        return json.dumps({
            "check_id":   "C2",
            "passed":     check_passed,
            "confidence": 1.00,
            "finding": (
                f"Subtotal matches: Rs {calculated_subtotal:,.2f}"
                if check_passed
                else f"Subtotal mismatch - calculated Rs {calculated_subtotal:,.2f} but invoice says Rs {stated_subtotal:,.2f} (diff Rs {difference:,.2f})"
            ),
            "evidence": {
                "calculated_subtotal": calculated_subtotal,
                "stated_subtotal":     stated_subtotal,
                "difference":          difference,
                "total_line_items":    len(line_items)
            }
        })

    except Exception as error:
        return json.dumps({
            "check_id": "C2", "passed": False, "confidence": 0.20,
            "finding":  f"Could not check subtotal: {str(error)}"
        })


# ============================================================
# CATEGORY D - TDS COMPLIANCE
# D1 and D2 are fully LLM-powered.
# One LLM call per invoice covers both checks together.
# The LLM reads tds_sections.json and reasons through all
# exceptions, special rules, and edge cases naturally.
# ============================================================

@tool("Check TDS Applicability D1")
def check_tds_applicability(invoice_json: str, vendor_registry_json: str) -> str:
    """Im using this tool to check if TDS applies to this invoice (D1 check).
    Uses LLM with full tds_sections.json context to handle all exceptions and edge cases.
    Input: invoice as JSON string, vendor_registry as JSON string.
    Returns: JSON string with check_id D1, passed, tds_applicable, confidence, and finding."""
    try:
        invoice         = json.loads(invoice_json)
        vendor_registry = json.loads(vendor_registry_json)

        # Getting vendor record from registry for LLM context
        vendor_data   = invoice.get("vendor", {})
        vendor_gstin  = vendor_data.get("gstin", "") or invoice.get("vendor_gstin", "")
        vendor_record = find_vendor_in_registry(vendor_gstin, vendor_registry)

        # Loading full TDS rules file to pass to LLM as context
        tds_rules_text = load_file_as_text("data/tds_sections.json")

        # FinanceGuard previous FY turnover from company_policy.yaml
        # This is Rs 15 Crore which is above the 194Q threshold of Rs 10 Crore
        buyer_turnover = 150000000

        # Single LLM call covers both D1 and D2 together
        llm_result = call_llm_for_tds_determination(
            invoice, vendor_record, tds_rules_text, buyer_turnover
        )

        # Extracting just the D1 portion of the LLM result
        d1_result      = llm_result.get("d1", {})
        tds_applicable = d1_result.get("tds_applicable", False)
        tds_section    = d1_result.get("tds_section", "UNKNOWN")
        confidence     = float(d1_result.get("confidence", 0.50))
        reasoning      = d1_result.get("reasoning", "")

        return json.dumps({
            "check_id":       "D1",
            "passed":         True,
            "confidence":     confidence,
            "tds_applicable": tds_applicable,
            "tds_section":    tds_section,
            "finding": (
                f"TDS {'APPLICABLE' if tds_applicable else 'not applicable'} | "
                f"Section: {tds_section} | "
                f"Reasoning: {reasoning}"
            ),
            "evidence": {
                "tds_applicable": tds_applicable,
                "tds_section":    tds_section,
                "reasoning":      reasoning
            }
        })

    except Exception as error:
        return json.dumps({
            "check_id":       "D1",
            "passed":         False,
            "confidence":     0.10,
            "tds_applicable": False,
            "finding":        f"Could not check TDS applicability: {str(error)}"
        })


@tool("Determine TDS Section and Calculate TDS Amount D2")
def check_tds_section_and_rate(invoice_json: str, vendor_registry_json: str) -> str:
    """Im using this tool to determine TDS section, rate, and calculate TDS amount (D2 check).
    Uses LLM with full tds_sections.json context including all exceptions.
    Handles GTA exemptions, DTAA rates, 206AB, LDC, no-PAN rates, composite supplies.
    Input: invoice as JSON string, vendor_registry as JSON string.
    Returns: JSON string with check_id D2, tds_section, tds_rate, tds_amount, and finding."""
    try:
        invoice         = json.loads(invoice_json)
        vendor_registry = json.loads(vendor_registry_json)

        # Getting vendor record from registry for LLM context
        vendor_data   = invoice.get("vendor", {})
        vendor_gstin  = vendor_data.get("gstin", "") or invoice.get("vendor_gstin", "")
        vendor_record = find_vendor_in_registry(vendor_gstin, vendor_registry)

        # Loading full TDS rules file to pass to LLM as context
        tds_rules_text = load_file_as_text("data/tds_sections.json")
        buyer_turnover = 150000000

        # Single LLM call covers both D1 and D2 together
        llm_result = call_llm_for_tds_determination(
            invoice, vendor_record, tds_rules_text, buyer_turnover
        )

        # Extracting the D2 portion of the LLM result
        d2_result              = llm_result.get("d2", {})
        tds_section            = d2_result.get("tds_section", "UNKNOWN")
        tds_rate               = float(d2_result.get("tds_rate", 0.0))
        tds_base_amount        = float(d2_result.get("tds_base_amount", 0.0))
        tds_amount_from_llm    = float(d2_result.get("tds_amount", 0.0))
        base_amount_note       = d2_result.get("base_amount_note", "")
        special_rules          = d2_result.get("special_rules_applied", [])
        confidence             = float(d2_result.get("confidence", 0.50))
        reasoning              = d2_result.get("reasoning", "")
        vendor_classification  = d2_result.get("vendor_classification", "")
        service_classification = d2_result.get("service_classification", "")

        # Running arithmetic sanity check to catch any LLM math errors
        amount_is_correct, expected_amount = run_tds_sanity_check(
            tds_base_amount, tds_rate, tds_amount_from_llm
        )

        if amount_is_correct:
            # LLM math checks out - using its amount
            final_tds_amount = tds_amount_from_llm
        else:
            # LLM reasoning was right but arithmetic was off - correcting it
            final_tds_amount = expected_amount
            special_rules.append(
                f"Arithmetic corrected: LLM said Rs {tds_amount_from_llm:,.2f} "
                f"but Rs {tds_base_amount:,.2f} x {tds_rate}% = Rs {expected_amount:,.2f}"
            )
            confidence = min(confidence, 0.80)

        # Building the finding text from all parts
        finding_parts = [
            f"Section {tds_section}",
            f"Rate: {tds_rate}%",
            f"TDS amount: Rs {final_tds_amount:,.2f}",
            base_amount_note
        ]
        if special_rules:
            finding_parts.extend(special_rules)
        if vendor_classification:
            finding_parts.append(f"Vendor: {vendor_classification}")
        if service_classification:
            finding_parts.append(f"Service: {service_classification}")

        return json.dumps({
            "check_id":              "D2",
            "passed":                True,
            "confidence":            round(confidence, 3),
            "tds_section":           tds_section,
            "tds_rate":              tds_rate,
            "tds_base_amount":       tds_base_amount,
            "tds_amount":            final_tds_amount,
            "arithmetic_verified":   amount_is_correct,
            "finding":               " | ".join(finding_parts),
            "evidence": {
                "tds_section":            tds_section,
                "tds_rate":               tds_rate,
                "tds_base_amount":        tds_base_amount,
                "tds_amount":             final_tds_amount,
                "base_amount_note":       base_amount_note,
                "special_rules_applied":  special_rules,
                "vendor_classification":  vendor_classification,
                "service_classification": service_classification,
                "reasoning":              reasoning,
                "arithmetic_verified":    amount_is_correct
            }
        })

    except Exception as error:
        return json.dumps({
            "check_id":              "D2",
            "passed":                False,
            "confidence":            0.00,
            "tds_section":           "ERROR",
            "finding":               f"Could not determine TDS section: {str(error)}",
            "requires_human_review": True
        })


# ============================================================
# CATEGORY E - POLICY AND BUSINESS RULES
# ============================================================

@tool("Check Invoice Amount Within PO Tolerance")
def check_po_amount_tolerance(invoice_json: str) -> str:
    """Im using this tool to check if the invoice total is within plus or minus 5 percent of the PO amount.
    If PO amount is missing the check is skipped and marked as not applicable.
    Input: full invoice as a JSON string.
    Returns: JSON string with check_id E1, passed, confidence, variance percentage, and finding."""
    try:
        invoice       = json.loads(invoice_json)
        invoice_total = float(invoice.get("total_amount", 0))
        po_amount     = float(invoice.get("po_amount", 0))
        po_reference  = invoice.get("po_reference", invoice.get("po_number", ""))

        # No PO amount means this check cannot be performed - skipping it
        # Skipping is different from failing - the invoice is not penalised
        if po_amount == 0:
            return json.dumps({
                "check_id":   "E1",
                "passed":     True,
                "confidence": 1.00,
                "skipped":    True,
                "finding":    "PO amount not provided - E1 check skipped, not applicable for this invoice",
                "evidence":   {"po_reference": po_reference if po_reference else "none"}
            })

        # Checking if invoice is within 5% band of the PO amount
        tolerance_amount  = po_amount * 0.05
        amount_difference = abs(invoice_total - po_amount)
        variance_percent  = round((amount_difference / po_amount) * 100, 2)
        check_passed      = amount_difference <= tolerance_amount

        return json.dumps({
            "check_id":   "E1",
            "passed":     check_passed,
            "confidence": 0.99,
            "skipped":    False,
            "finding": (
                f"Within PO tolerance - variance is {variance_percent}% which is under 5%"
                if check_passed
                else f"Exceeds PO tolerance - variance is {variance_percent}% which is over the 5% limit"
            ),
            "evidence": {
                "invoice_total":       invoice_total,
                "po_amount":           po_amount,
                "tolerance_5_percent": tolerance_amount,
                "amount_difference":   amount_difference,
                "variance_percent":    variance_percent,
                "po_reference":        po_reference
            }
        })

    except Exception as error:
        return json.dumps({
            "check_id": "E1", "passed": False, "confidence": 0.20,
            "finding":  f"Could not check PO tolerance: {str(error)}"
        })


@tool("Check Vendor is on Approved Vendor List")
def check_approved_vendor(vendor_gstin: str, vendor_registry_json: str) -> str:
    """Im using this tool to verify the vendor exists in vendor_registry.json and is currently ACTIVE.
    Vendor can be in the registry but have SUSPENDED status - both conditions must pass.
    Input: vendor_gstin as a plain string, vendor_registry as a JSON string.
    Returns: JSON string with check_id E3, passed, confidence, and finding."""
    try:
        vendor_registry = json.loads(vendor_registry_json)

        # Fixing OCR errors before doing the registry lookup
        normalized_gstin = fix_ocr_errors_in_gstin(vendor_gstin)
        vendor_record    = find_vendor_in_registry(normalized_gstin, vendor_registry)

        if not vendor_record:
            return json.dumps({
                "check_id":   "E3",
                "passed":     False,
                "confidence": 0.99,
                "finding":    f"Vendor GSTIN '{normalized_gstin}' not found in the approved vendor registry",
                "evidence":   {"normalized_gstin": normalized_gstin, "found_in_registry": False}
            })

        # Vendor found - now checking if they are actually active
        vendor_status    = vendor_record.get("status", "UNKNOWN")
        vendor_is_active = vendor_status == "ACTIVE"

        if not vendor_is_active:
            return json.dumps({
                "check_id":   "E3",
                "passed":     False,
                "confidence": 0.99,
                "finding":    f"Vendor '{vendor_record.get('legal_name', '')}' is in registry but status is '{vendor_status}'",
                "evidence": {
                    "normalized_gstin": normalized_gstin,
                    "vendor_name":      vendor_record.get("legal_name", ""),
                    "status":           vendor_status,
                    "suspension_date":  vendor_record.get("suspension_date", "not available")
                }
            })

        return json.dumps({
            "check_id":   "E3",
            "passed":     True,
            "confidence": 1.00,
            "finding":    f"Vendor '{vendor_record.get('legal_name', '')}' is on the approved list and is ACTIVE",
            "evidence": {
                "normalized_gstin": normalized_gstin,
                "vendor_id":        vendor_record.get("vendor_id", ""),
                "vendor_name":      vendor_record.get("legal_name", ""),
                "vendor_type":      vendor_record.get("vendor_type", ""),
                "status":           vendor_status
            }
        })

    except Exception as error:
        return json.dumps({
            "check_id": "E3", "passed": False, "confidence": 0.20,
            "finding":  f"Could not check approved vendor list: {str(error)}"
        })