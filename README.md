# Compliance Validator Agent

A multi-agent AI system for validating Indian GST and TDS compliance across invoice batches.
Built with CrewAI, LiteLLM, and a mock GSTIN verification API.

---

## Requirements

- Python 3.11+
- A Gemini or Groq API key (see Setup)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Setup

1. Copy the example env file and add your API key:

```bash
cp .env.example .env
```

2. Edit `.env` and paste your key:

```
GEMINI_API_KEY="your-key-here"   # or GROQ_API_KEY if using Groq
```

3. To switch between Gemini and Groq, edit the one line in `config/config.py`:

```python
LLM_PROVIDER = "gemini"   # or "groq"
```

---

## Running

The system requires the Mock GSTIN API to be running before you start the main pipeline.

**Terminal 1 — start the mock API:**

```bash
python mock_api/server.py
```

**Terminal 2 — run the validator:**

```bash
python main.py --input <path> --output <path>
```

### Input formats supported

| Format | Example |
|--------|---------|
| JSON file (array or single invoice) | `--input data/test_invoices.json` |
| Folder of JSON/PDF/DOCX/CSV files | `--input data/invoices/` |
| Single PDF invoice | `--input invoices/invoice.pdf` |
| Single Word document | `--input invoices/invoice.docx` |
| CSV (one invoice per row) | `--input invoices/batch.csv` |

### Example

```bash
python main.py --input data/test_invoices.json --output reports/
```

This processes every invoice in `test_invoices.json` and writes:
- One `<invoice_id>.json` report per invoice in `reports/`
- A `_summary.json` file with totals and decisions

---

## Output

Each invoice produces a JSON file matching this schema:

```json
{
  "invoice_id": "INV-2024-0001",
  "overall_decision": "APPROVED",
  "compliance_score": 96,
  "confidence": 0.95,
  "requires_human_review": false,
  "validation_results": {
    "category_a_authenticity": { "score": 2, "max_score": 2, "checks": {} },
    "category_b_gst":          { "score": 2, "max_score": 2, "checks": {} },
    "category_c_arithmetic":   { "score": 2, "max_score": 2, "checks": {} },
    "category_d_tds":          { "score": 2, "max_score": 2, "checks": {} },
    "category_e_policy":       { "score": 2, "max_score": 2, "checks": {} }
  },
  "tds_summary": {},
  "gst_summary": {},
  "audit_trail": []
}
```

`overall_decision` is always one of: `APPROVED` | `REJECTED` | `ESCALATE_TO_HUMAN` | `HOLD_FOR_VERIFICATION`

---

## Project Structure

```
main.py                  Entry point — format detection, batch loop, report writer
crew.py                  CrewAI pipeline wiring (4 agents, 4 tasks, sequential)
tools/check_tools.py     10 compliance check tools attached to the Validator agent
mock_api/server.py       Local Flask server mocking GSTIN verification API
config/
  config.py              All config — paths, model settings, thresholds
  agents.yaml            Agent roles, goals, backstories
  tasks.yaml             Task descriptions and expected outputs
data/
  test_invoices.json     Sample invoice batch for testing
  vendor_registry.json   Approved vendor list
  tds_sections.json      TDS rules (194C, 194J, 194H, 194I, 194Q, 195, 206AB)
  gst_rates_schedule.csv GST rate schedule by HSN/SAC code
  company_policy.yaml    FinanceGuard business rules and thresholds
reports/                 Generated compliance reports (one JSON per invoice)
logs/                    Structured run logs
```

---

## Rate Limits

Gemini free tier (flash): ~15 RPM. The pipeline sleeps 20 seconds between invoices.
Groq free tier: ~30 RPM. The pipeline sleeps 45 seconds between invoices.

To process a large batch quickly, use a paid API tier and set `SLEEP_BETWEEN_INVOICES_SECONDS = 0` in `config/config.py`.

---

## Assumptions

The following assumptions govern the design of each compliance check.

---

### Category A — Document Authenticity

#### A1 — Invoice Number Format Validation

A valid invoice number follows a recognizable structured pattern, such as `TS/MH/2024/001234` or `GL/DEL/24-25/5678`. The check rejects:

- Completely unstructured strings with no separators or segments
- Numbers that are suspiciously short (fewer than 5 characters) or excessively long (more than 30 characters)

Data quality issues such as lowercase characters or OCR substitutions (e.g., `O` for `0`, `I` for `1`) that appear in other fields (see INV-2024-0016 GSTIN trap) may also appear in invoice numbers. The check applies OCR normalization before matching patterns.

#### A2 — Duplicate Invoice Detection

Two invoices are considered duplicates if they share the same vendor GSTIN, invoice number, and invoice amount within a 365-day window — as defined in `company_policy.yaml` (`duplicate_window_days: 365`, `duplicate_fields: [vendor_gstin, invoice_number, invoice_amount]`).

A near-duplicate is flagged when the average similarity across those three fields is ≥ 95% (`near_duplicate_threshold: 0.95`). INV-2024-0010 is a known near-duplicate of INV-2024-0001 — same vendor GSTIN, same date, same amount, only the invoice number differs.

Duplicate detection runs within the processed batch only. It is not cross-checked against a historical database.

---

### Category B — GST Compliance

#### B1 — GSTIN Format and Active Status

A valid GSTIN is exactly 15 characters in the format:

```
SS PPPPPNNNNE E Z C
↑    ↑     ↑  ↑ ↑ ↑
State  PAN  Seq Ent Z Checksum
```

- Characters 1–2: state code (numeric)
- Characters 3–12: PAN of the registered entity
- Character 13: entity registration number (1–9 or A–Z)
- Character 14: always the letter `Z`
- Character 15: alphanumeric checksum

The check applies OCR correction before validation (`O→0`, `I→1`, `l→1` at digit positions). A GSTIN that passes the format check is then verified against the mock GSTIN API for active status. INV-2024-0016 is a known trap — the GSTIN ends in lowercase `j` instead of `J`; this must be caught and failed.

#### B7 — GST Type Consistency (CGST/SGST vs IGST)

The applicable GST type is determined by comparing the first two digits of the vendor GSTIN against the buyer GSTIN:

- The buyer is always registered in Maharashtra (state code `27`)
- Vendor state code `27` → intrastate supply → must use CGST + SGST, IGST must be zero
- Vendor state code other than `27` → interstate supply → must use IGST only, CGST and SGST must be zero
- Mixing CGST/SGST and IGST on the same invoice is always invalid
- For intrastate invoices, CGST rate must equal SGST rate
- RCM invoices and foreign vendors (no GSTIN) are excluded from this check

---

### Category C — Arithmetic

#### C1 — Line Item Arithmetic

For every line item: `quantity × rate = amount`. A tolerance of ±₹1 is allowed for rounding (configured in `company_policy.yaml`). Each line item is checked individually. INV-2024-0013 is a known multi-line trap with three different line items that must each be verified separately.

#### C2 — Subtotal and Total Consistency

Two checks are performed:

1. `sum(line_item.amount for all items) = subtotal`
2. `subtotal + total_tax = total_amount`

Both use the same ±₹1 rounding tolerance. This check runs on every invoice regardless of how simple the line items look.

---

### Category D — TDS Compliance

#### D1 — TDS Applicability

TDS applicability is determined by the vendor type from `vendor_registry.json`, cross-referenced against threshold rules in `tds_sections.json`:

| Vendor Type | Section |
|---|---|
| IT_SERVICES | 194J |
| PROFESSIONAL_SERVICES | 194J |
| CONTRACTOR | 194C |
| TRANSPORT | 194C (GTA exception applies) |
| RENT | 194I |
| GOODS_SUPPLIER | 194Q (threshold-gated) |
| FOREIGN_VENDOR | 195 |

Section 194C applies only when a single payment exceeds ₹30,000 or the aggregate in the financial year exceeds ₹1,00,000. Section 194Q applies only when the buyer's turnover in the previous FY exceeds ₹10 crore. GTA vendors that charge GST on a forward-charge basis are exempt from 194C.

#### D2 — TDS Section and Rate Determination

Once applicability is confirmed, the precise section and rate are determined. Known ambiguities and special rules:

- **194J sub-classification**: Software development and IT maintenance → technical services → 2%. IT consulting, data analytics, advisory → professional services → 10%. Where the classification is genuinely ambiguous, the check escalates rather than guesses.
- **194C entity type**: Payments to companies attract 2%; payments to individuals or HUFs attract 1%. Entity type is inferred from the GSTIN registration type returned by the mock API.
- **194I sub-classification**: Commercial office rent (land/building) → 10%. Plant and machinery → 2%.
- **Section 206AB override**: If `section_206ab_applicable: true` in the vendor registry (non-filer flag), TDS rate is the higher of double the normal rate or 5%. INV-2024-0008 (RK Electricals) is flagged — TDS should be 5%, not 1%.
- **194I base amount**: TDS under section 194I is calculated on the **gross amount including GST**, unlike all other sections which use the subtotal (pre-GST) as the base. For INV-2024-0009: TDS = 10% × ₹5,31,000 = ₹53,100, not 10% × ₹4,50,000.

D1 and D2 share a single LLM call per invoice (result is cached); D2 reads from the cache at no additional cost.

---

### Category E — Policy and Business Rules

#### E1 — Invoice Amount Within PO Tolerance

If a PO reference exists on the invoice, the invoice total must fall within ±5% or ±₹1,000 of the PO amount, whichever is greater (`amount_tolerance_percentage: 5.0`, `amount_tolerance_absolute: 1000` from `company_policy.yaml`).

The test invoices include a `po_reference` field but no corresponding PO amount. Where no PO amount is present, the check is skipped and noted as non-applicable rather than failed. This is a known data gap in the test dataset.

#### E3 — Approved Vendor List

The vendor GSTIN is looked up in `vendor_registry.json`. Three outcomes:

- GSTIN present and `status: ACTIVE` → pass
- GSTIN present but `status: SUSPENDED` or `CANCELLED` → fail (INV-2024-0005, Chennai Software Solutions is a known case)
- GSTIN not found → escalate for verification, do not auto-reject

Additional policy rules from `company_policy.yaml` that apply beyond the basic lookup:

- **First-time vendor** (no prior transaction history): minimum Level 3 approval required before payment
- **Related-party vendor** (same PAN registered under a different GSTIN, e.g. VND001 and VND011): Level 4 approval required, must be flagged in the report
- **Foreign vendor**: additional documentation (Form 10F, tax residency certificate) required before TDS rate reduction can be applied

---

## Design Decisions

### Why most tools don't use an LLM

The agent in `crew.py` already uses an LLM to orchestrate tool calls and synthesize results — adding LLMs inside individual tools would just pay for the same reasoning twice.

For each tool group, there's a concrete reason to keep it deterministic:

- **C1, C2 (arithmetic)** — Why ask an LLM if 2+2=4? These are exact calculations with a single right answer. LLMs are unreliable at precise math anyway, so a simple ₹1 tolerance check is just computed directly.
- **E3, A2 (vendor lookup, duplicate detection)** — The LLM has no access to `vendor_registry.json` or processed invoice history. It would hallucinate.
- **A1, B1, B7, E1 (regex and rules)** — GST/TDS rules like GSTIN format and CGST/IGST splits are exact and well-defined. An LLM risks confident wrong answers on domain-specific edge cases; a broken rule is easier to fix than a nondeterministic one.

**D1 and D2 (TDS)** are the exception — classifying a vendor's service from a text description genuinely requires language understanding, so that's where the LLM earns its cost.
