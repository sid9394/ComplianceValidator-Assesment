# Architecture — Compliance Validator Agent

## System Overview

The system is a sequential multi-agent pipeline built on CrewAI. Each invoice goes through
four specialized agents in order. No agent can delegate to another; each does exactly its
assigned job and passes structured output to the next.

```
Input File(s)
     │
     ▼
[ Format Parser ]  ← main.py (Python)
  JSON / PDF / DOCX / CSV → normalized entry with raw content
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  CrewAI Crew (Sequential)            │
│                                                     │
│  ┌──────────────────┐                               │
│  │  Extractor Agent │  Parse & normalize invoice    │
│  │  (no tools)      │  into canonical JSON schema   │
│  └────────┬─────────┘                               │
│           │ normalized invoice JSON                  │
│           ▼                                         │
│  ┌──────────────────┐                               │
│  │ Validator Agent  │  Run all 10 compliance checks │
│  │ (10 check tools) │  by calling tools in order    │
│  └────────┬─────────┘                               │
│           │ check results (A1–E3)                   │
│           ▼                                         │
│  ┌──────────────────┐                               │
│  │  Resolver Agent  │  Apply decision rules,        │
│  │  (no tools)      │  compute score + confidence   │
│  └────────┬─────────┘                               │
│           │ decision + reasoning                    │
│           ▼                                         │
│  ┌──────────────────┐                               │
│  │  Reporter Agent  │  Format final JSON report     │
│  │  (no tools)      │  matching required schema     │
│  └────────┬─────────┘                               │
└───────────┼─────────────────────────────────────────┘
            │
            ▼
     <invoice_id>.json  +  _summary.json
```

---

## Agents

### 1. Extractor Agent
- **Role**: Invoice Data Extraction Specialist
- **Tools**: None
- **Input**: Raw invoice content — may be native JSON, or a wrapper dict with `_source_format`
  and `_raw_content` extracted from a PDF, DOCX, or CSV file
- **Output**: Canonical invoice JSON with all fields normalized to snake_case, plus
  `missing_fields` and `data_quality_notes` lists
- **Key behaviour**: Copies GSTIN and PAN values exactly without alteration. Flags missing
  fields with null rather than inferring values.

### 2. Validator Agent
- **Role**: Senior GST and TDS Compliance Validator
- **Tools**: All 10 check tools (see Tools section below)
- **Input**: Normalized invoice from Extractor, vendor registry, seen-invoices list
- **Output**: Results for all 10 checks (A1–E3), each with `passed`, `confidence`, `finding`,
  and `evidence`
- **Key behaviour**: Calls tools in strict order. Never skips a check without a documented
  reason. Marks checks as skipped (not failed) when genuinely not applicable.

### 3. Resolver Agent
- **Role**: Lead Compliance Decision Officer
- **Tools**: None
- **Input**: All 10 check results from Validator
- **Output**: `overall_decision`, `compliance_score`, `confidence`, `requires_human_review`,
  lists of failed and skipped checks, decision reasoning
- **Decision rules** (applied in priority order):
  1. A2, B1, C1, or C2 failed → `REJECTED`
  2. Average confidence < 0.70 → `ESCALATE_TO_HUMAN`
  3. Critical fields missing → `HOLD_FOR_VERIFICATION`
  4. All checks pass → `APPROVED`
  5. Any other combination → reasoned judgment

### 4. Reporter Agent
- **Role**: Compliance Report Formatting Specialist
- **Tools**: None
- **Input**: All prior task outputs
- **Output**: Final JSON report in the exact required schema, with full audit trail
- **Key behaviour**: Enforces correct data types on every field (confidence is always float,
  score always int, passed always boolean). Never truncates the audit trail.

---

## Tools (attached to Validator Agent)

| Check | Tool | Method | External calls |
|-------|------|--------|---------------|
| A1 | Check Invoice Number Format | Regex (3 patterns for Indian invoice formats) | None |
| A2 | Check for Duplicate Invoice | GSTIN::invoice_number key lookup in seen-list | None |
| B1 | Validate GSTIN Format and Active Status | Regex + OCR correction + Mock API | Mock API (GSTIN verify) |
| B7 | Check GST Rate Consistency | Arithmetic: CGST=SGST for intra-state, IGST for inter-state | None |
| C1 | Check Line Item Arithmetic | qty × rate = amount per line, Rs 1 tolerance | None |
| C2 | Check Subtotal Matches Sum of Line Items | Sum(line items) = subtotal, Rs 1 tolerance | None |
| D1 | Check TDS Applicability | LLM call with TDS rules + vendor context | LLM (cached) |
| D2 | Determine TDS Section and Rate | Reads D1 cache, sanity-checks LLM math | None (cache hit) |
| E1 | Check Invoice Amount Within PO Tolerance | ±5% variance check vs PO amount | None |
| E3 | Check Vendor is on Approved Vendor List | Lookup in vendor_registry.json, check ACTIVE status | None |

---

## Data Flow

```
main.py
  │
  ├── load_invoices(input_path)
  │     ├── .json  → parse directly, return list of invoice dicts
  │     ├── .pdf   → pdfplumber text extraction → wrapper dict
  │     ├── .docx  → python-docx text extraction → wrapper dict
  │     └── .csv   → csv.DictReader rows → wrapper dicts
  │
  ├── load_vendor_registry()  → vendor_registry.json
  │
  └── for each invoice:
        ├── slim vendor registry to matching GSTIN (token optimisation)
        ├── crew.kickoff(invoice_json, vendor_registry_json, seen_invoices_json)
        │     └── [extract → validate → resolve → report]
        ├── write <invoice_id>.json to output/
        └── append to seen_invoices list (A2 duplicate tracking)
```

---

## External Dependencies

| Component | Purpose |
|-----------|---------|
| CrewAI 1.14.4 | Multi-agent orchestration, sequential task pipeline |
| LiteLLM | Provider-agnostic LLM calls (Gemini or Groq) |
| Google Gemini 2.5 Flash | Primary LLM (TDS reasoning, invoice normalization) |
| Groq llama-3.3-70b | Alternate LLM provider |
| pdfplumber | PDF text extraction |
| python-docx | Word document text extraction |
| Flask | Mock GSTIN verification API server |
| requests | HTTP calls to Mock API |

---

## Key Design Decisions

**Token optimisation**: Before passing the vendor registry to the crew, `main.py` filters it
to the single vendor matching the invoice GSTIN. This reduces registry tokens from ~3,500 to
~300 per invoice. For non-JSON inputs where the GSTIN is unknown upfront, the full registry
is passed and the extractor's normalized output is used by the validator.

**D1/D2 shared LLM call**: TDS determination (D1) and section/rate calculation (D2) share
a single LLM call via an in-process cache keyed by invoice_id. D1 stores the result; D2
reads it for free. The cache is cleared after each invoice to keep memory flat.

**Confidence-based escalation**: Any invoice with average check confidence below 0.70 is
escalated for human review rather than auto-decided. Low confidence typically arises from
Mock API unavailability or ambiguous TDS scenarios.

**Non-crashing design**: Every tool catches all exceptions and returns a structured JSON
error result rather than raising. `main.py` wraps the entire crew run in try/except and
writes a `HOLD_FOR_VERIFICATION` report if the crew itself fails.
