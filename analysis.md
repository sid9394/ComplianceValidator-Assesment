# Analysis — Compliance Validator Agent

## Approach

The core design goal was to make each agent do exactly one thing and produce a typed,
structured output that the next agent can rely on without guessing. Four agents maps cleanly
to the four concerns: data extraction, compliance checking, decision-making, and report
formatting. A sequential pipeline (not parallel or hierarchical) was chosen because each
stage genuinely depends on the previous stage's output.

All 10 checks are implemented as deterministic Python tools rather than asking the LLM to
reason about arithmetic or regex patterns directly. The LLM is only involved where judgment
is actually required: TDS section determination, which requires reading and interpreting
complex regulatory rules with vendor-specific exceptions.

---

## Decisions Made

### 1. Deterministic tools over LLM for most checks

Checks A1, A2, B1, B7, C1, C2, E1, E3 are pure logic — regex, arithmetic, or lookup. Running
them as Python functions (not LLM prompts) gives 100% confidence, makes failures unambiguous,
and eliminates the cost and latency of extra LLM calls. The LLM is reserved for TDS (D1/D2)
where Indian tax regulation genuinely requires reading rules, handling exceptions (206AB, DTAA,
LDC), and applying judgment about vendor classification.

### 2. Shared LLM call for D1 and D2

D1 (applicability) and D2 (section + rate + amount) are logically two checks but are answered
by the same reasoning process. Making two separate LLM calls would duplicate context, cost,
latency, and introduce the risk of inconsistent answers. A single call covers both; D1 stores
the result in a per-invoice cache and D2 reads it. The cache is cleared after each invoice
so memory stays flat across a large batch.

### 3. LLM arithmetic sanity check in D2

LLMs occasionally make arithmetic errors on TDS amount calculations. D2 computes the expected
amount independently (base × rate / 100) and compares it to the LLM's figure. If the difference
exceeds Rs 1, the calculated amount is used and confidence is capped at 0.80. This means TDS
amounts in the report are always mathematically correct even when the LLM's reasoning was sound
but its arithmetic was not.

### 4. Vendor registry token optimisation

The full vendor registry is ~3,500 tokens. Sending it with every invoice wastes tokens on
every call. Before the crew runs, `main.py` looks up the vendor by GSTIN and passes only that
one vendor record (~300 tokens) to the crew. For non-JSON input formats where the GSTIN is
unknown before the extractor runs, the full registry is passed as a safe fallback.

### 5. Format-agnostic input pipeline

The submission spec requires support for multiple input formats. The design separates file
parsing (main.py, deterministic) from data normalization (Extractor agent, LLM). The parsers
extract raw text from PDFs, DOCX files, and CSV rows and wrap it in a typed dict
(`_source_format`, `_raw_content`). The Extractor agent receives this wrapper and knows to
parse the raw content rather than treating it as structured JSON. This avoids building fragile
regex-based field extractors for unstructured documents.

### 6. OCR correction for GSTIN

B1 includes a pre-processing step that fixes common OCR misreads in the digit positions of
a GSTIN (positions 0, 1, 9–12): O→0, I→1, l→1. This prevents valid GSTINs from failing the
regex check due to scanner artefacts in PDF-sourced invoices.

### 7. Priority-based resolver decision rules

Checks are assigned one of three priority levels — HIGH, MEDIUM, LOW — that determine which
resolver rule fires on failure. HIGH checks (A2, B1, B7, C1, C2, E3) are hard compliance
violations: failure always produces `REJECTED` regardless of confidence. MEDIUM checks (A1,
D1, D2) require judgment: failure produces `ESCALATE_TO_HUMAN`. The LOW check (E1) is a
business policy gate that is frequently skipped and only escalates when a PO amount is
present and the variance is material.

This distinction is enforced at three levels: the `priority` field in every check tool's
JSON output, the resolver task instructions, and the resolver agent's backstory. Each check
tool returns its own priority so the resolver has the information inline rather than
relying on a separate lookup.

### 8. Confidence-based escalation at 70%

The threshold follows the spec exactly. Low confidence typically arises in two situations:
the Mock GSTIN API is unavailable (B1 drops to 0.70) or TDS determination is ambiguous (D1/D2
return low confidence for edge cases like unregistered foreign vendors). In both cases
`ESCALATE_TO_HUMAN` is correct — a human reviewer should make the final call rather than the
system auto-approving or auto-rejecting.

---

## Edge Cases Handled

### RCM invoices
B7 (GST rate consistency) is skipped for Reverse Charge Mechanism invoices because the buyer
pays GST directly to the government, not the vendor. The invoice will have no vendor-side GST
rate to validate.

### Foreign vendors
B1 GSTIN check: foreign vendors have no Indian GSTIN. The check is marked skipped with a note.
B7: no Indian GST applies; skipped.
D1/D2 TDS: section 195 applies for non-residents. If a DTAA country is registered in the
vendor record, the treaty rate is used. No-PAN vendors are charged 20%.

### Missing PO reference
E1 (PO tolerance) is skipped — not failed — when no PO amount is present on the invoice.
Many legitimate invoices are raised without a PO reference. The check is only meaningful when
a PO amount exists to compare against.

### Credit notes with negative amounts
C1 and C2 still run on credit notes. Negative quantities and amounts are arithmetically valid
(−qty × rate = −amount). The resolver is instructed to treat negative arithmetic correctness
as valid and not penalise it.

### Duplicate detection across a batch
The A2 check uses a `seen_invoices` list that grows across the entire batch run. The compound
key is `VENDOR_GSTIN::INVOICE_NUMBER::AMOUNT::DATE`. The first occurrence passes; any
subsequent occurrence that matches exactly or reaches 95% average similarity across those
three fields (GSTIN, invoice number, and amount — date is in the compound key to gate the
365-day rolling window but is not included in the similarity ratio) fails A2, which triggers
`REJECTED` immediately. Both exact and near-duplicates
are hard rejections — the 95% similarity threshold is the discriminator, not the confidence
score. For non-JSON sources where the invoice number is extracted by the LLM, the resolved
number from the crew report is added to the seen-list after processing.

### Temporal validity — invoice date vs processing date
All GST and TDS rules are applied based on `invoice_date`, not the current processing date.
For B7, invoices dated in March or April fall in the GST rate transition window; the tool
flags these with confidence 0.70 and "transition period" in the finding, which triggers
Rule 4 in the resolver and escalates to `ESCALATE_TO_HUMAN`. For D1/D2, the LLM is
explicitly instructed to use the financial year of the invoice date when applying TDS
threshold rules and section assignments.

### Regulatory conflicts between checks
Some invoices trigger two rules that are in tension. The primary case is an RCM invoice
where B7 is skipped (buyer pays GST directly) but D1 determines TDS still applies. This
is a genuine regulatory ambiguity: the payer cannot self-certify both that GST is reverse-
charged and that TDS is deductible without human verification. When this occurs both
findings include `REGULATORY CONFLICT`, which causes the resolver Rule 2 to fire and
produce `ESCALATE_TO_HUMAN` regardless of confidence.

### Malformed or unprocessable invoices
If the crew itself raises an unhandled exception (e.g. a catastrophic LLM API failure), the
pipeline catches it, logs the error, and writes a `HOLD_FOR_VERIFICATION` report with a zero
score. The batch continues. No invoice failure stops the remaining invoices from being
processed.

### Vendor in registry but suspended
E3 checks both presence and `status == "ACTIVE"`. A vendor that is in the registry but has
been suspended still fails E3. The suspension date is included in the finding for audit
purposes.

### 206AB higher-rate deduction
If the vendor record has `section_206ab_applicable: true`, D2 applies the 206AB rate
multiplier on top of the applicable section rate, with a minimum of 5%. This is handled
within the TDS LLM prompt and the special rules applied list in the D2 output records it.

---

## Historical Decisions — Deliberate Non-Use

The provided `historical_decisions.jsonl` was not used to build or train any validation logic
in this system. The file contains 26 past decisions, of which the challenge specification
explicitly states 15% are incorrect — errors include wrong TDS rates, missed 206AB escalations,
incorrect approval levels, and GST rate violations that were approved without challenge.

Using this data as a source of truth would propagate those errors into the new system. All
validation logic in this implementation is derived exclusively from the primary regulatory
sources: `tds_sections.json` (TDS rules and rates), `gst_rates_schedule.csv` (GST rates by
HSN/SAC code), `company_policy.yaml` (approval thresholds and business rules), and
`vendor_registry.json` (vendor status and TDS attributes). The historical file is present in
the data directory but is never loaded or referenced at runtime.

When this system's decisions diverge from a historical decision, it is because the historical
decision was wrong — not because the historical decision was treated as precedent and the system
disagrees. The audit trail for each invoice documents the specific regulation and evidence used
for every check, making deviations from historical patterns fully explainable.

---

## Limitations and Known Trade-offs

- **Mock API only**: B1's active-status check runs against a local mock server, not the live
  GSTIN portal. When the mock is unavailable, B1 confidence drops to 0.70 (triggering
  escalation) rather than failing hard. This is intentional — a network issue should not
  auto-reject valid invoices.

- **LLM non-determinism in TDS**: D1/D2 results can vary slightly between runs on ambiguous
  TDS scenarios. Temperature is set to 0.1 to minimise this. The arithmetic sanity check
  corrects the most impactful class of variation (calculation errors) deterministically.

- **Aggregate TDS threshold tracking**: TDS applicability is determined per-invoice against
  the single-payment threshold (e.g., Rs 30,000 for 194C/194J). Aggregate threshold tracking
  across multiple invoices to the same vendor within a financial year (e.g., 194C aggregate
  Rs 75,000; 194Q aggregate Rs 50 Lakh) is not implemented. In the provided test batch this
  does not affect any outcome — every relevant invoice individually exceeds its single-payment
  threshold, and 194Q applicability is correctly triggered by FinanceGuard's buyer turnover
  (Rs 15 Cr > Rs 10 Cr threshold). A production deployment would require a persistent
  vendor-payment ledger fed from the AP system to track running FY totals.

- **Rate limits**: Gemini free tier is 15 RPM with ~4 LLM calls per invoice. The default 20s
  sleep between invoices keeps the system within limits at the cost of throughput. A paid API
  tier removes this constraint.
