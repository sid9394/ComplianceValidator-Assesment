"""
main.py
This is the entry point. Run this to process invoices.

Usage:
    python main.py --input data/test_invoices.json --output reports/

What it does:
    1. checks if the mock API is running
    2. loads all the invoices from the input path
    3. loads the vendor registry
    4. runs the crew pipeline on each invoice one at a time
    5. writes a report file per invoice
    6. writes a summary at the end
"""

import argparse
import json
import re
import time
from logger import get_logger, log_invoice_start, log_invoice_end, log_error

log = get_logger("compliance_validator")
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from config.config import (
    VENDOR_REGISTRY_FILE,
    MOCK_API_BASE_URL,
    SLEEP_BETWEEN_INVOICES_SECONDS,
    BATCH_SIZE,
    BATCH_DELAY_SECONDS,
)
from crew import ComplianceValidatorCrew
from tools.check_tools import clear_tds_cache

load_dotenv()


# ========
# helpers
# ========

def check_mock_api_is_running():
    """
    I ping the API before starting so if someone forgot to run the mock server
    they get a clear message telling them to start it rather than mysterious errors
    appearing halfway through processing.
    """
    import requests
    try:
        requests.get(f"{MOCK_API_BASE_URL}/health", timeout=3)
        print("Mock API is running")
    except Exception:
        print(f"WARNING: Mock API not running at {MOCK_API_BASE_URL}")
        print("Start it in Terminal 2: python mock_api/server.py")
        print("Continuing anyway - B1 check will run with lower confidence")


def _parse_json_file(path: Path) -> list:
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in '{path.name}': {e}") from e
    items = data if isinstance(data, list) else [data]
    return [{"invoice_json": json.dumps(inv), "_parsed": inv} for inv in items]


def _parse_csv_file(path: Path) -> list:
    import csv
    entries = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wrapper = {"_source_format": "csv", "_source_file": path.name, "_raw_content": json.dumps(dict(row))}
                entries.append({"invoice_json": json.dumps(wrapper), "_parsed": None})
    except UnicodeDecodeError as e:
        raise ValueError(f"Cannot read '{path.name}' as UTF-8 CSV: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to parse CSV '{path.name}': {e}") from e
    return entries


def _parse_pdf_file(path: Path) -> list:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF support: pip install pdfplumber")
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as e:
        raise ValueError(f"Failed to parse PDF '{path.name}': {e}") from e
    text = "\n\n".join(p for p in pages if p.strip())
    wrapper = {"_source_format": "pdf", "_source_file": path.name, "_raw_content": text}
    return [{"invoice_json": json.dumps(wrapper), "_parsed": None}]


def _parse_docx_file(path: Path) -> list:
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx is required for Word support: pip install python-docx")
    try:
        doc = docx.Document(str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX '{path.name}': {e}") from e
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    wrapper = {"_source_format": "docx", "_source_file": path.name, "_raw_content": text}
    return [{"invoice_json": json.dumps(wrapper), "_parsed": None}]


_PARSERS = {
    ".json": _parse_json_file,
    ".csv":  _parse_csv_file,
    ".pdf":  _parse_pdf_file,
    ".docx": _parse_docx_file,
    ".doc":  _parse_docx_file,
}


def load_invoices(input_path: str) -> list:
    """
    Loads invoices from a file or folder. Handles JSON, CSV, PDF and DOCX.

    Each entry it returns looks like:
      {"invoice_json": str, "_parsed": dict or None}

    invoice_json is always a JSON string:
      - for JSON files it's the invoice itself
      - for everything else it's a wrapper with _source_format and _raw_content
        that the extractor agent knows how to unpack

    _parsed is the already-decoded dict for JSON inputs so I can grab the vendor GSTIN
    upfront without having to parse it again. It's None for non-JSON formats.
    """
    path = Path(input_path)

    if path.is_file():
        parser = _PARSERS.get(path.suffix.lower())
        if parser is None:
            raise ValueError(f"Unsupported file type: {path.suffix}. Supported: {list(_PARSERS)}")
        return parser(path)

    if path.is_dir():
        entries = []
        for f in sorted(path.iterdir()):
            if f.is_file() and f.suffix.lower() in _PARSERS:
                try:
                    entries.extend(_PARSERS[f.suffix.lower()](f))
                except (ValueError, ImportError) as e:
                    print(f"WARNING: Skipping '{f.name}': {e}", file=__import__("sys").stderr)
        if not entries:
            raise ValueError(f"No valid invoice files found in '{path}'")
        return entries

    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_vendor_registry() -> dict:
    """Loads vendor_registry.json from the data folder."""
    try:
        with open(VENDOR_REGISTRY_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Vendor registry not found: {VENDOR_REGISTRY_FILE}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in vendor registry '{VENDOR_REGISTRY_FILE}': {e}") from e


def extract_json_from_text(text: str) -> dict:
    """
    CrewAI sometimes wraps the output JSON in markdown code fences or adds extra text around it.
    This tries to parse it directly first, then strips the fences and tries again,
    then falls back to regex to find any JSON object in the text.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences and try again
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the first complete JSON object in the text
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Could not parse - return a minimal error report
    return {"parse_error": True, "raw_output": text[:500]}


def build_error_report(invoice_id: str, error: str) -> dict:
    """
    If an invoice fails to process for any reason I still want to write a valid output file
    so the summary doesn't end up with a gap. This builds a skeleton report in the right schema.
    """
    return {
        "invoice_id":           invoice_id,
        "overall_decision":     "HOLD_FOR_VERIFICATION",
        "compliance_score":     0,
        "confidence":           0.0,
        "requires_human_review": True,
        "validation_results": {
            "category_a_authenticity": {"score": 0, "max_score": 2, "checks": {}},
            "category_b_gst":          {"score": 0, "max_score": 2, "checks": {}},
            "category_c_arithmetic":   {"score": 0, "max_score": 2, "checks": {}},
            "category_d_tds":          {"score": 0, "max_score": 2, "checks": {}},
            "category_e_policy":       {"score": 0, "max_score": 2, "checks": {}}
        },
        "tds_summary":  {},
        "gst_summary":  {},
        "audit_trail":  [{
            "timestamp": datetime.now().isoformat(),
            "check":     "SYSTEM",
            "passed":    False,
            "finding":   f"Processing error: {error}",
            "confidence": 0.0
        }]
    }


# ==============
# main pipeline
# ==============

def _process_single_invoice(entry: dict, index: int, total: int,
                             vendor_registry: dict, seen_invoices: list,
                             crew_runner, output_dir: Path, summary: list):
    """Run the crew on one invoice, write the report, and add it to the summary."""
    invoice_json_str = entry["invoice_json"]
    parsed           = entry.get("_parsed")

    invoice_id   = (parsed or {}).get("invoice_id", f"INVOICE-{index + 1}")
    vendor_gstin = (parsed or {}).get("vendor", {}).get("gstin", "")

    print(f"[{index + 1}/{total}] Processing {invoice_id}...")

    log_invoice_start(log, invoice_id)
    start_time = time.time()

    try:
        if vendor_gstin:
            matching_vendor = next(
                (v for v in vendor_registry.get("vendors", [])
                 if (v.get("gstin") or "").upper() == vendor_gstin.upper()),
                {}
            )
            slim_registry = {"vendors": [matching_vendor]} if matching_vendor else {"vendors": []}
        else:
            slim_registry = vendor_registry

        result = crew_runner.crew().kickoff(inputs={
            "invoice_json":         invoice_json_str,
            "vendor_registry_json": json.dumps(slim_registry),
            "seen_invoices_json":   json.dumps(seen_invoices),
        })

        report               = extract_json_from_text(str(result))
        report["invoice_id"] = report.get("invoice_id", invoice_id)

        invoice_number = (parsed or {}).get("invoice_number", "") or report.get("invoice_number", "")
        resolved_gstin = vendor_gstin or report.get("vendor", {}).get("gstin", "")
        invoice_total  = str((parsed or {}).get("total_amount") or 0)
        invoice_date   = (parsed or {}).get("invoice_date", "") or ""
        if resolved_gstin and invoice_number:
            seen_invoices.append(f"{resolved_gstin.upper()}::{invoice_number}::{invoice_total}::{invoice_date}")

        decision   = report.get("overall_decision", "UNKNOWN")
        score      = report.get("compliance_score", 0)
        confidence = report.get("confidence", 0.0)
        print(f"  Decision: {decision} | Score: {score}%")
        log_invoice_end(log, invoice_id, decision, score, confidence, time.time() - start_time)

    except Exception as e:
        log_error(log, invoice_id, str(e))
        print(f"  ERROR: {e}")
        report   = build_error_report(invoice_id, str(e))
        decision = "HOLD_FOR_VERIFICATION"
        score    = 0

    finally:
        clear_tds_cache(invoice_id)

    report_file = output_dir / f"{invoice_id}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    summary.append({"invoice_id": invoice_id, "decision": decision, "score": score})


def process_invoices(input_path: str, output_path: str, batch_size: int = None):
    """
    Main loop. Goes through every invoice one at a time, runs the crew, writes the reports.

    batch_size overrides BATCH_SIZE from config if you want to run a smaller test batch.
    """
    print(f"\nCompliance Validator starting")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 50)

    check_mock_api_is_running()
    invoices        = load_invoices(input_path)
    vendor_registry = load_vendor_registry()
    output_dir      = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = batch_size or BATCH_SIZE
    total           = len(invoices)

    if effective_batch:
        batches = [invoices[i:i + effective_batch] for i in range(0, total, effective_batch)]
        print(f"Loaded {total} invoice(s) | batch size: {effective_batch} | {len(batches)} batch(es)\n")
    else:
        batches = [invoices]
        print(f"Loaded {total} invoice(s) to process\n")

    seen_invoices = []
    summary       = []
    crew_runner   = ComplianceValidatorCrew()
    global_index  = 0

    for batch_num, batch in enumerate(batches):
        if len(batches) > 1:
            print(f"\n--- Batch {batch_num + 1}/{len(batches)} ({len(batch)} invoice(s)) ---")

        for pos, entry in enumerate(batch):
            _process_single_invoice(
                entry, global_index, total,
                vendor_registry, seen_invoices,
                crew_runner, output_dir, summary
            )
            global_index += 1

            # wait between invoices to stay within rate limits
            if global_index < total:
                is_last_in_batch = (pos == len(batch) - 1)
                if is_last_in_batch and len(batches) > 1 and batch_num < len(batches) - 1:
                    # longer pause at the end of a batch
                    print(f"  Batch {batch_num + 1} complete. Waiting {BATCH_DELAY_SECONDS}s before next batch...")
                    time.sleep(BATCH_DELAY_SECONDS)
                else:
                    print(f"  Waiting {SLEEP_BETWEEN_INVOICES_SECONDS}s before next invoice...")
                    time.sleep(SLEEP_BETWEEN_INVOICES_SECONDS)

    # Write summary file
    summary_file = output_dir / "_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total_processed": len(summary),
            "run_timestamp":   datetime.now().isoformat(),
            "results":         summary
        }, f, indent=2)

    # Print final stats
    print("\n" + "=" * 50)
    print(f"Done. {len(summary)} invoice(s) processed.")
    print(f"Reports written to: {output_path}")

    decisions = [r["decision"] for r in summary]
    print(f"  APPROVED:             {decisions.count('APPROVED')}")
    print(f"  REJECTED:             {decisions.count('REJECTED')}")
    print(f"  ESCALATE_TO_HUMAN:    {decisions.count('ESCALATE_TO_HUMAN')}")
    print(f"  HOLD_FOR_VERIFICATION:{decisions.count('HOLD_FOR_VERIFICATION')}")


# =============
# entry point
# =============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compliance Validator - validates invoices against GST and TDS regulations"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to invoice JSON file or folder containing JSON files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output folder where reports will be written"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process N invoices per batch, then pause BATCH_DELAY_SECONDS. "
             "Overrides BATCH_SIZE in config. Use 1-3 when testing against free-tier limits."
    )
    args = parser.parse_args()
    try:
        process_invoices(args.input, args.output, batch_size=args.batch_size)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=__import__("sys").stderr)
        raise SystemExit(1)
