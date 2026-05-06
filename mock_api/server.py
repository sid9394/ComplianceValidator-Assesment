"""
server.py
A fake version of the GST government portal. I only implemented the endpoints
that the compliance checks actually need - no point building more than that.

Start this before running the main pipeline:
    python mock_api/server.py

Keep it running in a separate terminal the whole time.
"""

from flask import Flask, request, jsonify
from pathlib import Path
import json
import re

app = Flask(__name__)

# ============================================================
# load data
# ============================================================

# loading the vendor registry once when the server starts
# using a dict keyed by GSTIN so lookups are O(1) instead of scanning the list every time
DATA_DIR = Path(__file__).parent.parent / "data"

with open(DATA_DIR / "vendor_registry.json") as f:
    vendor_data = json.load(f)

VENDORS = {
    v["gstin"].upper(): v
    for v in vendor_data.get("vendors", [])
    if v.get("gstin")
}

# standard 15-character GSTIN format
GSTIN_PATTERN = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
API_KEY       = "test-api-key-12345"


# ============================================================
# endpoints
# ============================================================

@app.route("/health")
def health():
    """Just a health check so the main script can tell if the server is running."""
    return jsonify({"status": "ok"})


@app.route("/api/gst/validate-gstin", methods=["POST"])
def validate_gstin():
    """
    Validates a GSTIN and returns whether it's active.
    The B1 check in check_tools.py calls this.

    What it does:
    1. checks the auth header
    2. validates the GSTIN format with regex
    3. looks up the GSTIN in the vendor registry
    4. returns the status (ACTIVE, SUSPENDED, or CANCELLED)
    """

    # check auth header
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"valid": False, "error": "UNAUTHORIZED"}), 401

    # get the GSTIN from the request body
    gstin = (request.json or {}).get("gstin", "").upper().strip()

    # check the format
    if not gstin or not re.match(GSTIN_PATTERN, gstin):
        return jsonify({
            "valid":   False,
            "error":   "INVALID_FORMAT",
            "message": "GSTIN must be 15 characters alphanumeric"
        }), 400

    # look up the vendor
    vendor = VENDORS.get(gstin)
    if not vendor:
        return jsonify({
            "valid":   False,
            "error":   "NOT_FOUND",
            "message": "GSTIN not registered in GST system"
        }), 404

    # build the response
    response = {
        "valid":         True,
        "gstin":         gstin,
        "legal_name":    vendor.get("legal_name", ""),
        "status":        vendor.get("status", "ACTIVE"),
        "state_code":    vendor.get("state_code", ""),
        "state":         vendor.get("state", ""),
        "taxpayer_type": vendor.get("gst_filing_status", "Regular"),
    }

    # add extra info if the vendor isn't active
    if vendor.get("status") == "SUSPENDED":
        response["suspension_date"]   = vendor.get("suspension_date", "")
        response["suspension_reason"] = vendor.get("suspension_reason", "")

    if vendor.get("status") == "CANCELLED":
        response["cancellation_date"] = vendor.get("cancellation_date", "")

    return jsonify(response)


# ============================================================
# run the server
# ============================================================

if __name__ == "__main__":
    print(f"Mock API running on http://localhost:8080")
    print(f"Vendors loaded: {len(VENDORS)}")
    print(f"Endpoint: POST /api/gst/validate-gstin")
    app.run(port=8080, debug=True)
