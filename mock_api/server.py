from flask import Flask, request, jsonify
import re

app = Flask(__name__)

GSTIN_PATTERN = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/gst/validate-gstin", methods=["POST"])
def validate_gstin():
    data = request.json
    gstin = data.get("gstin", "").upper().strip()
    valid = bool(re.match(GSTIN_PATTERN, gstin))
    inactive = gstin in ["27ZZZZZ9999Z1ZZ"]
    return jsonify({
        "gstin": gstin,
        "valid": valid and not inactive,
        "status": "INACTIVE" if inactive else ("ACTIVE" if valid else "INVALID"),
        "message": "Valid and active" if (valid and not inactive) else "Invalid or inactive GSTIN"
    })

@app.route("/api/gst/verify-irn", methods=["POST"])
def verify_irn():
    data = request.json
    irn = data.get("irn", "")
    valid = len(irn) == 64
    return jsonify({
        "irn": irn,
        "valid": valid,
        "message": "Valid IRN" if valid else "IRN must be 64 characters"
    })

@app.route("/api/tds/check-206ab", methods=["POST"])
def check_206ab():
    data = request.json
    pan = data.get("pan", "").upper().strip()
    flagged = pan.startswith("X")
    return jsonify({
        "pan": pan,
        "flagged_206ab": flagged,
        "tds_rate_multiplier": 2 if flagged else 1,
        "reason": "Non-filer — higher TDS rate applies" if flagged else "Normal rate"
    })

if __name__ == "__main__":
    print("Mock API running on http://localhost:5000")
    app.run(port=5000, debug=True)