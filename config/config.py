"""
config.py
I put all the config values here so I only have to change things in one place.
Paths, API settings, model names, thresholds; everything lives here.
Nothing is hardcoded anywhere else in the project, it all points back to this file.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Must run before any os.getenv() calls below

# ============================================================
# paths
# ============================================================

# project root - everything else is relative to this
PROJECT_ROOT = Path(__file__).parent

# where all the reference data files are
DATA_DIR = Path("D:/Projects/Agentic Army Datamatics Eval Project/ComplianceValidator/data")

# where the output reports get written
REPORTS_DIR = PROJECT_ROOT / "reports"

# ============================================================
# data file locations
# ============================================================

TDS_SECTIONS_FILE    = DATA_DIR / "tds_sections.json"
VENDOR_REGISTRY_FILE = DATA_DIR / "vendor_registry.json"
COMPANY_POLICY_FILE  = DATA_DIR / "company_policy.yaml"
HSN_SAC_CODES_FILE   = DATA_DIR / "hsn_sac_codes.json"
GST_RATES_FILE       = DATA_DIR / "gst_rates_schedule.csv"
TEST_INVOICES_FILE   = DATA_DIR / "test_invoices.json"

# ============================================================
# mock api settings
# base URL is localhost:8080.
# all routes are under /api/gst/ and you need the auth header or it rejects the request.
# ============================================================

MOCK_API_BASE_URL = "http://localhost:8080"
MOCK_API_KEY      = "test-api-key-12345"
MOCK_API_HEADERS  = {"X-API-Key": MOCK_API_KEY, "Content-Type": "application/json"}
MOCK_API_TIMEOUT  = 5  # seconds

# endpoint paths from the mock api spec
MOCK_API_ENDPOINTS = {
    "validate_gstin":       "/api/gst/validate-gstin",
}

# ============================================================
# groq settings
# I tried splitting into 70B and 8B but they share the same 6000 TPM cap on the free tier
# so having two models doesn't actually help. Just using 70B for everything on groq.
# ============================================================

GROQ_MODEL        = "groq/llama-3.3-70b-versatile"  # LiteLLM format requires provider prefix
GROQ_MAX_TOKENS   = None  # Use model default; free tier: 6000 TPM, 30 RPM, 1000 RPD
GROQ_TEMPERATURE  = 0.1   # keeping temperature low so the compliance decisions are consistent

# ============================================================
# gemini settings
# gemini flash doesn't really have a token bottleneck so I don't need to split models.
# I mirrored GEMINI_FAST_MODEL to GEMINI_MODEL just to keep the code path the same.
# ============================================================

GEMINI_MODEL       = "gemini/gemini-2.5-flash"  # LiteLLM format
GEMINI_MAX_TOKENS  = None  # Use model default
GEMINI_TEMPERATURE = 0.1

# ============================================================
# pick which provider to use here
# just change the one line below to "groq" or "gemini" to switch
# ============================================================

#LLM_PROVIDER       = "groq"
LLM_PROVIDER       = "gemini"

# ACTIVE_MODEL is used for the actual TDS calls (D1 and D2)
# ACTIVE_FAST_MODEL is what the crewai agents use for their own reasoning
ACTIVE_MODEL       = GROQ_MODEL      if LLM_PROVIDER == "groq" else GEMINI_MODEL
ACTIVE_FAST_MODEL  = GROQ_MODEL      if LLM_PROVIDER == "groq" else GEMINI_MODEL
ACTIVE_MAX_TOKENS  = GROQ_MAX_TOKENS  if LLM_PROVIDER == "groq" else GEMINI_MAX_TOKENS
ACTIVE_TEMPERATURE = GROQ_TEMPERATURE if LLM_PROVIDER == "groq" else GEMINI_TEMPERATURE

# ============================================================
# business rules and thresholds
# pulled from company_policy.yaml so I don't have to change them in two places
# ============================================================

# FinanceGuard's previous FY turnover is Rs 15 Crore
# this matters for 194Q which only kicks in when the buyer's turnover is over Rs 10 Crore
BUYER_TURNOVER_PREVIOUS_FY = 150_000_000

# Invoice acceptance rules
MAX_INVOICE_AGE_DAYS       = 180    # Reject invoices older than 180 days
FUTURE_DATE_TOLERANCE_DAYS = 0      # No future-dated invoices
PO_TOLERANCE_PERCENT       = 5.0    # Invoice can be +/- 5% of PO amount
ARITHMETIC_TOLERANCE_RS    = 1.0    # Allow Rs 1 rounding difference in math checks

# E-invoice mandatory threshold - Rs 5 Crore
E_INVOICE_THRESHOLD = 50_000_000

# ============================================================
# rate limit settings
# groq free tier is 30 RPM and 6000 TPM for 70B which is pretty tight
# gemini flash is more generous, 15 RPM but the token limit is much higher
# most of my groq budget goes on the TDS call which is the only real LLM call per invoice
# ============================================================

SLEEP_BETWEEN_INVOICES_SECONDS = 45 if LLM_PROVIDER == "groq" else 20

# adding a small delay before each groq call to stay within 30 RPM
# gemini doesn't need this so it's 0
REQUEST_DELAY_SECONDS = 2 if LLM_PROVIDER == "groq" else 0

# Retry wait times for 429 / RESOURCE_EXHAUSTED errors
RETRY_WAIT_SECONDS = [30, 60, 90]

# ============================================================
# batch processing settings
# BATCH_SIZE lets me process invoices in groups with a longer pause between groups
# set it to None to just process everything with the regular sleep between invoices
# useful to set it to something like 3 when testing to not burn through the daily request limit
# ============================================================

BATCH_SIZE           = None  # None = no batching; set to e.g. 3 to test small batches first
BATCH_DELAY_SECONDS  = 60 if LLM_PROVIDER == "groq" else 20

# ============================================================
# caching settings
# ENABLE_RESPONSE_CACHE: litellm will cache identical prompts so the same call isn't made twice
# ENABLE_GSTIN_CACHE: skip the mock API call if we already validated that GSTIN this run
# ENABLE_VENDOR_TDS_CACHE: if the same vendor appears in multiple invoices, reuse the TDS result
# ============================================================

ENABLE_RESPONSE_CACHE  = True
ENABLE_GSTIN_CACHE     = True
ENABLE_VENDOR_TDS_CACHE = True
