"""
logger.py
Logging setup. I wanted all log files in one place with rotation so they don't eat disk space.
"""

import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from pathlib import Path

# keeping a dict of loggers so I don't create duplicates if get_logger is called twice with the same name
loggers = {}

# logs folder in the project root
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name. Creates it if it doesn't exist yet."""
    global loggers

    if loggers.get(name):
        return loggers.get(name)

    log_file  = LOG_DIR / f"{name}.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # 50MB max per file, keeps 10 backups before deleting old ones
        file_handler = ConcurrentRotatingFileHandler(
            str(log_file),
            maxBytes=1048576 * 50,
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # also log to console so you can see what's happening while it runs
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        logger.propagate = False
        loggers[name] = logger

    return logger


# ============================================================
# helper functions to make logging consistent across the codebase
# ============================================================

def log_invoice_start(log: logging.Logger, invoice_id: str):
    """Log when we start processing an invoice."""
    log.info("=" * 60)
    log.info(f"INVOICE START: {invoice_id}")


def log_invoice_end(log: logging.Logger, invoice_id: str, decision: str,
                    score: int, confidence: float, elapsed_seconds: float):
    """Log the final result once we're done with an invoice."""
    log.info(
        f"INVOICE END: {invoice_id} | "
        f"Decision: {decision} | "
        f"Score: {score}% | "
        f"Confidence: {confidence:.3f} | "
        f"Time: {elapsed_seconds:.2f}s"
    )
    log.info("=" * 60)


def log_agent_reasoning(log: logging.Logger, agent_name: str, reasoning: str):
    """Log what the agent was thinking."""
    log.debug(f"AGENT REASONING [{agent_name}]: {reasoning}")


def log_tool_call(log: logging.Logger, tool_name: str, inputs: dict):
    """Log which tool was called and what inputs it got."""
    log.debug(f"TOOL CALL [{tool_name}]: inputs={inputs}")


def log_tool_response(log: logging.Logger, tool_name: str, check_id: str,
                      passed: bool, confidence: float, finding: str):
    """Log what a tool returned."""
    status = "PASS" if passed else "FAIL"
    log.info(
        f"TOOL RESULT [{tool_name}] {check_id}: "
        f"{status} | confidence={confidence:.2f} | {finding}"
    )


def log_decision(log: logging.Logger, invoice_id: str, decision: str,
                 score: int, confidence: float, reasoning: str):
    """Log the resolver's final decision."""
    log.info(
        f"DECISION [{invoice_id}]: {decision} | "
        f"score={score}% | confidence={confidence:.3f} | {reasoning}"
    )


def log_error(log: logging.Logger, invoice_id: str, error: str):
    """Log when something goes wrong."""
    log.error(f"ERROR [{invoice_id}]: {error}")
