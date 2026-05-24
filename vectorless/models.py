"""Model pins for non-parser roles.

The parser model is per-category in vectorless/categories.py.
RETRIEVAL_MODEL can be overridden via the RETRIEVAL_MODEL_OVERRIDE
environment variable.
"""

import os

SUMMARY_MODEL = "gemini-2.5-flash-lite"
OCR_CLEAN_MODEL = "gemini-2.5-flash-lite"
JUDGE_MODEL = "gemini-2.5-pro"
RETRIEVAL_MODEL = os.environ.get("RETRIEVAL_MODEL_OVERRIDE", "deepseek-v4-flash")
