#!/usr/bin/env python3
"""
BRIGHT-specific wrapper around scripts/model_server.py.

Purpose:
  - keep the existing shared model_server.py untouched for BC+ experiments
  - allow BRIGHT runs to use subset-specific embedding query instructions via
    the QUERY_INSTRUCTION environment variable

This wrapper simply overrides the module-global QUERY_INSTRUCTION used by
EmbeddingEngine.encode_query() and then delegates to the original main().
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(SCRIPT_DIR))

import model_server as _base_model_server  # noqa: E402


_QUERY_INSTRUCTION_OVERRIDE = (os.environ.get("QUERY_INSTRUCTION") or "").strip()
if _QUERY_INSTRUCTION_OVERRIDE:
    _base_model_server.QUERY_INSTRUCTION = _QUERY_INSTRUCTION_OVERRIDE


if __name__ == "__main__":
    _base_model_server.main()
