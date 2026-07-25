"""BRIGHT agent configuration built on top of ts_mirror_agent config."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from ts_mirror_agent.config import AgentConfig as BaseAgentConfig  # type: ignore  # noqa: E402
from ts_mirror_agent.config import build_config_from_args as base_build_config_from_args  # type: ignore  # noqa: E402

@dataclass
class AgentConfig(BaseAgentConfig):
    bright_subset: str = ""
    query_instruction: str = ""
    max_retrieved_docs: int = 20
    force_answer_prompt_file: str = ""


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TS-mirror Python Agent (BRIGHT)")

    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--thinking-level", default="")

    parser.add_argument("--question", default="")
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--force-answer-turns", type=int, default=0)
    parser.add_argument("--keep-tool-calls", type=int, default=0)
    parser.add_argument("--compaction-threshold", type=int, default=0)
    parser.add_argument("--use-usage-based-compaction", action="store_true")
    parser.add_argument("--usage-based-compaction-min-tokens", type=int, default=0)
    parser.add_argument("--usage-based-compaction-max-tokens", type=int, default=0)

    parser.add_argument("--corpus-dir", default="")
    parser.add_argument("--scope-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--model-server-script", default="")
    parser.add_argument("--index-dir", default="")
    parser.add_argument("--embed-model-path", default="")
    parser.add_argument("--qr-model-path", default="")
    parser.add_argument("--qr-heads", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--embed-top-k", type=int, default=0)
    parser.add_argument("--no-reranker", action="store_true")

    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument("--append-system-prompt-file", default="")
    parser.add_argument("--force-answer-prompt-file", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--bright-subset", default="")
    parser.add_argument("--query-instruction", default="")
    parser.add_argument("--max-retrieved-docs", type=int, default=20)
    return parser.parse_args()


def build_config_from_args(args: Optional[argparse.Namespace] = None) -> AgentConfig:
    if args is None:
        args = parse_cli_args()

    base_config = base_build_config_from_args(args)
    config = AgentConfig(**base_config.__dict__)
    config.bright_subset = getattr(args, "bright_subset", "") or ""
    config.query_instruction = getattr(args, "query_instruction", "") or ""
    config.max_retrieved_docs = max(1, int(getattr(args, "max_retrieved_docs", 20) or 20))
    config.force_answer_prompt_file = getattr(args, "force_answer_prompt_file", "") or ""
    return config
