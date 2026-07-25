"""
Agent configuration — mirrors TS version's runtime settings.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_MODELS_DIR = _REPO_ROOT / "models"

DEFAULT_EMBED_MODEL = str(_MODELS_DIR / "Qwen3-Embedding-4B")
DEFAULT_QR_MODEL = str(_MODELS_DIR / "Qwen3-4B-Instruct-2507")
DEFAULT_QR_HEADS = "20-15,15-11,23-10,22-4,21-8,15-9,17-4,21-11,21-10,15-13,21-18,18-15,24-23,18-19,21-31,17-25"


@dataclass
class AgentConfig:
    """Mirrors TS agent runtime settings."""

    # === LLM ===
    provider: str = "openai"
    model: str = "gpt-5.4-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 16384
    thinking_level: str = "high"

    # === Agent ===
    max_turns: int = 150
    force_answer_turns: int = 0      # force a final answer when turn count reaches this threshold (0=disabled)
    keep_tool_calls: int = 30       # N most recent tool results kept intact
    compaction_threshold: int = 150000  # trigger compaction when estimated context exceeds this (tokens)
    use_usage_based_compaction: bool = False
    usage_based_compaction_min_tokens: int = 110000
    usage_based_compaction_max_tokens: int = 130000

    # === Corpus & Search ===
    corpus_dir: str = "."
    scope_dir: str = ""

    # === Model Server ===
    model_server_script: str = ""
    index_dir: str = ""
    embed_model_path: str = DEFAULT_EMBED_MODEL
    qr_model_path: str = DEFAULT_QR_MODEL
    qr_heads: str = DEFAULT_QR_HEADS
    device: str = "cuda:0"
    embed_top_k: int = 5000
    no_reranker: bool = False

    # === Bash Tool (rg post-processing) ===
    rg_max_matches: int = 30         # same as TS RG_MAX_MATCHES
    grep_max_line_length: int = 1000  # GREP_MAX_LINE_LENGTH
    paragraph_rerank_model: str = "qr"
    paragraph_rerank_doc_limit: int = 0
    rg_rerank_match: bool = False
    rg_rerank_model: str = "qr"
    use_contextual_bash_tool: bool = False
    rg_rerank_candidate_num: int = 200
    rg_rerank_timeout: int = 90

    # === Read Tool ===
    read_max_lines: int = 2000
    read_max_bytes: int = 50 * 1024

    # === Output ===
    output_dir: str = ""
    question: str = ""
    resume: bool = False

    # === Misc ===
    scope_size_prefix: str = ""
    scope_path_dot_prefix: bool = False
    sample_id: str = ""
    system_prompt_file: str = ""
    append_system_prompt_file: str = ""
    verbose: bool = False


def build_config_from_args(args: Optional[argparse.Namespace] = None) -> AgentConfig:
    if args is None:
        args = parse_cli_args()

    config = AgentConfig()

    config.provider = args.provider or _env("PROVIDER", config.provider)
    config.model = args.model or _env("MODEL", config.model)
    config.base_url = (
        args.base_url
        or _env("LLM_BASE_URL")
        or _env("OPENAI_BASE_URL")
        or config.base_url
    )
    config.api_key = (
        args.api_key
        or _env("LLM_API_KEY")
        or _env("OPENAI_API_KEY")
        or config.api_key
    )
    config.temperature = args.temperature if args.temperature is not None else 0.0
    config.max_output_tokens = args.max_output_tokens or _env_int("MAX_OUTPUT_TOKENS", config.max_output_tokens)
    config.thinking_level = args.thinking_level or _env("THINKING_LEVEL", config.thinking_level)

    config.max_turns = args.max_turns or _env_int("MAX_TURNS", config.max_turns)
    config.force_answer_turns = args.force_answer_turns or _env_int("FORCE_ANSWER_TURNS", config.force_answer_turns)
    config.keep_tool_calls = args.keep_tool_calls or _env_int("KEEP_TOOL_CALLS", config.keep_tool_calls)
    config.compaction_threshold = args.compaction_threshold or _env_int("COMPACTION_THRESHOLD", config.compaction_threshold)
    config.use_usage_based_compaction = args.use_usage_based_compaction or _env_bool("USE_USAGE_BASED_COMPACTION", config.use_usage_based_compaction)
    config.usage_based_compaction_min_tokens = (
        args.usage_based_compaction_min_tokens
        or _env_int("USAGE_BASED_COMPACTION_MIN_TOKENS", config.usage_based_compaction_min_tokens)
    )
    config.usage_based_compaction_max_tokens = (
        args.usage_based_compaction_max_tokens
        or _env_int("USAGE_BASED_COMPACTION_MAX_TOKENS", config.usage_based_compaction_max_tokens)
    )

    config.corpus_dir = args.corpus_dir or _env("CORPUS_DIR", config.corpus_dir)
    config.scope_dir = args.scope_dir or _env("EMBED_SCOPE_DIR", "")

    config.model_server_script = args.model_server_script or _env(
        "MODEL_SERVER_SCRIPT", str(_REPO_ROOT / "scripts" / "model_server.py")
    )
    config.index_dir = args.index_dir or _env("EMBED_INDEX_DIR", str(_REPO_ROOT / "data" / "indices" / "bc_plus_100k"))
    config.embed_model_path = args.embed_model_path or _env("EMBED_MODEL_PATH", config.embed_model_path)
    config.qr_model_path = args.qr_model_path or _env("QR_MODEL_PATH", config.qr_model_path)
    config.qr_heads = args.qr_heads or _env("QR_HEADS", config.qr_heads)
    config.device = args.device or _env("DEVICE", config.device)
    config.embed_top_k = args.embed_top_k or _env_int("EMBED_TOP_K", config.embed_top_k)
    config.no_reranker = args.no_reranker

    config.rg_max_matches = _env_int("RG_MAX_MATCHES", config.rg_max_matches)
    config.grep_max_line_length = _env_int("GREP_MAX_LINE_LENGTH", config.grep_max_line_length)
    config.paragraph_rerank_model = (_env("PARAGRAPH_RERANK_MODEL", config.paragraph_rerank_model) or config.paragraph_rerank_model).strip().lower()
    if config.paragraph_rerank_model not in {"qr", "emb", "none"}:
        config.paragraph_rerank_model = "qr"
    config.paragraph_rerank_doc_limit = _env_int("PARAGRAPH_RERANK_DOC_LIMIT", config.paragraph_rerank_doc_limit)
    config.rg_rerank_match = _env_bool("RG_RERANK_MATCH", config.rg_rerank_match)
    config.rg_rerank_model = (_env("RG_RERANK_MODEL", config.rg_rerank_model) or config.rg_rerank_model).strip().lower()
    if config.rg_rerank_model not in {"qr", "emb", "qwen3emb"}:
        config.rg_rerank_model = "qr"
    config.use_contextual_bash_tool = _env_bool("USE_CONTEXTUAL_BASH_TOOL", config.use_contextual_bash_tool)
    config.rg_rerank_candidate_num = _env_int("RG_RERANK_CANDIDATE_NUM", config.rg_rerank_candidate_num)
    config.rg_rerank_timeout = _env_int("RG_RERANK_TIMEOUT", config.rg_rerank_timeout)

    config.read_max_lines = _env_int("READ_MAX_LINES", config.read_max_lines)
    config.read_max_bytes = _env_int("READ_MAX_BYTES", config.read_max_bytes)

    config.output_dir = args.output_dir or _env("OUTPUT_DIR", "")
    config.question = args.question or ""
    config.resume = args.resume

    config.scope_size_prefix = _env("SCOPE_SIZE_PREFIX", "")
    config.scope_path_dot_prefix = _env_bool("SCOPE_PATH_DOT_PREFIX", config.scope_path_dot_prefix)
    config.sample_id = _env("SAMPLE_ID", "")
    config.system_prompt_file = getattr(args, "system_prompt_file", "") or ""
    config.append_system_prompt_file = getattr(args, "append_system_prompt_file", "") or ""
    config.verbose = args.verbose

    if not config.scope_dir:
        config.scope_dir = config.output_dir or "."

    return config


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TS-mirror Python Agent")

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
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()
