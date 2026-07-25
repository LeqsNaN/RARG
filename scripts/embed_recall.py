#!/usr/bin/env python3
"""
Embedding-based document recall tool for DCI-Agent-Lite.

Uses FAISS index + Qwen3-Embedding-4B (last-token pooling) to recall
relevant documents. Called by LLM via bash tool.

Usage:
    python embed_recall.py --query "search keywords" --top-k 5000
    python embed_recall.py --query "another query" --top-k 3000 --index-dir ./data/indices/bc_plus_100k

Each call generates a numbered scope file (scope_1.txt, scope_2.txt, ...).
Then search within the recalled subset:
    cat scope_N.txt | xargs -d '\\n' rg "pattern"
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

# Suppress transformers progress bars (Loading checkpoint shards...)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Limit OpenBLAS threads to avoid "too many memory regions" error with FAISS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, logging as tf_logging

# Disable transformers logging (progress bars, warnings)
tf_logging.set_verbosity_error()


DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-Embedding-4B")


def _scope_path_dot_prefix_enabled() -> bool:
    return os.environ.get("SCOPE_PATH_DOT_PREFIX", "").strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedding-based document recall. Generates a scope file for use with rg."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query for semantic recall.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5000,
        help="Number of documents to recall (default: 5000).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Directory containing index.faiss and paths.json. "
        "If not specified, uses EMBED_INDEX_DIR env var or ./data/indices/bc_plus_100k.",
    )
    parser.add_argument(
        "--scope-dir",
        type=str,
        default=".",
        help="Directory where scope files are written (default: current directory).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model path for query encoding. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for query tokenization (default: 512).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda', 'cpu', or None for auto.",
    )
    return parser.parse_args()


def find_next_scope_number(scope_dir: Path) -> int:
    """Detect existing scope files and return the next number."""
    existing = glob.glob(str(scope_dir / "scope_*.txt"))
    if not existing:
        return 1
    numbers: List[int] = []
    for f in existing:
        stem = Path(f).stem
        parts = stem.split("_")
        if len(parts) == 2:
            try:
                numbers.append(int(parts[1]))
            except ValueError:
                continue
    return max(numbers, default=0) + 1


def load_faiss_index(index_dir: str):
    """Load FAISS index and paths."""
    index_path = os.path.join(index_dir, "index.faiss")
    paths_path = os.path.join(index_dir, "paths.json")

    if not os.path.exists(index_path):
        print(f"Error: FAISS index not found: {index_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(paths_path):
        print(f"Error: paths file not found: {paths_path}", file=sys.stderr)
        sys.exit(1)

    index = faiss.read_index(index_path)
    with open(paths_path, "r", encoding="utf-8") as f:
        paths = json.load(f)

    return index, paths


QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def format_query_with_instruction(query: str) -> str:
    """Add instruction prefix for Qwen3-Embedding query encoding."""
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"


def last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract the last non-padding token's hidden state."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, sequence_lengths]


@torch.no_grad()
def encode_query(
    query: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_tokens: int,
    device: torch.device,
) -> np.ndarray:
    """Encode a single query using last-token pooling, return normalized vector."""
    inputs = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)
    embedding = last_token_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy().astype(np.float32)


def write_scope_file(scope_file: Path, paths: List[str]) -> None:
    """Write recalled document paths to scope file."""
    with open(scope_file, "w", encoding="utf-8") as f:
        for p in paths:
            path_text = p
            if _scope_path_dot_prefix_enabled() and path_text and not path_text.startswith("/") and not path_text.startswith("./"):
                path_text = f"./{path_text}"
            f.write(path_text + "\n")


def write_scope_meta(meta_file: Path, query: str, top_k: int, doc_count: int, elapsed_ms: float) -> None:
    """Write metadata about this scope."""
    meta = {
        "query": query,
        "top_k": top_k,
        "doc_count": doc_count,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def get_scope_mapping(scope_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read all existing scope files and their metadata."""
    mapping: Dict[str, Dict[str, Any]] = {}
    for f in sorted(glob.glob(str(scope_dir / "scope_*.txt"))):
        p = Path(f)
        meta_file = p.with_suffix(".meta")
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                mapping[p.name] = {
                    "docs": meta.get("doc_count", 0),
                    "query": meta.get("query", "(unknown)"),
                }
            except (json.JSONDecodeError, IOError):
                doc_count = sum(1 for _ in open(f))
                mapping[p.name] = {"docs": doc_count, "query": "(unknown)"}
        else:
            doc_count = sum(1 for _ in open(f))
            mapping[p.name] = {"docs": doc_count, "query": "(unknown)"}
    return mapping


def resolve_index_dir(args: argparse.Namespace) -> str:
    """Resolve index directory."""
    if args.index_dir:
        return args.index_dir
    env_dir = os.environ.get("EMBED_INDEX_DIR")
    if env_dir:
        return env_dir
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default = repo_root / "data" / "indices" / "bc_plus_100k"
    return str(default)


def resolve_model_path(args: argparse.Namespace) -> str:
    """Resolve model path."""
    if args.model:
        return args.model
    env_model = os.environ.get("EMBED_MODEL_PATH")
    if env_model:
        return env_model
    return DEFAULT_MODEL_PATH


def resolve_scope_dir(original_scope_dir: Path) -> tuple:
    """
    Determine scope file output directory.

    If SCOPE_SIZE_PREFIX (100k/200k) and SAMPLE_ID env vars are set,
    write to /tmp/{prefix}/{sample_id}/ for shorter paths in conversation context.
    The original_scope_dir is used as backup destination.

    Returns: (scope_dir, backup_dir_or_None)
    """
    size_prefix = os.environ.get("SCOPE_SIZE_PREFIX", "").strip()
    sample_id = os.environ.get("SAMPLE_ID", "").strip()

    if size_prefix and sample_id:
        tmp_scope_dir = Path(f"/tmp/{size_prefix}/{sample_id}")
        tmp_scope_dir.mkdir(parents=True, exist_ok=True)
        return tmp_scope_dir, original_scope_dir

    return original_scope_dir, None


def main() -> int:
    args = parse_args()
    original_scope_dir = Path(args.scope_dir).resolve()
    original_scope_dir.mkdir(parents=True, exist_ok=True)

    # Resolve actual scope output directory (may be /tmp/... for shorter paths)
    scope_dir, backup_dir = resolve_scope_dir(original_scope_dir)
    scope_dir.mkdir(parents=True, exist_ok=True)

    index_dir = resolve_index_dir(args)
    model_path = resolve_model_path(args)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.perf_counter()

    # Load FAISS index
    index, paths = load_faiss_index(index_dir)

    # Load model for query encoding
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device).eval()

    # Encode query (with instruction prefix per Qwen3-Embedding spec)
    formatted_query = format_query_with_instruction(args.query)
    query_vec = encode_query(formatted_query, tokenizer, model, args.max_tokens, device)

    # Search FAISS index
    top_k = min(args.top_k, index.ntotal)
    scores, indices = index.search(query_vec, top_k)

    # Get recalled paths
    recalled_paths = [paths[i] for i in indices[0] if i >= 0]

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Determine scope file number
    scope_num = find_next_scope_number(scope_dir)
    scope_file = scope_dir / f"scope_{scope_num}.txt"
    meta_file = scope_dir / f"scope_{scope_num}.meta"

    # Write scope file and metadata
    write_scope_file(scope_file, recalled_paths)
    write_scope_meta(meta_file, args.query, args.top_k, len(recalled_paths), elapsed_ms)

    # Backup to original outputs directory if using /tmp shortcut
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scope_file, backup_dir / scope_file.name)
        shutil.copy2(meta_file, backup_dir / meta_file.name)

    # Output for LLM
    mapping = get_scope_mapping(scope_dir)

    print(f'"{args.query}" → {scope_file} ({len(recalled_paths)} docs)')

    return 0


if __name__ == "__main__":
    sys.exit(main())
