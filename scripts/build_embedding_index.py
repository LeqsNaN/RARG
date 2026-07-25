#!/usr/bin/env python3
"""
Build a FAISS index for a document corpus using a supported embedding model.

Supports:
  - Qwen3-Embedding-4B via vLLM embedding runner
  - llama-nv-embed-reasoning-3b via Transformers + average pooling

The implementation is streaming and only keeps one embedding batch in memory
at a time, which is important for large corpora such as bc_plus_1m.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

import faiss
import numpy as np

from embedding_backends import (
    DEFAULT_QWEN3_EMBED_MODEL,
    create_vllm_embedder,
    encode_batch_with_vllm,
    format_passage_text,
    get_embedding_spec,
    normalize_backend_override,
    normalize_model_type,
    pool_hidden_states,
    preferred_torch_dtype,
    probe_vllm_embedder,
    torch_embeddings_to_numpy,
)


DEFAULT_MODEL_PATH = DEFAULT_QWEN3_EMBED_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS embedding index using a supported embedding backend.")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        required=True,
        help="Root directory of the document corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save index.faiss and paths.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Model path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        help="Embedding model family: auto, qwen3_embedding_4b, llama_nv_embed_reasoning_3b.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="Embedding runtime backend override: auto, transformers, vllm.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Maximum characters to pre-read from each document (0 = no limit, let tokenizer handle truncation).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum token length for the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--file-extensions",
        type=str,
        default=".txt,.html,.md",
        help="Comma-separated file extensions to include (default: .txt,.html,.md).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism when using the vLLM backend.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Torch device for Transformers backend (default: EMBED_DEVICE env, else auto).",
    )
    parser.add_argument(
        "--scan-progress-every",
        type=int,
        default=10000,
        help="Print scan progress every N discovered files.",
    )
    parser.add_argument(
        "--encode-progress-every",
        type=int,
        default=10000,
        help="Print encode progress every N encoded documents.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Current shard id for multi-process indexing (default: 0).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards for multi-process indexing (default: 1).",
    )
    return parser.parse_args()


def iter_file_paths(corpus_dir: Path, extensions: Sequence[str]) -> Iterator[Path]:
    """Yield matching document file paths without materializing the whole corpus list."""
    ext_set = set(extensions)
    stack = [corpus_dir]

    while stack:
        current = stack.pop()
        try:
            dirs: list[Path] = []
            files: list[Path] = []
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False) and any(entry.name.endswith(ext) for ext in ext_set):
                            files.append(Path(entry.path))
                    except OSError:
                        continue
            for file_path in sorted(files, key=lambda p: p.name):
                yield file_path
            for dir_path in sorted(dirs, key=lambda p: p.name, reverse=True):
                stack.append(dir_path)
        except OSError:
            continue


def file_belongs_to_shard(file_ordinal: int, shard_index: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard config: shard_index={shard_index}, num_shards={num_shards}")
    return (file_ordinal % num_shards) == shard_index


def read_document_text(path: Path, max_chars: int) -> str:
    """Read document text. If max_chars > 0, truncate; otherwise read all."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            if max_chars > 0:
                return f.read(max_chars)
            return f.read()
    except (IOError, OSError):
        return ""


def create_transformers_embedder(model_path: str, spec, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch_device = torch.device(device)
    dtype = preferred_torch_dtype(spec, torch_device)

    print(f"Loading model with Transformers: {model_path}")
    print(
        f"  device={torch_device}, dtype={dtype}, pooling={spec.pooling}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side=spec.tokenizer_padding_side,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(torch_device).eval()
    return tokenizer, model, torch_device


def encode_batch_with_transformers(tokenizer, model, device, texts: list[str], max_model_len: int, pooling: str) -> np.ndarray:
    import torch

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_model_len,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], pooling)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return torch_embeddings_to_numpy(embeddings)


def flush_batch(
    *,
    embedder,
    backend: str,
    spec,
    index: faiss.Index | None,
    pending_texts: list[str],
    pending_paths: list[str],
    valid_paths: list[str],
    max_model_len: int,
) -> tuple[faiss.Index, int]:
    if backend == "vllm":
        embeddings = encode_batch_with_vllm(embedder, pending_texts, max_model_len)
    else:
        tokenizer, model, device = embedder
        embeddings = encode_batch_with_transformers(
            tokenizer,
            model,
            device,
            pending_texts,
            max_model_len,
            spec.pooling,
        )
    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    valid_paths.extend(pending_paths)
    count = len(pending_paths)
    pending_texts.clear()
    pending_paths.clear()
    return index, count


def resolve_transformers_device(arg_device: str) -> str:
    if arg_device:
        return arg_device
    env_device = os.environ.get("EMBED_DEVICE", "").strip()
    if env_device:
        return env_device
    return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def resolve_backend(spec, backend_override: str) -> str:
    normalized = normalize_backend_override(backend_override)
    if normalized == "auto":
        return spec.backend
    return normalized


def create_embedder_with_fallback(args, spec):
    requested_backend = resolve_backend(spec, args.backend)
    explicit_backend = normalize_backend_override(args.backend)

    if requested_backend == "vllm":
        print("Trying vLLM backend...", flush=True)
        try:
            embedder = create_vllm_embedder(
                model_path=args.model,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            probe_dim = probe_vllm_embedder(
                embedder,
                probe_text=format_passage_text("vllm backend probe", spec),
                max_model_len=args.max_model_len,
            )
            print(f"  vLLM probe succeeded, embedding dim={probe_dim}", flush=True)
            return embedder, "vllm"
        except Exception as e:
            print(f"[WARN] vLLM backend failed, falling back to Transformers: {e}", flush=True)

    embedder = create_transformers_embedder(
        args.model,
        spec,
        device=resolve_transformers_device(args.device),
    )
    return embedder, "transformers"


def main() -> int:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    extensions = [ext.strip() for ext in args.file_extensions.split(",") if ext.strip()]

    spec = get_embedding_spec(
        model_path=args.model,
        explicit_model_type=normalize_model_type(args.model_type),
        index_dir=None,
    )
    requested_backend = resolve_backend(spec, args.backend)

    if not corpus_dir.exists():
        print(f"Error: corpus directory does not exist: {corpus_dir}", file=sys.stderr)
        return 1

    if args.batch_size <= 0:
        print("Error: --batch-size must be > 0", file=sys.stderr)
        return 1
    if args.num_shards <= 0:
        print("Error: --num-shards must be > 0", file=sys.stderr)
        return 1
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        print("Error: --shard-index must satisfy 0 <= shard-index < num-shards", file=sys.stderr)
        return 1

    print(f"Streaming corpus: {corpus_dir}")
    print(f"Extensions: {extensions}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model path: {args.model}")
    print(f"Model type: {spec.model_type}")
    print(f"Backend:    {requested_backend}")
    print(f"Pooling:    {spec.pooling}")
    print(f"Shard:      {args.shard_index + 1}/{args.num_shards}")
    start_time = time.perf_counter()

    embedder, actual_backend = create_embedder_with_fallback(args, spec)

    scanned_files = 0
    valid_docs = 0
    pending_texts: list[str] = []
    pending_paths: list[str] = []
    valid_paths: list[str] = []
    index: faiss.Index | None = None
    first_encode_start: float | None = None
    encode_time_total = 0.0
    eligible_files_seen = 0

    for fpath in iter_file_paths(corpus_dir, extensions):
        scanned_files += 1
        if scanned_files % args.scan_progress_every == 0:
            print(
                f"Scanned {scanned_files} files; encoded {valid_docs} non-empty docs so far...",
                flush=True,
            )

        relpath = str(fpath.relative_to(corpus_dir))
        current_ordinal = eligible_files_seen
        eligible_files_seen += 1
        if not file_belongs_to_shard(current_ordinal, args.shard_index, args.num_shards):
            continue
        text = read_document_text(fpath, args.max_chars)
        if not text.strip():
            continue

        pending_texts.append(format_passage_text(text, spec))
        pending_paths.append(relpath)

        if len(pending_texts) >= args.batch_size:
            if first_encode_start is None:
                first_encode_start = time.perf_counter()
            batch_start = time.perf_counter()
            index, added = flush_batch(
                embedder=embedder,
                backend=actual_backend,
                spec=spec,
                index=index,
                pending_texts=pending_texts,
                pending_paths=pending_paths,
                valid_paths=valid_paths,
                max_model_len=args.max_model_len,
            )
            encode_time_total += time.perf_counter() - batch_start
            valid_docs += added
            if valid_docs % args.encode_progress_every == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"Encoded {valid_docs} docs from {scanned_files} scanned files "
                    f"({valid_docs / max(elapsed, 1e-8):.1f} docs/s end-to-end)",
                    flush=True,
                )

    if pending_texts:
        if first_encode_start is None:
            first_encode_start = time.perf_counter()
        batch_start = time.perf_counter()
        index, added = flush_batch(
            embedder=embedder,
            backend=actual_backend,
            spec=spec,
            index=index,
            pending_texts=pending_texts,
            pending_paths=pending_paths,
            valid_paths=valid_paths,
            max_model_len=args.max_model_len,
        )
        encode_time_total += time.perf_counter() - batch_start
        valid_docs += added

    if index is None or valid_docs == 0:
        print("Error: no non-empty documents were indexed.", file=sys.stderr)
        return 1

    print(f"Finished scan: {scanned_files} files visited, {valid_docs} non-empty docs indexed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    paths_path = output_dir / "paths.json"
    meta_path = output_dir / "meta.json"

    print(f"Saving to: {output_dir}")
    faiss.write_index(index, str(index_path))
    with open(paths_path, "w", encoding="utf-8") as f:
        json.dump(valid_paths, f, ensure_ascii=False)

    total_time = time.perf_counter() - start_time
    pure_encode_docs_per_second = valid_docs / max(encode_time_total, 1e-8)
    end_to_end_docs_per_second = valid_docs / max(total_time, 1e-8)
    index_size_mb = index_path.stat().st_size / (1024 * 1024)

    meta = {
        "model": args.model,
        "embed_model_type": spec.model_type,
        "backend": actual_backend,
        "max_chars": args.max_chars,
        "max_model_len": args.max_model_len,
        "corpus_dir": str(corpus_dir),
        "num_scanned_files": scanned_files,
        "num_documents": len(valid_paths),
        "embedding_dim": index.d,
        "index_type": "IndexFlatIP",
        "file_extensions": extensions,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "pooling": spec.pooling,
        "query_prefix_style": spec.query_style,
        "query_prefix": spec.query_prefix,
        "passage_prefix": spec.passage_prefix,
        "encode_time_seconds": round(encode_time_total, 1),
        "pure_encode_docs_per_second": round(pure_encode_docs_per_second, 1),
        "end_to_end_docs_per_second": round(end_to_end_docs_per_second, 1),
        "total_time_seconds": round(total_time, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nIndex built successfully:")
    print(f"  Scanned files: {scanned_files}")
    print(f"  Indexed docs:  {len(valid_paths)}")
    print(f"  Dimension:     {index.d}")
    print(f"  Index size:    {index_size_mb:.1f} MB")
    print(f"  Encode speed:  {pure_encode_docs_per_second:.1f} docs/s (pure encode)")
    print(f"  Overall speed: {end_to_end_docs_per_second:.1f} docs/s")
    print(f"  Total time:    {total_time:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
