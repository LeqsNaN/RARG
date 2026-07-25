#!/usr/bin/env python3
"""
Merge two compatible FAISS flat indices without re-embedding.

This is intended for the DCI-Agent-Lite embedding indices built by
`scripts/build_embedding_index.py`, which currently writes:
  - index.faiss
  - paths.json
  - meta.json

The merged index preserves vector order:
  merged_paths = paths_a + paths_b
and the merged FAISS index stores vectors in the same order.

Typical use:
    python scripts/merge_embedding_indices.py \
        --index-a-dir data/indices/bc_plus_100k \
        --index-b-dir data/indices/bc_plus_900k \
        --output-dir data/indices/bc_plus_1m \
        --corpus-dir corpus/bc_plus_1m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two compatible FAISS flat indices.")
    parser.add_argument("--index-a-dir", type=Path, required=True, help="First index directory.")
    parser.add_argument("--index-b-dir", type=Path, required=True, help="Second index directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Merged output directory.")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="",
        help="Final corpus root used by the merged index. "
        "If empty, defaults to meta.json corpus_dir from index-b when available.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Chunk size when copying vectors if in-place merge is unavailable.",
    )
    parser.add_argument(
        "--allow-duplicate-paths",
        action="store_true",
        help="Allow duplicate relative paths across the two indices.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_bundle(index_dir: Path) -> Tuple[Any, List[str], Dict[str, Any]]:
    index_path = index_dir / "index.faiss"
    paths_path = index_dir / "paths.json"
    meta_path = index_dir / "meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.faiss: {index_path}")
    if not paths_path.exists():
        raise FileNotFoundError(f"Missing paths.json: {paths_path}")

    index = faiss.read_index(str(index_path))
    paths = load_json(paths_path)
    if not isinstance(paths, list):
        raise ValueError(f"paths.json is not a list: {paths_path}")
    meta = load_json(meta_path) if meta_path.exists() else {}
    if not isinstance(meta, dict):
        meta = {}

    return index, paths, meta


def metric_name(metric_type: int) -> str:
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return "IP"
    if metric_type == faiss.METRIC_L2:
        return "L2"
    return str(metric_type)


def ensure_compatible(
    index_a: Any,
    paths_a: List[str],
    meta_a: Dict[str, Any],
    index_b: Any,
    paths_b: List[str],
    meta_b: Dict[str, Any],
    *,
    allow_duplicate_paths: bool,
) -> None:
    if index_a.d != index_b.d:
        raise ValueError(f"Embedding dim mismatch: {index_a.d} vs {index_b.d}")

    metric_a = getattr(index_a, "metric_type", None)
    metric_b = getattr(index_b, "metric_type", None)
    if metric_a != metric_b:
        raise ValueError(f"Metric mismatch: {metric_name(metric_a)} vs {metric_name(metric_b)}")

    if len(paths_a) != index_a.ntotal:
        raise ValueError(
            f"index-a paths count != ntotal: len(paths)={len(paths_a)} ntotal={index_a.ntotal}"
        )
    if len(paths_b) != index_b.ntotal:
        raise ValueError(
            f"index-b paths count != ntotal: len(paths)={len(paths_b)} ntotal={index_b.ntotal}"
        )

    model_a = meta_a.get("model")
    model_b = meta_b.get("model")
    if model_a and model_b and model_a != model_b:
        raise ValueError(f"Model mismatch: {model_a} vs {model_b}")

    if not allow_duplicate_paths:
        dup = set(paths_a) & set(paths_b)
        if dup:
            sample = sorted(dup)[:10]
            raise ValueError(
                f"Found {len(dup)} duplicate relative paths across the two indices. "
                f"Examples: {sample}"
            )

    cls_a = index_a.__class__.__name__
    cls_b = index_b.__class__.__name__
    if not (cls_a.startswith("IndexFlat") and cls_b.startswith("IndexFlat")):
        raise ValueError(
            "This utility currently supports flat FAISS indices only. "
            f"Got {cls_a} and {cls_b}."
        )


def empty_like(index: Any) -> Any:
    metric = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT)
    if metric == faiss.METRIC_INNER_PRODUCT:
        return faiss.IndexFlatIP(index.d)
    if metric == faiss.METRIC_L2:
        return faiss.IndexFlatL2(index.d)
    raise ValueError(f"Unsupported metric type for flat merge: {metric_name(metric)}")


def iter_flat_chunks(index: Any, chunk_size: int):
    flat = faiss.downcast_index(index)
    if not hasattr(flat, "get_xb"):
        raise ValueError(f"Index does not expose flat xb storage: {flat.__class__.__name__}")

    total = flat.ntotal
    xb = faiss.rev_swig_ptr(flat.get_xb(), total * flat.d).reshape(total, flat.d)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield start, end, xb[start:end].copy()


def merge_indices(index_a: Any, index_b: Any, chunk_size: int) -> Any:
    # Prefer in-place merge when available because it uses less extra memory.
    if hasattr(index_a, "merge_from"):
        try:
            index_a.merge_from(index_b, 0)
            return index_a
        except TypeError:
            index_a.merge_from(index_b)
            return index_a
        except Exception:
            # Fall back to explicit vector copy below.
            pass

    merged = empty_like(index_a)
    for _, _, chunk in iter_flat_chunks(index_a, chunk_size):
        merged.add(chunk)
    for _, _, chunk in iter_flat_chunks(index_b, chunk_size):
        merged.add(chunk)
    return merged


def main() -> int:
    args = parse_args()

    start_time = time.perf_counter()
    print(f"Loading index A from {args.index_a_dir}", flush=True)
    index_a, paths_a, meta_a = load_bundle(args.index_a_dir)
    print(f"  A: ntotal={index_a.ntotal}, dim={index_a.d}, metric={metric_name(index_a.metric_type)}", flush=True)

    print(f"Loading index B from {args.index_b_dir}", flush=True)
    index_b, paths_b, meta_b = load_bundle(args.index_b_dir)
    print(f"  B: ntotal={index_b.ntotal}, dim={index_b.d}, metric={metric_name(index_b.metric_type)}", flush=True)

    ensure_compatible(
        index_a,
        paths_a,
        meta_a,
        index_b,
        paths_b,
        meta_b,
        allow_duplicate_paths=args.allow_duplicate_paths,
    )

    merged_paths = paths_a + paths_b
    print(f"Merging vectors: {len(paths_a)} + {len(paths_b)} = {len(merged_paths)}", flush=True)
    merged_index = merge_indices(index_a, index_b, args.chunk_size)

    if merged_index.ntotal != len(merged_paths):
        raise ValueError(
            f"Merged ntotal mismatch: index.ntotal={merged_index.ntotal} vs paths={len(merged_paths)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.faiss"
    paths_path = args.output_dir / "paths.json"
    meta_path = args.output_dir / "meta.json"

    print(f"Writing merged index to {index_path}", flush=True)
    faiss.write_index(merged_index, str(index_path))
    with paths_path.open("w", encoding="utf-8") as f:
        json.dump(merged_paths, f, ensure_ascii=False)

    total_time = time.perf_counter() - start_time
    meta = {
        "merged": True,
        "merged_from": [str(args.index_a_dir), str(args.index_b_dir)],
        "corpus_dir": args.corpus_dir or meta_b.get("corpus_dir") or meta_a.get("corpus_dir") or "",
        "num_documents": len(merged_paths),
        "embedding_dim": merged_index.d,
        "index_type": merged_index.__class__.__name__,
        "metric_type": metric_name(merged_index.metric_type),
        "model": meta_a.get("model") or meta_b.get("model") or "",
        "embed_model_type": meta_a.get("embed_model_type") or meta_b.get("embed_model_type") or "",
        "backend": "merged",
        "max_model_len": meta_a.get("max_model_len", meta_b.get("max_model_len")),
        "max_chars": meta_a.get("max_chars", meta_b.get("max_chars")),
        "file_extensions": meta_a.get("file_extensions") or meta_b.get("file_extensions") or [],
        "pooling": meta_a.get("pooling") or meta_b.get("pooling") or "",
        "query_prefix_style": meta_a.get("query_prefix_style") or meta_b.get("query_prefix_style") or "",
        "query_prefix": meta_a.get("query_prefix") or meta_b.get("query_prefix") or "",
        "passage_prefix": meta_a.get("passage_prefix") or meta_b.get("passage_prefix") or "",
        "source_a_num_documents": len(paths_a),
        "source_b_num_documents": len(paths_b),
        "total_time_seconds": round(total_time, 1),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done.", flush=True)
    print(f"  Output dir:     {args.output_dir}", flush=True)
    print(f"  Documents:      {len(merged_paths)}", flush=True)
    print(f"  Embedding dim:  {merged_index.d}", flush=True)
    print(f"  Metric:         {metric_name(merged_index.metric_type)}", flush=True)
    print(f"  Total time:     {total_time:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
