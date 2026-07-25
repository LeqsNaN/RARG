#!/usr/bin/env python3
"""
Evaluate pure embedding recall on BRIGHT without any agent / reranking flow.

This script:
  1. loads a prebuilt FAISS index for each BRIGHT subset
  2. encodes each query exactly once with the specified embedding model
  3. retrieves top-k documents from the index
  4. computes mean nDCG@10/20/50/100 (or custom ks)

Supported embedding families:
  - Qwen3-Embedding-4B
  - llama-nv-embed-reasoning-3b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import faiss
import numpy as np

SCRIPT_PATH = Path(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from embedding_backends import (  # type: ignore
    DEFAULT_LLAMA_NV_REASONING_EMBED_MODEL,
    DEFAULT_QWEN3_EMBED_MODEL,
    create_vllm_embedder,
    encode_batch_with_vllm,
    format_query_text,
    get_embedding_spec,
    normalize_backend_override,
    normalize_model_type,
    pool_hidden_states,
    preferred_torch_dtype,
    read_index_meta,
    torch_embeddings_to_numpy,
)


TASK_QUERY_INSTRUCTIONS: Dict[str, str] = {
    "biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "earth_science": "Given an Earth Science post, retrieve relevant passages that help answer the post.",
    "economics": "Given an Economics post, retrieve relevant passages that help answer the post.",
    "robotics": "Given a Robotics post, retrieve relevant passages that help answer the post.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pure embedding recall on BRIGHT.")
    parser.add_argument(
        "--index-root",
        type=Path,
        required=True,
        help="Root directory containing per-subset FAISS indices.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "bright" / "queries",
        help="Directory containing BRIGHT subset query jsonl files.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="biology,earth_science,economics,robotics",
        help="Comma-separated BRIGHT subsets to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Embedding model path. If omitted, inferred from --model-type.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        help="Embedding family: auto / qwen3_embedding_4b / llama_nv_embed_reasoning_3b.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="Embedding runtime backend: auto / transformers / vllm.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Torch device for Transformers fallback, e.g. cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Query batch size.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="Override max token length. 0 = read from index meta if available.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="10,20,50,100",
        help="Comma-separated k values for nDCG.",
    )
    parser.add_argument(
        "--recall-ks",
        type=str,
        default="10,50,100,1000,5000,10000",
        help="Comma-separated k values for recall.",
    )
    parser.add_argument(
        "--limit-per-subset",
        type=int,
        default=0,
        help="For quick smoke tests only: cap the number of queries per subset. 0 = full set.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save detailed metrics json.",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_k_values(value: str) -> List[int]:
    ks = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"Invalid --ks: {value}")
    return ks


def normalize_path(path: str) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_ndcg_at_k(retrieved: Sequence[str], gold_set: set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc in enumerate(retrieved[:k])
        if doc in gold_set
    )
    ideal_k = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall_at_k(retrieved: Sequence[str], gold_set: set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    top_k = {doc for doc in retrieved[:k]}
    return len(top_k & gold_set) / len(gold_set)


def resolve_model_path(model_type: str, model_path: str) -> str:
    if model_path:
        return model_path
    normalized = normalize_model_type(model_type)
    if normalized == "auto":
        return DEFAULT_QWEN3_EMBED_MODEL
    if normalized == "llama_nv_embed_reasoning_3b":
        return DEFAULT_LLAMA_NV_REASONING_EMBED_MODEL
    return DEFAULT_QWEN3_EMBED_MODEL


def resolve_transformers_device(arg_device: str) -> str:
    if arg_device:
        return arg_device
    env_device = os.environ.get("EMBED_DEVICE", "").strip()
    if env_device:
        return env_device
    return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def resolve_max_model_len(index_meta: Dict[str, Any], arg_max_model_len: int) -> int:
    if arg_max_model_len and arg_max_model_len > 0:
        return arg_max_model_len
    meta_value = int(index_meta.get("max_model_len", 0) or 0)
    if meta_value > 0:
        return meta_value
    return 512


def load_transformers_embedder(model_path: str, spec, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch_device = torch.device(device)
    dtype = preferred_torch_dtype(spec, torch_device)
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


def encode_batch_with_transformers(tokenizer, model, device, texts: List[str], max_model_len: int, pooling: str) -> np.ndarray:
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


@dataclass
class QueryEncoder:
    model_path: str
    model_type: str
    backend: str
    max_model_len: int
    spec: Any
    engine: Any

    @classmethod
    def create(
        cls,
        *,
        model_path: str,
        model_type: str,
        backend: str,
        index_dir: Path,
        device: str,
        max_model_len: int,
    ) -> "QueryEncoder":
        spec = get_embedding_spec(
            model_path=model_path,
            explicit_model_type=normalize_model_type(model_type),
            index_dir=str(index_dir),
        )
        requested_backend = normalize_backend_override(backend)
        if requested_backend == "auto":
            requested_backend = spec.backend

        if requested_backend == "vllm":
            try:
                print("Loading query encoder with vLLM...", flush=True)
                engine = create_vllm_embedder(
                    model_path=model_path,
                    max_model_len=max_model_len,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.9,
                )
                return cls(
                    model_path=model_path,
                    model_type=spec.model_type,
                    backend="vllm",
                    max_model_len=max_model_len,
                    spec=spec,
                    engine=engine,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] vLLM query encoder init failed, fallback to Transformers: {e}", flush=True)

        print("Loading query encoder with Transformers...", flush=True)
        engine = load_transformers_embedder(model_path, spec, resolve_transformers_device(device))
        return cls(
            model_path=model_path,
            model_type=spec.model_type,
            backend="transformers",
            max_model_len=max_model_len,
            spec=spec,
            engine=engine,
        )

    def encode_queries(self, queries: List[str], query_instruction: str) -> np.ndarray:
        formatted = [format_query_text(q, self.spec, query_instruction) for q in queries]
        if self.backend == "vllm":
            return encode_batch_with_vllm(self.engine, formatted, self.max_model_len)
        tokenizer, model, device = self.engine
        return encode_batch_with_transformers(
            tokenizer,
            model,
            device,
            formatted,
            self.max_model_len,
            self.spec.pooling,
        )


def load_index(index_dir: Path) -> tuple[faiss.Index, List[str], Dict[str, Any]]:
    index_path = index_dir / "index.faiss"
    paths_path = index_dir / "paths.json"
    meta = read_index_meta(str(index_dir))
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    if not paths_path.exists():
        raise FileNotFoundError(f"Missing paths file: {paths_path}")
    index = faiss.read_index(str(index_path))
    with paths_path.open("r", encoding="utf-8") as f:
        paths = json.load(f)
    if index.ntotal != len(paths):
        raise ValueError(
            f"Index/path mismatch in {index_dir}: ntotal={index.ntotal}, len(paths)={len(paths)}"
        )
    return index, [normalize_path(p) for p in paths], meta


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def evaluate_subset(
    *,
    subset: str,
    dataset_path: Path,
    index_dir: Path,
    encoder: QueryEncoder,
    ndcg_ks: List[int],
    recall_ks: List[int],
    batch_size: int,
    limit_per_subset: int,
) -> Dict[str, Any]:
    rows = read_jsonl(dataset_path)
    if limit_per_subset > 0:
        rows = rows[:limit_per_subset]

    index, paths, meta = load_index(index_dir)
    max_k = max(max(ndcg_ks), max(recall_ks))
    query_instruction = TASK_QUERY_INSTRUCTIONS.get(
        subset,
        "Given the following post, retrieve relevant passages that help answer the post.",
    )

    ndcg_sums = {k: 0.0 for k in ndcg_ks}
    recall_sums = {k: 0.0 for k in recall_ks}
    query_count = 0
    started = time.perf_counter()

    for batch_rows in batched(rows, batch_size):
        queries = [str(row.get("query", row.get("question", ""))) for row in batch_rows]
        query_vecs = encoder.encode_queries(list(queries), query_instruction)
        scores, indices = index.search(query_vecs.astype(np.float32), max_k)

        for row, retrieved_idx in zip(batch_rows, indices):
            gold_docs = row.get("gold_docs") or row.get("gold_ids") or []
            gold_set = {normalize_path(x) for x in gold_docs}
            excluded = {
                normalize_path(x)
                for x in row.get("excluded_ids", [])
                if x and x != "N/A"
            }
            retrieved = [
                paths[i]
                for i in retrieved_idx
                if i >= 0 and 0 <= i < len(paths) and paths[i] not in excluded
            ]
            for k in ndcg_ks:
                ndcg_sums[k] += compute_ndcg_at_k(retrieved, gold_set, k)
            for k in recall_ks:
                recall_sums[k] += compute_recall_at_k(retrieved, gold_set, k)
            query_count += 1

    elapsed = time.perf_counter() - started
    result = {
        "subset": subset,
        "dataset": str(dataset_path),
        "index_dir": str(index_dir),
        "index_documents": len(paths),
        "index_embedding_dim": int(index.d),
        "index_meta": meta,
        "num_queries": query_count,
        "elapsed_seconds": round(elapsed, 3),
        "queries_per_second": round(query_count / max(elapsed, 1e-8), 3),
    }
    for k in ndcg_ks:
        result[f"ndcg@{k}"] = ndcg_sums[k] / max(query_count, 1)
    for k in recall_ks:
        result[f"recall@{k}"] = recall_sums[k] / max(query_count, 1)
    return result


def aggregate_overall(subset_results: List[Dict[str, Any]], ndcg_ks: List[int], recall_ks: List[int]) -> Dict[str, Any]:
    total_queries = sum(int(item["num_queries"]) for item in subset_results)
    overall: Dict[str, Any] = {"num_queries": total_queries}
    for k in ndcg_ks:
        numerator = sum(float(item[f"ndcg@{k}"]) * int(item["num_queries"]) for item in subset_results)
        overall[f"ndcg@{k}"] = numerator / max(total_queries, 1)
    for k in recall_ks:
        numerator = sum(float(item[f"recall@{k}"]) * int(item["num_queries"]) for item in subset_results)
        overall[f"recall@{k}"] = numerator / max(total_queries, 1)
    return overall


def _print_metric_table(
    *,
    subset_results: List[Dict[str, Any]],
    overall: Dict[str, Any],
    metric_name: str,
    ks: List[int],
) -> None:
    headers = ["subset", "queries"] + [f"{metric_name}@{k}" for k in ks]
    rows: List[List[str]] = []
    for item in subset_results:
        rows.append(
            [item["subset"], str(item["num_queries"])]
            + [f"{float(item[f'{metric_name}@{k}']):.4f}" for k in ks]
        )
    rows.append(
        ["overall", str(overall["num_queries"])]
        + [f"{float(overall[f'{metric_name}@{k}']):.4f}" for k in ks]
    )

    widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(len(headers))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)

    print(f"{metric_name.upper()} summary:")
    print("")
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))
    print("")


def print_summary(
    subset_results: List[Dict[str, Any]],
    overall: Dict[str, Any],
    ndcg_ks: List[int],
    recall_ks: List[int],
) -> None:
    _print_metric_table(
        subset_results=subset_results,
        overall=overall,
        metric_name="ndcg",
        ks=ndcg_ks,
    )
    _print_metric_table(
        subset_results=subset_results,
        overall=overall,
        metric_name="recall",
        ks=recall_ks,
    )


def main() -> int:
    args = parse_args()
    subsets = parse_csv_list(args.subsets)
    ndcg_ks = parse_k_values(args.ks)
    recall_ks = parse_k_values(args.recall_ks)
    model_path = resolve_model_path(args.model_type, args.model_path)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    if not args.index_root.exists():
        raise FileNotFoundError(f"Index root does not exist: {args.index_root}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.dataset_root}")

    first_subset = subsets[0]
    first_index_dir = args.index_root / first_subset
    first_meta = read_index_meta(str(first_index_dir))
    max_model_len = resolve_max_model_len(first_meta, args.max_model_len)

    print("=== BRIGHT pure embedding recall evaluation ===")
    print(f"index_root:     {args.index_root}")
    print(f"dataset_root:   {args.dataset_root}")
    print(f"subsets:        {subsets}")
    print(f"model_path:     {model_path}")
    print(f"model_type:     {args.model_type}")
    print(f"backend:        {args.backend}")
    print(f"device:         {args.device or resolve_transformers_device(args.device)}")
    print(f"batch_size:     {args.batch_size}")
    print(f"max_model_len:  {max_model_len}")
    print(f"ndcg ks:        {ndcg_ks}")
    print(f"recall ks:      {recall_ks}")
    if args.limit_per_subset > 0:
        print(f"limit/subset:   {args.limit_per_subset}")

    encoder = QueryEncoder.create(
        model_path=model_path,
        model_type=args.model_type,
        backend=args.backend,
        index_dir=first_index_dir,
        device=args.device,
        max_model_len=max_model_len,
    )

    subset_results: List[Dict[str, Any]] = []
    for subset in subsets:
        dataset_path = args.dataset_root / f"{subset}.jsonl"
        index_dir = args.index_root / subset
        print(f"\n[subset={subset}] evaluating...", flush=True)
        result = evaluate_subset(
            subset=subset,
            dataset_path=dataset_path,
            index_dir=index_dir,
            encoder=encoder,
            ndcg_ks=ndcg_ks,
            recall_ks=recall_ks,
            batch_size=args.batch_size,
            limit_per_subset=args.limit_per_subset,
        )
        subset_results.append(result)
        metrics = ", ".join(
            [f"nDCG@{k}={result[f'ndcg@{k}']:.4f}" for k in ndcg_ks]
            + [f"Recall@{k}={result[f'recall@{k}']:.4f}" for k in recall_ks]
        )
        print(f"[subset={subset}] done: queries={result['num_queries']}, {metrics}", flush=True)

    overall = aggregate_overall(subset_results, ndcg_ks, recall_ks)
    print_summary(subset_results, overall, ndcg_ks, recall_ks)

    output = {
        "index_root": str(args.index_root),
        "dataset_root": str(args.dataset_root),
        "subsets": subsets,
        "model_path": model_path,
        "requested_model_type": args.model_type,
        "resolved_model_type": encoder.model_type,
        "backend": encoder.backend,
        "batch_size": args.batch_size,
        "max_model_len": max_model_len,
        "ndcg_ks": ndcg_ks,
        "recall_ks": recall_ks,
        "subset_results": subset_results,
        "overall": overall,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics json to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
