#!/usr/bin/env python3
"""
Evaluate one-shot Qwen3 embedding recall on BrowseComp-Plus using gold_doc_ids.

This does NOT run the agent. It:
  1. maps BC+ parquet docid -> exported corpus relative path
  2. optionally maps FineWeb ids / fw_* basenames -> exported corpus relative path
  2. loads a prebuilt FAISS index + paths.json
  3. encodes each query once
  4. computes Recall@K against gold_doc_ids

The mapping is based on the same filename logic used by BC+ export scripts:
  relative_path = <domain>/<sanitized title or URL basename>.txt
with duplicate suffixes matched against paths.json when needed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlparse

# Avoid OpenBLAS / OMP over-threading on large parquet + FAISS jobs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("GOTO_NUM_THREADS", "1")

import faiss
import numpy as np
import pyarrow.parquet as pq

SCRIPT_PATH = Path(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from embedding_backends import (  # type: ignore
    DEFAULT_QUERY_INSTRUCTION,
    DEFAULT_QWEN3_EMBED_MODEL,
    format_query_text,
    get_embedding_spec,
    normalize_model_type,
    pool_hidden_states,
    torch_embeddings_to_numpy,
)

try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass


TITLE_RE = re.compile(r"(?mi)^title:\s*(.+?)\s*$")
INVALID_CHARS_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
WHITESPACE_RE = re.compile(r"\s+")
# Must match the exporter used for corpus/bc_plus_100k.  The original BC+
# exporter truncates sanitized filename stems to 140 chars.
MAX_STEM_LEN = 140


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BC+ Qwen3 embedding Recall@K.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "bcplus_qa_sample100.jsonl",
        help="BC+ QA jsonl containing query and gold_doc_ids.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=REPO_ROOT / "corpus" / "browsecomp_plus" / "data.parquet",
        help="Original BC+ parquet with docid/text/url.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=REPO_ROOT / "data" / "indices" / "bc_plus_100k",
        help="FAISS index directory containing index.faiss and paths.json.",
    )
    parser.add_argument(
        "--fineweb-sample",
        type=Path,
        default=REPO_ROOT / "data" / "fineweb_edu_10bt_sample900k.jsonl",
        help="Optional FineWeb JSONL used for bc_plus_1m corpus construction.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_QWEN3_EMBED_MODEL,
        help="Qwen3 embedding model path.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="qwen3_embedding_4b",
        help="Embedding model type; default qwen3_embedding_4b.",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float16,float32",
        help="Comma-separated torch dtypes to compare: float16,bfloat16,float32.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Torch device. Default: EMBED_DEVICE env, else cuda if visible, else cpu.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="Query max token length. Default: index meta max_model_len or 4096.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Query encoding batch size.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="10,20,50,100,1000,10000",
        help="Comma-separated Recall@K values.",
    )
    parser.add_argument(
        "--query-instruction",
        type=str,
        default=DEFAULT_QUERY_INSTRUCTION,
        help="Qwen3 query instruction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate first N queries; 0 = all.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write detailed JSON results.",
    )
    return parser.parse_args()


def normalize_path(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


def sanitize_name(value: str, fallback: str) -> str:
    value = INVALID_CHARS_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip().strip(".")
    return value or fallback


def sanitize_doc_id(raw_id: object, fallback_index: int) -> str:
    text = str(raw_id) if raw_id is not None else f"{fallback_index:06d}"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or f"{fallback_index:06d}"


def extract_domain(url: str) -> str:
    try:
        hostname = urlparse(url).hostname or "unknown-domain"
        return sanitize_name(hostname.lower(), "unknown-domain")
    except Exception:
        return "unknown-domain"


def extract_title(text: str) -> str | None:
    match = TITLE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return None


def build_filename(title: str | None, url: str, docid: str, prefix: str = "") -> str:
    parsed = urlparse(url or "")
    path_name = Path(parsed.path).name
    fallback = path_name or f"doc-{docid}"
    stem = title or fallback
    stem = sanitize_name(stem, f"doc-{docid}")
    if len(stem) > MAX_STEM_LEN:
        stem = stem[:MAX_STEM_LEN].rstrip(" .")
    if not stem:
        stem = f"doc-{docid}"
    if prefix:
        stem = f"{prefix}{stem}"
    return f"{stem}.txt"


def candidate_relpaths(domain: str, filename: str, docid: str) -> List[str]:
    base = Path(filename)
    stem = base.stem
    suffix = base.suffix
    candidates = [f"{domain}/{filename}"]
    candidates.append(f"{domain}/{stem}__docid_{docid}{suffix}")
    for i in range(2, 200):
        candidates.append(f"{domain}/{stem}__docid_{docid}_{i}{suffix}")
    return [normalize_path(x) for x in candidates]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_index(index_dir: Path) -> tuple[faiss.Index, List[str], Dict[str, Any]]:
    index = faiss.read_index(str(index_dir / "index.faiss"))
    paths = json.loads((index_dir / "paths.json").read_text(encoding="utf-8"))
    paths = [normalize_path(p) for p in paths]
    meta_path = index_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if index.ntotal != len(paths):
        raise ValueError(f"FAISS/path mismatch: ntotal={index.ntotal}, paths={len(paths)}")
    return index, paths, meta


def build_basename_lookup(index_paths: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for rel in index_paths:
        base = Path(rel).name
        out.setdefault(base, []).append(rel)
    return out


def build_docid_to_path(parquet_path: Path, index_paths: Sequence[str]) -> tuple[Dict[str, str], List[str]]:
    path_set = set(index_paths)
    mapping: Dict[str, str] = {}
    misses: List[str] = []
    pf = pq.ParquetFile(parquet_path)

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["docid", "text", "url"])
        for row_idx, row in enumerate(table.to_pylist()):
            # row_idx is only used if docid is missing; parquet has docid for BC+.
            docid = sanitize_doc_id(row.get("docid"), row_idx)
            text = row.get("text") or ""
            url = row.get("url") or ""
            domain = extract_domain(url)
            filename = build_filename(extract_title(text), url, docid)
            found = None
            for rel in candidate_relpaths(domain, filename, docid):
                if rel in path_set:
                    found = rel
                    break
            if found is None:
                misses.append(docid)
            else:
                mapping[docid] = found
    return mapping, misses


def build_fineweb_id_to_path(fineweb_jsonl: Path, index_paths: Sequence[str]) -> tuple[Dict[str, str], List[str]]:
    path_set = set(index_paths)
    mapping: Dict[str, str] = {}
    misses: List[str] = []
    if not fineweb_jsonl.exists():
        return mapping, misses

    with fineweb_jsonl.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_id = row.get("id")
            docid = sanitize_doc_id(raw_id, idx)
            text = row.get("text") or ""
            url = row.get("url") or ""
            domain = extract_domain(url)
            filename = build_filename(extract_title(text), url, docid)
            found = None
            for rel in candidate_relpaths(domain, filename, docid):
                if rel in path_set:
                    found = rel
                    break
            if found is None:
                misses.append(docid)
            else:
                mapping[docid] = found
                mapping[f"fw_{idx}"] = found
    return mapping, misses


def resolve_device(arg_device: str) -> str:
    if arg_device:
        return arg_device
    env_device = os.environ.get("EMBED_DEVICE", "").strip()
    if env_device:
        return env_device
    return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def resolve_dtype(name: str):
    import torch

    value = name.strip().lower()
    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(f"Unsupported dtype: {name}")
    if normalized == "float16":
        return normalized, torch.float16
    if normalized == "bfloat16":
        return normalized, torch.bfloat16
    return normalized, torch.float32


def resolve_max_model_len(index_meta: Dict[str, Any], arg_value: int) -> int:
    if arg_value and arg_value > 0:
        return arg_value
    meta_value = int(index_meta.get("max_model_len", 0) or 0)
    return meta_value if meta_value > 0 else 4096


@dataclass
class Encoder:
    tokenizer: Any
    model: Any
    device: Any
    spec: Any
    max_model_len: int
    query_instruction: str

    @classmethod
    def load(
        cls,
        *,
        model_path: str,
        model_type: str,
        dtype_name: str,
        device_name: str,
        max_model_len: int,
        query_instruction: str,
    ) -> "Encoder":
        import torch
        from transformers import AutoModel, AutoTokenizer

        normalized_dtype, torch_dtype = resolve_dtype(dtype_name)
        device = torch.device(device_name)
        if device.type != "cuda":
            torch_dtype = torch.float32
            normalized_dtype = "float32"
        spec = get_embedding_spec(
            model_path=model_path,
            explicit_model_type=normalize_model_type(model_type),
            index_dir=None,
        )
        print(
            f"Loading encoder: model_type={spec.model_type}, dtype={normalized_dtype}, "
            f"device={device}, pooling={spec.pooling}, padding={spec.tokenizer_padding_side}",
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
            torch_dtype=torch_dtype,
        )
        model = model.to(device).eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            device=device,
            spec=spec,
            max_model_len=max_model_len,
            query_instruction=query_instruction,
        )

    def encode(self, queries: List[str]) -> np.ndarray:
        import torch

        formatted = [format_query_text(q, self.spec, self.query_instruction) for q in queries]
        inputs = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=self.max_model_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.spec.pooling)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        arr = torch_embeddings_to_numpy(embeddings)
        del inputs, outputs, embeddings
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return arr


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def recall_for_ranking(ranked_paths: Sequence[str], gold_paths: set[str], k: int) -> float:
    if not gold_paths:
        return 0.0
    hit = len(set(ranked_paths[:k]) & gold_paths)
    return hit / len(gold_paths)


def evaluate_dtype(
    *,
    dtype_name: str,
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    index: faiss.Index,
    index_paths: List[str],
    index_meta: Dict[str, Any],
    docid_to_path: Dict[str, str],
    basename_lookup: Dict[str, List[str]],
    ks: List[int],
) -> Dict[str, Any]:
    import torch

    device_name = resolve_device(args.device)
    max_model_len = resolve_max_model_len(index_meta, args.max_model_len)
    encoder = Encoder.load(
        model_path=args.model_path,
        model_type=args.model_type,
        dtype_name=dtype_name,
        device_name=device_name,
        max_model_len=max_model_len,
        query_instruction=args.query_instruction,
    )

    max_k = min(max(ks), index.ntotal)
    sums = {k: 0.0 for k in ks}
    per_query: List[Dict[str, Any]] = []
    evaluated = 0
    skipped = 0
    start_time = time.perf_counter()

    for batch_rows in batched(rows, args.batch_size):
        queries = [str(row.get("query", row.get("question", ""))) for row in batch_rows]
        qvecs = encoder.encode(queries)
        _scores, indices = index.search(qvecs.astype(np.float32, copy=False), max_k)

        for row, idxs in zip(batch_rows, indices):
            gold_docids = [str(x) for x in row.get("gold_doc_ids", [])]
            gold_paths = set()
            missing_gold_docids = []
            for gid in gold_docids:
                if gid in docid_to_path:
                    gold_paths.add(docid_to_path[gid])
                    continue
                # Fallback: if gid is already a filename-like key, match by basename.
                if gid in basename_lookup and len(basename_lookup[gid]) == 1:
                    gold_paths.add(basename_lookup[gid][0])
                    continue
                missing_gold_docids.append(gid)
            if not gold_paths:
                skipped += 1
                per_query.append(
                    {
                        "query_id": row.get("query_id"),
                        "skipped": True,
                        "gold_doc_ids": gold_docids,
                        "missing_gold_docids": missing_gold_docids,
                    }
                )
                continue

            ranked_paths = [index_paths[int(i)] for i in idxs if int(i) >= 0]
            recalls = {f"recall@{k}": recall_for_ranking(ranked_paths, gold_paths, min(k, len(ranked_paths))) for k in ks}
            for k in ks:
                sums[k] += recalls[f"recall@{k}"]
            evaluated += 1
            per_query.append(
                {
                    "query_id": row.get("query_id"),
                    "gold_doc_ids": gold_docids,
                    "gold_paths": sorted(gold_paths),
                    "missing_gold_docids": missing_gold_docids,
                    **recalls,
                    "top10": ranked_paths[:10],
                }
            )

    elapsed = time.perf_counter() - start_time
    metrics = {f"recall@{k}": (sums[k] / evaluated if evaluated else 0.0) for k in ks}
    print(f"\n=== dtype={dtype_name} ===")
    print(f"evaluated={evaluated}, skipped={skipped}, elapsed={elapsed:.1f}s")
    for k in ks:
        print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dtype": dtype_name,
        "evaluated": evaluated,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
        "per_query": per_query,
    }


def main() -> int:
    args = parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    dtypes = [x.strip() for x in args.dtypes.split(",") if x.strip()]

    print(f"dataset:   {args.dataset}")
    print(f"parquet:   {args.parquet}")
    print(f"fineweb:   {args.fineweb_sample}")
    print(f"index_dir: {args.index_dir}")
    print(f"model:     {args.model_path}")
    print(f"dtypes:    {dtypes}")
    print(f"ks:        {ks}")

    rows = read_jsonl(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]
    rows_with_gold = [r for r in rows if r.get("gold_doc_ids")]
    print(f"queries:   {len(rows)} ({len(rows_with_gold)} with gold_doc_ids)")
    if not rows_with_gold:
        raise ValueError(
            "No rows with gold_doc_ids were found in the dataset. "
            "Use a labeled dataset such as data/bcplus_qa_sample100.jsonl."
        )
    rows = rows_with_gold

    index, index_paths, index_meta = load_index(args.index_dir)
    basename_lookup = build_basename_lookup(index_paths)
    print(f"index:     ntotal={index.ntotal}, dim={index.d}, meta_max_len={index_meta.get('max_model_len')}")

    print("Building docid -> relative path map from parquet...", flush=True)
    docid_to_path, map_misses = build_docid_to_path(args.parquet, index_paths)
    print(f"bc doc map:{len(docid_to_path)} mapped, {len(map_misses)} misses")
    if map_misses[:5]:
        print(f"map miss examples: {map_misses[:5]}")

    fineweb_map, fineweb_misses = build_fineweb_id_to_path(args.fineweb_sample, index_paths)
    if fineweb_map:
        docid_to_path.update(fineweb_map)
        print(f"fw doc map: {len(fineweb_map)} mapped, {len(fineweb_misses)} misses")
        if fineweb_misses[:5]:
            print(f"fw miss examples: {fineweb_misses[:5]}")

    needed_gold = sorted({str(x) for row in rows for x in row.get("gold_doc_ids", [])})
    missing_needed = [x for x in needed_gold if x not in docid_to_path]
    print(f"gold ids:  {len(needed_gold)} unique, {len(missing_needed)} missing from map")
    if missing_needed[:10]:
        print(f"missing gold examples: {missing_needed[:10]}")

    results = []
    for dtype in dtypes:
        results.append(
            evaluate_dtype(
                dtype_name=dtype,
                args=args,
                rows=rows,
                index=index,
                index_paths=index_paths,
                index_meta=index_meta,
                docid_to_path=docid_to_path,
                basename_lookup=basename_lookup,
                ks=ks,
            )
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": str(args.dataset),
            "parquet": str(args.parquet),
            "index_dir": str(args.index_dir),
            "model_path": args.model_path,
            "ks": ks,
            "results": results,
        }
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
