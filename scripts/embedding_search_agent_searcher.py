#!/usr/bin/env python3
"""
Embedding searcher for the BrowseComp-Plus official search-agent style baseline.

This version is designed for DCI-Agent-Lite and reuses our existing:
  - FAISS index: index.faiss
  - document path map: paths.json
  - embedding model families / formatting rules from embedding_backends.py

Compared with the older embedding_agent_searcher.py, this module:
  - supports our shared embedding model configuration
  - can infer model type from index meta.json
  - exposes snippet truncation and cache-empty behavior explicitly
  - is intended to be paired with the new ts-mirror-style runner
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, logging as tf_logging

from embedding_backends import (
    DEFAULT_QUERY_INSTRUCTION,
    DEFAULT_QWEN3_EMBED_MODEL,
    create_vllm_embedder,
    encode_batch_with_vllm,
    format_query_text,
    get_embedding_spec,
    normalize_backend_override,
    normalize_model_type,
    pool_hidden_states,
    preferred_torch_dtype,
    probe_vllm_embedder,
    torch_embeddings_to_numpy,
)

tf_logging.set_verbosity_error()


class EmbeddingSearchAgentSearcher:
    """FAISS-backed embedding searcher aligned with our embedding stack."""

    def __init__(
        self,
        *,
        index_dir: str,
        corpus_dir: str,
        model_path: str = DEFAULT_QWEN3_EMBED_MODEL,
        model_type: str = "auto",
        backend: str = "auto",
        device: Optional[str] = None,
        max_model_len: int = 512,
        snippet_max_tokens: int = 512,
        encode_batch_size: int = 16,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        empty_cache_after_encode: bool = False,
    ):
        self.index_dir = str(index_dir)
        self.corpus_dir = str(corpus_dir)
        self.model_path = str(model_path or DEFAULT_QWEN3_EMBED_MODEL)
        self.max_model_len = int(max_model_len)
        self.snippet_max_tokens = int(snippet_max_tokens)
        self.encode_batch_size = max(1, int(encode_batch_size))
        self.query_instruction = str(query_instruction or DEFAULT_QUERY_INSTRUCTION)
        self.empty_cache_after_encode = bool(empty_cache_after_encode)

        device_text = device or os.environ.get("EMBED_DEVICE", "").strip()
        if device_text:
            self.device = torch.device(device_text)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.spec = get_embedding_spec(
            model_path=self.model_path,
            explicit_model_type=normalize_model_type(model_type),
            index_dir=self.index_dir,
        )
        backend_override = normalize_backend_override(backend)
        self.backend = self.spec.backend if backend_override == "auto" else backend_override
        self.tensor_parallel_size = max(
            1,
            len((os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(","))
            if os.environ.get("CUDA_VISIBLE_DEVICES")
            else 1,
        )

        self.index: faiss.Index = self._load_index()
        self.paths: List[str] = self._load_paths()
        self.tokenizer = None
        self.model = None
        self.llm = None
        self._load_embedder()
        self._load_snippet_tokenizer()

    def _log(self, msg: str) -> None:
        print(f"[embedding_search_agent_searcher] {msg}", file=sys.stderr, flush=True)

    def _load_index(self) -> faiss.Index:
        index_path = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self._log(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        self._log(f"  loaded {index.ntotal} documents")
        return index

    def _load_paths(self) -> List[str]:
        paths_path = os.path.join(self.index_dir, "paths.json")
        if not os.path.exists(paths_path):
            raise FileNotFoundError(f"paths.json not found: {paths_path}")
        with open(paths_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_embedder(self) -> None:
        self._log(
            f"Loading embedding model: model={self.model_path} "
            f"model_type={self.spec.model_type} backend={self.backend} device={self.device}"
        )
        if self.backend == "vllm":
            try:
                self.llm = create_vllm_embedder(
                    model_path=self.model_path,
                    max_model_len=self.max_model_len,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=0.9,
                )
                probe_dim = probe_vllm_embedder(
                    self.llm,
                    probe_text=format_query_text("probe", self.spec, self.query_instruction),
                    max_model_len=self.max_model_len,
                )
                self._log(f"  vLLM embedding backend ready, dim={probe_dim}")
                return
            except Exception as e:
                self._log(f"  [WARN] vLLM failed, falling back to transformers: {e}")
                self.backend = "transformers"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side=self.spec.tokenizer_padding_side,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=preferred_torch_dtype(self.spec, self.device),
        )
        self.model = self.model.to(self.device).eval()
        self._log("  transformers embedding model loaded")

    def _load_snippet_tokenizer(self) -> None:
        if self.tokenizer is not None:
            self.snippet_tokenizer = self.tokenizer
            return
        self.snippet_tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side=self.spec.tokenizer_padding_side,
            trust_remote_code=True,
        )

    def _read_document(self, rel_path: str) -> str:
        full_path = os.path.join(self.corpus_dir, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except (FileNotFoundError, IOError, OSError):
            return ""

    def _truncate_snippet(self, text: str) -> str:
        if not text:
            return ""
        if self.snippet_max_tokens <= 0:
            return text
        tokens = self.snippet_tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.snippet_max_tokens:
            return text
        return self.snippet_tokenizer.decode(
            tokens[: self.snippet_max_tokens],
            skip_special_tokens=True,
        )

    @torch.no_grad()
    def _encode_texts_transformers(self, texts: List[str]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        assert self.tokenizer is not None and self.model is not None
        for start in range(0, len(texts), self.encode_batch_size):
            batch = texts[start : start + self.encode_batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_model_len,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            embeddings = pool_hidden_states(
                outputs.last_hidden_state,
                inputs["attention_mask"],
                self.spec.pooling,
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(torch_embeddings_to_numpy(embeddings))
            del inputs, outputs, embeddings
            if self.empty_cache_after_encode and self.device.type == "cuda":
                torch.cuda.empty_cache()
        return np.concatenate(all_embeddings, axis=0)

    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        formatted = [
            format_query_text(query, self.spec, self.query_instruction)
            for query in queries
        ]
        if not formatted:
            return np.zeros((0, int(self.index.d)), dtype=np.float32)
        if self.backend == "vllm":
            assert self.llm is not None
            return encode_batch_with_vllm(self.llm, formatted, self.max_model_len)
        return self._encode_texts_transformers(formatted)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.encode_queries([query])
        effective_k = min(max(1, int(k)), self.index.ntotal)
        scores, indices = self.index.search(query_vec, effective_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rel_path = self.paths[idx]
            text = self._read_document(rel_path)
            results.append(
                {
                    "docid": rel_path,
                    "score": float(score),
                    "snippet": self._truncate_snippet(text),
                }
            )
        return results

    def search_description(self, k: int = 5) -> str:
        return (
            f"Perform a semantic search on the document corpus. "
            f"Returns top-{k} hits with docid, score, and snippet. "
            f"The snippet contains the document's contents (truncated to {self.snippet_max_tokens} tokens)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone test for embedding search agent searcher")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--index-dir", type=str, required=True, help="Index directory")
    parser.add_argument("--corpus-dir", type=str, required=True, help="Corpus directory")
    parser.add_argument("--model-path", type=str, default=DEFAULT_QWEN3_EMBED_MODEL, help="Embedding model path")
    parser.add_argument("--model-type", type=str, default="auto", help="Embedding model type")
    parser.add_argument("--backend", type=str, default="auto", help="Embedding backend")
    parser.add_argument("--device", type=str, default="", help="Torch device")
    parser.add_argument("--max-model-len", type=int, default=512, help="Query max length")
    parser.add_argument("--snippet-max-tokens", type=int, default=512, help="Snippet max token length")
    parser.add_argument("--encode-batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--query-instruction", type=str, default=DEFAULT_QUERY_INSTRUCTION, help="Query instruction")
    parser.add_argument("--empty-cache-after-encode", action="store_true", help="Call torch.cuda.empty_cache after each batch")
    parser.add_argument("--k", type=int, default=5, help="Top-k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    searcher = EmbeddingSearchAgentSearcher(
        index_dir=args.index_dir,
        corpus_dir=args.corpus_dir,
        model_path=args.model_path,
        model_type=args.model_type,
        backend=args.backend,
        device=args.device or None,
        max_model_len=args.max_model_len,
        snippet_max_tokens=args.snippet_max_tokens,
        encode_batch_size=args.encode_batch_size,
        query_instruction=args.query_instruction,
        empty_cache_after_encode=args.empty_cache_after_encode,
    )
    start = time.perf_counter()
    results = searcher.search(args.query, k=args.k)
    elapsed = time.perf_counter() - start
    print(f"Search completed in {elapsed:.2f}s")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
