#!/usr/bin/env python3
"""
Persistent model server for DCI-Agent-Lite.

Loads:
  - embedding model
  - FAISS index
  - optional QRReranker

Then serves requests via stdin/stdout JSON-line protocol or TCP mode.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, logging as tf_logging

from embedding_backends import (
    DEFAULT_QUERY_INSTRUCTION,
    DEFAULT_QWEN3_EMBED_MODEL,
    create_vllm_embedder,
    encode_batch_with_vllm,
    format_passage_text,
    format_query_text,
    get_embedding_spec,
    normalize_backend_override,
    normalize_model_type,
    pool_hidden_states,
    preferred_torch_dtype,
    probe_vllm_embedder,
    read_index_meta,
    torch_embeddings_to_numpy,
)

tf_logging.set_verbosity_error()


QUERY_INSTRUCTION = DEFAULT_QUERY_INSTRUCTION
DEFAULT_EMBED_MODEL = DEFAULT_QWEN3_EMBED_MODEL
DEFAULT_QR_MODEL = str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-4B-Instruct-2507")
DEFAULT_QR_HEADS = "20-15,15-11,23-10,22-4,21-8,15-9,17-4,21-11,21-10,15-13,21-18,18-15,24-23,18-19,21-31,17-25"


def log(msg: str):
    print(f"[model_server] {msg}", file=sys.stderr, flush=True)


def _scope_path_dot_prefix_enabled() -> bool:
    return os.environ.get("SCOPE_PATH_DOT_PREFIX", "").strip().lower() in {"1", "true", "yes", "on"}


def send_response(response: dict):
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def send_result(request_id: int, result: Any):
    send_response({"id": request_id, "result": result})


def send_error(request_id: int, error: str):
    send_response({"id": request_id, "error": error})


def ensure_qr_import_path() -> None:
    script_dir = Path(__file__).resolve().parent
    qr_dir = script_dir.parent / "qr_head_discovery"
    if str(qr_dir) not in sys.path:
        sys.path.insert(0, str(qr_dir))


class EmbeddingEngine:
    """Manages embedding model + FAISS index for semantic recall and emb reranking."""

    def __init__(
        self,
        model_path: str,
        index_dir: str,
        device: torch.device,
        model_type: str = "auto",
        backend: str = "auto",
        max_tokens: int = 512,
        rerank_max_tokens: int = 4096,
        rerank_batch_size: int = 32,
        query_instruction: str = "",
        load_index: bool = True,
    ):
        self.device = device
        self.max_tokens = max_tokens
        self.rerank_max_tokens = rerank_max_tokens
        self.rerank_batch_size = max(1, rerank_batch_size)
        self.query_instruction = query_instruction or QUERY_INSTRUCTION
        self.load_index = load_index
        self.empty_cache_after_encode = os.environ.get("EMBED_EMPTY_CACHE_AFTER_ENCODE", "0").strip().lower() in {"1", "true", "yes"}
        self.tensor_parallel_size = max(1, len((os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1)
        self.spec = get_embedding_spec(
            model_path=model_path,
            explicit_model_type=normalize_model_type(model_type),
            index_dir=index_dir,
        )
        requested_backend = normalize_backend_override(backend)
        self.backend = self.spec.backend if requested_backend == "auto" else requested_backend
        self.index = None
        self.paths: List[str] = []

        if self.load_index:
            log(f"Loading FAISS index from {index_dir}...")
            index_path = os.path.join(index_dir, "index.faiss")
            paths_path = os.path.join(index_dir, "paths.json")
            self.index = faiss.read_index(index_path)
            with open(paths_path, "r", encoding="utf-8") as f:
                self.paths = json.load(f)
            log(f"  FAISS index loaded: {self.index.ntotal} documents")
        else:
            log(f"Skipping FAISS index load for rerank-only embedding engine ({self.spec.model_type})")

        log(f"Loading embedding model from {model_path}...")
        log(
            f"  model_type={self.spec.model_type}, backend={self.backend}, pooling={self.spec.pooling}, "
            f"device={device}, query_style={self.spec.query_style}, "
            f"max_tokens={self.max_tokens}, rerank_max_tokens={self.rerank_max_tokens}"
        )
        self.tokenizer = None
        self.model = None
        self.llm = None

        if self.backend == "vllm":
            try:
                self.llm = create_vllm_embedder(
                    model_path=model_path,
                    max_model_len=max(self.max_tokens, self.rerank_max_tokens),
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=0.9,
                )
                probe_dim = probe_vllm_embedder(
                    self.llm,
                    probe_text=format_passage_text("vllm backend probe", self.spec),
                    max_model_len=max(self.max_tokens, self.rerank_max_tokens),
                )
                log(f"  vLLM embedding backend ready, dim={probe_dim}")
            except Exception as e:
                log(f"  [WARN] vLLM backend failed, falling back to Transformers: {e}")
                self.backend = "transformers"

        if self.backend == "transformers":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side=self.spec.tokenizer_padding_side,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=preferred_torch_dtype(self.spec, device),
            )
            self.model = self.model.to(device).eval()
            log(f"  Transformers embedding model loaded on {device}")

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], *, max_length: int) -> np.ndarray:
        if not texts:
            dim = int(self.index.d) if self.index is not None else 0
            return np.zeros((0, dim), dtype=np.float32)

        if self.backend == "vllm":
            assert self.llm is not None
            return encode_batch_with_vllm(self.llm, texts, max_length)

        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.rerank_batch_size):
            batch_texts = texts[start : start + self.rerank_batch_size]
            assert self.tokenizer is not None and self.model is not None
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            embeddings = pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.spec.pooling)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(torch_embeddings_to_numpy(embeddings))
            del inputs, outputs, embeddings
            if self.empty_cache_after_encode and self.device.type == "cuda":
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    @torch.no_grad()
    def encode_query(self, query: str) -> np.ndarray:
        formatted = format_query_text(query, self.spec, self.query_instruction)
        return self._encode_texts([formatted], max_length=self.max_tokens)

    def search(self, query: str, top_k: int) -> List[str]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded for this embedding engine")
        query_vec = self.encode_query(query)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, top_k)
        return [self.paths[i] for i in indices[0] if i >= 0]

    @torch.no_grad()
    def encode_passages(self, texts: List[str]) -> np.ndarray:
        formatted = [format_passage_text(text, self.spec) for text in texts]
        return self._encode_texts(formatted, max_length=self.rerank_max_tokens)

    def rank_texts_by_similarity(self, query: str, texts: List[str]) -> List[int]:
        if not texts:
            return []
        query_vec = self.encode_query(query)
        doc_vecs = self.encode_passages(texts)
        scores = np.matmul(doc_vecs, query_vec[0])
        return [int(i) for i in np.argsort(-scores)]


class RerankerEngine:
    """Manages QRReranker for paragraph-level reranking."""

    def __init__(self, model_path: str, qr_heads: str, device: str = "cuda:0"):
        log(f"Loading QRReranker from {model_path}...")
        log(f"  QR heads: {qr_heads}")

        ensure_qr_import_path()

        from qr_rerank import QRReranker

        self.reranker = QRReranker(
            model_path=model_path,
            qr_head_list=qr_heads,
            calibration=True,
            device=device,
        )
        log(f"  QRReranker loaded on {device}")

    def rerank_paragraphs(
        self,
        query: str,
        documents: List[tuple],
        topk: int = 10,
        max_total_tokens: int = 40000,
        min_chars: int = 400,
        max_chars: int = 1200,
        target_chars: int = 600,
    ) -> List[Dict[str, Any]]:
        from qr_rerank import rerank_paragraphs

        return rerank_paragraphs(
            question=query,
            documents=documents,
            reranker=self.reranker,
            topk_paragraphs=topk,
            max_total_tokens=max_total_tokens,
            min_chars=min_chars,
            max_chars=max_chars,
            target_chars=target_chars,
        )


def find_next_scope_number(scope_dir: Path) -> int:
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


def write_scope_file(scope_file: Path, paths: List[str]) -> None:
    with open(scope_file, "w", encoding="utf-8") as f:
        for p in paths:
            path_text = p
            if _scope_path_dot_prefix_enabled() and path_text and not path_text.startswith("/") and not path_text.startswith("./"):
                path_text = f"./{path_text}"
            f.write(path_text + "\n")


def write_scope_meta(meta_file: Path, query: str, top_k: int, doc_count: int, elapsed_ms: float) -> None:
    meta = {
        "query": query,
        "top_k": top_k,
        "doc_count": doc_count,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _strip_scope_display_prefix(path_text: str) -> str:
    text = str(path_text or "")
    if _scope_path_dot_prefix_enabled() and text.startswith("./"):
        return text[2:]
    return text


def resolve_scope_dir(scope_dir_str: str, scope_size_prefix: str = "", sample_id: str = "") -> tuple:
    original_scope_dir = Path(scope_dir_str).resolve()
    original_scope_dir.mkdir(parents=True, exist_ok=True)

    size_prefix = scope_size_prefix or os.environ.get("SCOPE_SIZE_PREFIX", "").strip()
    sid = sample_id or os.environ.get("SAMPLE_ID", "").strip()

    if size_prefix and sid:
        tmp_scope_dir = Path(f"/tmp/{size_prefix}/{sid}")
        tmp_scope_dir.mkdir(parents=True, exist_ok=True)
        return tmp_scope_dir, original_scope_dir

    return original_scope_dir, None


def _format_qr_paragraphs(question: str, documents: List[tuple], reranker_engine: RerankerEngine) -> str:
    ensure_qr_import_path()
    from qr_rerank import rerank_and_format, rerank_top50_and_format

    if len(documents) <= 20:
        return rerank_and_format(
            question=question,
            documents=documents,
            reranker=reranker_engine.reranker,
            top_tier1=min(10, len(documents)),
            top_tier2=min(20, len(documents)),
            topk_tier1=7,
            topk_tier2=3,
        )

    return rerank_top50_and_format(
        question=question,
        documents=documents,
        reranker=reranker_engine.reranker,
    )


def _format_emb_paragraphs(question: str, documents: List[tuple], embedding_engine: EmbeddingEngine) -> str:
    ensure_qr_import_path()
    from paragraph_splitter import truncate_documents, split_documents_to_paragraphs

    truncated_docs = truncate_documents(documents, max_total_tokens=40000)
    para_info = split_documents_to_paragraphs(
        truncated_docs,
        min_chars=400,
        max_chars=1200,
        target_chars=600,
    )
    if not para_info:
        return ""

    paragraph_results: List[Dict[str, Any]] = []
    candidate_texts: List[str] = []
    for para_idx, (para_text, doc_idx, start_line, end_line) in enumerate(para_info):
        text = para_text.rstrip("\n")
        paragraph_results.append(
            {
                "text": text,
                "doc_title": _strip_scope_display_prefix(truncated_docs[doc_idx][1]),
                "doc_index": doc_idx,
                "start_line": start_line,
                "end_line": end_line,
                "para_index": para_idx,
            }
        )
        candidate_texts.append(text)

    ranked_indices = embedding_engine.rank_texts_by_similarity(question, candidate_texts)
    selected = [paragraph_results[idx] for idx in ranked_indices[:10]]

    from collections import OrderedDict

    doc_groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for item in selected:
        title = item["doc_title"]
        doc_groups.setdefault(title, []).append(item)

    for title in doc_groups:
        doc_groups[title].sort(key=lambda r: r["start_line"])

    parts: List[str] = []
    for title, paragraphs in doc_groups.items():
        doc_lines = [f"{title}:"]
        for p in paragraphs:
            start = p["start_line"]
            end = p["end_line"]
            line_label = f"Line {start}-{end}" if end > start else f"Line {start}"
            doc_lines.append("")
            doc_lines.append(f"{line_label}: {p['text']}")
        parts.append("\n".join(doc_lines))

    return "\n\n---\n\n".join(parts)


def handle_embed_recall(
    embedding_engine: EmbeddingEngine,
    reranker_engine: Optional[RerankerEngine],
    params: Dict[str, Any],
    corpus_dir: str,
) -> Dict[str, Any]:
    query = params["query"]
    top_k = params.get("top_k", 5000)
    scope_dir_str = params.get("scope_dir", ".")
    scope_size_prefix = params.get("scope_size_prefix", "")
    sample_id = params.get("sample_id", "")
    paragraph_rerank_model = str(params.get("paragraph_rerank_model", "qr") or "qr").strip().lower()
    if paragraph_rerank_model not in {"qr", "emb", "none"}:
        paragraph_rerank_model = "qr"
    paragraph_rerank_doc_limit = int(params.get("paragraph_rerank_doc_limit", 0) or 0)

    start_time = time.perf_counter()
    scope_dir, backup_dir = resolve_scope_dir(scope_dir_str, scope_size_prefix, sample_id)
    recalled_paths = embedding_engine.search(query, top_k)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    scope_num = find_next_scope_number(scope_dir)
    scope_file = scope_dir / f"scope_{scope_num}.txt"
    meta_file = scope_dir / f"scope_{scope_num}.meta"

    write_scope_file(scope_file, recalled_paths)
    write_scope_meta(meta_file, query, top_k, len(recalled_paths), elapsed_ms)

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scope_file, backup_dir / scope_file.name)
        shutil.copy2(meta_file, backup_dir / meta_file.name)

    scope_output = f'"{query}" → {scope_file} ({len(recalled_paths)} docs)'
    log(f"  embed_recall: {scope_output} [{elapsed_ms:.0f}ms]")

    qr_output = ""
    if recalled_paths and paragraph_rerank_model != "none" and (paragraph_rerank_model == "emb" or reranker_engine is not None):
        try:
            rerank_start = time.perf_counter()
            default_doc_limit = 50 if paragraph_rerank_model == "qr" else 20
            doc_limit = paragraph_rerank_doc_limit if paragraph_rerank_doc_limit > 0 else default_doc_limit
            candidate_paths = recalled_paths[:doc_limit]
            documents = []
            for rel_path in candidate_paths:
                full_path = os.path.join(corpus_dir, rel_path)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    documents.append((text, rel_path))
                except (FileNotFoundError, IOError):
                    continue

            if documents:
                if paragraph_rerank_model == "emb":
                    formatted = _format_emb_paragraphs(query, documents, embedding_engine)
                else:
                    display_documents = [(doc_text, _strip_scope_display_prefix(doc_path)) for doc_text, doc_path in documents]
                    formatted = _format_qr_paragraphs(query, display_documents, reranker_engine)
                if formatted:
                    qr_output = (
                        "<qr_paragraphs>\n"
                        "Below are some relevant paragraphs from the top recalled documents (for reference only):\n\n"
                        f"{formatted}\n"
                        "</qr_paragraphs>"
                    )

            rerank_ms = (time.perf_counter() - rerank_start) * 1000
            log(f"  paragraph_rerank({paragraph_rerank_model}): {len(documents)} docs processed [{rerank_ms:.0f}ms]")
        except Exception as e:
            log(f"  [WARN] paragraph rerank ({paragraph_rerank_model}) failed, skipping: {e}")

    output_msg = scope_output
    if qr_output:
        output_msg += "\n" + qr_output

    return {
        "output": output_msg,
        "scope_file": str(scope_file),
        "count": len(recalled_paths),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def handle_rerank_paragraphs(reranker_engine: RerankerEngine, params: Dict[str, Any]) -> Dict[str, Any]:
    query = params["query"]
    documents = params["documents"]
    topk = params.get("topk", 10)
    max_total_tokens = params.get("max_total_tokens", 40000)
    min_chars = params.get("min_chars", 400)
    max_chars = params.get("max_chars", 1200)
    target_chars = params.get("target_chars", 600)

    start_time = time.perf_counter()
    results = reranker_engine.rerank_paragraphs(
        query=query,
        documents=documents,
        topk=topk,
        max_total_tokens=max_total_tokens,
        min_chars=min_chars,
        max_chars=max_chars,
        target_chars=target_chars,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    log(f"  rerank_paragraphs: {len(results)} paragraphs returned [{elapsed_ms:.0f}ms]")
    return {
        "paragraphs": results,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def handle_rerank_matches(
    embedding_engine: EmbeddingEngine,
    rg_embedding_engine: Optional[EmbeddingEngine],
    reranker_engine: Optional[RerankerEngine],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    query = params["query"]
    matches = params.get("matches", [])
    rerank_model = str(params.get("rerank_model", "qr") or "qr").strip().lower()
    if rerank_model not in {"qr", "emb", "qwen3emb"}:
        rerank_model = "qr"
    if not matches:
        return {"ranked_indices": [], "elapsed_ms": 0.0}

    start_time = time.perf_counter()
    texts = [str(item.get("text", "")) for item in matches]

    if rerank_model in {"emb", "qwen3emb"}:
        active_embedding_engine = rg_embedding_engine or embedding_engine
        ranked_indices = active_embedding_engine.rank_texts_by_similarity(query, texts)
    else:
        if reranker_engine is None:
            raise ValueError("QRReranker not loaded (started with --no-reranker)")
        ranked = reranker_engine.reranker.rerank(query, texts, topk=len(texts))
        ranked_indices = [int(idx) for idx, _score in ranked]

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    log(f"  rerank_matches({rerank_model}): {len(matches)} matches [{elapsed_ms:.0f}ms]")
    return {
        "ranked_indices": ranked_indices,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def dispatch_request(method: str, params: dict, embedding_engine, rg_embedding_engine, reranker_engine, corpus_dir, inference_lock):
    if method == "embed_recall":
        with inference_lock:
            result = handle_embed_recall(embedding_engine, reranker_engine, params, corpus_dir)
        return result, False
    elif method == "rerank_paragraphs":
        if reranker_engine is None:
            raise ValueError("QRReranker not loaded (started with --no-reranker)")
        with inference_lock:
            result = handle_rerank_paragraphs(reranker_engine, params)
        return result, False
    elif method == "rerank_matches":
        rerank_model = str(params.get("rerank_model", "qr") or "qr").strip().lower()
        if rerank_model == "qr" and reranker_engine is None:
            raise ValueError("QRReranker not loaded (started with --no-reranker)")
        with inference_lock:
            result = handle_rerank_matches(embedding_engine, rg_embedding_engine, reranker_engine, params)
        return result, False
    elif method == "shutdown":
        log("Shutdown requested.")
        return "ok", True
    elif method == "ping":
        return "pong", False
    else:
        raise ValueError(f"Unknown method: {method}")


def run_stdio_loop(embedding_engine, rg_embedding_engine, reranker_engine, corpus_dir, inference_lock):
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = request.get("id", 0)
            method = request.get("method", "")
            params = request.get("params", {})
            result, should_shutdown = dispatch_request(
                method, params, embedding_engine, rg_embedding_engine, reranker_engine, corpus_dir, inference_lock
            )
            send_result(request_id, result)
            if should_shutdown:
                break
        except Exception as e:
            request_id = 0
            try:
                request_id = request.get("id", 0)  # type: ignore[name-defined]
            except Exception:
                pass
            log(f"[ERROR] {e}\n{traceback.format_exc()}")
            send_error(request_id, str(e))


def _handle_tcp_client(conn, addr, embedding_engine, rg_embedding_engine, reranker_engine, corpus_dir, inference_lock):
    log(f"TCP client connected: {addr}")
    try:
        rfile = conn.makefile("r", encoding="utf-8")
        wfile = conn.makefile("w", encoding="utf-8")
        wfile.write(json.dumps({"id": 0, "result": {"status": "ready"}}) + "\n")
        wfile.flush()
        while True:
            line = rfile.readline()
            if not line:
                break
            try:
                request = json.loads(line)
                request_id = request.get("id", 0)
                method = request.get("method", "")
                params = request.get("params", {})
                result, should_shutdown = dispatch_request(
                    method, params, embedding_engine, rg_embedding_engine, reranker_engine, corpus_dir, inference_lock
                )
                wfile.write(json.dumps({"id": request_id, "result": result}, ensure_ascii=False) + "\n")
                wfile.flush()
                if should_shutdown:
                    break
            except Exception as e:
                request_id = 0
                try:
                    request_id = request.get("id", 0)  # type: ignore[name-defined]
                except Exception:
                    pass
                log(f"[ERROR] {e}\n{traceback.format_exc()}")
                wfile.write(json.dumps({"id": request_id, "error": str(e)}, ensure_ascii=False) + "\n")
                wfile.flush()
    finally:
        try:
            conn.close()
        except Exception:
            pass
        log(f"TCP client disconnected: {addr}")


def run_tcp_server(port: int, embedding_engine, rg_embedding_engine, reranker_engine, corpus_dir, inference_lock):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(16)
    log(f"TCP server listening on 127.0.0.1:{port}")
    try:
        while True:
            conn, addr = server.accept()
            thread = threading.Thread(
                target=_handle_tcp_client,
                args=(conn, addr, embedding_engine, rg_embedding_engine, reranker_engine, corpus_dir, inference_lock),
                daemon=True,
            )
            thread.start()
    finally:
        server.close()


def parse_server_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent model server for DCI-Agent-Lite")
    parser.add_argument("--index-dir", type=str, default=None, help="FAISS index directory (or EMBED_INDEX_DIR env)")
    parser.add_argument("--embed-model-path", type=str, default=None, help=f"Embedding model path (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--embed-model-type", type=str, default=None, help="Embedding model family override: auto/qwen3_embedding_4b/llama_nv_embed_reasoning_3b")
    parser.add_argument("--embed-backend", type=str, default=None, help="Embedding backend override: auto/transformers/vllm")
    parser.add_argument("--qr-model-path", type=str, default=None, help=f"QRReranker model path (default: {DEFAULT_QR_MODEL})")
    parser.add_argument("--qr-heads", type=str, default=None, help=f"QR head list (default: {DEFAULT_QR_HEADS})")
    parser.add_argument("--device", type=str, default=None, help="CUDA device for embedding model (default: auto)")
    parser.add_argument("--qr-device", type=str, default=None, help="CUDA device for QRReranker (default: same as --device)")
    parser.add_argument("--corpus-dir", type=str, default=None, help="Corpus directory for loading documents (or CORPUS_DIR env)")
    parser.add_argument("--embed-max-tokens", type=int, default=None, help="Max token length for query encoding (default: index meta max_model_len or 512)")
    parser.add_argument("--embed-rerank-max-tokens", type=int, default=None, help="Max token length for emb rerank passage encoding (default: model/index dependent)")
    parser.add_argument("--rg-embed-model-path", type=str, default=None, help="Optional separate embedding model path for rg match reranking")
    parser.add_argument("--rg-embed-model-type", type=str, default=None, help="Optional separate embedding model family for rg match reranking")
    parser.add_argument("--rg-embed-backend", type=str, default=None, help="Optional separate embedding backend for rg match reranking")
    parser.add_argument("--rg-embed-device", type=str, default=None, help="Optional separate device for rg embedding reranker")
    parser.add_argument("--rg-embed-max-tokens", type=int, default=None, help="Optional separate max token length for rg rerank query encoding")
    parser.add_argument("--rg-embed-rerank-max-tokens", type=int, default=None, help="Optional separate max token length for rg rerank match encoding")
    parser.add_argument("--no-reranker", action="store_true", help="Skip loading QRReranker (embed_recall only)")
    parser.add_argument("--port", type=int, default=None, help="TCP port for persistent server mode. If not set, use stdin/stdout.")
    return parser.parse_args()


def main():
    args = parse_server_args()

    index_dir = args.index_dir or os.environ.get("EMBED_INDEX_DIR")
    if not index_dir:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        index_dir = str(repo_root / "data" / "indices" / "bc_plus_100k")

    embed_model_path = args.embed_model_path or os.environ.get("EMBED_MODEL_PATH", DEFAULT_EMBED_MODEL)
    embed_model_type = args.embed_model_type or os.environ.get("EMBED_MODEL_TYPE", "auto")
    embed_backend = args.embed_backend or os.environ.get("EMBED_BACKEND", "auto")
    rg_embed_model_path = args.rg_embed_model_path or os.environ.get("RG_EMBED_MODEL_PATH", "").strip()
    rg_embed_model_type = args.rg_embed_model_type or os.environ.get("RG_EMBED_MODEL_TYPE", "")
    rg_embed_backend = args.rg_embed_backend or os.environ.get("RG_EMBED_BACKEND", "")
    if (os.environ.get("RG_RERANK_MODEL", "").strip().lower() == "qwen3emb") and not rg_embed_model_path:
        rg_embed_model_path = DEFAULT_QWEN3_EMBED_MODEL
    if (os.environ.get("RG_RERANK_MODEL", "").strip().lower() == "qwen3emb") and not (rg_embed_model_type or "").strip():
        rg_embed_model_type = "qwen3_embedding_4b"
    if (os.environ.get("RG_RERANK_MODEL", "").strip().lower() == "qwen3emb") and not (rg_embed_backend or "").strip():
        rg_embed_backend = "transformers"
    qr_model_path = args.qr_model_path or os.environ.get("QR_MODEL_PATH", DEFAULT_QR_MODEL)
    qr_heads = args.qr_heads or os.environ.get("QR_HEADS", DEFAULT_QR_HEADS)
    corpus_dir = args.corpus_dir or os.environ.get("CORPUS_DIR", ".")

    if not embed_model_path:
        raise SystemExit("Missing embedding model path. Set --embed-model-path or EMBED_MODEL_PATH.")

    embed_device_str = args.device or os.environ.get("EMBED_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    qr_device_str = args.qr_device or os.environ.get("QR_DEVICE", embed_device_str)
    rg_embed_device_str = args.rg_embed_device or os.environ.get("RG_EMBED_DEVICE", embed_device_str)
    embed_device = torch.device(embed_device_str)
    rg_embed_device = torch.device(rg_embed_device_str)
    index_meta = read_index_meta(index_dir)
    index_max_model_len = int(index_meta.get("max_model_len", 0) or 0)

    embed_max_tokens = args.embed_max_tokens
    if embed_max_tokens is None:
        env_value = os.environ.get("EMBED_MAX_TOKENS", "").strip()
        if env_value:
            embed_max_tokens = int(env_value)
    if embed_max_tokens is None:
        embed_max_tokens = index_max_model_len if index_max_model_len > 0 else 512

    embed_rerank_max_tokens = args.embed_rerank_max_tokens
    if embed_rerank_max_tokens is None:
        env_value = os.environ.get("EMBED_RERANK_MAX_TOKENS", "").strip()
        if env_value:
            embed_rerank_max_tokens = int(env_value)
    if embed_rerank_max_tokens is None:
        normalized_embed_model_type = normalize_model_type(embed_model_type)
        if normalized_embed_model_type == "llama_nv_embed_reasoning_3b":
            if index_max_model_len > 0:
                embed_rerank_max_tokens = max(512, index_max_model_len)
            else:
                embed_rerank_max_tokens = 8192
        elif index_max_model_len > 0:
            embed_rerank_max_tokens = max(512, min(index_max_model_len, 4096))
        else:
            embed_rerank_max_tokens = 4096

    embed_rerank_batch_size_env = os.environ.get("EMBED_RERANK_BATCH_SIZE", "").strip()
    if embed_rerank_batch_size_env:
        embed_rerank_batch_size = max(1, int(embed_rerank_batch_size_env))
    elif normalize_model_type(embed_model_type) == "llama_nv_embed_reasoning_3b":
        embed_rerank_batch_size = 1
    else:
        embed_rerank_batch_size = 64

    rg_embed_max_tokens = args.rg_embed_max_tokens
    if rg_embed_max_tokens is None:
        env_value = os.environ.get("RG_EMBED_MAX_TOKENS", "").strip()
        if env_value:
            rg_embed_max_tokens = int(env_value)
    if rg_embed_max_tokens is None:
        rg_embed_max_tokens = embed_max_tokens

    rg_embed_rerank_max_tokens = args.rg_embed_rerank_max_tokens
    if rg_embed_rerank_max_tokens is None:
        env_value = os.environ.get("RG_EMBED_RERANK_MAX_TOKENS", "").strip()
        if env_value:
            rg_embed_rerank_max_tokens = int(env_value)
    if rg_embed_rerank_max_tokens is None:
        rg_embed_rerank_max_tokens = embed_rerank_max_tokens

    rg_embed_rerank_batch_size_env = os.environ.get("RG_EMBED_RERANK_BATCH_SIZE", "").strip()
    if rg_embed_rerank_batch_size_env:
        rg_embed_rerank_batch_size = max(1, int(rg_embed_rerank_batch_size_env))
    else:
        rg_embed_rerank_batch_size = 64

    total_start = time.perf_counter()
    embedding_engine = EmbeddingEngine(
        embed_model_path,
        index_dir,
        embed_device,
        model_type=embed_model_type,
        backend=embed_backend,
        max_tokens=embed_max_tokens,
        rerank_max_tokens=embed_rerank_max_tokens,
        rerank_batch_size=embed_rerank_batch_size,
        query_instruction=os.environ.get("QUERY_INSTRUCTION", QUERY_INSTRUCTION),
    )

    rg_embedding_engine = None
    if rg_embed_model_path:
        rg_embedding_engine = EmbeddingEngine(
            rg_embed_model_path,
            index_dir,
            rg_embed_device,
            model_type=rg_embed_model_type or "auto",
            backend=rg_embed_backend or "auto",
            max_tokens=rg_embed_max_tokens,
            rerank_max_tokens=rg_embed_rerank_max_tokens,
            rerank_batch_size=rg_embed_rerank_batch_size,
            query_instruction=os.environ.get("QUERY_INSTRUCTION", QUERY_INSTRUCTION),
            load_index=False,
        )
        log(
            "Loaded separate rg rerank embedding engine: "
            f"model_path={rg_embed_model_path}, model_type={rg_embedding_engine.spec.model_type}, "
            f"backend={rg_embedding_engine.backend}, device={rg_embed_device}"
        )

    reranker_engine = None
    if not args.no_reranker:
        if not qr_model_path:
            raise SystemExit("Missing QR reranker model path. Set --qr-model-path or QR_MODEL_PATH, or pass --no-reranker.")
        reranker_engine = RerankerEngine(qr_model_path, qr_heads, qr_device_str)

    total_load_time = time.perf_counter() - total_start
    log(f"All models loaded in {total_load_time:.1f}s. Server ready.")

    inference_lock = threading.Lock()

    if args.port:
        run_tcp_server(args.port, embedding_engine, rg_embedding_engine, reranker_engine, corpus_dir, inference_lock)
    else:
        send_response({"id": 0, "result": {"status": "ready", "load_time_s": round(total_load_time, 1)}})
        run_stdio_loop(embedding_engine, rg_embedding_engine, reranker_engine, corpus_dir, inference_lock)

    log("Shutting down, releasing GPU memory...")
    del embedding_engine
    if rg_embedding_engine is not None:
        del rg_embedding_engine
    if reranker_engine is not None:
        del reranker_engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("Server stopped.")


if __name__ == "__main__":
    main()
