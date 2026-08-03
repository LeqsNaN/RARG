#!/usr/bin/env python3
"""
Run NeMo agentic retrieval as a BC+ question-answering agent using DCI-Agent-Lite
corpora / indices / embedding formatting.

This is a NEW QA branch and does not modify the existing BRIGHT code path.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from tqdm import tqdm

BASELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_RARG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = DEFAULT_RARG_ROOT / "data" / "bcplus_qa_sample100.jsonl"
DEFAULT_INDEX_DIR = DEFAULT_RARG_ROOT / "data" / "indices" / "bc_plus_100k"
DEFAULT_CORPUS_DIR = DEFAULT_RARG_ROOT / "corpus" / "bc_plus_100k"
DEFAULT_OUTPUT_DIR = BASELINE_ROOT / "outputs" / "nemo_agentic_results_dci_bcplus_qa"

RARG_SCRIPTS_DIR = DEFAULT_RARG_ROOT / "scripts"
if str(RARG_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(RARG_SCRIPTS_DIR))
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from embedding_backends import (  # type: ignore  # noqa: E402
    create_vllm_embedder,
    encode_batch_with_vllm,
    format_query_text,
    get_embedding_spec,
    normalize_backend_override,
    normalize_model_type,
    pool_hidden_states,
    preferred_torch_dtype,
    torch_embeddings_to_numpy,
)
from nemo_agentic import llm_handler, utils  # noqa: E402
from nemo_agentic.agent import Agent  # noqa: E402
from nemo_agentic.configs import AgentConfig, LLMConfig  # noqa: E402
from nemo_agentic.llm_handler import LLM, QuotaExhaustedError, is_error  # noqa: E402
from nemo_agentic.tool_helpers import BaseTool, RetrieveToolBase, ThinkTool  # noqa: E402
from nemo_agentic.utils import rrf_from_subquery_results  # noqa: E402

_RUN_NAME = ""
_OUTPUT_DIR = DEFAULT_OUTPUT_DIR

QUERY_INSTRUCTION = "Given the following question, retrieve relevant documents that help answer the question."
DEFAULT_QWEN3_TOKENIZER_PATH = str(DEFAULT_RARG_ROOT / "models" / "Qwen3-8B")


@dataclass
class QAQueryItem:
    query_id: str
    question: str
    gold_answer: str
    gold_doc_ids: List[str]


def normalize_path(value: str) -> str:
    text = str(value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


@lru_cache(maxsize=4)
def get_qwen3_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )


def truncate_with_qwen3_tokenizer(text: str, tokenizer_path: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    tokenizer = get_qwen3_tokenizer(tokenizer_path)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    trimmed = token_ids[:max_tokens]
    decoded = tokenizer.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return decoded + "\n...[truncated]"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_bcplus_dataset(dataset_path: Path) -> List[QAQueryItem]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing BC+ dataset: {dataset_path}")
    rows = read_jsonl(dataset_path)
    items: List[QAQueryItem] = []
    for row in rows:
        query_id = str(row.get("query_id", row.get("id", len(items))))
        question = str(row.get("query", row.get("question", ""))).strip()
        answer = str(row.get("answer", "")).strip()
        gold_doc_ids = [str(x) for x in (row.get("gold_doc_ids") or row.get("gold_ids") or [])]
        if not question:
            continue
        items.append(
            QAQueryItem(
                query_id=query_id,
                question=question,
                gold_answer=answer,
                gold_doc_ids=gold_doc_ids,
            )
        )
    print(f"BC+ QA: {len(items)} queries loaded from {dataset_path}", flush=True)
    return items


def get_checkpoint_path(run_name: str) -> Path:
    base_dir = Path(_OUTPUT_DIR)
    subdir = base_dir / "checkpoints" / run_name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / "bcplus_qa_checkpoint.jsonl"


def load_checkpoint(run_name: str) -> Dict[str, Dict[str, Any]]:
    ckpt_path = get_checkpoint_path(run_name)
    processed: Dict[str, Dict[str, Any]] = {}
    if ckpt_path.exists():
        with ckpt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    processed[str(item["query_id"])] = item
        print(f"  Loaded checkpoint: {len(processed)} items already processed", flush=True)
    return processed


def save_checkpoint_item(run_name: str, result: Dict[str, Any]) -> None:
    ckpt_path = get_checkpoint_path(run_name)
    with ckpt_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


class QueryEncoder:
    def __init__(
        self,
        *,
        model_path: str,
        model_type: str,
        backend: str,
        index_dir: Path,
        device: str,
        max_model_len: int,
    ) -> None:
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.spec = get_embedding_spec(
            model_path=model_path,
            explicit_model_type=normalize_model_type(model_type),
            index_dir=str(index_dir),
        )
        requested_backend = normalize_backend_override(backend)
        if requested_backend == "auto":
            requested_backend = self.spec.backend
        self.backend = requested_backend

        if self.backend == "vllm":
            try:
                print("Loading query encoder with vLLM...", flush=True)
                self.engine = create_vllm_embedder(
                    model_path=model_path,
                    max_model_len=max_model_len,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.9,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] vLLM query encoder init failed, fallback to Transformers: {e}", flush=True)
                self.backend = "transformers"
                self.engine = self._load_transformers(device)
        else:
            print("Loading query encoder with Transformers...", flush=True)
            self.engine = self._load_transformers(device)

    def _load_transformers(self, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch_device = torch.device(device)
        dtype = preferred_torch_dtype(self.spec, torch_device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side=self.spec.tokenizer_padding_side,
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model = model.to(torch_device).eval()
        return tokenizer, model, torch_device

    def encode_queries(self, queries: List[str], query_instruction: str) -> np.ndarray:
        formatted = [format_query_text(q, self.spec, query_instruction) for q in queries]
        if self.backend == "vllm":
            return encode_batch_with_vllm(self.engine, formatted, self.max_model_len)

        tokenizer, model, device = self.engine
        return self._encode_with_transformers(tokenizer, model, device, formatted)

    def _encode_with_transformers(self, tokenizer, model, device, texts: List[str]) -> np.ndarray:
        import torch

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_model_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.spec.pooling)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return torch_embeddings_to_numpy(embeddings)

    def unload(self) -> None:
        try:
            if getattr(self, "backend", None) == "transformers":
                engine = getattr(self, "engine", None)
                if engine is not None:
                    tokenizer, model, _device = engine
                    del model
                    del tokenizer
            elif hasattr(self, "engine"):
                del self.engine
        finally:
            self.engine = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class BCPlusIndexRetriever:
    def __init__(
        self,
        *,
        index_dir: Path,
        corpus_dir: Path,
        model_path: str,
        model_type: str,
        backend: str,
        device: str,
        max_length: int,
        query_instruction: str,
        retrieved_text_max_chars: int = 0,
        retrieved_text_max_tokens: int = 512,
        qwen3_tokenizer_path: str = DEFAULT_QWEN3_TOKENIZER_PATH,
        max_return_docs: int = 5,
    ) -> None:
        self.index_dir = index_dir
        self.corpus_dir = corpus_dir
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        with (index_dir / "paths.json").open("r", encoding="utf-8") as f:
            self.paths = [normalize_path(x) for x in json.load(f)]
        if self.index.ntotal != len(self.paths):
            raise ValueError(
                f"Index/path mismatch: ntotal={self.index.ntotal}, len(paths)={len(self.paths)} for {index_dir}"
            )
        self.encoder = QueryEncoder(
            model_path=model_path,
            model_type=model_type,
            backend=backend,
            index_dir=index_dir,
            device=device,
            max_model_len=max_length,
        )
        self.query_instruction = query_instruction
        self.retrieved_text_max_chars = int(retrieved_text_max_chars)
        self.retrieved_text_max_tokens = int(retrieved_text_max_tokens)
        self.qwen3_tokenizer_path = str(qwen3_tokenizer_path)
        self.max_return_docs = max(1, int(max_return_docs))
        self._text_cache: Dict[str, str] = {}

    def _load_doc_text(self, doc_id: str) -> str:
        doc_id = normalize_path(doc_id)
        cached = self._text_cache.get(doc_id)
        if cached is not None:
            return cached
        path = self.corpus_dir / doc_id
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        if self.retrieved_text_max_chars > 0 and len(text) > self.retrieved_text_max_chars:
            text = text[: self.retrieved_text_max_chars] + "\n...[truncated]"
        if self.retrieved_text_max_tokens > 0 and text:
            text = truncate_with_qwen3_tokenizer(
                text,
                tokenizer_path=self.qwen3_tokenizer_path,
                max_tokens=self.retrieved_text_max_tokens,
            )
        self._text_cache[doc_id] = text
        return text

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_markdown: bool = False,
        allow_overfetch: bool = False,
    ) -> Any:
        query_vec = self.encoder.encode_queries([str(query)], self.query_instruction)
        if allow_overfetch:
            top_k = max(1, int(top_k))
        else:
            top_k = max(1, min(int(top_k), self.max_return_docs))
        total_docs = len(self.paths)
        search_k = min(total_docs, max(top_k * 4, top_k + 32))

        picked_scores: Dict[str, float] = {}
        picked_texts: Dict[str, str] = {}
        while True:
            scores, indices = self.index.search(query_vec.astype(np.float32), search_k)
            for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
                if idx < 0 or idx >= total_docs:
                    continue
                doc_id = self.paths[idx]
                if doc_id in picked_scores:
                    continue
                picked_scores[doc_id] = float(score)
                if return_markdown:
                    picked_texts[doc_id] = self._load_doc_text(doc_id)
                if len(picked_scores) >= top_k:
                    break
            if len(picked_scores) >= top_k or search_k >= total_docs:
                break
            search_k = min(total_docs, max(search_k + top_k * 2, search_k * 2))

        sorted_items = sorted(picked_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        score_dict = {doc_id: score for doc_id, score in sorted_items}
        if not return_markdown:
            return score_dict
        text_dict = {doc_id: picked_texts.get(doc_id, self._load_doc_text(doc_id)) for doc_id, _ in sorted_items}
        return score_dict, text_dict

    def unload(self) -> None:
        self._text_cache.clear()
        self.paths = []
        if hasattr(self, "index"):
            del self.index
        encoder = getattr(self, "encoder", None)
        if encoder is not None:
            encoder.unload()
            self.encoder = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class RetrieveTool(RetrieveToolBase):
    def __init__(
        self,
        retriever: BCPlusIndexRetriever,
        top_k: int = 5,
        use_original_guarantee: bool = False,
        allow_model_topk: bool = False,
    ):
        self.retriever = retriever
        self._default_top_k = int(top_k)
        self.use_original_guarantee = bool(use_original_guarantee)
        self.allow_model_topk = bool(allow_model_topk)

    def _spec(self) -> Dict[str, Any]:
        properties: Dict[str, Any] = {
            "query": {"type": "string", "description": "Search query for retrieving evidence documents."},
        }
        if self.allow_model_topk:
            properties["top_k"] = {
                "type": "integer",
                "description": "Number of documents to retrieve.",
                "default": self._default_top_k,
            }
        description = "Search for documents related to a question using dense retrieval."
        if not self.allow_model_topk:
            description += f" This tool always returns a fixed top-{self._default_top_k} candidate target controlled by the system."
        return {
            "type": "function",
            "function": {
                "name": "retrieve",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": ["query"],
                },
            },
        }

    async def _acall(self, query: str, top_k: Optional[int] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        overfetch_top_k = int(kwargs.pop("__art_top_k", self._default_top_k))
        requested_top_k = int(top_k or self._default_top_k) if self.allow_model_topk else self._default_top_k
        if self.use_original_guarantee:
            effective_top_k = max(1, overfetch_top_k)
        else:
            effective_top_k = max(1, min(requested_top_k, self.retriever.max_return_docs))
        scores, markdowns = self.retriever.retrieve(
            str(query),
            top_k=effective_top_k,
            return_markdown=True,
            allow_overfetch=self.use_original_guarantee,
        )
        results = [
            {
                "id": str(doc_id),
                "score": float(score),
                "text": str(markdowns.get(doc_id, "")),
            }
            for doc_id, score in scores.items()
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:effective_top_k]


class FinalAnswerTool(BaseTool):
    _name: Optional[str] = "final_answer"

    def __init__(self) -> None:
        self.correct_call_return_value = "The final answer has been successfully logged and the interaction ended."
        self.spec_dict = {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": (
                    "Finish the task by providing the best grounded answer to the question together with the most "
                    "important supporting document IDs."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["answer", "supporting_doc_ids", "evidence_summary", "confidence"],
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Best concise final answer to the question.",
                        },
                        "supporting_doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Document IDs that most directly support the answer, ordered by usefulness.",
                        },
                        "evidence_summary": {
                            "type": "string",
                            "description": "Short explanation of the evidence chain supporting the answer.",
                        },
                        "confidence": {
                            "type": "integer",
                            "description": "Confidence in the answer from 0 to 100.",
                        },
                    },
                },
            },
        }

    def _spec(self) -> Dict[str, Any]:
        return self.spec_dict

    def _call(self, answer: str, supporting_doc_ids: List[str], evidence_summary: str, confidence: int) -> str:
        if not isinstance(answer, str) or not answer.strip():
            raise TypeError("`answer` must be a non-empty string.")
        if not isinstance(supporting_doc_ids, list) or not all(isinstance(x, str) for x in supporting_doc_ids):
            raise TypeError("`supporting_doc_ids` must be a list of strings.")
        if not isinstance(evidence_summary, str):
            raise TypeError("`evidence_summary` must be a string.")
        if not isinstance(confidence, int):
            raise TypeError("`confidence` must be an integer.")
        if confidence < 0 or confidence > 100:
            raise ValueError("`confidence` must be between 0 and 100.")
        return self.correct_call_return_value

    async def _acall(self, answer: str, supporting_doc_ids: List[str], evidence_summary: str, confidence: int) -> str:
        return self._call(
            answer=answer,
            supporting_doc_ids=supporting_doc_ids,
            evidence_summary=evidence_summary,
            confidence=confidence,
        )


def _count_tool_calls(traj: List[Dict[str, Any]]) -> Tuple[int, Dict[str, int]]:
    total = 0
    by_name: Dict[str, int] = {}
    for msg in traj:
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            name = str(fn.get("name", "?"))
            total += 1
            by_name[name] = by_name.get(name, 0) + 1
    return total, by_name


def _usage_int(usage: Dict[str, Any], *path: str) -> int:
    curr: Any = usage
    for key in path:
        if not isinstance(curr, dict):
            return 0
        curr = curr.get(key, 0)
    try:
        return int(curr or 0)
    except Exception:
        return 0


def _usage_float(usage: Dict[str, Any], *path: str) -> float:
    curr: Any = usage
    for key in path:
        if not isinstance(curr, dict):
            return 0.0
        curr = curr.get(key, 0.0)
    try:
        return float(curr or 0.0)
    except Exception:
        return 0.0


def _log_trajectory(query_id: str, question: str, output: Dict[str, Any], final_answer: str, source: str) -> None:
    traj = output.get("agent_trajectories", []) if output else []
    print(f"\n  [{query_id}] Q: {question[:120]}", flush=True)
    step = 0
    for msg in traj:
        role = msg.get("role", "")
        if role == "assistant":
            tool_calls = msg.get("tool_calls", []) or []
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "?")
                    args_str = fn.get("arguments", "")
                    if isinstance(args_str, str):
                        try:
                            args = json.loads(args_str)
                        except Exception:
                            args = {}
                    else:
                        args = args_str or {}
                    if fn_name == "retrieve":
                        query_arg = str(args.get("query", ""))[:80]
                        top_k_arg = args.get("top_k", "default")
                        print(f'    Step {step}: retrieve(query="{query_arg}", top_k={top_k_arg})', flush=True)
                    elif fn_name == "think":
                        thought = str(args.get("thought", ""))[:100]
                        print(f'    Step {step}: think("{thought}...")', flush=True)
                    elif fn_name == "final_answer":
                        ans = str(args.get("answer", ""))[:80]
                        docs = args.get("supporting_doc_ids", [])
                        print(
                            f"    Step {step}: final_answer(answer={ans!r}, docs={docs[:4]}{'...' if len(docs)>4 else ''})",
                            flush=True,
                        )
                step += 1
        elif role == "agent_error":
            print(f"    Step {step}: ERROR: {msg.get('content', '')[:120]}", flush=True)
            step += 1
    print(f"    => Final: [{source}] {final_answer[:160]}", flush=True)


def format_bcplus_answer(explanation: str, exact_answer: str, confidence: int) -> str:
    explanation = str(explanation or "").strip()
    exact_answer = str(exact_answer or "").strip()
    confidence = max(0, min(100, int(confidence)))
    return (
        f"Explanation: {explanation}\n\n"
        f"Exact Answer: {exact_answer}\n\n"
        f"Confidence: {confidence}%"
    )


def parse_bcplus_answer_fields(text: str) -> Tuple[str, str, int]:
    raw = str(text or "").strip()
    explanation = ""
    exact_answer = raw
    confidence = 0

    import re

    m_exp = re.search(r"Explanation:\s*(.*?)(?=\n\s*Exact Answer:|\Z)", raw, re.S | re.I)
    if m_exp:
        explanation = m_exp.group(1).strip()

    m_ans = re.search(r"Exact Answer:\s*(.*?)(?=\n\s*Confidence:|\Z)", raw, re.S | re.I)
    if m_ans:
        exact_answer = m_ans.group(1).strip()

    m_conf = re.search(r"Confidence:\s*(\d{1,3})\s*%", raw, re.I)
    if m_conf:
        try:
            confidence = int(m_conf.group(1))
        except Exception:
            confidence = 0

    if not explanation:
        explanation = "Derived from the retrieved supporting documents."
    return explanation, exact_answer, confidence


def extract_final_answer(output_artifacts: Dict[str, Any]) -> Tuple[str, List[str], str, int, str]:
    traj = output_artifacts.get("agent_trajectories", []) or []
    for msg in reversed(traj):
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            if fn.get("name") != "final_answer":
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = None
            if isinstance(args, dict):
                answer = str(args.get("answer", "")).strip()
                docs = [normalize_path(x) for x in (args.get("supporting_doc_ids") or []) if isinstance(x, str)]
                evidence_summary = str(args.get("evidence_summary", "")).strip()
                confidence_raw = args.get("confidence", 0)
                try:
                    confidence = int(confidence_raw)
                except Exception:
                    confidence = 0
                formatted = format_bcplus_answer(evidence_summary, answer, confidence)
                return formatted, docs, evidence_summary, confidence, "final_answer"
    return "", [], "", 0, "none"


    traj = output_artifacts.get("agent_trajectories", []) or []
    for msg in reversed(traj):
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            if fn.get("name") != "final_answer":
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = None
            if isinstance(args, dict):
                answer = str(args.get("answer", "")).strip()
                docs = [normalize_path(x) for x in (args.get("supporting_doc_ids") or []) if isinstance(x, str)]
                evidence_summary = str(args.get("evidence_summary", "")).strip()
                confidence_raw = args.get("confidence", 0)
                try:
                    confidence = int(confidence_raw)
                except Exception:
                    confidence = 0
                return answer, docs, evidence_summary, confidence, "final_answer"
    return "", [], "", 0, "none"


async def fallback_answer(
    *,
    llm: LLM,
    query_item: QAQueryItem,
    retriever: BCPlusIndexRetriever,
    fallback_top_k: int,
    session_id: str,
    retrieval_log: Optional[List[Dict[str, Any]]] = None,
    fallback_drop_docs: int = 5,
    fallback_min_docs: int = 1,
) -> Tuple[str, List[str], str, int, Dict[str, Any]]:
    retrieval_log = retrieval_log or []

    ranked_docs: List[str] = []
    if retrieval_log:
        try:
            rrf_scores = rrf_from_subquery_results([entry.get("output", []) for entry in retrieval_log if entry.get("output")])
            if rrf_scores:
                ranked_docs = [normalize_path(doc_id) for doc_id in sorted(rrf_scores, key=rrf_scores.get, reverse=True)]
        except Exception:
            ranked_docs = []

    rrf_scores_for_save: Dict[str, float] = {}
    if ranked_docs:
        try:
            rrf_scores_for_save = rrf_from_subquery_results([entry.get("output", []) for entry in retrieval_log if entry.get("output")])
        except Exception:
            rrf_scores_for_save = {}

    fallback_scores, fallback_texts = retriever.retrieve(query_item.question, top_k=fallback_top_k, return_markdown=True)
    fallback_docs = [normalize_path(x) for x in fallback_scores.keys()]

    merged_docs: List[str] = []
    seen = set()
    for doc_id in ranked_docs + fallback_docs:
        nd = normalize_path(doc_id)
        if nd not in seen:
            merged_docs.append(nd)
            seen.add(nd)

    if not merged_docs:
        merged_docs = fallback_docs

    merged_docs = merged_docs[: max(fallback_top_k, fallback_min_docs)]

    extra: Dict[str, Any] = {
        "fallback_rrf_scores": rrf_scores_for_save,
        "fallback_candidate_docs": list(merged_docs),
        "fallback_attempts": [],
    }

    current_docs = list(merged_docs)
    drop_n = max(1, int(fallback_drop_docs))
    min_docs = max(1, int(fallback_min_docs))

    while current_docs:
        doc_blocks = []
        for i, doc_id in enumerate(current_docs, start=1):
            text = retriever._load_doc_text(doc_id)
            doc_blocks.append(f"[Doc {i}] ID: {doc_id}\n{text}")

        user_prompt = (
            "Answer the question using only the retrieved documents below. "
            "If evidence is incomplete, still give the best grounded short answer. "
            "Your response must follow this exact format:\n"
            "Explanation: {brief grounded explanation}\n"
            "Exact Answer: {succinct final answer}\n"
            "Confidence: {0-100%}\n\n"
            f"Question:\n{query_item.question}\n\nRetrieved Documents:\n\n" + "\n\n".join(doc_blocks)
        )
        messages = [
            {"role": "system", "content": "You answer questions using only the provided retrieved documents."},
            {"role": "user", "content": user_prompt},
        ]
        response = await llm.acompletion(
            messages=messages,
            return_metadata=True,
            logging_kwargs={"step": f"fallback_{len(current_docs)}", "subdir": f"fallback_{session_id}", "log_exp_name": session_id},
        )

        if is_error(response):
            answer_text = str(response)
            extra["fallback_attempts"].append({
                "doc_count": len(current_docs),
                "doc_ids": list(current_docs),
                "error": answer_text,
            })
            lowered = answer_text.lower()
            is_ooc = ("context" in lowered and "window" in lowered) or ("context_length_exceeded" in lowered) or ("tokens" in lowered and "exceed" in lowered)
            if is_ooc and len(current_docs) > min_docs:
                next_count = max(min_docs, len(current_docs) - drop_n)
                if next_count < len(current_docs):
                    current_docs = current_docs[:next_count]
                    continue
            extra["fallback_error"] = answer_text
            formatted_error = format_bcplus_answer("Fallback generation failed.", answer_text, 0)
            return formatted_error, [normalize_path(x) for x in current_docs], "Fallback generation failed.", 0, extra

        chat_response = response["response"]
        text = ""
        try:
            text = chat_response.choices[0].message.content or ""
        except Exception:
            text = ""
        extra["fallback_response"] = chat_response.model_dump()
        extra["fallback_attempts"].append({
            "doc_count": len(current_docs),
            "doc_ids": list(current_docs),
            "success": True,
        })
        explanation, exact_answer, parsed_confidence = parse_bcplus_answer_fields(text.strip())
        final_confidence = parsed_confidence if parsed_confidence > 0 else 50
        formatted = format_bcplus_answer(explanation, exact_answer, final_confidence)
        return formatted, [normalize_path(x) for x in current_docs], explanation, final_confidence, extra

    extra["fallback_error"] = "No fallback documents available."
    formatted_empty = format_bcplus_answer("Fallback generation failed.", "", 0)
    return formatted_empty, [], "Fallback generation failed.", 0, extra


def get_query_run_dir(query_id: str) -> Path:
    return Path(_OUTPUT_DIR) / _RUN_NAME / str(query_id)


def build_partial_output(agent: Agent) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "agent_trajectories": getattr(agent, "message_history", []),
        "retrieval_log": getattr(agent, "retrieval_log", []),
        "api_response_extras": getattr(agent, "api_response_extras", []),
    }
    if getattr(agent, "extra_data", None):
        output["agent_extra_data"] = agent.extra_data
    return output


def save_query_inprogress_state(
    *,
    query_item: QAQueryItem,
    agent: Agent,
    started_at: float,
    stage: str,
    partial_result: Optional[Dict[str, Any]] = None,
) -> None:
    run_root = get_query_run_dir(query_item.query_id)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "item.json").write_text(
        json.dumps(
            {
                "query_id": query_item.query_id,
                "question": query_item.question,
                "gold_answer": query_item.gold_answer,
                "gold_doc_ids": query_item.gold_doc_ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "question.txt").write_text(query_item.question, encoding="utf-8")

    partial_output = build_partial_output(agent)
    (run_root / "conversation.json").write_text(json.dumps(partial_output, ensure_ascii=False, indent=2), encoding="utf-8")

    state = {
        "query_id": query_item.query_id,
        "stage": stage,
        "started_at": started_at,
        "agent_state": {
            "message_history": getattr(agent, "message_history", []),
            "current_user_msg": getattr(agent, "current_user_msg", None),
            "steps": int(getattr(agent, "steps", 0)),
            "extra_data": getattr(agent, "extra_data", {}),
            "retrieved_docs": sorted(getattr(agent, "retrieved_docs", set())),
            "retrieval_log": getattr(agent, "retrieval_log", []),
            "exclude_docs": sorted(getattr(agent, "exclude_docs", set())),
            "api_response_extras": getattr(agent, "api_response_extras", []),
        },
        "partial_result": partial_result or {},
    }
    (run_root / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_query_inprogress_state(query_id: str) -> Optional[Dict[str, Any]]:
    state_path = get_query_run_dir(query_id) / "state.json"
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def restore_agent_state(agent: Agent, state: Dict[str, Any]) -> float:
    agent_state = state.get("agent_state", {})
    agent.message_history = list(agent_state.get("message_history", []))
    agent.current_user_msg = agent_state.get("current_user_msg")
    agent.steps = int(agent_state.get("steps", 0))
    agent.extra_data = dict(agent_state.get("extra_data", {}))
    agent.retrieved_docs = set(agent_state.get("retrieved_docs", []))
    agent.retrieval_log = list(agent_state.get("retrieval_log", []))
    agent.exclude_docs = set(agent_state.get("exclude_docs", []))
    agent.api_response_extras = list(agent_state.get("api_response_extras", []))
    return float(state.get("started_at", time.time()))


async def run_agent_for_query_with_resume(
    *,
    query_item: QAQueryItem,
    agent: Agent,
    started_at: Optional[float] = None,
) -> Dict[str, Any]:
    state = load_query_inprogress_state(str(query_item.query_id))
    io_log_data = None

    if state is not None and not (get_query_run_dir(query_item.query_id) / "final.txt").exists():
        started_at = restore_agent_state(agent, state)
        print(f"  [{query_item.query_id}] Resuming in-progress sample from stage={state.get('stage', 'unknown')}", flush=True)
    else:
        agent.reset()
        if agent.tool_map is None:
            raise RuntimeError("Agent requires tool_map to be provided by the caller.")

        started_at = time.time() if started_at is None else started_at
        await agent.llm.log_extra_data_log_dir(subdir=agent.get_llm_raw_io_subdir(), info=None)
        agent.current_user_msg = query_item.question
        agent.exclude_docs = set()

        task_inst_query = f"Query:\n{query_item.question}"
        if agent.config.user_msg_type == "simple":
            agent.message_history.append({"role": "user", "content": [{"type": "text", "text": task_inst_query}]})
        elif agent.config.user_msg_type == "with_results":
            res = await agent.call_one_tool(
                fn_name="retrieve",
                fn_kwargs={"query": query_item.question},
                store_state=False,
                query_type="main",
            )
            user_msg = {
                "role": "user",
                "content": [{"type": "text", "text": task_inst_query}, {"type": "text", "text": "Retrieved Documents:"}] + res,
            }
            agent.message_history.append(user_msg)
        else:
            raise ValueError(f"`{agent.config.user_msg_type}` is not a valid user_msg_type.")
        save_query_inprogress_state(query_item=query_item, agent=agent, started_at=started_at, stage="initialized")

    while True:
        if agent.config.max_steps is not None and agent.steps >= agent.config.max_steps:
            agent.message_history.append(utils.AgentErrorMessage(content="Agent reached maximum allowed iterations").model_dump())
            save_query_inprogress_state(query_item=query_item, agent=agent, started_at=started_at, stage="max_steps_reached")

        if agent.is_last_msg_error():
            break

        new_io_log_data = await agent.step()
        if new_io_log_data is not None:
            io_log_data = new_io_log_data
            if not agent.llm.config.instant_log:
                await llm_handler.awrite_json(**io_log_data["input_json"])
                await llm_handler.awrite_json(**io_log_data["output_json"])
        save_query_inprogress_state(query_item=query_item, agent=agent, started_at=started_at, stage="after_step")

        if not agent.is_last_msg_error():
            _tc = agent.message_history[-1].get("tool_calls", [])
            if _tc is None or len(_tc) == 0:
                agent.message_history.append({"role": "user", "content": [{"type": "text", "text": agent.auto_user_msg}]})
                save_query_inprogress_state(query_item=query_item, agent=agent, started_at=started_at, stage="auto_continue")
            else:
                tool_calls = [tc["function"]["name"] for tc in agent.message_history[-1]["tool_calls"]]
                tool_messages = await agent.process_tool_calls()
                agent.message_history.extend(tool_messages)
                save_query_inprogress_state(query_item=query_item, agent=agent, started_at=started_at, stage="after_tool_calls")
                ended_successfully = False
                if agent.config.end_tool in tool_calls:
                    end_tool = agent.tool_map[agent.config.end_tool]
                    _correct_val = end_tool.correct_call_return_value  # type: ignore[attr-defined]
                    for tm in tool_messages:
                        if tm["name"] == agent.config.end_tool and tm["content"][0]["text"] == _correct_val:
                            ended_successfully = True
                            break
                if ended_successfully:
                    break
        else:
            break

    await agent.llm.log_extra_data_log_dir(
        subdir=agent.get_llm_raw_io_subdir(),
        info=agent.api_response_extras,
        filename="api_response_extras.json",
    )

    output = await agent.conclude_task(query=query_item.question, task_info=None)
    save_query_inprogress_state(query_item=query_item, agent=agent, started_at=started_at, stage="concluded")
    return output


def _usage_stats_from_usage_blob(usage: Dict[str, Any], *, stage: str = "agent") -> Dict[str, Any]:
    pt = _usage_int(usage, "prompt_tokens")
    ct = _usage_int(usage, "completion_tokens")
    tt = _usage_int(usage, "total_tokens")
    rt = _usage_int(usage, "completion_tokens_details", "reasoning_tokens")
    crt = _usage_int(usage, "cache_read_input_tokens")
    if crt <= 0:
        crt = _usage_int(usage, "prompt_tokens_details", "cached_tokens")
    cwt = max(0, pt - crt)
    return {
        "stage": stage,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt if tt > 0 else pt + ct,
        "reasoning_tokens": rt,
        "thinking_tokens": rt,
        "cache_read_input_tokens": crt,
        "cache_write_input_tokens": cwt,
        "cost_usd": _usage_float(usage, "cost_usd"),
        "prompt_cost_usd": _usage_float(usage, "prompt_cost_usd"),
        "completion_cost_usd": _usage_float(usage, "completion_cost_usd"),
        "cache_read_cost_usd": _usage_float(usage, "cache_read_cost_usd"),
    }


def _merge_usage_stats(base_stats: Dict[str, Any], step_stats: Dict[str, Any]) -> None:
    base_stats["total_prompt_tokens"] += int(step_stats.get("prompt_tokens", 0))
    base_stats["total_completion_tokens"] += int(step_stats.get("completion_tokens", 0))
    base_stats["total_reasoning_tokens"] += int(step_stats.get("reasoning_tokens", 0))
    base_stats["total_thinking_tokens"] += int(step_stats.get("thinking_tokens", 0))
    base_stats["total_cache_read_input_tokens"] += int(step_stats.get("cache_read_input_tokens", 0))
    base_stats["total_cache_write_input_tokens"] += int(step_stats.get("cache_write_input_tokens", 0))
    base_stats["total_tokens"] += int(step_stats.get("total_tokens", 0))
    base_stats["total_cost_usd"] += float(step_stats.get("cost_usd", 0.0))
    base_stats["total_prompt_cost_usd"] += float(step_stats.get("prompt_cost_usd", 0.0))
    base_stats["total_completion_cost_usd"] += float(step_stats.get("completion_cost_usd", 0.0))
    base_stats["total_cache_read_cost_usd"] += float(step_stats.get("cache_read_cost_usd", 0.0))
    base_stats["per_step_tokens"].append(step_stats)


def _extract_fallback_usage_steps(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    fallback = output.get("fallback") or {}
    response_blob = fallback.get("fallback_response")
    if not isinstance(response_blob, dict):
        return []
    usage = response_blob.get("usage")
    if not isinstance(usage, dict):
        return []
    return [_usage_stats_from_usage_blob(usage, stage="fallback")]


def _extract_stats(output: Dict[str, Any], agent: Agent, elapsed: float) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "elapsed_seconds": round(elapsed, 2),
        "num_turns": int(getattr(agent, "steps", 0)),
        "num_tool_calls": 0,
        "tool_call_breakdown": {},
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_reasoning_tokens": 0,
        "total_thinking_tokens": 0,
        "total_cache_read_input_tokens": 0,
        "total_cache_write_input_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "total_prompt_cost_usd": 0.0,
        "total_completion_cost_usd": 0.0,
        "total_cache_read_cost_usd": 0.0,
        "per_step_tokens": [],
        "all_retrieved_doc_ids": sorted(getattr(agent, "retrieved_docs", set())),
        "num_unique_docs_seen": len(getattr(agent, "retrieved_docs", set())),
        "num_retrieval_calls": len(getattr(agent, "retrieval_log", [])),
    }

    for extra in getattr(agent, "api_response_extras", []):
        if isinstance(extra, dict):
            usage = extra.get("usage", {})
            if isinstance(usage, dict):
                _merge_usage_stats(stats, _usage_stats_from_usage_blob(usage, stage="agent"))

    for fallback_step in _extract_fallback_usage_steps(output):
        _merge_usage_stats(stats, fallback_step)

    traj = output.get("agent_trajectories", []) or []
    tool_count, tool_breakdown = _count_tool_calls(traj)
    stats["num_tool_calls"] = tool_count
    stats["tool_call_breakdown"] = tool_breakdown
    return stats


def save_query_artifacts(*, query_item: QAQueryItem, result: Dict[str, Any], output: Dict[str, Any]) -> None:
    run_root = Path(_OUTPUT_DIR) / _RUN_NAME / str(query_item.query_id)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "item.json").write_text(
        json.dumps(
            {
                "query_id": query_item.query_id,
                "question": query_item.question,
                "gold_answer": query_item.gold_answer,
                "gold_doc_ids": query_item.gold_doc_ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "conversation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "question.txt").write_text(query_item.question, encoding="utf-8")
    (run_root / "final.txt").write_text(result.get("predicted_answer", ""), encoding="utf-8")


async def run_bcplus_qa(
    items: List[QAQueryItem],
    retriever: BCPlusIndexRetriever,
    llm: LLM,
    agent_config: AgentConfig,
    initial_retrieval_top_k: int,
    fallback_top_k: int,
    run_name: str,
    fallback_drop_docs: int,
    fallback_min_docs: int,
    use_original_retrieve_guarantee: bool,
    allow_model_topk: bool,
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}", flush=True)
    print(f"Running NeMo Agentic QA on BC+ ({len(items)} queries)", flush=True)
    print(f"{'=' * 60}", flush=True)

    processed = load_checkpoint(run_name)
    remaining = [q for q in items if q.query_id not in processed]
    print(f"  Already processed: {len(processed)}, remaining: {len(remaining)}", flush=True)

    if remaining:
        completed = len(processed)
        try:
            for query_item in tqdm(remaining, desc="BC+ QA"):
                t0 = time.time()
                retrieve_tool = RetrieveTool(
                    retriever=retriever,
                    top_k=initial_retrieval_top_k,
                    use_original_guarantee=use_original_retrieve_guarantee,
                    allow_model_topk=allow_model_topk,
                )
                tool_map: Dict[str, BaseTool] = {"retrieve": retrieve_tool}
                final_answer_tool = FinalAnswerTool()
                tool_map[final_answer_tool.name] = final_answer_tool
                if not agent_config.disable_think:
                    think = ThinkTool(extended_relevance=agent_config.extended_relevance)
                    tool_map[think.name] = think

                agent = Agent(config=agent_config, llm=llm, tool_map=tool_map, session_id=query_item.query_id)
                try:
                    output = await run_agent_for_query_with_resume(query_item=query_item, agent=agent, started_at=t0)
                    final_answer, support_docs, evidence_summary, confidence, source = extract_final_answer(output)
                    elapsed = time.time() - t0
                    stats = _extract_stats(output, agent, elapsed)
                    if not final_answer:
                        fallback_answer_text, fallback_docs, fallback_summary, fallback_conf, fallback_extra = await fallback_answer(
                            llm=llm,
                            query_item=query_item,
                            retriever=retriever,
                            fallback_top_k=fallback_top_k,
                            session_id=str(query_item.query_id),
                            retrieval_log=getattr(agent, "retrieval_log", []),
                            fallback_drop_docs=fallback_drop_docs,
                            fallback_min_docs=fallback_min_docs,
                        )
                        final_answer = fallback_answer_text
                        support_docs = fallback_docs
                        evidence_summary = fallback_summary
                        confidence = fallback_conf
                        source = "fallback_answer"
                        output["fallback"] = fallback_extra
                    _log_trajectory(query_item.query_id, query_item.question, output, final_answer, source)
                except QuotaExhaustedError:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"  Error for {query_item.query_id}: {type(e).__name__}: {e}", flush=True)
                    elapsed = time.time() - t0
                    final_answer, support_docs, evidence_summary, confidence, fallback_extra = await fallback_answer(
                        llm=llm,
                        query_item=query_item,
                        retriever=retriever,
                        fallback_top_k=fallback_top_k,
                        session_id=str(query_item.query_id),
                        retrieval_log=getattr(agent, "retrieval_log", []),
                        fallback_drop_docs=fallback_drop_docs,
                        fallback_min_docs=fallback_min_docs,
                    )
                    source = "fallback_answer_on_error"
                    output = {
                        "agent_trajectories": getattr(agent, "message_history", []),
                        "retrieval_log": getattr(agent, "retrieval_log", []),
                        "fallback": fallback_extra,
                        "fallback_error": f"{type(e).__name__}: {e}",
                    }
                    stats = _extract_stats(output, agent, elapsed)

                parsed_explanation, parsed_exact_answer, parsed_conf = parse_bcplus_answer_fields(final_answer)
                if confidence <= 0 and parsed_conf > 0:
                    confidence = parsed_conf
                if not evidence_summary:
                    evidence_summary = parsed_explanation
                result = {
                    "query_id": query_item.query_id,
                    "question": query_item.question,
                    "gold_answer": query_item.gold_answer,
                    "gold_doc_ids": query_item.gold_doc_ids,
                    "predicted_answer": final_answer,
                    "exact_answer": parsed_exact_answer,
                    "supporting_doc_ids": [normalize_path(x) for x in support_docs],
                    "evidence_summary": evidence_summary,
                    "confidence": confidence,
                    "source": source,
                    "stats": stats,
                }
                save_query_artifacts(query_item=query_item, result=result, output=output)
                save_query_inprogress_state(query_item=query_item, agent=agent, started_at=t0, stage="completed", partial_result=result)
                save_checkpoint_item(run_name, result)
                processed[query_item.query_id] = result

                completed += 1
                if completed % 10 == 0 or completed == len(items):
                    print(f"  Progress: {completed}/{len(items)}", flush=True)

        except QuotaExhaustedError as e:
            print(f"\n  !!! QUOTA EXHAUSTED: {e}", flush=True)
            print(f"  Saved {len(processed)} items to checkpoint. Exiting...", flush=True)
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n  !!! Interrupted. Saved {len(processed)} items to checkpoint.", flush=True)
            sys.exit(1)

    results = [processed[q.query_id] for q in items if q.query_id in processed]
    total_queries = len(items)
    processed_queries = len(results)
    avg_turns = sum(int(r.get("stats", {}).get("num_turns", 0)) for r in results) / processed_queries if processed_queries else 0.0
    avg_tool_calls = sum(int(r.get("stats", {}).get("num_tool_calls", 0)) for r in results) / processed_queries if processed_queries else 0.0
    avg_retrieval_calls = sum(int(r.get("stats", {}).get("num_retrieval_calls", 0)) for r in results) / processed_queries if processed_queries else 0.0
    avg_prompt_tokens = sum(int(r.get("stats", {}).get("total_prompt_tokens", 0)) for r in results) / processed_queries if processed_queries else 0.0
    avg_completion_tokens = sum(int(r.get("stats", {}).get("total_completion_tokens", 0)) for r in results) / processed_queries if processed_queries else 0.0
    avg_cost_usd = sum(float(r.get("stats", {}).get("total_cost_usd", 0.0)) for r in results) / processed_queries if processed_queries else 0.0
    source_distribution: Dict[str, int] = {}
    for r in results:
        src = str(r.get("source", "unknown"))
        source_distribution[src] = source_distribution.get(src, 0) + 1

    summary = {
        "total_queries": total_queries,
        "processed_queries": processed_queries,
        "avg_turns": round(avg_turns, 2),
        "avg_tool_calls": round(avg_tool_calls, 2),
        "avg_retrieval_calls": round(avg_retrieval_calls, 2),
        "avg_prompt_tokens": round(avg_prompt_tokens, 2),
        "avg_completion_tokens": round(avg_completion_tokens, 2),
        "avg_cost_usd": round(avg_cost_usd, 6),
        "source_distribution": source_distribution,
    }

    print("\n=== BC+ QA Summary ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeMo Agentic Retrieval for BC+ QA using DCI-Agent-Lite embeddings.")
    parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--index_dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--corpus_dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_RARG_ROOT / "models" / "Qwen3-Embedding-4B"),
    )
    parser.add_argument("--embed_model_type", type=str, default="qwen3_embedding_4b")
    parser.add_argument("--embed_backend", type=str, default="auto")
    parser.add_argument("--embed_device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--llm_model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--llm_base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--reasoning_effort", type=str, default="medium")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--initial_retrieval_top_k", type=int, default=5)
    parser.add_argument("--fallback_top_k", type=int, default=5)
    parser.add_argument("--run_name", type=str, default="bcplus_qa_qwen3emb_gpt-5.4-mini_medium")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieved_text_max_chars", type=int, default=0)
    parser.add_argument("--retrieved_text_max_tokens", type=int, default=512)
    parser.add_argument("--qwen3_tokenizer_path", type=str, default=DEFAULT_QWEN3_TOKENIZER_PATH)
    parser.add_argument("--max_return_docs", type=int, default=5)
    parser.add_argument("--fallback_drop_docs", type=int, default=5)
    parser.add_argument("--fallback_min_docs", type=int, default=1)
    parser.add_argument("--use_original_retrieve_guarantee", action="store_true")
    parser.add_argument("--allow_model_topk", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global _RUN_NAME, _OUTPUT_DIR
    _RUN_NAME = args.run_name
    _OUTPUT_DIR = args.output_dir
    Path(_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    dataset_items = load_bcplus_dataset(args.dataset_path)
    if args.limit is not None:
        dataset_items = dataset_items[: args.limit]
        print(f"Truncated to first {len(dataset_items)} samples", flush=True)

    llm_config = LLMConfig(
        model=args.llm_model,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=args.llm_base_url,
        reasoning_effort=args.reasoning_effort,
        raw_log_pardir=str(Path(_OUTPUT_DIR) / "agent_logs" / _RUN_NAME),
        drop_params=True,
    )
    llm = LLM(llm_config)

    agent_config = AgentConfig(
        system_prompt="03_bcplus_qa_v0.j2",
        enforce_top_k=False,
        target_top_k=None,
        max_steps=args.max_steps,
        end_tool="final_answer",
        main_agent_only=True,
        extended_relevance=True,
        user_msg_type="with_results",
        calculate_rrf=False,
        selection_topk_list=[],
        ensure_new_docs=bool(args.use_original_retrieve_guarantee),
    )

    retriever = BCPlusIndexRetriever(
        index_dir=args.index_dir,
        corpus_dir=args.corpus_dir,
        model_path=args.model_path,
        model_type=args.embed_model_type,
        backend=args.embed_backend,
        device=args.embed_device,
        max_length=args.max_length,
        query_instruction=QUERY_INSTRUCTION,
        retrieved_text_max_chars=args.retrieved_text_max_chars,
        retrieved_text_max_tokens=args.retrieved_text_max_tokens,
        qwen3_tokenizer_path=args.qwen3_tokenizer_path,
        max_return_docs=args.max_return_docs,
    )

    start_time = time.time()
    try:
        summary = asyncio.run(
            run_bcplus_qa(
                items=dataset_items,
                retriever=retriever,
                llm=llm,
                agent_config=agent_config,
                initial_retrieval_top_k=args.initial_retrieval_top_k,
                fallback_top_k=args.fallback_top_k,
                run_name=args.run_name,
                fallback_drop_docs=args.fallback_drop_docs,
                fallback_min_docs=args.fallback_min_docs,
                use_original_retrieve_guarantee=args.use_original_retrieve_guarantee,
                allow_model_topk=args.allow_model_topk,
            )
        )
    finally:
        print("Unloading retriever and clearing cache...", flush=True)
        retriever.unload()

    summary["dataset_path"] = str(args.dataset_path)
    summary["index_dir"] = str(args.index_dir)
    summary["corpus_dir"] = str(args.corpus_dir)
    summary["retrieved_text_max_tokens"] = int(args.retrieved_text_max_tokens)
    summary["max_return_docs"] = int(args.max_return_docs)
    summary["fallback_drop_docs"] = int(args.fallback_drop_docs)
    summary["fallback_min_docs"] = int(args.fallback_min_docs)
    summary["use_original_retrieve_guarantee"] = bool(args.use_original_retrieve_guarantee)
    summary["allow_model_topk"] = bool(args.allow_model_topk)
    summary["elapsed_seconds"] = round(time.time() - start_time, 1)
    summary_path = Path(_OUTPUT_DIR) / f"{_RUN_NAME}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
