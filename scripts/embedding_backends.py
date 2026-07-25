#!/usr/bin/env python3
"""
Shared embedding-model helpers for DCI-Agent-Lite.

This module centralizes model-family-specific behavior that must stay aligned
between:
  - offline corpus indexing
  - online query encoding / semantic recall
  - embedding-based passage reranking

Currently supported model families:
  - Qwen3-Embedding-4B
  - llama-nv-embed-reasoning-3b
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

DEFAULT_QWEN3_EMBED_MODEL = str(_MODELS_DIR / "Qwen3-Embedding-4B")
DEFAULT_LLAMA_NV_REASONING_EMBED_MODEL = str(_MODELS_DIR / "llama-nv-embed-reasoning-3b")
DEFAULT_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
DEFAULT_LLAMA_NV_QUERY_PREFIX_FALLBACK = (
    "Instruct: Given the following post, retrieve relevant passages that help answer the post.\nQuery:"
)


@dataclass(frozen=True)
class EmbeddingSpec:
    model_type: str
    backend: str
    pooling: str
    tokenizer_padding_side: str
    query_style: str
    query_prefix: str
    passage_prefix: str
    preferred_torch_dtype: str


EMBEDDING_SPECS = {
    "qwen3_embedding_4b": EmbeddingSpec(
        model_type="qwen3_embedding_4b",
        backend="vllm",
        pooling="last_token",
        tokenizer_padding_side="left",
        query_style="qwen_instruction",
        query_prefix="",
        passage_prefix="",
        preferred_torch_dtype="bfloat16",
    ),
    "llama_nv_embed_reasoning_3b": EmbeddingSpec(
        model_type="llama_nv_embed_reasoning_3b",
        backend="transformers",
        pooling="avg",
        tokenizer_padding_side="left",
        query_style="query_prefix",
        query_prefix="query: ",
        passage_prefix="passage: ",
        preferred_torch_dtype="float16",
    ),
}


def normalize_model_type(model_type: str | None) -> str:
    value = (model_type or "").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "qwen": "qwen3_embedding_4b",
        "qwen3": "qwen3_embedding_4b",
        "qwen3_emb": "qwen3_embedding_4b",
        "qwen3_embedding": "qwen3_embedding_4b",
        "qwen3_embedding_4b": "qwen3_embedding_4b",
        "llama_nv": "llama_nv_embed_reasoning_3b",
        "llama_nv_reasoning": "llama_nv_embed_reasoning_3b",
        "llama_nv_reasoning_3b": "llama_nv_embed_reasoning_3b",
        "llama_nv_embed_reasoning": "llama_nv_embed_reasoning_3b",
        "llama_nv_embed_reasoning_3b": "llama_nv_embed_reasoning_3b",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(f"Unsupported embedding model type: {model_type}")
    return normalized


def normalize_backend_override(backend: str | None) -> str:
    value = (backend or "").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "hf": "transformers",
        "torch": "transformers",
        "transformer": "transformers",
        "transformers": "transformers",
        "vllm": "vllm",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(f"Unsupported embedding backend override: {backend}")
    return normalized


def infer_model_type_from_model_path(model_path: str) -> str:
    lowered = (model_path or "").strip().lower()
    if "llama-nv-embed-reasoning-3b" in lowered or "llama_nv_embed_reasoning_3b" in lowered:
        return "llama_nv_embed_reasoning_3b"
    return "qwen3_embedding_4b"


def read_index_meta(index_dir: str) -> dict:
    meta_path = Path(index_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def resolve_model_type(
    *,
    explicit_model_type: str | None,
    model_path: str,
    index_dir: str | None = None,
) -> str:
    normalized = normalize_model_type(explicit_model_type)
    if normalized != "auto":
        return normalized

    if index_dir:
        meta = read_index_meta(index_dir)
        meta_type = meta.get("embed_model_type") or meta.get("model_type")
        if meta_type:
            return normalize_model_type(str(meta_type))

    return infer_model_type_from_model_path(model_path)


def get_embedding_spec(
    *,
    model_path: str,
    explicit_model_type: str | None = None,
    index_dir: str | None = None,
) -> EmbeddingSpec:
    model_type = resolve_model_type(
        explicit_model_type=explicit_model_type,
        model_path=model_path,
        index_dir=index_dir,
    )
    return EMBEDDING_SPECS[model_type]


def format_query_text(query: str, spec: EmbeddingSpec, query_instruction: str = "") -> str:
    if spec.query_style == "qwen_instruction":
        instruction = (query_instruction or DEFAULT_QUERY_INSTRUCTION).strip()
        return f"Instruct: {instruction}\nQuery: {query}"
    if spec.query_style == "query_prefix":
        if spec.model_type == "llama_nv_embed_reasoning_3b":
            instruction = (query_instruction or "").strip()
            if instruction:
                return f"Instruct: {instruction}\nQuery: {query}"
            return f"{DEFAULT_LLAMA_NV_QUERY_PREFIX_FALLBACK}{query}"
        return f"{spec.query_prefix}{query}"
    return query


def format_passage_text(text: str, spec: EmbeddingSpec) -> str:
    if spec.passage_prefix:
        return f"{spec.passage_prefix}{text}"
    return text


def last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Qwen3-Embedding uses left padding in the recommended setup.
    # In that case, the last valid token is always at the final sequence position.
    # For right padding, we need to gather the final non-pad token per sample.
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, sequence_lengths]


def average_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.to(torch.float32)
    masked_hidden_states = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    denom = attention_mask.sum(dim=1, keepdim=True).to(torch.float32).clamp(min=1.0)
    return masked_hidden_states.sum(dim=1) / denom


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "last_token":
        return last_token_pooling(hidden_states, attention_mask)
    if pooling == "avg":
        return average_pool(hidden_states, attention_mask)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def normalize_numpy_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return embeddings / norms


def normalize_torch_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def torch_embeddings_to_numpy(embeddings: torch.Tensor) -> np.ndarray:
    return embeddings.float().cpu().numpy().astype(np.float32, copy=False)


def preferred_torch_dtype(spec: EmbeddingSpec, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if spec.preferred_torch_dtype == "bfloat16":
        return torch.bfloat16
    if spec.preferred_torch_dtype == "float16":
        return torch.float16
    return torch.float32


def create_vllm_embedder(
    *,
    model_path: str,
    max_model_len: int,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
):
    from vllm import LLM

    common_kwargs = dict(
        model=model_path,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    errors = []
    for extra_kwargs in ({"task": "embed"}, {"runner": "pooling"}):
        try:
            return LLM(**common_kwargs, **extra_kwargs)
        except TypeError as e:
            errors.append(f"{extra_kwargs}: {e}")

    raise TypeError("Unable to construct vLLM embedder. Tried: " + " | ".join(errors))


def encode_batch_with_vllm(llm, texts: list[str], max_model_len: int) -> np.ndarray:
    outputs = llm.embed(texts, truncate_prompt_tokens=max_model_len)
    batch_embeddings = [output.outputs.embedding for output in outputs]
    embeddings = np.array(batch_embeddings, dtype=np.float32)
    return normalize_numpy_embeddings(embeddings)


def probe_vllm_embedder(llm, *, probe_text: str, max_model_len: int) -> int:
    embeddings = encode_batch_with_vllm(llm, [probe_text], max_model_len)
    if embeddings.ndim != 2 or embeddings.shape[0] != 1:
        raise RuntimeError(f"Unexpected vLLM probe shape: {embeddings.shape}")
    return int(embeddings.shape[1])
