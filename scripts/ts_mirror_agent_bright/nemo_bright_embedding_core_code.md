# NeMo Retriever Agentic 在 BRIGHT 上的 Embedding 核心代码

源码文件（均位于 `NVIDIA/NeMo-Retriever` 仓库 `retrieval-bench/src/retrieval_bench/`）：

- `pipelines/backends.py` —— backend 注册与初始化入口（决定用哪个模型、传哪些参数）
- `singletons/hf_dense_retriever.py` —— 真正做 embedding 的单例 retriever（模型加载 / tokenizer / pooling / 打分）
- `prompts/bright_instructions.py` —— BRIGHT 各子任务的 query/doc 前缀

embedding 模型：`nvidia/llama-nv-embed-reasoning-3b`（backend 名 `llama-nv-embed-reasoning-3b`），**max_length = 8192**。

---

## 1. backends.py —— 默认参数 & 初始化（关键：`max_length` 在这里定死）

```python
# pipelines/backends.py  (行 ~ 33-52)
_BACKEND_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "llama-nv-embed-reasoning-3b": {
        "model_id": "nvidia/llama-nv-embed-reasoning-3b",
        "max_length": 8192,                       # <-- max_token 上限
        "pooling": "mean",
        "score_scale": 100.0,
        "corpus_batch_size": 1,
        "max_scoring_batch_size": 4096,
        "query_prefix_fallback": (
            "Instruct: Given the following post, retrieve relevant passages that help answer the post.\nQuery:"
        ),
    },
    ...
}
```

```python
# pipelines/backends.py  (行 ~ 157-204)  init_backend() 里针对该 backend 的分支
if backend == "llama-nv-embed-reasoning-3b":
    from retrieval_bench.prompts.bright_instructions import (
        NEMO_REASONING_PASSAGE_PREFIX,
        get_bright_query_prefix_nemo,
    )

    query_prefix_fallback = str(cfg.pop("query_prefix_fallback"))
    query_prefix = get_bright_query_prefix_nemo(task_key=task_key, fallback=query_prefix_fallback)

    pooling = str(cfg.pop("pooling", "mean"))
    max_length = int(cfg.pop("max_length", 8192))
    score_scale = float(cfg.pop("score_scale", 100.0))
    corpus_batch_size = int(cfg.pop("corpus_batch_size", 1))
    max_scoring_batch_size = int(cfg.pop("max_scoring_batch_size", 4096))

    if cfg:
        raise ValueError(f"Unknown pipeline arg(s) for backend {backend!r}: {', '.join(sorted(cfg))}")

    retriever.init(
        dataset_name=dataset_name,
        corpus_ids=corpus_ids,
        corpus=corpus,
        model_id=model_id,
        device="cuda",
        top_k=top_k,
        max_length=max_length,            # 8192
        pooling=pooling,                  # "mean"
        doc_prefix=str(NEMO_REASONING_PASSAGE_PREFIX),   # "passage: "
        query_prefix=str(query_prefix),                  # 按 task 不同的 Instruct...Query: 前缀
        task_description="Given the following post, retrieve relevant passages that help answer the post.",
        score_scale=score_scale,          # 100.0
        batch_size=1,
        corpus_batch_size=corpus_batch_size,
        max_scoring_batch_size=max_scoring_batch_size,
        cache_dir="cache/hf_dense",
    )
```

---

## 2. hf_dense_retriever.py —— 真正处理 embedding 的核心逻辑

### 2.1 模型加载（强制 GPU + fp16）

```python
# singletons/hf_dense_retriever.py  (行 185-205)
def _load_model_and_tokenizer(self):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This dense retriever requires an NVIDIA GPU.")
    if not str(self.device).startswith("cuda"):
        raise RuntimeError(
            f"Invalid device '{self.device}'. This dense retriever is GPU-only; use 'cuda'/'cuda:0'."
        )

    from transformers import AutoModel, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
    model.eval()
    model.to(self.device)
    model.half()           # -> fp16
    return tokenizer, model
```

### 2.2 Tokenizer：max_length=8192 + 截断 + padding

```python
# singletons/hf_dense_retriever.py  (行 283-292)
def _tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
    batch = self.tokenizer(
        list(texts),
        max_length=int(self.max_length),     # 8192
        padding=True,
        truncation=True,                     # 超长截断
        return_tensors="pt",
        pad_to_multiple_of=8,
    )
    return {k: v.to(self.device) for k, v in batch.items()}
```

### 2.3 查询编码（拼接 BRIGHT 专属 query_prefix）

```python
# singletons/hf_dense_retriever.py  (行 353-367)
def embed_query(self, query_text: str) -> torch.Tensor:
    if isinstance(self.query_prefix, str) and self.query_prefix:
        q = str(self.query_prefix) + str(query_text)   # 前缀 + query
    else:
        q = _wrap_instruct(self.task_description, str(query_text))

    with torch.no_grad():
        batch = self._tokenize([q])
        outputs = self.model(**batch)
        pooled = self._pool(outputs.last_hidden_state, batch["attention_mask"])
        mode = str(self.pooling or "last_token").strip().lower()
        if mode not in ("mean", "avg", "average"):
            pooled = F.normalize(pooled, p=2, dim=1)

    return pooled[0].detach()  # [dim], stays on GPU
```

### 2.4 文档批量编码 + mean pooling

```python
# singletons/hf_dense_retriever.py  (行 294-309)
def _embed_texts_batched(self, texts: Sequence[str], *, batch_size: int) -> torch.Tensor:
    out: List[torch.Tensor] = []
    bs = max(1, int(batch_size))

    with torch.no_grad():
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            batch = self._tokenize(chunk)
            outputs = self.model(**batch)
            pooled = self._pool(outputs.last_hidden_state, batch["attention_mask"])
            mode = str(self.pooling or "last_token").strip().lower()
            if mode not in ("mean", "avg", "average"):
                pooled = F.normalize(pooled, p=2, dim=1)
            out.append(pooled.detach().to("cpu"))
    return torch.cat(out, dim=0) if out else torch.empty((0, 0), dtype=torch.float16, device="cpu")
```

```python
# singletons/hf_dense_retriever.py  (行 72-87)
def _mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Upcast is important for numeric stability (matches training/eval reference).
    last_hidden_states = last_hidden_states.to(torch.float32)
    last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    denom = attention_mask.sum(dim=1)[..., None].to(torch.float32).clamp(min=1.0)
    embedding = last_hidden_states_masked.sum(dim=1) / denom
    embedding = F.normalize(embedding, dim=-1)   # L2 归一化
    return embedding.to(torch.float16)            # 存为 fp16
```

### 2.5 打分（score_scale=100.0 缩放点积）

```python
# singletons/hf_dense_retriever.py  (行 369-391)
def score_query(self, query_embedding: torch.Tensor) -> torch.Tensor:
    emb_gpu = self.corpus_embeddings_gpu
    ...
    chunk = max(1, int(self.max_scoring_batch_size))   # 4096
    scale = float(self.score_scale)                    # 100.0

    with torch.no_grad():
        q_col = query_embedding.unsqueeze(1)            # [dim, 1]
        for c_start in range(0, num_docs, chunk):
            c_end = min(c_start + chunk, num_docs)
            c_chunk = emb_gpu[c_start:c_end] if emb_gpu is not None else emb_cpu[c_start:c_end].to(device)
            chunk_scores = torch.matmul(c_chunk, q_col).squeeze(1).float() * scale
            scores[c_start:c_end] = chunk_scores
    return scores
```

---

## 3. bright_instructions.py —— BRIGHT 专属前缀

```python
# prompts/bright_instructions.py
BRIGHT_NEMO_QUERY_PREFIXES: dict[str, str] = {
    "biology": "Instruct: Given a Biology post, retrieve relevant passages that help answer the post.\nQuery:",
    "earth_science": "Instruct: Given an Earth Science post, retrieve relevant passages that help answer the post.\nQuery:",
    "economics": "Instruct: Given an Economics post, retrieve relevant passages that help answer the post.\nQuery:",
    "psychology": "Instruct: Given a Psychology post, retrieve relevant passages that help answer the post.\nQuery:",
    "robotics": "Instruct: Given a Robotics post, retrieve relevant passages that help answer the post.\nQuery:",
    "stackoverflow": "Instruct: Given a Stack Overflow post, retrieve relevant passages that help answer the post.\nQuery:",
    "sustainable_living": "Instruct: Given a Sustainable Living post, retrieve relevant passages that help answer the post.\nQuery:",
    "leetcode": "Instruct: Given a Coding problem, retrieve relevant examples that help answer the problem.\nQuery:",
    "pony": "Instruct: Given a Pony question, retrieve relevant passages that help answer the question.\nQuery:",
    "aops": "Instruct: Given a Math problem, retrieve relevant examples that help answer the problem.\nQuery:",
    "theoremqa_questions": "Instruct: Given a Math problem, retrieve relevant examples that help answer the problem.\nQuery:",
    "theoremqa_theorems": "Instruct: Given a Math problem, retrieve relevant theorems that help answer the problem.\nQuery:",
}

NEMO_REASONING_PASSAGE_PREFIX: str = "passage: "   # 文档统一前缀
```

---

## 关键参数速查

| 参数 | 值 | 位置 |
|---|---|---|
| 模型 | `nvidia/llama-nv-embed-reasoning-3b` | backends.py |
| `max_length` (max_token) | **8192** | backends.py → hf_dense_retriever.py `_tokenize` |
| `pooling` | `mean` | backends.py |
| `score_scale` | `100.0` | backends.py → `score_query` |
| `doc_prefix` | `passage: ` | bright_instructions.py |
| `query_prefix` | 每个 BRIGHT task 一个 `Instruct...Query:` | bright_instructions.py |
| `device` | `cuda` + `model.half()` (fp16) | hf_dense_retriever.py |
| `batch_size` / `corpus_batch_size` | `1` | backends.py / hf_dense_retriever.py |
| `max_scoring_batch_size` | `4096` | backends.py → `score_query` |
| `top_k` | `100` | hf_dense_retriever.py |
