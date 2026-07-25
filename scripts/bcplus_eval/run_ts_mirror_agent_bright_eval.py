#!/usr/bin/env python3
"""
Evaluation runner for BRIGHT using ts_mirror_agent_bright.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_PATH = Path(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_PATH.parents[2]

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ts_mirror_agent_bright.agent import AgentSession, save_outputs  # type: ignore  # noqa: E402
from ts_mirror_agent_bright.config import AgentConfig  # type: ignore  # noqa: E402
from ts_mirror_agent.llm_client import LLMClient  # type: ignore  # noqa: E402
from ts_mirror_agent.model_client import ModelServerClient  # type: ignore  # noqa: E402
from ts_mirror_agent.tools import build_tools  # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)


def _suppress_noisy_third_party_debug_logs() -> None:
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


TASK_QUERY_INSTRUCTIONS: Dict[str, str] = {
    "biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "earth_science": "Given an Earth Science post, retrieve relevant passages that help answer the post.",
    "economics": "Given an Economics post, retrieve relevant passages that help answer the post.",
    "robotics": "Given a Robotics post, retrieve relevant passages that help answer the post.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS-Mirror BRIGHT Agent eval")

    parser.add_argument("--dataset", type=Path, required=True, help="JSONL dataset")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--corpus-dir", type=Path, required=True, help="Corpus directory")
    parser.add_argument("--subset", type=str, default="", help="BRIGHT subset name")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Parallel queries (kept at 1)")
    parser.add_argument("--limit", type=int, help="Only run first N questions")
    parser.add_argument("--skip-completed", action="store_true", default=True, help="Skip queries with existing final.txt")

    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--thinking-level", default="high")
    parser.add_argument("--max-turns", type=int, default=150)
    parser.add_argument("--force-answer-turns", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--keep-tool-calls", type=int, default=30)
    parser.add_argument("--compaction-threshold", type=int, default=150000)
    parser.add_argument("--use-usage-based-compaction", action="store_true")
    parser.add_argument("--usage-based-compaction-min-tokens", type=int, default=110000)
    parser.add_argument("--usage-based-compaction-max-tokens", type=int, default=130000)

    parser.add_argument("--index-dir", type=str, required=True)
    parser.add_argument("--embed-model-path", type=str, default="")
    parser.add_argument("--qr-model-path", type=str, default="")
    parser.add_argument("--qr-heads", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embed-top-k", type=int, default=5000)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--query-instruction", default="")
    parser.add_argument("--max-retrieved-docs", type=int, default=20)

    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument("--append-system-prompt-file", default="")
    parser.add_argument("--force-answer-prompt-file", default="")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_bright_prompt(query: str) -> str:
    return "Retrieve and rank the most relevant documents for the following query.\n\nQUERY:\n" + query


def parse_retrieved_docs(result_text: str) -> List[str]:
    result_text = result_text.replace("\\n", "\n")
    section_match = re.search(
        r"Relevant Documents.*?(1[\.\)].+?)(?:\n\n|\Z)",
        result_text,
        re.DOTALL,
    )
    if not section_match:
        return []

    paths: List[str] = []
    for line in section_match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line).strip()
        if not line or line.startswith("("):
            continue
        line = line.replace("`", "")
        line = re.split(r"\s[-—]\s", line)[0].strip()
        line = line.rstrip(".,;:")
        if line:
            paths.append(line)
    return paths


def normalize_retrieved_path(path: str, corpus_dir: Optional[Path]) -> str:
    path = path.replace("\\", "/")
    path = re.sub(r"^\.?/+", "", path)
    if corpus_dir is not None:
        prefix = str(corpus_dir).replace("\\", "/").rstrip("/") + "/"
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path


def compute_ndcg_at_k(retrieved: List[str], gold_set: set[str], k: int) -> float:
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


def compute_bright_ndcg(final_text: str, row: Dict[str, Any], corpus_dir: Optional[Path], k: int = 10) -> float:
    gold_docs = row.get("gold_docs") or row.get("gold_ids") or []
    gold_set = {normalize_retrieved_path(g, corpus_dir) for g in gold_docs}
    retrieved = [normalize_retrieved_path(p, corpus_dir) for p in parse_retrieved_docs(final_text)]
    return compute_ndcg_at_k(retrieved, gold_set, k)


def build_agent_config(args: argparse.Namespace, query_dir: Path, query_id: str) -> AgentConfig:
    config = AgentConfig()
    config.provider = args.provider
    config.model = args.model
    config.base_url = args.base_url
    config.api_key = args.api_key
    config.temperature = args.temperature if args.temperature is not None else 0.0
    config.max_output_tokens = args.max_output_tokens
    config.thinking_level = args.thinking_level

    config.max_turns = args.max_turns
    config.force_answer_turns = args.force_answer_turns
    config.keep_tool_calls = args.keep_tool_calls
    config.compaction_threshold = args.compaction_threshold
    config.use_usage_based_compaction = args.use_usage_based_compaction
    config.usage_based_compaction_min_tokens = args.usage_based_compaction_min_tokens
    config.usage_based_compaction_max_tokens = args.usage_based_compaction_max_tokens

    config.corpus_dir = str(args.corpus_dir)
    config.scope_dir = str(query_dir)
    config.output_dir = str(query_dir)

    config.model_server_script = os.environ.get("MODEL_SERVER_SCRIPT", str(REPO_ROOT / "scripts" / "model_server_bright.py"))
    config.index_dir = args.index_dir
    config.embed_model_path = args.embed_model_path or os.environ.get("EMBED_MODEL_PATH", config.embed_model_path)
    config.qr_model_path = args.qr_model_path or os.environ.get("QR_MODEL_PATH", config.qr_model_path)
    config.qr_heads = args.qr_heads or os.environ.get("QR_HEADS", config.qr_heads)
    config.device = args.device
    config.embed_top_k = args.embed_top_k
    config.no_reranker = args.no_reranker

    config.rg_max_matches = int(os.environ.get("RG_MAX_MATCHES", config.rg_max_matches))
    config.grep_max_line_length = int(os.environ.get("GREP_MAX_LINE_LENGTH", config.grep_max_line_length))
    config.paragraph_rerank_model = str(os.environ.get("PARAGRAPH_RERANK_MODEL", config.paragraph_rerank_model)).strip().lower()
    if config.paragraph_rerank_model not in {"qr", "emb", "none"}:
        config.paragraph_rerank_model = "qr"
    config.paragraph_rerank_doc_limit = int(os.environ.get("PARAGRAPH_RERANK_DOC_LIMIT", config.paragraph_rerank_doc_limit))
    config.rg_rerank_match = str(os.environ.get("RG_RERANK_MATCH", "")).strip().lower() in {"1", "true", "yes", "on"}
    config.rg_rerank_model = str(os.environ.get("RG_RERANK_MODEL", config.rg_rerank_model)).strip().lower()
    if config.rg_rerank_model not in {"qr", "emb", "qwen3emb"}:
        config.rg_rerank_model = "qr"
    config.use_contextual_bash_tool = str(os.environ.get("USE_CONTEXTUAL_BASH_TOOL", "")).strip().lower() in {"1", "true", "yes", "on"}
    config.rg_rerank_candidate_num = int(os.environ.get("RG_RERANK_CANDIDATE_NUM", config.rg_rerank_candidate_num))
    config.rg_rerank_timeout = int(os.environ.get("RG_RERANK_TIMEOUT", config.rg_rerank_timeout))
    config.read_max_lines = int(os.environ.get("READ_MAX_LINES", config.read_max_lines))
    config.read_max_bytes = int(os.environ.get("READ_MAX_BYTES", config.read_max_bytes))

    config.scope_size_prefix = os.environ.get("SCOPE_SIZE_PREFIX", "")
    config.sample_id = query_id
    config.system_prompt_file = args.system_prompt_file
    config.append_system_prompt_file = args.append_system_prompt_file
    config.force_answer_prompt_file = args.force_answer_prompt_file
    config.verbose = args.verbose

    config.bright_subset = args.subset or ""
    config.query_instruction = args.query_instruction or TASK_QUERY_INSTRUCTIONS.get(config.bright_subset, "")
    config.max_retrieved_docs = max(1, int(args.max_retrieved_docs))
    return config


def run_single_query(row: Dict[str, Any], args: argparse.Namespace, model_client: ModelServerClient) -> Dict[str, Any]:
    query_id = str(row.get("query_id", row.get("id", "")))
    question = row.get("query", row.get("question", ""))
    query_dir = args.output_root / query_id

    if args.skip_completed and (query_dir / "final.txt").exists():
        final = (query_dir / "final.txt").read_text(encoding="utf-8").strip()
        logger.info(f"[{query_id}] Skipping (already completed)")
        return {"query_id": query_id, "final_text": final, "skipped": True}

    query_dir.mkdir(parents=True, exist_ok=True)
    with (query_dir / "item.json").open("w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

    config = build_agent_config(args, query_dir, query_id)
    config.question = build_bright_prompt(question)

    os.environ["SAMPLE_ID"] = query_id
    if config.query_instruction:
        os.environ["QUERY_INSTRUCTION"] = config.query_instruction

    is_resume = (query_dir / "state.json").exists() and not (query_dir / "final.txt").exists()
    llm_client = LLMClient(config)
    tools = build_tools(config, model_client)

    if is_resume:
        session = AgentSession.resume_from_output_dir(config, llm_client, tools, model_client, query_dir)
        session.system_prompt = session.system_prompt
    else:
        session = AgentSession(config, llm_client, tools, model_client)

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        final_answer = session.run(config.question)
    except Exception as e:
        logger.error(f"[{query_id}] Agent failed: {e}", exc_info=True)
        final_answer = f"[Agent error: {e}]"
    finished_at = datetime.now(timezone.utc).isoformat()

    save_outputs(config, session, final_answer)
    ndcg_at_10 = compute_bright_ndcg(final_answer, row, args.corpus_dir, k=10)

    result = {
        "query_id": query_id,
        "question": question,
        "final_text": final_answer,
        "turn_count": session.turn_count,
        "started_at": started_at,
        "finished_at": finished_at,
        "usage": llm_client.usage.to_dict(),
        "ndcg_at_10": ndcg_at_10,
        "gold_ids": row.get("gold_ids", []),
        "skipped": False,
    }
    with (query_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    _suppress_noisy_third_party_debug_logs()

    dataset = read_jsonl(args.dataset)
    if args.limit:
        dataset = dataset[:args.limit]
    logger.info(f"Loaded {len(dataset)} questions from {args.dataset}")

    if args.max_concurrency != 1:
        logger.warning("TS-Mirror BRIGHT eval runner currently executes sequentially; ignoring max_concurrency != 1")

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_dict = {
        "dataset": str(args.dataset),
        "output_root": str(args.output_root),
        "corpus_dir": str(args.corpus_dir),
        "subset": args.subset,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "thinking_level": args.thinking_level,
        "max_turns": args.max_turns,
        "force_answer_turns": args.force_answer_turns,
        "max_output_tokens": args.max_output_tokens,
        "keep_tool_calls": args.keep_tool_calls,
        "compaction_threshold": args.compaction_threshold,
        "embed_top_k": args.embed_top_k,
        "no_reranker": args.no_reranker,
        "query_instruction": args.query_instruction or TASK_QUERY_INSTRUCTIONS.get(args.subset, ""),
        "paragraph_rerank_model": os.environ.get("PARAGRAPH_RERANK_MODEL", "qr"),
        "rg_rerank_match": os.environ.get("RG_RERANK_MATCH", ""),
        "rg_rerank_model": os.environ.get("RG_RERANK_MODEL", "qr"),
        "total_questions": len(dataset),
    }
    with (args.output_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    shared_config = AgentConfig()
    shared_config.model_server_script = os.environ.get("MODEL_SERVER_SCRIPT", str(REPO_ROOT / "scripts" / "model_server_bright.py"))
    shared_config.index_dir = args.index_dir
    shared_config.embed_model_path = args.embed_model_path or os.environ.get("EMBED_MODEL_PATH", shared_config.embed_model_path)
    shared_config.qr_model_path = args.qr_model_path or os.environ.get("QR_MODEL_PATH", shared_config.qr_model_path)
    shared_config.qr_heads = args.qr_heads or os.environ.get("QR_HEADS", shared_config.qr_heads)
    shared_config.device = args.device
    shared_config.corpus_dir = str(args.corpus_dir)
    shared_config.no_reranker = args.no_reranker
    shared_config.bright_subset = args.subset or ""
    shared_config.query_instruction = args.query_instruction or TASK_QUERY_INSTRUCTIONS.get(shared_config.bright_subset, "")

    if shared_config.query_instruction:
        os.environ["QUERY_INSTRUCTION"] = shared_config.query_instruction

    logger.info("Starting shared BRIGHT model server...")
    model_client = ModelServerClient(shared_config)
    model_client.ensure_started()
    logger.info("Model server ready")

    results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()
    try:
        for i, row in enumerate(dataset):
            query_id = str(row.get("query_id", row.get("id", i)))
            logger.info(f"=== Query {i + 1}/{len(dataset)} [{query_id}] ===")
            try:
                result = run_single_query(row, args, model_client)
                results.append(result)
            except Exception as e:
                logger.error(f"[{query_id}] Fatal error: {e}", exc_info=True)
                results.append({"query_id": query_id, "error": str(e), "skipped": False})

            with (args.output_root / "results.jsonl").open("w", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
    finally:
        model_client.shutdown()

    total_elapsed = time.perf_counter() - total_start
    completed = [r for r in results if not r.get("skipped") and not r.get("error")]
    skipped = sum(1 for r in results if r.get("skipped"))
    errors = sum(1 for r in results if r.get("error"))
    avg_turns = sum(r.get("turn_count", 0) for r in completed) / len(completed) if completed else 0.0
    avg_ndcg = sum(float(r.get("ndcg_at_10", 0.0)) for r in completed) / len(completed) if completed else 0.0

    summary = {
        "total_questions": len(dataset),
        "completed": len(completed),
        "skipped": skipped,
        "errors": errors,
        "total_time_seconds": round(total_elapsed, 1),
        "avg_turns": round(avg_turns, 1),
        "ndcg_at_10": round(avg_ndcg, 4) if completed else None,
    }
    with (args.output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=== BRIGHT Evaluation Complete ===")
    logger.info(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
