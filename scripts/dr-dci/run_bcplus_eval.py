#!/usr/bin/env python3
"""Independent Python port of DR-DCI's main BrowseComp-Plus harness path.

This deliberately ports the original hardlink dynamic-pull experiment contract:
empty per-query views, root/root_flat_disclosed/rank_aware settings, 300-turn
budget, level3 artifacts, resumability, result metrics, and incremental output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from adapters import OpenAICompletionsAdapter, Qwen3EmbeddingAdapter
from agent import Config, Session

LOG = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Python DR-DCI BrowseComp-Plus main-path harness")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, help="Accepted for original DR-DCI launcher compatibility; Python port has no Node package runtime.")
    parser.add_argument("--agent-dir", type=Path, help="Accepted for original DR-DCI launcher compatibility; Python port has no Pi agent-dir dependency.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--view-cache-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query-id", action="append", default=[], help="Run only these exact query_id values. May be repeated.")
    parser.add_argument("--max-concurrency", type=int, default=1)
    # Keep the original main-launcher command line accepted verbatim. Values
    # are fixed by that launcher but are validated below rather than ignored.
    parser.add_argument("--tools", default="read,bash,pull")
    parser.add_argument("--max-turns-mode", choices=("abort", "hard"), default="abort")
    parser.add_argument("--submit-now-trigger-ratio", type=float, default=0.0)
    parser.add_argument("--submit-now-turns-remaining", type=int, default=0)
    parser.add_argument("--submit-now-min-turns-remaining", type=int, default=0)
    parser.add_argument("--runtime-context-level", default="level3")
    parser.add_argument("--level3-persistent-micro-compact", action="store_true", help="Non-DR-DCI extension: persist level3 clears, then compact again only after new tool output reaches 240k chars.")
    parser.add_argument("--pull-view-mode", default="hardlink")
    parser.add_argument("--pull-base-url", default="", help="Original retriever URL; Qwen3 adapter uses the configured model_server instead.")
    parser.add_argument("--pull-layout", default="root")
    parser.add_argument("--pull-prompt-mode", default="rank_aware")
    parser.add_argument("--pull-materialization-mode", default="root_flat_disclosed")
    parser.add_argument("--pull-min-top-k", type=int, default=300)
    parser.add_argument("--pull-max-top-k", type=int, default=600)
    parser.add_argument("--pull-max-queries", type=int, default=1)
    parser.add_argument("--pull-preview-mode", default="ranked")
    parser.add_argument("--pull-preview-limit", type=int, default=20)
    parser.add_argument("--full-corpus-doc-count", type=int, default=100_195)
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--judge-timeout-seconds", type=int, default=120)
    parser.add_argument("--agent-input-price-per-1m", type=float, default=0.0, help="Cost for non-cached input tokens; 0 records tokens without estimating cost.")
    parser.add_argument("--agent-cache-read-price-per-1m", type=float, default=0.0, help="Cost for cached input tokens.")
    parser.add_argument("--agent-cache-write-price-per-1m", type=float, default=0.0, help="Cost for non-cached prompt tokens, named cacheWrite to match ts_mirror_agent artifacts.")
    parser.add_argument("--agent-output-price-per-1m", type=float, default=0.0, help="Cost for all completion tokens, including reasoning tokens when the provider bills them as output.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--thinking-level", default="medium")
    parser.add_argument("--pi-thinking-level", default="", help="Original launcher spelling; mapped to --thinking-level when supplied.")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-output-tokens", type=int, default=128000)
    parser.add_argument("--embed-model-path", default=Config.embed_model_path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pull-calls", type=int, default=0)
    parser.add_argument("--skip-judge", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def view_dir(args: argparse.Namespace, query_dir: Path) -> Path:
    if args.view_cache_root:
        return args.view_cache_root / args.output_root.name / "_pull_views" / query_dir.name
    return query_dir.parent / "_pull_views" / query_dir.name


def pull_meta_dir(args: argparse.Namespace, query_dir: Path) -> Path:
    workspace = view_dir(args, query_dir)
    return workspace.parent.parent / "_pull_meta" / workspace.name


def validate_main_path(args: argparse.Namespace) -> None:
    expected = {
        "tools": "read,bash,pull", "max_turns_mode": "abort", "runtime_context_level": "level3",
        "pull_view_mode": "hardlink", "pull_layout": "root", "pull_prompt_mode": "rank_aware",
        "pull_materialization_mode": "root_flat_disclosed", "pull_min_top_k": 300,
        "pull_max_top_k": 600, "pull_max_queries": 1, "pull_preview_mode": "ranked",
    }
    mismatches = [f"{key}={getattr(args, key)!r} (required {value!r})" for key, value in expected.items() if getattr(args, key) != value]
    if args.submit_now_trigger_ratio != 0 or args.submit_now_turns_remaining != 0 or args.submit_now_min_turns_remaining != 0:
        mismatches.append("main path requires all submit-now triggers to be 0")
    if mismatches:
        raise ValueError("This runner is the strict BCplus main-path port; unsupported configuration:\n" + "\n".join(mismatches))


def core_artifacts_exist(query_dir: Path) -> bool:
    return any((query_dir / name).exists() for name in ("state.json", "events.jsonl", "conversation.json", "latest_model_context.json", "final.txt", "question.txt"))


def archive_unfinished(query_dir: Path) -> Path:
    """Preserve a partial run for diagnosis, but never resume it.

    The lean artifact format intentionally does not reconstruct a live agent
    session from events.  Starting a sample anew also avoids reusing a partly
    materialized root-flat pull view.
    """
    if not query_dir.exists(): return query_dir
    status = str(read_json(query_dir / "state.json").get("status") or "incomplete")
    label = "failed" if status == "failed" else "incomplete"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = query_dir.with_name(f"{query_dir.name}.{label}_{stamp}")
    suffix = 1
    while archived.exists():
        archived = query_dir.with_name(f"{query_dir.name}.{label}_{stamp}_{suffix}")
        suffix += 1
    shutil.move(str(query_dir), str(archived))
    return archived


def is_failed(query_dir: Path) -> bool:
    state = read_json(query_dir / "state.json")
    return bool(state.get("error") or state.get("status") == "failed")


def is_agent_completed(query_dir: Path) -> bool:
    """True only when no live-session recovery is needed.

    This lets a process interrupted after a final answer proceed to judging and
    aggregation without paying for a second model run. It is not a resume of a
    partial conversation.
    """
    return read_json(query_dir / "state.json").get("status") == "completed" and (query_dir / "final.txt").is_file()


def seconds_between(started: Any, finished: Any) -> Optional[float]:
    if not isinstance(started, str) or not isinstance(finished, str): return None
    try: return (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    except ValueError: return None


def normalize_retrieved_path(path: str, corpus_dir: Path) -> str:
    value = path.strip().strip("`").replace("\\", "/")
    value = re.sub(r"^\{corpus\}/", "", value)
    prefix = str(corpus_dir).replace("\\", "/").rstrip("/") + "/"
    if value.startswith(prefix): return value[len(prefix):]
    corpus_name_prefix = corpus_dir.name.rstrip("/") + "/"
    if value.startswith(corpus_name_prefix): return value[len(corpus_name_prefix):]
    return re.sub(r"^\.?/+", "", value)


def read_string_list(path: Optional[str]) -> List[str]:
    if not path: return []
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return [item for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError): return []


def tool_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    totals: Dict[str, Dict[str, float]] = {}
    for call in state.get("tool_calls", []):
        if call.get("event") != "tool_execution_end": continue
        name = str(call.get("toolName") or "unknown")
        stats = totals.setdefault(name, {"call_count": 0.0, "error_count": 0.0, "duration_seconds": 0.0})
        stats["call_count"] += 1
        stats["error_count"] += float(bool(call.get("isError")))
        measured = seconds_between(call.get("started_at") or call.get("recorded_at"), call.get("finished_at") or call.get("recorded_at"))
        stats["duration_seconds"] += measured if measured is not None else float(call.get("duration_seconds") or 0.0)
    count = int(sum(row["call_count"] for row in totals.values()))
    return {"call_count": count, "error_count": int(sum(row["error_count"] for row in totals.values())), "duration_seconds": sum(row["duration_seconds"] for row in totals.values()), "duration_measured_call_count": count, "duration_missing_call_count": 0, "by_tool": totals}


def pull_metrics(state: Dict[str, Any], row: Dict[str, Any], corpus_dir: Path, full_corpus_doc_count: int) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = []
    candidates: set[str] = set()
    for call in state.get("tool_calls", []):
        if call.get("event") != "tool_execution_end" or call.get("toolName") != "pull": continue
        details = (call.get("result") or {}).get("details") or {}
        managed = read_string_list(details.get("managedPathsPath"))
        normalized = [normalize_retrieved_path(path, corpus_dir) for path in managed]
        candidates.update(normalized)
        calls.append({
            "queries": details.get("queries", []), "top_k": details.get("topK"), "view_dir": details.get("viewDir"),
            "managed_paths_path": details.get("managedPathsPath"), "materialized_created_count": details.get("materializedDocumentCount", 0),
            "materialized_missing_count": details.get("missingDocumentCount", 0), "already_visible_count": details.get("alreadyVisibleDocumentCount", 0),
            "candidate_count": len(normalized), "per_query_hit_counts": details.get("perQueryHitCounts", {}),
            "is_error": bool(call.get("isError")), "duration_seconds": call.get("duration_seconds", 0),
        })
    gold = {normalize_retrieved_path(str(path), corpus_dir) for path in (row.get("gold_docs") or [])}
    overlap = candidates & gold
    precision = len(overlap) / len(candidates) if candidates else 0.0
    recall = len(overlap) / len(gold) if gold else 0.0
    return {"call_count": len(calls), "error_count": sum(item["is_error"] for item in calls), "duration_seconds": sum(float(item["duration_seconds"] or 0) for item in calls), "total_query_count": sum(len(item["queries"]) for item in calls), "total_per_query_hits": sum(sum((item["per_query_hit_counts"] or {}).values()) for item in calls), "unique_candidate_count": len(candidates), "total_materialized_created_count": sum(int(item["materialized_created_count"] or 0) for item in calls), "total_materialized_missing_count": sum(int(item["materialized_missing_count"] or 0) for item in calls), "gold_precision": precision, "gold_recall": recall, "gold_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "corpus_reduction_ratio": len(candidates) / full_corpus_doc_count if full_corpus_doc_count else None, "calls": calls}


def usage_cost_estimate(usage: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Price provider-reported buckets without guessing a provider tariff.

    ``cache_write_tokens`` is the ts_mirror_agent-compatible name for prompt
    tokens that were not reported as cache reads.  Most APIs charge this bucket
    at the normal input rate: pass the same price to both corresponding flags
    when that is the provider's billing model.
    """
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cache_read_tokens = int(usage.get("cache_read_tokens", 0) or 0)
    cache_write_tokens = int(usage.get("cache_write_tokens", max(0, input_tokens - cache_read_tokens)) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    prices = {
        "input_price_per_1m": float(args.agent_input_price_per_1m),
        "cache_read_price_per_1m": float(args.agent_cache_read_price_per_1m),
        "cache_write_price_per_1m": float(args.agent_cache_write_price_per_1m),
        "output_price_per_1m": float(args.agent_output_price_per_1m),
    }
    # When an explicit cache-write price is supplied it owns all non-cached
    # prompt tokens. Otherwise use the ordinary input price for that bucket.
    cache_write_price = prices["cache_write_price_per_1m"] or prices["input_price_per_1m"]
    cache_read_cost = cache_read_tokens / 1_000_000 * prices["cache_read_price_per_1m"]
    cache_write_cost = cache_write_tokens / 1_000_000 * cache_write_price
    output_cost = output_tokens / 1_000_000 * prices["output_price_per_1m"]
    total_cost = cache_read_cost + cache_write_cost + output_cost
    return {
        "pricing": prices,
        "cache_read_cost": round(cache_read_cost, 12),
        "cache_write_cost": round(cache_write_cost, 12),
        "output_cost": round(output_cost, 12),
        "total_cost": round(total_cost, 12),
    }


def result_for(row: Dict[str, Any], query_dir: Path, workspace: Path, corpus_dir: Path, full_corpus_doc_count: int, args: argparse.Namespace, llm_usage: Optional[Dict[str, Any]] = None, judge_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = read_json(query_dir / "state.json")
    final = (query_dir / "final.txt").read_text(encoding="utf-8") if (query_dir / "final.txt").is_file() else state.get("assistant_text", "")
    agent_usage = llm_usage or read_json(query_dir / "usage.json")
    return {
        "query_id": str(row["query_id"]), "question": row.get("query"), "gold_answer": row.get("answer"),
        "final_text": final, "query_dir": str(query_dir), "workspace_dir": str(workspace), "run_status": state.get("status"),
        "run_error": state.get("error"), "turn_count": state.get("turn_count"), "event_count": state.get("event_count"),
        "agent_started_at": state.get("started_at"), "agent_finished_at": state.get("finished_at"), "wall_time_seconds": seconds_between(state.get("started_at"), state.get("finished_at")),
        "tool_metrics": tool_metrics(state), "pull_metrics": pull_metrics(state, row, corpus_dir, full_corpus_doc_count), "agent_usage": agent_usage,
        "agent_cost_estimate": usage_cost_estimate(agent_usage, args),
        "judge_result": judge_result, "is_correct": None if judge_result is None else judge_result.get("is_correct"),
        "runtime_context_management": {
            "level": "level3", "maxToolResultChars": 20_000, "microCompactKeepTurns": 12,
            "microCompactMode": "persistent" if args.level3_persistent_micro_compact else "request_time",
            "persistentMicroCompactEnabled": bool(args.level3_persistent_micro_compact),
        },
    }


async def run_one(args: argparse.Namespace, row: Dict[str, Any], retriever: Qwen3EmbeddingAdapter) -> Dict[str, Any]:
    sample_id = str(row["query_id"])
    query_dir = args.output_root / sample_id
    workspace = view_dir(args, query_dir)
    existing = read_json(query_dir / "result.json")
    if existing.get("run_status") == "completed" and (query_dir / "final.txt").is_file():
        return existing
    agent_completed = is_agent_completed(query_dir)
    if not agent_completed:
        if query_dir.exists() and core_artifacts_exist(query_dir):
            archive_unfinished(query_dir)
        elif query_dir.exists():
            shutil.rmtree(query_dir)
        # A non-completed sample always receives a fresh pull view and pull
        # metadata. Otherwise a fresh root-flat view would inherit the old
        # managed-path set and incorrectly treat documents as already visible.
        stale_pull_meta = pull_meta_dir(args, query_dir)
        if stale_pull_meta.exists():
            archive_unfinished(stale_pull_meta)
        if workspace.exists(): shutil.rmtree(workspace)
    query_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    write_json(query_dir / "item.json", row)
    question = str(row.get("query") or "")
    config = Config(question=question, hidden_corpus_dir=str(args.corpus_dir), workspace_dir=str(workspace), output_dir=str(query_dir), index_dir=args.index_dir, provider=args.provider, model=args.model, base_url=args.base_url, api_key=args.api_key, thinking_level=args.thinking_level, max_turns=args.max_turns, max_output_tokens=args.max_output_tokens, embed_model_path=args.embed_model_path, device=args.device, model_server_script=str(Path(__file__).resolve().parents[1] / "model_server.py"), pull_meta_dir=str(pull_meta_dir(args, query_dir)), pull_preview_limit=args.pull_preview_limit, max_pull_calls=args.max_pull_calls, sample_id=sample_id, level3_persistent_micro_compact=args.level3_persistent_micro_compact)
    (query_dir / "input_question.txt").write_text(question, encoding="utf-8")
    llm = OpenAICompletionsAdapter(base_url=config.base_url, api_key=config.api_key, model=config.model, thinking_level=config.thinking_level, max_output_tokens=config.max_output_tokens)
    judge_result: Optional[Dict[str, Any]] = None
    if not agent_completed:
        session = Session(config, llm, retriever)
        try:
            await asyncio.to_thread(session.run)
        except Exception as exc:
            LOG.exception("[%s] failed", sample_id)
            state = read_json(query_dir / "state.json")
            write_json(query_dir / "state.json", {**state, "status": "failed", "error": str(exc), "finished_at": utc_now()})
            (query_dir / "worker_exception.txt").write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
    if not args.skip_judge and not is_failed(query_dir):
        try:
            judge_client = OpenAICompletionsAdapter(base_url=args.base_url, api_key=args.api_key, model=args.judge_model, thinking_level="low", max_output_tokens=180)
            judge_result = await asyncio.to_thread(judge_client.judge, question=question, gold_answer=str(row.get("answer") or ""), predicted_answer=(query_dir / "final.txt").read_text(encoding="utf-8"))
            write_json(query_dir / "eval_result.json", judge_result)
        except Exception as exc:
            judge_result = {"error": str(exc), "is_correct": False}
            write_json(query_dir / "eval_result.json", judge_result)
    result = result_for(row, query_dir, workspace, args.corpus_dir, args.full_corpus_doc_count, args, None if agent_completed else llm.usage.as_dict(), judge_result)
    write_json(query_dir / "result.json", result)
    return result


def summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [row for row in results if row.get("run_status") == "completed"]
    token_keys = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens", "content_output_tokens", "total_tokens", "request_count")
    usage_totals = {key: sum(float((row.get("agent_usage") or {}).get(key) or 0) for row in completed) for key in token_keys}
    cost_totals = {
        key: sum(float((row.get("agent_cost_estimate") or {}).get(key) or 0) for row in completed)
        for key in ("cache_read_cost", "cache_write_cost", "output_cost", "total_cost")
    }
    return {
        "counts": {"total": len(results), "completed": len(completed), "failed": len(results) - len(completed), "judged": sum(row.get("is_correct") is not None for row in results), "correct": sum(bool(row.get("is_correct")) for row in results)},
        "accuracy": {"over_total": sum(bool(row.get("is_correct")) for row in results) / len(results) if results else 0.0},
        "averages": {
            "turn_count": sum(float(row.get("turn_count") or 0) for row in completed) / len(completed) if completed else 0.0,
            "tool_call_count": sum(float((row.get("tool_metrics") or {}).get("call_count") or 0) for row in completed) / len(completed) if completed else 0.0,
            "pull_call_count": sum(float((row.get("pull_metrics") or {}).get("call_count") or 0) for row in completed) / len(completed) if completed else 0.0,
        },
        "agent_usage": {"totals": usage_totals, "averages_over_completed": {key: value / len(completed) if completed else 0.0 for key, value in usage_totals.items()}},
        "agent_cost_estimate": {"totals": cost_totals, "averages_over_completed": {key: value / len(completed) if completed else 0.0 for key, value in cost_totals.items()}},
        "main_path": {"tools": "read,bash,pull", "pull_layout": "root", "pull_prompt_mode": "rank_aware", "pull_materialization_mode": "root_flat_disclosed", "pull_min_top_k": 300, "pull_max_top_k": 600, "pull_max_queries": 1, "runtime_context_level": "level3"},
        "finished_at": utc_now(),
    }


async def main_async() -> None:
    args = parse_args()
    if args.pi_thinking_level:
        args.thinking_level = args.pi_thinking_level
    validate_main_path(args)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.query_id:
        requested_ids = {str(value) for value in args.query_id}
        rows = [row for row in rows if str(row.get("query_id")) in requested_ids]
        found_ids = {str(row["query_id"]) for row in rows}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(f"query_id not found in {args.dataset}: {', '.join(missing_ids)}")
    if args.limit: rows = rows[:args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Exact main launcher defaults from run_full830_dynamic_pull_...sh.
    os.environ.setdefault("DCI_BASH_DEFAULT_TIMEOUT_SECONDS", "30")
    os.environ.setdefault("DCI_BASH_MAX_LINE_CHARS", "2000")
    os.environ.setdefault("DCI_BASH_LONG_MATCH_SNIPPET_CHARS", "1500")
    os.environ.setdefault("DCI_REFLOW_SINGLE_LINE_TEXT", "1")
    os.environ.setdefault("DCI_REFLOW_SINGLE_LINE_WIDTH", "1200")
    os.environ.setdefault("DCI_WRAP_LONG_TEXT_LINES", "0")
    write_json(args.output_root / "config.json", {
        "started_at": utc_now(), "dataset": str(args.dataset), "output_root": str(args.output_root),
        "corpus_dir": str(args.corpus_dir), "package_dir": str(args.package_dir) if args.package_dir else None,
        "agent_dir": str(args.agent_dir) if args.agent_dir else None, "provider": args.provider, "model": args.model,
        "tools": args.tools, "max_turns": args.max_turns, "max_turns_mode": args.max_turns_mode,
        "submit_now_turns_remaining": args.submit_now_turns_remaining, "submit_now_trigger_ratio": args.submit_now_trigger_ratio,
        "submit_now_min_turns_remaining": args.submit_now_min_turns_remaining, "runtime_context_level": args.runtime_context_level,
        "level3_persistent_micro_compact": args.level3_persistent_micro_compact,
        "pi_thinking_level": args.pi_thinking_level, "thinking_level": args.thinking_level, "max_concurrency": args.max_concurrency,
        "limit": args.limit, "query_id": list(args.query_id), "judge_model": args.judge_model, "judge_timeout_seconds": args.judge_timeout_seconds,
        "agent_input_price_per_1m": args.agent_input_price_per_1m,
        "agent_cache_read_price_per_1m": args.agent_cache_read_price_per_1m,
        "agent_cache_write_price_per_1m": args.agent_cache_write_price_per_1m,
        "agent_output_price_per_1m": args.agent_output_price_per_1m,
        "pull_view_mode": args.pull_view_mode, "pull_base_url": args.pull_base_url or None, "pull_layout": args.pull_layout,
        "pull_prompt_mode": args.pull_prompt_mode, "pull_materialization_mode": args.pull_materialization_mode,
        "pull_min_top_k": args.pull_min_top_k, "pull_max_top_k": args.pull_max_top_k,
        "pull_max_queries": args.pull_max_queries, "pull_preview_mode": args.pull_preview_mode,
        "pull_preview_limit": args.pull_preview_limit, "view_cache_root": str(args.view_cache_root) if args.view_cache_root else None,
        "full_corpus_doc_count": args.full_corpus_doc_count, "question_count": len(rows),
        "adapter_differences": {"llm": "OpenAI-compatible Chat Completions API", "retriever": "local Qwen3 model_server JSONL adapter"},
    })
    retriever = Qwen3EmbeddingAdapter(model_server_script=str(Path(__file__).resolve().parents[1] / "model_server.py"), index_dir=args.index_dir, embed_model_path=args.embed_model_path, corpus_dir=str(args.corpus_dir), device=args.device)
    results: Dict[str, Dict[str, Any]] = {}
    result_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))
    async def worker(index: int, row: Dict[str, Any]) -> None:
        async with semaphore:
            result = await run_one(args, row, retriever)
        async with result_lock:
            results[str(row["query_id"])] = result
            ordered = [results[str(item["query_id"])] for item in rows if str(item["query_id"]) in results]
            with (args.output_root / "results.jsonl").open("w", encoding="utf-8") as f:
                for item in ordered: f.write(json.dumps(item, ensure_ascii=False) + "\n")
            write_json(args.output_root / "summary.json", summary(ordered))
            LOG.info("[%d/%d] qid=%s status=%s turns=%s", len(results), len(rows), result["query_id"], result.get("run_status"), result.get("turn_count"))
    try:
        await asyncio.gather(*(worker(index, row) for index, row in enumerate(rows, start=1)))
    finally:
        retriever.close()


if __name__ == "__main__": asyncio.run(main_async())
