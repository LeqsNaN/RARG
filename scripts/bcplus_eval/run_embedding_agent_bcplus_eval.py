#!/usr/bin/env python3
"""
Run the adapted BC+ embedding agent inside DCI-Agent-Lite with OpenAI-compatible API / retry / resume / cost tracking.

Key goals of this runner:
  - keep the official BC+ search-only prompt semantics
  - reuse our corpus / FAISS indices / embedding model
  - use OpenAI-compatible chat.completions
  - add retry, real-time trace printing, and ts-mirror-style artifact persistence

Per-query artifacts:
  output_root/<query_id>/
    item.json
    question.txt
    input_question.txt
    log.txt
    events.jsonl
    state.json
    conversation.json
    usage.json
    latest_model_context.json
    search_results/*.json
    final.txt
    result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from embedding_backends import DEFAULT_QUERY_INSTRUCTION, DEFAULT_QWEN3_EMBED_MODEL
from scripts.embedding_search_agent_searcher import EmbeddingSearchAgentSearcher

logger = logging.getLogger(__name__)


DEFAULT_QUERY_TEMPLATE_NO_GET_DOCUMENT = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}""".strip()

FORCE_ANSWER_PROMPT = """You have reached the turn limit for tool use. Stop calling tools now and produce your best final answer using only the evidence already gathered.

Respond exactly in this format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{confidence score between 0% and 100%}}
""".strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_ms() -> int:
    return int(time.time() * 1000)


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def safe_write_json(path: Path, payload: Any) -> None:
    safe_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text_blocks(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    return [{"type": "text", "text": text}]


def parse_tool_arguments(raw_args: Any) -> Any:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return raw_args
    return raw_args


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += estimate_tokens(str(block.get("text", "")))
        for tc in msg.get("tool_calls", []):
            total += estimate_tokens(tc.get("function", {}).get("arguments", ""))
    return total


def tool_name_from_msg(msg: Dict[str, Any]) -> str:
    return str(msg.get("tool_name") or msg.get("name") or "")


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    role = msg.get("role")
    if role == "user":
        return estimate_tokens(msg.get("content", "")) + 8
    if role == "assistant":
        usage = msg.get("usage") or {}
        content_tokens = usage.get("output")
        reasoning_tokens = usage.get("reasoningTokens", 0)
        if isinstance(content_tokens, int):
            return max(0, content_tokens - reasoning_tokens) + 12
        content_len = len(msg.get("content", ""))
        tool_calls = msg.get("tool_calls") or []
        args_len = sum(len(tc.get("function", {}).get("arguments", "")) for tc in tool_calls)
        return (content_len + args_len) // 4 + 12 + len(tool_calls) * 8
    if role == "tool":
        return estimate_tokens(msg.get("content", "")) + 15
    return 0


def compacted_tool_tokens(msg: Dict[str, Any]) -> int:
    content = msg.get("content", "")
    if tool_name_from_msg(msg) == "embed_recall":
        compacted_content = re.sub(r"<qr_paragraphs>[\s\S]*?</qr_paragraphs>", "", content).strip()
        compacted_content = compacted_content or "[cleared]"
    else:
        compacted_content = "[cleared]"
    return estimate_tokens(compacted_content) + 15


def assistant_input_tokens(messages: List[Dict[str, Any]], assistant_idx: int) -> Optional[int]:
    msg = messages[assistant_idx]
    usage = msg.get("usage") or {}
    value = usage.get("input")
    if isinstance(value, int):
        return value
    return None


def assistant_content_tokens(messages: List[Dict[str, Any]], assistant_idx: int) -> int:
    return estimate_message_tokens(messages[assistant_idx])


def estimate_compacted_prefix_tokens(messages: List[Dict[str, Any]], boundary_assistant_idx: int) -> Optional[int]:
    assistant_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "assistant"]
    if not assistant_indices:
        return None
    first_input_tokens = assistant_input_tokens(messages, assistant_indices[0])
    if first_input_tokens is None:
        return None
    total = first_input_tokens
    for idx, msg in enumerate(messages[:boundary_assistant_idx]):
        role = msg.get("role")
        if role == "assistant":
            total += assistant_content_tokens(messages, idx)
        elif role == "tool":
            total += compacted_tool_tokens(msg)
    return total


def find_turn_start(messages: List[Dict[str, Any]], tool_msg_idx: int, tool_call_id: str) -> int:
    assistant_idx = None
    for i in range(tool_msg_idx - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("id") == tool_call_id:
                    assistant_idx = i
                    break
            if assistant_idx is not None:
                break
    if assistant_idx is None:
        return tool_msg_idx
    return assistant_idx + 1


def compute_boundary(messages: List[Dict[str, Any]], keep_n: int) -> int:
    tool_msg_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "tool"]
    if len(tool_msg_indices) <= keep_n:
        return 0
    boundary_idx = tool_msg_indices[-keep_n]
    boundary_tool_call_id = messages[boundary_idx].get("tool_call_id", "")
    return find_turn_start(messages, boundary_idx, boundary_tool_call_id)


def apply_compaction(messages: List[Dict[str, Any]], boundary: int) -> List[Dict[str, Any]]:
    compacted = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool" or i >= boundary:
            compacted.append(deepcopy(msg))
            continue
        compacted_msg = deepcopy(msg)
        content = compacted_msg.get("content", "")
        if tool_name_from_msg(compacted_msg) == "embed_recall":
            compacted_content = re.sub(r"<qr_paragraphs>[\s\S]*?</qr_paragraphs>", "", content).strip()
            compacted_msg["content"] = compacted_content or "[cleared]"
        else:
            compacted_msg["content"] = "[cleared]"
        compacted.append(compacted_msg)
    return compacted


def estimate_pending_tool_tokens(messages: List[Dict[str, Any]]) -> int:
    tool_result_overhead = 15
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx < 0:
        return 0
    total_chars = 0
    n_tool_results = 0
    for i in range(last_assistant_idx + 1, len(messages)):
        if messages[i].get("role") == "tool":
            total_chars += len(messages[i].get("content", ""))
            n_tool_results += 1
    return total_chars // 4 + n_tool_results * tool_result_overhead


def extract_exact_answer(text: str) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        if line.strip().lower().startswith("exact answer:"):
            return line.split(":", 1)[1].strip()
    return text.strip()


def read_prompt_file(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8").strip()


def build_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt_file:
        base = read_prompt_file(args.system_prompt_file)
    elif args.append_system_prompt_file:
        return read_prompt_file(args.append_system_prompt_file)
    else:
        base = ""

    if args.append_system_prompt_file:
        extra = read_prompt_file(args.append_system_prompt_file)
        if base and extra:
            return base + "\n\n" + extra
        return base or extra
    return base


def load_query_template(args: argparse.Namespace) -> str:
    if args.query_template_file:
        template_path = Path(args.query_template_file)
        if template_path.is_file():
            return template_path.read_text(encoding="utf-8").strip()
    return DEFAULT_QUERY_TEMPLATE_NO_GET_DOCUMENT


def build_user_prompt(question: str, query_template: str) -> str:
    placeholder = "__QUESTION_PLACEHOLDER__"
    safe_template = query_template.replace("{Question}", placeholder)
    safe_template = safe_template.replace("{", "{{").replace("}", "}}")
    safe_template = safe_template.replace(placeholder, "{Question}")
    return safe_template.format(Question=question)


def quote_for_log(text: str, max_len: int = 120) -> str:
    text = str(text).replace("\n", "\\n")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def build_search_tool_definition(searcher: EmbeddingSearchAgentSearcher, top_k: int) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": searcher.search_description(top_k),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    }
                },
                "required": ["query"],
            },
        },
    }


def assistant_message_to_dict(msg: Any, *, usage: Optional[Dict[str, Any]] = None, stop_reason: str = "stop") -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "role": "assistant",
        "content": msg.content or "",
        "timestamp": timestamp_ms(),
        "stopReason": stop_reason,
    }
    if usage:
        payload["usage"] = usage
    if msg.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return payload


def sanitize_messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            sanitized.append({"role": "system", "content": msg.get("content", "")})
        elif role == "user":
            sanitized.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if msg.get("content") is not None:
                assistant_msg["content"] = msg.get("content", "")
            if msg.get("tool_calls"):
                assistant_msg["tool_calls"] = msg["tool_calls"]
            sanitized.append(assistant_msg)
        elif role == "tool":
            tool_msg = {
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            sanitized.append(tool_msg)
    return sanitized


@dataclass
class UsageStats:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    total_latency_s: float = 0.0

    def record(self, usage: Any, latency_s: float) -> Dict[str, int]:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        cached_tokens = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        reasoning_tokens = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

        self.input_tokens += prompt_tokens
        self.cached_input_tokens += cached_tokens
        self.output_tokens += completion_tokens
        self.reasoning_tokens += reasoning_tokens
        self.total_tokens += total_tokens
        self.request_count += 1
        self.total_latency_s += latency_s

        return {
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_write_input_tokens": max(0, prompt_tokens - cached_tokens),
            "total_tokens": total_tokens,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": max(0, self.input_tokens - self.cached_input_tokens),
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "total_latency_s": round(self.total_latency_s, 2),
        }


class QueryRun:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        row: Dict[str, Any],
        searcher: EmbeddingSearchAgentSearcher,
        client: OpenAI,
        query_template: str,
        system_prompt: str,
    ):
        self.args = args
        self.row = row
        self.searcher = searcher
        self.client = client
        self.query_template = query_template
        self.system_prompt = system_prompt

        self.query_id = str(row.get("query_id", row.get("id", "")))
        self.question = str(row.get("query", row.get("question", "")))
        self.gold_answer = str(row.get("answer", ""))
        self.query_dir = args.output_root / self.query_id
        self.log_path = self.query_dir / "log.txt"
        self.events_path = self.query_dir / "events.jsonl"
        self.search_dir = self.query_dir / "search_results"
        self.aggregate_log_path = args.output_root / "agent_progress.log"

        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.turn_count: int = 0
        self.tool_call_count: int = 0
        self.search_queries: List[str] = []
        self.retrieved_docids: List[str] = []
        self.messages: List[Dict[str, Any]] = []
        self.usage = UsageStats()
        self.latest_error: Optional[str] = None
        self.status: str = "running"
        self.stop_reason: str = ""
        self.resumed: bool = False
        self._last_input_tokens: int = 0
        self._last_content_tokens: int = 0
        self._last_usage_this_call: Dict[str, int] = {
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cache_write_input_tokens": 0,
            "total_tokens": 0,
        }
        self._compact_boundary: Optional[int] = None
        self._keep_tool_calls_override: Optional[int] = None

    def _append_line(self, path: Path, line: str, *, spacer: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            if spacer:
                f.write("\n")
            f.write(line)
            f.write("\n")

    def log(self, message: str, *, indent: int = 0, spacer: bool = False) -> None:
        prefix = f"[sample {self.query_id}]"
        line = f"{prefix} {message}"
        if indent > 0:
            line = (" " * indent) + line
        self._append_line(self.log_path, line, spacer=spacer)
        self._append_line(self.aggregate_log_path, line, spacer=spacer)
        if spacer:
            print(file=sys.stderr, flush=True)
        print(line, file=sys.stderr, flush=True)

    def append_event(self, event: Dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": utc_now(), **event}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")

    def _message_to_external(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = msg.get("role")
        if role == "system":
            return {"role": "system", "content": text_blocks(msg.get("content", ""))}
        if role == "user":
            return {
                "role": "user",
                "content": text_blocks(msg.get("content", "")),
                "timestamp": msg.get("timestamp", timestamp_ms()),
            }
        if role == "assistant":
            content_blocks: List[Dict[str, Any]] = []
            if msg.get("content"):
                content_blocks.extend(text_blocks(msg.get("content", "")))
            for tc in msg.get("tool_calls", []):
                content_blocks.append(
                    {
                        "type": "toolCall",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": parse_tool_arguments(tc.get("function", {}).get("arguments", {})),
                    }
                )
            payload = {
                "role": "assistant",
                "content": content_blocks,
                "api": "openai-completions",
                "provider": self.args.provider,
                "model": self.args.model,
                "usage": msg.get("usage", {}),
                "stopReason": msg.get("stopReason", "toolUse" if msg.get("tool_calls") else "stop"),
                "timestamp": msg.get("timestamp", timestamp_ms()),
            }
            return payload
        if role == "tool":
            return {
                "role": "toolResult",
                "toolCallId": msg.get("tool_call_id", ""),
                "toolName": msg.get("tool_name", "search"),
                "content": text_blocks(msg.get("content", "")),
                "isError": bool(msg.get("is_error", False)),
                "timestamp": msg.get("timestamp", timestamp_ms()),
                "tool_execution": msg.get("tool_execution", {}),
            }
        return dict(msg)

    def conversation_payload(self, *, status: str, messages_override: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        source_messages = messages_override if messages_override is not None else self.messages
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": text_blocks(self.system_prompt),
                    "sources": {
                        "system_prompt_file": self.args.system_prompt_file or None,
                        "append_system_prompt_file": self.args.append_system_prompt_file or None,
                    },
                }
            )
        messages.extend(self._message_to_external(msg) for msg in source_messages)
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": status,
            "question": self.question,
            "cwd": str(self.args.corpus_dir),
            "provider": self.args.provider,
            "model": self.args.model,
            "tools": "search",
            "max_turns": self.args.max_iterations,
            "system_prompt_file": self.args.system_prompt_file or None,
            "append_system_prompt_file": self.args.append_system_prompt_file or None,
            "conversation_features": {
                "clear_tool_results": False,
                "clear_tool_results_keep_last": self.args.keep_recent_tool_results,
                "externalize_tool_results": False,
                "strip_thinking": False,
                "strip_usage": False,
            },
            "keep_session": True,
            "turn_count": self.turn_count,
            "event_count": self.tool_call_count,
            "assistant_text": extract_exact_answer(self.latest_final_text()),
            "messages": messages,
        }

    def latest_final_text(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and not msg.get("tool_calls"):
                return msg.get("content", "")
        return ""

    def internal_state(self, *, status: str) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": status,
            "stop_reason": self.stop_reason,
            "latest_error": self.latest_error,
            "turn_count": self.turn_count,
            "tool_call_count": self.tool_call_count,
            "search_queries": self.search_queries,
            "retrieved_docids": self.retrieved_docids,
            "messages": self.messages,
            "usage": self.usage.to_dict(),
            "query_template_file": self.args.query_template_file or None,
            "system_prompt_file": self.args.system_prompt_file or None,
            "append_system_prompt_file": self.args.append_system_prompt_file or None,
            "resumed": self.resumed,
            "last_input_tokens": self._last_input_tokens,
            "last_content_tokens": self._last_content_tokens,
            "compact_boundary": self._compact_boundary,
            "keep_tool_calls_override": self._keep_tool_calls_override,
        }

    def save_incremental(self, *, status: str, conversation_messages_override: Optional[List[Dict[str, Any]]] = None) -> None:
        safe_write_json(self.query_dir / "state.json", self.internal_state(status=status))
        safe_write_json(self.query_dir / "conversation.json", self.conversation_payload(status=status, messages_override=conversation_messages_override))
        safe_write_json(self.query_dir / "usage.json", self.usage.to_dict())
        safe_write_json(
            self.query_dir / "latest_model_context.json",
            {
                "captured_at": utc_now(),
                "turn": self.turn_count,
                "message_count": len(self.messages),
                "messages": sanitize_messages_for_api(self.messages),
            },
        )

    def load_existing_state_if_any(self) -> None:
        state_path = self.query_dir / "state.json"
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        messages = state.get("messages")
        if not isinstance(messages, list) or not messages:
            return

        self.messages = messages
        self.started_at = state.get("started_at") or utc_now()
        self.finished_at = state.get("finished_at")
        self.turn_count = int(state.get("turn_count", 0))
        self.tool_call_count = int(state.get("tool_call_count", 0))
        self.search_queries = list(state.get("search_queries", []))
        self.retrieved_docids = list(state.get("retrieved_docids", []))
        usage = state.get("usage", {})
        self.usage = UsageStats(
            input_tokens=int(usage.get("input_tokens", 0)),
            cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            request_count=int(usage.get("request_count", 0)),
            total_latency_s=float(usage.get("total_latency_s", 0.0)),
        )
        self.resumed = True
        self._last_input_tokens = int(state.get("last_input_tokens", 0))
        self._last_content_tokens = int(state.get("last_content_tokens", 0))
        self._compact_boundary = state.get("compact_boundary")
        self._keep_tool_calls_override = state.get("keep_tool_calls_override")

    def is_completed(self) -> bool:
        result_path = self.query_dir / "result.json"
        final_path = self.query_dir / "final.txt"
        if not result_path.exists() or not final_path.exists():
            return False
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        final_text = final_path.read_text(encoding="utf-8").strip()
        return payload.get("status") == "completed" and bool(final_text)

    def _chat_with_retry(self, request_messages: List[Dict[str, Any]], *, use_tools: bool = True) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.args.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.args.model,
                    "messages": request_messages,
                    "max_tokens": self.args.max_output_tokens,
                }
                if use_tools:
                    kwargs["tools"] = [build_search_tool_definition(self.searcher, self.args.search_top_k)]
                    kwargs["tool_choice"] = "auto"
                if self.args.temperature is not None:
                    kwargs["temperature"] = self.args.temperature
                if self.args.thinking_level and self.args.thinking_level != "none":
                    kwargs["extra_body"] = {
                        "reasoning_effort": self.args.thinking_level,
                        "reasoning": {
                            "effort": self.args.thinking_level,
                            "summary": "auto",
                        },
                    }
                start_time = time.perf_counter()
                response = self.client.chat.completions.create(**kwargs)
                latency_s = time.perf_counter() - start_time
                usage_this_call = self.usage.record(response.usage, latency_s)
                self._last_usage_this_call = dict(usage_this_call)
                self._last_input_tokens = int(usage_this_call.get("prompt_tokens", 0))
                self._last_content_tokens = max(0, int(usage_this_call.get("completion_tokens", 0)) - int(usage_this_call.get("reasoning_tokens", 0)))
                self.append_event(
                    {
                        "type": "llm_response",
                        "turn": self.turn_count + 1,
                        "latency_s": round(latency_s, 2),
                        "usage": usage_this_call,
                    }
                )
                self.log(
                    f"[turn {self.turn_count + 1}] llm in={usage_this_call['prompt_tokens']} "
                    f"(cached={usage_this_call['cached_prompt_tokens']}) "
                    f"out={usage_this_call['completion_tokens']} "
                    f"(thinking={usage_this_call['reasoning_tokens']}) "
                    f"cum_total={self.usage.total_tokens}",
                    indent=2,
                )
                return response
            except Exception as e:
                last_error = e
                err = str(e)
                self.latest_error = err
                self.append_event(
                    {
                        "type": "llm_retry",
                        "turn": self.turn_count + 1,
                        "attempt": attempt,
                        "max_retries": self.args.max_retries,
                        "error": err,
                    }
                )
                err_lower = err.lower()
                if "context_length_exceeded" in err_lower or (("context" in err_lower and "window" in err_lower) or ("tokens" in err_lower and "exceed" in err_lower)):
                    raise RuntimeError(f"CONTEXT_LENGTH_EXCEEDED: {err}")
                if attempt >= self.args.max_retries:
                    break
                delay = min(self.args.retry_delay_max, self.args.retry_delay_base * (2 ** (attempt - 1)))
                self.log(
                    f"[turn {self.turn_count + 1}] API retry {attempt}/{self.args.max_retries}: {quote_for_log(err)} -> sleep {delay:.1f}s",
                    indent=2,
                )
                time.sleep(delay)
        raise RuntimeError(f"LLM request failed after {self.args.max_retries} attempts: {last_error}")

    def _register_docids(self, results: List[Dict[str, Any]]) -> None:
        seen = set(self.retrieved_docids)
        for item in results:
            docid = str(item.get("docid", ""))
            if docid and docid not in seen:
                self.retrieved_docids.append(docid)
                seen.add(docid)

    def _write_search_artifact(
        self,
        *,
        turn_index: int,
        call_index: int,
        query: str,
        results: List[Dict[str, Any]],
        elapsed_s: float,
    ) -> None:
        payload = {
            "query_id": self.query_id,
            "turn": turn_index,
            "call_index": call_index,
            "query": query,
            "elapsed_s": round(elapsed_s, 3),
            "result_count": len(results),
            "results": results,
        }
        safe_write_json(self.search_dir / f"turn_{turn_index:03d}_call_{call_index:03d}.json", payload)

    def _execute_search_tool(self, *, tool_call_id: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(tool_args.get("query", "")).strip()
        self.tool_call_count += 1
        self.search_queries.append(query)
        start = time.perf_counter()
        results = self.searcher.search(query, k=self.args.search_top_k)
        elapsed_s = time.perf_counter() - start
        self._register_docids(results)

        top_docids = ", ".join(str(item.get("docid", "")) for item in results[:3])
        self.log(
            f"[turn {self.turn_count}] search(query=\"{quote_for_log(query, 80)}\") -> {len(results)} docs in {elapsed_s:.2f}s"
            + (f" | top={top_docids}" if top_docids else ""),
            indent=2,
        )
        self.append_event(
            {
                "type": "tool_result",
                "turn": self.turn_count,
                "tool_name": "search",
                "tool_call_id": tool_call_id,
                "query": query,
                "elapsed_s": round(elapsed_s, 3),
                "result_count": len(results),
                "top_docids": [item.get("docid", "") for item in results[:5]],
            }
        )
        self._write_search_artifact(
            turn_index=self.turn_count,
            call_index=self.tool_call_count,
            query=query,
            results=results,
            elapsed_s=elapsed_s,
        )
        return {
            "tool_message": {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "tool_name": "search",
                "content": json.dumps(results, ensure_ascii=False, indent=2),
                "timestamp": timestamp_ms(),
                "tool_execution": {
                    "query": query,
                    "elapsed_s": round(elapsed_s, 3),
                    "result_count": len(results),
                },
            },
            "results": results,
        }

    def initialize(self) -> None:
        self.query_dir.mkdir(parents=True, exist_ok=True)
        safe_write_json(self.query_dir / "item.json", self.row)
        safe_write_text(self.query_dir / "question.txt", self.question)
        safe_write_text(self.query_dir / "input_question.txt", build_user_prompt(self.question, self.query_template))

        self.load_existing_state_if_any()
        if self.started_at is None:
            self.started_at = utc_now()

        if not self.messages:
            if self.system_prompt:
                self.messages.append({"role": "system", "content": self.system_prompt, "timestamp": timestamp_ms()})
            self.messages.append(
                {
                    "role": "user",
                    "content": build_user_prompt(self.question, self.query_template),
                    "timestamp": timestamp_ms(),
                }
            )
            self.append_event({"type": "query_started", "query_id": self.query_id, "question": self.question})
        else:
            self.append_event({"type": "query_resumed", "query_id": self.query_id, "turn_count": self.turn_count})

    def _compute_boundary(self, keep_n: int) -> int:
        return compute_boundary(self.messages, keep_n)

    def _apply_compaction(self, boundary: int) -> List[Dict[str, Any]]:
        return apply_compaction(self.messages, boundary)

    def _compact_messages(self) -> List[Dict[str, Any]]:
        base_keep_n = self.args.keep_recent_tool_results
        keep_n = self._keep_tool_calls_override or base_keep_n
        pending_tool_tokens = estimate_pending_tool_tokens(self.messages)
        estimated_next_context = self._last_input_tokens + self._last_content_tokens + pending_tool_tokens

        if self._compact_boundary is None:
            if estimated_next_context < self.args.max_context_tokens:
                return list(self.messages)
            self._keep_tool_calls_override = None
            boundary = self._compute_boundary(base_keep_n)
            self._compact_boundary = boundary
            self.log(
                f"[threshold-compaction] Compaction activated: estimated context {estimated_next_context} > {self.args.max_context_tokens}, boundary={boundary}",
                indent=2,
            )
        elif estimated_next_context >= self.args.max_context_tokens:
            new_boundary = self._compute_boundary(base_keep_n)
            if new_boundary > (self._compact_boundary or 0):
                self._compact_boundary = new_boundary
                self.log(f"[threshold-compaction] Compaction boundary updated to msg[{self._compact_boundary}]", indent=2)

        return self._apply_compaction(self._compact_boundary) if self._compact_boundary is not None else list(self.messages)

    def force_answer(self) -> Optional[str]:
        if self.args.force_answer_at_limit <= 0:
            return None
        prompt = FORCE_ANSWER_PROMPT
        assistant_prompt = {"role": "user", "content": prompt, "timestamp": timestamp_ms()}
        self.messages.append(assistant_prompt)
        request_messages = self._compact_messages()
        response = self._chat_with_retry(sanitize_messages_for_api(request_messages), use_tools=False)
        choice = response.choices[0]
        final_msg = assistant_message_to_dict(choice.message, usage=self.usage.to_dict(), stop_reason="stop")
        self.messages.pop()  # remove transient user force-answer prompt from saved history
        self.messages.append(assistant_prompt)
        self.messages.append(final_msg)
        self.stop_reason = "force_answer_at_limit"
        self.append_event({"type": "force_answer", "turn": self.turn_count, "content_preview": quote_for_log(final_msg.get("content", ""), 160)})
        self.save_incremental(status="running", conversation_messages_override=self.messages)
        return final_msg.get("content", "") or ""

    def finalize(self, *, status: str, final_text: str, error: Optional[str] = None) -> Dict[str, Any]:
        self.status = status
        self.finished_at = utc_now()
        self.latest_error = error
        exact_answer = extract_exact_answer(final_text)

        if final_text:
            safe_write_text(self.query_dir / "final.txt", final_text)

        result = {
            "query_id": self.query_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "final_text": final_text,
            "exact_answer": exact_answer,
            "status": status,
            "stop_reason": self.stop_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "turn_count": self.turn_count,
            "tool_call_count": self.tool_call_count,
            "search_count": self.tool_call_count,
            "search_queries": self.search_queries,
            "retrieved_docids": self.retrieved_docids,
            "usage": self.usage.to_dict(),
            "error": error,
            "resumed": self.resumed,
            "last_input_tokens": self._last_input_tokens,
            "last_content_tokens": self._last_content_tokens,
            "compact_boundary": self._compact_boundary,
            "keep_tool_calls_override": self._keep_tool_calls_override,
        }
        safe_write_json(self.query_dir / "result.json", result)
        self.save_incremental(status=status, conversation_messages_override=self.messages)
        return result

    def run(self) -> Dict[str, Any]:
        self.initialize()
        if self.resumed:
            self.log(f"Resuming from turn {self.turn_count}, search_calls={self.tool_call_count}", spacer=True)
        else:
            self.log(f"Starting: {quote_for_log(self.question, 160)}", spacer=True)
        self.save_incremental(status="running", conversation_messages_override=self.messages)

        try:
            while self.turn_count < self.args.max_iterations:
                full_messages_snapshot = list(self.messages)
                request_messages = self._compact_messages()
                response = self._chat_with_retry(sanitize_messages_for_api(request_messages))
                choice = response.choices[0]
                assistant = choice.message
                assistant_msg = assistant_message_to_dict(
                    assistant,
                    usage={
                        "input": int(self._last_input_tokens),
                        "output": int(self._last_usage_this_call.get("completion_tokens", 0)),
                        "reasoningTokens": int(self._last_usage_this_call.get("reasoning_tokens", 0)),
                        "cacheRead": int(self._last_usage_this_call.get("cached_prompt_tokens", 0)),
                        "cacheWrite": int(self._last_usage_this_call.get("cache_write_input_tokens", 0)),
                        "totalTokens": int(self._last_usage_this_call.get("total_tokens", 0)),
                    },
                    stop_reason="toolUse" if assistant.tool_calls else "stop",
                )
                self.messages.append(assistant_msg)
                self.turn_count += 1
                self.save_incremental(status="running", conversation_messages_override=full_messages_snapshot)

                if assistant.tool_calls:
                    self.log(
                        f"[turn {self.turn_count}] assistant requested {len(assistant.tool_calls)} tool call(s)",
                        indent=2,
                    )
                    self.append_event(
                        {
                            "type": "assistant_tool_call",
                            "turn": self.turn_count,
                            "tool_count": len(assistant.tool_calls),
                            "content_preview": quote_for_log(assistant.content or "", 160),
                        }
                    )
                else:
                    final_text = assistant.content or ""
                    self.stop_reason = "no_tool_call"
                    self.append_event(
                        {
                            "type": "query_completed",
                            "turn": self.turn_count,
                            "stop_reason": self.stop_reason,
                            "final_preview": quote_for_log(final_text, 200),
                        }
                    )
                    self.log(
                        f"Finished in {self.turn_count} turns, search_calls={self.tool_call_count}, answer=\"{quote_for_log(extract_exact_answer(final_text), 120)}\"",
                        spacer=False,
                    )
                    return self.finalize(status="completed", final_text=final_text)

                for tool_call in assistant.tool_calls or []:
                    tool_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    try:
                        tool_args = json.loads(raw_args)
                    except Exception:
                        tool_args = {"query": raw_args}
                    self.log(
                        f"[turn {self.turn_count}] tool search(query=\"{quote_for_log(str(tool_args.get('query', '')), 80)}\")",
                        indent=4,
                    )
                    self.append_event(
                        {
                            "type": "tool_call",
                            "turn": self.turn_count,
                            "tool_name": tool_name,
                            "tool_call_id": tool_call.id,
                            "arguments": tool_args,
                        }
                    )
                    try:
                        execution = self._execute_search_tool(
                            tool_call_id=tool_call.id,
                            tool_args=tool_args,
                        )
                        self.messages.append(execution["tool_message"])
                    except Exception as e:
                        error_msg = f"Error executing search: {e}"
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "tool_name": tool_name,
                                "content": error_msg,
                                "is_error": True,
                                "timestamp": timestamp_ms(),
                            }
                        )
                        self.append_event(
                            {
                                "type": "tool_error",
                                "turn": self.turn_count,
                                "tool_name": tool_name,
                                "tool_call_id": tool_call.id,
                                "error": str(e),
                            }
                        )
                        self.log(f"[turn {self.turn_count}] tool error: {quote_for_log(str(e), 160)}", indent=4)
                    self.save_incremental(status="running", conversation_messages_override=full_messages_snapshot)

            self.stop_reason = "max_iterations"
            final_text = ""
            if self.args.force_answer_at_limit > 0:
                final_text = self.force_answer() or ""
            self.append_event(
                {
                    "type": "query_completed",
                    "turn": self.turn_count,
                    "stop_reason": self.stop_reason,
                    "final_preview": quote_for_log(final_text, 200),
                }
            )
            self.log(
                f"Stopped at max_iterations={self.args.max_iterations}, search_calls={self.tool_call_count}",
                spacer=False,
            )
            return self.finalize(status="completed", final_text=final_text)
        except Exception as e:
            err = str(e)
            self.stop_reason = self.stop_reason or "error"
            self.append_event(
                {
                    "type": "query_failed",
                    "turn": self.turn_count,
                    "error": err,
                }
            )
            self.log(f"FAILED: {quote_for_log(err, 200)}")
            return self.finalize(status="failed", final_text=self.latest_final_text(), error=err)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure embedding search agent on BrowseComp-Plus")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSONL")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--corpus-dir", type=Path, required=True, help="Corpus directory")
    parser.add_argument("--index-dir", type=str, required=True, help="Embedding index directory")

    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--thinking-level", type=str, default="medium")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--force-answer-at-limit", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=230000)
    parser.add_argument("--keep-recent-tool-results", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay-base", type=float, default=2.0)
    parser.add_argument("--retry-delay-max", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--embed-model-path", type=str, default=DEFAULT_QWEN3_EMBED_MODEL)
    parser.add_argument("--embed-model-type", type=str, default="auto")
    parser.add_argument("--embed-backend", type=str, default="auto")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--embed-max-model-len", type=int, default=512)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--query-instruction", type=str, default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--embed-empty-cache-after-encode", action="store_true")

    parser.add_argument("--query-template-file", type=str, default="")
    parser.add_argument("--system-prompt-file", type=str, default="")
    parser.add_argument("--append-system-prompt-file", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    query_template = load_query_template(args)
    system_prompt = build_system_prompt(args)

    searcher = EmbeddingSearchAgentSearcher(
        index_dir=args.index_dir,
        corpus_dir=str(args.corpus_dir),
        model_path=args.embed_model_path,
        model_type=args.embed_model_type,
        backend=args.embed_backend,
        device=args.device or None,
        max_model_len=args.embed_max_model_len,
        snippet_max_tokens=args.snippet_max_tokens,
        encode_batch_size=args.embed_batch_size,
        query_instruction=args.query_instruction,
        empty_cache_after_encode=args.embed_empty_cache_after_encode,
    )
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    config = {
        "started_at": utc_now(),
        "dataset": str(args.dataset),
        "output_root": str(args.output_root),
        "corpus_dir": str(args.corpus_dir),
        "index_dir": args.index_dir,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "thinking_level": args.thinking_level,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "max_iterations": args.max_iterations,
        "force_answer_at_limit": args.force_answer_at_limit,
        "max_context_tokens": args.max_context_tokens,
        "keep_recent_tool_results": args.keep_recent_tool_results,
        "max_retries": args.max_retries,
        "retry_delay_base": args.retry_delay_base,
        "retry_delay_max": args.retry_delay_max,
        "search_top_k": args.search_top_k,
        "snippet_max_tokens": args.snippet_max_tokens,
        "embed_model_path": args.embed_model_path,
        "embed_model_type": args.embed_model_type,
        "embed_backend": args.embed_backend,
        "device": args.device,
        "embed_max_model_len": args.embed_max_model_len,
        "embed_batch_size": args.embed_batch_size,
        "query_instruction": args.query_instruction,
        "query_template_file": args.query_template_file or None,
        "system_prompt_file": args.system_prompt_file or None,
        "append_system_prompt_file": args.append_system_prompt_file or None,
        "question_count": len(rows),
    }
    safe_write_json(args.output_root / "config.json", config)

    logger.info("=== Embedding Search Agent Evaluation ===")
    logger.info(f"  questions={len(rows)}")
    logger.info(f"  dataset={args.dataset}")
    logger.info(f"  model={args.model}")
    logger.info(f"  base_url={args.base_url}")
    logger.info(f"  output_root={args.output_root}")

    results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()
    for idx, row in enumerate(rows):
        query_id = str(row.get("query_id", row.get("id", idx)))
        logger.info(f"=== Query {idx + 1}/{len(rows)} [{query_id}] ===")
        run = QueryRun(
            args=args,
            row=row,
            searcher=searcher,
            client=client,
            query_template=query_template,
            system_prompt=system_prompt,
        )
        if run.is_completed():
            logger.info(f"[{query_id}] Skipping completed query")
            result = json.loads((run.query_dir / "result.json").read_text(encoding="utf-8"))
        else:
            result = run.run()
        results.append(result)
        with (args.output_root / "results.jsonl").open("w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_elapsed = time.perf_counter() - total_start
    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") == "failed"]
    avg_turns = sum(int(r.get("turn_count", 0)) for r in completed) / len(completed) if completed else 0.0
    avg_search = sum(int(r.get("search_count", 0)) for r in completed) / len(completed) if completed else 0.0

    summary = {
        "finished_at": utc_now(),
        "total_questions": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "avg_turns": round(avg_turns, 3),
        "avg_search_calls": round(avg_search, 3),
        "total_time_seconds": round(total_elapsed, 1),
    }
    safe_write_json(args.output_root / "summary.json", summary)

    logger.info("=== Evaluation Complete ===")
    logger.info(f"  completed={len(completed)} failed={len(failed)}")
    logger.info(f"  avg_turns={avg_turns:.2f} avg_search_calls={avg_search:.2f}")
    logger.info(f"  total_time_seconds={total_elapsed:.1f}")


if __name__ == "__main__":
    main()
