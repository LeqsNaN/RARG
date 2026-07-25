#!/usr/bin/env python3
"""
TS-Mirror Python Agent — strict port of the TypeScript pi-mono agent.

Key design choices that mirror TS:
  - Tools: embed_recall, bash (with rg auto-detection), read
  - grep is NOT a separate tool — the model writes rg commands in bash
  - Compaction: simple [cleared] for old tool results, embed_recall scope preserved
  - No QR reranking on grep results
  - System prompt uses embed_recall_native.txt style (cat scope | xargs rg)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import AgentConfig, build_config_from_args
from llm_client import LLMClient
from model_client import ModelServerClient
from tools import BaseTool, build_tools

logger = logging.getLogger(__name__)


FORCE_ANSWER_PROMPT = """You have reached the turn limit for tool use. Stop calling tools now and produce your best final answer using only the evidence already gathered.

Respond exactly in this format:
Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer}
Confidence: {confidence score between 0% and 100%}
"""


# =============================================================================
# System Prompt — mirrors TS embed_recall_native.txt + system-prompt.ts
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """\
IMPORTANT CONSTRAINTS:
- Do NOT use `ls`, `find`, or any command that lists all filenames. The corpus has 100,000+ files across 30,000+ directories — listing them will timeout and waste all your turns.

EMBEDDING RECALL TOOL:
The corpus contains a large number of documents. The large search space puts significant pressure on rg — too many matches, excessive noise, and difficulty pinpointing relevant files.

To address this, an embed_recall tool is available. By providing a query, it semantically recalls a batch of relevant documents and saves their paths to a scope file (scope_1.txt, scope_2.txt, ...), one path per line. This narrows the search space to a manageable subset. It also returns a few reference paragraphs from top-ranked documents — these are a rough preview to help you identify search directions for `rg` search.

To search within recalled documents, pipe the scope file to rg:
  cat /path/to/scope_N.txt | xargs -d '\\n' rg [OPTIONS] "PATTERN"

Results are returned ordered by document embedding similarity to your query (most relevant first). Only search within one scope file at a time — do not concatenate multiple scope files.

NOTE:
- Initially call embed_recall with the ORIGINAL QUESTION to establish a scope file, then use rg to explore within it with diverse keywords, entities, and patterns.
- `rg` calls from different angles are essential.
- Only call embed_recall again if your scope is clearly missing the topic entirely — do NOT call it repeatedly hoping for better preview paragraphs, use rg instead.
- Do NOT call embed_recall multiple times without rg exploration in between.
- Do NOT run rg directly on the corpus without a scope file.

When you are confident to answer the question, stop tool calling and respond with the following format:
Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer}
Confidence: {confidence score between 0% and 100%}
"""


def build_system_prompt(config: AgentConfig) -> str:
    # Align with the pi-mono launch pattern used in DCI:
    # the runner typically passes only --append-system-prompt-file, and the
    # resulting system prompt in artifacts is just that file's content.
    if config.system_prompt_file and os.path.isfile(config.system_prompt_file):
        with open(config.system_prompt_file, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    elif config.append_system_prompt_file and os.path.isfile(config.append_system_prompt_file):
        with open(config.append_system_prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        base_prompt = DEFAULT_SYSTEM_PROMPT.strip()

    if config.append_system_prompt_file and os.path.isfile(config.append_system_prompt_file):
        with open(config.append_system_prompt_file, "r", encoding="utf-8") as f:
            extra_prompt = f.read().strip()
        return "\n\n".join([base_prompt, extra_prompt])

    return base_prompt


# =============================================================================
# Helpers
# =============================================================================


def assistant_msg_to_dict(msg: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"role": "assistant"}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def extract_final_answer(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            return msg.get("content", "")
    return ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_ms() -> int:
    return int(time.time() * 1000)


def _text_blocks(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    return [{"type": "text", "text": text}]


def _parse_tool_arguments(raw_args: Any) -> Any:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return raw_args
    return raw_args


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _safe_write_json(path: Path, payload: Any) -> None:
    _safe_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _quote_arg(value: Any) -> str:
    text = "" if value is None else str(value)
    escaped = json.dumps(text, ensure_ascii=False)[1:-1]
    return f'"{escaped}"'

def _format_tool_args(tool_name: str, args: Dict[str, Any]) -> str:
    if tool_name == "embed_recall":
        return f'query={_quote_arg(args.get("query", ""))}'
    elif tool_name == "bash":
        parts = [f'command={_quote_arg(args.get("command", ""))}']
        if args.get("timeout") is not None:
            parts.append(f'timeout={args.get("timeout")}')
        if args.get("rerank_query") is not None:
            parts.append(f'rerank_query={_quote_arg(args.get("rerank_query", ""))}')
        return ", ".join(parts)
    elif tool_name == "read":
        path = args.get("path", "")
        parts = [f'path=".../{os.path.basename(path)}"']
        if args.get("offset"):
            parts.append(f'offset={args["offset"]}')
        if args.get("limit"):
            parts.append(f'limit={args["limit"]}')
        return ", ".join(parts)
    else:
        return json.dumps(args, ensure_ascii=False)[:100]


def _result_snippet(result_text: str, max_len: int = 300) -> str:
    first_line = result_text.split("\n")[0]
    if len(first_line) > max_len:
        return first_line[:max_len] + "..."
    if len(first_line) < 80 and "\n" in result_text:
        lines = result_text.split("\n")
        snippet = "\n".join(lines[:3])
        if len(snippet) > max_len:
            return snippet[:max_len] + "..."
        return snippet
    return first_line


# =============================================================================
# Agent Session
# =============================================================================

class AgentSession:
    """Turn loop with boundary-based compaction and incremental artifacts."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        tools: List[BaseTool],
        model_client: ModelServerClient,
    ):
        self.config = config
        self.llm_client = llm_client
        self.tools = {t.name: t for t in tools}
        self.tools_list = tools
        self.model_client = model_client

        self.messages: List[Dict[str, Any]] = []
        self.system_prompt = build_system_prompt(config)
        self.turn_count = 0
        self.tool_calls_log: List[Dict[str, Any]] = []
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self._last_input_tokens: int = 0
        self._last_content_tokens: int = 0
        self._assistant_usage: Dict[int, Dict[str, int]] = {}
        self._compact_boundary: Optional[int] = None
        self._keep_tool_calls_override: Optional[int] = None
        self._overflow_retry_keep_n: Optional[int] = None
        self.resume_count: int = 0

        self._output_dir = Path(config.output_dir) if config.output_dir else None
        self._events_path = self._output_dir / "events.jsonl" if self._output_dir else None
        self._sample_log_path = self._output_dir / "log.txt" if self._output_dir else None
        self._run_log_path = self._output_dir.parent / "agent_progress.log" if self._output_dir else None
        self._sample_tag = config.sample_id or (self._output_dir.name if self._output_dir else "-")

    def _append_event(self, event: Dict[str, Any]) -> None:
        if not self._events_path:
            return
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False))
            f.write("\n")

    def _append_sample_log(self, line: str, *, spacer: bool = False) -> None:
        if not self._sample_log_path:
            return
        self._sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._sample_log_path.open("a", encoding="utf-8") as f:
            if spacer:
                f.write("\n")
            f.write(line)
            f.write("\n")

    def _append_run_log(self, line: str, *, spacer: bool = False) -> None:
        if not self._run_log_path:
            return
        self._run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._run_log_path.open("a", encoding="utf-8") as f:
            if spacer:
                f.write("\n")
            f.write(line)
            f.write("\n")

    def _print_progress(self, message: str, *, indent: int = 0, spacer: bool = False) -> None:
        prefix = f"[sample {self._sample_tag}]"
        line = f"{prefix} {message}"
        if indent > 0:
            line = (" " * indent) + line
        self._append_sample_log(line, spacer=spacer)
        self._append_run_log(line, spacer=spacer)
        if spacer:
            print(file=sys.stderr, flush=True)
        print(line, file=sys.stderr, flush=True)

    def _tool_name_from_msg(self, msg: Dict[str, Any]) -> str:
        return msg.get("tool_name") or self._find_tool_name_for_call_id(msg.get("tool_call_id", ""))

    def _user_external_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": _text_blocks(msg.get("content", "")),
            "timestamp": msg.get("timestamp", _timestamp_ms()),
        }

    def _assistant_external_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        content_blocks: List[Dict[str, Any]] = []
        if msg.get("content"):
            content_blocks.extend(_text_blocks(msg["content"]))
        for tc in msg.get("tool_calls", []):
            content_blocks.append(
                {
                    "type": "toolCall",
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": _parse_tool_arguments(tc.get("function", {}).get("arguments", {})),
                }
            )
        external = {
            "role": "assistant",
            "content": content_blocks,
            "api": "openai-completions",
            "provider": self.config.provider,
            "model": self.config.model,
            "usage": msg.get("usage", {}),
            "stopReason": msg.get("stopReason", "toolUse" if msg.get("tool_calls") else "stop"),
            "timestamp": msg.get("timestamp", _timestamp_ms()),
        }
        if msg.get("responseId"):
            external["responseId"] = msg["responseId"]
        return external

    def _tool_external_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        external = {
            "role": "toolResult",
            "toolCallId": msg.get("tool_call_id", ""),
            "toolName": self._tool_name_from_msg(msg),
            "content": _text_blocks(msg.get("content", "")),
            "isError": bool(msg.get("is_error", False)),
            "timestamp": msg.get("timestamp", _timestamp_ms()),
        }
        if msg.get("tool_execution"):
            external["tool_execution"] = msg["tool_execution"]
        return external

    def _message_to_external(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        role = msg.get("role")
        if role == "user":
            return self._user_external_message(msg)
        if role == "assistant":
            return self._assistant_external_message(msg)
        if role == "tool":
            return self._tool_external_message(msg)
        return deepcopy(msg)

    def _conversation_payload(self, *, status: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": _text_blocks(self.system_prompt),
                "sources": {
                    "system_prompt_file": self.config.system_prompt_file or None,
                    "append_system_prompt_file": self.config.append_system_prompt_file or None,
                },
            }
        ]
        messages.extend(self._message_to_external(msg) for msg in self.messages)
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": status,
            "question": self.config.question,
            "cwd": self.config.corpus_dir,
            "provider": self.config.provider,
            "model": self.config.model,
            "tools": ",".join(t.name for t in self.tools_list),
            "max_turns": self.config.max_turns,
            "system_prompt_file": self.config.system_prompt_file or None,
            "append_system_prompt_file": self.config.append_system_prompt_file or None,
            "conversation_features": {
                "clear_tool_results": False,
                "clear_tool_results_keep_last": 3,
                "externalize_tool_results": False,
                "strip_thinking": False,
                "strip_usage": False,
            },
            "keep_session": True,
            "resume_count": self.resume_count,
            "event_count": len(self.tool_calls_log),
            "turn_count": self.turn_count,
            "assistant_text": extract_final_answer(self.messages),
            "messages": messages,
        }

    def _save_latest_context(self, full_messages: List[Dict[str, Any]]) -> None:
        if not self._output_dir:
            return
        payload = {
            "captured_at": _utc_now_iso(),
            "turn": self.turn_count,
            "message_count": len(full_messages),
            "messages": full_messages,
        }
        try:
            _safe_write_json(self._output_dir / "latest_model_context.json", payload)
        except Exception as e:
            logger.debug(f"Failed to save latest_model_context: {e}")

    def _save_incremental(self, *, status: str) -> None:
        if not self._output_dir:
            return
        try:
            _safe_write_json(self._output_dir / "state.json", self.get_state(status=status))
            conversation_payload = self._conversation_payload(status=status)
            _safe_write_json(self._output_dir / "conversation.json", conversation_payload)
            _safe_write_json(self._output_dir / "conversation_full.json", conversation_payload)
            if self.config.question:
                _safe_write_text(self._output_dir / "question.txt", self.config.question)
                _safe_write_text(self._output_dir / "input_question.txt", self.config.question)
            _safe_write_json(self._output_dir / "usage.json", self.llm_client.usage.to_dict())
        except Exception as e:
            logger.debug(f"Failed to save incremental state: {e}")

    @classmethod
    def resume_from_output_dir(
        cls,
        config: AgentConfig,
        llm_client: "LLMClient",
        tools: List[BaseTool],
        model_client: "ModelServerClient",
        output_dir: Path,
    ) -> "AgentSession":
        session = cls(config, llm_client, tools, model_client)
        state_path = output_dir / "state.json"
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        session.messages = state.get("messages", [])
        session.tool_calls_log = state.get("tool_calls", [])
        session.turn_count = state.get("turn_count", 0)
        session.started_at = state.get("started_at")
        session.finished_at = state.get("finished_at")
        session._last_input_tokens = state.get("last_input_tokens", 0)
        session._last_content_tokens = state.get("last_content_tokens", 0)
        session._compact_boundary = state.get("compact_boundary")
        session._keep_tool_calls_override = state.get("keep_tool_calls_override")
        session._overflow_retry_keep_n = state.get("overflow_retry_keep_n")
        session._assistant_usage = {int(k): v for k, v in state.get("assistant_usage", {}).items()}
        session.resume_count = int(state.get("resume_count", 0)) + 1
        logger.info(f"Resumed from {state_path}: {len(session.messages)} messages, {session.turn_count} turns")
        return session

    def _activate_or_adjust_compaction(self, reason: str, *, overflow_retry: bool = False) -> None:
        # Compaction has two trigger sources with different policies:
        # 1. Threshold compaction: the estimated next context crosses the normal threshold.
        #    This always starts from the full configured keep_tool_calls and does not start
        #    a shrinking chain.
        # 2. Overflow compaction: the provider actually returns context_length_exceeded.
        #    This is only a retry-time safeguard. It starts from the full configured
        #    keep_tool_calls once, then shrinks to keep_tool_calls-10, -20, ... on repeated
        #    overflow retries in the SAME request loop.
        #
        # Boundary semantics:
        # - Once activated, _compact_boundary is persistent state.
        # - Every subsequent request keeps clearing tool results before that boundary.
        # - The boundary only moves forward when compaction is triggered again.
        base_keep_n = self.config.keep_tool_calls

        if overflow_retry:
            if self._overflow_retry_keep_n is None:
                self._overflow_retry_keep_n = base_keep_n
                self._keep_tool_calls_override = None
                if self._compact_boundary is None:
                    self._compact_boundary = self._compute_boundary(base_keep_n)
                logger.warning(
                    f"[overflow-compaction] {reason}: starting from keep_tool_calls={base_keep_n}, "
                    f"boundary={self._compact_boundary}"
                )
                return

            new_keep = max(5, self._overflow_retry_keep_n - 10)
            self._overflow_retry_keep_n = new_keep
            self._keep_tool_calls_override = new_keep
            if self._compact_boundary is None:
                self._compact_boundary = self._compute_boundary(new_keep)
            logger.warning(
                f"[overflow-compaction] {reason}: reducing keep_tool_calls to {new_keep}, "
                f"boundary={self._compact_boundary}"
            )
            return

        self._overflow_retry_keep_n = None
        self._keep_tool_calls_override = None
        boundary = self._compute_boundary(base_keep_n)
        if self._compact_boundary is None:
            self._compact_boundary = boundary
            logger.info(
                f"[threshold-compaction] {reason}: activating compaction with keep_tool_calls={base_keep_n}, boundary={boundary}"
            )
        elif boundary > self._compact_boundary:
            self._compact_boundary = boundary
            logger.info(f"[threshold-compaction] {reason}: compaction boundary updated to msg[{self._compact_boundary}]")

    def _compact_messages(self) -> List[Dict[str, Any]]:
        # Existing boundary means previously compacted history must stay cleared.
        # This method only decides whether the boundary should be created or advanced
        # on the current turn; it does NOT disable clearing when compaction is idle.
        base_keep_n = self.config.keep_tool_calls
        keep_n = self._keep_tool_calls_override or base_keep_n
        pending_tool_tokens = self._estimate_pending_tool_tokens()
        estimated_next_context = self._last_input_tokens + self._last_content_tokens + pending_tool_tokens

        if self.config.use_usage_based_compaction:
            # Usage-based mode has its own boundary selection, but the persistence rule
            # is the same: once boundary exists, keep clearing from it every turn.
            if self._compact_boundary is None:
                if estimated_next_context < self.config.compaction_threshold:
                    return list(self.messages)
                self._activate_or_adjust_compaction(
                    f"Usage-based compaction activated: estimated context {estimated_next_context}, "
                    f"max target <= {self.config.usage_based_compaction_max_tokens} tokens",
                    overflow_retry=False,
                )
                boundary = self._compute_usage_based_boundary(pending_tool_tokens) or self._compute_boundary(base_keep_n)
                self._compact_boundary = max(self._compact_boundary or 0, boundary)
            elif estimated_next_context >= self.config.compaction_threshold:
                boundary = self._compute_usage_based_boundary(pending_tool_tokens) or self._compute_boundary(base_keep_n)
                if boundary > self._compact_boundary:
                    self._compact_boundary = boundary
                    logger.info(
                        f"Usage-based compaction boundary updated to msg[{boundary}] "
                        f"with max target <= {self.config.usage_based_compaction_max_tokens} tokens"
                    )
            return self._apply_compaction(self._compact_boundary)

        if self._compact_boundary is None:
            # No boundary yet: regular compaction only starts once threshold is crossed.
            if estimated_next_context < self.config.compaction_threshold:
                return list(self.messages)
            self._activate_or_adjust_compaction(
                f"Compaction activated: estimated context {estimated_next_context} > {self.config.compaction_threshold}",
                overflow_retry=False,
            )
        elif estimated_next_context >= self.config.compaction_threshold:
            # Boundary already exists, so clearing continues every turn.
            # We only recompute / advance the boundary when threshold compaction is
            # triggered again, preserving the free space created by earlier compaction
            # so fresh tool outputs can accumulate.
            new_boundary = self._compute_boundary(base_keep_n)
            if new_boundary > self._compact_boundary:
                self._compact_boundary = new_boundary
                logger.info(f"Compaction boundary updated to msg[{self._compact_boundary}]")

        return self._apply_compaction(self._compact_boundary)

    def _compute_boundary(self, keep_n: int) -> int:
        tool_msg_indices = [i for i, msg in enumerate(self.messages) if msg.get("role") == "tool"]
        if len(tool_msg_indices) <= keep_n:
            return 0

        boundary_idx = tool_msg_indices[-keep_n]
        boundary_tool_call_id = self.messages[boundary_idx].get("tool_call_id", "")
        return self._find_turn_start(boundary_idx, boundary_tool_call_id)

    def _estimate_message_tokens(self, msg: Dict[str, Any]) -> int:
        role = msg.get("role")
        if role == "user":
            return len(msg.get("content", "")) // 4 + 8
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
            return len(msg.get("content", "")) // 4 + 15
        return 0

    def _assistant_input_tokens(self, assistant_idx: int) -> Optional[int]:
        msg = self.messages[assistant_idx]
        usage = msg.get("usage") or {}
        value = usage.get("input")
        if isinstance(value, int):
            return value
        return None

    def _assistant_content_tokens(self, assistant_idx: int) -> int:
        return self._estimate_message_tokens(self.messages[assistant_idx])

    def _compacted_tool_tokens(self, msg: Dict[str, Any]) -> int:
        tool_name = self._tool_name_from_msg(msg)
        if tool_name == "embed_recall":
            compacted_content = re.sub(r"<qr_paragraphs>[\s\S]*?</qr_paragraphs>", "", msg.get("content", "")).strip()
            compacted_content = compacted_content or "[cleared]"
        else:
            compacted_content = "[cleared]"
        return len(compacted_content) // 4 + 15

    def _estimate_compacted_prefix_tokens(self, boundary_assistant_idx: int) -> Optional[int]:
        assistant_indices = [i for i, msg in enumerate(self.messages) if msg.get("role") == "assistant"]
        if not assistant_indices:
            return None

        first_input_tokens = self._assistant_input_tokens(assistant_indices[0])
        if first_input_tokens is None:
            return None

        total = first_input_tokens
        for idx, msg in enumerate(self.messages[:boundary_assistant_idx]):
            role = msg.get("role")
            if role == "assistant":
                total += self._assistant_content_tokens(idx)
            elif role == "tool":
                total += self._compacted_tool_tokens(msg)
        return total

    def _compute_usage_based_boundary(self, pending_tool_tokens: int) -> Optional[int]:
        if not self.messages:
            return None

        max_tokens = self.config.usage_based_compaction_max_tokens
        if max_tokens <= 0:
            return None

        assistant_indices = [i for i, msg in enumerate(self.messages) if msg.get("role") == "assistant"]
        if not assistant_indices:
            return None

        latest_assistant_idx = assistant_indices[-1]
        latest_input_tokens = self._assistant_input_tokens(latest_assistant_idx)
        if latest_input_tokens is None:
            return None

        latest_content_tokens = self._last_content_tokens
        best_boundary: Optional[int] = None
        best_estimated: Optional[int] = None
        fallback_boundary: Optional[int] = None
        fallback_excess: Optional[int] = None

        for assistant_idx in assistant_indices:
            candidate_input_tokens = self._assistant_input_tokens(assistant_idx)
            if candidate_input_tokens is None or candidate_input_tokens > latest_input_tokens:
                continue

            compacted_prefix_tokens = self._estimate_compacted_prefix_tokens(assistant_idx)
            if compacted_prefix_tokens is None:
                continue

            exact_middle_tokens = latest_input_tokens - candidate_input_tokens
            estimated_total = compacted_prefix_tokens + exact_middle_tokens + latest_content_tokens + pending_tool_tokens

            if estimated_total <= max_tokens:
                if best_estimated is None or estimated_total > best_estimated:
                    best_estimated = estimated_total
                    best_boundary = assistant_idx
                continue

            excess = estimated_total - max_tokens
            if fallback_excess is None or excess < fallback_excess:
                fallback_excess = excess
                fallback_boundary = assistant_idx

        return best_boundary if best_boundary is not None else fallback_boundary

    def _apply_compaction(self, boundary: int) -> List[Dict[str, Any]]:
        compacted = []
        for i, msg in enumerate(self.messages):
            if msg.get("role") != "tool" or i >= boundary:
                compacted.append(deepcopy(msg))
                continue

            compacted_msg = deepcopy(msg)
            content = compacted_msg.get("content", "")
            tool_name = self._tool_name_from_msg(compacted_msg)

            if tool_name == "embed_recall":
                compacted_content = re.sub(r"<qr_paragraphs>[\s\S]*?</qr_paragraphs>", "", content).strip()
                compacted_msg["content"] = compacted_content or "[cleared]"
            else:
                compacted_msg["content"] = "[cleared]"

            compacted.append(compacted_msg)

        return compacted

    def _find_turn_start(self, tool_msg_idx: int, tool_call_id: str) -> int:
        assistant_idx = None
        for i in range(tool_msg_idx - 1, -1, -1):
            msg = self.messages[i]
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

    def _estimate_pending_tool_tokens(self) -> int:
        tool_result_overhead = 15

        last_assistant_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx < 0:
            return 0

        total_chars = 0
        n_tool_results = 0
        for i in range(last_assistant_idx + 1, len(self.messages)):
            if self.messages[i].get("role") == "tool":
                total_chars += len(self.messages[i].get("content", ""))
                n_tool_results += 1

        return total_chars // 4 + n_tool_results * tool_result_overhead

    def _find_tool_name_for_call_id(self, tool_call_id: str) -> str:
        for msg in self.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("id") == tool_call_id:
                        return tc.get("function", {}).get("name", "")
        return ""

    def run(self, question: str) -> str:
        if self.started_at is None:
            self.started_at = _utc_now_iso()
            self._append_event({"type": "agent_start"})
            self._print_progress(f"START model={self.config.model} max_turns={self.config.max_turns}", spacer=True)

        if not self.messages:
            user_msg = {"role": "user", "content": question, "timestamp": _timestamp_ms()}
            self.messages.append(user_msg)
            external_user = self._user_external_message(user_msg)
            self._append_event({"type": "message_start", "message": external_user})
            self._append_event({"type": "message_end", "message": external_user})
            self._save_incremental(status="running")

        openai_tools = [t.to_openai_tool() for t in self.tools_list]

        while self.turn_count < self.config.max_turns:
            self._append_event({"type": "turn_start"})
            force_answer_now = (
                self.config.force_answer_turns > 0
                and self.turn_count >= self.config.force_answer_turns
            )

            response = None
            compact_retry_count = 0
            while True:
                compacted_messages = self._compact_messages()
                full_messages = [{"role": "system", "content": self.system_prompt}] + compacted_messages
                if force_answer_now:
                    full_messages = full_messages + [{"role": "user", "content": FORCE_ANSWER_PROMPT}]

                self._save_latest_context(full_messages)

                try:
                    response = self.llm_client.chat(
                        messages=full_messages,
                        tools=None if force_answer_now else (openai_tools if openai_tools else None),
                        tool_choice="none" if force_answer_now else "auto",
                    )
                    break
                except Exception as e:
                    error_str = str(e)
                    if "context_length_exceeded" in error_str or ("tokens" in error_str.lower() and "exceed" in error_str.lower()):
                        # Real overflow monitoring applies to every LLM request. When the
                        # provider rejects the request for length, we enter overflow retry
                        # compaction instead of the normal threshold-only path.
                        compact_retry_count += 1
                        if compact_retry_count > 20:
                            logger.error(
                                f"LLM call failed after repeated compaction attempts at turn {self.turn_count}: {e}"
                            )
                            self.messages.append({
                                "role": "assistant",
                                "content": f"[Error: LLM call failed: {e}]",
                                "timestamp": _timestamp_ms(),
                                "stopReason": "error",
                                "usage": {},
                            })
                            self._save_incremental(status="completed")
                            break
                        self._activate_or_adjust_compaction(
                            f"Context length exceeded at turn {self.turn_count}",
                            overflow_retry=True,
                        )
                        continue

                    logger.error(f"LLM call failed at turn {self.turn_count}: {e}")
                    self.messages.append({
                        "role": "assistant",
                        "content": f"[Error: LLM call failed: {e}]",
                        "timestamp": _timestamp_ms(),
                        "stopReason": "error",
                        "usage": {},
                    })
                    self._save_incremental(status="completed")
                    break

            if response is None:
                break

            # Overflow handling is only a retry-time safeguard. Once one LLM call
            # succeeds again, future turns should go back to the full configured
            # keep_tool_calls unless another overflow happens later.
            self._overflow_retry_keep_n = None
            self._keep_tool_calls_override = None

            choice = response.choices[0]
            assistant_msg = choice.message
            msg_dict = assistant_msg_to_dict(assistant_msg)
            msg_dict["timestamp"] = _timestamp_ms()
            msg_dict["responseId"] = getattr(response, "id", None)
            msg_dict["stopReason"] = "toolUse" if assistant_msg.tool_calls else "stop"

            usage_info = ""
            if response.usage:
                u = response.usage
                prompt_details = getattr(u, "prompt_tokens_details", None)
                completion_details = getattr(u, "completion_tokens_details", None)
                cache_read = getattr(prompt_details, "cached_tokens", 0) or 0 if prompt_details else 0
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0 if completion_details else 0
                self._last_input_tokens = u.prompt_tokens or 0
                self._last_content_tokens = (u.completion_tokens or 0) - reasoning_tokens
                msg_dict["usage"] = {
                    "input": u.prompt_tokens or 0,
                    "output": u.completion_tokens or 0,
                    "cacheRead": cache_read,
                    "cacheWrite": (u.prompt_tokens or 0) - cache_read,
                    "totalTokens": u.total_tokens or 0,
                    "rawPromptTokens": u.prompt_tokens or 0,
                    "rawCompletionTokens": u.completion_tokens or 0,
                    "reasoningTokens": reasoning_tokens,
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "total": 0,
                    },
                }
                self._assistant_usage[len(self.messages)] = {
                    "input": u.prompt_tokens or 0,
                    "output": u.completion_tokens or 0,
                    "reasoning": reasoning_tokens,
                }
                usage_info = f" | {u.prompt_tokens or 0}in/{u.completion_tokens or 0}out"
            else:
                msg_dict["usage"] = {}

            self.messages.append(msg_dict)
            self.turn_count += 1

            external_assistant = self._assistant_external_message(msg_dict)
            self._append_event({"type": "message_start", "message": external_assistant})
            self._append_event({"type": "message_end", "message": external_assistant})

            n_tools = len(assistant_msg.tool_calls) if assistant_msg.tool_calls else 0
            text_preview = ""
            if assistant_msg.content:
                text_preview = assistant_msg.content[:150].replace("\n", " ")
            turn_msg = f"TURN {self.turn_count:03d} tools={n_tools}{usage_info}"
            if text_preview:
                turn_msg += f" text={text_preview}"
            self._print_progress(turn_msg, spacer=True)

            if force_answer_now and assistant_msg.tool_calls:
                logger.warning("Force-answer turn unexpectedly produced tool calls; ignoring and stopping")
                self._save_incremental(status="completed")
                break

            if not assistant_msg.tool_calls:
                logger.info(f"Agent finished at turn {self.turn_count} (no tool calls)")
                self._save_incremental(status="completed")
                break

            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                tool_call_id = tc.id

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                args_summary = _format_tool_args(tool_name, args)
                self._print_progress(f"> {tool_name}({args_summary})", indent=2)

                tool_started_at = _utc_now_iso()
                tool_start = time.perf_counter()
                self.tool_calls_log.append({
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "args": args,
                    "event": "tool_execution_start",
                    "recorded_at": tool_started_at,
                })
                self._append_event({
                    "type": "tool_execution_start",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "args": args,
                    "recorded_at": tool_started_at,
                })

                tool = self.tools.get(tool_name)
                if tool is None:
                    result_text = f"(error: unknown tool '{tool_name}')"
                    is_error = True
                else:
                    try:
                        result_text = tool.execute(**args)
                        is_error = False
                    except Exception as e:
                        result_text = f"(error: {tool_name} raised: {e})"
                        is_error = True
                        logger.error(f"Tool {tool_name} raised: {e}")

                tool_elapsed = time.perf_counter() - tool_start
                tool_finished_at = _utc_now_iso()
                tool_execution = {
                    "tool_call_id": tool_call_id,
                    "status": "completed" if not is_error else "failed",
                    "started_at": tool_started_at,
                    "finished_at": tool_finished_at,
                    "duration_seconds": round(tool_elapsed, 6),
                    "duration_ms": int(round(tool_elapsed * 1000)),
                }

                snippet = _result_snippet(result_text)
                status = "ERR" if is_error else "OK"
                self._print_progress(f"< [{status}] ({tool_elapsed:.1f}s) {snippet}", indent=4)

                self.tool_calls_log.append({
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "event": "tool_execution_end",
                    "recorded_at": tool_finished_at,
                    "isError": is_error,
                    "duration_seconds": round(tool_elapsed, 3),
                })
                self._append_event({
                    "type": "tool_execution_end",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "recorded_at": tool_finished_at,
                    "isError": is_error,
                    "duration_seconds": round(tool_elapsed, 3),
                    "result": {"content": _text_blocks(result_text)},
                })

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "content": result_text,
                    "is_error": is_error,
                    "timestamp": _timestamp_ms(),
                    "tool_execution": tool_execution,
                }
                self.messages.append(tool_msg)
                external_tool = self._tool_external_message(tool_msg)
                self._append_event({"type": "message_start", "message": external_tool})
                self._append_event({"type": "message_end", "message": external_tool})

            self._save_incremental(status="running")

        self.finished_at = _utc_now_iso()
        if self.turn_count >= self.config.max_turns:
            logger.warning(f"Agent hit max turns ({self.config.max_turns})")
        final_answer = extract_final_answer(self.messages)
        self._append_event({
            "type": "agent_end",
            "final_answer": final_answer,
            "turn_count": self.turn_count,
            "timestamp": self.finished_at,
        })
        self._print_progress(f"END turns={self.turn_count} final={final_answer[:120].replace(chr(10), ' ')}", spacer=True)
        return final_answer

    def get_state(self, *, status: str = "completed") -> Dict[str, Any]:
        return {
            "status": status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "turn_count": self.turn_count,
            "event_count": len(self.tool_calls_log),
            "messages": self.messages,
            "tool_calls": self.tool_calls_log,
            "assistant_text": extract_final_answer(self.messages),
            "last_input_tokens": self._last_input_tokens,
            "last_content_tokens": self._last_content_tokens,
            "compact_boundary": self._compact_boundary,
            "keep_tool_calls_override": self._keep_tool_calls_override,
            "overflow_retry_keep_n": self._overflow_retry_keep_n,
            "assistant_usage": self._assistant_usage,
            "resume_count": self.resume_count,
        }


# =============================================================================
# Output
# =============================================================================

def save_outputs(config: AgentConfig, session: AgentSession, final_answer: str) -> None:
    if not config.output_dir:
        return

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session.finished_at = session.finished_at or _utc_now_iso()
    session._save_incremental(status="completed")
    _safe_write_text(output_dir / "final.txt", final_answer)
    _safe_write_json(output_dir / "usage.json", session.llm_client.usage.to_dict())
    if config.question:
        _safe_write_text(output_dir / "question.txt", config.question)
        _safe_write_text(output_dir / "input_question.txt", config.question)


# =============================================================================
# Main
# =============================================================================

def main():
    config = build_config_from_args()

    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    output_dir = Path(config.output_dir) if config.output_dir else None
    resume_requested = False
    if output_dir:
        resume_requested = config.resume or ((output_dir / "state.json").exists() and not (output_dir / "final.txt").exists())

    if not config.question and not resume_requested:
        print("Error: --question is required", file=sys.stderr)
        sys.exit(1)

    logger.info("TS-Mirror Agent starting")
    logger.info(f"  Model: {config.model}")
    logger.info(f"  Max turns: {config.max_turns}")
    logger.info(f"  Corpus: {config.corpus_dir}")
    if output_dir:
        logger.info(f"  Output: {output_dir}")

    model_client = ModelServerClient(config)
    llm_client = LLMClient(config)
    tools = build_tools(config, model_client)

    if resume_requested and output_dir and (output_dir / "state.json").exists():
        session = AgentSession.resume_from_output_dir(config, llm_client, tools, model_client, output_dir)
    else:
        session = AgentSession(config, llm_client, tools, model_client)

    try:
        final_answer = session.run(config.question)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        final_answer = extract_final_answer(session.messages)
    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        final_answer = f"[Agent error: {e}]"
    finally:
        save_outputs(config, session, final_answer)
        model_client.shutdown()

    print(final_answer)


if __name__ == "__main__":
    main()
