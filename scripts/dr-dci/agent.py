#!/usr/bin/env python3
"""Independent Python port of DR-DCI's main agent-loop contract.

Only OpenAI-compatible/Qwen3 transport is adapter code. The tool loop, level3 context
policy, prompt, and tools are implemented here from DR-DCI/Pi.  Artifact
storage is deliberately leaner than Pi RPC: event logs are observability
metadata, while the single canonical full transcript is conversation.json.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from adapters import OpenAICompletionsAdapter, Qwen3EmbeddingAdapter
from prompt import build_main_prompt, build_pi_system_prompt
from tools import Tool, build_main_tools

LOG = logging.getLogger(__name__)
LEVEL3_MAX_TOOL_RESULT_CHARS = 20_000
LEVEL3_MICRO_COMPACT_KEEP_TURNS = 12
LEVEL3_MICRO_COMPACT_MIN_TOOL_RESULT_CHARS = 240_000
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_ms() -> int:
    return int(time.time() * 1000)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _quote_log_value(value: Any, *, limit: int | None = None) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", "\\n")
    if limit is not None and len(text) > limit:
        text = text[:limit] + "…"
    return json.dumps(text, ensure_ascii=False)


def _tool_args_summary(name: str, args: Dict[str, Any]) -> str:
    """Human-readable progress log; it does not affect tool execution."""
    if name == "bash":
        return f"command={_quote_log_value(args.get('command'))}"
    if name == "pull":
        return f"query={_quote_log_value(args.get('query'))}, topK={args.get('topK')!r}"
    if name == "read":
        parts = [f"path={_quote_log_value(args.get('path'))}"]
        for key in ("offset", "limit", "charOffset", "charLimit", "byteOffset", "byteLimit"):
            if args.get(key) is not None:
                parts.append(f"{key}={args[key]!r}")
        return ", ".join(parts)
    return json.dumps(args, ensure_ascii=False, default=str)[:400]


def _tool_result_preview(text: str) -> str:
    """Match ts_mirror_agent's compact operator-facing result snippet.

    Full tool output remains in state.json/conversation.json and enters model
    context normally. This affects only log.txt and the launcher log.
    """
    max_len = 300
    first_line = text.split("\n")[0]
    if len(first_line) > max_len:
        return first_line[:max_len] + "..."
    if len(first_line) < 80 and "\n" in text:
        snippet = "\n".join(text.split("\n")[:3])
        return snippet[:max_len] + "..." if len(snippet) > max_len else snippet
    return first_line


@dataclass
class Config:
    question: str
    hidden_corpus_dir: str
    workspace_dir: str
    output_dir: str
    index_dir: str
    provider: str = "openai"
    model: str = "gpt-5.4-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    thinking_level: str = "medium"
    max_turns: int = 200
    max_output_tokens: int = 128000
    embed_model_path: str = "models/Qwen3-Embedding-4B"
    device: str = "cuda:0"
    model_server_script: str = ""
    pull_meta_dir: str = ""
    pull_preview_limit: int = 20
    max_pull_calls: int = 0
    sample_id: str = ""
    # Off by default: DR-DCI/Pi level3 creates a temporary request view, and
    # never mutates the canonical transcript while micro-compacting.
    level3_persistent_micro_compact: bool = False


class Session:
    def __init__(self, config: Config, llm: OpenAICompletionsAdapter, retriever: Qwen3EmbeddingAdapter):
        self.config, self.llm, self.retriever = config, llm, retriever
        self.output_dir = Path(config.output_dir)
        self.workspace = Path(config.workspace_dir)
        self.tools: List[Tool] = build_main_tools(
            self.workspace, Path(config.hidden_corpus_dir), retriever,
            pull_meta_dir=Path(config.pull_meta_dir) if config.pull_meta_dir else None,
            preview_limit=config.pull_preview_limit,
            max_pull_calls=config.max_pull_calls,
        )
        self.tool_map = {tool.name: tool for tool in self.tools}
        # ``messages`` is both the OpenAI-compatible agent context and the
        # canonical full transcript. It is stored only in conversation.json.
        # state.json and events.jsonl intentionally contain no tool text.
        self.messages: List[Dict[str, Any]] = []
        self.turn_count = 0
        self.tool_calls: List[Dict[str, Any]] = []
        self.event_count = 0
        self.started_at = ""
        self.finished_at = ""
        self.last_input_tokens = 0
        self.last_content_tokens = 0
        self.assistant_usage: Dict[str, Dict[str, Any]] = {}
        self._latest_context: List[Dict[str, Any]] = []
        self._latest_context_meta: Dict[str, Any] = {}
        self._level3_compactions: List[Dict[str, Any]] = []
        self._next_level3_request_meta: Dict[str, Any] = {}

    def _progress(self, message: str, *, indent: int = 0, spacer: bool = False) -> None:
        """Tee concise live diagnostics to the harness log and sample log.txt.

        This is intentionally not events.jsonl: events remain resume-free,
        bounded metadata, while a run log is an operator-facing trace.
        """
        sample = self.config.sample_id or self.output_dir.name or "-"
        line = f"[sample {sample}] {message}"
        if indent:
            line = " " * indent + line
        log_path = self.output_dir / "log.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            if spacer:
                handle.write("\n")
            handle.write(line + "\n")
        if spacer:
            LOG.info("")
        LOG.info("%s", line)

    @staticmethod
    def _short_error(value: Any, limit: int = 512) -> str:
        text = str(value or "").replace("\n", " ")
        return text if len(text) <= limit else text[:limit] + "…"

    def _event(self, payload: Dict[str, Any]) -> None:
        """Append a deliberately content-free progress event.

        Tool output can be tens of KB per call and a 300-turn run has many
        calls.  Recording it in an append-only event stream would retain many
        copies and can create multi-GB sample directories.  Full content is
        persisted once in conversation.json instead.
        """
        event_type = str(payload.get("type") or payload.get("event") or "unknown")
        record: Dict[str, Any] = {
            "type": event_type,
            "recorded_at": payload.get("recorded_at") or now(),
            "turn": self.turn_count,
        }
        if self.config.sample_id:
            record["sample_id"] = self.config.sample_id
        message = payload.get("message")
        if isinstance(message, dict):
            record["role"] = message.get("role")
            record["content_chars"] = len(str(message.get("content") or ""))
            calls = message.get("tool_calls") or []
            if calls:
                record["tool_call_count"] = len(calls)
                record["tool_calls"] = [
                    {"id": call.get("id"), "name": (call.get("function") or {}).get("name")}
                    for call in calls
                ]
        for key in ("toolCallId", "toolName", "isError", "started_at", "finished_at", "duration_seconds"):
            if key in payload:
                record[key] = payload[key]
        if "argument_chars" in payload:
            record["argument_chars"] = payload["argument_chars"]
        if "arguments" in payload:
            record["argument_chars"] = len(json.dumps(payload["arguments"], ensure_ascii=False, default=str))
        result = payload.get("result")
        if isinstance(result, dict):
            record["result_chars"] = len(str(result.get("content") or ""))
        if "final_answer" in payload:
            record["final_answer_chars"] = len(str(payload["final_answer"] or ""))
        if "error" in payload:
            record["error"] = self._short_error(payload["error"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.event_count += 1

    @staticmethod
    def _text_blocks(text: str) -> List[Dict[str, str]]:
        return [{"type": "text", "text": text}] if text else []

    @staticmethod
    def _parse_tool_arguments(raw: Any) -> Any:
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def _external_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Pi-style readable transcript, modelled after ts_mirror_agent."""
        role = message.get("role")
        if role == "user":
            return {
                "role": "user", "content": self._text_blocks(str(message.get("content") or "")),
                "timestamp": message.get("timestamp", timestamp_ms()),
            }
        if role == "assistant":
            blocks: List[Dict[str, Any]] = self._text_blocks(str(message.get("content") or ""))
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                blocks.append({
                    "type": "toolCall", "id": call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": self._parse_tool_arguments(function.get("arguments", {})),
                })
            return {
                "role": "assistant", "content": blocks,
                "api": "openai-completions", "provider": self.config.provider,
                "model": self.config.model, "usage": message.get("usage") or {},
                "stopReason": "toolUse" if message.get("tool_calls") else "stop",
                "timestamp": message.get("timestamp", timestamp_ms()),
            }
        if role == "tool":
            external = {
                "role": "toolResult", "toolCallId": message.get("tool_call_id", ""),
                "toolName": message.get("name", ""),
                "content": self._text_blocks(str(message.get("content") or "")),
                "isError": bool(message.get("is_error", False)),
                "timestamp": message.get("timestamp", timestamp_ms()),
            }
            if message.get("tool_execution"):
                external["tool_execution"] = message["tool_execution"]
            return external
        return dict(message)

    def _conversation(self, status: str) -> Dict[str, Any]:
        system_prompt = build_pi_system_prompt(str(self.workspace))
        messages: List[Dict[str, Any]] = [{
            "role": "system", "content": self._text_blocks(system_prompt),
            "sources": {"system_prompt": "scripts/dr-dci/prompt.py::build_pi_system_prompt"},
        }]
        messages.extend(self._external_message(message) for message in self.messages)
        return {
            "artifact_schema": "pi-style-conversation-v2",
            "started_at": self.started_at, "finished_at": self.finished_at or None,
            "status": status, "question": self.config.question, "cwd": str(self.workspace),
            "provider": self.config.provider, "model": self.config.model,
            "tools": "read,bash,pull", "max_turns": self.config.max_turns,
            "max_turns_mode": "abort", "keep_session": True,
            "turn_count": self.turn_count, "event_count": self.event_count,
            "assistant_text": self.final_text() or None,
            "usage": self.llm.usage.as_dict(),
            "conversation_features": self._state_features(),
            "messages": messages,
        }

    def _state_features(self) -> Dict[str, Any]:
        return {
            "runtime_context_management": "level3", "truncate_tool_results": True,
            "max_tool_result_chars": LEVEL3_MAX_TOOL_RESULT_CHARS,
            "micro_compact_min_tool_result_chars": LEVEL3_MICRO_COMPACT_MIN_TOOL_RESULT_CHARS,
            "micro_compact_keep_turns": LEVEL3_MICRO_COMPACT_KEEP_TURNS,
            "micro_compact_mode": "persistent" if self.config.level3_persistent_micro_compact else "request_time",
            "persistent_micro_compact_enabled": self.config.level3_persistent_micro_compact,
        }

    def _state(self, status: str) -> Dict[str, Any]:
        return {
            "status": status, "started_at": self.started_at, "finished_at": self.finished_at,
            "question": self.config.question, "provider": self.config.provider, "model": self.config.model,
            "tools": "read,bash,pull", "max_turns": self.config.max_turns, "max_turns_mode": "abort",
            "artifact_schema": "runtime-state-v2",
            "turn_count": self.turn_count, "event_count": self.event_count,
            # A full, raw runtime snapshot for audit/debugging. This is not
            # used to resume a partial sample under the lean-resume policy.
            "messages": self.messages, "tool_calls": self.tool_calls,
            "assistant_text": self.final_text(), "usage": self.llm.usage.as_dict(),
            "last_input_tokens": self.last_input_tokens,
            "last_content_tokens": self.last_content_tokens,
            "assistant_usage": self.assistant_usage,
            "latest_model_context": self._latest_context_meta,
            "level3_compactions": self._level3_compactions,
            "paths": {
                "output_dir": str(self.output_dir), "events_jsonl": str(self.output_dir / "events.jsonl"),
                "state_json": str(self.output_dir / "state.json"),
                "conversation_json": str(self.output_dir / "conversation.json"), "latest_model_context_json": str(self.output_dir / "latest_model_context.json"),
                "final_txt": str(self.output_dir / "final.txt"), "eval_result_json": str(self.output_dir / "eval_result.json"),
                "stderr_txt": str(self.output_dir / "stderr.txt"), "question_txt": str(self.output_dir / "question.txt"), "tool_results_dir": str(self.output_dir / "tool_results"),
                "cwd": str(self.workspace),
            },
            "conversation_features": self._state_features(),
        }

    def _save_latest_context(self) -> None:
        """Persist the exact context prepared for the most recent LLM call."""
        if not self._latest_context:
            return
        level3 = self._latest_context_meta
        cleared = list(level3.get("cleared_tool_call_ids") or [])
        payload = {
            "artifact_schema": "model-request-context-v2",
            "captured_at": self._latest_context_meta.get("captured_at"),
            "request_turn": self._latest_context_meta.get("request_turn"),
            "message_count": len(self._latest_context),
            "tools": [tool.openai_schema() for tool in self.tools],
            "tool_choice": "auto", "temperature": 0.0, "model": self.config.model,
            "provider": self.config.provider, "thinking_level": self.config.thinking_level,
            "max_output_tokens": self.config.max_output_tokens,
            "extra_body": {
                "reasoning_effort": self.config.thinking_level,
                "reasoning": {"effort": self.config.thinking_level, "summary": "auto"},
            } if self.config.thinking_level and self.config.thinking_level != "none" else None,
            "runtime_context_management": {
                "level": "level3",
                "micro_compact_mode": level3.get("micro_compact_mode", "request_time"),
                "persistent_micro_compact_enabled": bool(level3.get("persistent_micro_compact_enabled", False)),
                "total_tool_result_chars_before_micro_compact": level3.get("total_tool_result_chars_before_micro_compact", 0),
                "total_tool_result_chars_after_micro_compact": level3.get("total_tool_result_chars_after_micro_compact", 0),
                "micro_compaction_applied": bool(level3.get("micro_compaction_applied", False)),
                "cleared_tool_call_ids": cleared,
                "cleared_tool_result_count": len(cleared),
                "maxToolResultChars": LEVEL3_MAX_TOOL_RESULT_CHARS,
                "microCompactMinToolResultChars": LEVEL3_MICRO_COMPACT_MIN_TOOL_RESULT_CHARS,
                "microCompactKeepTurns": LEVEL3_MICRO_COMPACT_KEEP_TURNS,
            },
            "messages": self._latest_context,
        }
        atomic_json(self.output_dir / "latest_model_context.json", payload)

    def save(self, status: str) -> None:
        atomic_json(self.output_dir / "conversation.json", self._conversation(status))
        atomic_json(self.output_dir / "state.json", self._state(status))
        atomic_json(self.output_dir / "usage.json", self.llm.usage.as_dict())
        self._save_latest_context()

    def final_text(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "assistant" and not message.get("tool_calls"):
                return str(message.get("content") or "")
        return ""

    def _level3_tool_result(self, text: str) -> str:
        if len(text) <= LEVEL3_MAX_TOOL_RESULT_CHARS:
            return text
        return text[:LEVEL3_MAX_TOOL_RESULT_CHARS] + f"\n[...truncated, {len(text) - LEVEL3_MAX_TOOL_RESULT_CHARS} chars omitted]"

    def _execute_one_tool(self, name: str, args: Dict[str, Any]) -> tuple[str, Dict[str, Any], bool, float]:
        """Run one prepared call; Pi's default toolExecution mode is parallel."""
        started = time.perf_counter()
        try:
            tool = self.tool_map.get(name)
            if tool is None:
                raise RuntimeError(f"Tool {name} not found")
            validated_args = tool.validate(args)
            result, details = tool.execute(**validated_args)
            return self._level3_tool_result(result), details, False, time.perf_counter() - started
        except Exception as exc:
            return str(exc), {}, True, time.perf_counter() - started

    def _model_messages(self) -> List[Dict[str, Any]]:
        # DR-DCI/Pi has two distinct layers: Pi's generated system prompt and
        # the benchmark's rank-aware task prompt supplied as the user message.
        messages: List[Dict[str, Any]] = [{"role": "system", "content": build_pi_system_prompt(str(self.workspace))}]
        messages.extend(self.messages)
        total_tool_chars = sum(len(str(msg.get("content", ""))) for msg in messages if msg.get("role") == "tool")
        level3_meta: Dict[str, Any] = {
            "micro_compact_mode": "persistent" if self.config.level3_persistent_micro_compact else "request_time",
            "persistent_micro_compact_enabled": self.config.level3_persistent_micro_compact,
            "total_tool_result_chars_before_micro_compact": total_tool_chars,
            "total_tool_result_chars_after_micro_compact": total_tool_chars,
            "micro_compaction_applied": False,
            "cleared_tool_call_ids": [],
        }
        self._next_level3_request_meta = level3_meta
        if total_tool_chars <= LEVEL3_MICRO_COMPACT_MIN_TOOL_RESULT_CHARS:
            return messages
        assistant_seen = 0
        cutoff = 0
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                assistant_seen += 1
                if assistant_seen > LEVEL3_MICRO_COMPACT_KEEP_TURNS:
                    cutoff = idx + 1
                    break
        if not cutoff:
            return messages
        compacted = list(messages)
        cleared_ids: List[str] = []
        for idx in range(1, cutoff):
            if compacted[idx].get("role") == "tool" and compacted[idx].get("content") != "[cleared]":
                cleared_ids.append(str(compacted[idx].get("tool_call_id") or ""))
                compacted[idx] = {**compacted[idx], "content": "[cleared]"}
        if not cleared_ids:
            return messages
        compacted_tool_chars = sum(len(str(msg.get("content", ""))) for msg in compacted if msg.get("role") == "tool")
        level3_meta.update({
            "total_tool_result_chars_after_micro_compact": compacted_tool_chars,
            "micro_compaction_applied": True,
            "cleared_tool_call_ids": cleared_ids,
        })
        if self.config.level3_persistent_micro_compact:
            # Experimental extension: use the compacted context as the new
            # canonical baseline. Future tool results now accumulate from it,
            # and no second compaction occurs until the threshold is crossed.
            self.messages = compacted[1:]
            event = {
                "request_turn": self.turn_count + 1,
                "total_tool_result_chars_before_micro_compact": total_tool_chars,
                "total_tool_result_chars_after_micro_compact": compacted_tool_chars,
                "cleared_tool_call_ids": cleared_ids,
            }
            self._level3_compactions.append(event)
            self._progress(
                "LEVEL3 persistent compact "
                f"request_turn={event['request_turn']} cleared={len(cleared_ids)} "
                f"tool_chars={total_tool_chars}->{compacted_tool_chars}",
                spacer=True,
            )
        return compacted

    def _capture_model_context(self, messages: List[Dict[str, Any]]) -> None:
        # json roundtrip makes the snapshot independent from later mutations.
        self._latest_context = json.loads(json.dumps(messages, ensure_ascii=False))
        self._latest_context_meta = {
            "captured_at": now(), "request_turn": self.turn_count + 1,
            **self._next_level3_request_meta,
        }
        self._save_latest_context()

    def run(self) -> str:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "question.txt").write_text(self.config.question, encoding="utf-8")
        if not self.messages:
            user_message = {"role": "user", "content": build_main_prompt(self.config.question, "corpus"), "timestamp": timestamp_ms()}
            self.messages.append(user_message)
            self._event({"event": "message_end", "message": user_message, "recorded_at": now()})
        if not self.started_at: self.started_at = now()
        self._event({"event": "agent_start", "recorded_at": now()})
        self.save("running")
        self._progress(f"START model={self.config.model} thinking={self.config.thinking_level} max_turns={self.config.max_turns}", spacer=True)
        try:
            while self.turn_count < self.config.max_turns:
                request_messages = self._model_messages()
                # Must happen before the network request: this file is the
                # exact level3-transformed input sent to the provider, even if the
                # request subsequently fails.
                self._capture_model_context(request_messages)
                assistant, usage = self.llm.complete(request_messages, [tool.openai_schema() for tool in self.tools])
                self.turn_count += 1
                assistant["usage"] = usage
                assistant["timestamp"] = timestamp_ms()
                assistant["stopReason"] = "toolUse" if assistant.get("tool_calls") else "stop"
                self.last_input_tokens = int(usage.get("input", 0) or 0)
                self.last_content_tokens = max(0, int(usage.get("output", 0) or 0) - int(usage.get("reasoningTokens", 0) or 0))
                self.messages.append(assistant)
                self.assistant_usage[str(self.turn_count)] = dict(usage)
                self._event({"event": "message_end", "message": assistant, "recorded_at": now()})
                # Persist the canonical transcript immediately, but do not
                # duplicate its content in the event stream.
                self.save("running")
                tool_count = len(assistant.get("tool_calls") or [])
                usage_text = f" input={usage.get('input', 0)} output={usage.get('output', 0)}" if usage else ""
                text = str(assistant.get("content") or "").replace("\n", " ").strip()
                text_summary = f" text={_quote_log_value(text, limit=180)}" if text else ""
                self._progress(f"TURN {self.turn_count:03d} tools={tool_count}{usage_text}{text_summary}", spacer=True)
                if not assistant.get("tool_calls"):
                    break
                prepared_calls: List[tuple[Dict[str, Any], str, Dict[str, Any], str]] = []
                # Pi preflights calls in assistant source order and emits all starts
                # before allowed calls execute concurrently.
                for call in assistant["tool_calls"]:
                    call_id = call["id"]
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"]["arguments"] or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("tool arguments must be a JSON object")
                    except (json.JSONDecodeError, ValueError) as exc:
                        args = {"__invalid_arguments__": str(exc)}
                    started = now()
                    started_record = {
                        "event": "tool_execution_start", "toolCallId": call_id,
                        "toolName": name, "arguments": args,
                        "argument_chars": len(json.dumps(args, ensure_ascii=False, default=str)),
                        "recorded_at": started,
                    }
                    self.tool_calls.append(started_record)
                    self._event(started_record)
                    self.save("running")
                    self._progress(f"> {name}({_tool_args_summary(name, args)})", indent=2)
                    prepared_calls.append((call, name, args, started))

                with ThreadPoolExecutor(max_workers=max(1, len(prepared_calls))) as executor:
                    futures = []
                    for _, name, args, _ in prepared_calls:
                        if "__invalid_arguments__" in args:
                            futures.append(None)
                        else:
                            futures.append(executor.submit(self._execute_one_tool, name, args))
                    # Results are appended in original source order, matching Pi's
                    # executeToolCallsParallel finalization contract.
                    for (call, name, args, started), future in zip(prepared_calls, futures):
                        call_id = call["id"]
                        if future is None:
                            result, details, is_error, duration = (f"Invalid tool arguments: {args['__invalid_arguments__']}", {}, True, 0.0)
                        else:
                            result, details, is_error, duration = future.result()
                        finished_at = now()
                        tool_execution = {
                            "tool_call_id": call_id,
                            "status": "failed" if is_error else "completed",
                            "started_at": started, "finished_at": finished_at,
                            "duration_seconds": round(duration, 6),
                            "duration_ms": int(round(duration * 1000)),
                        }
                        tool_message = {
                            "role": "tool", "tool_call_id": call_id, "name": name,
                            "content": result, "is_error": is_error,
                            "timestamp": timestamp_ms(), "tool_execution": tool_execution,
                        }
                        self.messages.append(tool_message)
                        # state.json is an inspectable runtime snapshot, so it
                        # retains full result/details. events.jsonl receives
                        # only the length via _event's slim projection.
                        ended = {
                            "event": "tool_execution_end", "toolCallId": call_id,
                            "toolName": name,
                            "result": {"content": result, "details": details},
                            "isError": is_error, "started_at": started,
                            "finished_at": finished_at, "duration_seconds": duration,
                            "recorded_at": now(),
                        }
                        self.tool_calls.append(ended)
                        self._event(ended)
                        status = "ERR" if is_error else "OK"
                        header = f"< [{status}] {name} ({duration:.2f}s, {len(result)} chars)"
                        self._progress(header + "\n" + _tool_result_preview(result), indent=4)
                        self.save("running")
                self.save("running")
        except Exception as exc:
            self.finished_at = now()
            self._event({"event": "agent_end", "error": str(exc), "turn_count": self.turn_count, "recorded_at": self.finished_at})
            self.save("failed")
            atomic_json(self.output_dir / "state.json", {**self._state("failed"), "error": str(exc)})
            (self.output_dir / "stderr.txt").write_text(str(exc) + "\n", encoding="utf-8")
            self._progress(f"END status=failed turns={self.turn_count} error={_quote_log_value(exc, limit=500)}", spacer=True)
            raise
        self.finished_at = now()
        final = self.final_text()
        self._event({"event": "agent_end", "turn_count": self.turn_count, "final_answer": final, "recorded_at": self.finished_at})
        # Write the final recorder state after agent_end, so state/event counts
        # and persisted artifacts stay mutually consistent.
        self.save("completed")
        (self.output_dir / "final.txt").write_text(final, encoding="utf-8")
        (self.output_dir / "stderr.txt").touch(exist_ok=True)
        atomic_json(self.output_dir / "usage.json", self.llm.usage.as_dict())
        self._progress(f"END status=completed turns={self.turn_count} final={_quote_log_value(final, limit=300)}", spacer=True)
        return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Python port of DR-DCI main path")
    parser.add_argument("--question", required=True); parser.add_argument("--corpus-dir", required=True); parser.add_argument("--workspace-dir", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--index-dir", required=True)
    parser.add_argument("--provider", default="openai"); parser.add_argument("--model", default="gpt-5.4-mini"); parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")); parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "")); parser.add_argument("--thinking-level", default="medium")
    parser.add_argument("--max-turns", type=int, default=200); parser.add_argument("--max-output-tokens", type=int, default=128000); parser.add_argument("--embed-model-path", default=Config.embed_model_path); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--max-pull-calls", type=int, default=0); parser.add_argument("--level3-persistent-micro-compact", action="store_true", help="Non-DR-DCI extension: persist level3 clears, then compact again only after newly accumulated tool output reaches 240k chars."); parser.add_argument("--resume", action="store_true", help="Unsupported: incomplete samples must be rerun from the beginning."); parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume:
        raise SystemExit("--resume is unsupported by the lean artifact format; rerun this sample from the beginning.")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    workspace = Path(args.workspace_dir)
    config = Config(question=args.question, hidden_corpus_dir=args.corpus_dir, workspace_dir=args.workspace_dir, output_dir=args.output_dir, index_dir=args.index_dir, provider=args.provider, model=args.model, base_url=args.base_url, api_key=args.api_key, thinking_level=args.thinking_level, max_turns=args.max_turns, max_output_tokens=args.max_output_tokens, embed_model_path=args.embed_model_path, device=args.device, model_server_script=str(Path(__file__).resolve().parents[1] / "model_server.py"), pull_meta_dir=str(workspace.parent.parent / "_pull_meta" / workspace.name), max_pull_calls=args.max_pull_calls, level3_persistent_micro_compact=args.level3_persistent_micro_compact)
    llm = OpenAICompletionsAdapter(base_url=config.base_url, api_key=config.api_key, model=config.model, thinking_level=config.thinking_level, max_output_tokens=config.max_output_tokens)
    retriever = Qwen3EmbeddingAdapter(model_server_script=config.model_server_script, index_dir=config.index_dir, embed_model_path=config.embed_model_path, corpus_dir=config.hidden_corpus_dir, device=config.device)
    session = Session(config, llm, retriever)
    try: print(session.run())
    finally: retriever.close()


if __name__ == "__main__": main()
