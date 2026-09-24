"""Transport adapters used by the standalone Python DR-DCI port.

The agent runtime, tools, prompt, and harness deliberately live elsewhere and
do not import ``ts_mirror_agent``. The LLM adapter uses the standard OpenAI
Chat Completions interface, so it also works with compatible providers.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


class MalformedCompletionResponse(RuntimeError):
    """An HTTP-success response that is not a usable chat completion."""


def _response_summary(response: Any, *, limit: int = 2_000) -> str:
    """Return a bounded diagnostic without relying on a particular SDK model."""
    try:
        payload = response.model_dump(mode="json")
    except Exception:
        try:
            payload = dict(response)
        except Exception:
            return repr(response)[:limit]
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "… [response truncated]"


@dataclass
class Usage:
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    latency_seconds: float = 0.0

    def record(self, raw: Any, latency: float) -> Dict[str, Any]:
        prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
        completion = int(getattr(raw, "completion_tokens", 0) or 0)
        total = int(getattr(raw, "total_tokens", prompt + completion) or 0)
        prompt_details = getattr(raw, "prompt_tokens_details", None)
        completion_details = getattr(raw, "completion_tokens_details", None)
        cache_read = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        reasoning = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        self.input_tokens += prompt
        self.cache_read_tokens += cache_read
        self.cache_write_tokens += max(0, prompt - cache_read)
        self.output_tokens += completion
        self.reasoning_tokens += reasoning
        self.total_tokens += total
        self.request_count += 1
        self.latency_seconds += latency
        # This mirrors the per-assistant-message usage shape used by
        # ts_mirror_agent/Pi artifacts. Providers may omit cache/reasoning fields;
        # those are then accurately recorded as zero rather than invented.
        return {
            "input": prompt,
            "output": completion,
            "cacheRead": cache_read,
            "cacheWrite": max(0, prompt - cache_read),
            "totalTokens": total,
            "rawPromptTokens": prompt,
            "rawCompletionTokens": completion,
            "reasoningTokens": reasoning,
            "latencySeconds": round(latency, 3),
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            # Kept under the same name and definition as ts_mirror_agent:
            # prompt tokens not reported as cached by the provider.
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "content_output_tokens": max(0, self.output_tokens - self.reasoning_tokens),
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "total_latency_seconds": round(self.latency_seconds, 3),
        }


class OpenAICompletionsAdapter:
    """OpenAI Chat Completions transport for OpenAI-compatible endpoints."""

    def __init__(self, *, base_url: str, api_key: str, model: str, thinking_level: str, max_output_tokens: int):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.usage = Usage()

    def complete(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, int]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": self.max_output_tokens,
        }
        if self.thinking_level and self.thinking_level != "none":
            payload["extra_body"] = {
                "reasoning_effort": self.thinking_level,
                "reasoning": {"effort": self.thinking_level, "summary": "auto"},
            }
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                started = time.perf_counter()
                response = self.client.chat.completions.create(**payload)
                usage = self.usage.record(response.usage, time.perf_counter() - started)
                choices = getattr(response, "choices", None)
                if not isinstance(choices, list) or not choices:
                    raise MalformedCompletionResponse(
                        "The provider returned HTTP 200 but no usable Chat Completions choices; "
                        f"model={self.model!r}; response={_response_summary(response)}"
                    )
                message = getattr(choices[0], "message", None)
                if message is None:
                    raise MalformedCompletionResponse(
                        "The provider returned HTTP 200 but choices[0].message is null; "
                        f"model={self.model!r}; response={_response_summary(response)}"
                    )
                raw_calls = getattr(message, "tool_calls", None) or []
                tool_calls = []
                for call in raw_calls:
                    function = getattr(call, "function", None)
                    name = getattr(function, "name", None)
                    arguments = getattr(function, "arguments", None)
                    call_id = getattr(call, "id", None)
                    if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, str):
                        raise MalformedCompletionResponse(
                            "The provider returned an invalid function tool call; "
                            f"model={self.model!r}; response={_response_summary(response)}"
                        )
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    })
                return {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": tool_calls,
                }, usage
            except Exception as exc:  # network/provider retry policy from DR-DCI/Pi is provider-level
                last_error = exc
                if isinstance(exc, MalformedCompletionResponse) or attempt == 2 or "context_length_exceeded" in str(exc):
                    break
                # Use the same retry cadence as the companion Python agent:
                # retry after 2s, then after 4s (three attempts total).
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"OpenAI-compatible request failed after retries: {last_error}")

    def judge(self, *, question: str, gold_answer: str, predicted_answer: str) -> Dict[str, Any]:
        """Correctness judge transport using the same OpenAI-compatible client."""
        system = (
            "You are grading a question-answer benchmark. Mark the prediction correct only if it identifies the same "
            "final answer as the gold answer. Ignore case, surrounding punctuation, whitespace, and extra explanation "
            "or supporting file paths. Do not give partial credit. Return exactly one compact JSON object."
        )
        user = (
            f"Question:\n{question}\n\nGold answer:\n{gold_answer}\n\nPredicted answer:\n{predicted_answer or '[empty]'}\n\n"
            'Return JSON with keys "is_correct" (boolean), "normalized_prediction" (string), and "reason" (string).'
        )
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=180,
            extra_body={"reasoning": {"effort": "low", "summary": "auto"}},
        )
        usage = self.usage.record(response.usage, time.perf_counter() - started)
        text = response.choices[0].message.content or "{}"
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = __import__("re").search(r"\{.*\}", text, flags=__import__("re").S)
            value = json.loads(match.group(0)) if match else {"is_correct": False, "normalized_prediction": "", "reason": text}
        return {"is_correct": bool(value.get("is_correct")), "normalized_prediction": str(value.get("normalized_prediction", "")), "reason": str(value.get("reason", "")), "usage": usage}


class Qwen3EmbeddingAdapter:
    """Adapter for the pre-existing Qwen3 model_server JSON-line protocol."""

    def __init__(
        self,
        *,
        model_server_script: str,
        index_dir: str,
        embed_model_path: str,
        corpus_dir: str,
        device: str,
    ):
        self.model_server_script = model_server_script
        self.index_dir = index_dir
        self.embed_model_path = embed_model_path
        self.corpus_dir = corpus_dir
        self.device = device
        self._request_id = 0
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._socket: Optional[socket.socket] = None
        self._rfile = None
        self._wfile = None

    def start(self) -> None:
        if self._process is not None or self._socket is not None:
            return
        port = int(os.environ.get("MODEL_SERVER_PORT", "0") or 0)
        if port:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)
            sock.connect(("127.0.0.1", port))
            sock.settimeout(None)
            self._socket = sock
            self._rfile = sock.makefile("r", encoding="utf-8")
            self._wfile = sock.makefile("w", encoding="utf-8")
            ready = json.loads(self._rfile.readline())
            if "error" in ready:
                raise RuntimeError(f"Qwen3 model server startup error: {ready['error']}")
            return
        cmd = [
            os.environ.get("EMBED_PYTHON", sys.executable), self.model_server_script,
            "--index-dir", self.index_dir,
            "--embed-model-path", self.embed_model_path,
            "--corpus-dir", self.corpus_dir,
            "--device", self.device,
            "--no-reranker",
        ]
        self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert self._process.stdout is not None
        ready = json.loads(self._process.stdout.readline().decode("utf-8"))
        if "error" in ready:
            raise RuntimeError(f"Qwen3 model server startup error: {ready['error']}")

    def recall(self, query: str, top_k: int, scope_dir: str) -> Dict[str, Any]:
        with self._lock:
            self.start()
            self._request_id += 1
            request = {
                "id": self._request_id,
                "method": "embed_recall",
                "params": {
                    "query": query, "top_k": top_k, "scope_dir": scope_dir,
                    "scope_size_prefix": "", "sample_id": "", "paragraph_rerank_model": "none",
                    "paragraph_rerank_doc_limit": 0,
                },
            }
            if self._socket is not None:
                assert self._wfile is not None and self._rfile is not None
                self._wfile.write(json.dumps(request, ensure_ascii=False) + "\n")
                self._wfile.flush()
                response = json.loads(self._rfile.readline())
            else:
                assert self._process is not None and self._process.stdin is not None and self._process.stdout is not None
                self._process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
                self._process.stdin.flush()
                response = json.loads(self._process.stdout.readline().decode("utf-8"))
        if "error" in response:
            raise RuntimeError(f"Qwen3 embedding recall failed: {response['error']}")
        return response.get("result", {})

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
