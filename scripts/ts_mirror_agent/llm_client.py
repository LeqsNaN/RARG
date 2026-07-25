"""
LLM client wrapper using OpenAI-compatible API.

Supports function calling, extended thinking, retry, and usage statistics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Accumulated token usage across all LLM calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    total_latency_s: float = 0.0

    def record(self, usage: Any, latency_s: float):
        if usage:
            self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.total_tokens += getattr(usage, "total_tokens", 0) or 0
        self.request_count += 1
        self.total_latency_s += latency_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "total_latency_s": round(self.total_latency_s, 2),
        }


class LLMClient:
    """OpenAI-compatible LLM client with function calling and retry."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.model = config.model
        self.temperature = config.temperature
        self.max_output_tokens = config.max_output_tokens
        self.thinking_level = config.thinking_level
        self.usage = UsageStats()

        self.max_retries = 3
        self.retry_delay_base = 2.0

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> ChatCompletion:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        if self.max_output_tokens > 0:
            kwargs["max_tokens"] = self.max_output_tokens

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if self.thinking_level and self.thinking_level != "none":
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["reasoning_effort"] = self.thinking_level
            kwargs["extra_body"]["reasoning"] = {
                "effort": self.thinking_level,
                "summary": "auto",
            }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.perf_counter()
                response = self.client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - start_time

                self.usage.record(response.usage, latency)
                logger.debug(
                    f"LLM call #{self.usage.request_count}: "
                    f"{response.usage.prompt_tokens}in/{response.usage.completion_tokens}out "
                    f"[{latency:.1f}s]"
                    if response.usage else f"LLM call [{latency:.1f}s]"
                )
                return response

            except Exception as e:
                last_error = e
                error_str = str(e)
                if "context_length_exceeded" in error_str:
                    break
                if attempt >= self.max_retries:
                    break
                delay = self.retry_delay_base * (2 ** (attempt - 1))
                logger.warning(f"LLM request failed (attempt {attempt}/{self.max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)

        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _thinking_budget(level: str) -> int:
        budgets = {
            "low": 4096,
            "medium": 10240,
            "high": 32768,
        }
        return budgets.get(level, 10240)
