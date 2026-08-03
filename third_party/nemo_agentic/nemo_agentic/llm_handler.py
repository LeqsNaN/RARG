# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM handler using a standard OpenAI-compatible API.

Retains the retry / logging / metadata behavior from the in-repo NeMo adaptation,
but removes all iChat-specific authentication logic for open-source release.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import aiofiles
from dotenv import load_dotenv

from .configs import LLMConfig
from .logging_utils import get_logger_with_config

logger, _ = get_logger_with_config()

LLM_ERROR_PREFIX = "LLMError:"


class QuotaExhaustedError(Exception):
    """Raised when API quota/frequency limit is exhausted. Program should save state and exit."""
    pass


load_dotenv()
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def is_error(response: Any) -> bool:
    return isinstance(response, str) and response.startswith(LLM_ERROR_PREFIX)


def normalize_messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, list):
            text_parts: List[str] = []
            all_text = True
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                else:
                    all_text = False
                    break
            if all_text:
                if len(text_parts) == 0:
                    msg["content"] = None
                elif len(text_parts) == 1:
                    msg["content"] = text_parts[0]
                else:
                    msg["content"] = "\n".join(text_parts)
        normalized.append(msg)
    return normalized


def write_json(obj: Any, log_dir: Union[str, Path], filename: Union[str, Path]):
    if log_dir is None:
        return
    path = Path(log_dir, filename)
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


async def awrite_json(obj: Any, log_dir: Union[str, Path], filename: Union[str, Path]):
    if log_dir is None:
        return
    path = Path(log_dir, filename)
    path.parent.mkdir(exist_ok=True, parents=True)
    async with aiofiles.open(path.as_posix(), "w") as f:
        await f.write(json.dumps(obj, indent=2))


class LLM:
    def __init__(self, llm_config: LLMConfig) -> None:
        from openai import AsyncOpenAI

        self.config = llm_config
        base_url = self.config.base_url or BASE_URL
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = self.config.model

    async def log_extra_data_log_dir(
        self,
        subdir: Optional[str] = None,
        info: Optional[Any] = None,
        filename: str = "extra_info.json",
    ) -> None:
        if info is None or self.config.raw_log_pardir is None or subdir is None:
            return
        if not filename.endswith(".json"):
            raise ValueError(f"filename must end with '.json', got {filename!r}")
        json_log_dir = Path(self.config.raw_log_pardir, subdir)
        await awrite_json(obj=info, log_dir=json_log_dir, filename=filename)

    async def acompletion(self, messages: list[dict], tools: Optional[list[dict]] = None, **kwargs: Any):
        return_metadata = kwargs.pop("return_metadata", False)
        logging_kwargs = kwargs.pop("logging_kwargs", None)

        messages = normalize_messages_for_api(messages)
        messages = [m for m in messages if m.get("role") != "agent_error"]

        request_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "timeout": 600,
            "max_tokens": self.config.max_completion_tokens or 20480,
            "stream": False,
        }

        if tools is not None and len(tools) > 0:
            request_kwargs["tools"] = tools
            tool_choice = self.config.tool_choice or "auto"
            request_kwargs["tool_choice"] = tool_choice

        if self.config.reasoning_effort is not None:
            request_kwargs.setdefault("extra_body", {})
            request_kwargs["extra_body"]["reasoning_effort"] = self.config.reasoning_effort
            request_kwargs["extra_body"]["reasoning"] = {
                "effort": self.config.reasoning_effort,
                "summary": "auto",
            }

        curr_step = logging_kwargs.get("step", uuid4().hex) if logging_kwargs else uuid4().hex
        json_log_dir = None
        if self.config.raw_log_pardir and logging_kwargs and "subdir" in logging_kwargs:
            json_log_dir = Path(self.config.raw_log_pardir, logging_kwargs["subdir"])

        io_log_kwargs = {
            "input_json": dict(obj=request_kwargs, log_dir=json_log_dir, filename=f"{curr_step}_prompt.json"),
        }
        if self.config.instant_log and json_log_dir:
            await awrite_json(**io_log_kwargs["input_json"])

        response = None
        max_retries = int(self.config.num_retries or 5)
        for attempt in range(max_retries):
            try:
                response = await self._client.chat.completions.create(**request_kwargs)
                break
            except Exception as e:
                err_str = str(e)
                err_str_lower = err_str.lower()
                if ("context" in err_str_lower and "window" in err_str_lower) or \
                   "context_length_exceeded" in err_str_lower or \
                   ("tokens" in err_str_lower and "exceed" in err_str_lower and "limit" in err_str_lower):
                    error_msg = f"{LLM_ERROR_PREFIX} Context window exceeded: {e}"
                    if self.config.strict_error_handling:
                        raise
                    print(error_msg)
                    return error_msg
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                    continue
                error_msg = f"{LLM_ERROR_PREFIX} All {max_retries} attempts failed. Last error: {type(e).__name__}: {e}"
                if self.config.strict_error_handling:
                    raise
                print(error_msg)
                return error_msg

        if response is None:
            return f"{LLM_ERROR_PREFIX} All retry attempts failed (no response)."

        io_log_kwargs["output_json"] = dict(
            obj=response.model_dump(), log_dir=json_log_dir, filename=f"{curr_step}_response.json"
        )
        if self.config.instant_log and json_log_dir:
            await awrite_json(**io_log_kwargs["output_json"])

        metadata_kv = {}
        usage = getattr(response, "usage", None)
        if usage:
            metadata_kv["PT"] = str(getattr(usage, "prompt_tokens", 0))
            metadata_kv["CT"] = str(getattr(usage, "completion_tokens", 0))

        if logging_kwargs:
            step_str = str(logging_kwargs.get("step", "?"))
            log_str = f"S: {step_str} " + " ".join(f"{k}: {v}" for k, v in metadata_kv.items())
            logger.info(log_str)

        if return_metadata:
            output = dict(response=response, metadata_kv=metadata_kv, io_log_kwargs=io_log_kwargs)
            try:
                r = response.model_dump()
                r.pop("choices", None)
                output["api_response_extras"] = r
            except Exception:
                pass
            return output
        return response
