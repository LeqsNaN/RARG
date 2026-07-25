#!/usr/bin/env python3
"""
BRIGHT-specific TS-Mirror Python Agent.

This module intentionally reuses the original ts_mirror_agent implementation as
much as possible. The bottom-layer agent logic is not reimplemented here; only
the high-level BRIGHT task prompt / force-answer format are adapted.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict


_SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import ts_mirror_agent.agent as _base_agent_module  # type: ignore  # noqa: E402
from ts_mirror_agent.agent import *  # type: ignore  # noqa: F401,F403,E402
from ts_mirror_agent.agent import AgentSession as BaseAgentSession  # type: ignore  # noqa: E402
from ts_mirror_agent.agent import save_outputs  # type: ignore  # noqa: E402
from ts_mirror_agent_bright.config import AgentConfig, build_config_from_args  # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)


def _suppress_noisy_third_party_debug_logs() -> None:
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


BRIGHT_FORCE_ANSWER_PROMPT_DCI = """You have reached the turn limit for tool use. Stop calling tools now and produce your best final retrieval result using only the evidence already gathered.

Your response MUST follow this exact format:

Relevant Documents (ranked by relevance, most relevant first; maximum 20):
1. relative/path/to/doc1.txt
2. relative/path/to/doc2.txt
3. relative/path/to/doc3.txt
(use full relative paths from the working directory; list at most 20 documents; omit any document that is not directly relevant)

Explanation: {brief explanation of why these are the most relevant documents}
Exact Answer: {concise final answer only}
Confidence: {0–100%; use below 50% if evidence is weak, ambiguous, or missing}
"""


BRIGHT_FORCE_ANSWER_PROMPT_NEMO = """You have reached the turn limit for tool use. Stop calling tools now and produce your best final retrieval result using only the evidence already gathered.

Do not answer the query directly. Return only the final ranked document list.

Your response MUST follow this exact format:

Relevant Documents (ranked by relevance, most relevant first; exactly 10):
1. relative/path/to/doc1.txt
2. relative/path/to/doc2.txt
3. relative/path/to/doc3.txt
4. relative/path/to/doc4.txt
5. relative/path/to/doc5.txt
6. relative/path/to/doc6.txt
7. relative/path/to/doc7.txt
8. relative/path/to/doc8.txt
9. relative/path/to/doc9.txt
10. relative/path/to/doc10.txt
"""


def resolve_force_answer_prompt(config: AgentConfig) -> str:
    prompt_file = str(getattr(config, "force_answer_prompt_file", "") or "").strip()
    if prompt_file and os.path.isfile(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            return text
    return BRIGHT_FORCE_ANSWER_PROMPT_DCI


TASK_QUERY_INSTRUCTIONS: Dict[str, str] = {
    "biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "earth_science": "Given an Earth Science post, retrieve relevant passages that help answer the post.",
    "economics": "Given an Economics post, retrieve relevant passages that help answer the post.",
    "robotics": "Given a Robotics post, retrieve relevant passages that help answer the post.",
    "psychology": "Given a Psychology post, retrieve relevant passages that help answer the post.",
    "stackoverflow": "Given a Stack Overflow post, retrieve relevant passages that help answer the post.",
    "sustainable_living": "Given a Sustainable Living post, retrieve relevant passages that help answer the post.",
    "leetcode": "Given a Coding problem, retrieve relevant examples that help answer the problem.",
    "pony": "Given a Pony question, retrieve relevant passages that help answer the question.",
    "aops": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
}


def resolve_query_instruction(config: AgentConfig) -> str:
    manual = (config.query_instruction or "").strip()
    if manual:
        return manual
    subset = (config.bright_subset or "").strip()
    return TASK_QUERY_INSTRUCTIONS.get(
        subset,
        "Given the following post, retrieve relevant passages that help answer the post.",
    )


def build_system_prompt(config: AgentConfig) -> str:
    if config.system_prompt_file and os.path.isfile(config.system_prompt_file):
        with open(config.system_prompt_file, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    elif config.append_system_prompt_file and os.path.isfile(config.append_system_prompt_file):
        with open(config.append_system_prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        max_docs = max(1, int(getattr(config, "max_retrieved_docs", 20) or 20))
        base_prompt = (
            "IMPORTANT CONSTRAINTS:\n"
            "- Do NOT use `ls`, `find`, or any command that lists all filenames. The corpus is large and nested; listing filenames wastes turns and context.\n"
            "- All returned document identifiers must be full relative paths from the working directory and must correspond to actual corpus files.\n\n"
            "EMBEDDING RECALL TOOL:\n"
            "The corpus contains many documents. To narrow the search space semantically, an embed_recall tool is available.\n"
            "By providing a query, it recalls a batch of relevant documents and saves their paths to a scope file (scope_1.txt, scope_2.txt, ...), one path per line.\n"
            "Results are returned ordered by document embedding similarity to the query.\n"
            "To search within recalled documents, use rg with the scope file:\n"
            "  cat /path/to/scope_N.txt | xargs -d '\\n' rg [OPTIONS] \"PATTERN\"\n"
            "Only search within one scope file at a time.\n\n"
            "SEARCH STRATEGY:\n"
            "- Initially call embed_recall with the ORIGINAL QUERY to establish a scope file.\n"
            "- Then use rg calls to explore within it with diverse keywords, entities, terminology, clues, and paraphrases.\n"
            "- Only call embed_recall again if the current scope is clearly missing the topic entirely. Do NOT call embed_recall repeatedly without rg exploration in between.\n"
            "- Do NOT run rg directly on the corpus without a scope file.\n"
            "- Do NOT stop after finding a few documents — exhaust all plausible search angles with rg.\n\n"
            "RETRIEVAL INSTRUCTIONS:\n"
            "- Find EVERY document that is genuinely relevant.\n"
            "- Carefully read candidate files before including them in the final ranking.\n"
            "- A document is relevant only if it directly addresses the query or provides essential supporting evidence. Do NOT include tangential or loosely related documents.\n"
            "- Recall and precision both matter: the output is evaluated with NDCG, which penalizes both missing relevant documents and including irrelevant ones.\n\n"
            "RANKING INSTRUCTIONS:\n"
            "- Rank the final list by relevance: the most directly useful document should appear first.\n\n"
            "When you are confident in the ranking, stop tool calling and respond exactly in this format:\n\n"
            f"Relevant Documents (ranked by relevance, most relevant first; maximum {max_docs}):\n"
            "1. relative/path/to/doc1.txt\n"
            "2. relative/path/to/doc2.txt\n"
            "3. relative/path/to/doc3.txt\n"
            f"(use full relative paths from the working directory; list at most {max_docs} documents; omit any document that is not directly relevant)\n\n"
            "Explanation: {brief explanation of why these are the most relevant documents}\n"
            "Exact Answer: {concise final answer only}\n"
            "Confidence: {0–100%; use below 50% if evidence is weak, ambiguous, or missing}\n"
        )

    if config.append_system_prompt_file and os.path.isfile(config.append_system_prompt_file):
        with open(config.append_system_prompt_file, "r", encoding="utf-8") as f:
            extra_prompt = f.read().strip()
        return "\n\n".join([base_prompt, extra_prompt]) if extra_prompt else base_prompt

    return base_prompt


class AgentSession(BaseAgentSession):
    def __init__(self, config, llm_client, tools, model_client):
        super().__init__(config, llm_client, tools, model_client)
        self.system_prompt = build_system_prompt(config)

_base_agent_module.build_system_prompt = build_system_prompt
_base_agent_module.AgentSession = AgentSession


def main() -> None:
    config = build_config_from_args()

    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    _suppress_noisy_third_party_debug_logs()

    query_instruction = resolve_query_instruction(config)
    if query_instruction:
        os.environ["QUERY_INSTRUCTION"] = query_instruction

    _base_agent_module.FORCE_ANSWER_PROMPT = resolve_force_answer_prompt(config)

    if not getattr(config, "model_server_script", ""):
        config.model_server_script = str(_REPO_ROOT / "scripts" / "model_server_bright.py")
    elif os.path.basename(config.model_server_script) == "model_server.py":
        config.model_server_script = str(_REPO_ROOT / "scripts" / "model_server_bright.py")

    _base_agent_module.main()


if __name__ == "__main__":
    main()
