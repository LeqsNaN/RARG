"""
Alternative bash tool definitions.

This module keeps the original `bash` tool name but exposes a variant whose
schema explicitly includes a rerank-only query field. The rg command itself
remains unchanged; only the downstream rerank query differs.
"""

from __future__ import annotations

from typing import Optional

from config import AgentConfig
from tools import BashTool


class RerankAwareBashTool(BashTool):
    name = "bash"
    accepts_explicit_rerank_query = True
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr. "
        "Output is truncated to last 2000 lines or 50KB (whichever is hit first). "
        "If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds. "
        "For rg commands, you may also provide a rerank_query used only for semantic reranking of matches."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Bash command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (optional). rg commands default to 30s.",
            },
            "rerank_query": {
                "type": "string",
                "description": (
                    "Optional natural-language query used only to rerank rg matches. For rg commands, write exactly two short sentences. "
                    "Sentence 1 must state the final fact, entity, or relation the user is trying to determine from the question. "
                    "Sentence 2 must state what this specific rg step is trying to verify, identify, or retrieve. "
                    "Build the query by jointly considering: the question (what the user is ultimately asking), the relevant conversation context so far (what has already been learned, narrowed down, or hypothesized), and the current search step (what evidence this rg call is trying to find). "
                    "Use concrete entities and informative content. "
                    "The query should become more specific as the conversation narrows. "
                    "Example: The user is trying to determine which Jinx ability corresponds to the name Super Mega Death Rocket and whether it is her ultimate ability. This rg step is looking for text that explicitly links Super Mega Death Rocket to Jinx and states the ability's slot or letter mapping. "
                    # "Example: The goal is to determine which League of Legends champion has a global-range R ability that can conditionally trigger a passive effect. "
                    # "This step is looking for text that explicitly introduces Jinx's abilities and states the ability slot or letter mapping (Q, W, E, R). "
                    "For non-rg commands, omit this field."
                ),
            },
        },
        "required": ["command"],
    }

    def __init__(self, config: AgentConfig, model_client=None):
        super().__init__(config, model_client)
        self._last_rerank_query: Optional[str] = None

    def execute(self, *, command: str, timeout: int = 0, rerank_query: Optional[str] = None, **kwargs) -> str:
        self._last_rerank_query = rerank_query.strip() if isinstance(rerank_query, str) and rerank_query.strip() else None
        return super().execute(
            command=command,
            timeout=timeout,
            rerank_query=self._last_rerank_query,
            **kwargs,
        )
