"""Verbatim main-path DR-DCI rank-aware prompt construction port.

Source: DR-DCI/scripts/bcplus_eval/run_bcplus_eval.py,
build_rank_aware_pull_prompt(), constrained to the paper's main setup:
root + root_flat_disclosed + ranked preview + one query per pull.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def build_pi_system_prompt(cwd: str, *, current_date: str | None = None) -> str:
    """Port of Pi ``buildSystemPrompt`` for the selected read,bash,pull tools.

    The benchmark prompt remains a *user* message, exactly as it is passed by
    DR-DCI's harness to dci-agent-lite.
    """
    repo_root = Path(__file__).resolve().parents[3]
    # getPackageDir() in Pi resolves to the coding-agent package passed through
    # --package-dir, not the pi-mono repository root.
    package_dir = repo_root / "DR-DCI" / "pi-mono" / "packages" / "coding-agent"
    date = current_date or datetime.now(timezone.utc).date().isoformat()
    snippets = (
        "- read: Read file contents\n"
        "- bash: Execute bash commands (ls, grep, find, etc.)\n"
        "- pull: pull retrieves semantically relevant documents from the hidden corpus into the visible workspace. It accepts one query string per call and required topK 300-600. Documents are stored directly in the workspace root. Retrieval ranks are shown in the tool result, not encoded in filenames."
    )
    guidelines = (
        "- Use bash for file operations like ls, rg, find\n"
        "- Use read to examine files instead of cat or sed.\n"
        "- If output is clipped or truncated, continue with offset or charOffset windows.\n"
        "- The visible workspace starts empty; pull adds documents from the hidden corpus.\n"
        "- The query parameter is a single string. topK is required; choose topK between 300 and 600 for each call.\n"
        "- Rank-aware mode accepts one query string and topK; call pull again for a different clue.\n"
        "- The tool result shows retrieval ranks for newly added documents; lower numbers are more similar.\n"
        "- Each pull call adds new documents directly to the workspace root.\n"
        "- pull is not evidence. Final answers must come from document text actually searched or read in the workspace.\n"
        "- Be concise in your responses\n"
        "- Show file paths clearly when working with files"
    )
    return (
        "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.\n"
        "\nAvailable tools:\n"
        f"{snippets}\n"
        "\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n"
        "\nGuidelines:\n"
        f"{guidelines}\n"
        "\nPi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):\n"
        f"- Main documentation: {package_dir / 'README.md'}\n"
        f"- Additional docs: {package_dir / 'docs'}\n"
        f"- Examples: {package_dir / 'examples'} (extensions, custom tools, SDK)\n"
        "- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), pi packages (docs/packages.md)\n"
        "- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing\n"
        "- Always read pi .md files completely and follow links to related docs (e.g., tui.md for TUI API details)\n"
        f"Current date: {date}\n"
        f"Current working directory: {cwd.replace(chr(92), '/') }"
    )


def build_main_prompt(query: str, corpus_ref: str = "corpus") -> str:
    """Return the original DR-DCI main-experiment prompt text."""
    return (
        "You are a deep research agent answering a question using only the visible workspace and tools.\n"
        "\n"
        "Workspace and pull:\n"
        "- The full corpus is hidden and massive. The visible workspace starts empty.\n"
        "- pull(query, topK) retrieves semantically relevant documents from the hidden corpus into the visible workspace. Use one concise query per call. topK is required; choose topK between 300 and 600.\n"
        "- Each pull call adds newly retrieved files directly into the current workspace root. There are no pull_N folders to choose between.\n"
        "- pull returns a short ranked preview of newly added documents. The workspace filenames are safe slugs and do not include rank prefixes; use the returned ranks as navigation hints.\n"
        f"- The current working directory is the visible workspace. Use relative paths such as ./filename.txt in terminal commands. Use @{corpus_ref}/relative_path only for final citations.\n"
        "\n"
        "Workflow:\n"
        "1. Use pull with one concise lexical query string based on the original question. Always provide topK between 300 and 600 based on clue breadth.\n"
        "2. Prefer short clue/entity/title/date queries over long natural-language rewrites.\n"
        "3. After every pull, stop pulling and search/read the current workspace locally. Use the ranked preview returned by pull to prioritize newly added documents.\n"
        "4. Use rg/find/ls to screen candidates, then read promising documents with read.\n"
        "5. If output is clipped or truncated, use the suggested read offset/charOffset window to inspect only the relevant region.\n"
        "6. Use another pull only when it adds a genuinely new clue from what you already saw. Do not use pull as the first response to ordinary uncertainty; first search/read the workspace you just built.\n"
        "7. As soon as you have enough evidence, stop using tools and answer.\n"
        f"8. Final answer must cite documents actually read from @{corpus_ref} paths.\n"
        "9. Your final response must use exactly this format:\n"
        f"Explanation: {{your explanation for your final answer. Cite supporting documents inline as [@{corpus_ref}/relative_path] at the end of sentences when possible.}}\n"
        "Exact Answer: {your succinct, final answer}\n"
        "Confidence: {your confidence score between 0% and 100%}\n"
        "10. If you later receive a user steer telling you to submit now, stop using tools immediately and answer right away with the exact final response format below. Do not do more research after that steer.\n"
        "11. Keep Exact Answer concise and directly responsive to the question.\n"
        f"Question: {query}\n"
    )
