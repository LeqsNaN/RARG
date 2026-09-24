"""Python ports of DR-DCI's main-path read, bash, and pull tools.

Behavioral sources are respectively Pi's ``read.ts``, ``bash.ts``,
``truncate.ts``, and ``pull.ts``.  This module intentionally has no dependency
on ``ts_mirror_agent``.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_MAX_LINES = 2_000
DEFAULT_MAX_BYTES = 10 * 1024
DEFAULT_CHAR_WINDOW = 4_096
DEFAULT_BYTE_WINDOW = 4_096

def _bash_max_line_length() -> int:
    return int(os.environ.get("DCI_BASH_MAX_LINE_CHARS", "1500"))


def _bash_long_match_snippet_chars() -> int:
    return int(os.environ.get("DCI_BASH_LONG_MATCH_SNIPPET_CHARS", "1000"))


def _bash_long_match_read_window_chars() -> int:
    return int(os.environ.get("DCI_BASH_LONG_MATCH_READ_WINDOW_CHARS", "1600"))


def _bash_default_timeout_seconds() -> int:
    raw = os.environ.get("DCI_BASH_DEFAULT_TIMEOUT_SECONDS", "")
    try:
        parsed = float(raw)
        if parsed > 0:
            return int(parsed)
    except ValueError:
        pass
    return 30


def _format_size(value: int) -> str:
    if value < 1024:
        return f"{value}B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f}KB"
    return f"{value / (1024 * 1024):.1f}MB"


@dataclass
class Truncation:
    content: str
    truncated: bool
    truncated_by: Optional[str]
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool


def truncate_head(content: str) -> Truncation:
    """Port of DR-DCI ``truncateHead`` (line/UTF-8 limits are identical)."""
    raw = content.encode("utf-8")
    lines = content.split("\n")
    if len(lines) <= DEFAULT_MAX_LINES and len(raw) <= DEFAULT_MAX_BYTES:
        return Truncation(content, False, None, len(lines), len(raw), len(lines), len(raw), False, False)
    if len(lines[0].encode("utf-8")) > DEFAULT_MAX_BYTES:
        return Truncation("", True, "bytes", len(lines), len(raw), 0, 0, False, True)
    kept: List[str] = []
    used = 0
    reason = "lines"
    for idx, line in enumerate(lines[:DEFAULT_MAX_LINES]):
        size = len(line.encode("utf-8")) + (1 if idx else 0)
        if used + size > DEFAULT_MAX_BYTES:
            reason = "bytes"
            break
        kept.append(line)
        used += size
    output = "\n".join(kept)
    return Truncation(output, True, reason, len(lines), len(raw), len(kept), len(output.encode("utf-8")), False, False)


def _truncate_bytes_from_end(value: str, maximum: int) -> str:
    raw = value.encode("utf-8")
    if len(raw) <= maximum:
        return value
    start = max(0, len(raw) - maximum)
    while start < len(raw) and raw[start] & 0xC0 == 0x80:
        start += 1
    return raw[start:].decode("utf-8", errors="replace")


def truncate_tail(content: str) -> Truncation:
    """Port of DR-DCI ``truncateTail``."""
    raw = content.encode("utf-8")
    lines = content.split("\n")
    if len(lines) <= DEFAULT_MAX_LINES and len(raw) <= DEFAULT_MAX_BYTES:
        return Truncation(content, False, None, len(lines), len(raw), len(lines), len(raw), False, False)
    kept: List[str] = []
    used = 0
    partial = False
    reason = "lines"
    for line in reversed(lines):
        if len(kept) >= DEFAULT_MAX_LINES:
            break
        size = len(line.encode("utf-8")) + (1 if kept else 0)
        if used + size > DEFAULT_MAX_BYTES:
            reason = "bytes"
            retained = len("\n".join(kept).encode("utf-8"))
            if not kept or retained == 0:
                kept.insert(0, _truncate_bytes_from_end(line, DEFAULT_MAX_BYTES - retained))
                partial = True
            break
        kept.insert(0, line)
        used += size
    output = "\n".join(kept)
    return Truncation(output, True, reason, len(lines), len(raw), len(kept), len(output.encode("utf-8")), partial, False)


def _safe_relative(value: str) -> Optional[str]:
    normalized = posixpath.normpath(value.replace("\\", "/"))
    if not normalized or normalized == "." or normalized.startswith("/"):
        return None
    # Match Node's path.normalize guard in pull.ts.  The original deliberately
    # rejects every normalized path beginning with "..", including a benign
    # looking filename such as "..notes.txt".
    if normalized.startswith(".."):
        return None
    return normalized


def _safe_filename(path: str) -> str:
    name = Path(path).name
    stem, ext = os.path.splitext(name)
    clean_stem = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:96] or "document"
    clean_ext = re.sub(r"[^a-z0-9]+", "_", ext.lstrip(".").lower()).strip("_")[:16]
    return f"{clean_stem}.{clean_ext}" if clean_ext else clean_stem


def _is_inside(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _clamp_long_lines(content: str, command: str) -> Tuple[str, int]:
    """Main behavior of DR-DCI ``clampLongLines`` for text bash output."""
    if os.environ.get("DCI_DISABLE_BASH_LINE_CLAMP") == "1":
        return content, 0
    max_line_length = _bash_max_line_length()
    clipped = 0
    output: List[str] = []
    for line in content.split("\n"):
        if len(line) <= max_line_length:
            output.append(line)
            continue
        clipped += 1
        match = re.match(r"^(.+?):(\d+):\s?(.*)$", line)
        if not match:
            half = max_line_length // 2
            output.append(f"{line[:half]}... [line truncated, {len(line) - max_line_length} chars omitted] ...{line[-half:]}")
            continue
        path, line_number, text = match.groups()
        terms = re.findall(r"['\"]([^'\"]+)['\"]", command)
        offset = next((text.lower().find(term.lower()) for term in terms if text.lower().find(term.lower()) >= 0), 0)
        snippet_chars = _bash_long_match_snippet_chars()
        read_window_chars = _bash_long_match_read_window_chars()
        snippet_start = max(0, offset - snippet_chars // 2)
        snippet_end = min(len(text), snippet_start + snippet_chars)
        snippet = ("..." if snippet_start else "") + text[snippet_start:snippet_end] + ("..." if snippet_end < len(text) else "")
        if int(line_number) == 1:
            read_start = max(0, offset - read_window_chars // 4)
            hint = f'read={{"path":{json.dumps(path)},"charOffset":{read_start},"charLimit":{read_window_chars}}}'
        else:
            hint = f'read={{"path":{json.dumps(path)},"offset":{line_number},"limit":20}}'
        output.append(f"{path}:{line_number}: {snippet} [long line clipped; lineChars={len(text)}; {hint}]")
    return "\n".join(output), clipped


def _reflow_long_line_text(text: str, width: int) -> str:
    """Port of pull.ts ``reflowLongLineText`` used by the main launcher."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks = re.sub(r"([.!?。！？；;])\s+", r"\1\n", normalized).split("\n")
    lines: List[str] = []
    for chunk in chunks:
        trimmed = chunk.strip()
        if not trimmed:
            continue
        lines.extend(trimmed[offset:offset + width] for offset in range(0, len(trimmed), width))
    return "\n".join(lines) + "\n"


def _materialize_file(source: Path, target: Path) -> None:
    """Port of pull.ts materializeFile for hardlink/reflow main-path mode."""
    reflow = os.environ.get("DCI_REFLOW_SINGLE_LINE_TEXT", "").lower() in {"1", "true", "yes"}
    wrap = os.environ.get("DCI_WRAP_LONG_TEXT_LINES", "").lower() in {"1", "true", "yes"}
    if not reflow and not wrap:
        os.link(source, target)
        return
    text = source.read_text(encoding="utf-8", errors="replace")
    if wrap:
        width = int(os.environ.get("DCI_WRAP_LONG_TEXT_LINE_WIDTH", "2000"))
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        target.write_text("\n".join(line[offset:offset + width] for line in normalized.split("\n") for offset in range(0, len(line), width)), encoding="utf-8")
        return
    first = text.find("\n")
    second = text.find("\n", first + 1) if first >= 0 else -1
    min_bytes = int(os.environ["DCI_REFLOW_SINGLE_LINE_MIN_BYTES"]) if os.environ.get("DCI_REFLOW_SINGLE_LINE_MIN_BYTES") else None
    if (first >= 0 and second >= 0) or (min_bytes is not None and len(text.encode("utf-8")) < min_bytes):
        os.link(source, target)
        return
    width = int(os.environ.get("DCI_REFLOW_SINGLE_LINE_WIDTH", "1200"))
    target.write_text(_reflow_long_line_text(text, width), encoding="utf-8")


def _sanitize_bash_output(value: str) -> str:
    """Port the text-relevant part of Pi sanitizeBinaryOutput + stripAnsi."""
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value).replace("\r", "")
    return "".join(
        char for char in value
        if char in "\t\n" or (ord(char) > 0x1F and not 0xFFF9 <= ord(char) <= 0xFFFB)
    )


class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]

    def openai_schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def execute(self, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal TypeBox-equivalent validation for the fixed main-path tools."""
        return args


class ReadTool(Tool):
    name = "read"
    description = (
        "Read the contents of a file. Supports text files. For text files, output is truncated to 2000 lines or 10KB "
        "(whichever is hit first). Use offset/limit for normal multi-line files. If output is clipped or truncated, "
        "use charOffset/charLimit to inspect a small window."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "number", "description": "Maximum number of lines to read"},
            "charOffset": {"type": "number", "description": "Character offset for a window inside a long text file"},
            "charLimit": {"type": "number", "description": "Maximum characters for charOffset; default 4096"},
            "byteOffset": {"type": "number", "description": "Byte offset for a bounded window"},
            "byteLimit": {"type": "number", "description": "Maximum bytes for byteOffset; default 4096"},
        },
        "required": ["path"],
    }

    def __init__(self, cwd: Path):
        self.cwd = cwd

    def validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args.get("path"), str):
            raise ValueError("Invalid tool arguments: path must be a string")
        for key in ("offset", "limit", "charOffset", "charLimit", "byteOffset", "byteLimit"):
            if key in args and not isinstance(args[key], (int, float)):
                raise ValueError(f"Invalid tool arguments: {key} must be a number")
        return args

    def execute(self, *, path: str, offset: Any = None, limit: Any = None, charOffset: Any = None, charLimit: Any = None, byteOffset: Any = None, byteLimit: Any = None, **_: Any) -> Tuple[str, Dict[str, Any]]:
        # resolveReadPath() in Pi expands a leading home-directory marker
        # before resolving relative paths against cwd.
        target = Path(os.path.expanduser(path))
        if not target.is_absolute():
            target = self.cwd / target
        if not target.is_file():
            raise RuntimeError(f"File not found or not readable: {path}")
        buffer = target.read_bytes()
        text = buffer.decode("utf-8", errors="replace")
        if byteOffset is not None or byteLimit is not None:
            start = max(0, int(byteOffset or 0))
            if start >= len(buffer):
                raise RuntimeError(f"byteOffset {start} is beyond end of file ({len(buffer)} bytes total)")
            end = min(len(buffer), start + max(1, min(DEFAULT_MAX_BYTES, int(byteLimit or DEFAULT_BYTE_WINDOW))))
            result = buffer[start:end].decode("utf-8", errors="replace")
            if end < len(buffer): result += f"\n\n[Showing bytes {start}-{end} of {len(buffer)}. Use byteOffset={end} to continue.]"
            return result, {}
        if charOffset is not None or charLimit is not None:
            start = max(0, int(charOffset or 0))
            if start >= len(text):
                raise RuntimeError(f"charOffset {start} is beyond end of file ({len(text)} chars total)")
            end = min(len(text), start + max(1, min(DEFAULT_MAX_BYTES, int(charLimit or DEFAULT_CHAR_WINDOW))))
            result = text[start:end]
            if end < len(text): result += f"\n\n[Showing chars {start}-{end} of {len(text)}. Use charOffset={end} to continue.]"
            return result, {}
        lines = text.split("\n")
        start = max(0, int(offset or 1) - 1)
        if start >= len(lines):
            raise RuntimeError(f"Offset {offset} is beyond end of file ({len(lines)} lines total)")
        if limit is not None:
            selected = "\n".join(lines[start:min(len(lines), start + int(limit))])
            user_limited = min(len(lines), start + int(limit)) - start
        else:
            selected, user_limited = "\n".join(lines[start:]), None
        trunc = truncate_head(selected)
        if trunc.first_line_exceeds_limit:
            end = min(len(text), (0 if start == 0 else len("\n".join(lines[:start])) + 1) + DEFAULT_CHAR_WINDOW)
            begin = 0 if start == 0 else len("\n".join(lines[:start])) + 1
            return text[begin:end] + (f"\n\n[chars {begin}-{end}/{len(text)}; next charOffset={end}]" if end < len(text) else ""), {"truncation": trunc.__dict__}
        output = trunc.content
        if trunc.truncated:
            end_line = start + trunc.output_lines
            if trunc.truncated_by == "lines":
                output += f"\n\n[Showing lines {start + 1}-{end_line} of {len(lines)}. Use offset={end_line + 1} to continue.]"
            else:
                output += f"\n\n[Showing lines {start + 1}-{end_line} of {len(lines)} ({_format_size(DEFAULT_MAX_BYTES)} limit). Use offset={end_line + 1} to continue.]"
        elif user_limited is not None and start + user_limited < len(lines):
            output += f"\n\n[{len(lines) - start - user_limited} more lines in file. Use offset={start + user_limited + 1} to continue.]"
        return output, {"truncation": trunc.__dict__} if trunc.truncated else {}


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr. Commands are capped by the "
        "harness timeout. Output is truncated to last 2000 lines or 10KB (whichever is hit first), and individual long "
        "lines are shortened to the configured DCI_BASH_MAX_LINE_CHARS limit. If output is truncated, refine the command or use read with offset/charOffset."
    )
    parameters = {"type": "object", "properties": {"command": {"type": "string", "description": "Bash command to execute"}}, "required": ["command"]}

    def __init__(self, cwd: Path, timeout_seconds: Optional[int] = None):
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else _bash_default_timeout_seconds()

    def validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args.get("command"), str):
            raise ValueError("Invalid tool arguments: command must be a string")
        return args

    def execute(self, *, command: str, **_: Any) -> Tuple[str, Dict[str, Any]]:
        if os.environ.get("DCI_BASH_BLOCK_NETWORK") == "1" and re.search(r"\b(curl|wget|ssh|scp)\b|https?://", command, re.I):
            return "Network access is disabled for bash in this isolated environment. Use pull(query) for corpus retrieval; use bash only on local files.", {}
        started = time.perf_counter()
        if not self.cwd.exists():
            raise RuntimeError(f"Working directory does not exist: {self.cwd}\nCannot execute bash commands.")
        process = subprocess.Popen(
            ["/bin/bash", "-c", command], cwd=self.cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy(),
        )
        try:
            stdout, _ = process.communicate(timeout=self.timeout_seconds)
            raw = _sanitize_bash_output(stdout.decode("utf-8", errors="replace"))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            stdout, _ = process.communicate()
            raw = _sanitize_bash_output(stdout.decode("utf-8", errors="replace")) + f"\n\nCommand timed out after {self.timeout_seconds} seconds"
            process = None
        clamped, clipped = _clamp_long_lines(raw, command)
        trunc = truncate_tail(clamped)
        full_path: Optional[str] = None
        if len(raw.encode("utf-8")) > DEFAULT_MAX_BYTES or clipped:
            fd, full_path = tempfile.mkstemp(prefix="pi-bash-", suffix=".log")
            with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(raw)
        output = trunc.content or "(no output)"
        if clipped:
            output += f"\n\n[{clipped} long line(s) clipped; full={full_path}]"
        if trunc.truncated:
            start_line = trunc.total_lines - trunc.output_lines + 1
            if trunc.last_line_partial:
                output += f"\n\n[Showing last {_format_size(trunc.output_bytes)} of line {trunc.total_lines}. Full output: {full_path}]"
            elif trunc.truncated_by == "lines":
                output += f"\n\n[Showing lines {start_line}-{trunc.total_lines} of {trunc.total_lines}. Full output: {full_path}]"
            else:
                output += f"\n\n[Showing lines {start_line}-{trunc.total_lines} of {trunc.total_lines} ({_format_size(DEFAULT_MAX_BYTES)} limit). Full output: {full_path}]"
        if process is None:
            raise RuntimeError(output)
        if process.returncode:
            raise RuntimeError(output + f"\n\nCommand exited with code {process.returncode}")
        return output, {"truncation": trunc.__dict__ if trunc.truncated else None, "fullOutputPath": full_path, "duration_seconds": time.perf_counter() - started}


class PullTool(Tool):
    """Port of original main-path ``pull.ts``: root_flat_disclosed only."""
    name = "pull"
    description = "Pull semantically relevant documents into the visible workspace. Accepts one query and topK 300-600; returns a short ranked preview of newly added documents."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "description": "One concise lexical query. Call pull again for a different clue."},
            "topK": {"type": "integer", "minimum": 300, "maximum": 600, "description": "Required. Number of documents to retrieve. Choose 300-600."},
        },
        "required": ["query", "topK"],
    }

    def __init__(self, workspace_dir: Path, source_root: Path, retriever: Any, *, meta_root: Optional[Path] = None, preview_limit: int = 20, max_calls: int = 0):
        self.workspace_dir, self.source_root, self.retriever = workspace_dir, source_root, retriever
        self.meta_root = meta_root or (workspace_dir / ".dci_pull_meta")
        self.preview_limit, self.max_calls = preview_limit, max_calls

    def validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(args.get("query"), str) or not args["query"].strip():
            raise ValueError("Invalid tool arguments: query must be a non-empty string")
        # TypeBox Integer rejects floats and missing topK before tool execution.
        top_k = args.get("topK")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("Invalid tool arguments: topK must be an integer")
        if not 300 <= top_k <= 600:
            raise ValueError("Invalid tool arguments: topK must be between 300 and 600")
        return args

    def _previous_paths(self) -> Set[str]:
        paths: Set[str] = set()
        if not self.meta_root.exists(): return paths
        for child in self.meta_root.iterdir():
            managed = child / "managed_paths.json"
            if managed.is_file():
                try: paths.update(p for p in json.loads(managed.read_text(encoding="utf-8")) if isinstance(p, str))
                except (OSError, ValueError): pass
        return paths

    def _next_index(self) -> int:
        if not self.meta_root.exists(): return 1
        values = [int(item.name[5:]) for item in self.meta_root.iterdir() if item.is_dir() and re.fullmatch(r"pull_\d+", item.name)]
        return max(values, default=0) + 1

    def _target_path(self, safe_path: str) -> Path:
        # Exact root_flat_disclosed mapping in pull.ts: it uses only the safe
        # basename.  A collision consequently fails materialization (rather
        # than assigning a Python-only suffix) and is recorded as missing.
        return self.workspace_dir / _safe_filename(safe_path)

    def execute(self, *, query: str, topK: int, **_: Any) -> Tuple[str, Dict[str, Any]]:
        query = query.strip()
        if not query: raise RuntimeError("A non-empty query string is required")
        if not 300 <= int(topK) <= 600: raise RuntimeError("topK must be between 300 and 600")
        source_abs = Path(os.path.abspath(self.source_root))
        view_abs = Path(os.path.abspath(self.workspace_dir))
        if _is_inside(source_abs, view_abs) or _is_inside(view_abs, source_abs):
            raise RuntimeError("Pull viewDir must be separate from DCI_PULL_SOURCE_ROOT")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.meta_root.mkdir(parents=True, exist_ok=True)
        pull_index = self._next_index()
        if self.max_calls and pull_index > self.max_calls: raise RuntimeError(f"pull call limit reached ({self.max_calls}). Do not retrieve more documents. Search/read the existing workspace and answer with the available evidence.")
        meta_dir = self.meta_root / f"pull_{pull_index}"
        meta_dir.mkdir()
        recalled = self.retriever.recall(query, int(topK), str(meta_dir))
        scope_file = Path(str(recalled.get("scope_file", "")))
        if not scope_file.is_file(): raise RuntimeError("Pull retriever did not return a readable scope file")
        existing, created_this_call = self._previous_paths(), set()
        docs: List[Dict[str, Any]] = []
        missing = already_visible = 0
        for rank, raw in enumerate(scope_file.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            safe = _safe_relative(raw)
            if not safe: missing += 1; continue
            if safe in existing: already_visible += 1; continue
            if safe in created_this_call: continue
            source = self.source_root / safe
            if not source.is_file(): missing += 1; continue
            target = self._target_path(safe)
            try:
                _materialize_file(source, target)
            except OSError:
                missing += 1
                continue
            created_this_call.add(safe)
            docs.append({
                "sourcePath": safe,
                "workspacePath": target.name,
                "rank": rank,
                # Retriever score transport is intentionally adapter-specific,
                # but score is not exposed in the main-path pull text.
                "score": 0.0,
                # pull.ts uses /\\.txt$/i, not a case-sensitive suffix test.
                "title": re.sub(r"\.txt$", "", _safe_filename(safe), flags=re.I),
            })
        managed = meta_dir / "managed_paths.json"
        managed.write_text(json.dumps(sorted(created_this_call), ensure_ascii=False, indent=2), encoding="utf-8")
        details = {
            "toolKind": "pull", "queries": [query], "topK": int(topK), "viewMode": "hardlink", "layout": "root",
            "materializationMode": "root_flat_disclosed", "previewMode": "ranked", "viewDir": str(self.workspace_dir),
            "pullIndex": pull_index, "pullDir": str(self.workspace_dir), "workspaceDir": ".", "managedPathsPath": str(managed),
            "sourceDocumentCount": len(created_this_call), "materializedDocumentCount": len(docs), "missingDocumentCount": missing,
            "alreadyVisibleDocumentCount": already_visible, "topNewDocuments": docs[:self.preview_limit],
            "perQueryHitCounts": {query: int(recalled.get("count", 0))}, "queryDirs": {query: "."},
        }
        lines = ["Workspace root expanded.", f"New documents added: {len(docs)}. Already visible from previous pulls: {already_visible}.", "Top newly added documents by retrieval rank:"]
        lines.extend(f"- #{doc['rank']} {doc['workspacePath']} ({doc['title']})" for doc in docs[:self.preview_limit])
        if not docs: lines.append("- none")
        lines.append("Search/read the workspace root with local tools. Ranks are shown here only; filenames are not rank-prefixed.")
        return "\n".join(lines), details


def build_main_tools(
    workspace_dir: Path,
    source_root: Path,
    retriever: Any,
    *,
    pull_meta_dir: Optional[Path] = None,
    preview_limit: int = 20,
    max_pull_calls: int = 0,
) -> List[Tool]:
    return [
        ReadTool(workspace_dir),
        BashTool(workspace_dir),
        PullTool(
            workspace_dir,
            source_root,
            retriever,
            meta_root=pull_meta_dir,
            preview_limit=preview_limit,
            max_calls=max_pull_calls,
        ),
    ]
