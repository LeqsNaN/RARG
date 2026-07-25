"""
Tool implementations for the TS-mirror agent.

Tools: embed_recall, bash (with rg special handling), read.

This module intentionally mirrors the pi-mono bash/read behavior closely while
still exposing simple text outputs to the Python agent loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import AgentConfig

logger = logging.getLogger(__name__)


DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
LEFT_TRUNCATION_MARKER = "[truncated]..."
RIGHT_TRUNCATION_MARKER = "...[truncated]"
MIDDLE_TRUNCATION_MARKER = "...[truncated]..."
WORD_BOUNDARY_EXPANSION_LIMIT = 10


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def _get_temp_file_path(prefix: str = "pi-bash-") -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".log")
    os.close(fd)
    return path


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text
    start = len(raw) - max_bytes
    while start < len(raw) and (raw[start] & 0xC0) == 0x80:
        start += 1
    return raw[start:].decode("utf-8", errors="replace")


def _truncate_line(line: str, max_chars: int) -> tuple[str, bool]:
    if len(line) <= max_chars:
        return line, False
    marker = RIGHT_TRUNCATION_MARKER if max_chars > len(RIGHT_TRUNCATION_MARKER) else "..."
    keep = max(0, max_chars - len(marker))
    return line[:keep] + marker, True


def _normalize_submatches(submatches: List[Dict[str, Any]], text_len: int) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    for submatch in submatches:
        start = submatch.get("start")
        end = submatch.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        start = max(0, min(start, text_len))
        end = max(start, min(end, text_len))
        if end > start:
            spans.append((start, end))

    if not spans:
        return []

    spans.sort()
    merged: List[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _cluster_spans(spans: List[tuple[int, int]], max_chars: int) -> List[tuple[int, int]]:
    if not spans:
        return []

    gap_threshold = max(16, min(120, max_chars // 8 if max_chars > 0 else 16))
    clusters: List[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = clusters[-1]
        if start - last_end <= gap_threshold:
            clusters[-1] = (last_start, end)
        else:
            clusters.append((start, end))
    return clusters


def _reduce_clusters_for_budget(clusters: List[tuple[int, int]], text_len: int, max_chars: int) -> List[tuple[int, int]]:
    reduced = list(clusters)
    min_avg_cluster_budget = 18

    while len(reduced) > 1:
        boundary_cost = 0
        if reduced[0][0] > 0:
            boundary_cost += len(LEFT_TRUNCATION_MARKER)
        if reduced[-1][1] < text_len:
            boundary_cost += len(RIGHT_TRUNCATION_MARKER)
        middle_count = len(reduced) - 1
        content_budget = max_chars - boundary_cost - middle_count * len(MIDDLE_TRUNCATION_MARKER)
        avg_cluster_budget = content_budget // len(reduced)
        if avg_cluster_budget >= min_avg_cluster_budget:
            break

        merge_idx = min(
            range(len(reduced) - 1),
            key=lambda i: reduced[i + 1][0] - reduced[i][1],
        )
        merged = (reduced[merge_idx][0], reduced[merge_idx + 1][1])
        reduced = reduced[:merge_idx] + [merged] + reduced[merge_idx + 2:]

    return reduced


def _window_around_span(start: int, end: int, text_len: int, budget: int) -> tuple[int, int]:
    span_len = end - start
    if budget <= 0:
        return start, start
    if span_len >= budget:
        return start, min(end, start + budget)

    extra = budget - span_len
    left_extra = extra // 2
    right_extra = extra - left_extra
    window_start = max(0, start - left_extra)
    window_end = min(text_len, end + right_extra)

    missing = budget - (window_end - window_start)
    if missing > 0:
        shift_left = min(missing, window_start)
        window_start -= shift_left
        missing -= shift_left
    if missing > 0:
        window_end = min(text_len, window_end + missing)

    return window_start, window_end


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _expand_to_word_boundaries(
    text: str,
    start: int,
    end: int,
    max_expand: int = WORD_BOUNDARY_EXPANSION_LIMIT,
    *,
    left_bound: int = 0,
    right_bound: Optional[int] = None,
) -> tuple[int, int]:
    if right_bound is None:
        right_bound = len(text)

    new_start = start
    new_end = end

    if left_bound < start < len(text) and _is_word_char(text[start]) and _is_word_char(text[start - 1]):
        steps = 0
        while new_start > left_bound and _is_word_char(text[new_start - 1]) and steps < max_expand:
            new_start -= 1
            steps += 1
        if new_start > left_bound and _is_word_char(text[new_start - 1]):
            new_start = start

    if 0 < end < right_bound and _is_word_char(text[end - 1]) and _is_word_char(text[end]):
        steps = 0
        while new_end < right_bound and _is_word_char(text[new_end]) and steps < max_expand:
            new_end += 1
            steps += 1
        if new_end < right_bound and _is_word_char(text[new_end]):
            new_end = end

    return new_start, new_end


def _expand_windows_to_word_boundaries(text: str, windows: List[List[int]]) -> List[List[int]]:
    expanded: List[List[int]] = []
    for idx, (start, end) in enumerate(windows):
        left_bound = 0 if idx == 0 else windows[idx - 1][1]
        right_bound = len(text) if idx == len(windows) - 1 else windows[idx + 1][0]
        new_start, new_end = _expand_to_word_boundaries(
            text,
            start,
            end,
            left_bound=left_bound,
            right_bound=right_bound,
        )
        expanded.append([new_start, new_end])
    return expanded


def _merge_touching_windows(windows: List[List[int]]) -> List[List[int]]:
    if not windows:
        return []

    merged: List[List[int]] = [[windows[0][0], windows[0][1]]]
    for start, end in windows[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


def _build_window_excerpt(text: str, start: int, end: int, budget: int) -> str:
    if budget <= 0:
        return ""

    min_marker_budget = len(LEFT_TRUNCATION_MARKER) + len(RIGHT_TRUNCATION_MARKER) + 8
    if budget < min_marker_budget:
        clipped_end = min(end, start + budget)
        expanded_start, expanded_end = _expand_to_word_boundaries(text, start, clipped_end)
        return text[expanded_start:expanded_end]

    content_budget = budget
    window_start = start
    window_end = end

    for _ in range(3):
        window_start, window_end = _window_around_span(start, end, len(text), content_budget)
        expanded_start, expanded_end = _expand_to_word_boundaries(text, window_start, window_end)
        window_start, window_end = expanded_start, expanded_end
        left_truncated = window_start > 0
        right_truncated = window_end < len(text)
        prefix = LEFT_TRUNCATION_MARKER if left_truncated else ""
        suffix = RIGHT_TRUNCATION_MARKER if right_truncated else ""
        marker_cost = len(prefix) + len(suffix)
        new_content_budget = budget - marker_cost
        if new_content_budget == content_budget or new_content_budget <= 0:
            body = text[window_start:window_end]
            return prefix + body + suffix
        content_budget = new_content_budget

    prefix = LEFT_TRUNCATION_MARKER if window_start > 0 else ""
    suffix = RIGHT_TRUNCATION_MARKER if window_end < len(text) else ""
    body = text[window_start:window_end]
    return prefix + body + suffix


def _render_windows_with_markers(text: str, windows: List[List[int]]) -> str:
    if not windows:
        return ""

    parts: List[str] = []
    first_start = windows[0][0]
    last_end = windows[-1][1]

    if first_start > 0:
        parts.append(LEFT_TRUNCATION_MARKER)

    for idx, (start, end) in enumerate(windows):
        parts.append(text[start:end])
        if idx < len(windows) - 1:
            parts.append(MIDDLE_TRUNCATION_MARKER)

    if last_end < len(text):
        parts.append(RIGHT_TRUNCATION_MARKER)

    return "".join(parts)


def _build_multi_cluster_excerpt(text: str, clusters: List[tuple[int, int]], max_chars: int) -> str:
    if not clusters or max_chars <= 0:
        return ""

    boundary_cost = 0
    if clusters[0][0] > 0:
        boundary_cost += len(LEFT_TRUNCATION_MARKER)
    if clusters[-1][1] < len(text):
        boundary_cost += len(RIGHT_TRUNCATION_MARKER)
    middle_count = max(0, len(clusters) - 1)
    available_content_budget = max_chars - boundary_cost - middle_count * len(MIDDLE_TRUNCATION_MARKER)
    min_cluster_chars = min(12, max(6, available_content_budget // max(1, len(clusters) * 3)))
    min_required_content = len(clusters) * min_cluster_chars

    if available_content_budget <= 0 or min_required_content > available_content_budget:
        return _compress_cluster_text(text, clusters[0][0], clusters[-1][1], max_chars)

    windows = [[start, end] for start, end in clusters]
    target_content_budget = available_content_budget
    current_content = sum(end - start for start, end in windows)
    windows = _expand_cluster_windows(text, windows, target_content_budget - current_content)
    windows = _expand_windows_to_word_boundaries(text, windows)
    windows = _merge_touching_windows(windows)
    return _render_windows_with_markers(text, windows)


def _compress_cluster_text(text: str, start: int, end: int, budget: int) -> str:
    if budget <= 0:
        return ""

    min_marker_budget = len(MIDDLE_TRUNCATION_MARKER) + 8
    if budget < min_marker_budget:
        return text[start:min(end, start + budget)]

    cluster_len = end - start
    if cluster_len <= budget:
        return _build_window_excerpt(text, start, end, budget)

    marker = MIDDLE_TRUNCATION_MARKER if budget > len(MIDDLE_TRUNCATION_MARKER) else "..."
    marker_len = len(marker)
    if budget <= marker_len + 2:
        return text[start:start + max(0, budget - marker_len)] + marker

    left_budget = max(1, (budget - marker_len) // 2)
    right_budget = max(1, budget - marker_len - left_budget)

    left_excerpt = _build_window_excerpt(text[start:end], 0, min(cluster_len, left_budget), left_budget)
    right_relative_start = max(0, cluster_len - right_budget)
    right_excerpt = _build_window_excerpt(text[start:end], right_relative_start, cluster_len, right_budget)

    left_excerpt = left_excerpt.removeprefix(LEFT_TRUNCATION_MARKER)
    left_excerpt = left_excerpt.removesuffix(RIGHT_TRUNCATION_MARKER)
    right_excerpt = right_excerpt.removeprefix(LEFT_TRUNCATION_MARKER)
    right_excerpt = right_excerpt.removesuffix(RIGHT_TRUNCATION_MARKER)

    return left_excerpt + marker + right_excerpt


def _expand_cluster_windows(text: str, windows: List[List[int]], remaining_budget: int) -> List[List[int]]:
    if remaining_budget <= 0 or not windows:
        return windows

    text_len = len(text)
    while remaining_budget > 0:
        progressed = False
        for idx, window in enumerate(windows):
            left_bound = 0 if idx == 0 else windows[idx - 1][1]
            right_bound = text_len if idx == len(windows) - 1 else windows[idx + 1][0]

            if remaining_budget > 0 and window[0] > left_bound:
                window[0] -= 1
                remaining_budget -= 1
                progressed = True

            if remaining_budget > 0 and window[1] < right_bound:
                window[1] += 1
                remaining_budget -= 1
                progressed = True

            if remaining_budget <= 0:
                break

        if not progressed:
            break

    return windows


def _budgeted_match_excerpt(text: str, submatches: List[Dict[str, Any]], max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False

    spans = _normalize_submatches(submatches, len(text))
    if not spans:
        return _truncate_line(text, max_chars)

    clusters = _cluster_spans(spans, max_chars)
    clusters = _reduce_clusters_for_budget(clusters, len(text), max_chars)
    cluster_lengths = [end - start for start, end in clusters]
    content_budget = max_chars
    total_cluster_len = sum(cluster_lengths)

    if total_cluster_len <= content_budget:
        return _build_multi_cluster_excerpt(text, clusters, max_chars), True

    min_per_cluster = min(24, max(8, content_budget // max(1, len(clusters) * 2)))
    allocations = [min(length, min_per_cluster) for length in cluster_lengths]
    remaining = content_budget - sum(allocations)

    while remaining > 0:
        progressed = False
        for idx in sorted(range(len(clusters)), key=lambda i: cluster_lengths[i] - allocations[i], reverse=True):
            unmet = cluster_lengths[idx] - allocations[idx]
            if unmet <= 0:
                continue
            allocations[idx] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    parts = [
        _compress_cluster_text(text, start, end, budget)
        for (start, end), budget in zip(clusters, allocations)
    ]
    excerpt = MIDDLE_TRUNCATION_MARKER.join(parts)
    if len(excerpt) > max_chars:
        return _truncate_line(excerpt, max_chars)
    return excerpt, True


def _format_rg_match(
    file_path: str,
    line_number: int,
    match_text: str,
    submatches: List[Dict[str, Any]],
    max_chars: int,
) -> tuple[str, bool]:
    display_path = file_path
    if _scope_path_dot_prefix_enabled() and isinstance(file_path, str) and file_path.startswith("./"):
        display_path = file_path[2:]
    prefix = f"{display_path}:{line_number}: "
    excerpt, was_truncated = _budgeted_match_excerpt(match_text, submatches, max_chars)
    return prefix + excerpt, was_truncated


def _scope_path_dot_prefix_enabled() -> bool:
    return os.environ.get("SCOPE_PATH_DOT_PREFIX", "").strip().lower() in {"1", "true", "yes", "on"}


def _is_scope_file_path(path_text: str) -> bool:
    text = str(path_text or "")
    return bool(re.fullmatch(r"/tmp/.+/scope_\d+\.txt", text))


def _command_mentions_scope_file(command: str) -> bool:
    return bool(re.search(r"/tmp/[^\s'\";|&()<>]+/scope_\d+\.txt", command or ""))


def _strip_scope_prefix_from_output(content: str) -> str:
    if not _scope_path_dot_prefix_enabled():
        return content
    if not content:
        return content
    lines = content.split("\n")
    normalized = []
    for line in lines:
        if line.startswith("./"):
            normalized.append(line[2:])
        else:
            normalized.append(line)
    return "\n".join(normalized)


def _truncate_head(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> Dict[str, Any]:
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return {
            "content": content,
            "truncated": False,
            "truncated_by": None,
            "total_lines": total_lines,
            "output_lines": total_lines,
            "output_bytes": total_bytes,
            "first_line_exceeds_limit": False,
        }

    if lines and len(lines[0].encode("utf-8")) > max_bytes:
        return {
            "content": "",
            "truncated": True,
            "truncated_by": "bytes",
            "total_lines": total_lines,
            "output_lines": 0,
            "output_bytes": 0,
            "first_line_exceeds_limit": True,
        }

    kept: List[str] = []
    used_bytes = 0
    truncated_by = "lines"
    for i, line in enumerate(lines[:max_lines]):
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)
        if used_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        kept.append(line)
        used_bytes += line_bytes

    output = "\n".join(kept)
    return {
        "content": output,
        "truncated": True,
        "truncated_by": truncated_by,
        "total_lines": total_lines,
        "output_lines": len(kept),
        "output_bytes": len(output.encode("utf-8")),
        "first_line_exceeds_limit": False,
    }


def _truncate_tail(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> Dict[str, Any]:
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return {
            "content": content,
            "truncated": False,
            "truncated_by": None,
            "total_lines": total_lines,
            "output_lines": total_lines,
            "output_bytes": total_bytes,
            "last_line_partial": False,
        }

    kept: List[str] = []
    used_bytes = 0
    truncated_by = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (1 if kept else 0)
        if used_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not kept:
                partial = _truncate_string_to_bytes_from_end(line, max_bytes)
                kept.insert(0, partial)
                used_bytes = len(partial.encode("utf-8"))
                last_line_partial = True
            break
        kept.insert(0, line)
        used_bytes += line_bytes
        if len(kept) >= max_lines:
            truncated_by = "lines"
            break

    output = "\n".join(kept)
    return {
        "content": output,
        "truncated": True,
        "truncated_by": truncated_by,
        "total_lines": total_lines,
        "output_lines": len(kept),
        "output_bytes": len(output.encode("utf-8")),
        "last_line_partial": last_line_partial,
    }


# =============================================================================
# Base Tool
# =============================================================================

class BaseTool(ABC):
    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    def execute(self, **kwargs) -> str:
        ...

    def to_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# =============================================================================
# embed_recall
# =============================================================================

class EmbedRecallTool(BaseTool):
    name = "embed_recall"
    description = (
        "Semantic search to recall relevant documents from the corpus. "
        "Returns a scope file path containing document paths. "
        "Use the scope file with: cat scope_N.txt | xargs -d '\\n' rg \"pattern\" "
        "to search only within recalled documents. "
        "You can call this tool multiple times with different queries to create multiple scopes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query for semantic document recall",
            },
        },
        "required": ["query"],
    }

    def __init__(self, config: AgentConfig, model_client):
        self.config = config
        self.model_client = model_client

    def execute(self, *, query: str, **kwargs) -> str:
        start = time.perf_counter()
        try:
            result = self.model_client.embed_recall(
                query=query,
                top_k=self.config.embed_top_k,
                scope_dir=self.config.scope_dir,
                scope_size_prefix=self.config.scope_size_prefix,
                sample_id=self.config.sample_id,
                paragraph_rerank_model=self.config.paragraph_rerank_model,
                paragraph_rerank_doc_limit=self.config.paragraph_rerank_doc_limit,
            )
            elapsed = (time.perf_counter() - start) * 1000
            output = result.get("output", "")
            logger.info(f"embed_recall: {result.get('count', 0)} docs [{elapsed:.0f}ms]")
            return output
        except Exception as e:
            logger.error(f"embed_recall failed: {e}")
            return f"(error: embed_recall failed: {e})"


# =============================================================================
# bash
# =============================================================================

def _inject_rg_json_flags(command: str) -> str:
    """
    Mirror of pi-mono injectRgJsonFlags().

    It intentionally stays simple:
    - strip a trailing `| head ...` / `| tail ...`
    - inject `--json --color=never`
    - force `-j1` when no thread option is present, to preserve scope order
    """
    command = re.sub(r"\|\s*(head|tail)(\s+-[^\|]*)?$", "", command)
    command = re.sub(
        r"(?<![\w/.-])rg(?=(?:\s|$))",
        "rg --json --color=never",
        command,
        count=1,
    )
    if not re.search(r"(?:\s-j\s*\d+|\s--threads\s*\d+)", command):
        command = re.sub(
            r"(?<![\w/.-])rg --json --color=never(?=(?:\s|$))",
            "rg --json --color=never -j1",
            command,
            count=1,
        )
    return command


def _extract_rg_pattern(command: str) -> str:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    after_rg = False
    consume_next = False
    option_with_arg = {
        '-e', '-f', '-g', '-m', '-M', '-A', '-B', '-C', '-j', '-t', '-T',
        '--regexp', '--file', '--glob', '--max-count', '--max-columns',
        '--after-context', '--before-context', '--context', '--threads',
        '--type', '--type-not', '--pre', '--pre-glob',
    }

    for token in tokens:
        if not after_rg:
            if token == 'rg' or token.endswith('/rg'):
                after_rg = True
            continue

        if token == '|':
            break
        if consume_next:
            consume_next = False
            continue
        if token == '--':
            continue
        if token in option_with_arg:
            consume_next = True
            continue
        if token.startswith('--'):
            continue
        if token.startswith('-'):
            if re.match(r'^-(?:[mMABCj])\d+$', token):
                continue
            if token in {'-n', '-i', '-w', '-S', '-s', '-F', '-P', '-U', '-u', '-uu', '-uuu', '-H', '-h', '-o', '-v', '-l', '-L', '-c'}:
                continue
            continue
        return token

    return ''


def _regexish_query_from_pattern(pattern: str) -> str:
    query = pattern.strip()
    if not query:
        return ''
    query = re.sub(r'\\[AbBdDsSwWZzGQEK]', ' ', query)
    query = query.replace('|', ' ')
    query = re.sub(r'\\.', ' ', query)
    query = re.sub(r'[\^$*+?()\[\]{}]', ' ', query)
    query = re.sub(r'[^0-9A-Za-z_./:-]+', ' ', query)
    tokens: List[str] = []
    seen = set()
    for token in query.split():
        cleaned = token.strip('._/:-')
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(cleaned)
    return ' '.join(tokens[:32]) or pattern.strip()


def _build_rg_rerank_query(command: str) -> str:
    pattern = _extract_rg_pattern(command)
    if not pattern:
        return command.strip()
    return _regexish_query_from_pattern(pattern)


def _extract_scope_file_from_command(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""

    for token in tokens:
        if re.search(r"(?:^|/)scope_\d+\.txt$", token):
            return token
    return ""


def _read_scope_query(scope_file: str) -> str:
    if not scope_file:
        return ""

    meta_path = Path(scope_file).with_suffix(".meta")
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""

    query = meta.get("query", "")
    return query.strip() if isinstance(query, str) else ""


def _build_combined_rg_rerank_query(command: str) -> str:
    rg_query = _build_rg_rerank_query(command)
    scope_file = _extract_scope_file_from_command(command)
    scope_query = _read_scope_query(scope_file)

    if scope_query and rg_query:
        return f"{scope_query}\n\nRG focus: {rg_query}"
    if scope_query:
        return scope_query
    return rg_query


def _use_scope_query_only_for_rg_rerank() -> bool:
    return os.environ.get("RG_RERANK_SCOPE_QUERY_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def _build_scope_only_rg_rerank_query(command: str) -> str:
    scope_file = _extract_scope_file_from_command(command)
    scope_query = _read_scope_query(scope_file)
    if scope_query:
        return scope_query
    return _build_combined_rg_rerank_query(command)


def _strip_truncation_markers_for_rerank(text: str) -> str:
    return (
        re.sub(r"(?:\.\.\.)?\[truncated\](?:\.\.\.)?", "...", text)
    )


class BashTool(BaseTool):
    name = "bash"
    accepts_explicit_rerank_query = False
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr. "
        "Output is truncated to last 2000 lines or 50KB (whichever is hit first). "
        "If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds."
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
        },
        "required": ["command"],
    }

    def __init__(self, config: AgentConfig, model_client=None):
        self.config = config
        self.model_client = model_client

    def execute(self, *, command: str, timeout: int = 0, **kwargs) -> str:
        rerank_query_override = None
        if self.accepts_explicit_rerank_query:
            rerank_query_override = kwargs.get("rerank_query") or kwargs.get("rerank_query_override")
        is_rg = self._is_rg_command(command)
        effective_command = _inject_rg_json_flags(command) if is_rg else command
        use_rg_rerank = self._use_rg_rerank()
        default_rg_timeout = self.config.rg_rerank_timeout if use_rg_rerank else 30
        effective_timeout = timeout if timeout > 0 else (default_rg_timeout if is_rg else 0)

        try:
            proc = subprocess.Popen(
                effective_command,
                shell=True,
                cwd=self.config.corpus_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        except Exception as e:
            return f"(error: failed to start command: {e})"

        assert proc.stdout is not None
        if is_rg:
            return self._process_rg(
                proc,
                effective_timeout,
                command=command,
                use_rerank=use_rg_rerank,
                rerank_query_override=rerank_query_override,
            )
        return self._process_generic(proc, effective_timeout, command=command)

    def _is_rg_command(self, command: str) -> bool:
        return bool(re.search(r"\brg\b", command or ""))

    def _use_rg_rerank(self) -> bool:
        if not self.config.rg_rerank_match or self.model_client is None:
            return False
        if self.config.rg_rerank_model in {"emb", "qwen3emb"}:
            return True
        return not self.config.no_reranker

    def _kill_proc(self, proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass

    def _process_rg(
        self,
        proc: subprocess.Popen,
        timeout: int,
        *,
        command: str,
        use_rerank: bool,
        rerank_query_override: Optional[str] = None,
    ) -> str:
        if use_rerank:
            return self._process_rg_rerank(
                proc,
                timeout,
                command=command,
                rerank_query_override=rerank_query_override,
            )
        return self._process_rg_standard(proc, timeout)

    def _process_rg_standard(self, proc: subprocess.Popen, timeout: int) -> str:
        rg_matches: List[str] = []
        rg_match_limit_reached = False
        rg_lines_truncated = False
        rg_line_buf = ""
        raw_chunks: List[bytes] = []
        raw_bytes = 0
        temp_file_path: Optional[str] = None
        timed_out = False
        read_error: Optional[Exception] = None

        def _handle_chunk(data: bytes) -> None:
            nonlocal rg_line_buf, rg_match_limit_reached, rg_lines_truncated, raw_bytes, temp_file_path
            raw_chunks.append(data)
            raw_bytes += len(data)
            if raw_bytes > DEFAULT_MAX_BYTES and temp_file_path is None:
                temp_file_path = _get_temp_file_path()
                with open(temp_file_path, "wb") as f:
                    for chunk in raw_chunks:
                        f.write(chunk)
            elif temp_file_path is not None:
                with open(temp_file_path, "ab") as f:
                    f.write(data)

            if rg_match_limit_reached:
                return

            rg_line_buf += data.decode("utf-8", errors="replace")
            lines = rg_line_buf.split("\n")
            rg_line_buf = lines.pop() or ""
            for line in lines:
                if not line.strip():
                    continue
                if len(rg_matches) >= self.config.rg_max_matches:
                    rg_match_limit_reached = True
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "match":
                    continue
                data_obj = event.get("data", {})
                file_path = data_obj.get("path", {}).get("text", "")
                line_number = data_obj.get("line_number", 0)
                match_text = data_obj.get("lines", {}).get("text", "").replace("\r\n", "\n").rstrip("\n")
                formatted, was_truncated = _format_rg_match(
                    file_path=file_path,
                    line_number=line_number,
                    match_text=match_text,
                    submatches=data_obj.get("submatches", []),
                    max_chars=self.config.grep_max_line_length,
                )
                if was_truncated:
                    rg_lines_truncated = True
                rg_matches.append(formatted)

        def _reader() -> None:
            nonlocal read_error
            assert proc.stdout is not None
            try:
                for chunk in iter(lambda: proc.stdout.read(4096), b""):
                    if not chunk:
                        break
                    _handle_chunk(chunk)
            except Exception as e:
                read_error = e

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=timeout if timeout > 0 else None)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_proc(proc)
            proc.wait()

        reader.join(timeout=2)
        if reader.is_alive():
            self._kill_proc(proc)
            reader.join(timeout=1)

        if read_error:
            logger.debug(f"bash rg stream reader error: {read_error}")

        if rg_line_buf.strip() and len(rg_matches) < self.config.rg_max_matches:
            try:
                event = json.loads(rg_line_buf)
                if event.get("type") == "match":
                    data_obj = event.get("data", {})
                    file_path = data_obj.get("path", {}).get("text", "")
                    line_number = data_obj.get("line_number", 0)
                    match_text = data_obj.get("lines", {}).get("text", "").replace("\r\n", "\n").rstrip("\n")
                    formatted, was_truncated = _format_rg_match(
                        file_path=file_path,
                        line_number=line_number,
                        match_text=match_text,
                        submatches=data_obj.get("submatches", []),
                        max_chars=self.config.grep_max_line_length,
                    )
                    if was_truncated:
                        rg_lines_truncated = True
                    rg_matches.append(formatted)
            except json.JSONDecodeError:
                pass

        if timed_out and rg_matches:
            return (
                "\n".join(rg_matches)
                + f"\n\n[rg timed out after {timeout}s, showing {len(rg_matches)} matches collected before timeout]"
            )

        if not rg_matches:
            exit_code = proc.returncode if proc.returncode is not None else 0
            if timed_out:
                return f"(error: rg timed out after {timeout}s)"
            if exit_code >= 2:
                return "(error: rg failed)"
            return "(no matches found)"

        output = "\n".join(rg_matches)
        notices: List[str] = []
        if rg_match_limit_reached:
            notices.append(f"{self.config.rg_max_matches} matches limit reached")
        if rg_lines_truncated:
            notices.append(f"some lines truncated to {self.config.grep_max_line_length} chars")
        if temp_file_path and (rg_match_limit_reached or rg_lines_truncated):
            notices.append(f"full raw output: {temp_file_path}")
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        exit_code = proc.returncode if proc.returncode is not None else 0
        if exit_code >= 2:
            return f"(error: rg failed)\n\n{output}".strip()
        return output

    def _process_rg_rerank(
        self,
        proc: subprocess.Popen,
        timeout: int,
        *,
        command: str,
        rerank_query_override: Optional[str] = None,
    ) -> str:
        candidate_limit = max(self.config.rg_max_matches, self.config.rg_rerank_candidate_num)
        rg_matches: List[Dict[str, Any]] = []
        candidate_limit_reached = False
        rg_lines_truncated = False
        rg_line_buf = ""
        raw_chunks: List[bytes] = []
        raw_bytes = 0
        temp_file_path: Optional[str] = None
        timed_out = False
        read_error: Optional[Exception] = None

        def _append_match(data_obj: Dict[str, Any]) -> None:
            nonlocal rg_lines_truncated
            file_path = data_obj.get("path", {}).get("text", "")
            display_path = file_path
            if _scope_path_dot_prefix_enabled() and isinstance(file_path, str) and file_path.startswith("./"):
                display_path = file_path[2:]
            line_number = data_obj.get("line_number", 0)
            match_text = data_obj.get("lines", {}).get("text", "").replace("\r\n", "\n").rstrip("\n")
            excerpt, was_truncated = _budgeted_match_excerpt(
                match_text,
                data_obj.get("submatches", []),
                self.config.grep_max_line_length,
            )
            if was_truncated:
                rg_lines_truncated = True
            rg_matches.append({
                "file": file_path,
                "line": line_number,
                "text": excerpt,
                "formatted": f"{display_path}:{line_number}: {excerpt}",
            })

        def _handle_chunk(data: bytes) -> None:
            nonlocal rg_line_buf, candidate_limit_reached, raw_bytes, temp_file_path
            raw_chunks.append(data)
            raw_bytes += len(data)
            if raw_bytes > DEFAULT_MAX_BYTES and temp_file_path is None:
                temp_file_path = _get_temp_file_path()
                with open(temp_file_path, "wb") as f:
                    for chunk in raw_chunks:
                        f.write(chunk)
            elif temp_file_path is not None:
                with open(temp_file_path, "ab") as f:
                    f.write(data)

            if candidate_limit_reached:
                return

            rg_line_buf += data.decode("utf-8", errors="replace")
            lines = rg_line_buf.split("\n")
            rg_line_buf = lines.pop() or ""
            for line in lines:
                if not line.strip():
                    continue
                if len(rg_matches) >= candidate_limit:
                    candidate_limit_reached = True
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "match":
                    continue
                _append_match(event.get("data", {}))
                if len(rg_matches) >= candidate_limit:
                    candidate_limit_reached = True
                    break

        def _reader() -> None:
            nonlocal read_error
            assert proc.stdout is not None
            try:
                for chunk in iter(lambda: proc.stdout.read(4096), b""):
                    if not chunk:
                        break
                    _handle_chunk(chunk)
                    if candidate_limit_reached:
                        self._kill_proc(proc)
                        break
            except Exception as e:
                read_error = e

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=timeout if timeout > 0 else None)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_proc(proc)
            proc.wait()

        reader.join(timeout=2)
        if reader.is_alive():
            self._kill_proc(proc)
            reader.join(timeout=1)

        if read_error:
            logger.debug(f"bash rg rerank stream reader error: {read_error}")

        if rg_line_buf.strip() and len(rg_matches) < candidate_limit:
            try:
                event = json.loads(rg_line_buf)
                if event.get("type") == "match":
                    _append_match(event.get("data", {}))
            except json.JSONDecodeError:
                pass

        output_limit = self.config.rg_max_matches
        reranked = False
        rerank_notice: Optional[str] = None
        selected_matches = rg_matches[:output_limit]

        if len(rg_matches) > output_limit:
            default_rerank_query = (
                _build_scope_only_rg_rerank_query(command)
                if _use_scope_query_only_for_rg_rerank()
                else _build_combined_rg_rerank_query(command)
            )
            rerank_query = (
                rerank_query_override
                or default_rerank_query
                or ""
            ).strip()
            rerank_payload = [
                {
                    "text": _strip_truncation_markers_for_rerank(m["text"]),
                    "file": m["file"],
                    "line": m["line"],
                }
                for m in rg_matches
            ]
            if rerank_query:
                try:
                    rerank_timeout = max(60, timeout)
                    ranked = self.model_client.rerank_matches(
                        rerank_query,
                        rerank_payload,
                        timeout=rerank_timeout,
                        rerank_model=self.config.rg_rerank_model,
                    )
                    ranked_indices = ranked.get("ranked_indices", [])
                    ordered: List[Dict[str, Any]] = []
                    seen = set()
                    for idx in ranked_indices:
                        if not isinstance(idx, int) or idx < 0 or idx >= len(rg_matches) or idx in seen:
                            continue
                        ordered.append(rg_matches[idx])
                        seen.add(idx)
                        if len(ordered) >= output_limit:
                            break
                    if len(ordered) < output_limit:
                        for idx, match in enumerate(rg_matches):
                            if idx in seen:
                                continue
                            ordered.append(match)
                            if len(ordered) >= output_limit:
                                break
                    selected_matches = ordered
                    reranked = True
                    rerank_notice = f"reranked top {len(selected_matches)} from {len(rg_matches)} collected matches"
                    logger.info(
                        "bash rg rerank(%s): %s candidates -> top %s [query=%r]",
                        self.config.rg_rerank_model,
                        len(rg_matches),
                        len(selected_matches),
                        rerank_query,
                    )
                except Exception as e:
                    logger.warning(f"bash rg rerank failed, using collected order: {e}")
            else:
                logger.info(
                    "bash rg rerank skipped: %s candidates collected but rerank_query missing",
                    len(rg_matches),
                )

        if timed_out and selected_matches:
            output = "\n".join(match["formatted"] for match in selected_matches)
            notices: List[str] = [f"rg timed out after {timeout}s, showing {len(selected_matches)} matches collected before timeout"]
            if reranked and rerank_notice:
                notices.append(rerank_notice)
            if candidate_limit_reached:
                notices.append(f"{candidate_limit} rerank candidates limit reached")
            if rg_lines_truncated:
                notices.append(f"some lines truncated to {self.config.grep_max_line_length} chars")
            if temp_file_path and (candidate_limit_reached or rg_lines_truncated):
                notices.append(f"full raw output: {temp_file_path}")
            return output + f"\n\n[{'. '.join(notices)}]"

        if not selected_matches:
            exit_code = proc.returncode if proc.returncode is not None else 0
            if timed_out:
                return f"(error: rg timed out after {timeout}s)"
            if exit_code >= 2:
                return "(error: rg failed)"
            return "(no matches found)"

        output = "\n".join(match["formatted"] for match in selected_matches)
        notices: List[str] = []
        if reranked and rerank_notice:
            notices.append(rerank_notice)
        if candidate_limit_reached:
            notices.append(f"{candidate_limit} rerank candidates limit reached")
        if rg_lines_truncated:
            notices.append(f"some lines truncated to {self.config.grep_max_line_length} chars")
        if temp_file_path and (candidate_limit_reached or rg_lines_truncated):
            notices.append(f"full raw output: {temp_file_path}")
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        exit_code = proc.returncode if proc.returncode is not None else 0
        if exit_code >= 2:
            return f"(error: rg failed)\n\n{output}".strip()
        return output

    def _process_generic(self, proc: subprocess.Popen, timeout: int, *, command: str = "") -> str:
        chunks: List[bytes] = []
        total_bytes = 0
        temp_file_path: Optional[str] = None
        timed_out = False

        def _handle_chunk(data: bytes) -> None:
            nonlocal total_bytes, temp_file_path
            chunks.append(data)
            total_bytes += len(data)
            if total_bytes > DEFAULT_MAX_BYTES and temp_file_path is None:
                temp_file_path = _get_temp_file_path()
                with open(temp_file_path, "wb") as f:
                    for chunk in chunks:
                        f.write(chunk)
            elif temp_file_path is not None:
                with open(temp_file_path, "ab") as f:
                    f.write(data)

        def _reader() -> None:
            assert proc.stdout is not None
            for chunk in iter(lambda: proc.stdout.read(4096), b""):
                if not chunk:
                    break
                _handle_chunk(chunk)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=timeout if timeout > 0 else None)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_proc(proc)
            proc.wait()

        reader.join(timeout=2)

        full_output = b"".join(chunks).decode("utf-8", errors="replace")
        if _scope_path_dot_prefix_enabled() and _command_mentions_scope_file(command):
            full_output = _strip_scope_prefix_from_output(full_output)
        truncation = _truncate_tail(full_output)
        output = truncation["content"] or "(no output)"

        if truncation["truncated"]:
            start_line = truncation["total_lines"] - truncation["output_lines"] + 1
            end_line = truncation["total_lines"]
            if truncation["last_line_partial"]:
                last_line = full_output.split("\n").pop() or ""
                output += (
                    f"\n\n[Showing last {_format_size(truncation['output_bytes'])} of line {end_line} "
                    f"(line is {_format_size(len(last_line.encode('utf-8')))})."
                )
            elif truncation["truncated_by"] == "lines":
                output += f"\n\n[Showing lines {start_line}-{end_line} of {truncation['total_lines']}."
            else:
                output += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation['total_lines']} "
                    f"({_format_size(DEFAULT_MAX_BYTES)} limit)."
                )
            if temp_file_path:
                output += f" Full output: {temp_file_path}]"
            else:
                output += "]"

        if timed_out:
            if output and output != "(no output)":
                output += f"\n\nCommand timed out after {timeout} seconds"
            else:
                output = f"(command timed out after {timeout}s)"

        if proc.returncode not in (0, None):
            output += f"\n\n(exit code {proc.returncode})"

        return output if output.strip() else "(no output)"


# =============================================================================
# read
# =============================================================================

class ReadTool(BaseTool):
    name = "read"
    description = (
        "Read the contents of a file. Output is truncated to 2000 lines or 50KB "
        "(whichever is hit first). Use offset/limit for large files. "
        "When you need the full file, continue with offset until complete."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative or absolute).",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, config: AgentConfig):
        self.config = config

    def execute(self, *, path: str, offset: int = 1, limit: int = 0, **kwargs) -> str:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(self.config.corpus_dir) / file_path

        if not file_path.exists():
            return f"(error: file not found: {path})"
        if not file_path.is_file():
            return f"(error: not a file: {path})"

        try:
            text_content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"(error reading file: {e})"

        all_lines = text_content.split("\n")
        total_lines = len(all_lines)
        start_idx = max(0, offset - 1)
        start_line = start_idx + 1
        if start_idx >= total_lines:
            return f"(error: offset {offset} is beyond end of file ({total_lines} lines total))"

        if limit > 0:
            end_idx = min(start_idx + limit, total_lines)
            selected = "\n".join(all_lines[start_idx:end_idx])
            user_limited_lines = end_idx - start_idx
        else:
            selected = "\n".join(all_lines[start_idx:])
            user_limited_lines = None

        if _scope_path_dot_prefix_enabled() and _is_scope_file_path(path):
            selected = _strip_scope_prefix_from_output(selected)

        truncation = _truncate_head(selected)
        if truncation["first_line_exceeds_limit"]:
            first_line_size = _format_size(len(all_lines[start_idx].encode("utf-8")))
            return (
                f"[Line {start_line} is {first_line_size}, exceeds {_format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_line}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )

        output = truncation["content"]
        if truncation["truncated"]:
            end_line = start_line + truncation["output_lines"] - 1
            next_offset = end_line + 1
            if truncation["truncated_by"] == "lines":
                output += f"\n\n[Showing lines {start_line}-{end_line} of {total_lines}. Use offset={next_offset} to continue.]"
            else:
                output += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {total_lines} "
                    f"({_format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
        elif user_limited_lines is not None and start_idx + user_limited_lines < total_lines:
            remaining = total_lines - (start_idx + user_limited_lines)
            next_offset = start_idx + user_limited_lines + 1
            output += f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"

        return output


# =============================================================================
# Registry
# =============================================================================

def build_tools(config: AgentConfig, model_client) -> List[BaseTool]:
    bash_tool_cls = BashTool
    if config.use_contextual_bash_tool:
        from contextual_bash_tool import RerankAwareBashTool

        bash_tool_cls = RerankAwareBashTool

    return [
        EmbedRecallTool(config, model_client),
        bash_tool_cls(config, model_client),
        ReadTool(config),
    ]
