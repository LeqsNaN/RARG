#!/usr/bin/env python3
"""Create a bc_plus_1m corpus from BC+ docs plus sampled FineWeb docs.

The preferred path is to reconstruct the original BC+ document folder layout
directly from ``corpus/browsecomp_plus`` using the same exporter used by the
standard BC+ setup, then append the sampled FineWeb distractors into the same
output corpus.

Usage:
    python scripts/bcplus_eval/create_bcplus_1m_corpus.py \
        --source-browsecomp corpus/browsecomp_plus \
        --fineweb-sample data/fineweb_edu_10bt_sample900k_min8000.jsonl \
        --output corpus/bc_plus_1m \
        --build-index \
        --index-output-dir data/indices/bc_plus_1m
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_EMBED_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "Qwen3-Embedding-4B")
TITLE_RE = re.compile(r"(?mi)^title:\s*(.+?)\s*$")
INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WHITESPACE_RE = re.compile(r"\s+")
MAX_STEM_LEN = 140


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a standalone bc_plus_1m corpus")
    parser.add_argument(
        "--source-browsecomp",
        type=Path,
        default=None,
        help="BrowseComp-Plus parquet dir, e.g. corpus/browsecomp_plus",
    )
    parser.add_argument(
        "--source-original",
        type=Path,
        default=None,
        help="Legacy fallback: pre-exported BC+ corpus dir, e.g. corpus/bc_plus_100k",
    )
    parser.add_argument("--fineweb-sample", type=Path, required=True, help="Sampled FineWeb JSONL path")
    parser.add_argument("--output", type=Path, required=True, help="Output corpus dir, e.g. corpus/bc_plus_1m")
    parser.add_argument("--fineweb-prefix", default="", help="Optional prefix for exported FineWeb filenames (default: none)")
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing output directory")
    parser.add_argument("--clean-output", action="store_true", help="If set together with --force, remove existing files under --output before rebuilding")
    parser.add_argument("--build-index", action="store_true", help="Also build a FAISS embedding index after corpus creation")
    parser.add_argument("--index-output-dir", type=Path, default=None, help="Index output dir; required when --build-index is set")
    parser.add_argument("--clean-index-output", action="store_true", help="If set, delete the existing index output dir before rebuilding the index")
    parser.add_argument("--embed-python", type=str, default=sys.executable, help="Python executable for build_embedding_index.py")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL_PATH, help="Embedding model path for index building")
    parser.add_argument("--index-max-chars", type=int, default=0, help="Passed to build_embedding_index.py --max-chars")
    parser.add_argument("--index-max-model-len", type=int, default=4096, help="Passed to build_embedding_index.py --max-model-len")
    parser.add_argument("--index-batch-size", type=int, default=64, help="Passed to build_embedding_index.py --batch-size")
    parser.add_argument("--index-tensor-parallel-size", type=int, default=1, help="Passed to build_embedding_index.py --tensor-parallel-size")
    parser.add_argument("--index-gpu-memory-utilization", type=float, default=0.9, help="Passed to build_embedding_index.py --gpu-memory-utilization")
    return parser.parse_args()


def extract_domain(url: str) -> str:
    try:
        hostname = urlparse(url).hostname or "unknown-domain"
        return sanitize_name(hostname.lower(), "unknown-domain")
    except Exception:
        return ""


def sanitize_doc_id(raw_id: object, fallback_index: int) -> str:
    text = str(raw_id) if raw_id is not None else f"{fallback_index:06d}"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or f"{fallback_index:06d}"


def extract_title(text: str) -> str | None:
    match = TITLE_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def sanitize_name(value: str, fallback: str) -> str:
    value = INVALID_CHARS_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip().strip(".")
    return value or fallback


def build_filename(title: str | None, url: str, docid: str, prefix: str) -> str:
    parsed = urlparse(url)
    path_name = Path(parsed.path).name
    fallback = path_name or f"doc-{docid}"
    stem = title or fallback
    stem = sanitize_name(stem, f"doc-{docid}")
    if len(stem) > MAX_STEM_LEN:
        stem = stem[:MAX_STEM_LEN].rstrip(" .")
    if not stem:
        stem = f"doc-{docid}"
    if prefix:
        stem = f"{prefix}{stem}"
    return f"{stem}.txt"


def unique_path(path: Path, docid: str, text: str) -> Path:
    if not path.exists():
        return path
    try:
        if path.read_text(encoding="utf-8") == text:
            return path
    except OSError:
        pass
    stem = path.stem
    suffix = path.suffix
    candidate = path.with_name(f"{stem}__docid_{docid}{suffix}")
    if not candidate.exists():
        return candidate
    try:
        if candidate.read_text(encoding="utf-8") == text:
            return candidate
    except OSError:
        pass
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}__docid_{docid}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        try:
            if candidate.read_text(encoding="utf-8") == text:
                return candidate
        except OSError:
            pass
        counter += 1


def prepare_dir(path: Path, force: bool, clean: bool, *, label: str) -> None:
    if path.exists():
        has_entries = any(path.iterdir())
        if has_entries and not force:
            raise SystemExit(f"{label} directory {path} already exists and is not empty; pass --force to continue")
        if has_entries and clean:
            print(f"Cleaning existing {label} directory: {path}", flush=True)
            for child in path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def copy_original_docs(source: Path, output: Path) -> int:
    copied = 0
    for domain_dir in sorted(source.iterdir()):
        if not domain_dir.is_dir():
            continue
        out_domain = output / domain_dir.name
        out_domain.mkdir(parents=True, exist_ok=True)
        for src_file in domain_dir.iterdir():
            if not src_file.is_file():
                continue
            dst_file = out_domain / src_file.name
            if dst_file.exists():
                continue
            shutil.copy2(src_file, dst_file)
            copied += 1
            if copied % 10000 == 0:
                print(f"Copied {copied} original docs...", flush=True)
    return copied


def export_original_docs_from_browsecomp(source: Path, output: Path) -> int | None:
    cmd = [
        sys.executable,
        "-m",
        "dci.benchmark.export_bc_plus_docs",
        "--source-dir",
        str(source),
        "--output-dir",
        str(output),
    ]
    print("Exporting original BC+ docs with dci.benchmark.export_bc_plus_docs...", flush=True)
    print("  Command: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return None


def export_fineweb_docs(sample_path: Path, output: Path, prefix: str) -> tuple[int, int]:
    written = 0
    skipped = 0
    seen_paths: set[Path] = set()

    with sample_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            url = doc.get("url", "")
            text = doc.get("text", "")
            domain = extract_domain(url)
            if not domain:
                skipped += 1
                continue

            domain_dir = output / domain
            domain_dir.mkdir(parents=True, exist_ok=True)

            docid = sanitize_doc_id(doc.get("id"), idx)
            title = extract_title(text)
            out_name = build_filename(title, url, docid, prefix)
            out_path = unique_path(domain_dir / out_name, docid, text)
            while out_path in seen_paths:
                stem = out_path.stem
                suffix = out_path.suffix
                out_path = out_path.with_name(f"{stem}__dup{suffix}")

            out_path.write_text(text, encoding="utf-8")
            seen_paths.add(out_path)
            written += 1

            if written % 10000 == 0:
                print(f"Exported {written} FineWeb docs...", flush=True)

    return written, skipped


def build_embedding_index(args: argparse.Namespace) -> None:
    if args.index_output_dir is None:
        raise SystemExit("--index-output-dir is required when --build-index is set")

    prepare_dir(
        args.index_output_dir,
        force=True,
        clean=args.clean_index_output,
        label="index output",
    )

    script_path = Path(__file__).resolve().parents[1] / "build_embedding_index.py"
    cmd = [
        args.embed_python,
        str(script_path),
        "--corpus-dir",
        str(args.output),
        "--output-dir",
        str(args.index_output_dir),
        "--model",
        args.embed_model,
        "--max-chars",
        str(args.index_max_chars),
        "--max-model-len",
        str(args.index_max_model_len),
        "--batch-size",
        str(args.index_batch_size),
        "--tensor-parallel-size",
        str(args.index_tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.index_gpu_memory_utilization),
    ]

    print("Building embedding index...", flush=True)
    print("  Command: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    if args.source_browsecomp is None and args.source_original is None:
        raise SystemExit("One of --source-browsecomp or --source-original is required")

    if args.source_browsecomp is not None and not args.source_browsecomp.exists():
        raise SystemExit(f"BrowseComp-Plus source not found: {args.source_browsecomp}")
    if args.source_original is not None and not args.source_original.exists():
        raise SystemExit(f"Original corpus not found: {args.source_original}")
    if not args.fineweb_sample.exists():
        raise SystemExit(f"FineWeb sample not found: {args.fineweb_sample}")

    prepare_dir(args.output, args.force, args.clean_output, label="output corpus")

    original_count: int | None
    if args.source_browsecomp is not None:
        print(f"Reconstructing original docs from {args.source_browsecomp} -> {args.output}", flush=True)
        original_count = export_original_docs_from_browsecomp(args.source_browsecomp, args.output)
    else:
        print(f"Copying original docs from {args.source_original} -> {args.output}", flush=True)
        original_count = copy_original_docs(args.source_original, args.output)

    print(f"Exporting FineWeb docs from {args.fineweb_sample}", flush=True)
    written, skipped = export_fineweb_docs(args.fineweb_sample, args.output, args.fineweb_prefix)

    print("Done!", flush=True)
    if original_count is None:
        print("  Original docs exported from browsecomp_plus", flush=True)
    else:
        print(f"  Original docs copied: {original_count}", flush=True)
    print(f"  FineWeb docs written: {written}", flush=True)
    print(f"  FineWeb docs skipped (no domain): {skipped}", flush=True)
    print(f"  Output: {args.output}", flush=True)

    if args.build_index:
        build_embedding_index(args)


if __name__ == "__main__":
    main()
