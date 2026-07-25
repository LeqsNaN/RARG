#!/usr/bin/env python3
"""
Create the BrowseComp-Plus 100K corpus as real files.

Preferred workflow:
  - use the DCI exporter directly on `corpus/browsecomp_plus`
  - write the exported 100K document tree into `corpus/bc_plus_100k`

Legacy fallback:
  - copy an existing `corpus/bc_plus_docs` tree into `corpus/bc_plus_100k`
  - skip any `fw_*` distractor files if they appear

This script intentionally writes real files rather than symlinks so downstream
index builders and experiment scripts see an ordinary corpus tree.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create BrowseComp-Plus 100K corpus with DCI-compatible export logic")
    parser.add_argument(
        "--source-browsecomp",
        type=Path,
        default=None,
        help="Raw BrowseComp-Plus parquet dir, e.g. corpus/browsecomp_plus",
    )
    parser.add_argument(
        "--source-docs",
        type=Path,
        default=None,
        help="Existing exported docs dir, e.g. corpus/bc_plus_docs",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Legacy alias. If parquet files are found, treated as --source-browsecomp; otherwise as --source-docs.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output corpus dir, e.g. corpus/bc_plus_100k")
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing non-empty output dir")
    parser.add_argument("--clean-output", action="store_true", help="If set with --force, remove existing files under output first")
    return parser.parse_args()


def prepare_dir(path: Path, force: bool, clean: bool) -> None:
    if path.exists():
        has_entries = any(path.iterdir())
        if has_entries and not force:
            raise SystemExit(f"Output directory {path} already exists and is not empty; pass --force to continue")
        if has_entries and clean:
            print(f"Cleaning existing output directory: {path}", flush=True)
            for child in path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def infer_sources(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    source_browsecomp = args.source_browsecomp
    source_docs = args.source_docs

    if args.source is not None:
        parquet_candidates = list(args.source.glob("*.parquet"))
        if parquet_candidates:
            source_browsecomp = args.source
        else:
            source_docs = args.source

    if source_browsecomp is None and source_docs is None:
        default_browsecomp = REPO_ROOT / "corpus" / "browsecomp_plus"
        default_docs = REPO_ROOT / "corpus" / "bc_plus_docs"
        if default_browsecomp.exists():
            source_browsecomp = default_browsecomp
        elif default_docs.exists():
            source_docs = default_docs

    return source_browsecomp, source_docs


def export_with_dci(source_browsecomp: Path, output: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "dci.benchmark.export_bc_plus_docs",
        "--source-dir",
        str(source_browsecomp),
        "--output-dir",
        str(output),
    ]
    print("Creating bc_plus_100k via DCI exporter...", flush=True)
    print("  Command: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def copy_exported_docs(source_docs: Path, output: Path) -> tuple[int, int]:
    copied = 0
    skipped = 0

    for domain_dir in sorted(source_docs.iterdir()):
        if not domain_dir.is_dir():
            continue

        out_domain = output / domain_dir.name
        out_domain.mkdir(parents=True, exist_ok=True)

        for src_file in domain_dir.iterdir():
            if not src_file.is_file():
                continue
            if src_file.name.startswith("fw_"):
                skipped += 1
                continue
            dst_file = out_domain / src_file.name
            if dst_file.exists():
                continue
            shutil.copy2(src_file, dst_file)
            copied += 1
            if copied % 10000 == 0:
                print(f"  Copied {copied} files...", flush=True)

    return copied, skipped


def main() -> None:
    args = parse_args()
    source_browsecomp, source_docs = infer_sources(args)

    if source_browsecomp is None and source_docs is None:
        raise SystemExit(
            "No source provided. Pass --source-browsecomp / --source-docs / --source, "
            "or place corpus/browsecomp_plus or corpus/bc_plus_docs under the repo root."
        )

    if source_browsecomp is not None and not source_browsecomp.exists():
        raise SystemExit(f"BrowseComp-Plus source not found: {source_browsecomp}")
    if source_docs is not None and not source_docs.exists():
        raise SystemExit(f"Exported docs source not found: {source_docs}")

    prepare_dir(args.output, args.force, args.clean_output)

    if source_browsecomp is not None:
        export_with_dci(source_browsecomp, args.output)
        print("\nDone!", flush=True)
        print("  Source type: BrowseComp parquet/raw corpus", flush=True)
        print(f"  Output:      {args.output}", flush=True)
        return

    assert source_docs is not None
    copied, skipped = copy_exported_docs(source_docs, args.output)
    print("\nDone!", flush=True)
    print("  Source type: exported bc_plus_docs tree", flush=True)
    print(f"  Copied:      {copied} files", flush=True)
    print(f"  Skipped:     {skipped} fw_* files", flush=True)
    print(f"  Output:      {args.output}", flush=True)


if __name__ == "__main__":
    main()
