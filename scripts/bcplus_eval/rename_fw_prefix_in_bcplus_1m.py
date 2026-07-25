#!/usr/bin/env python3
"""Remove legacy ``fw_`` filename prefixes from an existing bc_plus_1m corpus.

This script updates two things in a coordinated way:

1. Rename physical files under ``corpus/bc_plus_1m`` whose basename starts
   with ``fw_``.
2. Rewrite ``data/indices/bc_plus_1m/paths.json`` with the new relative paths,
   preserving the original order exactly so the existing FAISS embeddings remain
   aligned with their path entries.

The vector index itself is not modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus" / "bc_plus_1m"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "indices" / "bc_plus_1m"


@dataclass(frozen=True)
class RenameEntry:
    old_relpath: str
    new_relpath: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove legacy fw_ prefixes from bc_plus_1m corpus filenames and "
            "update paths.json without rebuilding embeddings."
        )
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Corpus root, default: corpus/bc_plus_1m",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Index root, default: data/indices/bc_plus_1m",
    )
    parser.add_argument(
        "--paths-json",
        type=Path,
        default=None,
        help="Optional explicit paths.json path. Default: <index-dir>/paths.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report planned renames; do not touch files.",
    )
    return parser.parse_args()


def choose_unique_relpath(
    candidate: Path,
    reserved: set[str],
) -> str:
    candidate_str = candidate.as_posix()
    if candidate_str not in reserved:
        return candidate_str

    stem = candidate.stem
    suffix = candidate.suffix
    parent = candidate.parent

    counter = 1
    while True:
        alt = (parent / f"{stem}__dup{counter}{suffix}").as_posix()
        if alt not in reserved:
            return alt
        counter += 1


def load_paths(paths_json: Path) -> list[str]:
    with paths_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise SystemExit(f"Expected a JSON list[str] in {paths_json}")
    return data


def plan_renames(paths: list[str]) -> tuple[list[str], list[RenameEntry], int]:
    reserved = {p for p in paths if not Path(p).name.startswith("fw_")}
    rewritten: list[str] = []
    renames: list[RenameEntry] = []
    conflicts = 0

    for relpath in paths:
        rel = Path(relpath)
        name = rel.name
        if not name.startswith("fw_"):
            rewritten.append(relpath)
            continue

        stripped = rel.with_name(name[3:])
        new_relpath = choose_unique_relpath(stripped, reserved)
        if new_relpath != stripped.as_posix():
            conflicts += 1
        reserved.add(new_relpath)
        rewritten.append(new_relpath)
        renames.append(RenameEntry(old_relpath=relpath, new_relpath=new_relpath))

    return rewritten, renames, conflicts


def apply_file_renames(corpus_dir: Path, renames: list[RenameEntry]) -> None:
    for idx, entry in enumerate(renames, start=1):
        old_path = corpus_dir / entry.old_relpath
        new_path = corpus_dir / entry.new_relpath
        new_path.parent.mkdir(parents=True, exist_ok=True)

        if old_path.exists():
            if new_path.exists():
                raise SystemExit(
                    f"Refusing to overwrite existing target during rename:\n"
                    f"  old: {old_path}\n"
                    f"  new: {new_path}"
                )
            old_path.rename(new_path)
        else:
            if not new_path.exists():
                raise SystemExit(
                    f"Neither source nor target exists for planned rename:\n"
                    f"  old: {old_path}\n"
                    f"  new: {new_path}"
                )

        if idx % 10000 == 0:
            print(f"Renamed {idx}/{len(renames)} files...", flush=True)


def backup_file(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak_before_fw_prefix_fix_{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def write_paths_json(paths_json: Path, rewritten: list[str]) -> None:
    tmp_path = paths_json.with_name(f"{paths_json.name}.tmp_fw_prefix_fix")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(rewritten, f, ensure_ascii=False)
    tmp_path.replace(paths_json)


def write_manifest(index_dir: Path, renames: list[RenameEntry], backup_path: Path | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = index_dir / f"fw_prefix_rename_manifest_{timestamp}.json"
    payload = {
        "renamed_count": len(renames),
        "paths_json_backup": str(backup_path) if backup_path is not None else None,
        "renames": [
            {"old_relpath": entry.old_relpath, "new_relpath": entry.new_relpath}
            for entry in renames
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()
    corpus_dir = args.corpus_dir
    index_dir = args.index_dir
    paths_json = args.paths_json or (index_dir / "paths.json")

    if not corpus_dir.exists():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")
    if not paths_json.exists():
        raise SystemExit(f"paths.json not found: {paths_json}")

    paths = load_paths(paths_json)
    rewritten, renames, conflicts = plan_renames(paths)

    print(f"Loaded {len(paths)} path entries from {paths_json}", flush=True)
    print(f"Entries needing fw_ removal: {len(renames)}", flush=True)
    print(f"Conflicts resolved with __dupN suffix: {conflicts}", flush=True)

    if renames:
        print("Examples:", flush=True)
        for entry in renames[:5]:
            print(f"  {entry.old_relpath}  ->  {entry.new_relpath}", flush=True)

    if args.dry_run:
        print("Dry run only; no files changed.", flush=True)
        return

    backup_path = backup_file(paths_json)
    print(f"Backed up paths.json to {backup_path}", flush=True)

    apply_file_renames(corpus_dir, renames)
    write_paths_json(paths_json, rewritten)
    manifest_path = write_manifest(index_dir, renames, backup_path)

    print("Done.", flush=True)
    print(f"  Renamed files: {len(renames)}", flush=True)
    print(f"  Updated paths.json: {paths_json}", flush=True)
    print(f"  Manifest: {manifest_path}", flush=True)
    print("  index.faiss unchanged; embedding order preserved.", flush=True)


if __name__ == "__main__":
    main()
