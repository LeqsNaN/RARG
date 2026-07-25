#!/usr/bin/env python3
"""
Export SRA-Bench skills corpus to individual files (like DCI BRIGHT format).

Each skill becomes a separate .txt file:
    data/sra_corpus/skills/{skill_id}.txt

This allows the agent to:
    rg -l "keyword" .    → get list of matching skill files (fast, compact output)
    read skills/xxx.txt  → read full skill content

Usage:
    python3 scripts/sra_bench/prepare_corpus.py --format files
    python3 scripts/sra_bench/prepare_corpus.py --format jsonl   (original single-file)
    python3 scripts/sra_bench/prepare_corpus.py --format both    (default)
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CORPUS_JSON = REPO_ROOT.parents[1] / "SR-Agents" / "data" / "bench" / "corpus" / "corpus.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "sra_corpus"


def export_jsonl(corpus: list, output_dir: Path) -> None:
    """Export as single JSONL file."""
    output_path = output_dir / "skills.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for skill in corpus:
            record = {
                "skill_id": skill["skill_id"],
                "name": skill["name"],
                "description": skill["description"],
                "content": skill["content"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  JSONL: {len(corpus)} skills → {output_path} ({size_mb:.1f} MB)")


def export_files(corpus: list, output_dir: Path) -> None:
    """Export each skill as individual .txt file (like DCI BRIGHT format).

    File format:
        skill_id: {skill_id}
        name: {name}
        description: {description}

        {content}
    """
    skills_dir = output_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    for skill in corpus:
        skill_id = skill["skill_id"]
        file_path = skills_dir / f"{skill_id}.txt"

        text = (
            f"skill_id: {skill_id}\n"
            f"name: {skill['name']}\n"
            f"description: {skill['description']}\n"
            f"\n"
            f"{skill['content']}"
        )
        file_path.write_text(text, encoding="utf-8")

    print(f"  Files: {len(corpus)} skills → {skills_dir}/ ({len(corpus)} .txt files)")


def main():
    parser = argparse.ArgumentParser(description="Prepare SRA-Bench skill corpus for DCI Agent")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_CORPUS_JSON,
        help=f"Path to corpus.json. Default: {DEFAULT_CORPUS_JSON}",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--format", choices=["jsonl", "files", "both"], default="both",
        help="Export format. Default: both",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"corpus.json not found: {args.input}")

    print(f"Loading corpus from {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  Total skills: {len(corpus)}")

    args.output.mkdir(parents=True, exist_ok=True)

    if args.format in ("jsonl", "both"):
        export_jsonl(corpus, args.output)

    if args.format in ("files", "both"):
        export_files(corpus, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
