#!/usr/bin/env python3
"""
Export FineWeb distractors as .txt files in domain/docid.txt structure.

Randomly samples N documents from fineweb_distractors_1m.jsonl and exports them
into the BrowseComp-Plus corpus directory, matching the existing structure.

Usage:
    python scripts/bcplus_eval/export_distractors.py \
        --input data/fineweb_distractors_1m.jsonl \
        --output-dir corpus/bc_plus_docs \
        --num-docs 100000 \
        --seed 42
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


def extract_domain(url: str) -> str:
    """Extract domain from URL, stripping www. prefix."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        # Strip port
        domain = domain.split(":")[0]
        # Strip www.
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return ""


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Make a string safe for use as a filename."""
    # Replace unsafe characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Trim
    name = name.strip('_. ')
    if len(name) > max_len:
        name = name[:max_len]
    return name or "doc"


def main():
    parser = argparse.ArgumentParser(description="Export FineWeb distractors as .txt files")
    parser.add_argument("--input", type=Path, required=True, help="Path to fineweb_distractors_1m.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Corpus directory (e.g. corpus/bc_plus_docs)")
    parser.add_argument("--num-docs", type=int, default=100000, help="Number of documents to sample (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # The file has 1M lines (already shuffled with seed=42).
    # We take the first num_docs lines as our sample (since it's pre-shuffled).
    print(f"Reading first {args.num_docs} documents from {args.input}...")
    docs = []
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_docs:
                break
            docs.append(json.loads(line))

    print(f"Loaded {len(docs)} documents")

    # Export as domain/docid.txt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_domain = 0
    domain_counts = {}

    for doc in docs:
        url = doc.get("url", "")
        text = doc.get("text", "")
        doc_id = doc.get("id", f"fineweb_{written:06d}")

        domain = extract_domain(url)
        if not domain:
            skipped_no_domain += 1
            continue

        # Create domain directory
        domain_dir = args.output_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: use a hash-based short ID to avoid collisions
        # Format: fw_{index:06d}.txt (fw prefix to distinguish from original docs)
        filename = f"fw_{written:06d}.txt"
        filepath = domain_dir / filename

        # Write content
        filepath.write_text(text, encoding="utf-8")
        written += 1

        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if written % 10000 == 0:
            print(f"  {written} files written ({len(domain_counts)} domains)...", flush=True)

    print(f"\nDone!")
    print(f"  Written: {written} files")
    print(f"  Skipped (no domain): {skipped_no_domain}")
    print(f"  Unique domains: {len(domain_counts)}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
