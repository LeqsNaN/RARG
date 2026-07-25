#!/usr/bin/env python3
"""Sample FineWeb documents from downloaded parquet shards.

This script is intended to run on the cluster after the parquet files have
already been downloaded locally/shared to the filesystem.

It performs a two-pass sampling process.

If ``--min-text-chars`` is 0:
1. Read parquet metadata to count the total number of rows.
2. Sample global row indices uniformly without replacement.
3. Stream the selected rows into a JSONL file with fields: id, url, text.

If ``--min-text-chars`` is > 0:
1. Stream all shards once and count only rows whose text length is at least the threshold.
2. Sample eligible-row indices uniformly without replacement.
3. Stream all shards a second time and write only the selected eligible rows.

Usage:
    python scripts/bcplus_eval/sample_fineweb_from_parquet.py \
        --input-dir data/fineweb_edu_10bt/sample/10BT \
        --output data/fineweb_edu_10bt_sample900k.jsonl \
        --sample-size 900000 \
        --min-text-chars 12000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample FineWeb rows from parquet shards")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing downloaded parquet shards")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--sample-size", type=int, default=900000, help="Number of documents to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=2048, help="Parquet streaming batch size")
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=0,
        help="Only sample documents whose text length is at least this many characters (default: 0). Uses >= semantics.",
    )
    return parser.parse_args()


def load_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "pyarrow is required. Install on the cluster first, e.g.\n"
            "uv pip install pyarrow -i https://mirrors.cloud.tencent.com/pypi/simple"
        ) from exc
    return pq


def discover_parquet_files(input_dir: Path) -> list[Path]:
    files = sorted(p for p in input_dir.rglob("*.parquet") if p.is_file())
    if not files:
        raise SystemExit(f"No parquet files found under {input_dir}")
    return files


def count_rows(parquet_files: list[Path], pq) -> tuple[int, list[tuple[Path, int]]]:
    per_file: list[tuple[Path, int]] = []
    total = 0
    for path in parquet_files:
        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows
        per_file.append((path, rows))
        total += rows
    return total, per_file


def count_eligible_rows(
    parquet_files: list[Path],
    batch_size: int,
    min_text_chars: int,
    pq,
) -> tuple[int, list[tuple[Path, int]]]:
    per_file: list[tuple[Path, int]] = []
    total = 0

    for file_idx, path in enumerate(parquet_files, start=1):
        pf = pq.ParquetFile(path)
        eligible_rows = 0
        for batch in pf.iter_batches(columns=["text"], batch_size=batch_size):
            texts = batch.column(0).to_pylist()
            eligible_rows += sum(1 for text in texts if isinstance(text, str) and len(text) >= min_text_chars)
        per_file.append((path, eligible_rows))
        total += eligible_rows
        print(
            f"[count {file_idx}/{len(parquet_files)}] eligible rows so far: {total} (latest file: {path.name}, file_eligible={eligible_rows})",
            flush=True,
        )

    return total, per_file


def sample_indices(total_rows: int, sample_size: int, seed: int) -> list[int]:
    if sample_size > total_rows:
        raise SystemExit(f"sample_size={sample_size} exceeds total_rows={total_rows}")
    rng = random.Random(seed)
    indices = rng.sample(range(total_rows), sample_size)
    indices.sort()
    return indices


def stream_selected_rows(
    per_file: list[tuple[Path, int]],
    selected: list[int],
    output_path: Path,
    batch_size: int,
    pq,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_ptr = 0
    global_row_start = 0
    written = 0

    columns = ["id", "url", "text"]

    with output_path.open("w", encoding="utf-8") as out_f:
        for file_idx, (path, row_count) in enumerate(per_file, start=1):
            file_global_end = global_row_start + row_count

            if selected_ptr >= len(selected):
                break
            if selected[selected_ptr] >= file_global_end:
                global_row_start = file_global_end
                continue

            pf = pq.ParquetFile(path)
            local_target_rows = []
            while selected_ptr < len(selected) and selected[selected_ptr] < file_global_end:
                local_target_rows.append(selected[selected_ptr] - global_row_start)
                selected_ptr += 1

            local_ptr = 0
            local_batch_start = 0
            for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
                batch_py = batch.to_pydict()
                batch_len = len(batch_py.get("text", []))
                local_batch_end = local_batch_start + batch_len

                while local_ptr < len(local_target_rows) and local_target_rows[local_ptr] < local_batch_end:
                    rel = local_target_rows[local_ptr] - local_batch_start
                    doc = {
                        "id": batch_py.get("id", [None] * batch_len)[rel],
                        "url": batch_py.get("url", [""] * batch_len)[rel],
                        "text": batch_py.get("text", [""] * batch_len)[rel],
                    }
                    out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    written += 1
                    local_ptr += 1

                local_batch_start = local_batch_end
                if local_ptr >= len(local_target_rows):
                    break

            print(
                f"[{file_idx}/{len(per_file)}] wrote {written} sampled docs so far from {path.name}",
                flush=True,
            )
            global_row_start = file_global_end

    if written != len(selected):
        raise SystemExit(f"Expected to write {len(selected)} docs, but wrote {written}")


def stream_selected_eligible_rows(
    parquet_files: list[Path],
    selected: list[int],
    output_path: Path,
    batch_size: int,
    min_text_chars: int,
    pq,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_ptr = 0
    eligible_seen = 0
    written = 0

    columns = ["id", "url", "text"]

    with output_path.open("w", encoding="utf-8") as out_f:
        for file_idx, path in enumerate(parquet_files, start=1):
            if selected_ptr >= len(selected):
                break

            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
                batch_py = batch.to_pydict()
                batch_len = len(batch_py.get("text", []))

                ids = batch_py.get("id", [None] * batch_len)
                urls = batch_py.get("url", [""] * batch_len)
                texts = batch_py.get("text", [""] * batch_len)

                for rel in range(batch_len):
                    text = texts[rel]
                    if not isinstance(text, str) or len(text) < min_text_chars:
                        continue

                    if selected_ptr < len(selected) and eligible_seen == selected[selected_ptr]:
                        doc = {
                            "id": ids[rel],
                            "url": urls[rel],
                            "text": text,
                        }
                        out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        written += 1
                        selected_ptr += 1
                    eligible_seen += 1

                    if selected_ptr >= len(selected):
                        break

                if selected_ptr >= len(selected):
                    break

            print(
                f"[write {file_idx}/{len(parquet_files)}] wrote {written} sampled eligible docs so far from {path.name}",
                flush=True,
            )

    if written != len(selected):
        raise SystemExit(f"Expected to write {len(selected)} docs, but wrote {written}")


def main() -> None:
    args = parse_args()
    pq = load_pyarrow()

    parquet_files = discover_parquet_files(args.input_dir)
    print(f"Found {len(parquet_files)} parquet files under {args.input_dir}", flush=True)

    if args.min_text_chars > 0:
        total_rows, per_file = count_eligible_rows(
            parquet_files,
            args.batch_size,
            args.min_text_chars,
            pq,
        )
        print(
            f"Total eligible rows available (text >= {args.min_text_chars} chars): {total_rows}",
            flush=True,
        )
    else:
        total_rows, per_file = count_rows(parquet_files, pq)
        print(f"Total rows available: {total_rows}", flush=True)

    selected = sample_indices(total_rows, args.sample_size, args.seed)
    print(
        f"Sampled {len(selected)} {'eligible ' if args.min_text_chars > 0 else ''}row indices with seed={args.seed}; writing to {args.output}",
        flush=True,
    )

    if args.min_text_chars > 0:
        stream_selected_eligible_rows(
            parquet_files,
            selected,
            args.output,
            args.batch_size,
            args.min_text_chars,
            pq,
        )
    else:
        stream_selected_rows(per_file, selected, args.output, args.batch_size, pq)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
