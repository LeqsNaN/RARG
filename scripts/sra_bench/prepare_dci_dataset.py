#!/usr/bin/env python3
"""
Convert SRA-Bench instances to DCI-Agent-Lite dataset JSONL format.

DCI expects: {"query_id": "...", "query": "...", "answer": "...", "gold_docs": [...]}
SRA-Bench has: {"instance_id": "...", "question": "...", "skill_annotations": [...], "eval_data": {...}}

Usage:
    python3 scripts/sra_bench/prepare_dci_dataset.py --dataset champ
    python3 scripts/sra_bench/prepare_dci_dataset.py --dataset theoremqa
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SR_AGENTS_ROOT = REPO_ROOT.parents[1] / "SR-Agents"
DEFAULT_INSTANCES_DIR = SR_AGENTS_ROOT / "data" / "bench" / "instances"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "sra_bench"

ALL_DATASETS = ["theoremqa", "logicbench", "toolqa", "medcalcbench", "champ", "bigcodebench"]


def main():
    parser = argparse.ArgumentParser(description="Convert SRA-Bench instances to DCI JSONL format.")
    parser.add_argument("--dataset", choices=ALL_DATASETS + ["all"], default="all",
                        help="Dataset to convert. Default: all")
    parser.add_argument("--instances-dir", type=Path, default=DEFAULT_INSTANCES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        src = args.instances_dir / f"{dataset}.json"
        if not src.exists():
            print(f"  SKIP {dataset}: {src} not found")
            continue

        instances = json.loads(src.read_text())
        output_path = args.output_dir / f"{dataset}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for inst in instances:
                # Map SRA-Bench fields to DCI format
                # gold_docs: skill_annotations serve as "gold documents" for nDCG evaluation
                record = {
                    "query_id": inst["instance_id"],
                    "query": inst["question"],
                    "answer": inst.get("eval_data", {}).get("answer", ""),
                    "gold_docs": inst.get("skill_annotations", []),
                    "source": f"sra_bench_{dataset}",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  {dataset}: {len(instances)} instances → {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
