#!/usr/bin/env python3
"""
LLM-as-Judge for NeMo BC+ QA results.

Reads NeMo QA run directories of the form:
  <output_dir>/<query_id>/result.json
  <output_dir>/<query_id>/final.txt

Unlike DCI-Agent-Lite's generic judge_results.py, this script does not depend on
state.json/status conventions from other runners. It directly uses result.json /
final.txt produced by the NeMo BC+ QA branch.
"""

from __future__ import annotations

import argparse
import os
import os
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(path)


def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Judge response did not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Judge response JSON was not an object")
    return payload


def judge_answer(
    *,
    base_url: str,
    model: str,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    timeout_seconds: int = 120,
    max_retries: int = 3,
) -> Dict[str, Any]:
    system_prompt = (
        "You are grading a question-answer benchmark. "
        "Mark the prediction correct only if it identifies the same final answer as the gold answer. "
        "Ignore case, surrounding punctuation, whitespace, and extra explanation or supporting file paths. "
        "Do not give partial credit. Return exactly one compact JSON object."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Predicted answer:\n{predicted_answer or '[empty]'}\n\n"
        'Return JSON with keys "is_correct" (boolean), "normalized_prediction" (string), and "reason" (string).'
    )

    request_payload = {
        "model": model,
        "max_tokens": 180,
        "temperature": 0,
        "reasoning_effort": "low",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request_body = json.dumps(request_payload).encode("utf-8")

    last_error = None
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(2)
        try:
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=request_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))

            choices = response_payload.get("choices", [])
            if not choices:
                last_error = "No choices in response"
                continue
            response_text = choices[0].get("message", {}).get("content", "")
            parsed = extract_json_object(response_text)
            usage = response_payload.get("usage", {})

            return {
                "judge_model": model,
                "judged_at": utc_now(),
                "judge_status": "success",
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "is_correct": bool(parsed.get("is_correct")),
                "normalized_prediction": str(parsed.get("normalized_prediction", "")),
                "reason": str(parsed.get("reason", "")),
                "usage": usage,
                "raw_response_text": response_text,
                "error": None,
            }
        except ValueError as exc:
            last_error = f"JSON parse error: {exc}"
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)

    return {
        "judge_model": model,
        "judged_at": utc_now(),
        "judge_status": "failed",
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "is_correct": None,
        "normalized_prediction": "",
        "reason": "",
        "error": f"Judge failed after {max_retries} attempts: {last_error}",
    }


def load_gold_answers(dataset_path: Path) -> Dict[str, Dict[str, str]]:
    gold_answers: Dict[str, Dict[str, str]] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = str(item["query_id"])
            gold_answers[qid] = {
                "answer": str(item["answer"]),
                "question": str(item.get("question", item.get("query", ""))),
            }
    return gold_answers


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge for NeMo BC+ QA results")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory containing per-query subdirectories")
    parser.add_argument("--dataset", type=Path, required=True, help="BC+ dataset JSONL")
    parser.add_argument("--judge-model", type=str, default="gpt-5.4", help="Judge model")
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="Judge API base URL")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per judge request")
    parser.add_argument("--force", action="store_true", help="Re-judge all queries")
    args = parser.parse_args()

    gold_answers = load_gold_answers(args.dataset)
    query_dirs = sorted(
        [d for d in args.output_dir.iterdir() if d.is_dir() and (d / "result.json").exists()],
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
    )

    total = 0
    correct = 0
    failed_judge = 0
    skipped_no_final = 0
    results: List[Dict[str, Any]] = []

    print("=== LLM-as-Judge Evaluation (NeMo BC+ QA) ===")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Judge model:  {args.judge_model}")
    print(f"  Base URL:     {args.base_url}")
    print()

    for query_dir in query_dirs:
        qid = query_dir.name
        if qid not in gold_answers:
            continue

        result_data = read_json(query_dir / "result.json") or {}
        final_path = query_dir / "final.txt"
        predicted = ""
        if final_path.exists():
            predicted = final_path.read_text(encoding="utf-8").strip()
        if not predicted:
            predicted = str(result_data.get("predicted_answer", "")).strip()
        if not predicted:
            skipped_no_final += 1
            print(f"  [skip]   qid={qid:>5} no final content")
            continue

        eval_path = query_dir / "eval_result.json"
        if not args.force and eval_path.exists():
            cached = read_json(eval_path)
            if cached and cached.get("is_correct") is not None and cached.get("judge_model") == args.judge_model:
                total += 1
                if cached["is_correct"]:
                    correct += 1
                results.append({"qid": qid, **cached})
                print(
                    f"  [cached] qid={qid:>5} {'CORRECT' if cached['is_correct'] else 'WRONG':>8} | "
                    f"gold={gold_answers[qid]['answer'][:40]:<40} | pred={cached.get('normalized_prediction', predicted[:40])}"
                )
                continue

        gold = gold_answers[qid]
        eval_result = judge_answer(
            base_url=args.base_url,
            model=args.judge_model,
            question=gold["question"],
            gold_answer=gold["answer"],
            predicted_answer=predicted,
            timeout_seconds=args.timeout,
        )
        write_json(eval_path, eval_result)

        if eval_result.get("error"):
            failed_judge += 1
            print(f"  [ERROR]  qid={qid:>5} {eval_result['error'][:80]}")
        else:
            total += 1
            is_correct = bool(eval_result["is_correct"])
            if is_correct:
                correct += 1
            results.append({"qid": qid, **eval_result})
            print(
                f"  [judged] qid={qid:>5} {'CORRECT' if is_correct else 'WRONG':>8} | "
                f"gold={gold['answer'][:40]:<40} | pred={eval_result.get('normalized_prediction', predicted[:40])}"
            )

    acc = correct / total if total > 0 else 0.0
    print()
    print("=" * 60)
    print(f"  Total judged:           {total}")
    print(f"  Correct:                {correct}")
    print(f"  Accuracy:               {acc:.4f} ({correct}/{total})")
    print(f"  Judge failures:         {failed_judge}")
    print(f"  Skipped (no final):     {skipped_no_final}")
    print("=" * 60)

    summary = {
        "judge_model": args.judge_model,
        "evaluated_at": utc_now(),
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "failed_judge": failed_judge,
        "skipped_no_final": skipped_no_final,
        "per_query": [
            {
                "qid": r["qid"],
                "is_correct": r.get("is_correct"),
                "normalized_prediction": r.get("normalized_prediction", ""),
                "reason": r.get("reason", ""),
            }
            for r in results
        ],
    }
    summary_path = args.output_dir / "judge_summary.json"
    write_json(summary_path, summary)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
