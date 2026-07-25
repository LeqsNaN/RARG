#!/usr/bin/env python3
"""
DCI-style LLM-as-Judge for BrowseComp-Plus results.

Open-source-oriented copy of DCI-Agent-Lite/scripts/bcplus_eval/judge_results.py
with the following changes:
- keeps only the original DCI judge style
- removes all RISE-specific prompt / schema logic
- uses OpenAI-compatible base URL + API key settings
- preserves per-query outputs immediately so judging progress is inspectable
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = "https://api.openai.com/v1"


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


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


def normalize_api_key(api_key: str) -> str:
    token = (api_key or "").strip()
    if not token or token.lower() in {"none", "null", "not-needed"}:
        return ""
    return token


def build_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = normalize_api_key(api_key)
    if token:
        if token.lower().startswith("bearer "):
            headers["Authorization"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def cached_result_matches(cached: Optional[Dict[str, Any]], judge_model: str) -> bool:
    if not cached:
        return False
    return cached.get("is_correct") is not None and cached.get("judge_model") == judge_model


def result_answer_preview(result: Dict[str, Any], fallback: str) -> str:
    return str(result.get("normalized_prediction") or fallback)


def load_gold_answers(dataset_path: Path) -> Dict[str, Dict[str, str]]:
    gold_answers: Dict[str, Dict[str, str]] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = str(item["query_id"])
            gold_answers[qid] = {
                "answer": str(item.get("answer", "")),
                "question": str(item.get("query") or item.get("question") or ""),
            }
    return gold_answers


def list_query_dirs(output_dir: Path) -> List[Path]:
    return sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and (d / "state.json").exists()],
        key=lambda d: int(d.name) if d.name.isdigit() else d.name,
    )


def judge_answer(
    *,
    base_url: str,
    api_key: str,
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
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request_body = json.dumps(request_payload).encode("utf-8")

    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = build_headers(api_key)
    last_error = None

    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(2)
        try:
            req = urllib.request.Request(
                endpoint,
                data=request_body,
                headers=headers,
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
                "judge_style": "dci",
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
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            last_error = f"HTTP {exc.code}: {body or exc.reason}"
        except urllib.error.URLError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)

    return {
        "judge_style": "dci",
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


def write_summary(
    *,
    summary_path: Path,
    judge_model: str,
    total: int,
    correct: int,
    failed_judge: int,
    skipped: int,
    per_query: List[Dict[str, Any]],
) -> None:
    acc = correct / total if total > 0 else 0.0
    summary = {
        "judge_style": "dci",
        "judge_model": judge_model,
        "evaluated_at": utc_now(),
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "failed_judge": failed_judge,
        "skipped": skipped,
        "per_query": per_query,
    }
    write_json(summary_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="DCI-style LLM-as-Judge for BrowseComp-Plus results")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory with query subdirectories")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSONL with gold answers")
    parser.add_argument("--judge-model", type=str, default="gpt-5.4", help="Model for judging")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help=f"LLM API base URL (default: OPENAI_BASE_URL or {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key for the judge model (default: OPENAI_API_KEY)",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per judge request (seconds)")
    parser.add_argument("--force", action="store_true", help="Re-judge all queries, ignore cached results")
    parser.add_argument("--result-file", type=str, default="eval_result.json", help="Per-query result filename")
    parser.add_argument("--summary-file", type=str, default="judge_summary.json", help="Summary filename")
    args = parser.parse_args()

    gold_answers = load_gold_answers(args.dataset)
    query_dirs = list_query_dirs(args.output_dir)

    total = 0
    correct = 0
    failed_judge = 0
    skipped = 0
    per_query: List[Dict[str, Any]] = []
    summary_path = args.output_dir / args.summary_file

    print("=== LLM-as-Judge Evaluation (DCI style) ===")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Judge model:  {args.judge_model}")
    print(f"  Base URL:     {args.base_url}")
    print(f"  Result file:  {args.result_file}")
    print(f"  Summary file: {args.summary_file}")
    print()

    for query_dir in query_dirs:
        qid = query_dir.name
        if qid not in gold_answers:
            continue

        state = read_json(query_dir / "state.json")
        if not state or state.get("status") != "completed":
            continue

        final_path = query_dir / "final.txt"
        if not final_path.exists():
            skipped += 1
            write_summary(
                summary_path=summary_path,
                judge_model=args.judge_model,
                total=total,
                correct=correct,
                failed_judge=failed_judge,
                skipped=skipped,
                per_query=per_query,
            )
            continue

        predicted = final_path.read_text(encoding="utf-8").strip()
        eval_path = query_dir / args.result_file

        if not args.force and eval_path.exists():
            cached = read_json(eval_path)
            if cached_result_matches(cached, args.judge_model):
                total += 1
                is_correct = bool(cached.get("is_correct"))
                if is_correct:
                    correct += 1
                per_query.append(
                    {
                        "qid": qid,
                        "is_correct": cached.get("is_correct"),
                        "normalized_prediction": cached.get("normalized_prediction", ""),
                        "reason": cached.get("reason", ""),
                    }
                )
                print(
                    f"  [cached] qid={qid:>5} {'CORRECT' if is_correct else 'WRONG':>8} | "
                    f"gold={gold_answers[qid]['answer'][:40]:<40} | pred={result_answer_preview(cached, predicted[:40])}"
                )
                write_summary(
                    summary_path=summary_path,
                    judge_model=args.judge_model,
                    total=total,
                    correct=correct,
                    failed_judge=failed_judge,
                    skipped=skipped,
                    per_query=per_query,
                )
                continue

        gold = gold_answers[qid]
        eval_result = judge_answer(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.judge_model,
            question=gold["question"],
            gold_answer=gold["answer"],
            predicted_answer=predicted,
            timeout_seconds=args.timeout,
        )
        write_json(eval_path, eval_result)

        if eval_result.get("error"):
            failed_judge += 1
            print(f"  [ERROR]  qid={qid:>5} {eval_result['error'][:120]}")
        else:
            total += 1
            is_correct = bool(eval_result.get("is_correct"))
            if is_correct:
                correct += 1
            per_query.append(
                {
                    "qid": qid,
                    "is_correct": eval_result.get("is_correct"),
                    "normalized_prediction": eval_result.get("normalized_prediction", ""),
                    "reason": eval_result.get("reason", ""),
                }
            )
            print(
                f"  [judged] qid={qid:>5} {'CORRECT' if is_correct else 'WRONG':>8} | "
                f"gold={gold['answer'][:40]:<40} | pred={result_answer_preview(eval_result, predicted[:40])}"
            )

        write_summary(
            summary_path=summary_path,
            judge_model=args.judge_model,
            total=total,
            correct=correct,
            failed_judge=failed_judge,
            skipped=skipped,
            per_query=per_query,
        )

    acc = correct / total if total > 0 else 0.0
    print()
    print("=" * 60)
    print(f"  Total judged:   {total}")
    print(f"  Correct:        {correct}")
    print(f"  Accuracy:       {acc:.4f} ({correct}/{total})")
    print(f"  Judge failures: {failed_judge}")
    print(f"  Skipped (no final.txt): {skipped}")
    print("=" * 60)
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
