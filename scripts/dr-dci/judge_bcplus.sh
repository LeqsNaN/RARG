#!/usr/bin/env bash
# Foreground LLM-as-judge for a standalone DR-DCI BrowseComp-Plus run.
# Uses OPENAI_API_KEY and OPENAI_BASE_URL (or API_KEY and BASE_URL).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.1}"
DATASET="${DATASET:-$REPO_ROOT/data/bcplus_qa_sample100.jsonl}"
DR_DCI_RUN_NAME="${DR_DCI_RUN_NAME:-dr_dci_bcplus100k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/dr_dci/$DR_DCI_RUN_NAME}"
RESULT_FILE="${RESULT_FILE:-eval_result_openai.json}"
SUMMARY_FILE="${SUMMARY_FILE:-judge_summary_openai.json}"
JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-120}"
JUDGE_FORCE="${JUDGE_FORCE:-0}"

if [ -z "$API_KEY" ]; then
    echo "ERROR: Set OPENAI_API_KEY (or API_KEY) before judging." >&2
    exit 1
fi
if [ ! -d "$OUTPUT_ROOT" ] || [ ! -f "$DATASET" ]; then
    echo "ERROR: Output root or dataset does not exist." >&2
    exit 1
fi

CMD=(
    "$REPO_ROOT/.venv/bin/python" -u "$REPO_ROOT/scripts/bcplus_eval/judge_results.py"
    --output-dir "$OUTPUT_ROOT"
    --dataset "$DATASET"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --judge-model "$JUDGE_MODEL"
    --judge-style dci
    --timeout "$JUDGE_TIMEOUT_SECONDS"
    --include-noncompleted-with-final
    --result-file "$RESULT_FILE"
    --summary-file "$SUMMARY_FILE"
)
if [ "$JUDGE_FORCE" = "1" ]; then
    CMD+=(--force)
fi

echo "=== Judge standalone DR-DCI / BrowseComp-Plus ==="
echo "  Output root:  $OUTPUT_ROOT"
echo "  API endpoint: $BASE_URL"
echo "  Judge model:  $JUDGE_MODEL"
echo "  Force rejudge:$([ "$JUDGE_FORCE" = "1" ] && echo yes || echo no)"
exec "${CMD[@]}"
