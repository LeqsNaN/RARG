#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

BRIGHT_SOURCE_ROOT="${BRIGHT_SOURCE_ROOT:-$REPO_ROOT/data/bright/docs}"
BRIGHT_OUTPUT_ROOT="${BRIGHT_OUTPUT_ROOT:-$REPO_ROOT/bright_corpus}"
SUBSETS="${SUBSETS:-biology,earth_science,economics,robotics}"

mkdir -p "$REPO_ROOT/logs"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/prepare_bright_corpus_$(date '+%Y%m%d_%H%M%S').log}"

echo "=== Prepare BRIGHT document corpora ==="
echo "  Source root: $BRIGHT_SOURCE_ROOT"
echo "  Output root: $BRIGHT_OUTPUT_ROOT"
echo "  Subsets:     $SUBSETS"
echo "  Log:         $LOG_FILE"

IFS=',' read -r -a SUBSET_ARRAY <<< "$SUBSETS"

CMD=(
  "$REPO_ROOT/.venv/bin/python" -m dci.benchmark.export_bright_docs
  --source-root "$BRIGHT_SOURCE_ROOT"
  --output-root "$BRIGHT_OUTPUT_ROOT"
)

for subset in "${SUBSET_ARRAY[@]}"; do
  subset="${subset// /}"
  if [ -n "$subset" ]; then
    CMD+=(--subset "$subset")
  fi
done

nohup stdbuf -oL -eL "${CMD[@]}" > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Tail log: tail -f $LOG_FILE"
