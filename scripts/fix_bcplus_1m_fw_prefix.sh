#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/corpus/bc_plus_1m}"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/indices/bc_plus_1m}"

mkdir -p "$REPO_ROOT/logs"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/fix_bcplus_1m_fw_prefix_$(date '+%Y%m%d_%H%M%S').log}"

echo "=== Fix bc_plus_1m legacy fw_ prefix ==="
echo "  Corpus dir: $CORPUS_DIR"
echo "  Index dir:  $INDEX_DIR"
echo "  Log file:   $LOG_FILE"

nohup bash -lc '
  stdbuf -oL uv run python -u "'$REPO_ROOT/scripts/bcplus_eval/rename_fw_prefix_in_bcplus_1m.py'" \
    --corpus-dir "'$CORPUS_DIR'" \
    --index-dir "'$INDEX_DIR'"
' >> "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Tail log: tail -f $LOG_FILE"
