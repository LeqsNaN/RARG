#!/usr/bin/env bash
# Run the standalone Python DR-DCI port on BrowseComp-Plus.
#
# This launcher uses only standard OpenAI-compatible configuration:
#   OPENAI_API_KEY   required API key
#   OPENAI_BASE_URL  optional endpoint; defaults to https://api.openai.com/v1
#
# Retrieval is served by RARG's local model_server. Set MODEL_SERVER_PORT to
# reuse a server already started with scripts/start_model_server.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
MODEL="${MODEL:-gpt-5.4-mini}"
THINKING_LEVEL="${THINKING_LEVEL:-medium}"

CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/corpus/bc_plus_100k}"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/indices/bc_plus_100k}"
DATASET="${DATASET:-$REPO_ROOT/data/bcplus_qa_sample100.jsonl}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-$REPO_ROOT/models/Qwen3-Embedding-4B}"
MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-9010}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
BCP_LIMIT="${BCP_LIMIT:-1}"
BCP_QUERY_ID="${BCP_QUERY_ID:-}"
MAX_TURNS="${MAX_TURNS:-300}"
LEVEL3_PERSISTENT_MICRO_COMPACT="${LEVEL3_PERSISTENT_MICRO_COMPACT:-0}"
DR_DCI_RUN_NAME="${DR_DCI_RUN_NAME:-dr_dci_bcplus100k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/dr_dci/$DR_DCI_RUN_NAME}"
VIEW_CACHE_ROOT="${VIEW_CACHE_ROOT:-$REPO_ROOT/outputs/dr_dci_views/$DR_DCI_RUN_NAME}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/dr_dci_$(date '+%Y%m%d_%H%M%S').log}"

if [ -z "$API_KEY" ]; then
    echo "ERROR: Set OPENAI_API_KEY (or API_KEY) before running." >&2
    exit 1
fi
for required in "$CORPUS_DIR" "$INDEX_DIR" "$DATASET" "$EMBED_MODEL_PATH"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: Required path does not exist: $required" >&2
        exit 1
    fi
done
if [ ! -f "$INDEX_DIR/index.faiss" ]; then
    echo "ERROR: FAISS index not found: $INDEX_DIR/index.faiss" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$VIEW_CACHE_ROOT" "$REPO_ROOT/logs"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export EMBED_DEVICE="cuda:0"
export EMBED_INDEX_DIR="$INDEX_DIR"
export EMBED_MODEL_PATH
export EMBED_MODEL_TYPE="${EMBED_MODEL_TYPE:-qwen3_embedding_4b}"
export EMBED_BACKEND="${EMBED_BACKEND:-transformers}"
export EMBED_EMPTY_CACHE_AFTER_ENCODE=1
export EMBED_PYTHON="${EMBED_PYTHON:-$REPO_ROOT/.venv/bin/python}"
export CORPUS_DIR MODEL_SERVER_PORT
export NO_RERANKER=1

# Starts a local retrieval server if no live server with this managed port is
# already registered. No LLM proxy or provider-specific credential handling is
# performed here.
source "$REPO_ROOT/scripts/start_model_server.sh" --port "$MODEL_SERVER_PORT"

CMD=(
    "$REPO_ROOT/.venv/bin/python" -u "$SCRIPT_DIR/run_bcplus_eval.py"
    --dataset "$DATASET"
    --output-root "$OUTPUT_ROOT"
    --view-cache-root "$VIEW_CACHE_ROOT"
    --corpus-dir "$CORPUS_DIR"
    --index-dir "$INDEX_DIR"
    --max-concurrency "$MAX_CONCURRENCY"
    --provider openai
    --model "$MODEL"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --thinking-level "$THINKING_LEVEL"
    --max-turns "$MAX_TURNS"
    --max-output-tokens 128000
    --tools read,bash,pull
    --max-turns-mode abort
    --runtime-context-level level3
    --pull-view-mode hardlink
    --pull-layout root
    --pull-prompt-mode rank_aware
    --pull-materialization-mode root_flat_disclosed
    --pull-min-top-k 300
    --pull-max-top-k 600
    --pull-max-queries 1
    --pull-preview-mode ranked
    --limit "$BCP_LIMIT"
)
if [ -n "$BCP_QUERY_ID" ]; then
    CMD+=(--query-id "$BCP_QUERY_ID")
fi
if [ "$LEVEL3_PERSISTENT_MICRO_COMPACT" = "1" ]; then
    CMD+=(--level3-persistent-micro-compact)
fi

echo "=== Standalone DR-DCI / BrowseComp-Plus ==="
echo "  API endpoint:         $BASE_URL"
echo "  Model:                $MODEL"
echo "  Model server:         127.0.0.1:$MODEL_SERVER_PORT"
echo "  Dataset:              $DATASET"
echo "  Query ID filter:      ${BCP_QUERY_ID:-first $BCP_LIMIT sample(s)}"
echo "  Max turns:            $MAX_TURNS"
echo "  Level3 compact mode:  $([ "$LEVEL3_PERSISTENT_MICRO_COMPACT" = "1" ] && echo persistent || echo request_time)"
echo "  Output root:          $OUTPUT_ROOT"
echo "  View cache root:      $VIEW_CACHE_ROOT"
echo "  Agent log:            $LOG_FILE"

nohup stdbuf -oL -eL "${CMD[@]}" > "$LOG_FILE" 2>&1 &
echo "Agent PID: $!"
echo "Follow progress: tail -f $LOG_FILE"
