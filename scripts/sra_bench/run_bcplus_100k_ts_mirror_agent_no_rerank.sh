#!/usr/bin/env bash
# ==========================================================================
# Run TS-Mirror Python Agent on BrowseComp-Plus (100K corpus) over sample100.
# Full no-rerank ablation:
#   - no embed_recall paragraph reranking
#   - no rg match reranking
# Other retrieval / agent budgets stay aligned with the 100k rerank script.
# ==========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

# ==========================================================================
# LLM
# ==========================================================================
PROVIDER="openai"
MODEL="gpt-5.4-mini"
BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
THINKING_LEVEL="medium"
MAX_OUTPUT_TOKENS=128000
# TEMPERATURE=0.0

# ==========================================================================
# Agent behavior
# ==========================================================================
MAX_TURNS=101
FORCE_ANSWER_TURNS=100
KEEP_TOOL_CALLS=40
COMPACTION_THRESHOLD=230000

# ==========================================================================
# Corpus / retrieval
# ==========================================================================
CORPUS_DIR="$REPO_ROOT/corpus/bc_plus_100k"
INDEX_DIR="$REPO_ROOT/data/indices/bc_plus_100k"
DATASET="$REPO_ROOT/data/bcplus_qa_sample100.jsonl"
EMBED_MODEL_PATH="$REPO_ROOT/models/Qwen3-Embedding-4B"
EMBED_MODEL_TYPE="qwen3_embedding_4b"
EMBED_BACKEND="transformers"
EMBED_TOP_K=10000
SCOPE_SIZE_PREFIX="NA100"
GREP_MAX_LINE_LENGTH=1000
RG_MAX_MATCHES=30
RG_RERANK_MATCH=0
RG_RERANK_CANDIDATE_NUM=30
RG_RERANK_TIMEOUT=30
PARAGRAPH_RERANK_MODEL="none"
NO_RERANKER=1
USE_CONTEXTUAL_BASH_TOOL=0

# ==========================================================================
# Prompt
# ==========================================================================
APPEND_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/bcplus/embed_recall_native.txt"
SYSTEM_PROMPT_FILE=""

# ==========================================================================
# Runtime / device
# ==========================================================================
CUDA_DEVICE=4
MODEL_SERVER_PORT=9013

# ==========================================================================
# Output
# ==========================================================================
OUTPUT_NAME="${PROVIDER}_${MODEL}_100k_ts_mirror_agent_sample100_no_rerank"
OUTPUT_ROOT="$REPO_ROOT/outputs/bcplus_eval/$OUTPUT_NAME"
LOG_FILE="$REPO_ROOT/logs/bcplus_100k_ts_mirror_agent_no_rerank_$(date '+%Y%m%d_%H%M%S').log"
MAX_CONCURRENCY=1
RUN_IN_FOREGROUND="${RUN_IN_FOREGROUND:-0}"
KILL_MODEL_SERVER_ON_EXIT="${KILL_MODEL_SERVER_ON_EXIT:-1}"

mkdir -p "$OUTPUT_ROOT" "$REPO_ROOT/logs"

if [ -z "$API_KEY" ]; then
    echo "ERROR: Set OPENAI_API_KEY (or API_KEY) before running."
    exit 1
fi

# ==========================================================================
# Infrastructure checks
# ==========================================================================

if [ ! -f "$INDEX_DIR/index.faiss" ]; then
    echo "ERROR: FAISS index not found at $INDEX_DIR/index.faiss"
    exit 1
fi

# ==========================================================================
# Environment
# ==========================================================================
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export EMBED_DEVICE="cuda:0"
export EMBED_INDEX_DIR="$INDEX_DIR"
export EMBED_MODEL_PATH="$EMBED_MODEL_PATH"
export EMBED_MODEL_TYPE="$EMBED_MODEL_TYPE"
export EMBED_BACKEND="$EMBED_BACKEND"
export EMBED_EMPTY_CACHE_AFTER_ENCODE=1
export EMBED_PYTHON="$REPO_ROOT/.venv/bin/python"
export EMBED_TOP_K="$EMBED_TOP_K"
export SCOPE_SIZE_PREFIX="$SCOPE_SIZE_PREFIX"
export CORPUS_DIR="$CORPUS_DIR"
export MODEL_SERVER_SCRIPT="$REPO_ROOT/scripts/model_server.py"
export MODEL_SERVER_PORT="$MODEL_SERVER_PORT"
export RG_MAX_MATCHES="$RG_MAX_MATCHES"
export GREP_MAX_LINE_LENGTH="$GREP_MAX_LINE_LENGTH"
export RG_RERANK_MATCH="$RG_RERANK_MATCH"
export RG_RERANK_CANDIDATE_NUM="$RG_RERANK_CANDIDATE_NUM"
export RG_RERANK_TIMEOUT="$RG_RERANK_TIMEOUT"
export PARAGRAPH_RERANK_MODEL="$PARAGRAPH_RERANK_MODEL"
export NO_RERANKER="$NO_RERANKER"
export USE_CONTEXTUAL_BASH_TOOL="$USE_CONTEXTUAL_BASH_TOOL"

source "$REPO_ROOT/scripts/start_model_server.sh"

# ==========================================================================
# Print config
# ==========================================================================
echo "=== BrowseComp-Plus 100K — TS-Mirror Agent No-rerank Ablation ==="
echo "  Dataset:              $DATASET"
echo "  Provider:             $PROVIDER"
echo "  Model:                $MODEL"
echo "  Base URL:             $BASE_URL"
echo "  Thinking:             $THINKING_LEVEL"
echo "  Max turns:            $MAX_TURNS"
echo "  Force answer:         $FORCE_ANSWER_TURNS"
echo "  Keep tool calls:      $KEEP_TOOL_CALLS"
echo "  Compact thresh:       $COMPACTION_THRESHOLD"
echo "  Temperature:          ${TEMPERATURE:-<not set>}"
echo "  Embed top_k:          $EMBED_TOP_K"
echo "  Embed model:          $EMBED_MODEL_PATH"
echo "  Embed model type:     $EMBED_MODEL_TYPE"
echo "  Embed backend:        $EMBED_BACKEND"
echo "  Paragraph rerank:     $PARAGRAPH_RERANK_MODEL"
echo "  RG max matches:       $RG_MAX_MATCHES"
echo "  RG rerank:            $RG_RERANK_MATCH"
echo "  No reranker engine:   $NO_RERANKER"
echo "  Contextual bash:      $USE_CONTEXTUAL_BASH_TOOL"
echo "  RG candidates:        $RG_RERANK_CANDIDATE_NUM"
echo "  RG timeout:           $RG_RERANK_TIMEOUT"
echo "  Line budget:          $GREP_MAX_LINE_LENGTH"
echo "  Scope prefix:         $SCOPE_SIZE_PREFIX"
echo "  Output root:          $OUTPUT_ROOT"
echo "  Log file:             $LOG_FILE"
echo "  Model server port:    $MODEL_SERVER_PORT"
echo ""

CMD=(
    "$REPO_ROOT/.venv/bin/python" -u "$REPO_ROOT/scripts/bcplus_eval/run_ts_mirror_agent_eval.py"
    --dataset "$DATASET"
    --output-root "$OUTPUT_ROOT"
    --corpus-dir "$CORPUS_DIR"
    --max-concurrency "$MAX_CONCURRENCY"
    --provider "$PROVIDER"
    --model "$MODEL"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
    --max-output-tokens "$MAX_OUTPUT_TOKENS"
    --thinking-level "$THINKING_LEVEL"
    --max-turns "$MAX_TURNS"
    --force-answer-turns "$FORCE_ANSWER_TURNS"
    --keep-tool-calls "$KEEP_TOOL_CALLS"
    --compaction-threshold "$COMPACTION_THRESHOLD"
    --index-dir "$INDEX_DIR"
    --device "cuda:0"
    --embed-top-k "$EMBED_TOP_K"
    --append-system-prompt-file "$APPEND_SYSTEM_PROMPT_FILE"
    --no-reranker
)

if [ -n "$SYSTEM_PROMPT_FILE" ]; then
    CMD+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
fi
if [ -n "${TEMPERATURE:-}" ]; then
    CMD+=(--temperature "$TEMPERATURE")
fi
if [ -n "${USE_USAGE_BASED_COMPACTION:-}" ]; then
    CMD+=(--use-usage-based-compaction)
fi
if [ -n "${USAGE_BASED_COMPACTION_MIN_TOKENS:-}" ]; then
    CMD+=(--usage-based-compaction-min-tokens "$USAGE_BASED_COMPACTION_MIN_TOKENS")
fi
if [ -n "${USAGE_BASED_COMPACTION_MAX_TOKENS:-}" ]; then
    CMD+=(--usage-based-compaction-max-tokens "$USAGE_BASED_COMPACTION_MAX_TOKENS")
fi

if [ "$RUN_IN_FOREGROUND" = "1" ]; then
    if [ "$KILL_MODEL_SERVER_ON_EXIT" = "1" ]; then
        "$REPO_ROOT/scripts/run_with_model_server_cleanup.sh" \
            --pid-file "$REPO_ROOT/logs/model_server_${MODEL_SERVER_PORT}.pid" \
            -- "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
        cmd_status=${PIPESTATUS[0]}
    else
        stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
        cmd_status=${PIPESTATUS[0]}
    fi
    exit "$cmd_status"
fi

if [ "$KILL_MODEL_SERVER_ON_EXIT" = "1" ]; then
    nohup "$REPO_ROOT/scripts/run_with_model_server_cleanup.sh" \
            --pid-file "$REPO_ROOT/logs/model_server_${MODEL_SERVER_PORT}.pid" \
            -- "${CMD[@]}" > "$LOG_FILE" 2>&1 &
else
    nohup stdbuf -oL -eL "${CMD[@]}" > "$LOG_FILE" 2>&1 &
fi

echo "PID: $!"
echo "Tail log: tail -f $LOG_FILE"
