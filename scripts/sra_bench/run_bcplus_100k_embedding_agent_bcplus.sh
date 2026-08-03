#!/usr/bin/env bash
# ==========================================================================
# BC+ embedding agent adaptation inside DCI-Agent-Lite
# - uses Qwen3-Embedding-4B FAISS index on 100k corpus
# - uses OpenAI-compatible API (gpt-5.4-mini, medium thinking)
# - retry / in-sample resume / realtime logs / token usage persistence
# - ts_mirror_agent-style compaction at threshold 230000 + max-turn force answer
# ==========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

PROVIDER="openai"
MODEL="gpt-5.4-mini"
BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
THINKING_LEVEL="${THINKING_LEVEL:-medium}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-128000}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
FORCE_ANSWER_AT_LIMIT="${FORCE_ANSWER_AT_LIMIT:-1}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-230000}"
KEEP_RECENT_TOOL_RESULTS="${KEEP_RECENT_TOOL_RESULTS:-50}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_DELAY_BASE="${RETRY_DELAY_BASE:-2}"
RETRY_DELAY_MAX="${RETRY_DELAY_MAX:-30}"

DATASET="${DATASET:-$REPO_ROOT/data/bcplus_qa_sample100.jsonl}"
CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/corpus/bc_plus_100k}"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/indices/bc_plus_100k}"
SEARCH_TOP_K="${SEARCH_TOP_K:-5}"
SNIPPET_MAX_TOKENS="${SNIPPET_MAX_TOKENS:-512}"
QUERY_TEMPLATE_FILE="${QUERY_TEMPLATE_FILE:-$REPO_ROOT/prompts/bcplus/embedding_search_agent_no_get_document.txt}"

EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-$REPO_ROOT/models/Qwen3-Embedding-4B}"
EMBED_MODEL_TYPE="${EMBED_MODEL_TYPE:-qwen3_embedding_4b}"
EMBED_BACKEND="${EMBED_BACKEND:-transformers}"
DEVICE="${DEVICE:-cuda:7}"
EMBED_MAX_MODEL_LEN="${EMBED_MAX_MODEL_LEN:-512}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-16}"
QUERY_INSTRUCTION="${QUERY_INSTRUCTION:-Given a web search query, retrieve relevant passages that answer the query}"
EMBED_EMPTY_CACHE_AFTER_ENCODE="${EMBED_EMPTY_CACHE_AFTER_ENCODE:-1}"

SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
APPEND_SYSTEM_PROMPT_FILE="${APPEND_SYSTEM_PROMPT_FILE:-}"
LIMIT="${LIMIT:-}"

OUTPUT_NAME="${OUTPUT_NAME:-openai_gpt-5.4-mini_100k_embedding_agent_bcplus_medium}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/bcplus_eval/$OUTPUT_NAME}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/${OUTPUT_NAME}_$(date '+%Y%m%d_%H%M%S').log}"

mkdir -p "$OUTPUT_ROOT" "$REPO_ROOT/logs"

if [ ! -f "$INDEX_DIR/index.faiss" ]; then
  echo "ERROR: missing FAISS index: $INDEX_DIR/index.faiss" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" -u "$REPO_ROOT/scripts/bcplus_eval/run_embedding_agent_bcplus_eval.py"
  --dataset "$DATASET"
  --output-root "$OUTPUT_ROOT"
  --corpus-dir "$CORPUS_DIR"
  --index-dir "$INDEX_DIR"
  --provider "$PROVIDER"
  --model "$MODEL"
  --base-url "$BASE_URL"
  --api-key "$API_KEY"
  --thinking-level "$THINKING_LEVEL"
  --temperature "$TEMPERATURE"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --max-iterations "$MAX_ITERATIONS"
  --force-answer-at-limit "$FORCE_ANSWER_AT_LIMIT"
  --max-context-tokens "$MAX_CONTEXT_TOKENS"
  --keep-recent-tool-results "$KEEP_RECENT_TOOL_RESULTS"
  --max-retries "$MAX_RETRIES"
  --retry-delay-base "$RETRY_DELAY_BASE"
  --retry-delay-max "$RETRY_DELAY_MAX"
  --search-top-k "$SEARCH_TOP_K"
  --snippet-max-tokens "$SNIPPET_MAX_TOKENS"
  --embed-model-path "$EMBED_MODEL_PATH"
  --embed-model-type "$EMBED_MODEL_TYPE"
  --embed-backend "$EMBED_BACKEND"
  --device "$DEVICE"
  --embed-max-model-len "$EMBED_MAX_MODEL_LEN"
  --embed-batch-size "$EMBED_BATCH_SIZE"
  --query-instruction "$QUERY_INSTRUCTION"
  --query-template-file "$QUERY_TEMPLATE_FILE"
)

if [ "$EMBED_EMPTY_CACHE_AFTER_ENCODE" = "1" ]; then
  CMD+=(--embed-empty-cache-after-encode)
fi
if [ -n "$SYSTEM_PROMPT_FILE" ]; then
  CMD+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
fi
if [ -n "$APPEND_SYSTEM_PROMPT_FILE" ]; then
  CMD+=(--append-system-prompt-file "$APPEND_SYSTEM_PROMPT_FILE")
fi
if [ -n "$LIMIT" ]; then
  CMD+=(--limit "$LIMIT")
fi

nohup stdbuf -oL -eL "${CMD[@]}" > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Log: $LOG_FILE"
echo "Tail log: tail -f $LOG_FILE"
