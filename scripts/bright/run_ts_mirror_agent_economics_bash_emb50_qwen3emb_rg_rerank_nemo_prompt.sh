#!/usr/bin/env bash
# ==========================================================================
# Run TS-Mirror BRIGHT agent on the full Economics subset.
#
# Mode:
#   - use NeMo-style BRIGHT system prompt
#   - use NeMo-style forced-answer prompt
#   - embed_recall uses embedding paragraph reranking on top-50 recalled docs
#   - rg match reranking uses a separate Qwen3 embedding model
#   - standard Bash tool
#   - no doc-level QR reranker is loaded
#   - this script owns model_server on MODEL_SERVER_PORT and can kill stale one first
# ==========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-5.4-mini}"
BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
THINKING_LEVEL="${THINKING_LEVEL:-medium}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-128000}"

MAX_TURNS="${MAX_TURNS:-41}"
FORCE_ANSWER_TURNS="${FORCE_ANSWER_TURNS:-40}"
KEEP_TOOL_CALLS="${KEEP_TOOL_CALLS:-40}"
COMPACTION_THRESHOLD="${COMPACTION_THRESHOLD:-230000}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MAX_RETRIEVED_DOCS="${MAX_RETRIEVED_DOCS:-10}"

SUBSET="${SUBSET:-economics}"
CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/bright_corpus/$SUBSET}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-$REPO_ROOT/models/llama-nv-embed-reasoning-3b}"
EMBED_MODEL_TYPE="${EMBED_MODEL_TYPE:-llama_nv_embed_reasoning_3b}"
INDEX_ROOT_NAME="${INDEX_ROOT_NAME:-bright_llama_nv_reasoning_3b}"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/indices/${INDEX_ROOT_NAME}/$SUBSET}"
DATASET="${DATASET:-$REPO_ROOT/data/bright/queries/${SUBSET}.jsonl}"
EMBED_TOP_K="${EMBED_TOP_K:-10000}"
EMBED_MAX_TOKENS="${EMBED_MAX_TOKENS:-8192}"
EMBED_RERANK_MAX_TOKENS="${EMBED_RERANK_MAX_TOKENS:-8192}"
EMBED_RERANK_BATCH_SIZE="${EMBED_RERANK_BATCH_SIZE:-1}"
EMBED_EMPTY_CACHE_AFTER_ENCODE="${EMBED_EMPTY_CACHE_AFTER_ENCODE:-1}"
SCOPE_SIZE_PREFIX="${SCOPE_SIZE_PREFIX:-ECQ3}"
SCOPE_PATH_DOT_PREFIX="${SCOPE_PATH_DOT_PREFIX:-0}"
GREP_MAX_LINE_LENGTH="${GREP_MAX_LINE_LENGTH:-500}"
RG_MAX_MATCHES="${RG_MAX_MATCHES:-60}"
RG_RERANK_MATCH="${RG_RERANK_MATCH:-1}"
RG_RERANK_MODEL="${RG_RERANK_MODEL:-qwen3emb}"
RG_EMBED_MODEL_PATH="${RG_EMBED_MODEL_PATH:-$REPO_ROOT/models/Qwen3-Embedding-4B}"
RG_EMBED_MODEL_TYPE="${RG_EMBED_MODEL_TYPE:-qwen3_embedding_4b}"
RG_EMBED_BACKEND="${RG_EMBED_BACKEND:-transformers}"
RG_EMBED_MAX_TOKENS="${RG_EMBED_MAX_TOKENS:-8192}"
RG_EMBED_RERANK_MAX_TOKENS="${RG_EMBED_RERANK_MAX_TOKENS:-8192}"
RG_EMBED_RERANK_BATCH_SIZE="${RG_EMBED_RERANK_BATCH_SIZE:-1}"
RG_RERANK_CANDIDATE_NUM="${RG_RERANK_CANDIDATE_NUM:-500}"
RG_RERANK_TIMEOUT="${RG_RERANK_TIMEOUT:-500}"
PARAGRAPH_RERANK_MODEL="${PARAGRAPH_RERANK_MODEL:-emb}"
PARAGRAPH_RERANK_DOC_LIMIT="${PARAGRAPH_RERANK_DOC_LIMIT:-50}"
NO_RERANKER="${NO_RERANKER:-1}"
USE_CONTEXTUAL_BASH_TOOL="${USE_CONTEXTUAL_BASH_TOOL:-0}"
QUERY_INSTRUCTION="${QUERY_INSTRUCTION:-Given an Economics post, retrieve relevant passages that help answer the post.}"

SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$REPO_ROOT/prompts/bright/nemo_style_system_prompt.txt}"
FORCE_ANSWER_PROMPT_FILE="${FORCE_ANSWER_PROMPT_FILE:-$REPO_ROOT/prompts/bright/nemo_style_force_answer_prompt.txt}"

CUDA_DEVICE="${CUDA_DEVICE:-2}"
RG_EMBED_CUDA_DEVICE="${RG_EMBED_CUDA_DEVICE:-3}"
MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-9114}"
KILL_MODEL_SERVER_BEFORE_START="${KILL_MODEL_SERVER_BEFORE_START:-1}"
KILL_MODEL_SERVER_ON_EXIT="${KILL_MODEL_SERVER_ON_EXIT:-1}"
RUN_IN_FOREGROUND="${RUN_IN_FOREGROUND:-0}"

kill_model_server_port() {
    local port="$1"
    local pid_file="$REPO_ROOT/logs/model_server_${port}.pid"
    if [ -f "$pid_file" ]; then
        local old_pid
        old_pid="$(cat "$pid_file" 2>/dev/null || true)"
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            echo "Stopping model_server on port $port (PID $old_pid)..."
            kill "$old_pid" 2>/dev/null || true
            for _ in $(seq 1 30); do
                if ! kill -0 "$old_pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$old_pid" 2>/dev/null; then
                echo "Force killing model_server on port $port (PID $old_pid)..."
                kill -9 "$old_pid" 2>/dev/null || true
            fi
        fi
        rm -f "$pid_file"
    fi
}

EMBED_TAG="${EMBED_TAG:-$(python - <<'PY2'
import os
name = os.environ.get("INDEX_ROOT_NAME", "bright").strip()
print(name.replace("bright_", "").replace("/", "_"))
PY2
)}"
OUTPUT_NAME="${OUTPUT_NAME:-${PROVIDER}_${MODEL}_bright_${SUBSET}_ts_mirror_agent_bash_emb50_qwen3emb_rg_rerank_nemo_prompt_${EMBED_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/bright_ts_mirror_agent/$OUTPUT_NAME}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/bright_${SUBSET}_ts_mirror_agent_bash_emb50_qwen3emb_rg_rerank_nemo_prompt_$(date '+%Y%m%d_%H%M%S').log}"

mkdir -p "$OUTPUT_ROOT" "$REPO_ROOT/logs"

if [ -z "$API_KEY" ]; then
    echo "ERROR: Set OPENAI_API_KEY (or API_KEY) before running."
    exit 1
fi

if [ ! -f "$INDEX_DIR/index.faiss" ]; then
    echo "ERROR: FAISS index not found at $INDEX_DIR/index.faiss"
    exit 1
fi

if [ ! -f "$SYSTEM_PROMPT_FILE" ]; then
    echo "ERROR: system prompt file not found at $SYSTEM_PROMPT_FILE"
    exit 1
fi

if [ ! -f "$FORCE_ANSWER_PROMPT_FILE" ]; then
    echo "ERROR: force-answer prompt file not found at $FORCE_ANSWER_PROMPT_FILE"
    exit 1
fi

if [ "$KILL_MODEL_SERVER_BEFORE_START" = "1" ]; then
    kill_model_server_port "$MODEL_SERVER_PORT"
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE,$RG_EMBED_CUDA_DEVICE"
export EMBED_DEVICE="cuda:0"
export RG_EMBED_DEVICE="cuda:1"
export EMBED_INDEX_DIR="$INDEX_DIR"
export EMBED_MODEL_PATH="$EMBED_MODEL_PATH"
export EMBED_MODEL_TYPE="$EMBED_MODEL_TYPE"
export EMBED_PYTHON="$REPO_ROOT/.venv/bin/python"
export EMBED_TOP_K="$EMBED_TOP_K"
export EMBED_MAX_TOKENS="$EMBED_MAX_TOKENS"
export EMBED_RERANK_MAX_TOKENS="$EMBED_RERANK_MAX_TOKENS"
export EMBED_RERANK_BATCH_SIZE="$EMBED_RERANK_BATCH_SIZE"
export EMBED_EMPTY_CACHE_AFTER_ENCODE="$EMBED_EMPTY_CACHE_AFTER_ENCODE"
export SCOPE_SIZE_PREFIX="$SCOPE_SIZE_PREFIX"
export SCOPE_PATH_DOT_PREFIX="$SCOPE_PATH_DOT_PREFIX"
export CORPUS_DIR="$CORPUS_DIR"
export MODEL_SERVER_SCRIPT="$REPO_ROOT/scripts/model_server_bright.py"
export MODEL_SERVER_PORT="$MODEL_SERVER_PORT"
export QUERY_INSTRUCTION="$QUERY_INSTRUCTION"
export RG_MAX_MATCHES="$RG_MAX_MATCHES"
export GREP_MAX_LINE_LENGTH="$GREP_MAX_LINE_LENGTH"
export RG_RERANK_MATCH="$RG_RERANK_MATCH"
export RG_RERANK_MODEL="$RG_RERANK_MODEL"
export RG_EMBED_MODEL_PATH="$RG_EMBED_MODEL_PATH"
export RG_EMBED_MODEL_TYPE="$RG_EMBED_MODEL_TYPE"
export RG_EMBED_BACKEND="$RG_EMBED_BACKEND"
export RG_EMBED_DEVICE="$RG_EMBED_DEVICE"
export RG_EMBED_MAX_TOKENS="$RG_EMBED_MAX_TOKENS"
export RG_EMBED_RERANK_MAX_TOKENS="$RG_EMBED_RERANK_MAX_TOKENS"
export RG_EMBED_RERANK_BATCH_SIZE="$RG_EMBED_RERANK_BATCH_SIZE"
export RG_RERANK_CANDIDATE_NUM="$RG_RERANK_CANDIDATE_NUM"
export RG_RERANK_TIMEOUT="$RG_RERANK_TIMEOUT"
export PARAGRAPH_RERANK_MODEL="$PARAGRAPH_RERANK_MODEL"
export PARAGRAPH_RERANK_DOC_LIMIT="$PARAGRAPH_RERANK_DOC_LIMIT"
export NO_RERANKER="$NO_RERANKER"
export USE_CONTEXTUAL_BASH_TOOL="$USE_CONTEXTUAL_BASH_TOOL"

source "$REPO_ROOT/scripts/start_model_server.sh"


echo "=== BRIGHT ${SUBSET} — TS-Mirror Agent Bash Emb50 + Qwen3Emb RG Rerank + NeMo Prompt ==="
echo "  Dataset:            $DATASET"
echo "  Corpus:             $CORPUS_DIR"
echo "  Index:              $INDEX_DIR"
echo "  Embed model:        $EMBED_MODEL_PATH"
echo "  Embed type:         $EMBED_MODEL_TYPE"
echo "  Embed q max:        $EMBED_MAX_TOKENS"
echo "  Embed doc max:      $EMBED_RERANK_MAX_TOKENS"
echo "  Embed bs:           $EMBED_RERANK_BATCH_SIZE"
echo "  Empty cache:        $EMBED_EMPTY_CACHE_AFTER_ENCODE"
echo "  Scope prefix:       $SCOPE_SIZE_PREFIX"
echo "  Scope dot prefix:   $SCOPE_PATH_DOT_PREFIX"
echo "  Query instr:        $QUERY_INSTRUCTION"
echo "  System prompt:      $SYSTEM_PROMPT_FILE"
echo "  Force prompt:       $FORCE_ANSWER_PROMPT_FILE"
echo "  Paragraph rr:       $PARAGRAPH_RERANK_MODEL"
echo "  Paragraph doc lim:  $PARAGRAPH_RERANK_DOC_LIMIT"
echo "  RG rerank:          $RG_RERANK_MATCH"
echo "  RG rr model:        $RG_RERANK_MODEL"
echo "  RG embed model:     $RG_EMBED_MODEL_PATH"
echo "  RG embed type:      $RG_EMBED_MODEL_TYPE"
echo "  RG embed backend:   $RG_EMBED_BACKEND"
echo "  RG embed device:    $RG_EMBED_DEVICE"
echo "  RG candidates:      $RG_RERANK_CANDIDATE_NUM"
echo "  RG timeout:         $RG_RERANK_TIMEOUT"
echo "  No reranker:        $NO_RERANKER"
echo "  Contextual bash:    $USE_CONTEXTUAL_BASH_TOOL"
echo "  Output root:        $OUTPUT_ROOT"
echo "  Log file:           $LOG_FILE"
echo "  Model server port:  $MODEL_SERVER_PORT"

CMD=(
    "$REPO_ROOT/.venv/bin/python" -u "$REPO_ROOT/scripts/bcplus_eval/run_ts_mirror_agent_bright_eval.py"
    --dataset "$DATASET"
    --output-root "$OUTPUT_ROOT"
    --corpus-dir "$CORPUS_DIR"
    --subset "$SUBSET"
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
    --embed-model-path "$EMBED_MODEL_PATH"
    --device "cuda:0"
    --embed-top-k "$EMBED_TOP_K"
    --query-instruction "$QUERY_INSTRUCTION"
    --max-retrieved-docs "$MAX_RETRIEVED_DOCS"
    --system-prompt-file "$SYSTEM_PROMPT_FILE"
    --force-answer-prompt-file "$FORCE_ANSWER_PROMPT_FILE"
    --no-reranker
)

if [ -n "${TEMPERATURE:-}" ]; then
    CMD+=(--temperature "$TEMPERATURE")
fi

if [ "$RUN_IN_FOREGROUND" = "1" ]; then
    stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    cmd_status=${PIPESTATUS[0]}
    if [ "$KILL_MODEL_SERVER_ON_EXIT" = "1" ]; then
        kill_model_server_port "$MODEL_SERVER_PORT"
    fi
    exit "$cmd_status"
fi

cmd_escaped="stdbuf -oL -eL "
for arg in "${CMD[@]}"; do
    printf -v q '%q' "$arg"
    cmd_escaped+="$q "
done

if [ "$KILL_MODEL_SERVER_ON_EXIT" = "1" ]; then
    cleanup_script="$(mktemp /tmp/rarg_bright_cleanup_XXXXXX.sh)"
    cat > "$cleanup_script" <<EOF
#!/usr/bin/env bash
set -uo pipefail
${cmd_escaped}
cmd_status=$?
pid_file="$REPO_ROOT/logs/model_server_${MODEL_SERVER_PORT}.pid"
if [ -f "$pid_file" ]; then
    old_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        kill "$old_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            if ! kill -0 "$old_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$old_pid" 2>/dev/null; then
            kill -9 "$old_pid" 2>/dev/null || true
        fi
    fi
    rm -f "$pid_file"
fi
rm -f "$cleanup_script"
exit "$cmd_status"
EOF
    chmod +x "$cleanup_script"
    nohup "$cleanup_script" > "$LOG_FILE" 2>&1 &
else
    nohup stdbuf -oL -eL "${CMD[@]}" > "$LOG_FILE" 2>&1 &
fi

echo "PID: $!"
echo "Tail log: tail -f $LOG_FILE"
