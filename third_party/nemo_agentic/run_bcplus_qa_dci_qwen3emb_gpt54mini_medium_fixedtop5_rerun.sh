#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
RARG_ROOT="$REPO_ROOT"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/bcplus_qa_sample100.jsonl}"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/indices/bc_plus_100k}"
CORPUS_DIR="${CORPUS_DIR:-$REPO_ROOT/corpus/bc_plus_100k}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen3-Embedding-4B}"
EMBED_MODEL_TYPE="${EMBED_MODEL_TYPE:-qwen3_embedding_4b}"
EMBED_BACKEND="${EMBED_BACKEND:-transformers}"
EMBED_DEVICE="${EMBED_DEVICE:-cuda:3}"
LLM_MODEL="${LLM_MODEL:-gpt-5.4-mini}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
MAX_STEPS="${MAX_STEPS:-100}"
INITIAL_RETRIEVAL_TOP_K="${INITIAL_RETRIEVAL_TOP_K:-5}"
FALLBACK_TOP_K="${FALLBACK_TOP_K:-5}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LIMIT="${LIMIT:-}"
RETRIEVED_TEXT_MAX_CHARS="${RETRIEVED_TEXT_MAX_CHARS:-0}"
RETRIEVED_TEXT_MAX_TOKENS="${RETRIEVED_TEXT_MAX_TOKENS:-512}"
QWEN3_TOKENIZER_PATH="${QWEN3_TOKENIZER_PATH:-$REPO_ROOT/models/Qwen3-8B}"
MAX_RETURN_DOCS="${MAX_RETURN_DOCS:-5}"
FALLBACK_DROP_DOCS="${FALLBACK_DROP_DOCS:-5}"
FALLBACK_MIN_DOCS="${FALLBACK_MIN_DOCS:-1}"
USE_ORIGINAL_RETRIEVE_GUARANTEE="${USE_ORIGINAL_RETRIEVE_GUARANTEE:-1}"
ALLOW_MODEL_TOPK="${ALLOW_MODEL_TOPK:-0}"
RUN_IN_FOREGROUND="${RUN_IN_FOREGROUND:-0}"
RUN_NAME="${RUN_NAME:-bcplus_qa_qwen3emb_gpt-5.4-mini_medium_fixedtop5_rerun_sample100}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/nemo_agentic_results_dci_bcplus_qa_qwen3emb_gpt-5.4-mini_medium_fixedtop5_rerun}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}.log}"

CMD=(
  "$PYTHON_BIN" -u "$REPO_ROOT/run_agentic_bcplus_qa_dci.py"
  --dataset_path "$DATASET_PATH"
  --index_dir "$INDEX_DIR"
  --corpus_dir "$CORPUS_DIR"
  --model_path "$MODEL_PATH"
  --embed_model_type "$EMBED_MODEL_TYPE"
  --embed_backend "$EMBED_BACKEND"
  --embed_device "$EMBED_DEVICE"
  --max_length "$MAX_LENGTH"
  --llm_model "$LLM_MODEL"
  --llm_base_url "$LLM_BASE_URL"
  --reasoning_effort "$REASONING_EFFORT"
  --max_steps "$MAX_STEPS"
  --initial_retrieval_top_k "$INITIAL_RETRIEVAL_TOP_K"
  --fallback_top_k "$FALLBACK_TOP_K"
  --run_name "$RUN_NAME"
  --output_dir "$OUTPUT_DIR"
  --retrieved_text_max_chars "$RETRIEVED_TEXT_MAX_CHARS"
  --retrieved_text_max_tokens "$RETRIEVED_TEXT_MAX_TOKENS"
  --qwen3_tokenizer_path "$QWEN3_TOKENIZER_PATH"
  --max_return_docs "$MAX_RETURN_DOCS"
  --fallback_drop_docs "$FALLBACK_DROP_DOCS"
  --fallback_min_docs "$FALLBACK_MIN_DOCS"
)

if [ -n "$LIMIT" ]; then
  CMD+=(--limit "$LIMIT")
fi

if [ "$USE_ORIGINAL_RETRIEVE_GUARANTEE" = "1" ]; then
  CMD+=(--use_original_retrieve_guarantee)
fi
# Fixed-top5 rerun: do NOT expose model-controlled top_k.
if [ "$ALLOW_MODEL_TOPK" = "1" ]; then
  CMD+=(--allow_model_topk)
fi

if [ "$RUN_IN_FOREGROUND" = "1" ]; then
  stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
  nohup stdbuf -oL -eL "${CMD[@]}" > "$LOG_FILE" 2>&1 &
  echo "PID: $!"
  echo "Log: $LOG_FILE"
  echo "Tail log: tail -f $LOG_FILE"
fi
