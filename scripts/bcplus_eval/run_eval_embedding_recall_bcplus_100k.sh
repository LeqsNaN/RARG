#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

DATASET="${DATASET:-$REPO_ROOT/data/bcplus_qa_sample100.jsonl}"
PARQUET="${PARQUET:-$REPO_ROOT/corpus/browsecomp_plus/data.parquet}"
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/indices/bc_plus_100k}"
# 100k 评测不需要 FineWeb 映射；传一个默认不存在的路径即可跳过。
FINEWEB_SAMPLE="${FINEWEB_SAMPLE:-$REPO_ROOT/data/__skip_fineweb_for_bcplus_100k__.jsonl}"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen3-Embedding-4B}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_embedding_4b}"
DTYPES="${DTYPES:-float16}"
CUDA_DEVICE="${CUDA_DEVICE:-6}"
DEVICE="${DEVICE:-cuda:0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
KS="${KS:-10,20,50,100,1000,10000}"
LIMIT="${LIMIT:-0}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/bcplus_100k_embedding_recall_${TIMESTAMP}.log}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPO_ROOT/logs/bcplus_100k_embedding_recall_${TIMESTAMP}.json}"

mkdir -p "$REPO_ROOT/logs"

if [ ! -f "$PARQUET" ]; then
    echo "ERROR: parquet not found: $PARQUET"
    exit 1
fi

if [ ! -f "$INDEX_DIR/index.faiss" ]; then
    echo "ERROR: index.faiss not found: $INDEX_DIR/index.faiss"
    exit 1
fi

if [ ! -f "$INDEX_DIR/paths.json" ]; then
    echo "ERROR: paths.json not found: $INDEX_DIR/paths.json"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

echo "=== BC+ 100K embedding recall eval ==="
echo "  Dataset:        $DATASET"
echo "  Parquet:        $PARQUET"
echo "  Index dir:      $INDEX_DIR"
echo "  FineWeb sample: $FINEWEB_SAMPLE"
echo "  Model path:     $MODEL_PATH"
echo "  Model type:     $MODEL_TYPE"
echo "  Dtypes:         $DTYPES"
echo "  CUDA device:    $CUDA_DEVICE"
echo "  Device:         $DEVICE"
echo "  Batch size:     $BATCH_SIZE"
echo "  Max model len:  $MAX_MODEL_LEN"
echo "  Recall ks:      $KS"
echo "  Limit:          $LIMIT"
echo "  Output json:    $OUTPUT_JSON"
echo "  Log file:       $LOG_FILE"

"$REPO_ROOT/.venv/bin/python" -u "$REPO_ROOT/scripts/bcplus_eval/eval_embedding_recall_bcplus.py" \
    --dataset "$DATASET" \
    --parquet "$PARQUET" \
    --index-dir "$INDEX_DIR" \
    --fineweb-sample "$FINEWEB_SAMPLE" \
    --model-path "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    --dtypes "$DTYPES" \
    --device "$DEVICE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --batch-size "$BATCH_SIZE" \
    --ks "$KS" \
    --limit "$LIMIT" \
    --output-json "$OUTPUT_JSON" \
    2>&1 | tee "$LOG_FILE"
