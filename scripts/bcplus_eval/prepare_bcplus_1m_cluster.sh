#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/activate.sh"
cd "$REPO_ROOT"

FINEWEB_PARQUET_DIR="${FINEWEB_PARQUET_DIR:-$REPO_ROOT/data/fineweb_edu_10bt/sample/10BT}"
FINEWEB_SAMPLE_JSONL="${FINEWEB_SAMPLE_JSONL:-$REPO_ROOT/data/fineweb_edu_10bt_sample900k_min8000.jsonl}"
SOURCE_BROWSECOMP_DIR="${SOURCE_BROWSECOMP_DIR:-$REPO_ROOT/corpus/browsecomp_plus}"
OUTPUT_CORPUS_DIR="${OUTPUT_CORPUS_DIR:-$REPO_ROOT/corpus/bc_plus_1m}"
INDEX_OUTPUT_DIR="${INDEX_OUTPUT_DIR:-$REPO_ROOT/data/indices/bc_plus_1m}"

SAMPLE_SIZE="${SAMPLE_SIZE:-900000}"
SEED="${SEED:-42}"
MIN_TEXT_CHARS="${MIN_TEXT_CHARS:-8000}"
EMBED_MODEL="${EMBED_MODEL:-$REPO_ROOT/models/Qwen3-Embedding-4B}"
INDEX_MAX_CHARS="${INDEX_MAX_CHARS:-0}"
INDEX_MAX_MODEL_LEN="${INDEX_MAX_MODEL_LEN:-4096}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-64}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
INDEX_TP="${INDEX_TP:-4}"
INDEX_GPU_MEMORY_UTILIZATION="${INDEX_GPU_MEMORY_UTILIZATION:-0.9}"

mkdir -p "$REPO_ROOT/logs"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/prepare_bcplus_1m_$(date '+%Y%m%d_%H%M%S').log}"

echo "=== Rebuild BrowseComp-Plus 1M (standalone corpus) ==="
echo "  FineWeb parquet:   $FINEWEB_PARQUET_DIR"
echo "  FineWeb sample:    $FINEWEB_SAMPLE_JSONL"
echo "  BrowseComp source: $SOURCE_BROWSECOMP_DIR"
echo "  Output corpus:     $OUTPUT_CORPUS_DIR"
echo "  Output index:      $INDEX_OUTPUT_DIR"
echo "  Sample size:       $SAMPLE_SIZE"
echo "  Seed:              $SEED"
echo "  Min text chars:    >= $MIN_TEXT_CHARS"
echo "  Embed model:       $EMBED_MODEL"
echo "  CUDA devices:      $CUDA_VISIBLE_DEVICES"
echo "  Tensor parallel:   $INDEX_TP"
echo "  Log:               $LOG_FILE"

nohup bash -lc '
  export CUDA_VISIBLE_DEVICES="'$CUDA_VISIBLE_DEVICES'"

  stdbuf -oL uv run python -u "'$REPO_ROOT/scripts/bcplus_eval/sample_fineweb_from_parquet.py'" \
    --input-dir "'$FINEWEB_PARQUET_DIR'" \
    --output "'$FINEWEB_SAMPLE_JSONL'" \
    --sample-size "'$SAMPLE_SIZE'" \
    --seed "'$SEED'" \
    --min-text-chars "'$MIN_TEXT_CHARS'"

  stdbuf -oL uv run python -u "'$REPO_ROOT/scripts/bcplus_eval/create_bcplus_1m_corpus.py'" \
    --source-browsecomp "'$SOURCE_BROWSECOMP_DIR'" \
    --fineweb-sample "'$FINEWEB_SAMPLE_JSONL'" \
    --output "'$OUTPUT_CORPUS_DIR'" \
    --force \
    --clean-output \
    --build-index \
    --index-output-dir "'$INDEX_OUTPUT_DIR'" \
    --clean-index-output \
    --embed-python "'$REPO_ROOT/.venv/bin/python'" \
    --embed-model "'$EMBED_MODEL'" \
    --index-max-chars "'$INDEX_MAX_CHARS'" \
    --index-max-model-len "'$INDEX_MAX_MODEL_LEN'" \
    --index-batch-size "'$INDEX_BATCH_SIZE'" \
    --index-tensor-parallel-size "'$INDEX_TP'" \
    --index-gpu-memory-utilization "'$INDEX_GPU_MEMORY_UTILIZATION'"
' >> "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Tail log: tail -f $LOG_FILE"
