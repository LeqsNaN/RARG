#!/usr/bin/env bash
# Build FAISS embedding index for bc_plus_100k corpus
# Uses 4 GPUs (4,5,6,7) with vLLM + Qwen3-Embedding-4B

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
source activate.sh

mkdir -p logs

LOG_FILE="logs/build_index_$(date '+%Y%m%d_%H%M%S').log"

echo "=== Building FAISS index for bc_plus_100k ==="
echo "  Model:    Qwen3-Embedding-4B"
echo "  GPUs:     4,5,6,7 (TP=4)"
echo "  Corpus:   corpus/bc_plus_100k"
echo "  Output:   data/indices/bc_plus_100k"
echo "  Log:      $LOG_FILE"

if [ ! -d "corpus/bc_plus_100k" ]; then
    echo "ERROR: corpus/bc_plus_100k not found."
    echo "Create it first with:"
    echo "  uv run python scripts/bcplus_eval/create_100k_corpus.py \\"
    echo "    --source-browsecomp corpus/browsecomp_plus \\"
    echo "    --output corpus/bc_plus_100k \\"
    echo "    --force"
    exit 1
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup uv run python -u scripts/build_embedding_index.py \
    --corpus-dir corpus/bc_plus_100k \
    --output-dir data/indices/bc_plus_100k \
    --batch-size 64 \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Tail log: tail -f $LOG_FILE"
