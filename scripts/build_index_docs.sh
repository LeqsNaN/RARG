#!/usr/bin/env bash
# Build FAISS embedding index for bc_plus_docs corpus (200K docs)
# Uses 4 GPUs (4,5,6,7) with vLLM + Qwen3-Embedding-4B

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
source activate.sh

mkdir -p logs

LOG_FILE="logs/build_index_docs_$(date '+%Y%m%d_%H%M%S').log"

echo "=== Building FAISS index for bc_plus_docs (200K) ==="
echo "  Model:    Qwen3-Embedding-4B"
echo "  GPUs:     4,5,6,7 (TP=4)"
echo "  Corpus:   corpus/bc_plus_docs"
echo "  Output:   data/indices/bc_plus_docs"
echo "  Log:      $LOG_FILE"

if [ ! -d "corpus/bc_plus_docs" ]; then
    echo "ERROR: corpus/bc_plus_docs not found."
    echo "Prepare it first with one of:"
    echo "  uv run python scripts/download_corpus.py"
    echo "or"
    echo "  uv run dci-export-bc-plus-docs --source-dir \"$PWD/corpus/browsecomp_plus\" --output-dir \"$PWD/corpus/bc_plus_docs\""
    exit 1
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup uv run python -u scripts/build_embedding_index.py \
    --corpus-dir corpus/bc_plus_docs \
    --output-dir data/indices/bc_plus_docs \
    --batch-size 64 \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Tail log: tail -f $LOG_FILE"
