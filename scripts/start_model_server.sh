#!/usr/bin/env bash
# ==========================================================================
# Start model_server.py in persistent TCP mode.
#
# Usage:
#   bash scripts/start_model_server.sh [--port PORT]
#
# The server stays alive until killed. Multiple agent processes can connect.
# Set CUDA_VISIBLE_DEVICES before calling this script to control GPU mapping.
#
# Required env vars (or will use defaults):
#   EMBED_INDEX_DIR   - FAISS index directory
#   EMBED_DEVICE      - Embedding model device (default: cuda:0)
#   QR_DEVICE         - QRReranker device (default: cuda:1)
#   QR_MODEL_PATH     - QRReranker model path
#   QR_HEADS          - QR head list
#   CORPUS_DIR        - Corpus directory
#
# Exports MODEL_SERVER_PORT for child processes.
# ==========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse --port argument
PORT="${MODEL_SERVER_PORT:-9010}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PYTHON="${EMBED_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SERVER_SCRIPT="${MODEL_SERVER_SCRIPT:-$REPO_ROOT/scripts/model_server.py}"
LOG_FILE="$REPO_ROOT/logs/model_server_${PORT}.log"
PID_FILE="$REPO_ROOT/logs/model_server_${PORT}.pid"

mkdir -p "$REPO_ROOT/logs"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Model server already running (PID $OLD_PID) on port $PORT"
        export MODEL_SERVER_PORT="$PORT"
        return 0 2>/dev/null || exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# Build command
CMD=("$PYTHON" -u "$SERVER_SCRIPT" --port "$PORT")

if [ -n "${EMBED_INDEX_DIR:-}" ]; then
    CMD+=(--index-dir "$EMBED_INDEX_DIR")
fi
if [ -n "${EMBED_MODEL_PATH:-}" ]; then
    CMD+=(--embed-model-path "$EMBED_MODEL_PATH")
fi
if [ -n "${EMBED_MODEL_TYPE:-}" ]; then
    CMD+=(--embed-model-type "$EMBED_MODEL_TYPE")
fi
if [ -n "${EMBED_BACKEND:-}" ]; then
    CMD+=(--embed-backend "$EMBED_BACKEND")
fi
if [ -n "${EMBED_MAX_TOKENS:-}" ]; then
    CMD+=(--embed-max-tokens "$EMBED_MAX_TOKENS")
fi
if [ -n "${EMBED_RERANK_MAX_TOKENS:-}" ]; then
    CMD+=(--embed-rerank-max-tokens "$EMBED_RERANK_MAX_TOKENS")
fi
if [ -n "${RG_EMBED_MODEL_PATH:-}" ]; then
    CMD+=(--rg-embed-model-path "$RG_EMBED_MODEL_PATH")
fi
if [ -n "${RG_EMBED_MODEL_TYPE:-}" ]; then
    CMD+=(--rg-embed-model-type "$RG_EMBED_MODEL_TYPE")
fi
if [ -n "${RG_EMBED_BACKEND:-}" ]; then
    CMD+=(--rg-embed-backend "$RG_EMBED_BACKEND")
fi
if [ -n "${RG_EMBED_DEVICE:-}" ]; then
    CMD+=(--rg-embed-device "$RG_EMBED_DEVICE")
fi
if [ -n "${RG_EMBED_MAX_TOKENS:-}" ]; then
    CMD+=(--rg-embed-max-tokens "$RG_EMBED_MAX_TOKENS")
fi
if [ -n "${RG_EMBED_RERANK_MAX_TOKENS:-}" ]; then
    CMD+=(--rg-embed-rerank-max-tokens "$RG_EMBED_RERANK_MAX_TOKENS")
fi
if [ -n "${EMBED_DEVICE:-}" ]; then
    CMD+=(--device "$EMBED_DEVICE")
fi
if [ -n "${QR_DEVICE:-}" ]; then
    CMD+=(--qr-device "$QR_DEVICE")
fi
if [ -n "${QR_MODEL_PATH:-}" ]; then
    CMD+=(--qr-model-path "$QR_MODEL_PATH")
fi
if [ -n "${QR_HEADS:-}" ]; then
    CMD+=(--qr-heads "$QR_HEADS")
fi
if [ -n "${CORPUS_DIR:-}" ]; then
    CMD+=(--corpus-dir "$CORPUS_DIR")
fi
if [ "${NO_RERANKER:-0}" = "1" ] || [ "${NO_RERANKER:-}" = "true" ] || [ "${NO_RERANKER:-}" = "True" ]; then
    CMD+=(--no-reranker)
fi

echo "Starting model server on port $PORT..."
echo "  Command: ${CMD[*]}"
echo "  Log: $LOG_FILE"

# Start in background
"${CMD[@]}" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Wait for "Server ready" or "TCP server listening" in log
echo -n "  Waiting for server to be ready"
TIMEOUT=300
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: Server process died. Check $LOG_FILE"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    if grep -q "TCP server listening" "$LOG_FILE" 2>/dev/null; then
        echo " OK (${ELAPSED}s)"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo -n "."
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo ""
    echo "ERROR: Server startup timed out after ${TIMEOUT}s. Check $LOG_FILE"
    kill "$SERVER_PID" 2>/dev/null
    rm -f "$PID_FILE"
    exit 1
fi

echo "  Model server ready (PID $SERVER_PID, port $PORT)"
export MODEL_SERVER_PORT="$PORT"
