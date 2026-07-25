#!/usr/bin/env bash
set -euo pipefail

PID_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid-file)
            PID_FILE="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$PID_FILE" ]; then
    echo "Missing required --pid-file" >&2
    exit 2
fi

if [ $# -eq 0 ]; then
    echo "Missing command after --" >&2
    exit 2
fi

run_status=0
if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "$@" || run_status=$?
else
    "$@" || run_status=$?
fi

if [ -f "$PID_FILE" ]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
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
    rm -f "$PID_FILE"
fi

exit "$run_status"
