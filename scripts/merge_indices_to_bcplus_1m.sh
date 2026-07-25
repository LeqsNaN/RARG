#!/usr/bin/env bash
# Deprecated helper kept for compatibility.
# The current workflow does not merge indices anymore; it rebuilds the final
# bc_plus_1m corpus and index directly.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[WARN] scripts/merge_indices_to_bcplus_1m.sh is deprecated."
echo "[WARN] The new workflow no longer merges bc_plus_100k and bc_plus_900k."
echo "[WARN] Please use scripts/build_index_bcplus_1m.sh instead."

exec bash "$SCRIPT_DIR/build_index_bcplus_1m.sh" "$@"
