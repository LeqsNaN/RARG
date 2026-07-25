#!/usr/bin/env bash
# Deprecated helper kept only for compatibility with older notes.
# The recommended workflow is now to rebuild a standalone bc_plus_1m corpus
# directly, instead of building a distractor-only 900k index and merging.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[WARN] scripts/build_index_bcplus_900k.sh is deprecated."
echo "[WARN] Please use scripts/bcplus_eval/prepare_bcplus_1m_cluster.sh instead."
echo "[WARN] The new workflow:"
echo "       1) samples 900k FineWeb docs with text length >= 8000 chars"
echo "       2) reconstructs BC+ docs from corpus/browsecomp_plus into corpus/bc_plus_1m"
echo "       3) builds data/indices/bc_plus_1m directly without merge"

exec bash "$SCRIPT_DIR/bcplus_eval/prepare_bcplus_1m_cluster.sh" "$@"
