#!/usr/bin/env bash
# Rebuild a standalone bc_plus_1m corpus and embedding index directly.
#
# New workflow:
#   - sample 900k FineWeb documents with text length >= 8000 chars
#   - reconstruct original BC+ docs from corpus/browsecomp_plus
#   - export the 900k FineWeb docs into the same corpus/bc_plus_1m
#   - build data/indices/bc_plus_1m directly with the same embedding-index
#     pipeline/settings used for bc_plus_100k

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/bcplus_eval/prepare_bcplus_1m_cluster.sh" "$@"
