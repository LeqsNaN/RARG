#!/usr/bin/env bash
# Generic RARG environment activation script for open-source use.
# Usage: source /path/to/RARG/activate.sh

_RARG_ACTIVATE_SOURCE="${BASH_SOURCE[0]:-$0}"
RARG_ROOT="$(cd "$(dirname "$_RARG_ACTIVATE_SOURCE")" && pwd)"
export RARG_ROOT

# ---------------------------------------------------------------------------
# Optional DCI-style local-tools activation block
#
# We keep this block as comments on purpose.
#
# Background:
# - In the original DCI-Agent-Lite environment, some machines did not provide
#   a usable system-wide Node.js 20 or ripgrep.
# - To work around that, those tools were installed under local-tools/ and then
#   prepended to PATH from activate.sh.
# - RARG does not ship those binaries, but we keep the exact pattern here so
#   readers with the same machine constraints can reproduce the setup.
#
# Versions used in that environment:
# - Node.js v20.18.1
# - ripgrep 14.1.1
#
# If you want to follow the same approach, place the tools under:
#   local-tools/node-v20.18.1-linux-x64/
#   local-tools/ripgrep-14.1.1-x86_64-unknown-linux-musl/
#   local-tools/bin/                      # optional helper wrappers / symlinks
#
# Then uncomment / adapt these lines:
#
# export PATH="$RARG_ROOT/local-tools/node-v20.18.1-linux-x64/bin:$PATH"
# export PATH="$RARG_ROOT/local-tools/ripgrep-14.1.1-x86_64-unknown-linux-musl:$PATH"
# export PATH="$RARG_ROOT/local-tools/bin:$PATH"
#
# For readers comparing with DCI-Agent-Lite: that repo also exported some
# internal-only variables in activate.sh. RARG keeps only the open-source
# Python / OpenAI-compatible path active by default.
# ---------------------------------------------------------------------------

if [ -d "$RARG_ROOT/.venv/bin" ]; then
    # shellcheck disable=SC1091
    source "$RARG_ROOT/.venv/bin/activate"
fi

if [ -f "$RARG_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$RARG_ROOT/.env" 2>/dev/null || true
    set +a
fi

export PYTHONPATH="$RARG_ROOT/src:$RARG_ROOT:${PYTHONPATH:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

echo "✅ RARG environment activated"
echo "   Python:     $(python3 --version 2>/dev/null || echo 'NOT FOUND')"
echo "   Node:       $(node --version 2>/dev/null || echo 'NOT FOUND')"
echo "   npm:        $(npm --version 2>/dev/null || echo 'NOT FOUND')"
echo "   uv:         $(uv --version 2>/dev/null || echo 'NOT FOUND')"
echo "   rg:         $(rg --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "   dci export: $(python3 -m dci.benchmark.export_bc_plus_docs --help >/dev/null 2>&1 && echo 'OK' || echo 'NOT FOUND')"
echo "   OPENAI_BASE_URL: ${OPENAI_BASE_URL}"
echo "   RARG_ROOT:  $RARG_ROOT"
