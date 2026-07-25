# Local tools placeholder

This directory is intentionally a **placeholder** in the open-source RARG tree.

We do **not** vendor large prebuilt binaries here, but we keep this directory so
readers can understand how our internal environment was aligned with
`DCI-Agent-Lite`.

## Why this exists

On some machines, the system environment does not provide:

- a sufficiently new **Node.js 20**
- a working **ripgrep (`rg`)**

In our internal setup, we solved that by installing these tools under
`local-tools/` and prepending them to `PATH` from `activate.sh`.

The corresponding activation block is present in `activate.sh` as a **commented
example** so others with the same machine constraints can follow the same idea.

## Versions we used

- **Node.js**: `v20.18.1`
- **ripgrep**: `14.1.1`

The DCI-style internal directory names were:

- `local-tools/node-v20.18.1-linux-x64/`
- `local-tools/ripgrep-14.1.1-x86_64-unknown-linux-musl/`
- optionally `local-tools/bin/` for helper wrappers / symlinks

## If your machine already has these tools

If `node`, `npm`, and `rg` are already available on your machine, you do not
need to put anything here.

## If your machine is missing them

You can either:

1. install them system-wide, or
2. place prebuilt copies under `local-tools/` using the structure above, then
   uncomment / adapt the optional `PATH` exports in `activate.sh`.

## Notes

- RARG's current TS-Mirror workflow mainly needs:
  - `python`
  - `uv`
  - `rg`
- The heavier `Node.js` / Pi-related setup comes from the original
  `DCI-Agent-Lite` environment and is documented here for compatibility /
  reproducibility context.
