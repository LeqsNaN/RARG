# Python DR-DCI port

This is RARG's standalone, sanitized copy of the Python implementation of
DR-DCI's main BrowseComp-Plus path. It is independent of the original
workspace copy under `DCI-Agent-Lite/`.

The port preserves the DR-DCI agent loop, prompts, `read`/`bash`/`pull` tools,
root-flat dynamic document view, and Level 3 request-time context handling.
The local retrieval integration uses RARG's Qwen3 embedding model server.

## API configuration

LLM requests use the standard OpenAI Chat Completions API. Set:

```bash
export OPENAI_API_KEY=...
# Optional for another OpenAI-compatible provider:
# export OPENAI_BASE_URL=https://your-endpoint.example/v1
```

No proxy, account-specific environment variables, or provider-specific startup
logic is included in this copy.

## Run BrowseComp-Plus

Place the embedding model under `models/Qwen3-Embedding-4B`, prepare
`corpus/bc_plus_100k` and `data/indices/bc_plus_100k`, then run:

```bash
cd RARG
source activate.sh
OPENAI_API_KEY=... BCP_LIMIT=100 MAX_TURNS=300 \
  bash scripts/dr-dci/run_bcplus.sh
```

The launcher uses one sample at a time by default (`MAX_CONCURRENCY=1`) and
starts or reuses RARG's local retrieval `model_server` on port `9010`. Override
all corpus, index, model, server, output, and view locations through the
similarly named environment variables in `run_bcplus.sh`.

To run a single benchmark id:

```bash
BCP_QUERY_ID=96 bash scripts/dr-dci/run_bcplus.sh
```

## Judge results

```bash
cd RARG
source activate.sh
OPENAI_API_KEY=... DR_DCI_RUN_NAME=dr_dci_bcplus100k \
  bash scripts/dr-dci/judge_bcplus.sh
```

The judge runs in the foreground and reuses matching per-query results by
default. Set `JUDGE_FORCE=1` only when intentionally rejudging every sample.

## Level 3 behavior

The default `request_time` mode matches the DR-DCI/Pi behavior: when tool text
in the canonical transcript exceeds 240k characters, the request temporarily
replaces older tool results with `[cleared]`, retaining the latest 12 assistant
turns. The canonical transcript itself is unchanged.

`LEVEL3_PERSISTENT_MICRO_COMPACT=1` enables an experimental, non-original
mode that persists clears and re-accumulates tool text before the next compact.
