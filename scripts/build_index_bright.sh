#!/usr/bin/env bash
# Build FAISS embedding indices for BRIGHT subset corpora.
# Default configuration is tuned for llama-nv-embed-reasoning-3b on BRIGHT:
#   - subsets: biology, earth_science, economics, robotics
#   - backend: Transformers
#   - GPUs: 2,3
#   - TP: 2
#   - batch size: 1
#   - max model len: 8192
# Override any setting via environment variables when needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
source activate.sh

BRIGHT_CORPUS_ROOT="${BRIGHT_CORPUS_ROOT:-$REPO_ROOT/bright_corpus}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-$REPO_ROOT/models/llama-nv-embed-reasoning-3b}"
EMBED_MODEL_TYPE="${EMBED_MODEL_TYPE:-llama_nv_embed_reasoning_3b}"
EMBED_BACKEND="${EMBED_BACKEND:-transformers}"
INDEX_NAME_SUFFIX="${INDEX_NAME_SUFFIX:-llama_nv_reasoning_3b_nemo_aligned}"
BRIGHT_INDEX_ROOT="${BRIGHT_INDEX_ROOT:-$REPO_ROOT/data/indices/bright_${INDEX_NAME_SUFFIX}}"
SUBSETS="${SUBSETS:-biology,earth_science,economics,robotics}"
CUDA_DEVICES="${CUDA_DEVICES:-2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
EMBED_DEVICE="${EMBED_DEVICE:-cuda:0}"
SHARD_PREFIX="${SHARD_PREFIX:-shard}"
VALIDATE_INDEX="${VALIDATE_INDEX:-1}"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$BRIGHT_INDEX_ROOT"

LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/build_index_bright_$(date '+%Y%m%d_%H%M%S').log}"

echo "=== Building FAISS indices for BRIGHT subsets ==="
echo "  Model path: $EMBED_MODEL_PATH"
echo "  Model type: $EMBED_MODEL_TYPE"
echo "  Backend:    $EMBED_BACKEND"
echo "  GPUs:       $CUDA_DEVICES"
echo "  TP size:    $TENSOR_PARALLEL_SIZE"
echo "  Device:     $EMBED_DEVICE"
echo "  Corpus root:$BRIGHT_CORPUS_ROOT"
echo "  Index root: $BRIGHT_INDEX_ROOT"
echo "  Subsets:    $SUBSETS"
echo "  Batch size: $BATCH_SIZE"
echo "  Max len:    $MAX_MODEL_LEN"
echo "  Validate:   $VALIDATE_INDEX"
echo "  Log:        $LOG_FILE"

for subset in ${SUBSETS//,/ }; do
    subset="${subset// /}"
    if [ -z "$subset" ]; then
        continue
    fi
    if [ ! -d "$BRIGHT_CORPUS_ROOT/$subset" ]; then
        echo "[ERROR] Missing BRIGHT subset corpus: $BRIGHT_CORPUS_ROOT/$subset" >&2
        echo "Run: bash scripts/prepare_bright_corpus.sh" >&2
        exit 1
    fi
done

RUN_SCRIPT="$(mktemp /tmp/build_index_bright_XXXXXX.sh)"
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$REPO_ROOT"
source activate.sh

BRIGHT_CORPUS_ROOT="$BRIGHT_CORPUS_ROOT"
BRIGHT_INDEX_ROOT="$BRIGHT_INDEX_ROOT"
EMBED_MODEL_PATH="$EMBED_MODEL_PATH"
EMBED_MODEL_TYPE="$EMBED_MODEL_TYPE"
EMBED_BACKEND="$EMBED_BACKEND"
SUBSETS="$SUBSETS"
CUDA_DEVICES="$CUDA_DEVICES"
TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE"
BATCH_SIZE="$BATCH_SIZE"
MAX_MODEL_LEN="$MAX_MODEL_LEN"
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
EMBED_DEVICE="$EMBED_DEVICE"
SHARD_PREFIX="$SHARD_PREFIX"
VALIDATE_INDEX="$VALIDATE_INDEX"

validate_index_dir() {
    local index_dir="\$1"
    local label="\$2"
    local expected_model="\$3"
    local expected_model_type="\$4"
    local expected_backend="\$5"

    uv run python - "\$index_dir" "\$label" "\$expected_model" "\$expected_model_type" "\$expected_backend" <<'PY'
import json
import sys
from pathlib import Path

import faiss

index_dir = Path(sys.argv[1])
label = sys.argv[2]
expected_model = sys.argv[3]
expected_model_type = sys.argv[4]
expected_backend = sys.argv[5]

index_path = index_dir / "index.faiss"
paths_path = index_dir / "paths.json"
meta_path = index_dir / "meta.json"

missing = [str(p) for p in (index_path, paths_path, meta_path) if not p.exists()]
if missing:
    raise SystemExit(f"[validate:{label}] Missing files: {missing}")

index = faiss.read_index(str(index_path))
with paths_path.open("r", encoding="utf-8") as f:
    paths = json.load(f)
with meta_path.open("r", encoding="utf-8") as f:
    meta = json.load(f)

if len(paths) != index.ntotal:
    raise SystemExit(
        f"[validate:{label}] paths.json length ({len(paths)}) != index.ntotal ({index.ntotal})"
    )

duplicates = len(paths) - len(set(paths))
if duplicates:
    raise SystemExit(f"[validate:{label}] Found {duplicates} duplicate document paths")

if expected_model and meta.get("model") != expected_model:
    raise SystemExit(
        f"[validate:{label}] model mismatch: {meta.get('model')} != {expected_model}"
    )

if expected_model_type and meta.get("embed_model_type") != expected_model_type:
    raise SystemExit(
        f"[validate:{label}] embed_model_type mismatch: {meta.get('embed_model_type')} != {expected_model_type}"
    )

if expected_backend and meta.get("backend") != expected_backend:
    raise SystemExit(
        f"[validate:{label}] backend mismatch: {meta.get('backend')} != {expected_backend}"
    )

print(
    f"[validate:{label}] ok: ntotal={index.ntotal}, dim={index.d}, "
    f"backend={meta.get('backend')}, model_type={meta.get('embed_model_type')}"
)
PY
}

for subset in \${SUBSETS//,/ }; do
    subset="\${subset// /}"
    if [ -z "\$subset" ]; then
        continue
    fi

    corpus_dir="\$BRIGHT_CORPUS_ROOT/\$subset"
    output_dir="\$BRIGHT_INDEX_ROOT/\$subset"

    echo
    echo ">>> [\$(date '+%Y-%m-%d %H:%M:%S')] Building BRIGHT index for subset: \$subset"
    echo "    Corpus: \$corpus_dir"
    echo "    Output: \$output_dir"

    mkdir -p "\$output_dir"
    IFS=',' read -r -a gpu_array <<< "\$CUDA_DEVICES"
    num_gpus="\${#gpu_array[@]}"

    if [ "\$EMBED_BACKEND" = "transformers" ] && [ "\$num_gpus" -gt 1 ]; then
        echo "    Using sharded Transformers indexing across \$num_gpus GPUs"
        shard_dirs=()
        pids=()
        for shard_idx in "\${!gpu_array[@]}"; do
            gpu_id="\${gpu_array[\$shard_idx]}"
            shard_dir="\$output_dir/\$SHARD_PREFIX-\$shard_idx"
            shard_dirs+=("\$shard_dir")
            mkdir -p "\$shard_dir"

            echo "    Launch shard \$shard_idx on GPU \$gpu_id -> \$shard_dir"
            CUDA_VISIBLE_DEVICES="\$gpu_id" EMBED_DEVICE="cuda:0" \
                uv run python -u scripts/build_embedding_index.py \
                    --corpus-dir "\$corpus_dir" \
                    --output-dir "\$shard_dir" \
                    --model "\$EMBED_MODEL_PATH" \
                    --model-type "\$EMBED_MODEL_TYPE" \
                    --backend "\$EMBED_BACKEND" \
                    --batch-size "\$BATCH_SIZE" \
                    --tensor-parallel-size 1 \
                    --max-model-len "\$MAX_MODEL_LEN" \
                    --gpu-memory-utilization "\$GPU_MEMORY_UTILIZATION" \
                    --device "cuda:0" \
                    --shard-index "\$shard_idx" \
                    --num-shards "\$num_gpus" \
                    > "\$shard_dir/build.log" 2>&1 &
            pids+=("\$!")
        done

        for pid in "\${pids[@]}"; do
            wait "\$pid"
        done

        if [ "\$VALIDATE_INDEX" = "1" ]; then
            for shard_idx in "\${!shard_dirs[@]}"; do
                validate_index_dir "\${shard_dirs[\$shard_idx]}" "subset=\$subset shard=\$shard_idx" "\$EMBED_MODEL_PATH" "\$EMBED_MODEL_TYPE" "transformers"
            done
        fi

        rm -f "\$output_dir/index.faiss" "\$output_dir/paths.json" "\$output_dir/meta.json"
        first_shard="\${shard_dirs[0]}"
        cp "\$first_shard/index.faiss" "\$output_dir/index.faiss"
        cp "\$first_shard/paths.json" "\$output_dir/paths.json"
        cp "\$first_shard/meta.json" "\$output_dir/meta.json"

        for ((merge_i=1; merge_i<\${#shard_dirs[@]}; merge_i++)); do
            shard_dir="\${shard_dirs[\$merge_i]}"
            tmp_merged_dir="\$output_dir/merged_tmp_\$merge_i"
            rm -rf "\$tmp_merged_dir"
            uv run python -u scripts/merge_embedding_indices.py \
                --index-a-dir "\$output_dir" \
                --index-b-dir "\$shard_dir" \
                --output-dir "\$tmp_merged_dir" \
                --corpus-dir "\$corpus_dir"
            mv "\$tmp_merged_dir/index.faiss" "\$output_dir/index.faiss"
            mv "\$tmp_merged_dir/paths.json" "\$output_dir/paths.json"
            mv "\$tmp_merged_dir/meta.json" "\$output_dir/meta.json"
            rmdir "\$tmp_merged_dir" 2>/dev/null || true
        done

        if [ "\$VALIDATE_INDEX" = "1" ]; then
            uv run python - "\$output_dir" "\${shard_dirs[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
shard_dirs = [Path(p) for p in sys.argv[2:]]

with (output_dir / "paths.json").open("r", encoding="utf-8") as f:
    final_paths = json.load(f)

final_set = set(final_paths)
shard_sets = []
for shard_dir in shard_dirs:
    with (shard_dir / "paths.json").open("r", encoding="utf-8") as f:
        shard_paths = json.load(f)
    shard_set = set(shard_paths)
    shard_sets.append((shard_dir, shard_set, len(shard_paths)))

expected_total = sum(size for _dir, _set, size in shard_sets)
if len(final_paths) != expected_total:
    raise SystemExit(
        f"[validate:final] final paths count {len(final_paths)} != shard sum {expected_total}"
    )

union = set().union(*(s for _d, s, _n in shard_sets)) if shard_sets else set()
if final_set != union:
    missing = sorted(union - final_set)[:10]
    extra = sorted(final_set - union)[:10]
    raise SystemExit(
        f"[validate:final] final path set mismatch; missing={missing}, extra={extra}"
    )

for i, (dir_a, set_a, _) in enumerate(shard_sets):
    for dir_b, set_b, _ in shard_sets[i + 1 :]:
        dup = set_a & set_b
        if dup:
            sample = sorted(dup)[:10]
            raise SystemExit(
                f"[validate:final] duplicate paths across shards {dir_a.name} and {dir_b.name}: {sample}"
            )

print(
    f"[validate:final] ok: final_docs={len(final_paths)}, "
    f"shards={len(shard_sets)}, shard_sum={expected_total}"
)
PY
            validate_index_dir "\$output_dir" "subset=\$subset final" "\$EMBED_MODEL_PATH" "\$EMBED_MODEL_TYPE" "merged"
        fi
    else
        CUDA_VISIBLE_DEVICES="\$CUDA_DEVICES" EMBED_DEVICE="\$EMBED_DEVICE" uv run python -u scripts/build_embedding_index.py \
            --corpus-dir "\$corpus_dir" \
            --output-dir "\$output_dir" \
            --model "\$EMBED_MODEL_PATH" \
            --model-type "\$EMBED_MODEL_TYPE" \
            --backend "\$EMBED_BACKEND" \
            --batch-size "\$BATCH_SIZE" \
            --tensor-parallel-size "\$TENSOR_PARALLEL_SIZE" \
            --max-model-len "\$MAX_MODEL_LEN" \
            --gpu-memory-utilization "\$GPU_MEMORY_UTILIZATION" \
            --device "\$EMBED_DEVICE"

        if [ "\$VALIDATE_INDEX" = "1" ]; then
            validate_index_dir "\$output_dir" "subset=\$subset final" "\$EMBED_MODEL_PATH" "\$EMBED_MODEL_TYPE" "\$EMBED_BACKEND"
        fi
    fi

    echo "<<< [\$(date '+%Y-%m-%d %H:%M:%S')] Finished subset: \$subset"
done

echo
echo "All requested BRIGHT subset indices finished."
EOF
chmod +x "$RUN_SCRIPT"

nohup stdbuf -oL -eL bash "$RUN_SCRIPT" > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Tail log: tail -f $LOG_FILE"
echo "Worker script: $RUN_SCRIPT"
