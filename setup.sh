#!/usr/bin/env bash
set -euo pipefail

echo "==> Setting up RARG environment..."

if [ -f ".env" ]; then
    echo "==> Loading .env..."
    set -a
    # shellcheck disable=SC1091
    source ".env" 2>/dev/null || true
    set +a
fi

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Please install uv first."
    exit 1
fi

if ! command -v rg &> /dev/null; then
    echo "WARN: ripgrep (rg) not found. Please install it before running agents."
fi

echo "==> Syncing Python dependencies..."
uv sync

_has_files_at_depth() {
    local dir="$1"
    local min_depth="$2"
    [ -d "$dir" ] && find "$dir" -mindepth "$min_depth" -type f -print -quit | grep -q .
}

if ! _has_files_at_depth "corpus/browsecomp_plus" 1 \
    || ! _has_files_at_depth "corpus/bc_plus_docs" 2 \
    || [ ! -f "corpus/bright_corpus/biology/.dci_export_complete" ] \
    || [ ! -f "corpus/bright_corpus/earth_science/.dci_export_complete" ] \
    || [ ! -f "corpus/bright_corpus/economics/.dci_export_complete" ] \
    || [ ! -f "corpus/bright_corpus/robotics/.dci_export_complete" ]; then
    echo ""
    echo "==> Downloading/exporting corpus from HuggingFace (DCI-Agent/corpus)..."
    echo "    Note: This dataset is gated. Run 'huggingface-cli login' first if needed."
    uv run python scripts/download_corpus.py || {
        echo ""
        echo "WARN: Corpus download failed."
        echo "      1. Run 'huggingface-cli login' to authenticate"
        echo "      2. Then re-run: uv run python scripts/download_corpus.py"
    }
else
    echo ""
    echo "==> Corpus already present and exported in corpus/, skipping download."
fi

if [ ! -d "data/dci-bench" ]; then
    echo ""
    echo "==> Downloading benchmark datasets from HuggingFace (DCI-Agent/dci-bench)..."
    uv run python scripts/download_dci_bench.py || {
        echo ""
        echo "WARN: Benchmark dataset download failed."
        echo "      1. Run 'huggingface-cli login' to authenticate"
        echo "      2. Then re-run: uv run python scripts/download_dci_bench.py"
    }
else
    echo ""
    echo "==> Benchmark datasets already present in data/dci-bench/, skipping download."
fi

if [ ! -f "data/bcplus_qa.jsonl" ]; then
    echo ""
    echo "==> Extracting BrowseComp-Plus QA pairs to data/bcplus_qa.jsonl..."
    uv run python scripts/bcplus_eval/extract_bcplus_qa.py || {
        echo ""
        echo "WARN: QA extraction failed."
        echo "      Make sure data/dci-bench/data/browsecomp-plus/ exists with parquet files."
    }
else
    echo ""
    echo "==> data/bcplus_qa.jsonl already present, skipping extraction."
fi

if ! _has_files_at_depth "corpus/bc_plus_100k" 2; then
    if _has_files_at_depth "corpus/browsecomp_plus" 1 || _has_files_at_depth "corpus/bc_plus_docs" 2; then
        echo ""
        echo "==> Preparing corpus/bc_plus_100k with DCI-compatible 100K construction..."
        if [ -d "corpus/browsecomp_plus" ]; then
            uv run python scripts/bcplus_eval/create_100k_corpus.py \
                --source-browsecomp corpus/browsecomp_plus \
                --output corpus/bc_plus_100k \
                --force \
                --clean-output || {
                echo ""
                echo "WARN: corpus/bc_plus_100k creation failed."
            }
        else
            uv run python scripts/bcplus_eval/create_100k_corpus.py \
                --source-docs corpus/bc_plus_docs \
                --output corpus/bc_plus_100k \
                --force \
                --clean-output || {
                echo ""
                echo "WARN: corpus/bc_plus_100k creation failed."
            }
        fi
    fi
fi

if ! _has_files_at_depth "corpus/bc_plus_100k" 2; then
    echo ""
    echo "WARN: corpus/bc_plus_100k is missing."
    echo "      The TS-Mirror sample100 scripts in scripts/sra_bench/ expect this directory."
    echo "      Prepare it with:"
    echo "      uv run python scripts/bcplus_eval/create_100k_corpus.py --source-browsecomp corpus/browsecomp_plus --output corpus/bc_plus_100k --force"
fi

if [ ! -f "data/indices/bc_plus_100k/index.faiss" ]; then
    echo ""
    echo "WARN: data/indices/bc_plus_100k/index.faiss is missing."
    echo "      100K TS-Mirror / embedding runs will not start until this index is available."
fi

if [ ! -f "data/indices/bc_plus_1m/index.faiss" ]; then
    echo ""
    echo "WARN: data/indices/bc_plus_1m/index.faiss is missing."
    echo "      1M TS-Mirror runs will not start until this index is available."
fi

if [ ! -d "models/Qwen3-Embedding-4B" ]; then
    echo ""
    echo "WARN: models/Qwen3-Embedding-4B is missing."
    echo "      The copied TS-Mirror scripts expect this embedding model by default."
fi

if [ ! -d "models/Qwen3-4B-Instruct-2507" ]; then
    echo ""
    echo "WARN: models/Qwen3-4B-Instruct-2507 is missing."
    echo "      QR / paragraph reranking paths may require this model by default."
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo ""
    echo "WARN: No OPENAI_API_KEY detected in environment."
    echo "      Set it before agent runs / LLM-as-judge."
fi

echo ""
echo "==> Setup complete!"
echo "    Next steps:"
echo "    1. source activate.sh"
echo "    2. Run data/corpus prep again if needed: bash setup.sh"
echo "    3. Launch an agent script under scripts/sra_bench/ or scripts/bright/"
