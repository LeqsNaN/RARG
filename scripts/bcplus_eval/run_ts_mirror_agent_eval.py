#!/usr/bin/env python3
"""
Evaluation runner for TS-Mirror Python Agent on BrowseComp-Plus.

Runs the in-process ts_mirror_agent over a JSONL dataset, persists per-query
artifacts under output_root/<query_id>/, and reuses a single model server.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'ts_mirror_agent'))

from agent import AgentSession, save_outputs
from config import AgentConfig
from llm_client import LLMClient
from model_client import ModelServerClient
from tools import build_tools

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run TS-Mirror Agent eval on BrowseComp-Plus')

    parser.add_argument('--dataset', type=Path, required=True, help='JSONL dataset')
    parser.add_argument('--output-root', type=Path, required=True, help='Output root directory')
    parser.add_argument('--corpus-dir', type=Path, required=True, help='Corpus directory')
    parser.add_argument('--max-concurrency', type=int, default=1, help='Parallel queries (kept at 1)')
    parser.add_argument('--limit', type=int, help='Only run first N questions')
    parser.add_argument('--skip-completed', action='store_true', default=True, help='Skip queries with existing final.txt')

    parser.add_argument('--provider', default='openai')
    parser.add_argument('--model', default='gpt-5.4-mini')
    parser.add_argument('--base-url', default='https://api.openai.com/v1')
    parser.add_argument('--api-key', default='')
    parser.add_argument('--thinking-level', default='high')
    parser.add_argument('--max-turns', type=int, default=150)
    parser.add_argument('--force-answer-turns', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max-output-tokens', type=int, default=16384)
    parser.add_argument('--keep-tool-calls', type=int, default=30)
    parser.add_argument('--compaction-threshold', type=int, default=150000)
    parser.add_argument('--use-usage-based-compaction', action='store_true')
    parser.add_argument('--usage-based-compaction-min-tokens', type=int, default=110000)
    parser.add_argument('--usage-based-compaction-max-tokens', type=int, default=130000)

    parser.add_argument('--index-dir', type=str, required=True)
    parser.add_argument('--embed-model-path', type=str, default='')
    parser.add_argument('--qr-model-path', type=str, default='')
    parser.add_argument('--qr-heads', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--embed-top-k', type=int, default=5000)
    parser.add_argument('--no-reranker', action='store_true')

    parser.add_argument('--system-prompt-file', default='')
    parser.add_argument('--append-system-prompt-file', default='')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_benchmark_prompt(query: str) -> str:
    return (
        "Answer the following question. The answer is contained in the corpus directory "
        "(your current working directory). **Do Not use web search!** Use ripgrep (rg) "
        "instead of grep for fast searching.\n\nQUESTION:\n" + query
    )


def build_agent_config(args: argparse.Namespace, query_dir: Path, query_id: str) -> AgentConfig:
    config = AgentConfig()
    config.provider = args.provider
    config.model = args.model
    config.base_url = args.base_url
    config.api_key = args.api_key
    config.temperature = args.temperature if args.temperature is not None else 0.0
    config.max_output_tokens = args.max_output_tokens
    config.thinking_level = args.thinking_level

    config.max_turns = args.max_turns
    config.force_answer_turns = args.force_answer_turns
    config.keep_tool_calls = args.keep_tool_calls
    config.compaction_threshold = args.compaction_threshold
    config.use_usage_based_compaction = args.use_usage_based_compaction
    config.usage_based_compaction_min_tokens = args.usage_based_compaction_min_tokens
    config.usage_based_compaction_max_tokens = args.usage_based_compaction_max_tokens

    config.corpus_dir = str(args.corpus_dir)
    config.scope_dir = str(query_dir)
    config.output_dir = str(query_dir)

    config.model_server_script = os.environ.get('MODEL_SERVER_SCRIPT', str(REPO_ROOT / 'scripts' / 'model_server.py'))
    config.index_dir = args.index_dir
    config.embed_model_path = args.embed_model_path or os.environ.get('EMBED_MODEL_PATH', AgentConfig.embed_model_path)
    config.qr_model_path = args.qr_model_path or os.environ.get('QR_MODEL_PATH', AgentConfig.qr_model_path)
    config.qr_heads = args.qr_heads or os.environ.get('QR_HEADS', AgentConfig.qr_heads)
    config.device = args.device
    config.embed_top_k = args.embed_top_k
    config.no_reranker = args.no_reranker

    config.rg_max_matches = int(os.environ.get('RG_MAX_MATCHES', config.rg_max_matches))
    config.grep_max_line_length = int(os.environ.get('GREP_MAX_LINE_LENGTH', config.grep_max_line_length))
    config.paragraph_rerank_model = str(os.environ.get('PARAGRAPH_RERANK_MODEL', config.paragraph_rerank_model)).strip().lower()
    if config.paragraph_rerank_model not in {'qr', 'emb', 'none'}:
        config.paragraph_rerank_model = 'qr'
    config.paragraph_rerank_doc_limit = int(os.environ.get('PARAGRAPH_RERANK_DOC_LIMIT', config.paragraph_rerank_doc_limit))
    config.rg_rerank_match = str(os.environ.get('RG_RERANK_MATCH', '')).strip().lower() in {'1', 'true', 'yes', 'on'}
    config.rg_rerank_model = str(os.environ.get('RG_RERANK_MODEL', config.rg_rerank_model)).strip().lower()
    if config.rg_rerank_model not in {'qr', 'emb', 'qwen3emb'}:
        config.rg_rerank_model = 'qr'
    config.use_contextual_bash_tool = str(os.environ.get('USE_CONTEXTUAL_BASH_TOOL', '')).strip().lower() in {'1', 'true', 'yes', 'on'}
    config.rg_rerank_candidate_num = int(os.environ.get('RG_RERANK_CANDIDATE_NUM', config.rg_rerank_candidate_num))
    config.rg_rerank_timeout = int(os.environ.get('RG_RERANK_TIMEOUT', config.rg_rerank_timeout))
    config.read_max_lines = int(os.environ.get('READ_MAX_LINES', config.read_max_lines))
    config.read_max_bytes = int(os.environ.get('READ_MAX_BYTES', config.read_max_bytes))

    config.scope_size_prefix = os.environ.get('SCOPE_SIZE_PREFIX', '')
    config.sample_id = query_id
    config.system_prompt_file = args.system_prompt_file
    config.append_system_prompt_file = args.append_system_prompt_file
    config.verbose = args.verbose
    return config


def run_single_query(row: Dict[str, Any], args: argparse.Namespace, model_client: ModelServerClient) -> Dict[str, Any]:
    query_id = str(row.get('query_id', row.get('id', '')))
    question = row.get('query', row.get('question', ''))
    query_dir = args.output_root / query_id

    if args.skip_completed and (query_dir / 'final.txt').exists():
        final = (query_dir / 'final.txt').read_text(encoding='utf-8').strip()
        logger.info(f'[{query_id}] Skipping (already completed)')
        return {'query_id': query_id, 'final_text': final, 'skipped': True}

    query_dir.mkdir(parents=True, exist_ok=True)
    sample_log_path = query_dir / 'log.txt'
    aggregate_log_path = query_dir.parent / 'agent_progress.log'

    def _append_log(path: Path, line: str, *, spacer: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            if spacer:
                f.write('\n')
            f.write(line)
            f.write('\n')

    def _append_sample_log(line: str, *, spacer: bool = False) -> None:
        _append_log(sample_log_path, line, spacer=spacer)
        _append_log(aggregate_log_path, line, spacer=spacer)

    with (query_dir / 'item.json').open('w', encoding='utf-8') as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

    config = build_agent_config(args, query_dir, query_id)
    config.question = build_benchmark_prompt(question)

    os.environ['SAMPLE_ID'] = query_id

    is_resume = (query_dir / 'state.json').exists() and not (query_dir / 'final.txt').exists()
    if is_resume:
        logger.info(f'[{query_id}] Resuming from state.json...')
        _append_sample_log(f'[{query_id}] Resuming from state.json...', spacer=True)
    else:
        logger.info(f'[{query_id}] Starting: {question[:80]}...')
        _append_sample_log(f'[{query_id}] Starting: {question[:80]}...', spacer=True)

    started_at = datetime.now(timezone.utc).isoformat()
    llm_client = LLMClient(config)
    tools = build_tools(config, model_client)
    if is_resume:
        session = AgentSession.resume_from_output_dir(config, llm_client, tools, model_client, query_dir)
    else:
        session = AgentSession(config, llm_client, tools, model_client)

    try:
        final_answer = session.run(config.question)
    except Exception as e:
        logger.error(f'[{query_id}] Agent failed: {e}', exc_info=True)
        _append_sample_log(f'[{query_id}] Agent failed: {e}')
        final_answer = f'[Agent error: {e}]'

    finished_at = datetime.now(timezone.utc).isoformat()
    save_outputs(config, session, final_answer)
    logger.info(f'[{query_id}] Finished in {session.turn_count} turns')
    _append_sample_log(f'[{query_id}] Finished in {session.turn_count} turns')

    return {
        'query_id': query_id,
        'question': question,
        'final_text': final_answer,
        'turn_count': session.turn_count,
        'started_at': started_at,
        'finished_at': finished_at,
        'usage': llm_client.usage.to_dict(),
        'skipped': False,
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stderr,
    )

    dataset = read_jsonl(args.dataset)
    if args.limit:
        dataset = dataset[:args.limit]
    logger.info(f'Loaded {len(dataset)} questions from {args.dataset}')

    if args.max_concurrency != 1:
        logger.warning('TS-Mirror eval runner currently executes sequentially; ignoring max_concurrency != 1')

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_dict = {
        'dataset': str(args.dataset),
        'output_root': str(args.output_root),
        'corpus_dir': str(args.corpus_dir),
        'provider': args.provider,
        'model': args.model,
        'base_url': args.base_url,
        'thinking_level': args.thinking_level,
        'max_turns': args.max_turns,
        'force_answer_turns': args.force_answer_turns,
        'max_output_tokens': args.max_output_tokens,
        'keep_tool_calls': args.keep_tool_calls,
        'compaction_threshold': args.compaction_threshold,
        'embed_top_k': args.embed_top_k,
        'no_reranker': args.no_reranker,
        'paragraph_rerank_model': os.environ.get('PARAGRAPH_RERANK_MODEL', 'qr'),
        'rg_rerank_match': os.environ.get('RG_RERANK_MATCH', ''),
        'rg_rerank_model': os.environ.get('RG_RERANK_MODEL', 'qr'),
        'total_questions': len(dataset),
    }
    with (args.output_root / 'config.json').open('w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    logger.info('Starting shared model server...')
    shared_config = AgentConfig()
    shared_config.model_server_script = os.environ.get('MODEL_SERVER_SCRIPT', str(REPO_ROOT / 'scripts' / 'model_server.py'))
    shared_config.index_dir = args.index_dir
    shared_config.embed_model_path = args.embed_model_path or os.environ.get('EMBED_MODEL_PATH', AgentConfig.embed_model_path)
    shared_config.qr_model_path = args.qr_model_path or os.environ.get('QR_MODEL_PATH', AgentConfig.qr_model_path)
    shared_config.qr_heads = args.qr_heads or os.environ.get('QR_HEADS', AgentConfig.qr_heads)
    shared_config.device = args.device
    shared_config.corpus_dir = str(args.corpus_dir)
    shared_config.no_reranker = args.no_reranker

    model_client = ModelServerClient(shared_config)
    model_client.ensure_started()
    logger.info('Model server ready')

    results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()
    try:
        for i, row in enumerate(dataset):
            query_id = str(row.get('query_id', row.get('id', i)))
            logger.info(f'=== Query {i + 1}/{len(dataset)} [{query_id}] ===')
            try:
                result = run_single_query(row, args, model_client)
                results.append(result)
            except Exception as e:
                logger.error(f'[{query_id}] Fatal error: {e}', exc_info=True)
                results.append({'query_id': query_id, 'error': str(e), 'skipped': False})

            with (args.output_root / 'results.jsonl').open('w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
    finally:
        model_client.shutdown()

    total_elapsed = time.perf_counter() - total_start
    completed = [r for r in results if not r.get('skipped') and not r.get('error')]
    skipped = sum(1 for r in results if r.get('skipped'))
    errors = sum(1 for r in results if r.get('error'))
    avg_turns = sum(r.get('turn_count', 0) for r in completed) / len(completed) if completed else 0.0

    logger.info('=== Evaluation Complete ===')
    logger.info(f'  Total: {len(dataset)} questions')
    logger.info(f'  Completed: {len(completed)}')
    logger.info(f'  Skipped: {skipped}')
    logger.info(f'  Errors: {errors}')
    logger.info(f'  Total time: {total_elapsed:.1f}s')
    if completed:
        logger.info(f'  Avg turns: {avg_turns:.1f}')

    summary = {
        'total_questions': len(dataset),
        'completed': len(completed),
        'skipped': skipped,
        'errors': errors,
        'total_time_seconds': round(total_elapsed, 1),
        'avg_turns': round(avg_turns, 1),
    }
    with (args.output_root / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info('Done.')


if __name__ == '__main__':
    main()
