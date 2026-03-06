"""
LLM Logprob Entropy Sequence Calculator.

Runs LLM queries in batch with logprobs enabled, then computes approximate
per-token entropy from the returned top-K log probabilities.

The entropy at each token position is computed as:
    H = -sum(exp(lp) * lp for lp in top_logprobs)

This is an approximation (lower bound) since only the top-K logprobs are
available, not the full vocabulary distribution.

Usage:
    # From JSONL input
    python -m myllmutils.entropy_seq \
        --input_file queries.jsonl \
        --output_file results.jsonl \
        --api_config config.yaml \
        --top_logprobs 5

    # From dataset
    python -m myllmutils.entropy_seq \
        --dataset tatsu-lab/alpaca_eval \
        --text_column instruction \
        --max_prompts 50 \
        --output_file results.jsonl \
        --api_config config.yaml \
        --top_logprobs 5
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional

import httpx

from myllmutils.batch_process import load_api_config, process_single_query_async

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading (mirrors test_determinism.py pattern)
# ---------------------------------------------------------------------------

def _infer_format(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    format_map = {
        ".json": "json",
        ".jsonl": "json",
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
        ".txt": "text",
    }
    return format_map.get(ext, "json")


def load_prompts(args) -> List[str]:
    from datasets import load_dataset

    if os.path.exists(args.dataset):
        ds = load_dataset(
            _infer_format(args.dataset),
            data_files=args.dataset,
            split=args.dataset_split,
        )
    else:
        ds = load_dataset(
            args.dataset,
            name=args.dataset_config,
            split=args.dataset_split,
        )

    if args.text_column not in ds.column_names:
        logger.critical(
            "Column '%s' not found in dataset. Available columns: %s",
            args.text_column,
            ds.column_names,
        )
        sys.exit(1)

    prompts = list(ds[args.text_column])
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    return prompts


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------

def create_task(
    prompt: str, prompt_idx: int, protocol: str
) -> Dict[str, Any]:
    if protocol == "gemini":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    else:
        body = {"messages": [{"role": "user", "content": prompt}]}
    return {
        "custom_id": f"prompt-{prompt_idx}",
        "body": body,
    }


def tasks_from_dataset(prompts: List[str], protocol: str) -> List[Dict[str, Any]]:
    return [create_task(prompt, idx, protocol) for idx, prompt in enumerate(prompts)]


def load_jsonl_tasks(input_file: str) -> List[Dict[str, Any]]:
    tasks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

def load_completed_ids(output_file: str) -> set:
    completed = set()
    if not os.path.exists(output_file):
        return completed
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                completed.add(entry["custom_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# ---------------------------------------------------------------------------
# Entropy computation
# ---------------------------------------------------------------------------

def compute_entropy(logprobs_content: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-token entropy from OpenAI logprobs content array.

    Each element in logprobs_content has:
        {"token": str, "logprob": float, "top_logprobs": [{"token": str, "logprob": float}, ...]}

    Returns dict with per_token list, mean_entropy, and num_tokens.
    """
    per_token = []
    for token_info in logprobs_content:
        token = token_info.get("token", "")
        token_logprob = token_info.get("logprob", 0.0)
        top_lps = token_info.get("top_logprobs", [])

        if top_lps:
            entropy = 0.0
            for entry in top_lps:
                lp = entry.get("logprob", 0.0)
                p = math.exp(lp)
                if p > 0:
                    entropy -= p * lp
        else:
            entropy = 0.0

        per_token.append({
            "token": token,
            "logprob": token_logprob,
            "entropy": round(entropy, 6),
        })

    num_tokens = len(per_token)
    mean_entropy = (
        sum(t["entropy"] for t in per_token) / num_tokens
        if num_tokens > 0
        else 0.0
    )

    return {
        "per_token": per_token,
        "mean_entropy": round(mean_entropy, 6),
        "num_tokens": num_tokens,
    }


def _extract_entropy(response: Dict[str, Any], custom_id: str = "") -> Dict[str, Any]:
    """Extract logprobs from a response and compute entropy.

    Raises RuntimeError if logprobs are missing.
    """
    logprobs_data = (
        response.get("choices", [{}])[0]
        .get("logprobs", {})
        .get("content")
    )
    if not logprobs_data:
        label = f"Task {custom_id}: " if custom_id else ""
        raise RuntimeError(
            f"{label}response contains no logprobs. "
            "Ensure the model and API support logprobs."
        )
    return compute_entropy(logprobs_data)


# ---------------------------------------------------------------------------
# Single-prompt API
# ---------------------------------------------------------------------------

async def get_entropy_seq(
    prompt: str,
    api_config: str | Dict[str, Any],
    top_logprobs: int = 5,
    stream: bool = True,
    max_retries: int = 5,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Compute per-token entropy for a single prompt.

    Args:
        prompt: The user prompt text.
        api_config: Path to API config file or config dict.
        top_logprobs: Number of top logprobs per token (1-20).
        stream: Whether to use streaming mode.
        max_retries: Max retries for server errors.
        client: Optional httpx.AsyncClient to reuse. If None, one is created.

    Returns:
        Dict with keys: per_token, mean_entropy, num_tokens, response.

    Raises:
        RuntimeError: If the API call fails or response contains no logprobs.
    """
    config = load_api_config(api_config)
    protocol = config.get("protocol", "openai")

    task = create_task(prompt, 0, protocol)
    task["body"]["logprobs"] = True
    task["body"]["top_logprobs"] = top_logprobs

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=60, read=None if stream else 60),
        )

    try:
        success, result = await process_single_query_async(
            task, api_config, client, "", stream, max_retries,
        )
    finally:
        if owns_client:
            await client.aclose()

    if not success:
        raise RuntimeError(f"API call failed: {result}")

    response = result["response"]
    entropy = _extract_entropy(response)
    entropy["response"] = response
    return entropy

async def run_batch(
    tasks: List[Dict[str, Any]],
    api_config_path: str,
    args,
    completed_ids: set,
) -> None:
    """Run all tasks and write results to output file."""
    pending = [t for t in tasks if t["custom_id"] not in completed_ids]

    if not pending:
        logger.info("All %d tasks already completed.", len(tasks))
        return

    logger.info(
        "%d tasks to run (%d already completed).",
        len(pending), len(completed_ids),
    )

    stream = not args.no_stream

    # Inject logprobs parameters into all task bodies
    for task in pending:
        task["body"]["logprobs"] = True
        task["body"]["top_logprobs"] = args.top_logprobs

    timeout_config = httpx.Timeout(
        timeout=args.timeout,
        read=None if stream else args.timeout,
    )
    limits = httpx.Limits(
        max_connections=args.max_workers * 2,
        max_keepalive_connections=args.max_workers,
    )

    completed = 0
    semaphore = asyncio.Semaphore(args.max_workers)
    file_lock = asyncio.Lock()

    async def bounded_process(task):
        async with semaphore:
            success, result = await process_single_query_async(
                task,
                api_config_path,
                client,
                "",  # mask_input_fields
                stream,
                args.max_retries,
            )
            return task["custom_id"], success, result

    async with httpx.AsyncClient(
        timeout=timeout_config,
        verify=not args.no_verify,
        limits=limits,
    ) as client:
        coroutines = [bounded_process(task) for task in pending]

        logger.info(
            "Submitted %d tasks with max %d concurrent workers.",
            len(pending), args.max_workers,
        )

        for coro in asyncio.as_completed(coroutines):
            try:
                custom_id, success, result = await coro
                if success:
                    response = result["response"]
                    entropy = _extract_entropy(response, custom_id)

                    output_entry = {
                        "custom_id": custom_id,
                        "response": response,
                        "query": result.get("query"),
                        "entropy": entropy,
                    }

                    async with file_lock:
                        with open(args.output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

                    completed += 1
                    if completed % 10 == 0 or completed == len(pending):
                        logger.info(
                            "Progress: %d/%d completed", completed, len(pending)
                        )
                else:
                    logger.warning("Task %s failed: %s", custom_id, result)
            except Exception as e:
                logger.warning("Task %s exception: %s", custom_id, e)

    logger.info("Done. %d/%d tasks completed.", completed, len(pending))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM queries in batch with logprobs and compute per-token entropy.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_argument_group("input (choose one)")
    input_group.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input JSONL file (same format as batch_process.py).",
    )
    input_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a local file or a HuggingFace dataset identifier.\n"
        "Requires --text_column.",
    )
    input_group.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Column name containing prompt text (used with --dataset).",
    )
    input_group.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use (default: 'train').",
    )
    input_group.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Config/subset name for HuggingFace dataset.",
    )
    input_group.add_argument(
        "--max_prompts",
        type=int,
        default=0,
        help="Maximum number of prompts from dataset (0 = all).",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--api_config",
        type=str,
        required=True,
        help="Path to API config file (YAML/JSON).",
    )
    parser.add_argument(
        "--top_logprobs",
        type=int,
        default=5,
        help="Number of top logprobs per token (1-20, default: 5).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Thread pool concurrency (default: 4).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retries for server errors (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Disable SSL verification.",
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable streaming mode (default: streaming enabled).",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_file and not args.dataset:
        parser.error("Either --input_file or --dataset is required.")
    if args.input_file and args.dataset:
        parser.error("--input_file and --dataset are mutually exclusive.")
    if args.dataset and not args.text_column:
        parser.error("--text_column is required when using --dataset.")
    if not 1 <= args.top_logprobs <= 20:
        parser.error("--top_logprobs must be between 1 and 20.")

    api_config = load_api_config(args.api_config)
    protocol = api_config.get("protocol", "openai")

    # Load tasks
    if args.input_file:
        tasks = load_jsonl_tasks(args.input_file)
        logger.info("Loaded %d tasks from %s", len(tasks), args.input_file)
    else:
        prompts = load_prompts(args)
        tasks = tasks_from_dataset(prompts, protocol)
        logger.info("Loaded %d prompts from dataset '%s'", len(tasks), args.dataset)

    # Load completed ids for resumability
    completed_ids = load_completed_ids(args.output_file)
    if completed_ids:
        logger.info("Found %d already-completed tasks in %s", len(completed_ids), args.output_file)

    asyncio.run(run_batch(tasks, args.api_config, args, completed_ids))


if __name__ == "__main__":
    main()
