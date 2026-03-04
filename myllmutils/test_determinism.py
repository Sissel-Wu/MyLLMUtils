"""
LLM API Determinism Tester.

Tests whether an LLM API produces deterministic outputs by sending the same prompts
multiple times at various concurrency levels and measuring response consistency.

Default metrics (fast): exact match, longest common prefix, SequenceMatcher similarity.
Optional metric (slow): Levenshtein edit distance (--edit_distance flag).

Supports caching LLM responses to a JSONL file (--cache_file) to avoid re-querying
on restarts.

Usage:
    python -m myllmutils.test_determinism \
        --dataset <path_or_hf_name> \
        --text_column <column> \
        --api_config <config.yaml> \
        --batch_sizes 1,2,4,8 \
        --num_repetitions 8

Each batch_size value controls the max_workers concurrency level. The --num_repetitions
parameter controls how many total responses to collect per prompt per batch setting.
Requests are dispatched in rounds of batch_size concurrent workers.
"""

import argparse
import difflib
import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import httpx

from myllmutils.batch_process import load_api_config, process_single_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
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
    prompt: str, prompt_idx: int, rep_idx: int, batch_size: int, protocol: str
) -> Dict[str, Any]:
    if protocol == "gemini":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    else:
        body = {"messages": [{"role": "user", "content": prompt}]}
    return {
        "custom_id": f"prompt-{prompt_idx}_bs-{batch_size}_rep-{rep_idx}",
        "body": body,
    }


# ---------------------------------------------------------------------------
# Response text extraction
# ---------------------------------------------------------------------------

def extract_response_text(result: Dict[str, Any], protocol: str) -> str:
    response = result["response"]
    if protocol == "gemini":
        parts = response["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts)
    else:
        msg = response["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if reasoning:
            return reasoning + "\n" + content
        return content


# ---------------------------------------------------------------------------
# Response cache (JSONL-based resumability)
# ---------------------------------------------------------------------------

class ResponseCache:
    """Thread-safe JSONL cache for LLM responses."""

    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self._lock = threading.Lock()
        self._cache: Dict[str, str] = {}  # custom_id -> response_text
        self._load()

    def _load(self):
        if not os.path.exists(self.cache_file):
            return
        with open(self.cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._cache[entry["custom_id"]] = entry["response_text"]
        logger.info("Loaded %d cached responses from %s", len(self._cache), self.cache_file)

    def get(self, custom_id: str) -> Optional[str]:
        return self._cache.get(custom_id)

    def put(self, custom_id: str, response_text: str):
        with self._lock:
            self._cache[custom_id] = response_text
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"custom_id": custom_id, "response_text": response_text}, ensure_ascii=False) + "\n")

    def __contains__(self, custom_id: str) -> bool:
        return custom_id in self._cache


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def longest_common_prefix_length(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def similarity_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m < n:
        return levenshtein_distance(b, a)
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_pair_metrics(a: str, b: str, use_edit_distance: bool) -> Dict[str, Any]:
    if a == b:
        metrics = {
            "lcp_length": len(a),
            "lcp_ratio": 1.0,
        }
        if use_edit_distance:
            metrics["similarity"] = 1.0
            metrics["edit_distance"] = 0
        return metrics

    lcp = longest_common_prefix_length(a, b)
    max_len = max(len(a), len(b))
    metrics = {
        "lcp_length": lcp,
        "lcp_ratio": lcp / max_len if max_len > 0 else 1.0,
    }
    if use_edit_distance:
        metrics["similarity"] = similarity_ratio(a, b)
        metrics["edit_distance"] = levenshtein_distance(a, b)
    return metrics


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def run_batch(
    prompts: List[str],
    batch_size: int,
    num_repetitions: int,
    api_config_path: str,
    protocol: str,
    args,
    cache: Optional[ResponseCache] = None,
) -> Dict[int, List[Optional[str]]]:
    """Run all prompts with given concurrency level, collecting num_repetitions responses.

    Dispatches requests in rounds of batch_size concurrent workers.
    Returns dict mapping prompt_idx -> list of response texts (None on failure).
    """
    results: Dict[int, List[Optional[str]]] = {}

    verify = not args.no_verify
    client = httpx.Client(verify=verify, timeout=args.timeout)

    try:
        for prompt_idx, prompt in enumerate(prompts):
            responses: List[Tuple[int, Optional[str]]] = []

            # Check cache for already-completed reps
            tasks_to_run = []
            for rep_idx in range(num_repetitions):
                task = create_task(prompt, prompt_idx, rep_idx, batch_size, protocol)
                custom_id = task["custom_id"]
                if cache and custom_id in cache:
                    responses.append((rep_idx, cache.get(custom_id)))
                else:
                    tasks_to_run.append((rep_idx, task))

            # Dispatch uncached tasks in rounds of batch_size
            for round_start in range(0, len(tasks_to_run), batch_size):
                round_tasks = tasks_to_run[round_start:round_start + batch_size]
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = {}
                    for rep_idx, task in round_tasks:
                        future = executor.submit(
                            process_single_query,
                            task,
                            api_config_path,
                            client,
                            "",  # mask_input_fields
                            args.stream,
                            args.max_retries,
                            args.timeout,
                            args.no_verify,
                        )
                        futures[future] = (rep_idx, task["custom_id"])

                    for future in as_completed(futures):
                        rep_idx, custom_id = futures[future]
                        try:
                            success, result = future.result()
                            if success:
                                text = extract_response_text(result, protocol)
                                responses.append((rep_idx, text))
                                if cache:
                                    cache.put(custom_id, text)
                            else:
                                logger.warning(
                                    "Prompt %d rep %d failed: %s",
                                    prompt_idx, rep_idx, result,
                                )
                                responses.append((rep_idx, None))
                        except Exception as e:
                            logger.warning(
                                "Prompt %d rep %d exception: %s",
                                prompt_idx, rep_idx, e,
                            )
                            responses.append((rep_idx, None))

            cached = num_repetitions - len(tasks_to_run)
            responses.sort(key=lambda x: x[0])
            results[prompt_idx] = [text for _, text in responses]

            status = f"  batch_size={batch_size} | prompt {prompt_idx + 1}/{len(prompts)} done"
            if cached > 0:
                status += f" ({cached} cached)"
            logger.info(status)
    finally:
        client.close()

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_intra_batch(
    responses: List[Optional[str]], use_edit_distance: bool
) -> Dict[str, Any]:
    valid = [r for r in responses if r is not None]
    if len(valid) < 2:
        result = {
            "all_identical": len(set(valid)) <= 1,
            "num_unique": len(set(valid)),
            "mean_lcp_ratio": 1.0 if valid else 0.0,
            "min_lcp_length": len(valid[0]) if valid else 0,
        }
        if use_edit_distance:
            result["mean_similarity"] = 1.0 if valid else 0.0
            result["max_edit_distance"] = 0
        return result

    unique = list(set(valid))
    num_unique = len(unique)

    if num_unique == 1:
        result = {
            "all_identical": True,
            "num_unique": 1,
            "mean_lcp_ratio": 1.0,
            "min_lcp_length": len(unique[0]),
        }
        if use_edit_distance:
            result["mean_similarity"] = 1.0
            result["max_edit_distance"] = 0
        return result

    # Compute metrics only between distinct unique strings
    pair_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
    unique_idx = {s: i for i, s in enumerate(unique)}
    for i in range(num_unique):
        for j in range(i + 1, num_unique):
            pair_cache[(i, j)] = compute_pair_metrics(
                unique[i], unique[j], use_edit_distance
            )

    # Aggregate over all pairs (using cache for repeated values)
    lcp_ratios = []
    lcp_lengths = []
    sims = [] if use_edit_distance else None
    edit_dists = [] if use_edit_distance else None
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if valid[i] == valid[j]:
                lcp_ratios.append(1.0)
                lcp_lengths.append(len(valid[i]))
                if use_edit_distance:
                    sims.append(1.0)
                    edit_dists.append(0)
            else:
                ui, uj = unique_idx[valid[i]], unique_idx[valid[j]]
                key = (min(ui, uj), max(ui, uj))
                m = pair_cache[key]
                lcp_ratios.append(m["lcp_ratio"])
                lcp_lengths.append(m["lcp_length"])
                if use_edit_distance:
                    sims.append(m["similarity"])
                    edit_dists.append(m["edit_distance"])

    result = {
        "all_identical": False,
        "num_unique": num_unique,
        "mean_lcp_ratio": sum(lcp_ratios) / len(lcp_ratios),
        "min_lcp_length": min(lcp_lengths),
    }
    if use_edit_distance:
        result["mean_similarity"] = sum(sims) / len(sims)
        result["max_edit_distance"] = max(edit_dists)
    return result


def analyze_cross_batch(
    all_batch_results: Dict[int, List[Optional[str]]],
    use_edit_distance: bool,
) -> Dict[str, Any]:
    representatives = {}
    for bs, responses in all_batch_results.items():
        valid = [r for r in responses if r is not None]
        if valid:
            representatives[bs] = valid[0]

    batch_sizes = sorted(representatives.keys())
    if len(batch_sizes) < 2:
        return {"all_identical": True, "pairwise_comparisons": []}

    comparisons = []
    for i in range(len(batch_sizes)):
        for j in range(i + 1, len(batch_sizes)):
            bs_a, bs_b = batch_sizes[i], batch_sizes[j]
            m = compute_pair_metrics(
                representatives[bs_a], representatives[bs_b], use_edit_distance
            )
            comp = {
                "batch_size_a": bs_a,
                "batch_size_b": bs_b,
                "lcp_length": m["lcp_length"],
                "lcp_ratio": m["lcp_ratio"],
            }
            if use_edit_distance:
                comp["similarity"] = m["similarity"]
                comp["edit_distance"] = m["edit_distance"]
            comparisons.append(comp)

    all_texts = list(representatives.values())
    return {
        "all_identical": len(set(all_texts)) <= 1,
        "pairwise_comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    model_name: str,
    dataset_name: str,
    num_prompts: int,
    batch_sizes: List[int],
    num_repetitions: int,
    per_prompt: List[Dict[str, Any]],
    use_edit_distance: bool,
):
    sep = "=" * 64
    print(f"\n{sep}")
    print("LLM Determinism Test Report")
    print(sep)
    print(f"Model: {model_name} | Dataset: {dataset_name} ({num_prompts} prompts)")
    print(f"Batch sizes (concurrency): {batch_sizes} | Repetitions: {num_repetitions}")
    print("-" * 64)

    for bs in batch_sizes:
        print(f"\n--- Intra-Batch (batch_size={bs}) ---")
        if num_repetitions < 2:
            print("  (Only 1 repetition - no intra-batch comparison)")
            continue

        identical = sum(1 for p in per_prompt if p["intra_batch"][bs]["all_identical"])
        total = len(per_prompt)
        pct = identical / total * 100 if total else 0

        lcp_ratios = [p["intra_batch"][bs]["mean_lcp_ratio"] for p in per_prompt]
        mean_lcp = sum(lcp_ratios) / len(lcp_ratios) if lcp_ratios else 0

        print(f"  Exact match: {identical}/{total} ({pct:.1f}%)")
        print(f"  Mean LCP ratio: {mean_lcp:.4f}")

        if use_edit_distance:
            sims = [p["intra_batch"][bs]["mean_similarity"] for p in per_prompt]
            mean_sim = sum(sims) / len(sims) if sims else 0
            print(f"  Mean similarity: {mean_sim:.4f}")

        divergent = [
            (p["prompt_idx"], p["intra_batch"][bs])
            for p in per_prompt
            if not p["intra_batch"][bs]["all_identical"]
        ]
        if divergent:
            print(f"  Divergent prompts ({len(divergent)}):")
            for idx, analysis in divergent[:10]:
                line = f"    Prompt {idx}: lcp={analysis['min_lcp_length']}"
                if use_edit_distance:
                    line += f", sim={analysis['mean_similarity']:.4f}"
                    line += f", max_edit_dist={analysis['max_edit_distance']}"
                print(line)
            if len(divergent) > 10:
                print(f"    ... and {len(divergent) - 10} more")

    # Cross-batch
    print("\n--- Cross-Batch ---")
    identical = sum(1 for p in per_prompt if p["cross_batch"]["all_identical"])
    total = len(per_prompt)
    pct = identical / total * 100 if total else 0
    print(f"  Identical across all batch sizes: {identical}/{total} ({pct:.1f}%)")

    all_cross_lcps = []
    for p in per_prompt:
        for comp in p["cross_batch"]["pairwise_comparisons"]:
            all_cross_lcps.append(comp["lcp_ratio"])
    if all_cross_lcps:
        mean_cross_lcp = sum(all_cross_lcps) / len(all_cross_lcps)
        print(f"  Mean cross-batch LCP ratio: {mean_cross_lcp:.4f}")

    if use_edit_distance:
        all_cross_sims = []
        for p in per_prompt:
            for comp in p["cross_batch"]["pairwise_comparisons"]:
                all_cross_sims.append(comp["similarity"])
        if all_cross_sims:
            mean_cross = sum(all_cross_sims) / len(all_cross_sims)
            print(f"  Mean cross-batch similarity: {mean_cross:.4f}")

    # Overall determinism score based on LCP ratio
    all_lcps = []
    for p in per_prompt:
        for bs in batch_sizes:
            if bs >= 2:
                all_lcps.append(p["intra_batch"][bs]["mean_lcp_ratio"])
        for comp in p["cross_batch"]["pairwise_comparisons"]:
            all_lcps.append(comp["lcp_ratio"])

    if all_lcps:
        overall = sum(all_lcps) / len(all_lcps)
        print(f"\n--- Overall Determinism Score (LCP): {overall:.4f} ---")

    print(sep + "\n")


def save_json_report(
    output_file: str,
    args,
    model_name: str,
    batch_sizes: List[int],
    prompts: List[str],
    per_prompt: List[Dict[str, Any]],
):
    report = {
        "metadata": {
            "api_config": args.api_config,
            "model": model_name,
            "dataset": args.dataset,
            "text_column": args.text_column,
            "batch_sizes": batch_sizes,
            "num_repetitions": args.num_repetitions,
            "num_prompts": len(prompts),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "per_prompt": per_prompt,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Detailed report saved to %s", output_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test LLM API determinism by repeating prompts and measuring output consistency.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a local file or a HuggingFace dataset identifier.\n"
        "Loaded via datasets.load_dataset().",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
        help="Name of the column containing prompt text.",
    )
    parser.add_argument(
        "--api_config",
        type=str,
        required=True,
        help="File path to the API config file (json or yaml).\n"
        "Should contain greedy decoding settings and max_tokens.",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        required=True,
        help="Comma-separated list of concurrency levels (e.g., '1,2,4,8').\n"
        "Each value = max_workers for ThreadPoolExecutor.\n"
        "Requests are dispatched in rounds of this many concurrent workers.",
    )
    parser.add_argument(
        "--num_repetitions",
        type=int,
        required=True,
        help="Number of responses to collect per prompt per batch setting.\n"
        "Same count is used for every batch_size, ensuring fair comparison.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Which split of the dataset to use (default: 'train').",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Config/subset name for the HuggingFace dataset.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=0,
        help="Maximum number of prompts to test (0 = all).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode for API calls.",
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
        "--output_file",
        type=str,
        default=None,
        help="Path to save detailed JSON results.",
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default=None,
        help="Path to JSONL cache file for LLM responses.\n"
        "Enables resumability: cached responses are reused on restart.\n"
        "New responses are appended as they complete.",
    )
    parser.add_argument(
        "--edit_distance",
        action="store_true",
        help="Compute Levenshtein edit distance (slow for long responses).\n"
        "Default metrics (LCP, similarity ratio) are always computed.",
    )

    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    for bs in batch_sizes:
        if bs < 1:
            logger.critical("Batch sizes must be >= 1, got %d", bs)
            sys.exit(1)
    if args.num_repetitions < 1:
        logger.critical("--num_repetitions must be >= 1, got %d", args.num_repetitions)
        sys.exit(1)

    api_config = load_api_config(args.api_config)
    protocol = api_config.get("protocol", "openai")
    model_name = api_config.get("model", "unknown")

    cache = ResponseCache(args.cache_file) if args.cache_file else None

    logger.info("Loading dataset: %s (split=%s, column=%s)", args.dataset, args.dataset_split, args.text_column)
    prompts = load_prompts(args)
    logger.info("Loaded %d prompts.", len(prompts))

    # all_results[batch_size][prompt_idx] = list of response texts
    all_results: Dict[int, Dict[int, List[Optional[str]]]] = {}

    for bs in batch_sizes:
        num_rounds = (args.num_repetitions + bs - 1) // bs
        logger.info(
            "Running batch_size=%d (%d repetitions, %d rounds of %d workers)...",
            bs, args.num_repetitions, num_rounds, bs,
        )
        all_results[bs] = run_batch(prompts, bs, args.num_repetitions, args.api_config, protocol, args, cache)

    # Analyze
    per_prompt = []
    for prompt_idx in range(len(prompts)):
        intra = {}
        batch_responses = {}
        for bs in batch_sizes:
            responses = all_results[bs][prompt_idx]
            batch_responses[bs] = responses
            intra[bs] = analyze_intra_batch(responses, args.edit_distance)

        cross = analyze_cross_batch(batch_responses, args.edit_distance)

        per_prompt.append({
            "prompt_idx": prompt_idx,
            "prompt_text_preview": prompts[prompt_idx][:100],
            "intra_batch": intra,
            "cross_batch": cross,
            "responses": {bs: all_results[bs][prompt_idx] for bs in batch_sizes},
        })

    print_report(model_name, args.dataset, len(prompts), batch_sizes, args.num_repetitions, per_prompt, args.edit_distance)

    if args.output_file:
        save_json_report(args.output_file, args, model_name, batch_sizes, prompts, per_prompt)


if __name__ == "__main__":
    main()
