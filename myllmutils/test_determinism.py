"""
LLM API Determinism Tester.

Tests whether an LLM API produces deterministic outputs by sending the same prompts
multiple times at various concurrency levels and measuring response consistency
using edit distances.

Usage:
    python -m myllmutils.test_determinism \
        --dataset <path_or_hf_name> \
        --text_column <column> \
        --api_config <config.yaml> \
        --batch_sizes 1,2,4,8

Each batch_size value controls both the number of repetitions per prompt AND the
max_workers concurrency level.
"""

import argparse
import difflib
import json
import logging
import os
import sys
import time
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
# Edit distance utilities
# ---------------------------------------------------------------------------

def similarity_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def levenshtein_distance(a: str, b: str) -> int:
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


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def run_batch(
    prompts: List[str],
    batch_size: int,
    api_config_path: str,
    protocol: str,
    args,
) -> Dict[int, List[Optional[str]]]:
    """Run all prompts with given batch_size (repetitions + concurrency).

    Returns dict mapping prompt_idx -> list of response texts (None on failure).
    """
    results: Dict[int, List[Optional[str]]] = {}

    verify = not args.no_verify
    client = httpx.Client(verify=verify, timeout=args.timeout)

    try:
        for prompt_idx, prompt in enumerate(prompts):
            tasks = [
                create_task(prompt, prompt_idx, rep_idx, batch_size, protocol)
                for rep_idx in range(batch_size)
            ]

            responses: List[Tuple[int, Optional[str]]] = []

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for rep_idx, task in enumerate(tasks):
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
                    futures[future] = rep_idx

                for future in as_completed(futures):
                    rep_idx = futures[future]
                    try:
                        success, result = future.result()
                        if success:
                            text = extract_response_text(result, protocol)
                            responses.append((rep_idx, text))
                        else:
                            logger.warning(
                                "Prompt %d rep %d failed: %s",
                                prompt_idx,
                                rep_idx,
                                result,
                            )
                            responses.append((rep_idx, None))
                    except Exception as e:
                        logger.warning(
                            "Prompt %d rep %d exception: %s",
                            prompt_idx,
                            rep_idx,
                            e,
                        )
                        responses.append((rep_idx, None))

            responses.sort(key=lambda x: x[0])
            results[prompt_idx] = [text for _, text in responses]

            logger.info(
                "  batch_size=%d | prompt %d/%d done",
                batch_size,
                prompt_idx + 1,
                len(prompts),
            )
    finally:
        client.close()

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_intra_batch(responses: List[Optional[str]]) -> Dict[str, Any]:
    valid = [r for r in responses if r is not None]
    if len(valid) < 2:
        return {
            "all_identical": len(set(valid)) <= 1,
            "num_unique": len(set(valid)),
            "mean_similarity": 1.0 if valid else 0.0,
            "min_similarity": 1.0 if valid else 0.0,
            "pairwise_edit_distances": [],
        }

    ratios = []
    distances = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            ratios.append(similarity_ratio(valid[i], valid[j]))
            distances.append(levenshtein_distance(valid[i], valid[j]))

    unique = len(set(valid))
    return {
        "all_identical": unique == 1,
        "num_unique": unique,
        "mean_similarity": sum(ratios) / len(ratios),
        "min_similarity": min(ratios),
        "pairwise_edit_distances": distances,
    }


def analyze_cross_batch(
    all_batch_results: Dict[int, List[Optional[str]]],
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
            ratio = similarity_ratio(representatives[bs_a], representatives[bs_b])
            dist = levenshtein_distance(representatives[bs_a], representatives[bs_b])
            comparisons.append({
                "batch_size_a": bs_a,
                "batch_size_b": bs_b,
                "similarity": ratio,
                "edit_distance": dist,
            })

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
    per_prompt: List[Dict[str, Any]],
):
    sep = "=" * 64
    print(f"\n{sep}")
    print("LLM Determinism Test Report")
    print(sep)
    print(f"Model: {model_name} | Dataset: {dataset_name} ({num_prompts} prompts)")
    print(f"Batch sizes: {batch_sizes}")
    print("-" * 64)

    for bs in batch_sizes:
        print(f"\n--- Intra-Batch (batch_size={bs}) ---")
        if bs < 2:
            print("  (Only 1 repetition - no intra-batch comparison)")
            continue

        identical = sum(1 for p in per_prompt if p["intra_batch"][bs]["all_identical"])
        total = len(per_prompt)
        pct = identical / total * 100 if total else 0

        sims = [p["intra_batch"][bs]["mean_similarity"] for p in per_prompt]
        mean_sim = sum(sims) / len(sims) if sims else 0

        print(f"  Exact match: {identical}/{total} ({pct:.1f}%)")
        print(f"  Mean similarity: {mean_sim:.4f}")

        divergent = [
            (p["prompt_idx"], p["intra_batch"][bs])
            for p in per_prompt
            if not p["intra_batch"][bs]["all_identical"]
        ]
        if divergent:
            print(f"  Divergent prompts ({len(divergent)}):")
            for idx, analysis in divergent[:10]:
                dists = analysis["pairwise_edit_distances"]
                max_dist = max(dists) if dists else 0
                print(
                    f"    Prompt {idx}: sim={analysis['min_similarity']:.4f}, "
                    f"max_edit_dist={max_dist}"
                )
            if len(divergent) > 10:
                print(f"    ... and {len(divergent) - 10} more")

    # Cross-batch
    print("\n--- Cross-Batch ---")
    identical = sum(1 for p in per_prompt if p["cross_batch"]["all_identical"])
    total = len(per_prompt)
    pct = identical / total * 100 if total else 0
    print(f"  Identical across all batch sizes: {identical}/{total} ({pct:.1f}%)")

    all_cross_sims = []
    for p in per_prompt:
        for comp in p["cross_batch"]["pairwise_comparisons"]:
            all_cross_sims.append(comp["similarity"])
    if all_cross_sims:
        mean_cross = sum(all_cross_sims) / len(all_cross_sims)
        print(f"  Mean cross-batch similarity: {mean_cross:.4f}")

    # Overall
    all_sims = []
    for p in per_prompt:
        for bs in batch_sizes:
            if bs >= 2:
                all_sims.append(p["intra_batch"][bs]["mean_similarity"])
        for comp in p["cross_batch"]["pairwise_comparisons"]:
            all_sims.append(comp["similarity"])

    if all_sims:
        overall = sum(all_sims) / len(all_sims)
        print(f"\n--- Overall Determinism Score: {overall:.4f} ---")

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
        help="Comma-separated list of batch sizes (e.g., '1,2,4,8').\n"
        "Each value = number of repetitions AND max_workers concurrency.",
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

    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    for bs in batch_sizes:
        if bs < 1:
            logger.critical("Batch sizes must be >= 1, got %d", bs)
            sys.exit(1)

    api_config = load_api_config(args.api_config)
    protocol = api_config.get("protocol", "openai")
    model_name = api_config.get("model", "unknown")

    logger.info("Loading dataset: %s (split=%s, column=%s)", args.dataset, args.dataset_split, args.text_column)
    prompts = load_prompts(args)
    logger.info("Loaded %d prompts.", len(prompts))

    # all_results[batch_size][prompt_idx] = list of response texts
    all_results: Dict[int, Dict[int, List[Optional[str]]]] = {}

    for bs in batch_sizes:
        logger.info("Running batch_size=%d (%d repetitions, %d workers)...", bs, bs, bs)
        all_results[bs] = run_batch(prompts, bs, args.api_config, protocol, args)

    # Analyze
    per_prompt = []
    for prompt_idx in range(len(prompts)):
        intra = {}
        batch_responses = {}
        for bs in batch_sizes:
            responses = all_results[bs][prompt_idx]
            batch_responses[bs] = responses
            intra[bs] = analyze_intra_batch(responses)

        cross = analyze_cross_batch(batch_responses)

        per_prompt.append({
            "prompt_idx": prompt_idx,
            "prompt_text_preview": prompts[prompt_idx][:100],
            "intra_batch": intra,
            "cross_batch": cross,
            "responses": {bs: all_results[bs][prompt_idx] for bs in batch_sizes},
        })

    print_report(model_name, args.dataset, len(prompts), batch_sizes, per_prompt)

    if args.output_file:
        save_json_report(args.output_file, args, model_name, batch_sizes, prompts, per_prompt)


if __name__ == "__main__":
    main()
