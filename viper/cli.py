"""``viper-eval`` console entry point.

Run a vision-language model end-to-end against the VIPER benchmark and
write a paper-aligned ``results.json`` to ``eval_logs/<model>/<timestamp>/``.

Examples:

    # OpenAI GPT
    viper-eval --model gpt-4o-mini

    # vLLM-served local model (OpenAI-compatible)
    viper-eval --model my-org/my-vlm \\
        --api-base http://localhost:8000/v1 --api-key dummy

    # Smoke test
    viper-eval --model gpt-4o-mini --limit 5

    # Image ablation (paper §3)
    viper-eval --model gpt-4o-mini --ablation no-image
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from viper import __version__
from viper.config import (
    EvalConfig,
    resolve_extract_model,
    resolve_int_env,
    resolve_judge_model,
)
from viper.data import load_viper
from viper.inference import InferenceItem, run_inference
from viper.scoring import score


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viper-eval",
        description=(
            "Run a VLM against the VIPER veterinary pathology benchmark. "
            "Reproduces the paper's MCQ / KPrim / free-text scoring exactly."
        ),
    )
    parser.add_argument("--version", action="version", version=f"viper-bench {__version__}")

    parser.add_argument(
        "--model",
        required=True,
        help="Model name to send in the OpenAI Chat Completions request.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override the API base URL (defaults to OpenAI). Example: http://localhost:8000/v1.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override the API key. Defaults to OPENAI_API_KEY from env / .env.",
    )

    data_grp = parser.add_mutually_exclusive_group()
    data_grp.add_argument(
        "--hf-dataset",
        default="MahmoodLab/viper",
        help="Hugging Face dataset ID to load (default: MahmoodLab/viper).",
    )
    data_grp.add_argument(
        "--data",
        default=None,
        help="Path to a local viper.parquet (overrides --hf-dataset).",
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Optional Hub revision (commit / branch / tag) for --hf-dataset.",
    )
    parser.add_argument("--split", default="test", help='HF split name (default: "test").')

    parser.add_argument(
        "--mcq-rotations",
        type=int,
        default=5,
        help="Number of cyclic-shift permutations per MCQ (paper default: 5).",
    )
    parser.add_argument(
        "--ablation",
        choices=("none", "black-image", "no-image", "random-image"),
        default="none",
        help="Optional ablation transform applied to every image.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N samples (smoke test).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens generated per request.",
    )

    parser.add_argument(
        "--judge-model",
        default=None,
        help="LLM judge model for free-text scoring. Default: $JUDGE_MODEL or gpt-5.4.",
    )
    parser.add_argument(
        "--judge-api-base",
        default=None,
        help="Override API base for the judge (defaults to --api-base if unset, then OpenAI).",
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        help="Override API key for the judge (defaults to OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--extract-model",
        default=None,
        help="LLM model for MCQ/TF extraction fallback. Default: $EXTRACT_MODEL or gpt-4o-mini.",
    )

    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=resolve_int_env("REQUEST_CONCURRENCY", 32),
        help="Concurrent requests to the model under test.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=resolve_int_env("JUDGE_CONCURRENCY", 50),
        help="Concurrent calls to the judge model.",
    )

    parser.add_argument(
        "--output",
        default="eval_logs",
        help="Root directory for results (default: eval_logs/).",
    )
    parser.add_argument(
        "--judge-cache",
        default=None,
        help="Override judge cache directory (default: <output>/.judge_cache/).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the progress bar.",
    )
    return parser


def _load_dotenv_if_present() -> None:
    """Best-effort .env loader so OPENAI_API_KEY etc. are picked up automatically."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        return


def _resolve_api_key(cli_key: str | None) -> str | None:
    if cli_key:
        return cli_key
    return os.environ.get("OPENAI_API_KEY")


def _items_to_json(items: list[InferenceItem]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in items]


def _samples_for_scoring(items: list[InferenceItem]) -> list[dict[str, Any]]:
    """Convert per-trial InferenceItems into the dicts ``score`` expects."""
    out = []
    for it in items:
        d = it.to_dict()
        out.append(d)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    _load_dotenv_if_present()

    api_key = _resolve_api_key(args.api_key)
    judge_model = resolve_judge_model(args.judge_model)
    extract_model = resolve_extract_model(args.extract_model)

    dataset_source = args.data or args.hf_dataset
    print(f"[viper-eval] Loading dataset: {dataset_source}", file=sys.stderr)
    ds = load_viper(
        source=dataset_source,
        split=args.split,
        revision=args.dataset_revision,
    )
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"[viper-eval] --limit applied: evaluating first {len(ds)} samples", file=sys.stderr)

    cfg = EvalConfig.build(
        model=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        dataset_source=dataset_source,
        dataset_revision=args.dataset_revision,
        limit=args.limit,
        mcq_rotations=args.mcq_rotations,
        ablation=args.ablation,
        judge_model=judge_model,
        extract_model=extract_model,
        request_concurrency=args.request_concurrency,
        judge_concurrency=args.judge_concurrency,
    )
    out_root = (
        Path(args.output)
        / cfg.output_dir_name()
        / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    judge_cache_dir = (
        Path(args.judge_cache) if args.judge_cache else out_root.parent.parent / ".judge_cache"
    )

    print(f"[viper-eval] Running {args.model} on {len(ds)} samples", file=sys.stderr)
    print(
        f"[viper-eval] MCQ rotations: {args.mcq_rotations}, ablation: {args.ablation}",
        file=sys.stderr,
    )
    print(f"[viper-eval] Output: {out_root}", file=sys.stderr)

    items = run_inference(
        dataset=ds,
        model=args.model,
        api_base=args.api_base,
        api_key=api_key,
        judge_model=judge_model,
        judge_api_base=args.judge_api_base or args.api_base,
        judge_api_key=args.judge_api_key or api_key,
        extract_model=extract_model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        mcq_rotations=args.mcq_rotations,
        ablation=args.ablation,
        request_concurrency=args.request_concurrency,
        judge_concurrency=args.judge_concurrency,
        judge_cache_dir=judge_cache_dir,
        progress=not args.quiet,
    )

    metrics = score(_samples_for_scoring(items))

    payload = {
        "config": dataclasses.asdict(cfg),
        **metrics,
    }
    (out_root / "results.json").write_text(json.dumps(payload, indent=2))
    samples_path = out_root / "samples.jsonl"
    with samples_path.open("w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")

    overall = metrics.get("overall_score", 0.0)
    mcq = metrics.get("mcq_accuracy")
    kprim = metrics.get("kprim_score")
    ftj = metrics.get("free_text_judge")
    ftr = metrics.get("free_text_rouge")

    print("", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"VIPER results for {args.model}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Overall score:     {overall:.4f}", file=sys.stderr)
    if mcq is not None:
        print(
            f"  MCQ accuracy:      {mcq:.4f} (mean over {args.mcq_rotations} rotations)",
            file=sys.stderr,
        )
    if kprim is not None:
        print(f"  KPrim score:       {kprim:.4f}", file=sys.stderr)
    if ftj is not None:
        print(f"  Free-text judge:   {ftj:.4f} ({judge_model})", file=sys.stderr)
    if ftr is not None:
        print(f"  Free-text ROUGE-L: {ftr:.4f}", file=sys.stderr)
    print(f"  Results:  {out_root / 'results.json'}", file=sys.stderr)
    print(f"  Samples:  {samples_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
