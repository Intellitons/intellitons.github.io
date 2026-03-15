#!/usr/bin/env python3
"""Standalone runner for the paper-focused Intelliton experiment subset.

Supports running one model or all five paper models in sequence::

    # Single model
    python run_paper.py --model-path /data/users/xiongzhaoping/Intel2/Qwen3-8B-Base

    # All five models
    python run_paper.py --all-models

    # Specific subset
    python run_paper.py --model-path /path/to/ModelA --model-path /path/to/ModelB
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from typing import List

import torch

from src.config import IntellitonConfig
from src.paper_pipeline import PaperIntellitonPipeline

logger = logging.getLogger(__name__)

# ── Canonical model list for the paper ──────────────────────────────
ALL_MODELS: List[str] = [
    "/data/users/xiongzhaoping/Intel2/Qwen3-4B-Base",
    "/data/users/xiongzhaoping/Intel2/Qwen3-8B-Base",
    "/data/users/xiongzhaoping/Intel2/Mistral-7B-v0.3",
    "/data/users/xiongzhaoping/Intel2/Qwen3-4B",
    "/data/users/xiongzhaoping/Intel2/Qwen3-8B",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-focused Intelliton experiments: discovery, characterization, applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Model selection (mutually informative, not exclusive) ───────
    parser.add_argument(
        "--model-path",
        action="append",
        default=None,
        help="Path to a model directory (can be repeated for multiple models)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all five paper models in sequence",
    )

    # ── Output ──────────────────────────────────────────────────────
    parser.add_argument(
        "--output-root",
        default="./results_paper",
        help="Root output directory; per-model results go into <root>/<model_name>/",
    )

    # ── Analysis knobs ──────────────────────────────────────────────
    parser.add_argument("--seq-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--n-modes", type=int, default=20, help="Number of SVD modes")
    parser.add_argument("--max-prompts", type=int, default=None, help="Max prompts per category")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--skip-hallucination", action="store_true", help="Skip hallucination diagnostics")
    parser.add_argument("--skip-trajectory", action="store_true", help="Skip trajectory analysis")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Max new tokens for trajectory generation")
    parser.add_argument("--local-window", type=int, default=6, help="Local wave-packet window size for trajectory analysis")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling for trajectory generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for trajectory generation sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for trajectory generation sampling")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _resolve_model_list(args) -> List[str]:
    """Return the list of model paths to process."""
    if args.all_models:
        return list(ALL_MODELS)
    if args.model_path:
        return list(args.model_path)
    # Default: all models
    return list(ALL_MODELS)


def _model_short_name(model_path: str) -> str:
    """Derive a filesystem-safe short name from a model path."""
    return os.path.basename(model_path.rstrip("/"))


def run_single_model(model_path: str, args) -> None:
    """Run the full paper pipeline for one model."""
    model_name = _model_short_name(model_path)
    output_dir = os.path.join(args.output_root, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Per-model log file (console handler is shared)
    file_handler = logging.FileHandler(os.path.join(output_dir, "analysis.log"))
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    cfg = IntellitonConfig(
        model_path=model_path,
        output_dir=output_dir,
        analysis_seq_len=args.seq_len,
        n_top_modes=args.n_modes,
        max_prompts_per_category=args.max_prompts,
    )
    if args.gpu is not None:
        cfg.device_map = "cuda:0"
    cfg.auto_configure_from_model()

    logger.info("=" * 60)
    logger.info("PAPER INTELLITON ANALYSIS")
    logger.info("=" * 60)
    logger.info("Model : %s", cfg.model_name)
    logger.info("Path  : %s", cfg.model_path)
    logger.info("Layers: %d, Hidden: %d, Heads: %d, HeadDim: %d",
                cfg.num_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim)
    logger.info("Output: %s", cfg.output_dir)
    logger.info("Seq len: %s, Modes: %s", cfg.analysis_seq_len, cfg.n_top_modes)

    pipeline = PaperIntellitonPipeline(cfg)
    catalog = pipeline.run(
        skip_hallucination=args.skip_hallucination,
        skip_trajectory=args.skip_trajectory,
        max_new_tokens=args.max_new_tokens,
        local_window=args.local_window,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n")
    catalog.print_table()
    print(f"\nResults for {model_name} saved to: {output_dir}/")

    # Cleanup to free GPU memory before loading the next model
    del pipeline, catalog
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Remove per-model file handler so the next model gets its own
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model_paths = _resolve_model_list(args)
    n_models = len(model_paths)

    logger.info("=" * 60)
    logger.info("INTELLITON PAPER EXPERIMENT SUITE")
    logger.info("Models to process: %d", n_models)
    for i, mp in enumerate(model_paths, 1):
        logger.info("  [%d/%d] %s", i, n_models, mp)
    logger.info("=" * 60)

    for idx, model_path in enumerate(model_paths, 1):
        model_name = _model_short_name(model_path)
        logger.info("\n" + "#" * 60)
        logger.info("# MODEL %d/%d: %s", idx, n_models, model_name)
        logger.info("#" * 60)
        try:
            run_single_model(model_path, args)
        except Exception:
            logger.exception("Failed on model %s — skipping", model_name)
            continue

    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE — %d model(s) processed", n_models)
    logger.info("Results root: %s", args.output_root)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
