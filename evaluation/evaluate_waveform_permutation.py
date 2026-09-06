#!/usr/bin/env python3
"""Evaluate whether raw separator outputs already use the target source order."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.evaluate_checkpoint import (
    build_bundle,
    load_yaml,
    resolve_directory,
    resolve_file,
    seed_everything,
    unique_output_path,
)
from evaluation.extended_metrics import _ensure_2d, si_sdr_sources


LOGGER = logging.getLogger("waveform_permutation_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        choices=("dars", "spmamba", "tdanet", "tflocoformer"),
        required=True,
    )
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-name",
        default="waveform_permutation_accuracy.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for diagnostic output (default: checkpoint/results).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tie-tolerance-db", type=float, default=1e-6)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def permutation_scores(
    estimate: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[float, float, float]:
    """Score the raw output order and a swapped diagnostic without reordering."""

    estimate = _ensure_2d(estimate.detach().float().cpu())
    reference = _ensure_2d(reference.detach().float().cpu())
    if estimate.shape[0] != 2 or reference.shape[0] != 2:
        raise ValueError(
            "Waveform permutation accuracy requires exactly two sources, got "
            "estimate={} reference={}".format(
                tuple(estimate.shape), tuple(reference.shape)
            )
        )
    common_length = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[..., :common_length]
    reference = reference[..., :common_length]
    direct = float(si_sdr_sources(estimate, reference).mean().item())
    swapped = float(
        si_sdr_sources(estimate[[1, 0]], reference).mean().item()
    )
    return direct, swapped, direct - swapped


def classify_margin(margin_db: float, tolerance_db: float) -> str:
    if margin_db > tolerance_db:
        return "direct"
    if margin_db < -tolerance_db:
        return "swapped"
    return "tie"


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    summary_path = path.with_suffix(".summary.json")
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    temporary_path.replace(summary_path)


def main(args: argparse.Namespace) -> Path:
    if args.tie_tolerance_db < 0:
        raise ValueError("tie_tolerance_db cannot be negative")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("max_examples must be positive")

    project_dir = resolve_directory(args.project_dir)
    config_path = resolve_file(args.config)
    checkpoint_path = resolve_file(args.checkpoint)
    sys.path.insert(0, str(project_dir))
    os.chdir(project_dir)

    seed_everything(args.seed)
    config = load_yaml(config_path)
    bundle = build_bundle(args.adapter, config, checkpoint_path)
    # Dataset cropping in several projects draws from NumPy during __getitem__.
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    bundle.model.to(device)
    bundle.model.eval()

    dataset_size = len(bundle.test_set)
    start_index = max(0, int(args.start_index))
    end_index = dataset_size if args.end_index is None else int(args.end_index)
    end_index = min(dataset_size, max(start_index, end_index))
    if args.max_examples is not None:
        end_index = min(end_index, start_index + int(args.max_examples))
    requested_count = end_index - start_index
    if requested_count <= 0:
        raise ValueError("The requested dataset slice is empty")

    results_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint_path.parent / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = unique_output_path(results_dir / args.output_name)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    fieldnames = [
        "snt_id",
        "dataset_index",
        "direct_order_si_sdr_db",
        "swapped_order_diagnostic_si_sdr_db",
        "direct_minus_swapped_margin_db",
        "best_diagnostic_order",
        "raw_output_order_correct",
        "tie",
    ]

    counts = {"direct": 0, "swapped": 0, "tie": 0}
    started_at = time.time()
    try:
        with temporary_path.open("x", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            with torch.inference_mode():
                for processed, sample_index in enumerate(
                    range(start_index, end_index), start=1
                ):
                    mixture, sources, key = bundle.decode_sample(
                        bundle.test_set[sample_index]
                    )
                    output = bundle.model(mixture.unsqueeze(0).to(device))
                    if bundle.model_output_key is not None:
                        if not isinstance(output, dict):
                            raise TypeError(
                                "Expected dict output for key {}".format(
                                    bundle.model_output_key
                                )
                            )
                        output = output[bundle.model_output_key]
                    if not isinstance(output, torch.Tensor):
                        raise TypeError(
                            "Expected tensor model output, got {}".format(
                                type(output).__name__
                            )
                        )
                    estimate = output.squeeze(0).detach().cpu()
                    direct, swapped, margin = permutation_scores(
                        estimate, sources
                    )
                    decision = classify_margin(
                        margin, args.tie_tolerance_db
                    )
                    counts[decision] += 1
                    writer.writerow(
                        {
                            "snt_id": key,
                            "dataset_index": sample_index,
                            "direct_order_si_sdr_db": direct,
                            "swapped_order_diagnostic_si_sdr_db": swapped,
                            "direct_minus_swapped_margin_db": margin,
                            "best_diagnostic_order": decision,
                            "raw_output_order_correct": int(
                                decision == "direct"
                            ),
                            "tie": int(decision == "tie"),
                        }
                    )
                    if (
                        processed == 1
                        or processed % args.progress_every == 0
                        or processed == requested_count
                    ):
                        LOGGER.info(
                            "progress=%d/%d direct=%d swapped=%d ties=%d",
                            processed,
                            requested_count,
                            counts["direct"],
                            counts["swapped"],
                            counts["tie"],
                        )
            csv_file.flush()
            os.fsync(csv_file.fileno())
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    decisive = counts["direct"] + counts["swapped"]
    summary = {
        "adapter": args.adapter,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "dataset_size": dataset_size,
        "device": str(device),
        "end_index": end_index,
        "direct_order_count": counts["direct"],
        "num_examples": requested_count,
        "seed": args.seed,
        "start_index": start_index,
        "swapped_count": counts["swapped"],
        "tie_count": counts["tie"],
        "tie_tolerance_db": args.tie_tolerance_db,
        "wall_seconds": time.time() - started_at,
        "waveform_permutation_accuracy": (
            counts["direct"] / requested_count
        ),
        "waveform_permutation_accuracy_decisive": (
            counts["direct"] / decisive if decisive else None
        ),
    }
    _write_summary(output_path, summary)
    LOGGER.info("summary=%s", summary)
    LOGGER.info("saved=%s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main(parse_args())
