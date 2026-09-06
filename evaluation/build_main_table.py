#!/usr/bin/env python3
"""Validate complete fixed-identity evaluations and build the main table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "outputs" / "main_table"
METRICS = (
    "si_sdr",
    "si_sdr_i",
    "sdr",
    "sdr_i",
    "pesq_nb",
    "stoi",
    "estoi",
    "sir",
    "sar",
    "dnsmos_p835_sig",
    "dnsmos_p835_bak",
    "dnsmos_p835_ovrl",
)


@dataclass(frozen=True)
class ModelResult:
    key: str
    model: str
    checkpoint: str
    sample_rate_hz: int = 8000
    segment_seconds: float = 4.0
    expected_samples: int = 2801


RESULTS = (
    ModelResult("dars", "DARS (ES patience 10)", "dars_mixed_p10"),
    ModelResult("spmamba", "SPMamba", "mixed_g3090_p5_0227"),
    ModelResult("tdanet", "TDANet-Large", "tdanet_large200_nopit"),
    ModelResult(
        "tflocoformer",
        "TF-Locoformer-M",
        "tflocoformer_m_nopit20_mixed_dataset_official_stable",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--input-name",
        default="metrics_extended_fixed_identity",
        help="Metric filename stem emitted by evaluate_main_table.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "main_separation_metrics.csv",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "main_separation_metrics.md",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_result(
    results_root: Path, input_name: str, spec: ModelResult
) -> Tuple[Dict[str, object], Sequence[str]]:
    summary_path = results_root / spec.key / (input_name + ".summary.json")
    csv_path = results_root / spec.key / (input_name + ".csv")
    if not summary_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(
            "Missing complete evaluation for {} under {}".format(spec.model, summary_path.parent)
        )

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("alignment") != "fixed":
        raise ValueError("{} is not a fixed-identity evaluation".format(summary_path))
    if int(summary.get("num_samples", -1)) != spec.expected_samples:
        raise ValueError(
            "{} has {} samples, expected {}".format(
                spec.model, summary.get("num_samples"), spec.expected_samples
            )
        )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    example_rows = [row for row in raw_rows if str(row.get("snt_id", "")).endswith(".wav")]
    if len(example_rows) != spec.expected_samples:
        raise ValueError(
            "{} has {} per-example rows, expected {}".format(
                csv_path, len(example_rows), spec.expected_samples
            )
        )
    sample_ids = [str(row["snt_id"]) for row in example_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("{} contains duplicate sample IDs".format(csv_path))
    if {row.get("alignment") for row in example_rows} != {"fixed"}:
        raise ValueError("{} contains a non-fixed row".format(csv_path))

    means = summary.get("mean")
    if not isinstance(means, dict):
        raise ValueError("Missing mean metrics in {}".format(summary_path))
    result: Dict[str, object] = {
        "model": spec.model,
        "checkpoint": spec.checkpoint,
        "dataset": "HETMIXR",
        "training_assignment": "fixed distance order",
        "evaluation_assignment": "fixed identity",
        "sample_rate_hz": spec.sample_rate_hz,
        "segment_seconds": spec.segment_seconds,
        "num_samples": spec.expected_samples,
    }
    for metric in METRICS:
        value = float(means[metric])
        if not math.isfinite(value):
            raise ValueError("{} is non-finite in {}".format(metric, summary_path))
        result[metric] = value
    return result, sample_ids


def ensure_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = (
        "Model",
        "N",
        "SI-SDR",
        "SI-SDRi",
        "SDR",
        "SDRi",
        "PESQ-NB",
        "STOI",
        "ESTOI",
        "SIR",
        "SAR",
        "DNSMOS SIG",
        "BAK",
        "OVRL",
    )
    lines = [
        "# Main separation results",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index < 2 else "---:" for index in range(len(headers))) + " |",
    ]
    for row in rows:
        values = (str(row["model"]), str(row["num_samples"])) + tuple(
            "{:.3f}".format(float(row[metric])) for metric in METRICS
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        (
            "",
            "All systems use the same 8 kHz, 4 s, 2,801-example HETMIXR test set. "
            "Output identity is preserved at evaluation time.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    results_root = args.results_root.expanduser().resolve()
    loaded = [load_result(results_root, args.input_name, spec) for spec in RESULTS]
    rows = [item[0] for item in loaded]
    reference_ids = list(loaded[0][1])
    for spec, (_, sample_ids) in zip(RESULTS[1:], loaded[1:]):
        if list(sample_ids) != reference_ids:
            raise ValueError("{} does not use the same ordered test set".format(spec.model))

    output_csv = args.output_csv.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    ensure_output(output_csv, args.overwrite)
    ensure_output(output_markdown, args.overwrite)
    write_csv(output_csv, rows)
    write_markdown(output_markdown, rows)
    print("csv={}".format(output_csv))
    print("markdown={}".format(output_markdown))


if __name__ == "__main__":
    main(parse_args())
