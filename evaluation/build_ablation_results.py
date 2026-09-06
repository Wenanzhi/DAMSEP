#!/usr/bin/env python3
"""Validate the released patience-10 ablation data and render a table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MIXTURES = 2801
EXPECTED_RESPONSES = 5602
VARIANTS: Tuple[Tuple[str, str, float, float], ...] = (
    ("SPMamba (L_sep)", "dars_lsep.yml", 0.0, 0.0),
    ("DARS + L_rev", "dars_lsep_lrev.yml", 0.1, 0.0),
    ("DARS + L_resp", "dars_lsep_lresp.yml", 0.0, 0.5),
    ("DARS full", "dars_full.yml", 0.1, 0.5),
)
SIGNAL_METRICS = (
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
RIR_METRICS = (
    "recon_complex_nmse_db",
    "recon_si_sdr_db",
    "rir50_rmse",
    "edc_rmse_db",
    "response_si_nmse_db",
    "response_corr",
    "lsd_db",
    "t20_mae_s",
    "t20_valid_rate",
    "drr_mae_db",
    "drr_pearson",
    "drr_spearman",
    "c50_mae_db",
    "c80_mae_db",
    "near_far_accuracy",
    "drr_rank_accuracy",
    "drr_gap_mae_db",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT / "results" / "ablation_metrics.csv",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=REPO_ROOT / "outputs" / "ablation_metrics.md",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite(row: Dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError("Non-finite {} for {}".format(field, row.get("variant")))
    return value


def validate_config(filename: str, w_rev: float, w_recon: float) -> None:
    path = REPO_ROOT / "configs" / "ablation" / filename
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for split in ("train", "val"):
        loss = config["loss"][split]["config"]
        if loss["pit_from"] != "no_pit":
            raise ValueError("{} is not a no-PIT config".format(path))
        actual = (float(loss["w_rev"]), float(loss["w_recon"]), float(loss["w_rir"]))
        if actual != (w_rev, w_recon, 0.0):
            raise ValueError("Unexpected loss weights in {}".format(path))
    if int(config["training"]["early_stop"]["patience"]) != 10:
        raise ValueError("{} does not use early-stopping patience 10".format(path))


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_labels = [variant[0] for variant in VARIANTS]
    if [row.get("variant") for row in rows] != expected_labels:
        raise ValueError("Unexpected ablation order or labels in {}".format(path))
    for row, (_, config_name, w_rev, w_recon) in zip(rows, VARIANTS):
        validate_config(config_name, w_rev, w_recon)
        if int(row["early_stop_patience"]) != 10:
            raise ValueError("Unexpected patience for {}".format(row["variant"]))
        if row["training_assignment"] != "no_pit":
            raise ValueError("Unexpected training assignment for {}".format(row["variant"]))
        if row["evaluation_assignment"] != "fixed_identity":
            raise ValueError("Unexpected evaluation assignment for {}".format(row["variant"]))
        if int(row["num_mixtures"]) != EXPECTED_MIXTURES:
            raise ValueError("Unexpected mixture count for {}".format(row["variant"]))
        if int(row["num_responses"]) != EXPECTED_RESPONSES:
            raise ValueError("Unexpected response count for {}".format(row["variant"]))
        if (finite(row, "w_rev"), finite(row, "w_recon"), finite(row, "w_rir")) != (
            w_rev,
            w_recon,
            0.0,
        ):
            raise ValueError("CSV/config weight mismatch for {}".format(row["variant"]))
        for metric in SIGNAL_METRICS + RIR_METRICS:
            finite(row, metric)
    return rows


def render_markdown(rows: List[Dict[str, str]]) -> str:
    lines = [
        "# DARS patience-10 ablation",
        "",
        "All variants use fixed-distance-order training and fixed-identity evaluation on "
        "the same 2,801-mixture HETMIXR test set.",
        "",
        "## Separation metrics",
        "",
        "| Variant | SI-SDRi | SDRi | PESQ-NB | STOI | ESTOI | SIR | SAR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {si_sdr_i:.3f} | {sdr_i:.3f} | {pesq_nb:.3f} | "
            "{stoi:.3f} | {estoi:.3f} | {sir:.3f} | {sar:.3f} |".format(
                variant=row["variant"],
                **{name: finite(row, name) for name in SIGNAL_METRICS},
            )
        )
    lines.extend(
        (
            "",
            "## Response metrics",
            "",
            "| Variant | RIR-50 RMSE | EDC RMSE | SI-NMSE | Corr. | LSD | DRR MAE | Near/Far |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in rows:
        lines.append(
            "| {variant} | {rir50_rmse:.4f} | {edc_rmse_db:.3f} | "
            "{response_si_nmse_db:.3f} | {response_corr:.3f} | {lsd_db:.3f} | "
            "{drr_mae_db:.3f} | {near_far_accuracy:.2%} |".format(
                variant=row["variant"],
                **{name: finite(row, name) for name in RIR_METRICS},
            )
        )
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> None:
    input_csv = args.input_csv.expanduser().resolve()
    output = args.output_markdown.expanduser().resolve()
    rows = load_rows(input_csv)
    if output.exists() and not args.overwrite:
        raise FileExistsError("Refusing to overwrite {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(rows), encoding="utf-8")
    print("Validated {} variants from {}".format(len(rows), input_csv))
    print("markdown={}".format(output))


if __name__ == "__main__":
    main(parse_args())
