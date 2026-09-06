#!/usr/bin/env python3
"""Build paper-facing measured-RIR comparison tables from unified summaries."""

import argparse
import csv
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "measured_rir" / "results"
BASELINES = (
    ("Rec-RIR", "recrir"),
    ("VINP (TCN+SA+S, epoch 120)", "vinp_tcnsas_epoch120"),
    ("BUDDy", "buddy_full24"),
    ("Speech2RIR", "speech2rir"),
    ("FiNS (local epoch 250)", "fins_local_epoch250"),
)
METHOD_SETS = {
    "dars-fixed": (("DARS (ES patience 10)", "dars"),) + BASELINES,
    "oracle-ground-truth": BASELINES,
}
EXPECTED_COUNTS = {
    "0716": (390, 780),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--method-set",
        choices=tuple(METHOD_SETS),
        default="dars-fixed",
    )
    parser.add_argument("--expected-assignment", default=None)
    parser.add_argument("--input-description", default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "measured_rir_baseline_comparison.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "outputs" / "measured_rir_baseline_comparison.md",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_panel(summary, panel):
    if panel == "overall":
        shape = summary["groups"]["overall"]["response_shape"]
        acoustic = summary["groups"]["overall"]["acoustic_parameters"]
        mixture = summary["mixture_metrics"]
        gap = summary["drr_gap"]
        n_mixtures = summary["metadata"]["evaluated_utterances"]
        n_responses = summary["metadata"]["evaluated_source_rows"]
    else:
        room = summary["rooms"][panel]
        shape = room["response_shape"]
        acoustic = room["acoustic_parameters"]
        mixture = room["mixture_metrics"]
        gap = room["drr_gap"]
        n_mixtures = room["n_mixtures"]
        n_responses = room["n_responses"]
    return {
        "n_mixtures": n_mixtures,
        "n_responses": n_responses,
        "rir50_rmse": shape["rir50_rmse"]["mean"],
        "edc_rmse_db": shape["edc_rmse_db"]["mean"],
        "si_nmse_db": shape["si_nmse_db"]["mean"],
        "rir_waveform_corr": shape["waveform_corr"]["mean"],
        "lsd_db": shape["lsd_db"]["mean"],
        "t20_mae_s": acoustic["t20_s"]["mae"],
        "t20_valid_rate": acoustic["t20_s"]["valid_rate"],
        "drr_mae_db": acoustic["drr_db"]["mae"],
        "drr_pearson": acoustic["drr_db"]["pearson_r"],
        "drr_spearman": acoustic["drr_db"]["spearman_r"],
        "c50_mae_db": acoustic["c50_db"]["mae"],
        "c80_mae_db": acoustic["c80_db"]["mae"],
        "near_far_accuracy": mixture["near_far_correct"]["mean"],
        "drr_rank_fidelity": mixture["drr_rank_fidelity"]["mean"],
        "drr_gap_mae_db": gap["mae"],
    }


def validate_summary(name, summary, expected_assignment):
    metadata = summary.get("metadata", {})
    if metadata.get("assignment_mode") != expected_assignment:
        raise ValueError(
            "{} assignment mismatch: expected {}, got {}".format(
                name, expected_assignment, metadata.get("assignment_mode")
            )
        )
    if set(summary.get("rooms", {})) != {"0716"}:
        raise ValueError("{} does not contain exactly the 0716 room".format(name))
    for panel, expected in EXPECTED_COUNTS.items():
        row = extract_panel(summary, panel)
        actual = (row["n_mixtures"], row["n_responses"])
        if actual != expected:
            raise ValueError(
                "{} {} count mismatch: expected {}, got {}".format(
                    name, panel, expected, actual
                )
            )
        for key, value in row.items():
            if key in ("n_mixtures", "n_responses"):
                continue
            if (
                key == "t20_mae_s"
                and value is None
                and float(row["t20_valid_rate"]) == 0.0
            ):
                continue
            if value is None or not math.isfinite(float(value)):
                raise ValueError("{} {} has invalid {}={}".format(name, panel, key, value))


def fmt(value, digits=3, percent=False):
    if value is None:
        return "--"
    if percent:
        return ("{:.%df}" % digits).format(100.0 * float(value))
    return ("{:.%df}" % digits).format(float(value))


def main():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    methods = METHOD_SETS[args.method_set]
    expected_assignment = args.expected_assignment or "fixed_identity"
    input_description = args.input_description or (
        "ground-truth reverberant source images rendered with paired measured RIRs"
        if args.method_set == "oracle-ground-truth"
        else "fixed-identity DARS reverberant stems"
    )
    summaries = []
    input_manifests = set()
    for name, directory in methods:
        path = results_dir / directory / "summary.json"
        if not path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(path)
        summary = load(path)
        validate_summary(name, summary, expected_assignment)
        input_manifests.add(str(Path(summary["metadata"]["input_manifest"]).resolve()))
        summaries.append((name, summary))
    if len(input_manifests) != 1:
        raise ValueError("Methods do not share one input manifest: {}".format(input_manifests))
    rows = []
    for panel in ("0716",):
        for name, summary in summaries:
            row = {"panel": panel, "model": name}
            row.update(extract_panel(summary, panel))
            rows.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    title = {
        "oracle-ground-truth": "Oracle-input measured-RIR estimator comparison",
    }.get(args.method_set, "Measured-RIR baseline comparison")
    lines = [
        "# {}".format(title),
        "",
        "All external estimators receive the same {}. Predictions are peak-aligned to channel 0 of the measured full RIR; no additional RIR PIT is used. Near/far is selected only by comparing predicted DRR.".format(input_description),
        "",
    ]
    for panel in ("0716",):
        lines.extend(
            [
                "## {}".format(panel),
                "",
                "| Model | N mix/src | RIR-50 | EDC | SI-NMSE | RIR corr | LSD | T20 MAE / valid | DRR MAE | DRR r / rho | C50 | C80 | Near/Far | DRR Rank | Gap MAE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in (item for item in rows if item["panel"] == panel):
            lines.append(
                "| {model} | {n_mixtures}/{n_responses} | {rir50} | {edc} | {sinmse} | {corr} | {lsd} | {t20} / {t20v}% | {drr} | {pr} / {sr} | {c50} | {c80} | {nf}% | {rank}% | {gap} |".format(
                    model=row["model"],
                    n_mixtures=row["n_mixtures"],
                    n_responses=row["n_responses"],
                    rir50=fmt(row["rir50_rmse"], 4),
                    edc=fmt(row["edc_rmse_db"]),
                    sinmse=fmt(row["si_nmse_db"]),
                    corr=fmt(row["rir_waveform_corr"]),
                    lsd=fmt(row["lsd_db"]),
                    t20=fmt(row["t20_mae_s"]),
                    t20v=fmt(row["t20_valid_rate"], 2, True),
                    drr=fmt(row["drr_mae_db"]),
                    pr=fmt(row["drr_pearson"]),
                    sr=fmt(row["drr_spearman"]),
                    c50=fmt(row["c50_mae_db"]),
                    c80=fmt(row["c80_mae_db"]),
                    nf=fmt(row["near_far_accuracy"], 2, True),
                    rank=fmt(row["drr_rank_fidelity"], 2, True),
                    gap=fmt(row["drr_gap_mae_db"]),
                )
            )
        lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote {} and {}".format(args.output_csv, args.output_md))


if __name__ == "__main__":
    main()
