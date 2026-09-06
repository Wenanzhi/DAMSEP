#!/usr/bin/env python3
"""Build a joint-model-versus-blind-RIR-estimator comparison table."""

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DARS_SUMMARY = REPO_ROOT / "outputs" / "rir_metrics" / "summary.json"
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "rir_baselines" / "results"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reference-summary", type=Path, default=DARS_SUMMARY)
    parser.add_argument("--reference-name", default="DARS (ES patience 10)")
    parser.add_argument("--title", default="RIR baseline comparison")
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Build a table containing only the five released RIR estimators.",
    )
    parser.add_argument(
        "--input-description",
        default="fixed-identity DARS patience=10 `x_sep` reverberant outputs",
    )
    parser.add_argument(
        "--comparison-note",
        default=(
            "External models receive DARS-separated reverberant stems, while DARS "
            "estimates its response jointly; this is the intended cascaded baseline comparison."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "rir_baseline_comparison.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "outputs" / "rir_baseline_comparison.md",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract(name, path):
    summary = load_json(path)
    overall = summary["groups"]["overall"]
    shape = overall["response_shape"]
    acoustic = overall["acoustic_parameters"]
    mix = summary["mixture_metrics"]
    metadata = summary["metadata"]
    return {
        "model": name,
        "n_mixtures": metadata["evaluated_utterances"],
        "n_responses": metadata["evaluated_source_rows"],
        "rir50_rmse": shape["rir50_rmse"]["mean"],
        "edc_rmse_db": shape["edc_rmse_db"]["mean"],
        "si_nmse_db": shape["si_nmse_db"]["mean"],
        "waveform_corr": shape["waveform_corr"]["mean"],
        "lsd_db": shape["lsd_db"]["mean"],
        "t20_mae_s": acoustic["t20_s"]["mae"],
        "t20_valid_rate": acoustic["t20_s"]["valid_rate"],
        "drr_mae_db": acoustic["drr_db"]["mae"],
        "drr_pearson": acoustic["drr_db"]["pearson_r"],
        "drr_spearman": acoustic["drr_db"]["spearman_r"],
        "c50_mae_db": acoustic["c50_db"]["mae"],
        "c80_mae_db": acoustic["c80_db"]["mae"],
        "near_far_accuracy": mix["near_far_correct"]["mean"],
        "drr_rank_fidelity": mix["drr_rank_fidelity"]["mean"],
        "drr_gap_mae_db": summary["drr_gap"]["mae"],
    }


def display(value, digits=3, percent=False):
    if value is None:
        return "--"
    if percent:
        return "{:.2f}".format(100.0 * float(value))
    return ("{:.%df}" % digits).format(float(value))


def main():
    args = parse_args()
    paths = [
        ("Rec-RIR", args.results_dir / "recrir" / "summary.json"),
        ("VINP (TCN+SA+S, epoch 120)", args.results_dir / "vinp_tcnsas_epoch120" / "summary.json"),
        ("BUDDy", args.results_dir / "buddy" / "summary.json"),
        ("Speech2RIR", args.results_dir / "speech2rir" / "summary.json"),
        ("FiNS (local epoch 250)", args.results_dir / "fins_local_epoch250" / "summary.json"),
    ]
    if not args.external_only:
        paths.insert(0, (args.reference_name, args.reference_summary))
    missing = [path for _, path in paths if not path.exists()]
    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing summaries: {}".format(", ".join(str(path) for path in missing)))
    rows = [extract(name, path) for name, path in paths if path.exists()]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# {}".format(args.title),
        "",
        "All external estimators take the {}. All rows use the same 8 kHz reference RIRs, direct-arrival alignment, 1 s analysis tail, and full untrimmed test set.".format(args.input_description),
        "",
        "| Model | N mix/src | RIR-50 RMSE | EDC RMSE | SI-NMSE | T20 MAE / valid | DRR MAE | DRR r / rho | C50 MAE | C80 MAE | Near/Far | DRR Rank | Gap MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {n_mixtures}/{n_responses} | {rir50} | {edc} | {sinmse} | {t20} / {t20v}% | {drr} | {pr} / {sr} | {c50} | {c80} | {nf}% | {rank}% | {gap} |".format(
                model=row["model"],
                n_mixtures=row["n_mixtures"],
                n_responses=row["n_responses"],
                rir50=display(row["rir50_rmse"], 4),
                edc=display(row["edc_rmse_db"]),
                sinmse=display(row["si_nmse_db"]),
                t20=display(row["t20_mae_s"]),
                t20v=display(row["t20_valid_rate"], 2, True),
                drr=display(row["drr_mae_db"]),
                pr=display(row["drr_pearson"]),
                sr=display(row["drr_spearman"]),
                c50=display(row["c50_mae_db"]),
                c80=display(row["c80_mae_db"]),
                nf=display(row["near_far_accuracy"], 2, True),
                rank=display(row["drr_rank_fidelity"], 2, True),
                gap=display(row["drr_gap_mae_db"]),
            )
        )
    lines.extend(
        [
            "",
            "- Lower is better for RMSE/MAE/LSD; lower (more negative) is better for SI-NMSE.",
            "- Higher is better for correlation and ranking accuracy.",
            "- {}".format(args.comparison_note),
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote {} and {}".format(args.output_csv, args.output_md))


if __name__ == "__main__":
    main()
