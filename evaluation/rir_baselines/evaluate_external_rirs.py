#!/usr/bin/env python3
"""Evaluate externally estimated full RIRs with the DARS metric protocol."""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


DARS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = DARS_ROOT / "outputs" / "rir_baselines" / "dars_stems" / "manifest.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--assignment-mode",
        default="fixed_identity",
        help="Assignment already applied when the estimator input manifest was exported.",
    )
    parser.add_argument(
        "--input-description",
        default="DARS patience=10 fixed-identity x_sep reverberant estimates",
    )
    parser.add_argument("--native-model-output", default="time-domain RIR")
    parser.add_argument("--prediction-subdir", default="rir")
    parser.add_argument("--tail-seconds", type=float, default=1.0)
    parser.add_argument("--pre-direct-ms", type=float, default=2.5)
    parser.add_argument("--drr-analysis-pre-ms", type=float, default=5.0)
    parser.add_argument("--direct-half-ms", type=float, default=2.5)
    parser.add_argument(
        "--drr-sensitivity-half-ms", type=float, nargs="+", default=(1.25, 2.5, 5.0)
    )
    parser.add_argument("--rir50-ms", type=float, default=50.0)
    parser.add_argument("--edc-limit-db", type=float, default=-35.0)
    parser.add_argument("--lsd-floor-db", type=float, default=-80.0)
    parser.add_argument("--clarity-min-late-db", type=float, default=-80.0)
    parser.add_argument("--min-decay-r2", type=float, default=0.9)
    parser.add_argument("--drr-tie-tolerance-db", type=float, default=1e-6)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-examples", type=int, default=5)
    args = parser.parse_args()
    if args.max_sources is not None and args.max_sources < 1:
        parser.error("--max-sources must be positive")
    return args


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def read_manifest(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Empty manifest: {}".format(path))
    return rows


def prediction_path(prediction_dir, prediction_subdir, row):
    filename = Path(row["input_path"]).name
    candidates = []
    if prediction_subdir:
        candidates.append(prediction_dir / prediction_subdir / filename)
    candidates.extend((prediction_dir / filename, prediction_dir / "estimate_rir" / filename))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing predicted RIR for {} (checked {})".format(
            filename, ", ".join(str(path) for path in candidates)
        )
    )


def load_mono(path, target_rate):
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        waveform = np.mean(waveform, axis=1, keepdims=True)
    waveform = waveform[:, 0]
    if sample_rate != target_rate:
        divisor = int(np.gcd(sample_rate, target_rate))
        waveform = resample_poly(
            waveform, target_rate // divisor, sample_rate // divisor
        )
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise ValueError("Invalid RIR waveform: {}".format(path))
    return waveform


def load_reference(path, source, target_rate, channel=None):
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if channel is None:
        if waveform.shape[1] != 2:
            raise ValueError("Expected paired two-source RIR at {}".format(path))
        channel = source - 1
    if channel < 0 or channel >= waveform.shape[1]:
        raise ValueError("Reference channel {} unavailable at {}".format(channel, path))
    waveform = waveform[:, channel]
    if sample_rate != target_rate:
        divisor = int(np.gcd(sample_rate, target_rate))
        waveform = resample_poly(
            waveform, target_rate // divisor, sample_rate // divisor
        )
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise ValueError("Invalid reference RIR at {}".format(path))
    return waveform


def write_csv(path, rows, fieldnames):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    sys.path.insert(0, str(DARS_ROOT))
    import evaluate_rir_metrics as rir_eval

    manifest = args.manifest.expanduser().resolve()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        output_dir / "per_source_metrics.csv",
        output_dir / "per_mixture_metrics.csv",
        output_dir / "summary.csv",
        output_dir / "summary.json",
        output_dir / "REPORT.md",
    ]
    if any(path.exists() for path in protected) and not args.overwrite:
        raise FileExistsError("Outputs exist in {}; pass --overwrite".format(output_dir))

    rows = read_manifest(manifest)
    if args.max_sources is not None:
        rows = rows[: args.max_sources]
    sample_rate = 8000
    shape_pre = int(round(args.pre_direct_ms * sample_rate / 1000.0))
    drr_pre = int(round(args.drr_analysis_pre_ms * sample_rate / 1000.0))
    direct_half = int(round(args.direct_half_ms * sample_rate / 1000.0))
    output_samples = shape_pre + int(round(args.tail_seconds * sample_rate))
    drr_output_samples = drr_pre + int(round(args.tail_seconds * sample_rate))
    acoustic_parameters = rir_eval.acoustic_parameter_names(args.drr_sensitivity_half_ms)
    source_rows = []
    mixture_lookup = defaultdict(list)
    example_dir = output_dir / "examples"
    if args.save_examples:
        example_dir.mkdir(parents=True, exist_ok=True)

    for index, manifest_row in enumerate(rows, 1):
        source = int(manifest_row["source"])
        predicted_path = prediction_path(
            prediction_dir, args.prediction_subdir, manifest_row
        )
        predicted = load_mono(predicted_path, sample_rate)
        reference_channel = manifest_row.get("reference_rir_channel", "").strip()
        reference_channel = int(reference_channel) if reference_channel else None
        full = load_reference(
            manifest_row["full_rir_path"], source, sample_rate, reference_channel
        )
        direct_path = manifest_row.get("direct_rir_path", "").strip()
        direct = (
            load_reference(direct_path, source, sample_rate, reference_channel)
            if direct_path
            else full
        )
        predicted_peak = int(np.argmax(np.abs(predicted)))
        direct_peak = int(np.argmax(np.abs(direct)))
        estimated_aligned = rir_eval.crop_around_anchor(
            predicted, predicted_peak, shape_pre, output_samples
        )
        target_aligned = rir_eval.crop_around_anchor(
            full, direct_peak, shape_pre, output_samples
        )
        shape = rir_eval.response_shape_metrics(
            estimated_aligned,
            target_aligned,
            shape_pre,
            sample_rate,
            args.rir50_ms,
            args.edc_limit_db,
            args.lsd_floor_db,
        )
        target_parameters = rir_eval.room_parameters(
            target_aligned,
            sample_rate,
            shape_pre,
            direct_half,
            (),
            args.clarity_min_late_db,
        )
        estimated_parameters = rir_eval.room_parameters(
            estimated_aligned,
            sample_rate,
            shape_pre,
            direct_half,
            (),
            args.clarity_min_late_db,
        )
        drr_target = rir_eval.crop_around_anchor(
            full, direct_peak, drr_pre, drr_output_samples
        )
        drr_estimated = rir_eval.crop_around_anchor(
            predicted, predicted_peak, drr_pre, drr_output_samples
        )
        target_parameters["drr_db"] = rir_eval.direct_to_reverberant_ratio(
            drr_target, drr_pre, direct_half
        )
        estimated_parameters["drr_db"] = rir_eval.direct_to_reverberant_ratio(
            drr_estimated, drr_pre, direct_half
        )
        for half_ms in args.drr_sensitivity_half_ms:
            name = rir_eval.drr_parameter_name(half_ms)
            half_samples = int(round(half_ms * sample_rate / 1000.0))
            target_parameters[name] = rir_eval.direct_to_reverberant_ratio(
                drr_target, drr_pre, half_samples
            )
            estimated_parameters[name] = rir_eval.direct_to_reverberant_ratio(
                drr_estimated, drr_pre, half_samples
            )

        source_row = {
            "utterance": manifest_row["utterance"],
            "dataset_index": int(manifest_row["dataset_index"]),
            "crop_start": int(manifest_row["crop_start"]),
            "source": source,
            "role": manifest_row["role"],
            "distance_m": float(manifest_row["distance_m"]),
            "rir_set": manifest_row.get("rir_set", ""),
            "repeat": manifest_row.get("repeat", ""),
            "level_protocol": manifest_row.get("level_protocol", ""),
            "effective_peak_raw": predicted_peak,
            "effective_peak_scale": float(np.max(np.abs(predicted))),
            "direct_path_peak": direct_peak,
        }
        for name in rir_eval.RECONSTRUCTION_METRICS:
            source_row[name] = float("nan")
        source_row.update(shape)
        for name in acoustic_parameters:
            target_value = target_parameters[name]
            estimate_value = estimated_parameters[name]
            source_row["target_" + name] = target_value
            source_row["est_" + name] = estimate_value
            source_row["error_" + name] = (
                estimate_value - target_value
                if finite(target_value) and finite(estimate_value)
                else float("nan")
            )
        for name in ("edt_r2", "t20_r2", "t30_r2"):
            source_row["target_" + name] = target_parameters[name]
            source_row["est_" + name] = estimated_parameters[name]
        source_rows.append(source_row)
        mixture_lookup[manifest_row["utterance"]].append(source_row)

        if index <= args.save_examples:
            name = "{:04d}_s{}".format(int(manifest_row["dataset_index"]), source)
            sf.write(example_dir / (name + "_estimated.wav"), estimated_aligned, sample_rate, subtype="FLOAT")
            sf.write(example_dir / (name + "_target.wav"), target_aligned, sample_rate, subtype="FLOAT")
        if index % 100 == 0 or index == len(rows):
            print("Evaluated {}/{} source responses".format(index, len(rows)), flush=True)

    mixture_rows = []
    for key in sorted(mixture_lookup, key=lambda value: int(mixture_lookup[value][0]["dataset_index"])):
        pair = sorted(mixture_lookup[key], key=lambda row: int(row["source"]))
        if len(pair) != 2:
            continue
        item = {
            "key": key,
            "dataset_index": pair[0]["dataset_index"],
            "crop_start": pair[0]["crop_start"],
            "s1_distance_m": pair[0]["distance_m"],
            "s2_distance_m": pair[1]["distance_m"],
        }
        mixture_row = rir_eval.build_mixture_row(
            item, pair, args.drr_tie_tolerance_db
        )
        mixture_row["rir_set"] = pair[0].get("rir_set", "")
        mixture_row["repeat"] = pair[0].get("repeat", "")
        mixture_row["level_protocol"] = pair[0].get("level_protocol", "")
        mixture_rows.append(mixture_row)

    metadata = {
        "method": args.method,
        "checkpoint": args.method,
        "input_manifest": str(manifest),
        "prediction_dir": str(prediction_dir),
        "assignment_mode": args.assignment_mode,
        "input_description": args.input_description,
        "native_model_output": args.native_model_output,
        "reference_direct_anchor": (
            "separate direct-path RIR"
            if all(row.get("direct_rir_path", "").strip() for row in rows)
            else "strongest peak of measured full RIR"
        ),
        "evaluated_utterances": len(mixture_rows),
        "evaluated_source_rows": len(source_rows),
        "eligible_utterances": len(mixture_rows),
        "manifest_utterances": len({row["utterance"] for row in read_manifest(manifest)}),
        "sample_rate": sample_rate,
        "analysis_pre_direct_ms": args.pre_direct_ms,
        "drr_analysis_pre_ms": args.drr_analysis_pre_ms,
        "tail_seconds": args.tail_seconds,
        "clarity_min_late_db": args.clarity_min_late_db,
        "drr_tie_tolerance_db": args.drr_tie_tolerance_db,
        "min_decay_r2": args.min_decay_r2,
    }
    summary = rir_eval.build_summary(
        source_rows, mixture_rows, acoustic_parameters, metadata, args.min_decay_r2
    )
    room_names = sorted({row["rir_set"] for row in source_rows if row.get("rir_set")})
    summary["rooms"] = {}
    for room_name in room_names:
        room_sources = [row for row in source_rows if row.get("rir_set") == room_name]
        room_mixtures = [row for row in mixture_rows if row.get("rir_set") == room_name]
        room_summary = rir_eval.build_summary(
            room_sources,
            room_mixtures,
            acoustic_parameters,
            metadata,
            args.min_decay_r2,
        )
        summary["rooms"][room_name] = {
            "n_mixtures": len(room_mixtures),
            "n_responses": len(room_sources),
            "response_shape": room_summary["groups"]["overall"]["response_shape"],
            "acoustic_parameters": room_summary["groups"]["overall"]["acoustic_parameters"],
            "mixture_metrics": room_summary["mixture_metrics"],
            "drr_gap": room_summary["drr_gap"],
        }
    source_fields = rir_eval.source_fieldnames(acoustic_parameters)
    source_fields[6:6] = ["rir_set", "repeat", "level_protocol"]
    mixture_fields = rir_eval.mixture_fieldnames()
    mixture_fields[3:3] = ["rir_set", "repeat", "level_protocol"]
    write_csv(
        output_dir / "per_source_metrics.csv",
        source_rows,
        source_fields,
    )
    write_csv(
        output_dir / "per_mixture_metrics.csv",
        mixture_rows,
        mixture_fields,
    )
    rir_eval.write_summary_csv(summary, output_dir / "summary.csv")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, allow_nan=False)

    overall = summary["groups"]["overall"]
    shape = overall["response_shape"]
    acoustic = overall["acoustic_parameters"]
    mix = summary["mixture_metrics"]
    report = [
        "# External RIR evaluation: {}".format(args.method),
        "",
        "- Input: {}.".format(args.input_description),
        "- Alignment anchor: {}.".format(metadata["reference_direct_anchor"]),
        "- Evaluated: {} mixtures / {} source responses.".format(len(mixture_rows), len(source_rows)),
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| RIR-50 RMSE | {:.6f} |".format(shape["rir50_rmse"]["mean"]),
        "| EDC RMSE (dB) | {:.6f} |".format(shape["edc_rmse_db"]["mean"]),
        "| SI-NMSE (dB) | {:.6f} |".format(shape["si_nmse_db"]["mean"]),
        "| DRR MAE (dB) | {:.6f} |".format(acoustic["drr_db"]["mae"]),
        "| DRR Pearson | {:.6f} |".format(acoustic["drr_db"]["pearson_r"] or float("nan")),
        "| C50 MAE (dB) | {:.6f} |".format(acoustic["c50_db"]["mae"]),
        "| C80 MAE (dB) | {:.6f} |".format(acoustic["c80_db"]["mae"]),
        "| Near/far accuracy | {:.4%} |".format(mix["near_far_correct"]["mean"]),
    ]
    if summary["rooms"]:
        report.extend(
            [
                "",
                "## Per-room headline metrics",
                "",
                "| Room | N mix/src | RIR-50 | EDC RMSE | SI-NMSE | DRR MAE | DRR r / rho | C50 MAE | C80 MAE | DRR Near/Far |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for room_name, room in summary["rooms"].items():
            room_shape = room["response_shape"]
            room_acoustic = room["acoustic_parameters"]
            room_mix = room["mixture_metrics"]
            report.append(
                "| {} | {}/{} | {:.4f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} / {:.3f} | {:.3f} | {:.3f} | {:.2%} |".format(
                    room_name,
                    room["n_mixtures"],
                    room["n_responses"],
                    room_shape["rir50_rmse"]["mean"],
                    room_shape["edc_rmse_db"]["mean"],
                    room_shape["si_nmse_db"]["mean"],
                    room_acoustic["drr_db"]["mae"],
                    room_acoustic["drr_db"]["pearson_r"],
                    room_acoustic["drr_db"]["spearman_r"],
                    room_acoustic["c50_db"]["mae"],
                    room_acoustic["c80_db"]["mae"],
                    room_mix["near_far_correct"]["mean"],
                )
            )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Wrote unified metrics to {}".format(output_dir))


if __name__ == "__main__":
    main()
