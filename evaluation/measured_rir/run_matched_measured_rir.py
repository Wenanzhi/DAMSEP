#!/usr/bin/env python3
"""Build and evaluate deterministic measured-RIR test mixtures for DARS.

The generated mixtures reuse eligible 4-second source pairs and level statistics
from the ordered HETMIXR test set.  Each measured RIR is spatially normalized in
the same spirit as the HETMIXR generator: convolution changes the room response,
while the source's pre-convolution RMS is preserved.  Distance therefore does
not determine source loudness.
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy import signal, stats


DARS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = DARS_ROOT / "outputs" / "measured_rir" / "matched_0716"
DEFAULT_EXP = DARS_ROOT / "checkpoints" / "dars_mixed_p10"
DEFAULT_MIXED_TEST = DARS_ROOT / "data" / "hetmixr" / "wav8k" / "min" / "tt"
RIR_SETS = {
    "0716": "rir_IOT-meidi12.3-0716",
}
EXPECTED_RIR_COUNTS = {"0716": 13}
SAMPLE_RATE = 8000
SEGMENT_SAMPLES = 4 * SAMPLE_RATE
EPS = np.finfo(np.float64).eps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "evaluate", "all"), default="all")
    parser.add_argument(
        "--rir-root",
        type=Path,
        required=True,
        help="Directory containing the 0716 measured-RIR folder.",
    )
    parser.add_argument("--mixed-test-dir", type=Path, default=DEFAULT_MIXED_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--rir-channel", type=int, default=0)
    parser.add_argument(
        "--level-protocol",
        choices=("as_recorded", "balanced"),
        default="balanced",
        help=(
            "as_recorded preserves the source-level direction in ordered HETMIXR; "
            "balanced preserves its absolute level gap but balances which distance is louder"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_size < 1:
        parser.error("--repeats and --batch-size must be positive")
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    return args


def resolve_storage_path(value):
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    parsed = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            raise ValueError("Malformed manifest row in {}".format(path))
        parsed.append((resolve_storage_path(row[0]), int(row[1])))
    return parsed


def load_source_pool(mixed_test_dir):
    manifests = {
        name: load_json_manifest(mixed_test_dir / (name + ".json"))
        for name in ("mix_both_reverb", "s1_anechoic", "s2_anechoic")
    }
    lengths = {len(rows) for rows in manifests.values()}
    if len(lengths) != 1:
        raise ValueError("Ordered HETMIXR manifests have different row counts")
    pool = []
    for index, rows in enumerate(zip(*manifests.values())):
        mix_row, s1_row, s2_row = rows
        basenames = {item[0].name for item in rows}
        row_lengths = {item[1] for item in rows}
        if len(basenames) != 1 or len(row_lengths) != 1:
            raise ValueError("Manifest mismatch at index {}".format(index))
        length = row_lengths.pop()
        if length < SEGMENT_SAMPLES:
            continue
        pool.append(
            {
                "manifest_index": index,
                "key": basenames.pop(),
                "length": length,
                "mix_path": mix_row[0],
                "s1_path": s1_row[0],
                "s2_path": s2_row[0],
            }
        )
    if not pool:
        raise ValueError("No eligible source pairs")
    return pool


def parse_rir_distance(path, rir_set):
    prefix = RIR_SETS[rir_set] + "_"
    name = path.stem
    if not name.startswith(prefix):
        raise ValueError("Unexpected RIR basename: {}".format(path.name))
    return round(float(name[len(prefix) :].split("_")[0]), 1)


def load_rir_paths(rir_root):
    result = {}
    for rir_set, directory in RIR_SETS.items():
        paths = sorted((rir_root / directory).glob("*.wav"))
        by_distance = {parse_rir_distance(path, rir_set): path.resolve() for path in paths}
        if len(by_distance) != len(paths):
            raise ValueError("Duplicate distance in {}".format(directory))
        expected = EXPECTED_RIR_COUNTS[rir_set]
        if len(paths) != expected:
            raise ValueError("Expected {} RIRs in {}, got {}".format(expected, directory, len(paths)))
        result[rir_set] = by_distance
    return result


def read_segment(path, start, samples=SEGMENT_SAMPLES):
    audio, sample_rate = sf.read(path, start=start, stop=start + samples, dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Expected 8 kHz source {}, got {}".format(path, sample_rate))
    if audio.ndim != 1 or audio.shape[0] != samples:
        raise ValueError("Unexpected source shape {} for {}".format(audio.shape, path))
    if not np.all(np.isfinite(audio)):
        raise ValueError("Non-finite source {}".format(path))
    return audio.astype(np.float64)


def read_measured_rir(path, channel):
    rir, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if channel < 0 or channel >= rir.shape[1]:
        raise ValueError("RIR channel {} unavailable in {}".format(channel, path))
    rir = rir[:, channel].astype(np.float64)
    if sample_rate != SAMPLE_RATE:
        rir = signal.resample_poly(rir, SAMPLE_RATE, sample_rate)
    if not np.all(np.isfinite(rir)) or np.sum(rir * rir) <= EPS:
        raise ValueError("Invalid RIR {}".format(path))
    return rir


def rms(audio):
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))


def convolve_and_spatially_normalize(source, rir):
    reverberant = signal.fftconvolve(source, rir, mode="full")[: source.shape[0]]
    source_energy = float(np.sum(source * source))
    reverb_energy = float(np.sum(reverberant * reverberant))
    if source_energy <= EPS or reverb_energy <= EPS:
        raise ValueError("Zero-energy source or convolved response")
    return reverberant * math.sqrt(source_energy / reverb_energy)


def apply_level_protocol(s1, s2, near, far, repeat, protocol):
    if protocol == "as_recorded":
        return s1, s2
    rms1 = rms(s1)
    rms2 = rms(s2)
    if rms1 <= EPS or rms2 <= EPS:
        raise ValueError("Cannot balance silent sources")
    absolute_gap_db = abs(20.0 * math.log10(rms1 / rms2))
    # The rule is independent of room ID and deterministic for each pair/repeat.
    near_is_louder = (int(round(near * 10)) + int(round(far * 10)) + repeat) % 2 == 0
    signed_gap_db = absolute_gap_db if near_is_louder else -absolute_gap_db
    geometric_rms = math.sqrt(rms1 * rms2)
    target_rms1 = geometric_rms * 10.0 ** (signed_gap_db / 40.0)
    target_rms2 = geometric_rms * 10.0 ** (-signed_gap_db / 40.0)
    return s1 * (target_rms1 / rms1), s2 * (target_rms2 / rms2)


def rir_drr_db(rir, half_samples=int(round(0.0025 * SAMPLE_RATE))):
    """Measure DRR around the strongest direct-path candidate."""
    rir = np.asarray(rir, dtype=np.float64).reshape(-1)
    direct_index = int(np.argmax(np.abs(rir)))
    start = max(direct_index - half_samples, 0)
    stop = min(direct_index + half_samples, rir.size)
    direct_energy = float(np.sum(np.square(rir[start:stop]), dtype=np.float64))
    reverberant_energy = float(
        np.sum(np.square(rir[:start]), dtype=np.float64)
        + np.sum(np.square(rir[stop:]), dtype=np.float64)
    )
    if direct_energy <= EPS or reverberant_energy <= EPS:
        return float("nan")
    return 10.0 * math.log10(direct_energy / reverberant_energy)


def subset_flags(near, far):
    gap = far - near
    ratio = far / near
    train_near_min, train_near_max = 0.9014809446, 1.4144062973
    train_far_min, train_far_max = 2.3372589340, 5.7242063691
    train_gap_min, train_ratio_min = 1.3420365681, 2.1810395534
    return {
        "shared_grid": int(near >= 0.8 and far <= 2.0),
        "gap_ge_0p5": int(gap >= 0.5 - 1e-8),
        "gap_ge_0p8": int(gap >= 0.8 - 1e-8),
        "ratio_in_train_support": int(ratio >= train_ratio_min),
        "strict_train_geometry_overlap": int(
            near >= train_near_min
            and near <= train_near_max
            and far >= train_far_min
            and far <= train_far_max
            and gap >= train_gap_min
            and ratio >= train_ratio_min
        ),
    }


def prepare_dataset(args):
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        config_path = output_dir / "run_config.json"
        if not config_path.is_file():
            raise ValueError("Prepared manifest has no run_config.json: {}".format(output_dir))
        with open(config_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        expected = {
            "seed": args.seed,
            "repeats": args.repeats,
            "rir_channel": args.rir_channel,
            "level_protocol": args.level_protocol,
            "rir_root": str(args.rir_root.resolve()),
            "mixed_test_dir": str(args.mixed_test_dir.resolve()),
        }
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError("Prepared dataset configuration mismatch: {}".format(mismatches))
        print("Prepared manifest already exists: {}".format(manifest_path))
        return manifest_path
    if args.overwrite and output_dir.exists():
        for relative in ("manifest.csv", "dataset_summary.json", "run_config.json"):
            (output_dir / relative).unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rir_set in RIR_SETS:
        for directory in ("mix", "targets_reverb", "targets_clean"):
            (output_dir / rir_set / directory).mkdir(parents=True, exist_ok=True)

    source_pool = load_source_pool(args.mixed_test_dir.resolve())
    rir_paths = load_rir_paths(args.rir_root.resolve())
    reference_grid = next(iter(rir_paths.values()))
    union_pairs = list(itertools.combinations(sorted(reference_grid), 2))
    required_sources = len(union_pairs) * args.repeats
    if required_sources > len(source_pool):
        raise ValueError(
            "Need {} unique source pairs, only {} eligible".format(required_sources, len(source_pool))
        )
    random_generator = random.Random(args.seed)
    selected_pool = list(source_pool)
    random_generator.shuffle(selected_pool)
    content_for_pair = {}
    cursor = 0
    for repeat in range(args.repeats):
        for near, far in union_pairs:
            content_for_pair[(near, far, repeat)] = selected_pool[cursor]
            cursor += 1

    rir_cache = {}
    fieldnames = [
        "example_id",
        "rir_set",
        "repeat",
        "level_protocol",
        "near_distance_m",
        "far_distance_m",
        "distance_gap_m",
        "distance_ratio",
        "manifest_index",
        "source_key",
        "crop_start",
        "near_rir_path",
        "far_rir_path",
        "mix_path",
        "targets_reverb_path",
        "targets_clean_path",
        "source_level_difference_db",
        "measured_rir_drr_near_db",
        "measured_rir_drr_far_db",
        "measured_rir_drr_gap_db",
        "measured_rir_drr_order_correct",
        "source_mix_rms",
        "generated_mix_rms_before_match",
        "global_level_scale",
        "peak_guard_scale",
        "shared_grid",
        "gap_ge_0p5",
        "gap_ge_0p8",
        "ratio_in_train_support",
        "strict_train_geometry_overlap",
    ]
    rows = []
    for rir_set, by_distance in rir_paths.items():
        for repeat in range(args.repeats):
            for near, far in itertools.combinations(sorted(by_distance), 2):
                item = content_for_pair[(near, far, repeat)]
                crop_rng = random.Random("{}:{}".format(args.seed, item["manifest_index"]))
                crop_start = crop_rng.randrange(item["length"] - SEGMENT_SAMPLES + 1)
                s1 = read_segment(item["s1_path"], crop_start)
                s2 = read_segment(item["s2_path"], crop_start)
                s1, s2 = apply_level_protocol(
                    s1, s2, near, far, repeat, args.level_protocol
                )
                original_mix = read_segment(item["mix_path"], crop_start)
                near_key = (rir_set, near)
                far_key = (rir_set, far)
                if near_key not in rir_cache:
                    rir_cache[near_key] = read_measured_rir(by_distance[near], args.rir_channel)
                if far_key not in rir_cache:
                    rir_cache[far_key] = read_measured_rir(by_distance[far], args.rir_channel)
                measured_drr_near = rir_drr_db(rir_cache[near_key])
                measured_drr_far = rir_drr_db(rir_cache[far_key])
                near_reverb = convolve_and_spatially_normalize(s1, rir_cache[near_key])
                far_reverb = convolve_and_spatially_normalize(s2, rir_cache[far_key])
                mixture = near_reverb + far_reverb
                before_rms = rms(mixture)
                target_rms = rms(original_mix)
                if before_rms <= EPS or target_rms <= EPS:
                    raise ValueError("Silent mixture for {}".format(item["key"]))
                global_scale = target_rms / before_rms
                near_reverb *= global_scale
                far_reverb *= global_scale
                s1 *= global_scale
                s2 *= global_scale
                mixture = near_reverb + far_reverb
                peak = float(np.max(np.abs(mixture)))
                peak_guard = min(1.0, 0.99 / peak) if peak > 0.0 else 1.0
                mixture *= peak_guard
                near_reverb *= peak_guard
                far_reverb *= peak_guard
                s1 *= peak_guard
                s2 *= peak_guard

                example_id = "{}-n{:03d}-f{:03d}-r{:02d}".format(
                    rir_set, int(round(near * 10)), int(round(far * 10)), repeat
                )
                mix_path = output_dir / rir_set / "mix" / (example_id + ".wav")
                reverb_path = output_dir / rir_set / "targets_reverb" / (example_id + ".wav")
                clean_path = output_dir / rir_set / "targets_clean" / (example_id + ".wav")
                sf.write(mix_path, mixture.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
                sf.write(
                    reverb_path,
                    np.stack((near_reverb, far_reverb), axis=1).astype(np.float32),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )
                sf.write(
                    clean_path,
                    np.stack((s1, s2), axis=1).astype(np.float32),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )
                source_level_difference_db = 20.0 * math.log10(max(rms(s1), EPS) / max(rms(s2), EPS))
                row = {
                    "example_id": example_id,
                    "rir_set": rir_set,
                    "repeat": repeat,
                    "level_protocol": args.level_protocol,
                    "near_distance_m": "{:.1f}".format(near),
                    "far_distance_m": "{:.1f}".format(far),
                    "distance_gap_m": "{:.1f}".format(far - near),
                    "distance_ratio": "{:.8f}".format(far / near),
                    "manifest_index": item["manifest_index"],
                    "source_key": item["key"],
                    "crop_start": crop_start,
                    "near_rir_path": str(by_distance[near]),
                    "far_rir_path": str(by_distance[far]),
                    "mix_path": str(mix_path),
                    "targets_reverb_path": str(reverb_path),
                    "targets_clean_path": str(clean_path),
                    "source_level_difference_db": "{:.8f}".format(source_level_difference_db),
                    "measured_rir_drr_near_db": "{:.8f}".format(measured_drr_near),
                    "measured_rir_drr_far_db": "{:.8f}".format(measured_drr_far),
                    "measured_rir_drr_gap_db": "{:.8f}".format(
                        measured_drr_near - measured_drr_far
                    ),
                    "measured_rir_drr_order_correct": int(measured_drr_near > measured_drr_far),
                    "source_mix_rms": "{:.10f}".format(target_rms),
                    "generated_mix_rms_before_match": "{:.10f}".format(before_rms),
                    "global_level_scale": "{:.10f}".format(global_scale),
                    "peak_guard_scale": "{:.10f}".format(peak_guard),
                }
                row.update(subset_flags(near, far))
                rows.append(row)
                if len(rows) % 100 == 0:
                    total_expected = sum(
                        math.comb(len(by_distance), 2) for by_distance in rir_paths.values()
                    ) * args.repeats
                    print("Prepared {}/{} examples".format(len(rows), total_expected))

    temporary = manifest_path.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest_path)
    summary = {
        "sample_rate": SAMPLE_RATE,
        "segment_samples": SEGMENT_SAMPLES,
        "segment_seconds": SEGMENT_SAMPLES / SAMPLE_RATE,
        "seed": args.seed,
        "repeats": args.repeats,
        "rir_channel": args.rir_channel,
        "level_protocol": args.level_protocol,
        "eligible_source_pairs": len(source_pool),
        "selected_unique_source_pairs": required_sources,
        "total_examples": len(rows),
        "by_rir_set": {
            rir_set: sum(row["rir_set"] == rir_set for row in rows) for rir_set in RIR_SETS
        },
        "subset_counts": {
            rir_set: {
                name: sum(row["rir_set"] == rir_set and int(row[name]) == 1 for row in rows)
                for name in (
                    "shared_grid",
                    "gap_ge_0p5",
                    "gap_ge_0p8",
                    "ratio_in_train_support",
                    "strict_train_geometry_overlap",
                )
            }
            for rir_set in RIR_SETS
        },
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    config = vars(args).copy()
    config.update(
        {
            "rir_root": str(args.rir_root.resolve()),
            "mixed_test_dir": str(args.mixed_test_dir.resolve()),
            "output_dir": str(output_dir),
            "manifest_sha256": sha256(manifest_path),
            "construction": (
                "ordered HETMIXR 4-second source pairs; measured-RIR convolution; "
                "per-source RMS-preserving spatial scaling; original mixture RMS matching; "
                "level protocol={}".format(args.level_protocol)
            ),
        }
    )
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    print("Prepared {} examples at {}".format(len(rows), output_dir))
    return manifest_path


def si_sdr(estimate, target):
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_energy = float(np.dot(target, target))
    if target_energy <= EPS:
        return float("nan")
    projection = float(np.dot(estimate, target)) / target_energy * target
    noise = estimate - projection
    return 10.0 * math.log10(max(float(np.dot(projection, projection)), EPS) / max(float(np.dot(noise, noise)), EPS))


def assignment_metrics(estimate, target, mixture):
    matrix = np.array(
        [[si_sdr(estimate[i], target[j]) for j in range(2)] for i in range(2)],
        dtype=np.float64,
    )
    fixed = float((matrix[0, 0] + matrix[1, 1]) / 2.0)
    swapped = float((matrix[0, 1] + matrix[1, 0]) / 2.0)
    baseline = float(np.mean([si_sdr(mixture, target[index]) for index in range(2)]))
    return {
        "fixed_si_sdr_db": fixed,
        "oracle_si_sdr_db": max(fixed, swapped),
        "fixed_si_sdri_db": fixed - baseline,
        "oracle_si_sdri_db": max(fixed, swapped) - baseline,
        "fixed_order_correct": int(fixed > swapped),
        "assignment_margin_db": fixed - swapped,
    }


def load_evaluator_helpers():
    module_path = DARS_ROOT / "evaluate_rir_metrics.py"
    spec = importlib.util.spec_from_file_location("dars_rir_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dars_model(exp_dir, device):
    if str(DARS_ROOT) not in sys.path:
        sys.path.insert(0, str(DARS_ROOT))
    import look2hear.models

    with open(exp_dir / "conf.yml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_class = getattr(look2hear.models, config["audionet"]["audionet_name"])
    model = model_class.from_pretrain(
        str(exp_dir / "best.pth"),
        sample_rate=SAMPLE_RATE,
        **config["audionet"]["audionet_config"],
    )
    model.to(device)
    model.eval()
    return model


def read_generated(path, channels):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape != (SEGMENT_SAMPLES, channels):
        raise ValueError("Unexpected generated audio {} for {}".format(audio.shape, path))
    return audio.T.astype(np.float64)


def read_completed(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, "r", encoding="utf-8") as handle:
        return {row["example_id"] for row in csv.DictReader(handle)}


def evaluation_signature(args, manifest_path):
    checkpoint = args.exp_dir.resolve() / "best.pth"
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "rir_decoder": "inference_rir sweep deconvolution",
        "response_drr_half_window_ms": 2.5,
        "response_assignment": "fixed identity",
    }


def evaluate_dataset(args, manifest_path):
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.output_dir.resolve()
    result_path = output_dir / "per_example_metrics.csv"
    summary_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    eval_config_path = output_dir / "evaluation_config.json"
    if result_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError("Result exists; use --resume or --overwrite: {}".format(result_path))
    if args.overwrite:
        result_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        summary_csv_path.unlink(missing_ok=True)
        eval_config_path.unlink(missing_ok=True)
    signature = evaluation_signature(args, manifest_path)
    if args.resume and eval_config_path.exists():
        with open(eval_config_path, "r", encoding="utf-8") as handle:
            if json.load(handle) != signature:
                raise ValueError("Resume configuration does not match existing evaluation")
    elif result_path.exists() and args.resume:
        raise ValueError("Cannot safely resume without evaluation_config.json")
    with open(eval_config_path, "w", encoding="utf-8") as handle:
        json.dump(signature, handle, indent=2, sort_keys=True)
    completed = read_completed(result_path) if args.resume else set()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    pending = [row for row in rows if row["example_id"] not in completed]
    if not pending and result_path.exists():
        print("No pending examples; rebuilding summary")
        return summarize_results(output_dir, manifest_path, signature)
    if pending:
        summary_path.unlink(missing_ok=True)
        summary_csv_path.unlink(missing_ok=True)
        (output_dir / "accuracy_heatmaps.png").unlink(missing_ok=True)

    device = torch.device(args.device)
    model = load_dars_model(args.exp_dir.resolve(), device)
    helpers = load_evaluator_helpers()
    decoder = helpers.SweepRIRDecoder(SAMPLE_RATE, device)
    fieldnames = list(rows[0].keys()) + [
        "xsep_fixed_si_sdr_db",
        "xsep_oracle_si_sdr_db",
        "xsep_fixed_si_sdri_db",
        "xsep_oracle_si_sdri_db",
        "xsep_fixed_order_correct",
        "xsep_assignment_margin_db",
        "xderev_fixed_si_sdr_db",
        "xderev_oracle_si_sdr_db",
        "xderev_fixed_si_sdri_db",
        "xderev_oracle_si_sdri_db",
        "xderev_fixed_order_correct",
        "xderev_assignment_margin_db",
        "response_drr_near_db",
        "response_drr_far_db",
        "response_drr_gap_db",
        "response_drr_order_correct",
        "waveform_response_order_agree",
        "response_measured_drr_order_agree",
    ]
    write_header = not result_path.exists()
    with open(result_path, "a", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        with torch.no_grad():
            for batch_start in range(0, len(pending), args.batch_size):
                batch_rows = pending[batch_start : batch_start + args.batch_size]
                mixtures = []
                reverb_targets = []
                clean_targets = []
                for row in batch_rows:
                    mixtures.append(read_generated(row["mix_path"], 1)[0])
                    reverb_targets.append(read_generated(row["targets_reverb_path"], 2))
                    clean_targets.append(read_generated(row["targets_clean_path"], 2))
                mixture_array = np.stack(mixtures, axis=0)
                mixture_tensor = torch.from_numpy(mixture_array.astype(np.float32)).to(device)
                output = model(mixture_tensor)
                xsep = output["x_sep"].detach().cpu().numpy().astype(np.float64)
                xderev = output["x_derev"].detach().cpu().numpy().astype(np.float64)
                ctf = helpers.decode_complex_ctf(output["rir"])
                decoded = decoder.decode(ctf)
                if decoded.shape[0] != 2 * len(batch_rows):
                    raise ValueError("Unexpected decoded response count {}".format(decoded.shape[0]))
                for batch_index, row in enumerate(batch_rows):
                    reverb_target = reverb_targets[batch_index]
                    clean_target = clean_targets[batch_index]
                    xsep_metrics = assignment_metrics(xsep[batch_index], reverb_target, mixture_array[batch_index])
                    xderev_metrics = assignment_metrics(xderev[batch_index], clean_target, mixture_array[batch_index])
                    drr_values = []
                    for source in range(2):
                        raw_response = decoded[batch_index * 2 + source]
                        response, direct_index, _, _ = helpers.prepare_effective_response(
                            raw_response,
                            int(round(0.005 * SAMPLE_RATE)),
                            int(round(1.005 * SAMPLE_RATE)),
                        )
                        drr_values.append(
                            helpers.direct_to_reverberant_ratio(
                                response,
                                direct_index,
                                int(round(0.0025 * SAMPLE_RATE)),
                            )
                        )
                    response_correct = int(drr_values[0] > drr_values[1])
                    output_row = dict(row)
                    for prefix, metrics in (("xsep", xsep_metrics), ("xderev", xderev_metrics)):
                        for name, value in metrics.items():
                            output_row[prefix + "_" + name] = value
                    output_row.update(
                        {
                            "response_drr_near_db": drr_values[0],
                            "response_drr_far_db": drr_values[1],
                            "response_drr_gap_db": drr_values[0] - drr_values[1],
                            "response_drr_order_correct": response_correct,
                            "waveform_response_order_agree": int(
                                xsep_metrics["fixed_order_correct"] == response_correct
                            ),
                            "response_measured_drr_order_agree": int(
                                response_correct == int(row["measured_rir_drr_order_correct"])
                            ),
                        }
                    )
                    writer.writerow(output_row)
                output_handle.flush()
                done = batch_start + len(batch_rows)
                if done % 25 == 0 or done == len(pending):
                    print("Evaluated {}/{} pending examples".format(done, len(pending)))
    return summarize_results(output_dir, manifest_path, signature)


def wilson_interval(correct, total, z=1.959963984540054):
    if total <= 0:
        return float("nan"), float("nan")
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def finite_mean(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def finite_correlation(target, estimate, method):
    target = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(estimate)
    target = target[valid]
    estimate = estimate[valid]
    if target.size < 2 or np.std(target) <= EPS or np.std(estimate) <= EPS:
        return float("nan")
    if method == "pearson":
        return float(stats.pearsonr(target, estimate)[0])
    if method == "spearman":
        return float(stats.spearmanr(target, estimate)[0])
    raise ValueError("Unknown correlation method {}".format(method))


def summarize_group(rows, rir_set, subset):
    count = len(rows)
    result = {"rir_set": rir_set, "subset": subset, "n_examples": count}
    for field in (
        "xsep_fixed_order_correct",
        "xderev_fixed_order_correct",
        "response_drr_order_correct",
        "measured_rir_drr_order_correct",
        "waveform_response_order_agree",
        "response_measured_drr_order_agree",
    ):
        correct = sum(int(row[field]) for row in rows)
        low, high = wilson_interval(correct, count)
        prefix = field.replace("_correct", "")
        result[prefix + "_correct"] = correct
        result[prefix + "_accuracy"] = correct / count if count else float("nan")
        result[prefix + "_wilson_low"] = low
        result[prefix + "_wilson_high"] = high
    for field in (
        "xsep_fixed_si_sdri_db",
        "xsep_oracle_si_sdri_db",
        "xderev_fixed_si_sdri_db",
        "xderev_oracle_si_sdri_db",
        "xsep_assignment_margin_db",
        "xderev_assignment_margin_db",
        "response_drr_gap_db",
        "measured_rir_drr_gap_db",
        "source_level_difference_db",
    ):
        result[field + "_mean"] = finite_mean(rows, field)
    pair_groups = {}
    for row in rows:
        key = (row["near_distance_m"], row["far_distance_m"])
        pair_groups.setdefault(key, []).append(row)
    pair_accuracies = [
        np.mean([int(row["xsep_fixed_order_correct"]) for row in pair_rows])
        for pair_rows in pair_groups.values()
    ]
    result["n_distance_pairs"] = len(pair_groups)
    result["xsep_pair_macro_accuracy"] = float(np.mean(pair_accuracies)) if pair_accuracies else float("nan")
    result["xsep_pairs_majority_correct"] = sum(value > 0.5 for value in pair_accuracies)
    result["xsep_pairs_unanimously_correct"] = sum(value == 1.0 for value in pair_accuracies)
    response_pair_accuracies = [
        np.mean([int(row["response_drr_order_correct"]) for row in pair_rows])
        for pair_rows in pair_groups.values()
    ]
    result["response_pair_macro_accuracy"] = (
        float(np.mean(response_pair_accuracies)) if response_pair_accuracies else float("nan")
    )
    result["response_pairs_majority_correct"] = sum(
        value > 0.5 for value in response_pair_accuracies
    )
    result["response_pairs_unanimously_correct"] = sum(
        value == 1.0 for value in response_pair_accuracies
    )
    measured_drr = np.asarray(
        [
            float(row[field])
            for row in rows
            for field in ("measured_rir_drr_near_db", "measured_rir_drr_far_db")
        ],
        dtype=np.float64,
    )
    predicted_drr = np.asarray(
        [
            float(row[field])
            for row in rows
            for field in ("response_drr_near_db", "response_drr_far_db")
        ],
        dtype=np.float64,
    )
    valid_drr = np.isfinite(measured_drr) & np.isfinite(predicted_drr)
    drr_error = predicted_drr[valid_drr] - measured_drr[valid_drr]
    result["response_measured_drr_n"] = int(drr_error.size)
    result["response_measured_drr_mae_db"] = float(np.mean(np.abs(drr_error)))
    result["response_measured_drr_rmse_db"] = float(np.sqrt(np.mean(np.square(drr_error))))
    result["response_measured_drr_bias_db"] = float(np.mean(drr_error))
    result["response_measured_drr_pearson"] = finite_correlation(
        measured_drr, predicted_drr, "pearson"
    )
    result["response_measured_drr_spearman"] = finite_correlation(
        measured_drr, predicted_drr, "spearman"
    )
    near_louder = [row for row in rows if float(row["source_level_difference_db"]) > 0.0]
    far_louder = [row for row in rows if float(row["source_level_difference_db"]) < 0.0]
    result["near_louder_n"] = len(near_louder)
    result["far_louder_n"] = len(far_louder)
    for label, selected in (("near_louder", near_louder), ("far_louder", far_louder)):
        for metric in ("xsep_fixed_order_correct", "response_drr_order_correct"):
            if selected:
                result["{}_{}_accuracy".format(label, metric.replace("_correct", ""))] = float(
                    np.mean([int(row[metric]) for row in selected])
                )
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.floating):
        return float(value)
    return value


def plot_accuracy_heatmaps(rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("xsep_fixed_order_correct", "Waveform fixed order"),
        ("response_drr_order_correct", "Predicted-response DRR order"),
        ("measured_rir_drr_order_correct", "Measured-RIR DRR monotonicity"),
    )
    figure, axes = plt.subplots(
        len(RIR_SETS),
        3,
        figsize=(17, 5 * len(RIR_SETS)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_index, rir_set in enumerate(RIR_SETS):
        room_rows = [row for row in rows if row["rir_set"] == rir_set]
        distances = sorted(
            {float(row["near_distance_m"]) for row in room_rows}
            | {float(row["far_distance_m"]) for row in room_rows}
        )
        distance_to_index = {distance: index for index, distance in enumerate(distances)}
        for column_index, (metric, title) in enumerate(metrics):
            values = np.full((len(distances), len(distances)), np.nan, dtype=np.float64)
            buckets = {}
            for row in room_rows:
                key = (float(row["near_distance_m"]), float(row["far_distance_m"]))
                buckets.setdefault(key, []).append(int(row[metric]))
            for (near, far), observations in buckets.items():
                values[distance_to_index[near], distance_to_index[far]] = np.mean(observations)
            axis = axes[row_index, column_index]
            image = axis.imshow(
                values,
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                cmap="coolwarm",
                interpolation="nearest",
                extent=(distances[0] - 0.05, distances[-1] + 0.05, distances[0] - 0.05, distances[-1] + 0.05),
                aspect="equal",
            )
            overall = np.mean([int(row[metric]) for row in room_rows])
            axis.set_title("{} {}\nAccuracy: {:.2f}%".format(rir_set, title, 100.0 * overall))
            axis.set_xlabel("Far distance (m)")
            axis.set_ylabel("Near distance (m)")
            tick_stride = 2 if len(distances) > 15 else 1
            ticks = distances[::tick_stride]
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            axis.tick_params(axis="x", rotation=45)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.85)
    colorbar.set_label("Accuracy across content repeats")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def summarize_results(output_dir, manifest_path, signature):
    result_path = output_dir / "per_example_metrics.csv"
    with open(result_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No result rows")
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate result IDs")
    subsets = (
        ("all", None),
        ("shared_grid", "shared_grid"),
        ("gap_ge_0p5", "gap_ge_0p5"),
        ("gap_ge_0p8", "gap_ge_0p8"),
        ("ratio_in_train_support", "ratio_in_train_support"),
        ("strict_train_geometry_overlap", "strict_train_geometry_overlap"),
    )
    summaries = []
    for rir_set in RIR_SETS:
        set_rows = [row for row in rows if row["rir_set"] == rir_set]
        for subset_name, flag in subsets:
            selected = set_rows if flag is None else [row for row in set_rows if int(row[flag]) == 1]
            if selected:
                summaries.append(summarize_group(selected, rir_set, subset_name))
    fieldnames = sorted({key for summary in summaries for key in summary})
    summary_csv_path = output_dir / "summary.csv"
    with open(summary_csv_path.with_suffix(".csv.tmp"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    os.replace(summary_csv_path.with_suffix(".csv.tmp"), summary_csv_path)
    payload = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "checkpoint": signature["checkpoint"],
        "checkpoint_sha256": signature["checkpoint_sha256"],
        "n_result_rows": len(rows),
        "summaries": summaries,
        "notes": [
            "All subset rules are geometry-only and fixed before examining model outputs.",
            "xsep order uses fixed-versus-swapped pairwise SI-SDR on reverberant targets.",
            "response order uses the DRR of each decoded effective response; no response PIT.",
            "Wilson intervals treat examples as Bernoulli trials and are descriptive because RIRs repeat.",
        ],
    }
    with open(output_dir / "summary.json.tmp", "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(output_dir / "summary.json.tmp", output_dir / "summary.json")
    plot_accuracy_heatmaps(rows, output_dir / "accuracy_heatmaps.png")
    print("Wrote summaries to {}".format(summary_csv_path))
    return summary_csv_path


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    manifest_path = args.output_dir / "manifest.csv"
    if args.mode in ("prepare", "all"):
        manifest_path = prepare_dataset(args)
    if args.mode in ("evaluate", "all"):
        if not manifest_path.is_file():
            raise FileNotFoundError("Prepare the dataset first: {}".format(manifest_path))
        evaluate_dataset(args, manifest_path)


if __name__ == "__main__":
    main()
