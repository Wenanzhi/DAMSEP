#!/usr/bin/env python3
"""Export DARS reverberant stems and decoded responses on measured-RIR mixtures."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


DARS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = DARS_ROOT / "outputs" / "measured_rir" / "matched_0716"
DEFAULT_EXP = DARS_ROOT / "checkpoints" / "dars_mixed_p10"
DEFAULT_OUTPUT = DARS_ROOT / "outputs" / "measured_rir" / "baseline_inputs_0716"
SAMPLE_RATE = 8000
TARGET_SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size < 1 or args.print_every < 1:
        parser.error("--batch-size and --print-every must be positive")
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def output_name(index, source, example_id):
    return "{:04d}__s{}__{}.wav".format(index, source, example_id)


def load_mixture(path):
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or waveform.shape != (4 * SAMPLE_RATE, 1):
        raise ValueError("Unexpected measured mixture shape/rate for {}".format(path))
    if not np.all(np.isfinite(waveform)):
        raise ValueError("Non-finite measured mixture: {}".format(path))
    return waveform[:, 0]


def resample(waveform, source_rate, target_rate):
    if source_rate == target_rate:
        return np.asarray(waveform, dtype=np.float32)
    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(
        np.asarray(waveform, dtype=np.float32),
        target_rate // divisor,
        source_rate // divisor,
    ).astype(np.float32)


def load_helpers():
    module_path = Path(__file__).resolve().parent / "run_matched_measured_rir.py"
    spec = importlib.util.spec_from_file_location("measured_rir_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rir_helpers():
    module_path = DARS_ROOT / "evaluate_rir_metrics.py"
    spec = importlib.util.spec_from_file_location("dars_rir_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_existing(manifest_path):
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return {}
    return {
        (int(row["dataset_index"]), int(row["source"])): row
        for row in read_csv(manifest_path)
    }


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    source_manifest = dataset_dir / "manifest.csv"
    exp_dir = args.exp_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.csv"
    config_path = output_dir / "run_config.json"
    audio_dir = output_dir / "wav16k"
    dars_rir_dir = output_dir / "dars_predictions" / "rir"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    dars_rir_dir.mkdir(parents=True, exist_ok=True)

    protected = (manifest_path, config_path)
    if args.overwrite:
        for path in protected:
            path.unlink(missing_ok=True)
        for directory in (audio_dir, dars_rir_dir):
            for path in directory.glob("*.wav"):
                path.unlink()
    elif any(path.exists() for path in protected) and not args.resume:
        raise FileExistsError("Output exists; pass --resume or --overwrite: {}".format(output_dir))

    rows = read_csv(source_manifest)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    expected_config = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "checkpoint": str((exp_dir / "best.pth").resolve()),
        "checkpoint_sha256": sha256(exp_dir / "best.pth"),
        "assignment": "fixed_identity",
        "rir_channel": 0,
        "input_sample_rate": SAMPLE_RATE,
        "baseline_input_sample_rate": TARGET_SAMPLE_RATE,
        "max_examples": args.max_examples,
    }
    if args.resume and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != expected_config:
                raise ValueError("Resume configuration does not match existing export")

    existing = load_existing(manifest_path) if args.resume else {}
    helpers = load_helpers()
    rir_helpers = load_rir_helpers()
    device = torch.device(args.device)
    model = helpers.load_dars_model(exp_dir, device)
    decoder = rir_helpers.SweepRIRDecoder(SAMPLE_RATE, device)
    fieldnames = [
        "utterance",
        "dataset_index",
        "crop_start",
        "source",
        "model_output_source",
        "role",
        "distance_m",
        "rir_set",
        "repeat",
        "level_protocol",
        "input_path",
        "full_rir_path",
        "direct_rir_path",
        "reference_rir_channel",
        "anechoic_path",
        "reverberant_target_path",
    ]

    pending = []
    for index, row in enumerate(rows):
        keys = ((index, 1), (index, 2))
        complete = all(
            key in existing
            and Path(existing[key]["input_path"]).is_file()
            and (dars_rir_dir / Path(existing[key]["input_path"]).name).is_file()
            for key in keys
        )
        if not complete:
            pending.append((index, row))

    with torch.inference_mode():
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            mixture = np.stack([load_mixture(row["mix_path"]) for _, row in batch])
            output = model(torch.from_numpy(mixture).to(device))
            xsep = output["x_sep"].detach().cpu().numpy()
            ctf = rir_helpers.decode_complex_ctf(output["rir"])
            decoded = decoder.decode(ctf)
            if xsep.shape != (len(batch), 2, 4 * SAMPLE_RATE):
                raise ValueError("Unexpected x_sep shape {}".format(xsep.shape))
            if decoded.shape[0] != 2 * len(batch):
                raise ValueError("Unexpected decoded response shape {}".format(decoded.shape))

            for batch_index, (dataset_index, row) in enumerate(batch):
                for source in (1, 2):
                    role = "near" if source == 1 else "far"
                    distance = row["near_distance_m"] if source == 1 else row["far_distance_m"]
                    rir_path = row["near_rir_path"] if source == 1 else row["far_rir_path"]
                    filename = output_name(dataset_index, source, row["example_id"])
                    input_path = audio_dir / filename
                    predicted_rir_path = dars_rir_dir / filename
                    stem = resample(xsep[batch_index, source - 1], SAMPLE_RATE, TARGET_SAMPLE_RATE)
                    predicted_rir = np.asarray(
                        decoded[batch_index * 2 + source - 1], dtype=np.float32
                    ).reshape(-1)
                    if not np.all(np.isfinite(stem)) or np.max(np.abs(stem)) <= 0.0:
                        raise ValueError("Invalid DARS stem for {} source {}".format(row["example_id"], source))
                    if not np.all(np.isfinite(predicted_rir)) or np.max(np.abs(predicted_rir)) <= 0.0:
                        raise ValueError("Invalid DARS response for {} source {}".format(row["example_id"], source))
                    sf.write(input_path, stem, TARGET_SAMPLE_RATE, subtype="FLOAT")
                    sf.write(predicted_rir_path, predicted_rir, SAMPLE_RATE, subtype="FLOAT")
                    existing[(dataset_index, source)] = {
                        "utterance": row["example_id"],
                        "dataset_index": dataset_index,
                        "crop_start": row["crop_start"],
                        "source": source,
                        "model_output_source": source,
                        "role": role,
                        "distance_m": distance,
                        "rir_set": row["rir_set"],
                        "repeat": row["repeat"],
                        "level_protocol": row.get("level_protocol", "balanced"),
                        "input_path": str(input_path),
                        "full_rir_path": rir_path,
                        "direct_rir_path": "",
                        "reference_rir_channel": 0,
                        "anechoic_path": row["targets_clean_path"],
                        "reverberant_target_path": row["targets_reverb_path"],
                    }
            write_csv(manifest_path, [existing[key] for key in sorted(existing)], fieldnames)
            done = batch_start + len(batch)
            if done % args.print_every == 0 or done == len(pending):
                print("Exported {}/{} pending mixtures".format(done, len(pending)), flush=True)

    expected_rows = 2 * len(rows)
    if len(existing) != expected_rows:
        raise RuntimeError("Expected {} source rows, found {}".format(expected_rows, len(existing)))
    write_csv(manifest_path, [existing[key] for key in sorted(existing)], fieldnames)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(expected_config, handle, indent=2, sort_keys=True)
    print("Wrote {} source rows to {}".format(len(existing), manifest_path))


if __name__ == "__main__":
    main()
