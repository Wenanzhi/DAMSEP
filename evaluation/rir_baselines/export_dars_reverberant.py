#!/usr/bin/env python3
"""Export PIT- or fixed-aligned reverberant estimates for blind RIR baselines."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


DARS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONF = DARS_ROOT / "configs" / "dars.yml"
DEFAULT_CHECKPOINT = DARS_ROOT / "checkpoints" / "dars_mixed_p10" / "best.pth"
DEFAULT_OUTPUT = DARS_ROOT / "outputs" / "rir_baselines" / "dars_stems"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-root", type=Path, default=DARS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--distance-metadata", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--assignment-mode",
        choices=("auto", "fixed_identity", "waveform_pit"),
        default="auto",
    )
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.end_index is not None and args.end_index <= args.start_index:
        parser.error("--end-index must be larger than --start-index")
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    return args


def safe_stem(key):
    return Path(key).stem.replace(" ", "_")


def output_name(item, source):
    return "{:04d}__s{}__{}.wav".format(
        int(item["dataset_index"]), source, safe_stem(item["key"])
    )


def resample_audio(waveform, source_rate, target_rate):
    waveform = np.asarray(waveform, dtype=np.float32)
    if source_rate == target_rate:
        return waveform
    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(
        waveform, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)


def load_existing_manifest(path):
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["dataset_index"]), int(row["source"])): row for row in rows
    }


def write_manifest(path, rows, fieldnames):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    sys.path.insert(0, str(DARS_ROOT))
    import torch
    import yaml
    import evaluate_rir_metrics as rir_eval

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    conf_path = args.conf.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    if not model_root.exists():
        raise FileNotFoundError("Model root does not exist: {}".format(model_root))
    with conf_path.open("r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    assignment_mode = rir_eval.resolve_assignment_mode(conf, args.assignment_mode)
    sys.path.insert(0, str(model_root))
    items, sample_rate, manifest_count, distance_path = rir_eval.load_eval_items(
        conf, args.seed, args.distance_metadata
    )
    eligible_utterances = len(items)
    if args.max_examples is not None:
        items = items[: args.max_examples]
    end_index = len(items) if args.end_index is None else min(args.end_index, len(items))
    items = items[args.start_index:end_index]
    if not items:
        raise RuntimeError("No evaluation items selected")

    output_dir = args.output_dir.expanduser().resolve()
    audio_dir = output_dir / "wav16k"
    manifest_path = output_dir / "manifest.csv"
    run_path = output_dir / "run_config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        manifest_path.unlink(missing_ok=True)
        run_path.unlink(missing_ok=True)
    elif manifest_path.exists() and not args.resume:
        raise FileExistsError(
            "{} exists; pass --resume or --overwrite".format(manifest_path)
        )

    fieldnames = [
        "utterance",
        "dataset_index",
        "crop_start",
        "source",
        "model_output_source",
        "role",
        "distance_m",
        "input_path",
        "full_rir_path",
        "direct_rir_path",
        "anechoic_path",
        "reverberant_target_path",
    ]
    existing = load_existing_manifest(manifest_path) if args.resume else {}
    model = rir_eval.load_model(conf, checkpoint, sample_rate, torch.device(args.device))

    assignment_ties = 0
    for count, item in enumerate(items, 1):
        expected = [(int(item["dataset_index"]), source) for source in (1, 2)]
        if all(key in existing and Path(existing[key]["input_path"]).exists() for key in expected):
            continue
        mixture, clean, _ = rir_eval.load_item_audio(item, sample_rate)
        tensor = torch.from_numpy(mixture).float().unsqueeze(0).to(args.device)
        with torch.inference_mode():
            output = model(tensor)
        if not isinstance(output, dict) or "x_sep" not in output:
            raise RuntimeError("DARS model did not return the x_sep reverberant estimates")
        estimates_tensor = output["x_sep"]
        if estimates_tensor.ndim == 2:
            estimates_tensor = estimates_tensor.reshape(1, 2, -1)
        if assignment_mode == "fixed_identity":
            permutation = (0, 1)
        else:
            if "x_derev" not in output:
                raise RuntimeError("waveform_pit requires model output x_derev")
            target_tensor = torch.from_numpy(clean).float().unsqueeze(0)
            permutations, ties = rir_eval.waveform_pit_assignments(
                output["x_derev"], target_tensor
            )
            permutation = permutations[0]
            assignment_ties += ties
        estimates = estimates_tensor[:, list(permutation), :].detach().cpu().numpy()
        if estimates.shape != (1, 2, mixture.size):
            raise ValueError("Unexpected x_sep shape {}".format(estimates.shape))
        for source in (1, 2):
            filename = output_name(item, source)
            output_path = audio_dir / filename
            waveform = resample_audio(
                estimates[0, source - 1], sample_rate, args.target_sample_rate
            )
            if not np.all(np.isfinite(waveform)) or np.max(np.abs(waveform)) <= 0.0:
                raise ValueError("Invalid x_sep waveform for {} source {}".format(item["key"], source))
            sf.write(output_path, waveform, args.target_sample_rate, subtype="FLOAT")
            role, distance = rir_eval.geometry_role(item, source)
            existing[(int(item["dataset_index"]), source)] = {
                "utterance": item["key"],
                "dataset_index": item["dataset_index"],
                "crop_start": item["crop_start"],
                "source": source,
                "model_output_source": permutation[source - 1] + 1,
                "role": role,
                "distance_m": distance,
                "input_path": str(output_path),
                "full_rir_path": item["full_rir_path"],
                "direct_rir_path": item["direct_rir_path"],
                "anechoic_path": item["anechoic_paths"][source - 1],
                "reverberant_target_path": item["reverberant_paths"][source - 1],
            }
        if count % args.print_every == 0 or count == len(items):
            write_manifest(
                manifest_path,
                [existing[key] for key in sorted(existing)],
                fieldnames,
            )
            print("Exported {}/{} mixtures".format(count, len(items)), flush=True)

    write_manifest(
        manifest_path, [existing[key] for key in sorted(existing)], fieldnames
    )
    metadata = {
        "config": str(conf_path),
        "checkpoint": str(checkpoint),
        "model_root": str(model_root),
        "sample_rate": sample_rate,
        "target_sample_rate": args.target_sample_rate,
        "seed": args.seed,
        "assignment": assignment_mode,
        "assignment_ties": assignment_ties,
        "native_output": "model x_sep reverberant source estimate",
        "original_manifest_utterances": manifest_count,
        "eligible_utterances": eligible_utterances,
        "exported_source_rows": len(existing),
        "distance_metadata": str(distance_path) if distance_path else None,
    }
    with run_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("Wrote {} source rows to {}".format(len(existing), manifest_path))


if __name__ == "__main__":
    main()
