#!/usr/bin/env python3
"""Run released Rec-RIR or VINP inference on a manifest slice."""

import argparse
import csv
import errno
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("recrir", "vinp"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Path to the separately obtained Rec-RIR or VINP checkout.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more checkpoints; Rec-RIR accepts exactly one.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.end_index is not None and args.end_index <= args.start_index:
        parser.error("--end-index must be larger than --start-index")
    return args


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    import toml
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = read_rows(args.manifest.expanduser().resolve())
    end = len(rows) if args.end_index is None else min(args.end_index, len(rows))
    rows = rows[args.start_index:end]
    if not rows:
        raise RuntimeError("No manifest rows selected")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("Output already exists: {}".format(output_dir))
    input_dir = output_dir / "input_wav"
    rir_dir = output_dir / "rir"
    input_dir.mkdir(parents=True)
    rir_dir.mkdir()
    if args.method == "vinp":
        (output_dir / "normed").mkdir()
    for row in rows:
        source = Path(row["input_path"]).resolve()
        destination = input_dir / source.name
        if destination.exists():
            raise FileExistsError(destination)
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            os.symlink(source, destination)

    project_root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_paths = [path.expanduser().resolve() for path in args.checkpoint]
    if not project_root.is_dir():
        raise FileNotFoundError("Project root not found: {}".format(project_root))
    if not config_path.is_file():
        raise FileNotFoundError("Config not found: {}".format(config_path))
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    if args.method == "recrir" and len(checkpoint_paths) != 1:
        raise ValueError("Rec-RIR accepts exactly one checkpoint")
    config = toml.load(str(config_path))
    previous_cwd = Path.cwd()
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    try:
        if args.method == "recrir":
            from inference import inference

            inference(
                input_path=str(input_dir),
                output_path=str(output_dir),
                ckpt=str(checkpoint_paths[0]),
                device=args.device,
                **config
            )
        else:
            from enhance_rir_avg import enhance_avg

            enhance_avg(
                input_path=str(input_dir),
                output_path=str(output_dir),
                ckpt=[str(path) for path in checkpoint_paths],
                device=args.device,
                **config
            )
    finally:
        os.chdir(previous_cwd)

    expected = {Path(row["input_path"]).name for row in rows}
    actual = {path.name for path in rir_dir.glob("*.wav")}
    if actual != expected:
        raise RuntimeError(
            "Prediction set mismatch: expected {}, found {}".format(len(expected), len(actual))
        )
    metadata = {
        "method": args.method,
        "manifest": str(args.manifest.expanduser().resolve()),
        "config": str(config_path.resolve()),
        "checkpoints": [str(path.resolve()) for path in checkpoint_paths],
        "start_index": args.start_index,
        "end_index": end,
        "prediction_count": len(actual),
        "seed": args.seed,
    }
    with (output_dir / "adapter_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("{} wrote {} RIRs to {}".format(args.method, len(actual), rir_dir))


if __name__ == "__main__":
    main()
