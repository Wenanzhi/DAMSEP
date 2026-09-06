#!/usr/bin/env python3
"""Run the official BUDDy blind RIR estimator on exported DARS stems."""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "rir_baselines" / "dars_stems" / "manifest.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Path to a separately obtained BUDDy checkout.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def load_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fit_length(waveform, length):
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size >= length:
        return waveform[:length]
    if waveform.size == 0:
        raise ValueError("Empty input waveform")
    repeats = int(np.ceil(length / waveform.size))
    return np.tile(waveform, repeats)[:length]


def main():
    args_cli = parse_args()
    if args_cli.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.cuda_visible_devices
    import hydra
    import torch
    from hydra import compose, initialize_config_dir

    buddy_root = args_cli.project_root.expanduser().resolve()
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not (buddy_root / "conf").is_dir():
        raise FileNotFoundError("Invalid BUDDy project root: {}".format(buddy_root))
    if not checkpoint.is_file():
        raise FileNotFoundError("BUDDy checkpoint not found: {}".format(checkpoint))

    sys.path.insert(0, str(buddy_root))
    from testing.operators.subband_filtering import BlindSubbandFiltering
    from testing.tester import Tester

    if args_cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args_cli.device)
    with initialize_config_dir(config_dir=str(buddy_root / "conf"), version_base=None):
        args = compose(
            config_name="conf_VCTK",
            overrides=["tester=blind_dereverberation_BUDDy"],
        )
    args.model_dir = str(buddy_root / "experiments")
    args.exp.model_dir = args.model_dir
    diff_params = hydra.utils.instantiate(args.diff_params)
    network = hydra.utils.instantiate(args.network).to(device)
    tester = Tester(args=args, network=network, diff_params=diff_params, device=device)
    tester.load_checkpoint(str(checkpoint))
    sampler = tester.sampler

    rows = load_rows(args_cli.manifest.expanduser().resolve())
    if args_cli.max_sources is not None:
        rows = rows[: args_cli.max_sources]
    end_index = len(rows) if args_cli.end_index is None else min(args_cli.end_index, len(rows))
    rows = rows[args_cli.start_index:end_index]
    output_dir = args_cli.output_dir.expanduser().resolve()
    rir_dir = output_dir / "rir"
    rir_dir.mkdir(parents=True, exist_ok=True)
    audio_len = int(args.exp.audio_len)
    sample_rate = int(args.exp.sample_rate)
    scale = float(args.tester.posterior_sampling.warm_initialization.scaling_factor)

    for index, row in enumerate(rows, 1):
        output_path = rir_dir / Path(row["input_path"]).name
        if output_path.exists() and args_cli.resume:
            continue
        if output_path.exists() and not args_cli.overwrite:
            raise FileExistsError("{} exists; pass --resume or --overwrite".format(output_path))
        waveform, rate = sf.read(row["input_path"], dtype="float32", always_2d=True)
        if rate != sample_rate:
            raise ValueError("Expected {} Hz input".format(sample_rate))
        waveform = waveform[:, 0] if waveform.shape[1] == 1 else waveform.mean(axis=1)
        waveform = fit_length(waveform, audio_len)
        source_seed = args_cli.seed + int(row["dataset_index"]) * 2 + int(row["source"])
        random.seed(source_seed)
        np.random.seed(source_seed)
        torch.manual_seed(source_seed)
        torch.cuda.manual_seed_all(source_seed)
        y = torch.from_numpy(waveform).float().to(device)
        y = scale * y / y.std().clamp_min(1e-8)
        y = y.unsqueeze(0)
        operator = BlindSubbandFiltering(
            args.tester.informed_dereverberation.op_hp,
            sample_rate=sample_rate,
        )
        with torch.no_grad():
            operator.update_H(use_noise=True)
        sampler.predict_conditional(y, operator, shape=(1, audio_len), blind=True)
        rir = sampler.operator.get_time_RIR().detach().cpu().numpy().reshape(-1)
        peak = float(np.max(np.abs(rir)))
        if not np.isfinite(peak) or peak <= 1e-8:
            raise ValueError("BUDDy returned a silent or invalid RIR")
        sf.write(output_path, (rir / peak).astype(np.float32), sample_rate, subtype="FLOAT")
        print("BUDDy: {}/{} {}".format(index, len(rows), output_path.name), flush=True)


if __name__ == "__main__":
    main()
