#!/usr/bin/env python3
"""Run Speech2RIR or FiNS on exported DARS stems."""

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
    parser.add_argument("--method", choices=("speech2rir", "fins"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Path to the separately obtained Speech2RIR or FiNS checkout.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_input(path):
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise ValueError("Expected 16 kHz input: {}".format(path))
    waveform = waveform[:, 0] if waveform.shape[1] == 1 else waveform.mean(axis=1)
    if not np.all(np.isfinite(waveform)) or np.max(np.abs(waveform)) <= 1e-8:
        raise ValueError("Silent or invalid input: {}".format(path))
    return waveform.astype(np.float32)


def normalize_output(waveform):
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(waveform)))
    if not np.isfinite(peak) or peak <= 1e-8:
        raise ValueError("Estimated RIR is silent or invalid")
    return waveform / peak


def load_speech2rir(project_root, checkpoint, config_path, device):
    import torch
    import yaml

    sys.path.insert(0, str(project_root))
    from models.autoencoder.AudioDec import Generator

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    original_torch_load = torch.load

    def mapped_torch_load(path, *args, **kwargs):
        if Path(str(path)).name == "resnet18-f37072fd.pth" and not Path(str(path)).exists():
            path = project_root / "resnet18-f37072fd.pth"
        return original_torch_load(path, *args, **kwargs)

    torch.load = mapped_torch_load
    try:
        model = Generator(**config["generator_params"])
    finally:
        torch.load = original_torch_load
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"]["generator"], strict=True)
    model.eval().to(device)
    return model


def infer_speech2rir(model, waveform, device, seed):
    import torch

    target_samples = 20000
    if waveform.size < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.size))
    else:
        waveform = waveform[:target_samples]
    tensor = torch.from_numpy(waveform).view(1, 1, -1).to(device)
    with torch.inference_mode():
        estimate = model(tensor)
    return normalize_output(estimate.squeeze().detach().cpu().numpy())


def load_fins(project_root, checkpoint, config_path, device):
    import torch

    sys.path.insert(0, str(project_root))
    from model_wpy2 import FilteredNoiseShaper
    from utils.utils import load_config

    config = load_config(str(config_path))
    model = FilteredNoiseShaper(config.model.params)
    payload = torch.load(checkpoint, map_location="cpu")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, config


def infer_fins(model_and_config, waveform, device, seed):
    import torch
    from utils.audio_wpy import audio_normalize_batch

    model, _ = model_and_config
    target_samples = 43840
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    waveform = waveform - np.mean(waveform)
    nonzero = np.flatnonzero(np.abs(waveform) > 1e-8)
    if nonzero.size:
        waveform = waveform[nonzero[0] : nonzero[-1] + 1]
    if waveform.size < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.size))
    else:
        waveform = waveform[:target_samples]
    tensor = torch.from_numpy(waveform.astype(np.float32)).view(1, 1, -1).to(device)
    tensor = audio_normalize_batch(tensor, "rms", 0.01)
    generator = torch.Generator(device=device).manual_seed(seed)
    stochastic_noise = torch.randn((1, 10, 15360), generator=generator, device=device)
    noise_condition = torch.randn((1, 16), generator=generator, device=device)
    with torch.inference_mode():
        estimate = model(tensor, stochastic_noise, noise_condition)
    return normalize_output(estimate.squeeze().detach().cpu().numpy())


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = read_rows(args.manifest.expanduser().resolve())
    if args.max_sources is not None:
        rows = rows[: args.max_sources]
    end_index = len(rows) if args.end_index is None else min(args.end_index, len(rows))
    rows = rows[args.start_index:end_index]
    output_dir = args.output_dir.expanduser().resolve()
    rir_dir = output_dir / "rir"
    rir_dir.mkdir(parents=True, exist_ok=True)

    project_root = args.project_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError("Project root not found: {}".format(project_root))
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))

    if args.method == "speech2rir":
        config = (args.config or checkpoint.parent / "config.yml").expanduser().resolve()
        model = load_speech2rir(project_root, checkpoint, config, args.device)
        infer = infer_speech2rir
    else:
        config = (args.config or project_root / "config.yaml").expanduser().resolve()
        model = load_fins(project_root, checkpoint, config, args.device)
        infer = infer_fins

    for index, row in enumerate(rows, 1):
        output_path = rir_dir / Path(row["input_path"]).name
        if output_path.exists() and args.resume:
            continue
        if output_path.exists() and not args.overwrite:
            raise FileExistsError("{} exists; pass --resume or --overwrite".format(output_path))
        waveform = load_input(row["input_path"])
        deterministic_seed = args.seed + int(row["dataset_index"]) * 2 + int(row["source"])
        estimate = infer(model, waveform, args.device, deterministic_seed)
        sf.write(output_path, estimate, 16000, subtype="FLOAT")
        if index % args.print_every == 0 or index == len(rows):
            print("{}: {}/{}".format(args.method, index, len(rows)), flush=True)


if __name__ == "__main__":
    main()
