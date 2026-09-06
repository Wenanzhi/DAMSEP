#!/usr/bin/env python3
"""Run DARS on monaural mixtures and export waveforms plus decoded responses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dars.yml"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "dars_mixed_p10" / "best.pth"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Monaural mixture WAV files.")
    parser.add_argument("--conf-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--response-seconds", type=float, default=1.0)
    parser.add_argument("--pre-direct-ms", type=float, default=2.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.response_seconds <= 0.0:
        parser.error("--response-seconds must be positive")
    if args.pre_direct_ms < 0.0:
        parser.error("--pre-direct-ms must be non-negative")
    return args


def load_mixture(path: Path, expected_rate: int) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != expected_rate:
        raise ValueError(
            "{} uses {} Hz; the checkpoint expects {} Hz".format(
                path, sample_rate, expected_rate
            )
        )
    if waveform.shape[1] != 1:
        raise ValueError("DARS expects a monaural mixture: {}".format(path))
    waveform = waveform[:, 0]
    if waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise ValueError("Empty or non-finite mixture: {}".format(path))
    return waveform


def allocate_paths(output_dir: Path, stem: str) -> Dict[str, Path]:
    return {
        "reverberant": output_dir / "separated_reverberant" / (stem + ".wav"),
        "clean": output_dir / "estimated_clean" / (stem + ".wav"),
        "response": output_dir / "estimated_responses" / (stem + ".wav"),
    }


def main(args: argparse.Namespace) -> None:
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch

    import evaluate_rir_metrics as rir_metrics
    from look2hear.models import SPMamba

    config_path = args.conf_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError("Config not found: {}".format(config_path))
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError("{} exists; pass --overwrite".format(manifest_path))
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    sample_rate = int(config["datamodule"]["data_config"]["sample_rate"])

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = SPMamba.from_pretrain(
        str(checkpoint),
        sample_rate=sample_rate,
        **config["audionet"]["audionet_config"],
    ).to(device)
    model.eval()
    decoder = rir_metrics.SweepRIRDecoder(sample_rate, device)
    pre_samples = int(round(args.pre_direct_ms * sample_rate / 1000.0))
    response_samples = pre_samples + int(round(args.response_seconds * sample_rate))

    for directory in ("separated_reverberant", "estimated_clean", "estimated_responses"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    seen_stems = set()
    with torch.inference_mode():
        for input_path in args.inputs:
            input_path = input_path.expanduser().resolve()
            stem = input_path.stem
            if stem in seen_stems:
                raise ValueError("Input basenames must be unique: {}".format(stem))
            seen_stems.add(stem)
            paths = allocate_paths(output_dir, stem)
            existing = [path for path in paths.values() if path.exists()]
            if existing and not args.overwrite:
                raise FileExistsError(
                    "Outputs already exist for {}; pass --overwrite".format(input_path)
                )

            mixture = load_mixture(input_path, sample_rate)
            output = model(torch.from_numpy(mixture).unsqueeze(0).to(device))
            reverberant = output["x_sep"][0].detach().cpu().numpy().T
            clean = output["x_derev"][0].detach().cpu().numpy().T
            ctf = rir_metrics.decode_complex_ctf(output["rir"])
            decoded = decoder.decode(ctf)
            responses = []
            direct_indices = []
            for raw_response in decoded:
                response, direct_index, _, _ = rir_metrics.prepare_effective_response(
                    raw_response, pre_samples, response_samples
                )
                responses.append(response)
                direct_indices.append(direct_index)

            sf.write(paths["reverberant"], reverberant, sample_rate, subtype="FLOAT")
            sf.write(paths["clean"], clean, sample_rate, subtype="FLOAT")
            sf.write(
                paths["response"],
                np.stack(responses, axis=1).astype(np.float32),
                sample_rate,
                subtype="FLOAT",
            )
            records.append(
                {
                    "input": str(input_path),
                    "num_samples": int(mixture.size),
                    "sample_rate": sample_rate,
                    "output_source_order": ["near", "far"],
                    "direct_indices": direct_indices,
                    "separated_reverberant": str(paths["reverberant"]),
                    "estimated_clean": str(paths["clean"]),
                    "estimated_responses": str(paths["response"]),
                }
            )
            print("Processed {}".format(input_path))

    manifest = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "response_seconds": args.response_seconds,
        "pre_direct_ms": args.pre_direct_ms,
        "items": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("manifest={}".format(manifest_path))


if __name__ == "__main__":
    main(parse_args())
