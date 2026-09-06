#!/usr/bin/env python3
"""Evaluate every ordered-main-table checkpoint with fixed output identity.

Each model runs in a separate Python process because the projects contain
different top-level ``look2hear`` packages. Existing metric CSV files are never
overwritten; ``evaluate_checkpoint`` allocates a suffixed path on collision.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    adapter: str
    project_dir: str
    config: str
    checkpoint: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "dars": ModelSpec(
        adapter="dars",
        project_dir=".",
        config="configs/dars.yml",
        checkpoint="checkpoints/dars_mixed_p10/best.pth",
    ),
    "spmamba": ModelSpec(
        adapter="spmamba",
        project_dir="third_party/SPMamba",
        config="third_party/SPMamba/Experiments/checkpoint/mixed_g3090_p5_0227/conf.yml",
        checkpoint="third_party/SPMamba/Experiments/checkpoint/mixed_g3090_p5_0227/best_model.pth",
    ),
    "tdanet": ModelSpec(
        adapter="tdanet",
        project_dir="third_party/TDANet-Large",
        config="third_party/TDANet-Large/Experiments/checkpoint/tdanet_large200_nopit/conf.yml",
        checkpoint="third_party/TDANet-Large/Experiments/checkpoint/tdanet_large200_nopit/best_model.pth",
    ),
    "tflocoformer": ModelSpec(
        adapter="tflocoformer",
        project_dir="third_party/TF-Locoformer-M",
        config="third_party/TF-Locoformer-M/Experiments/checkpoint/tflocoformer_m_nopit20_mixed_dataset_official_stable/config.yaml",
        checkpoint="third_party/TF-Locoformer-M/Experiments/checkpoint/tflocoformer_m_nopit20_mixed_dataset_official_stable/best_model.pth",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all",) + tuple(MODEL_SPECS),
        default=("all",),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=("cuda:0",),
        help="Devices assigned round-robin to selected models.",
    )
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--output-tag", default="fixed_identity")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--dnsmos-threads", type=int, default=8)
    parser.add_argument("--bss-filter-length", type=int, default=512)
    parser.add_argument(
        "--logs-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "main_table" / "logs"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_models(requested: Sequence[str]) -> List[str]:
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("Use --models all by itself.")
        return list(MODEL_SPECS)
    seen = set()
    selected = []
    for model_name in requested:
        if model_name not in seen:
            selected.append(model_name)
            seen.add(model_name)
    return selected


def resolve_workspace_path(relative_path: str, expect_directory: bool) -> Path:
    path = (WORKSPACE_ROOT / relative_path).resolve()
    valid = path.is_dir() if expect_directory else path.is_file()
    if not valid:
        kind = "directory" if expect_directory else "file"
        raise FileNotFoundError("Missing {}: {}".format(kind, path))
    return path


def build_command(
    model_name: str,
    device: str,
    args: argparse.Namespace,
) -> List[str]:
    spec = MODEL_SPECS[model_name]
    project_dir = resolve_workspace_path(spec.project_dir, True)
    config = resolve_workspace_path(spec.config, False)
    checkpoint = resolve_workspace_path(spec.checkpoint, False)
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_checkpoint",
        "--adapter",
        spec.adapter,
        "--project-dir",
        str(project_dir),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--alignment",
        "fixed",
        "--device",
        device,
        "--output-name",
        "metrics_extended_{}.csv".format(args.output_tag),
        "--output-dir",
        str(WORKSPACE_ROOT / "outputs" / "main_table" / model_name),
        "--dnsmos-threads",
        str(args.dnsmos_threads),
        "--bss-filter-length",
        str(args.bss_filter_length),
        "--start-index",
        str(args.start_index),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.end_index is not None:
        command.extend(("--end-index", str(args.end_index)))
    if args.max_examples is not None:
        command.extend(("--max-examples", str(args.max_examples)))
    return command


def main(args: argparse.Namespace) -> int:
    models = selected_models(args.models)
    if not args.devices:
        raise ValueError("At least one device is required.")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("max_examples must be positive.")
    if args.start_index < 0:
        raise ValueError("start_index cannot be negative.")

    logs_dir = Path(args.logs_dir).expanduser().resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for index, model_name in enumerate(models):
        device = args.devices[index % len(args.devices)]
        commands.append((model_name, device, build_command(model_name, device, args)))

    for model_name, device, command in commands:
        print("[{} on {}] {}".format(model_name, device, shlex.join(command)))
    if args.dry_run:
        return 0

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(WORKSPACE_ROOT)
    started_at = time.time()
    records = []

    if args.parallel:
        running = []
        for model_name, device, command in commands:
            log_path = logs_dir / "{}_{}.log".format(model_name, args.output_tag)
            log_file = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(WORKSPACE_ROOT),
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((model_name, device, command, log_path, log_file, process))

        failed = False
        for model_name, device, command, log_path, log_file, process in running:
            return_code = process.wait()
            log_file.close()
            records.append(
                {
                    "model": model_name,
                    "device": device,
                    "return_code": return_code,
                    "log": str(log_path),
                    "command": command,
                }
            )
            failed = failed or return_code != 0
            print("[{}] return_code={} log={}".format(model_name, return_code, log_path))
        exit_code = 1 if failed else 0
    else:
        exit_code = 0
        for model_name, device, command in commands:
            log_path = logs_dir / "{}_{}.log".format(model_name, args.output_tag)
            with log_path.open("x", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    command,
                    cwd=str(WORKSPACE_ROOT),
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            records.append(
                {
                    "model": model_name,
                    "device": device,
                    "return_code": completed.returncode,
                    "log": str(log_path),
                    "command": command,
                }
            )
            print(
                "[{}] return_code={} log={}".format(
                    model_name, completed.returncode, log_path
                )
            )
            if completed.returncode != 0:
                exit_code = 1
                break

    manifest_path = logs_dir / "run_{}.json".format(args.output_tag)
    manifest = {
        "alignment": "fixed",
        "elapsed_seconds": time.time() - started_at,
        "models": records,
        "output_tag": args.output_tag,
    }
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    print("manifest={}".format(manifest_path))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
