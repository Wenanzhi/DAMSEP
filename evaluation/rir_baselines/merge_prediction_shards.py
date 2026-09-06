#!/usr/bin/env python3
"""Merge disjoint RIR prediction shards and verify exact manifest coverage."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--shards-dir",
        type=Path,
        action="append",
        required=True,
        help="Shard root containing shard_*/rir. Repeat for disjoint shard roots.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    with args.manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {Path(row["input_path"]).name for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Manifest input basenames are not unique")
    if args.output_dir.exists():
        raise FileExistsError("Output already exists: {}".format(args.output_dir))
    predictions = {}
    shard_configs = []
    shard_roots = []
    for shards_dir in args.shards_dir:
        shards_dir = shards_dir.expanduser().resolve()
        rir_dirs = sorted(shards_dir.glob("shard_*/rir"))
        if not rir_dirs:
            raise FileNotFoundError("No shard_*/rir directories under {}".format(shards_dir))
        shard_roots.append(str(shards_dir))
        for rir_dir in rir_dirs:
            config_path = rir_dir.parent / "adapter_run_config.json"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as handle:
                    shard_configs.append(json.load(handle))
            for path in rir_dir.glob("*.wav"):
                if path.name in predictions:
                    raise ValueError("Duplicate prediction basename: {}".format(path.name))
                predictions[path.name] = path.resolve()
    actual = set(predictions)
    if actual != expected:
        raise ValueError(
            "Coverage mismatch: missing {}, extra {}".format(
                sorted(expected - actual)[:5], sorted(actual - expected)[:5]
            )
        )
    output_rir = args.output_dir / "rir"
    output_rir.mkdir(parents=True)
    for name in sorted(predictions):
        destination = output_rir / name
        try:
            os.link(predictions[name], destination)
        except OSError:
            destination.symlink_to(predictions[name])
    metadata = {
        "method": args.method,
        "manifest": str(args.manifest.resolve()),
        "prediction_count": len(predictions),
        "shard_roots": shard_roots,
        "shard_configs": shard_configs,
    }
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metadata["checkpoint"] = str(checkpoint)
        metadata["checkpoint_sha256"] = sha256(checkpoint)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("Merged {} {} predictions".format(len(predictions), args.method))


if __name__ == "__main__":
    main()
