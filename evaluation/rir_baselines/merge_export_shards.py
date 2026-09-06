#!/usr/bin/env python3
"""Merge disjoint reverberant-stem export shards into one validated manifest."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-mixtures", type=int, required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def main():
    args = parse_args()
    shard_dirs = sorted(path.parent for path in args.shards_dir.glob("shard_*/manifest.csv"))
    if not shard_dirs:
        raise FileNotFoundError("No shard manifests found under {}".format(args.shards_dir))
    if args.output_dir.exists():
        raise FileExistsError("Output already exists: {}".format(args.output_dir))

    fieldnames = None
    rows_by_key = {}
    run_configs = []
    for shard_dir in shard_dirs:
        current_fields, rows = read_csv(shard_dir / "manifest.csv")
        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            raise ValueError("Manifest schema mismatch in {}".format(shard_dir))
        with (shard_dir / "run_config.json").open("r", encoding="utf-8") as handle:
            run_configs.append(json.load(handle))
        for row in rows:
            key = (int(row["dataset_index"]), int(row["source"]))
            if key in rows_by_key:
                raise ValueError("Duplicate source row {}".format(key))
            rows_by_key[key] = row

    expected_sources = args.expected_mixtures * 2
    if len(rows_by_key) != expected_sources:
        raise ValueError(
            "Expected {} source rows, found {}".format(expected_sources, len(rows_by_key))
        )
    by_utterance = defaultdict(set)
    for row in rows_by_key.values():
        by_utterance[row["utterance"]].add(int(row["source"]))
    if len(by_utterance) != args.expected_mixtures:
        raise ValueError(
            "Expected {} mixtures, found {}".format(args.expected_mixtures, len(by_utterance))
        )
    invalid = [key for key, sources in by_utterance.items() if sources != {1, 2}]
    if invalid:
        raise ValueError("Mixtures without exactly sources 1/2: {}".format(invalid[:5]))

    stable_keys = ("config", "checkpoint", "model_root", "sample_rate", "target_sample_rate", "seed", "assignment")
    for key in stable_keys:
        values = {json.dumps(config.get(key), sort_keys=True) for config in run_configs}
        if len(values) != 1:
            raise ValueError("Shard run_config mismatch for {}".format(key))

    audio_dir = args.output_dir / "wav16k"
    audio_dir.mkdir(parents=True)
    merged_rows = []
    for key in sorted(rows_by_key):
        row = dict(rows_by_key[key])
        source_path = Path(row["input_path"]).resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        destination = audio_dir / source_path.name
        if destination.exists():
            raise FileExistsError(destination)
        os.link(source_path, destination)
        row["input_path"] = str(destination.resolve())
        merged_rows.append(row)

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    metadata = dict(run_configs[0])
    metadata.update(
        {
            "eligible_utterances": args.expected_mixtures,
            "exported_source_rows": expected_sources,
            "merged_shards": [str(path.resolve()) for path in shard_dirs],
            "assignment_ties": sum(int(config.get("assignment_ties", 0)) for config in run_configs),
        }
    )
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(
        "Merged {} shards into {} mixtures / {} source rows".format(
            len(shard_dirs), args.expected_mixtures, expected_sources
        )
    )


if __name__ == "__main__":
    main()
