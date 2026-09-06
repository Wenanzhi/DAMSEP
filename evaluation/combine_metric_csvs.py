from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.extended_metrics import ExtendedMetricsTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine non-overlapping extended-metric CSV shards."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser.parse_args()


def finite_reduce(values: Iterable[float], reducer: str) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan")
    if reducer == "mean":
        return float(array.mean())
    if reducer == "std":
        return float(array.std())
    raise ValueError("Unknown reducer: {}".format(reducer))


def read_rows(paths: List[Path]) -> tuple:
    rows: List[Dict[str, str]] = []
    columns = None
    alignment = None
    seen_ids = set()
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if columns is None:
                columns = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != columns:
                raise ValueError("CSV columns differ: {}".format(path))
            for row in reader:
                sample_id = row.get("snt_id", "")
                if sample_id in ("avg", "std"):
                    continue
                if sample_id in seen_ids:
                    raise ValueError(
                        "Duplicate sample id {} in {}".format(sample_id, path)
                    )
                seen_ids.add(sample_id)
                row_alignment = row.get("alignment", "")
                if alignment is None:
                    alignment = row_alignment
                elif row_alignment != alignment:
                    raise ValueError("Alignment differs across shards.")
                rows.append(row)
    if columns is None or alignment is None:
        raise ValueError("No metric rows found.")
    return columns, alignment, rows


def combine(args: argparse.Namespace) -> Path:
    input_paths = [Path(path).expanduser().resolve() for path in args.inputs]
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
    output_path = Path(args.output).expanduser().resolve()
    summary_path = output_path.with_suffix(".summary.json")
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(output_path)

    columns, alignment, rows = read_rows(input_paths)
    if len(rows) != args.expected_count:
        raise ValueError(
            "Expected {} rows, found {}".format(args.expected_count, len(rows))
        )

    summary = {
        "alignment": alignment,
        "num_samples": len(rows),
        "mean": {},
        "std": {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        for row_name, reducer in (("avg", "mean"), ("std", "std")):
            aggregate: Dict[str, object] = {
                "snt_id": row_name,
                "alignment": alignment,
            }
            for metric_name in ExtendedMetricsTracker.metric_columns:
                value = finite_reduce(
                    (float(row[metric_name]) for row in rows), reducer
                )
                aggregate[metric_name] = value
                summary[reducer][metric_name] = value
            writer.writerow(aggregate)

    with summary_path.open("x", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
    return output_path


if __name__ == "__main__":
    print(combine(parse_args()))
