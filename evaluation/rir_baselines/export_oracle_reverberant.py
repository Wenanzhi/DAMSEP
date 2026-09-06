#!/usr/bin/env python3
"""Export ground-truth reverberant source images for blind RIR estimators."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-every", type=int, default=250)
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.target_sample_rate <= 0:
        parser.error("--target-sample-rate must be positive")
    if args.max_sources is not None and args.max_sources <= 0:
        parser.error("--max-sources must be positive")
    return args


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Empty source manifest: {}".format(path))
    required = {
        "utterance",
        "dataset_index",
        "crop_start",
        "source",
        "input_path",
        "reverberant_target_path",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("Missing manifest fields: {}".format(sorted(missing)))
    return rows


def resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(
        waveform,
        target_rate // divisor,
        source_rate // divisor,
    ).astype(np.float32)


def expected_output_frames(row: dict[str, str], target_rate: int) -> int:
    reference = Path(row["input_path"]).expanduser().resolve()
    info = sf.info(str(reference))
    duration = info.frames / float(info.samplerate)
    frames = int(round(duration * target_rate))
    if frames <= 0:
        raise ValueError("Invalid reference input duration: {}".format(reference))
    return frames


def load_oracle_source(
    row: dict[str, str], target_rate: int, output_frames: int
) -> tuple[np.ndarray, int, int, int]:
    path = Path(row["reverberant_target_path"]).expanduser().resolve()
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    source = int(row["source"])
    if waveform.shape[1] == 1:
        channel = 0
    elif 1 <= source <= waveform.shape[1]:
        channel = source - 1
    else:
        raise ValueError(
            "Source {} unavailable in {}-channel target {}".format(
                source, waveform.shape[1], path
            )
        )
    waveform = waveform[:, channel]
    source_frames = int(round(output_frames * sample_rate / float(target_rate)))
    if waveform.size == source_frames:
        crop_start = 0
    else:
        crop_start = int(row["crop_start"])
    crop_end = crop_start + source_frames
    if crop_start < 0 or crop_end > waveform.size:
        raise ValueError(
            "Crop [{}, {}) exceeds {} frames for {}".format(
                crop_start, crop_end, waveform.size, path
            )
        )
    cropped = waveform[crop_start:crop_end]
    exported = resample(cropped, sample_rate, target_rate)
    if exported.size > output_frames:
        exported = exported[:output_frames]
    elif exported.size < output_frames:
        exported = np.pad(exported, (0, output_frames - exported.size))
    if not np.all(np.isfinite(exported)) or float(np.max(np.abs(exported))) <= 1e-8:
        raise ValueError("Silent or invalid oracle reverberant input: {}".format(path))
    return exported.astype(np.float32), sample_rate, crop_start, channel


def write_manifest(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    source_manifest = args.source_manifest.expanduser().resolve()
    rows = read_manifest(source_manifest)
    if args.max_sources is not None:
        rows = rows[: args.max_sources]

    output_dir = args.output_dir.expanduser().resolve()
    audio_dir = output_dir / "wav16k"
    manifest_path = output_dir / "manifest.csv"
    config_path = output_dir / "run_config.json"
    if output_dir.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            "{} exists; pass --resume or --overwrite".format(output_dir)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        manifest_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)

    fieldnames = list(rows[0])
    input_index = fieldnames.index("input_path")
    fieldnames[input_index:input_index] = [
        "estimated_input_path",
        "oracle_target_channel",
        "oracle_crop_start",
        "oracle_native_sample_rate",
    ]
    output_rows: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, row in enumerate(rows, 1):
        filename = Path(row["input_path"]).name
        if filename in seen_names:
            raise ValueError("Duplicate input basename: {}".format(filename))
        seen_names.add(filename)
        output_path = audio_dir / filename
        output_frames = expected_output_frames(row, args.target_sample_rate)
        exported, native_rate, crop_start, target_channel = load_oracle_source(
            row, args.target_sample_rate, output_frames
        )
        if not (args.resume and output_path.exists()):
            sf.write(
                output_path,
                exported,
                args.target_sample_rate,
                subtype="FLOAT",
            )
        saved_info = sf.info(str(output_path))
        if (
            saved_info.samplerate != args.target_sample_rate
            or saved_info.frames != output_frames
            or saved_info.channels != 1
        ):
            raise ValueError("Exported audio header mismatch: {}".format(output_path))
        output_row: dict[str, object] = dict(row)
        output_row["estimated_input_path"] = row["input_path"]
        output_row["oracle_target_channel"] = target_channel
        output_row["oracle_crop_start"] = crop_start
        output_row["oracle_native_sample_rate"] = native_rate
        output_row["input_path"] = str(output_path)
        output_rows.append(output_row)
        if index % args.print_every == 0 or index == len(rows):
            print("Exported {}/{} source inputs".format(index, len(rows)), flush=True)

    write_manifest(manifest_path, output_rows, fieldnames)
    metadata = {
        "source_manifest": str(source_manifest),
        "input_semantics": "ground-truth reverberant source image",
        "waveform_source": "reverberant_target_path",
        "target_sample_rate": args.target_sample_rate,
        "source_rows": len(output_rows),
        "mixtures": len({row["utterance"] for row in output_rows}),
        "no_additional_normalization": True,
        "crop_rule": (
            "reuse source-manifest crop_start for long mono targets; use the "
            "saved full segment for already-cropped multichannel measured targets"
        ),
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("Wrote {} and {}".format(manifest_path, config_path))


if __name__ == "__main__":
    main()
