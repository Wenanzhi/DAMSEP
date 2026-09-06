#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import random
from pathlib import Path


HEADER = [
    "utterance_id",
    "room_x", "room_y", "room_z",
    "micL_x", "micL_y",
    "micR_x", "micR_y",
    "mic_z",
    "s1_x", "s1_y", "s1_z",
    "s2_x", "s2_y", "s2_z",
    "T60",
]


def u(rng: random.Random, a: float, b: float) -> float:
    return rng.uniform(a, b)


def sample_t60(rng: random.Random, mode: str) -> float:
    if mode == "low":
        return u(rng, 0.1, 0.3)
    if mode == "med":
        return u(rng, 0.2, 0.6)
    if mode == "high":
        return u(rng, 0.4, 1.0)
    bucket = rng.choice(["low", "med", "high"])
    return sample_t60(rng, bucket)


def within_room(x: float, y: float, L: float, W: float) -> bool:
    return (0.0 <= x <= L) and (0.0 <= y <= W)

def within_room_margin(x: float, y: float, L: float, W: float, margin: float) -> bool:
    return (margin <= x <= L - margin) and (margin <= y <= W - margin)

def max_horizontal_dist_to_inset_corners(x: float, y: float, L: float, W: float, margin: float) -> float:
    # 以内缩后的“可用区域”四角作为最远点候选，避免far_max鼓励贴墙角
    corners = [
        (margin, margin),
        (L - margin, margin),
        (margin, W - margin),
        (L - margin, W - margin),
    ]
    return max(math.hypot(cx - x, cy - y) for (cx, cy) in corners)


def sample_mic_positions(rng: random.Random, L: float, W: float, mic_sep: float):
    for _ in range(1000):
        cx = L / 2.0 + u(rng, -0.2, 0.2)
        cy = W / 2.0 + u(rng, -0.2, 0.2)
        cz = u(rng, 0.9, 1.8)

        theta = u(rng, 0.0, 2.0 * math.pi)
        dx = (mic_sep / 2.0) * math.cos(theta)
        dy = (mic_sep / 2.0) * math.sin(theta)

        micL_x, micL_y = cx + dx, cy + dy
        micR_x, micR_y = cx - dx, cy - dy

        if within_room(micL_x, micL_y, L, W) and within_room(micR_x, micR_y, L, W):
            return micL_x, micL_y, micR_x, micR_y, cz

    raise RuntimeError("Failed to sample mic positions within room bounds.")


def dist3(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def max_horizontal_dist_to_corners(x: float, y: float, L: float, W: float) -> float:
    corners = [(0.0, 0.0), (L, 0.0), (0.0, W), (L, W)]
    return max(math.hypot(cx - x, cy - y) for (cx, cy) in corners)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_source_with_distance(
    rng: random.Random,
    L: float, W: float,
    ref_x: float, ref_y: float,
    dist_min: float, dist_max: float,
    bias: str,
    z_min: float = 0.9, z_max: float = 1.8,
    max_tries: int = 6000,
    wall_margin: float = 0.0,
):
    if dist_min <= 0:
        dist_min = 1e-6
    if dist_max <= dist_min:
        raise ValueError(f"Invalid distance range: [{dist_min}, {dist_max}]")

    for _ in range(max_tries):
        if bias == "near":
            dist = rng.triangular(dist_min, dist_max, dist_min)
        elif bias == "far":
            dist = rng.triangular(dist_min, dist_max, dist_max)
        else:
            dist = rng.uniform(dist_min, dist_max)

        theta = rng.uniform(0.0, 2.0 * math.pi)
        x = ref_x + dist * math.cos(theta)
        y = ref_y + dist * math.sin(theta)

        # 用“内缩后的可用区域”判断
        if within_room_margin(x, y, L, W, wall_margin):
            z = rng.uniform(z_min, z_max)
            return (x, y, z)

    raise RuntimeError("Failed to sample a source position within room bounds (distance constrained).")


def sample_two_sources_near_far(
    rng: random.Random,
    L: float, W: float,
    mic_x: float, mic_y: float, mic_z: float,
    min_src_sep: float,
    strong_prob: float = 0.8,
    wall_margin: float = 0.3,
    mic_near_min: float = 0.9,
):
    # 合法性：margin不能让可用区域消失
    wall_margin = max(0.0, wall_margin)
    if (L - 2 * wall_margin) <= 0.5 or (W - 2 * wall_margin) <= 0.5:
        # 太夸张就自动缩小一点，避免直接失败
        wall_margin = max(0.0, min(L, W) * 0.05)

    d_corner = max_horizontal_dist_to_inset_corners(mic_x, mic_y, L, W, wall_margin)
    d_max = 0.95 * d_corner
    if d_max <= mic_near_min + 0.5:
        # 可用空间太小就放宽一点（避免死）
        d_max = max(d_max, mic_near_min + 0.8)

    # 近声源：显式保证不太近
    near_min = max(mic_near_min, 0.08 * d_max)
    near_max = min(max(near_min + 0.2, 1.2), 0.35 * d_max)
    if near_max <= near_min:
        near_max = near_min + 0.3

    far_max = d_max

    delta_target = clamp(0.4 * min(L, W), 1.5, 4.5)
    strong = (rng.random() < strong_prob)

    if strong:
        delta_min = min(delta_target, 0.65 * d_max)
        far_min_floor = 0.65 * d_max
    else:
        delta_min = min(max(1.0, 0.25 * min(L, W)), 0.55 * d_max)
        far_min_floor = 0.50 * d_max

    for _ in range(800):
        s1 = sample_source_with_distance(
            rng, L, W, mic_x, mic_y,
            near_min, near_max,
            bias="near",
            wall_margin=wall_margin,
        )

        d1 = dist3(s1, (mic_x, mic_y, mic_z))
        far_min = max(d1 + delta_min, far_min_floor)

        if far_min >= far_max:
            if strong:
                continue
            far_min = max(d1 + 0.6, 0.45 * d_max)
            if far_min >= far_max:
                continue

        for _ in range(6000):
            s2 = sample_source_with_distance(
                rng, L, W, mic_x, mic_y,
                far_min, far_max,
                bias="far",
                wall_margin=wall_margin,
            )

            min_sep_eff = min(min_src_sep, 0.85 * d_max)
            if dist3(s1, s2) >= min_sep_eff:
                d2 = dist3(s2, (mic_x, mic_y, mic_z))
                if d2 > d1:
                    return s1, s2

    raise RuntimeError("Failed to sample near/far sources under constraints.")



def fmt(x: float) -> str:
    return format(x, ".15g")


def collect_wavs_recursive(root: Path):
    wavs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".wav":
            wavs.append(p)
    return sorted(wavs)


def make_utterance_id(wav_path: Path, root: Path, mode: str) -> str:
    if mode == "name":
        return wav_path.name
    rel = wav_path.relative_to(root)
    return rel.as_posix()


def main():
    ap = argparse.ArgumentParser(description="Generate metadata CSV from wav files (recursive) with random room/mic/source params.")
    ap.add_argument("wav_root", type=str, help="Root folder containing subfolders with wav files.")
    ap.add_argument("-o", "--out_csv", type=str, default="metadata.csv", help="Output CSV path.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    ap.add_argument("--mic-sep", type=float, default=0.152, help="Mic separation in meters (default: 0.152).")
    ap.add_argument("--t60-mode", type=str, default="random", choices=["low", "med", "high", "random"],
                    help="T60 sampling mode (default: random among low/med/high).")
    ap.add_argument("--src-min-sep", type=float, default=2.0,
                    help="Minimum separation between s1 and s2 in meters (default: 2.0).")
    ap.add_argument("--utt-id-mode", type=str, default="relative", choices=["relative", "name"],
                    help="utterance_id format: relative path to root (default) or filename only.")
    ap.add_argument("--strong-prob", type=float, default=0.8,
                    help="Probability that a sample enforces strong near/far distance gap (default: 0.8).")
    ap.add_argument("--wall-margin", type=float, default=0.3,
                    help="Minimum distance from sources to walls (default: 0.3m).")
    ap.add_argument("--mic-near-min", type=float, default=0.9,
                    help="Minimum distance from s1 to micL in meters (default: 0.9m).")

    args = ap.parse_args()

    root = Path(args.wav_root)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[ERROR] wav_root not found or not a directory: {root}")

    rng = random.Random(args.seed)

    wav_files = collect_wavs_recursive(root)
    if not wav_files:
        raise SystemExit(f"[ERROR] No .wav files found under: {root}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for wav in wav_files:
            utterance_id = make_utterance_id(wav, root, args.utt_id_mode)

            # Room
            room_x = u(rng, 5.0, 10.0)
            room_y = u(rng, 5.0, 10.0)
            room_z = u(rng, 3.0, 4.0)

            # T60
            T60 = sample_t60(rng, args.t60_mode)

            # Mic
            micL_x, micL_y, micR_x, micR_y, mic_z = sample_mic_positions(rng, room_x, room_y, args.mic_sep)

            # Use the first-channel microphone (micL) as the reference point.
            mic_ref_x, mic_ref_y, mic_ref_z = micL_x, micL_y, mic_z

            # Sources: s1 near, s2 far (80% 强约束)
            (s1_x, s1_y, s1_z), (s2_x, s2_y, s2_z) = sample_two_sources_near_far(
                rng,
                room_x, room_y,
                mic_ref_x, mic_ref_y, mic_ref_z,
                min_src_sep=args.src_min_sep,
                strong_prob=args.strong_prob,
                wall_margin=args.wall_margin,
                mic_near_min=args.mic_near_min,
            )


            writer.writerow([
                utterance_id,
                fmt(room_x), fmt(room_y), fmt(room_z),
                fmt(micL_x), fmt(micL_y),
                fmt(micR_x), fmt(micR_y),
                fmt(mic_z),
                fmt(s1_x), fmt(s1_y), fmt(s1_z),
                fmt(s2_x), fmt(s2_y), fmt(s2_z),
                fmt(T60),
            ])

    print(f"[OK] Found {len(wav_files)} wav files under: {root}")
    print(f"[OK] Wrote CSV to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

# Example (use seeds 2004/2005/2003 for tr/cv/tt, respectively):
# python data/gen_meta_csv.py outputs/wav8k/min/tt/mix \
#   -o outputs/reverb_params_tt.csv --seed 2003 \
#   --src-min-sep 1.5 --strong-prob 0.8 \
#   --wall-margin 0.6 --mic-near-min 0.9
