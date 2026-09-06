#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path
from typing import List


def read_wav_paths_from_scp(scp_file: str) -> List[str]:
    """
    scp 每行两列：<id> <wav_path>
    取第二列作为 wav 路径。
    """
    wavs: List[str] = []
    with open(scp_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            wavs.append(parts[1])
    return wavs


def main():
    ap = argparse.ArgumentParser(
        description="Generate txt lines: wav1 gain wav2 -gain, sampled from 2-col scp (<id> <path>)."
    )
    ap.add_argument("--scp", required=True, help="Input scp (2 columns: id path).")
    ap.add_argument("--out", required=True, help="Output txt file.")
    ap.add_argument("--num", type=int, default=20000, help="Number of lines. Default=20000")
    ap.add_argument("--max_gain", type=float, default=2.5, help="Gain range: [0, max_gain]. Default=2.5")
    ap.add_argument("--seed", type=int, default=0, help="Random seed. Default=0")
    ap.add_argument("--decimals", type=int, default=5, help="Decimal places for gain. Default=5")
    ap.add_argument("--no_same", action="store_true", help="Ensure wav1 != wav2 for each line.")
    args = ap.parse_args()

    if not Path(args.scp).is_file():
        raise FileNotFoundError("scp not found: {}".format(args.scp))

    wavs = read_wav_paths_from_scp(args.scp)
    if len(wavs) < 2:
        raise RuntimeError("Need at least 2 wav paths, got {}".format(len(wavs)))

    rng = random.Random(args.seed)
    fmt = "{:." + str(args.decimals) + "f}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fo:
        for _ in range(args.num):
            if args.no_same:
                w1, w2 = rng.sample(wavs, 2)  # 保证不同
            else:
                w1 = rng.choice(wavs)
                w2 = rng.choice(wavs)

            g = rng.uniform(0.0, args.max_gain)
            fo.write("{} {} {} {}\n".format(w1, fmt.format(g), w2, fmt.format(-g)))

    print("Done. wavs={}, lines={}, out={}".format(len(wavs), args.num, out_path))


if __name__ == "__main__":
    main()


# python code/gen_mix_list.py --scp scp/all.scp --out metadata/mix_2spk_tr_1.txt --num 20000 --max_gain 2.5 --seed 26 --no_same
# python code/gen_mix_list.py --scp scp/all.scp --out metadata/mix_2spk_cv.txt --num 5000 --max_gain 2.5 --seed 2025 --no_same
# python code/gen_mix_list.py --scp scp/all.scp --out metadata/mix_2spk_tt.txt --num 3000 --max_gain 2.5 --seed 2024 --no_same

# python code/gen_mix_list.py --scp scp/all_new.scp --out metadata/mix_2spk_tt_new.txt --num 3000 --max_gain 2.5 --seed 2024 --no_same