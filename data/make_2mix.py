#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly
from tqdm import tqdm
import csv

# =========================
# I/O utils
# =========================
def read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    x, fs = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, fs


def write_wav_int16(path: str, x: np.ndarray, fs: int) -> None:
    x = np.asarray(x, dtype=np.float64)
    # mimic MATLAB int16(round(2^15 * x))
    x = np.clip(x, -1.0, 1.0 - 1.0 / 32768.0)
    y = np.int16(np.round((2**15) * x))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, fs, subtype="PCM_16")


def resample_to(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x.astype(np.float64, copy=False)
    g = math.gcd(fs_in, fs_out)
    up = fs_out // g
    down = fs_in // g
    return resample_poly(x, up=up, down=down).astype(np.float64)


def parse_task_txt(txt_path: str) -> List[Tuple[str, str, str, str]]:
    """
    Each line: abs_path1 gain1 abs_path2 gain2
    gains are strings kept for naming; float(gain) used for computation.
    """
    items = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 4:
                raise ValueError(f"Bad line (need 4 cols): {line}")
            p1, g1, p2, g2 = toks[0], toks[1], toks[2], toks[3]
            items.append((p1, g1, p2, g2))
    return items


def causal_sliding_max(x: np.ndarray, win: int) -> np.ndarray:
    """Causal max over last `win` samples (includes current)."""
    x = np.asarray(x)
    y = np.empty_like(x)
    dq = deque()  # (idx, value)
    for i, v in enumerate(x):
        while dq and dq[-1][1] <= v:
            dq.pop()
        dq.append((i, v))
        while dq and dq[0][0] <= i - win:
            dq.popleft()
        y[i] = dq[0][1]
    return y


# =========================
# P.56 Method B (VOICEBOX-like) active level
# =========================
@dataclass
class P56Config:
    nbin: int = 20
    thresh_db: float = 15.9  # Method B constant
    env_tau: float = 0.03    # 30 ms envelope smoothing
    hangover_s: float = 0.2  # 200 ms hangover
    zpad_s: float = 0.35     # 350 ms tail padding
    flh_default: Tuple[float, float] = (200.0, 5500.0)  # HPF/LPF nominal


def _voicebox_szp(mode: str) -> np.ndarray:
    """
    Precomputed s-domain zeros/poles like VOICEBOX activlev.m
    """
    c25zp = np.array([
        [0.0 + 0.0j, 0.0 + 0.37843443673309j, 0.0 + 0.23388534441447j,
         0.0 - 0.37843443673309j, 0.0 - 0.23388534441447j],
        [-0.66793268833792 + 0.0j,
         -0.20640255179496 + 0.73942185906851j,
         -0.54036889596392 + 0.45698784092898j,
         -0.20640255179496 - 0.73942185906851j,
         -0.54036889596392 - 0.45698784092898j]
    ], dtype=np.complex128)

    c15zp = np.array([
        [0, 0, 0, 0, 0],
        [-2.288586431066945 + 0.0j,
         -0.659002835294875 + 1.195798636925079j,
         -0.123261821596263 + 0.947463030958881j,
         -0.659002835294875 - 1.195798636925079j,
         -0.123261821596263 - 0.947463030958881j]
    ], dtype=np.complex128)

    e5zp = np.array([
        [0.0 + 0.0j, 0.0 + 0.406667680649209j, 0.0 + 0.613849362744881j,
         0.0 - 0.406667680649209j, 0.0 - 0.613849362744881j],
        [-1.964538608244084 + 0.0j,
         -0.538736390607201 + 1.130245082677107j,
         -0.092723126159100 + 0.958193646330194j,
         -0.538736390607201 - 1.130245082677107j,
         -0.092723126159100 - 0.958193646330194j]
    ], dtype=np.complex128)

    if "1" in mode:
        return c15zp
    if "e" in mode:
        return e5zp
    return c25zp


def _design_voicebox_hpf_biquads(fs: int, fc: float, mode: str):
    """
    Return 3 cascaded sections: (b0,a0),(b1,a1),(b2,a2) like activlev.m
    """
    szp = _voicebox_szp(mode)
    t = math.tan(fc * math.pi / fs)
    zl = 2.0 / (1.0 - szp * t) - 1.0

    abl = np.concatenate([
        np.ones((2, 1)),
        -zl[:, [0]],
        -2.0 * np.real(zl[:, 1:3]),
        (np.abs(zl[:, 1:3]) ** 2)
    ], axis=1)  # (2,6)

    v1 = np.array([1, -1, 0, 0, 0, 0], dtype=np.float64)
    v2 = np.array([1, 0, -1, 0, 1, 0], dtype=np.float64)
    v3 = np.array([1, 0, 0, -1, 0, 1], dtype=np.float64)

    hfg = (abl @ v1) * (abl @ v2) * (abl @ v3)
    idx = [0, 1, 0, 2, 4, 0, 3, 5]  # 0-based for MATLAB [1 2 1 3 5 1 4 6]
    abl2 = abl[:, idx].copy()

    scale = (hfg[1] / hfg[0]) if hfg[0] != 0 else 1.0
    abl2[0, 0:2] *= scale

    b0, a0 = np.real(abl2[0, 0:2]), np.real(abl2[1, 0:2])
    b1, a1 = np.real(abl2[0, 2:5]), np.real(abl2[1, 2:5])
    b2, a2 = np.real(abl2[0, 5:8]), np.real(abl2[1, 5:8])
    return (b0, a0, b1, a1, b2, a2)


def _design_voicebox_lpf(fs: int, fc: float, mode: str):
    szp = _voicebox_szp(mode)
    t = math.tan(fc * math.pi / fs)
    zh = 2.0 / (szp / t - 1.0) + 1.0
    a = np.real(np.poly(zh[1, :])).astype(np.float64)
    b = np.real(np.poly(zh[0, :])).astype(np.float64)
    b = b * (np.sum(a) / np.sum(b))  # normalization like MATLAB
    return b, a


def p56_active_level_power(
    x: np.ndarray,
    fs: int,
    mode: str = "n",
    cfg: P56Config = P56Config()
) -> float:
    """
    Compute active speech level in POWER (lp) using P.56 Method B style.
    mode supports: '0' omit HPF, 'h' omit LPF, 'w'/'W' wideband, 'z' no zpad.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    # select band
    flh1, flh2 = cfg.flh_default
    if "w" in mode:
        flh1, flh2 = 70.0, 12000.0
    if "W" in mode:
        flh1, flh2 = 30.0, 18000.0

    # zero-pad unless 'z'
    if "z" not in mode:
        nz = int(math.ceil(cfg.zpad_s * fs))
        x = np.concatenate([x, np.zeros(nz, dtype=np.float64)])

    # filtering
    sq = x
    if "0" not in mode:
        b0, a0, b1, a1, b2, a2 = _design_voicebox_hpf_biquads(fs, flh1, mode)
        sq = lfilter(b0, a0, sq)
        sq = lfilter(b1, a1, sq)
        sq = lfilter(b2, a2, sq)
    if "h" not in mode:
        b, a = _design_voicebox_lpf(fs, flh2, mode)
        sq = lfilter(b, a, sq)

    ns = len(sq)
    ssq = float(np.sum(sq * sq))
    if ns == 0 or ssq <= 0:
        return 0.0

    # envelope smoothing: filter(1, ae, abs(sq))
    ti = 1.0 / fs
    g = math.exp(-ti / cfg.env_tau)
    ae = np.array([1.0, -2.0 * g, g * g], dtype=np.float64) / ((1.0 - g) ** 2)
    s = lfilter([1.0], ae, np.abs(sq))

    # log2(pwr) via frexp
    pwr = s * s
    mant, expo = np.frexp(pwr)  # pwr = mant * 2**expo
    expo = expo.astype(np.float64)
    expo[mant == 0] = -np.inf

    # hangover = causal max over last nh
    nh = int(math.ceil(cfg.hangover_s / ti)) + 1
    expo_h = causal_sliding_max(expo, nh)

    emax = np.max(expo_h)
    if not np.isfinite(emax):
        return 0.0
    emax = int(emax) + 1

    # binning
    q = emax - expo_h
    q[~np.isfinite(q)] = cfg.nbin
    q = np.minimum(q, cfg.nbin).astype(np.int64)
    q[q < 1] = 1
    hist = np.bincount(q - 1, minlength=cfg.nbin).astype(np.float64)
    kc = np.cumsum(hist)

    with np.errstate(divide="ignore", invalid="ignore"):
        aj = 10.0 * np.log10(ssq / kc)
    j = np.arange(1, cfg.nbin + 1, dtype=np.float64)
    cj = 10.0 * np.log10(2.0) * (emax - j - 1.0)
    mj = aj - cj - cfg.thresh_db

    # find crossing
    jj = None
    for idx in range(cfg.nbin - 1):
        if mj[idx] < 0 and mj[idx + 1] >= 0:
            jj = idx
            break
    if jj is None:
        if mj[-1] <= 0:
            jj, jf = cfg.nbin - 2, 1.0
        else:
            jj, jf = 0, 0.0
    else:
        jf = 1.0 / (1.0 - mj[jj + 1] / mj[jj]) if mj[jj] != 0 else 0.0

    lev_db = float(aj[jj] + jf * (aj[jj + 1] - aj[jj]))
    lp = float(10.0 ** (lev_db / 10.0))
    return lp


# =========================
# Cache and mixing
# =========================
def _compute_one_lev(args):
    wav_path, fs_lev, lev_no_lpf = args
    x, fs = read_wav_mono(wav_path)
    x8 = resample_to(x, fs, fs_lev)

    # force disable low-pass if requested: mode includes 'h'
    mode = "hn" if lev_no_lpf else "n"
    lev = p56_active_level_power(x8, fs_lev, mode=mode)
    return wav_path, float(lev)


def build_lev_cache(
    task_txt: str,
    cache_path: str,
    fs_lev: int = 8000,
    lev_no_lpf: bool = True,
    num_workers: int = 0
) -> Dict[str, float]:
    tasks = parse_task_txt(task_txt)
    uniq = sorted(set([t[0] for t in tasks] + [t[2] for t in tasks]))

    cache: Dict[str, float] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    to_do = [p for p in uniq if p not in cache]
    if not to_do:
        return cache

    # compute
    if num_workers is None or num_workers <= 0:
        for p in tqdm(to_do, desc="Compute lev(8k)"):
            k, v = _compute_one_lev((p, fs_lev, lev_no_lpf))
            cache[k] = v
    else:
        nw = min(num_workers, cpu_count())
        with Pool(processes=nw) as pool:
            it = pool.imap_unordered(_compute_one_lev, [(p, fs_lev, lev_no_lpf) for p in to_do])
            for k, v in tqdm(it, total=len(to_do), desc=f"Compute lev(8k) x{nw}"):
                cache[k] = v

    # save
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)
    return cache


def make_mix_name(abs1: str, g1s: str, abs2: str, g2s: str) -> str:
    n1 = os.path.splitext(os.path.basename(abs1))[0]
    n2 = os.path.splitext(os.path.basename(abs2))[0]
    # keep original gain strings to match MATLAB-like naming
    return f"{n1}_{g1s}_{n2}_{g2s}"


def align_minmax(a: np.ndarray, b: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "max":
        L = max(len(a), len(b))
        if len(a) < L:
            a = np.pad(a, (0, L - len(a)))
        if len(b) < L:
            b = np.pad(b, (0, L - len(b)))
        return a, b
    else:
        L = min(len(a), len(b))
        return a[:L], b[:L]


def peak_normalize_triplet(s1: np.ndarray, s2: np.ndarray, mix: np.ndarray, peak: float = 0.9):
    max_amp = float(np.max(np.abs(np.concatenate([s1, s2, mix])))) if len(mix) else 0.0
    scale = (peak / max_amp) if max_amp > 0 else 1.0
    return s1 * scale, s2 * scale, mix * scale, scale


def run_mixing(
    task_txt: str,
    cache_path: str,
    out8k: str,
    out16k: str,
    minmax: str,
    subset: str,
    fs_lev: int = 8000,
    fs_out16k: int = 16000,
    lev_no_lpf: bool = True,
    num_workers: int = 0,
    write_lists: bool = True,
    csv_manifest: str = "",
    strip_prefix: str = "",
):
    assert minmax in ("min", "max")

    cache = build_lev_cache(
        task_txt=task_txt,
        cache_path=cache_path,
        fs_lev=fs_lev,
        lev_no_lpf=lev_no_lpf,
        num_workers=num_workers
    )

    tasks = parse_task_txt(task_txt)

    # list files like MATLAB (optional)
    if write_lists:
        os.makedirs(out8k, exist_ok=True)
        list_s1 = os.path.join(out8k, f"mix_2_spk_{minmax}_{subset}_1")
        list_s2 = os.path.join(out8k, f"mix_2_spk_{minmax}_{subset}_2")
        list_m  = os.path.join(out8k, f"mix_2_spk_{minmax}_{subset}_mix")
        fid_s1 = open(list_s1, "w", encoding="utf-8")
        fid_s2 = open(list_s2, "w", encoding="utf-8")
        fid_m  = open(list_m,  "w", encoding="utf-8")
    else:
        fid_s1 = fid_s2 = fid_m = None

    scaling_8k = []
    scaling_16k = []
    scaling16bit_8k = []
    scaling16bit_16k = []

    csv_f = None
    csv_writer = None
    if csv_manifest:
        os.makedirs(os.path.dirname(csv_manifest) or ".", exist_ok=True)
        csv_f = open(csv_manifest, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["output_filename", "s1_path", "s2_path"])

    def _maybe_strip(p: str) -> str:
        if strip_prefix and p.startswith(strip_prefix):
            p = p[len(strip_prefix):]
            p = p.lstrip("/\\")
        return p


    for abs1, g1s, abs2, g2s in tqdm(tasks, desc=f"Mix({subset},{minmax})"):
        if fid_s1:
            fid_s1.write(abs1 + "\n")
            fid_s2.write(abs2 + "\n")

        mix_name = make_mix_name(abs1, g1s, abs2, g2s)
        if csv_writer:
            csv_writer.writerow([mix_name + ".wav", _maybe_strip(abs1), _maybe_strip(abs2)])

        if fid_m:
            fid_m.write(mix_name + "\n")

        g1 = float(g1s)
        g2 = float(g2s)
        w1 = 10.0 ** (g1 / 20.0)
        w2 = 10.0 ** (g2 / 20.0)

        lev1 = float(cache.get(abs1, 0.0))
        lev2 = float(cache.get(abs2, 0.0))
        slev1 = math.sqrt(lev1) if lev1 > 0 else 1.0
        slev2 = math.sqrt(lev2) if lev2 > 0 else 1.0

        x1, fs1 = read_wav_mono(abs1)
        x2, fs2 = read_wav_mono(abs2)
        if fs1 != fs2:
            raise ValueError(f"Sample rate mismatch: {abs1}({fs1}) vs {abs2}({fs2})")

        # -------- 8k branch (use lev from 8k cache)
        x1_8 = resample_to(x1, fs1, fs_lev) / slev1
        x2_8 = resample_to(x2, fs1, fs_lev) / slev2
        s1_8 = w1 * x1_8
        s2_8 = w2 * x2_8
        s1_8, s2_8 = align_minmax(s1_8, s2_8, minmax)
        mix_8 = s1_8 + s2_8
        s1_8, s2_8, mix_8, mix_scaling_8 = peak_normalize_triplet(s1_8, s2_8, mix_8, peak=0.9)

        # -------- 16k branch: resample to fs_out16k (default 16k), then normalize by SAME lev
        x1_16 = resample_to(x1, fs1, fs_out16k)
        x2_16 = resample_to(x2, fs1, fs_out16k)
        s1_16 = (w1 * x1_16) / slev1
        s2_16 = (w2 * x2_16) / slev2
        s1_16, s2_16 = align_minmax(s1_16, s2_16, minmax)
        mix_16 = s1_16 + s2_16
        s1_16, s2_16, mix_16, mix_scaling_16 = peak_normalize_triplet(s1_16, s2_16, mix_16, peak=0.9)

        # record scaling (match MATLAB meaning)
        scaling_8k.append([w1 * mix_scaling_8 / slev1, w2 * mix_scaling_8 / slev2])
        scaling_16k.append([w1 * mix_scaling_16 / slev1, w2 * mix_scaling_16 / slev2])
        scaling16bit_8k.append(mix_scaling_8)
        scaling16bit_16k.append(mix_scaling_16)

        # write wavs
        p8_s1  = os.path.join(out8k,  minmax, subset, "s1",  mix_name + ".wav")
        p8_s2  = os.path.join(out8k,  minmax, subset, "s2",  mix_name + ".wav")
        p8_mix = os.path.join(out8k,  minmax, subset, "mix", mix_name + ".wav")
        p16_s1  = os.path.join(out16k, minmax, subset, "s1",  mix_name + ".wav")
        p16_s2  = os.path.join(out16k, minmax, subset, "s2",  mix_name + ".wav")
        p16_mix = os.path.join(out16k, minmax, subset, "mix", mix_name + ".wav")

        write_wav_int16(p8_s1,  s1_8,  fs_lev)
        write_wav_int16(p8_s2,  s2_8,  fs_lev)
        write_wav_int16(p8_mix, mix_8, fs_lev)
        write_wav_int16(p16_s1,  s1_16,  fs_out16k)
        write_wav_int16(p16_s2,  s2_16,  fs_out16k)
        write_wav_int16(p16_mix, mix_16, fs_out16k)

    if fid_s1:
        fid_s1.close(); fid_s2.close(); fid_m.close()

    # save scaling
    os.makedirs(os.path.join(out8k, minmax, subset), exist_ok=True)
    os.makedirs(os.path.join(out16k, minmax, subset), exist_ok=True)
    np.savez(
        os.path.join(out8k, minmax, subset, "scaling.npz"),
        scaling_8k=np.array(scaling_8k, dtype=np.float64),
        scaling16bit_8k=np.array(scaling16bit_8k, dtype=np.float64),
    )
    if csv_f:
        csv_f.close()

    np.savez(
        os.path.join(out16k, minmax, subset, "scaling.npz"),
        scaling_16k=np.array(scaling_16k, dtype=np.float64),
        scaling16bit_16k=np.array(scaling16bit_16k, dtype=np.float64),
    )


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="Create 2-speaker mixtures with P.56 active level normalization (VOICEBOX-like)."
    )
    ap.add_argument("--task_txt", type=str, required=True,
                    help="Path to mix_2_spk_*.txt (absolute wav paths).")
    ap.add_argument("--cache", type=str, default="lev_cache.json",
                    help="Path to lev cache JSON (abs_path -> lev_power).")
    ap.add_argument("--out8k", type=str, default="wav8k",
                    help="Output root for 8k wavs.")
    ap.add_argument("--out16k", type=str, default="wav16k",
                    help="Output root for 16k wavs.")
    ap.add_argument("--subset", type=str, default="tr",
                    help="Subset name used in output folder (e.g., tr/cv/tt).")
    ap.add_argument("--minmax", type=str, choices=["min", "max"], default="min",
                    help="Length alignment mode.")
    ap.add_argument("--fs_lev", type=int, default=8000,
                    help="Sampling rate used to compute lev (must be 8000 to match MATLAB flow).")
    ap.add_argument("--fs_out16k", type=int, default=16000,
                    help="Sampling rate for 16k branch output (default 16000).")
    ap.add_argument("--lev_no_lpf", type=int, default=1,
                    help="Force disable low-pass in lev computation (1=yes, 0=no). Recommended 1 for MATLAB-like 8k behavior.")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="Workers for lev precompute (0=single process).")
    ap.add_argument("--only_cache", action="store_true",
                    help="Only build lev cache, do not mix.")
    ap.add_argument("--no_lists", action="store_true",
                    help="Do not write mix_2_spk_* list files.")
    ap.add_argument("--csv_manifest", type=str, default="",
                help="Optional CSV manifest path. If empty, no CSV is written.")
    ap.add_argument("--strip_prefix", type=str, default="",
                help="Optional prefix to strip from s1/s2 paths when writing CSV (e.g., /path/to/wsj0_root/).")

    args = ap.parse_args()

    # sanity: enforce 8k lev as requested
    if args.fs_lev != 8000:
        raise ValueError("To match your requirement, please set --fs_lev 8000 (default).")

    if args.only_cache:
        build_lev_cache(
            task_txt=args.task_txt,
            cache_path=args.cache,
            fs_lev=args.fs_lev,
            lev_no_lpf=bool(args.lev_no_lpf),
            num_workers=args.num_workers
        )
        print(f"[OK] lev cache saved: {args.cache}")
        return

    run_mixing(
        task_txt=args.task_txt,
        cache_path=args.cache,
        out8k=args.out8k,
        out16k=args.out16k,
        minmax=args.minmax,
        subset=args.subset,
        fs_lev=args.fs_lev,
        fs_out16k=args.fs_out16k,
        lev_no_lpf=bool(args.lev_no_lpf),
        num_workers=args.num_workers,
        write_lists=(not args.no_lists),
        csv_manifest=args.csv_manifest,
        strip_prefix=args.strip_prefix,
    )
    print("[OK] Done.")


if __name__ == "__main__":
    main()

# Build the active-level cache, then generate a split:
# python data/make_2mix.py \
#   --task_txt outputs/pairs.txt \
#   --cache outputs/active_levels.json \
#   --only_cache --num_workers 8
#
# python data/make_2mix.py \
#   --task_txt outputs/pairs.txt \
#   --cache outputs/active_levels.json \
#   --out8k outputs/wav8k --out16k outputs/wav16k \
#   --subset tt --minmax min --num_workers 8 \
#   --csv_manifest outputs/mix_2_spk_filenames_tt.csv
