#!/usr/bin/env python3
"""Evaluate source-specific CTF outputs as composed full room responses.

The CTF-to-time-response conversion follows ``inference_rir.py``: an
exponential sine sweep is filtered by the predicted CTF and deconvolved with
the corresponding inverse filter.  The decoded effective response is then
composed with the paired direct-path RIR before comparison with the paired
full reverberant RIR.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy import signal, stats


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = SCRIPT_DIR / "checkpoints" / "dars_mixed_p10"
EPS = np.finfo(np.float64).tiny
SHAPE_METRICS = (
    "rir50_rmse",
    "si_nmse_db",
    "waveform_corr",
    "lsd_db",
    "edc_rmse_db",
)
RECONSTRUCTION_METRICS = (
    "recon_rimag",
    "recon_rimag_normalized",
    "recon_complex_nmse_db",
    "recon_si_sdr_db",
)
BASE_ACOUSTIC_PARAMETERS = (
    "edt_s",
    "t20_s",
    "t30_s",
    "drr_db",
    "c50_db",
    "c80_db",
    "d50_pct",
    "ts_ms",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conf_dir",
        "--conf-dir",
        dest="conf_dir",
        default=str(SCRIPT_DIR / "configs" / "dars.yml"),
    )
    parser.add_argument("--exp_dir", "--exp-dir", dest="exp_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--assignment_mode",
        "--assignment-mode",
        dest="assignment_mode",
        choices=("auto", "fixed_identity", "waveform_pit"),
        default="auto",
        help="Source matching policy; auto follows the training pit_from setting.",
    )
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default=None)
    parser.add_argument("--distance_metadata", "--distance-metadata", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cuda_visible_devices",
        "--cuda-visible-devices",
        dest="cuda_visible_devices",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--max_examples", "--max-examples", dest="max_examples", type=int, default=None)
    parser.add_argument("--tail_seconds", "--tail-seconds", dest="tail_seconds", type=float, default=1.0)
    parser.add_argument(
        "--effective_response_seconds",
        "--effective-response-seconds",
        dest="effective_response_seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--pre_direct_ms", "--pre-direct-ms", dest="pre_direct_ms", type=float, default=2.5
    )
    parser.add_argument("--rir50_ms", "--rir50-ms", dest="rir50_ms", type=float, default=50.0)
    parser.add_argument(
        "--direct_half_ms", "--direct-half-ms", dest="direct_half_ms", type=float, default=2.5
    )
    parser.add_argument(
        "--drr_sensitivity_half_ms",
        "--drr-sensitivity-half-ms",
        dest="drr_sensitivity_half_ms",
        type=float,
        nargs="+",
        default=(1.25, 2.5, 5.0),
    )
    parser.add_argument(
        "--drr_analysis_pre_ms",
        "--drr-analysis-pre-ms",
        dest="drr_analysis_pre_ms",
        type=float,
        default=5.0,
    )
    parser.add_argument("--edc_limit_db", "--edc-limit-db", dest="edc_limit_db", type=float, default=-35.0)
    parser.add_argument("--lsd_floor_db", "--lsd-floor-db", dest="lsd_floor_db", type=float, default=-80.0)
    parser.add_argument(
        "--clarity_min_late_db",
        "--clarity-min-late-db",
        dest="clarity_min_late_db",
        type=float,
        default=-80.0,
    )
    parser.add_argument(
        "--min_decay_r2", "--min-decay-r2", dest="min_decay_r2", type=float, default=0.9
    )
    parser.add_argument(
        "--drr_tie_tolerance_db",
        "--drr-tie-tolerance-db",
        dest="drr_tie_tolerance_db",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--save_example_rirs",
        "--save-example-rirs",
        dest="save_example_rirs",
        type=int,
        default=5,
    )
    parser.add_argument("--print_every", "--print-every", dest="print_every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self_test", "--self-test", dest="self_test", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    if not all(
        math.isfinite(value)
        for value in (
            args.tail_seconds,
            args.effective_response_seconds,
            args.pre_direct_ms,
            args.rir50_ms,
            args.direct_half_ms,
            args.drr_analysis_pre_ms,
            args.edc_limit_db,
            args.lsd_floor_db,
            args.clarity_min_late_db,
            args.min_decay_r2,
            args.drr_tie_tolerance_db,
            *args.drr_sensitivity_half_ms,
        )
    ):
        parser.error("metric arguments must be finite")
    if args.tail_seconds <= 0 or args.effective_response_seconds <= 0:
        parser.error("response durations must be positive")
    if args.pre_direct_ms < 0 or args.direct_half_ms <= 0 or args.rir50_ms <= 0:
        parser.error("time-window arguments are invalid")
    if not 0.0 <= args.min_decay_r2 <= 1.0:
        parser.error("--min-decay-r2 must be between 0 and 1")
    if args.edc_limit_db >= 0.0 or args.lsd_floor_db >= 0.0:
        parser.error("--edc-limit-db and --lsd-floor-db must be negative")
    if args.clarity_min_late_db >= 0.0:
        parser.error("--clarity-min-late-db must be negative")
    if args.drr_tie_tolerance_db < 0.0:
        parser.error("--drr-tie-tolerance-db must be non-negative")
    if any(value <= 0.0 for value in args.drr_sensitivity_half_ms):
        parser.error("--drr-sensitivity-half-ms values must be positive")
    required_drr_pre_ms = max(args.direct_half_ms, *args.drr_sensitivity_half_ms)
    if args.drr_analysis_pre_ms < required_drr_pre_ms:
        parser.error(
            "--drr-analysis-pre-ms must be at least the largest DRR half-window "
            "({} ms)".format(required_drr_pre_ms)
        )
    return args


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def db_ratio(numerator, denominator):
    if numerator < 0.0 or denominator <= 0.0:
        return float("nan")
    return 10.0 * math.log10(max(float(numerator), EPS) / max(float(denominator), EPS))


def sanitize_key(key):
    stem = Path(str(key)).stem
    return stem.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def drr_parameter_name(half_ms):
    token = ("{:g}".format(float(half_ms))).replace(".", "p")
    return "drr_half_{}ms_db".format(token)


def acoustic_parameter_names(sensitivity_half_ms):
    names = list(BASE_ACOUSTIC_PARAMETERS)
    for half_ms in sensitivity_half_ms:
        name = drr_parameter_name(half_ms)
        if name not in names:
            names.append(name)
    return tuple(names)


def apply_ramp(waveform, left_samples, right_samples):
    waveform = np.asarray(waveform, dtype=np.float64).copy()
    waveform[:left_samples] *= np.hanning(left_samples * 2)[:left_samples]
    waveform[-right_samples:] *= np.hanning(right_samples * 2)[:right_samples][::-1]
    return waveform


class SweepRIRDecoder:
    """Pseudo-intrusive CTF decoder matching ``inference_rir.py``."""

    def __init__(self, sample_rate, device):
        import torch
        from torch.nn.functional import pad

        from look2hear.losses.feature import transforms

        self.torch = torch
        self.pad = pad
        self.sample_rate = int(sample_rate)
        self.device = device
        self.tf = transforms(
            sr=sample_rate,
            n_fft=512,
            win_len=256,
            hop_len=128,
            win_type="sqrthann",
        )
        duration = 8.192
        f1 = 62.5
        f2 = sample_rate / 2.0
        w1 = 2.0 * np.pi * f1 / sample_rate
        w2 = 2.0 * np.pi * f2 / sample_rate
        num_samples = int(duration * sample_rate)
        taxis = np.arange(num_samples, dtype=np.float64) / (num_samples - 1)
        log_ratio = np.log(w2 / w1)
        sweep = np.sin(w1 * (num_samples - 1) / log_ratio * (np.exp(taxis * log_ratio) - 1.0))
        envelope = (w2 / w1) ** (-taxis)
        invfilter = np.flipud(sweep) * envelope
        scaling = (
            np.pi
            * num_samples
            * (w1 / w2 - 1.0)
            / (2.0 * (w2 - w1) * np.log(w1 / w2))
        )
        invfilter /= scaling
        sweep = apply_ramp(sweep, 256, 128)
        sweep = np.pad(sweep.astype(np.float32), (512, 512))
        self.invfilter = np.pad(invfilter.astype(np.float32), (512, 512))
        self.sweep_spec = self.tf.stft(torch.from_numpy(sweep).to(device), "complex")
        self._unfolded = {}

    def decode(self, ctf_complex):
        if ctf_complex.ndim != 3:
            raise ValueError(
                "Expected CTF [response, frequency, tap], got {}".format(tuple(ctf_complex.shape))
            )
        _, frequencies, taps = ctf_complex.shape
        if frequencies != self.sweep_spec.shape[0]:
            raise ValueError(
                "CTF/sweep frequency mismatch: {} versus {}".format(
                    frequencies, self.sweep_spec.shape[0]
                )
            )
        if taps not in self._unfolded:
            padded = self.pad(self.sweep_spec, (taps - 1, taps - 1))
            self._unfolded[taps] = padded.unfold(1, taps, 1).unsqueeze(0)
        ctf = ctf_complex.flip(-1).unsqueeze(-1)
        ir_spec = self.torch.matmul(self._unfolded[taps], ctf).squeeze(-1)
        source_irs = self.tf.istft(ir_spec, "complex").detach().cpu().numpy()
        return np.stack(
            [signal.fftconvolve(self.invfilter, source_ir, mode="full") for source_ir in source_irs],
            axis=0,
        )


def decode_complex_ctf(rir_output):
    if rir_output.ndim != 4 or rir_output.shape[1] != 2:
        raise ValueError(
            "Expected model response output [B*S, 2, F, L], got {}".format(
                tuple(rir_output.shape)
            )
        )
    return rir_output[:, 0] + 1j * rir_output[:, 1]


def prepare_effective_response(raw_response, pre_samples, output_samples):
    response = np.asarray(raw_response, dtype=np.float64).reshape(-1)
    if response.size == 0 or not np.all(np.isfinite(response)):
        raise ValueError("Decoded effective response is empty or non-finite")
    peak = int(np.argmax(np.abs(response)))
    start = max(peak - pre_samples, 0)
    response = response[start:]
    direct_index = peak - start
    max_abs = float(np.max(np.abs(response)))
    if max_abs <= 0.0:
        raise ValueError("Decoded effective response has zero energy")
    if max_abs > 1.0:
        response = response / max_abs
    if response.size < output_samples:
        response = np.pad(response, (0, output_samples - response.size))
    else:
        response = response[:output_samples]
    return response, direct_index, peak, max_abs


def crop_around_anchor(waveform, anchor, pre_samples, output_samples):
    waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)
    start = int(anchor) - int(pre_samples)
    left_pad = max(-start, 0)
    start = max(start, 0)
    cropped = waveform[start : start + output_samples - left_pad]
    if left_pad:
        cropped = np.pad(cropped, (left_pad, 0))
    if cropped.size < output_samples:
        cropped = np.pad(cropped, (0, output_samples - cropped.size))
    return cropped[:output_samples]


def compose_full_response(effective_response, direct_response):
    return signal.fftconvolve(
        np.asarray(direct_response, dtype=np.float64),
        np.asarray(effective_response, dtype=np.float64),
        mode="full",
    )


def schroeder_edc_db(rir):
    power = np.square(np.asarray(rir, dtype=np.float64))
    total = float(np.sum(power))
    if total <= 0.0 or not math.isfinite(total):
        return np.full(power.shape, np.nan, dtype=np.float64)
    edc = np.cumsum(power[::-1], dtype=np.float64)[::-1]
    return 10.0 * np.log10(np.maximum(edc / total, EPS))


def fit_decay_time(edc_db, sample_rate, upper_db, lower_db):
    edc_db = np.asarray(edc_db, dtype=np.float64)
    if edc_db.size < 2 or not np.any(edc_db <= lower_db):
        return float("nan"), float("nan")
    mask = np.isfinite(edc_db) & (edc_db <= upper_db) & (edc_db >= lower_db)
    indices = np.flatnonzero(mask)
    if indices.size < max(10, int(round(0.005 * sample_rate))):
        return float("nan"), float("nan")
    x = indices.astype(np.float64) / sample_rate
    y = edc_db[indices]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    x_energy = float(np.dot(x_centered, x_centered))
    y_energy = float(np.dot(y_centered, y_centered))
    cross = float(np.dot(x_centered, y_centered))
    if x_energy <= 0.0:
        return float("nan"), float("nan")
    slope = cross / x_energy
    if not math.isfinite(slope) or slope >= 0.0:
        return float("nan"), float("nan")
    r_squared = cross * cross / (x_energy * y_energy) if y_energy > 0.0 else float("nan")
    return float(-60.0 / slope), float(r_squared)


def direct_to_reverberant_ratio(rir, direct_index, half_samples):
    rir = np.asarray(rir, dtype=np.float64)
    power = np.square(rir)
    start = max(0, int(direct_index) - int(half_samples))
    stop = min(rir.size, int(direct_index) + int(half_samples))
    direct_energy = float(np.sum(power[start:stop]))
    reverb_energy = float(np.sum(power[:start]) + np.sum(power[stop:]))
    return db_ratio(direct_energy, reverb_energy)


def clarity_ratio(early_energy, late_energy, total_energy, min_late_db):
    late_level_db = db_ratio(late_energy, total_energy)
    if not finite(late_level_db) or late_level_db < min_late_db:
        return float("nan")
    return db_ratio(early_energy, late_energy)


def room_parameters(
    rir,
    sample_rate,
    direct_index,
    direct_half_samples,
    sensitivity_half_ms,
    clarity_min_late_db,
):
    rir = np.asarray(rir, dtype=np.float64)
    direct_index = min(max(int(direct_index), 0), max(rir.size - 1, 0))
    power = np.square(rir)
    total = float(np.sum(power))
    if total <= 0.0:
        result = {name: float("nan") for name in BASE_ACOUSTIC_PARAMETERS}
        result.update({"edt_r2": float("nan"), "t20_r2": float("nan"), "t30_r2": float("nan")})
        for half_ms in sensitivity_half_ms:
            result[drr_parameter_name(half_ms)] = float("nan")
        return result
    onset_rir = rir[direct_index:]
    onset_power = power[direct_index:]
    onset_total = float(np.sum(onset_power))
    if onset_total <= 0.0:
        result = {name: float("nan") for name in BASE_ACOUSTIC_PARAMETERS}
        result.update({"edt_r2": float("nan"), "t20_r2": float("nan"), "t30_r2": float("nan")})
        for half_ms in sensitivity_half_ms:
            result[drr_parameter_name(half_ms)] = float("nan")
        return result
    edc_db = schroeder_edc_db(onset_rir)
    edt, edt_r2 = fit_decay_time(edc_db, sample_rate, 0.0, -10.0)
    t20, t20_r2 = fit_decay_time(edc_db, sample_rate, -5.0, -25.0)
    t30, t30_r2 = fit_decay_time(edc_db, sample_rate, -5.0, -35.0)
    n50 = min(onset_power.size, int(round(0.050 * sample_rate)))
    n80 = min(onset_power.size, int(round(0.080 * sample_rate)))
    e50 = float(np.sum(onset_power[:n50]))
    e80 = float(np.sum(onset_power[:n80]))
    late50 = float(np.sum(onset_power[n50:]))
    late80 = float(np.sum(onset_power[n80:]))
    c50 = clarity_ratio(e50, late50, onset_total, clarity_min_late_db)
    c80 = clarity_ratio(e80, late80, onset_total, clarity_min_late_db)
    onset_times = np.arange(onset_power.size, dtype=np.float64) / sample_rate
    result = {
        "edt_s": edt,
        "edt_r2": edt_r2,
        "t20_s": t20,
        "t20_r2": t20_r2,
        "t30_s": t30,
        "t30_r2": t30_r2,
        "drr_db": direct_to_reverberant_ratio(rir, direct_index, direct_half_samples),
        "c50_db": c50,
        "c80_db": c80,
        "d50_pct": (
            float(np.clip(100.0 * e50 / (e50 + late50), 0.0, 100.0))
            if finite(c50)
            else float("nan")
        ),
        "ts_ms": (
            1000.0 * float(np.sum(onset_times * onset_power)) / onset_total
            if onset_total > 0.0
            else float("nan")
        ),
    }
    for half_ms in sensitivity_half_ms:
        half_samples = int(round(float(half_ms) * sample_rate / 1000.0))
        result[drr_parameter_name(half_ms)] = direct_to_reverberant_ratio(
            rir, direct_index, half_samples
        )
    return result


def rir50_rmse(estimate, target, direct_index, sample_rate, window_ms):
    start = int(direct_index)
    stop = min(estimate.size, target.size, start + int(round(window_ms * sample_rate / 1000.0)))
    if stop <= start:
        return float("nan")
    estimate = np.asarray(estimate[start:stop], dtype=np.float64)
    target = np.asarray(target[start:stop], dtype=np.float64)
    estimate_peak = float(np.max(np.abs(estimate)))
    target_peak = float(np.max(np.abs(target)))
    if estimate_peak <= 0.0 or target_peak <= 0.0:
        return float("nan")
    estimate = estimate / estimate_peak
    target = target / target_peak
    if float(np.dot(estimate, target)) < 0.0:
        estimate = -estimate
    return float(np.sqrt(np.mean(np.square(estimate - target))))


def edc_rmse_db(estimate, target, direct_index, limit_db):
    estimate_edc = schroeder_edc_db(np.asarray(estimate)[int(direct_index) :])
    target_edc = schroeder_edc_db(np.asarray(target)[int(direct_index) :])
    if not np.any(target_edc <= limit_db):
        return float("nan")
    mask = np.isfinite(target_edc) & (target_edc >= limit_db)
    if not np.any(mask) or not np.all(np.isfinite(estimate_edc[mask])):
        return float("nan")
    estimate_edc = np.maximum(estimate_edc, -80.0)
    target_edc = np.maximum(target_edc, -80.0)
    return float(np.sqrt(np.mean(np.square(estimate_edc[mask] - target_edc[mask]))))


def log_spectral_distance(estimate, target, sample_rate, floor_db):
    n_fft = 1 << int(math.ceil(math.log2(max(estimate.size, target.size, 2))))
    estimate_magnitude = np.abs(np.fft.rfft(estimate, n=n_fft))
    target_magnitude = np.abs(np.fft.rfft(target, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mask = (frequencies >= 62.5) & (frequencies <= 0.95 * sample_rate / 2.0)
    if not np.any(mask) or np.max(estimate_magnitude) <= 0.0 or np.max(target_magnitude) <= 0.0:
        return float("nan")
    common_floor = np.max(target_magnitude) * 10.0 ** (floor_db / 20.0)
    estimate_db = 20.0 * np.log10(np.maximum(estimate_magnitude, common_floor))
    target_db = 20.0 * np.log10(np.maximum(target_magnitude, common_floor))
    return float(np.sqrt(np.mean(np.square(estimate_db[mask] - target_db[mask]))))


def response_shape_metrics(
    estimate,
    target,
    direct_index,
    sample_rate,
    rir50_ms,
    edc_limit_db,
    lsd_floor_db,
):
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    estimate_energy = float(np.dot(estimate, estimate))
    target_energy = float(np.dot(target, target))
    if estimate_energy <= 0.0 or target_energy <= 0.0:
        return {name: float("nan") for name in SHAPE_METRICS}
    optimal_gain = float(np.dot(target, estimate) / estimate_energy)
    gain_aligned = optimal_gain * estimate
    error_energy = float(np.dot(gain_aligned - target, gain_aligned - target))
    if np.std(gain_aligned) > 0.0 and np.std(target) > 0.0:
        correlation = float(np.corrcoef(gain_aligned, target)[0, 1])
    else:
        correlation = float("nan")
    return {
        "rir50_rmse": rir50_rmse(estimate, target, direct_index, sample_rate, rir50_ms),
        "si_nmse_db": db_ratio(error_energy, target_energy),
        "waveform_corr": correlation,
        "lsd_db": log_spectral_distance(gain_aligned, target, sample_rate, lsd_floor_db),
        "edc_rmse_db": edc_rmse_db(estimate, target, direct_index, edc_limit_db),
    }


def scale_invariant_sdr(estimate, target):
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    length = min(estimate.size, target.size)
    if length == 0:
        return float("nan")
    estimate = estimate[:length]
    target = target[:length]
    target_energy = float(np.dot(target, target))
    if target_energy <= 0.0:
        return float("nan")
    projection = float(np.dot(estimate, target)) * target / target_energy
    noise = estimate - projection
    projection_energy = float(np.dot(projection, projection))
    noise_energy = float(np.dot(noise, noise))
    if projection_energy <= 0.0:
        return float("nan")
    return 10.0 * math.log10(max(projection_energy, EPS) / max(noise_energy, EPS))


def waveform_pit_assignments(estimates, targets):
    """Choose one source permutation from the dereverberated waveform outputs."""
    estimates = estimates.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    if estimates.ndim == 2:
        estimates = estimates[None, ...]
    if targets.ndim == 2:
        targets = targets[None, ...]
    if estimates.ndim != 3 or targets.ndim != 3:
        raise ValueError(
            "Expected waveform tensors [B, S, T], got {} and {}".format(
                estimates.shape, targets.shape
            )
        )
    if estimates.shape[:2] != targets.shape[:2] or estimates.shape[1] != 2:
        raise ValueError(
            "PIT assignment requires matching two-source shapes, got {} and {}".format(
                estimates.shape, targets.shape
            )
        )
    permutations = []
    ties = 0
    for batch_index in range(estimates.shape[0]):
        identity = sum(
            scale_invariant_sdr(estimates[batch_index, source], targets[batch_index, source])
            for source in range(2)
        )
        swapped = scale_invariant_sdr(estimates[batch_index, 1], targets[batch_index, 0]) + scale_invariant_sdr(
            estimates[batch_index, 0], targets[batch_index, 1]
        )
        if not finite(identity) or not finite(swapped):
            raise ValueError("Non-finite waveform PIT score at batch index {}".format(batch_index))
        if abs(identity - swapped) <= 1e-8:
            ties += 1
        permutations.append((0, 1) if identity >= swapped else (1, 0))
    return permutations, ties


def reorder_ctf_sources(ctf, permutations):
    import torch

    if ctf.ndim != 4 or ctf.shape[1] != 2 or len(permutations) != ctf.shape[0]:
        raise ValueError(
            "Expected CTF [B,2,F,L] and one permutation per batch, got {} and {}".format(
                tuple(ctf.shape), len(permutations)
            )
        )
    index = torch.as_tensor(permutations, dtype=torch.long, device=ctf.device)
    return ctf.gather(1, index[:, :, None, None].expand(-1, -1, ctf.shape[2], ctf.shape[3]))


def reconstruction_metrics(clean, reverberant, ctf, transform):
    import torch
    import torchaudio

    clean_spec = transform.stft(clean, "complex").to(torch.complex64)
    target_spec = transform.stft(reverberant, "complex").to(torch.complex64)
    if ctf.shape[:2] != clean_spec.shape[:2] or ctf.shape[2] != clean_spec.shape[2]:
        raise ValueError(
            "Reconstruction shape mismatch: clean {}, CTF {}".format(
                tuple(clean_spec.shape), tuple(ctf.shape)
            )
        )
    reconstructed_spec = torchaudio.functional.convolve(clean_spec, ctf, mode="full")
    reconstructed_spec = reconstructed_spec[..., : clean_spec.shape[-1]]
    difference = reconstructed_spec - target_spec
    reduce_dims = (-2, -1)
    rimag = (
        difference.real.abs() + difference.imag.abs() + (reconstructed_spec.abs() - target_spec.abs()).abs()
    ).mean(dim=reduce_dims)
    rimag_numerator = (
        difference.real.abs() + difference.imag.abs() + (reconstructed_spec.abs() - target_spec.abs()).abs()
    ).sum(dim=reduce_dims)
    rimag_denominator = (
        target_spec.real.abs() + target_spec.imag.abs() + target_spec.abs()
    ).sum(dim=reduce_dims)
    rimag_normalized = rimag_numerator / torch.clamp(
        rimag_denominator, min=torch.finfo(torch.float32).tiny
    )
    error_energy = torch.sum(torch.abs(difference) ** 2, dim=reduce_dims)
    target_energy = torch.sum(torch.abs(target_spec) ** 2, dim=reduce_dims)
    complex_nmse = 10.0 * torch.log10(
        torch.clamp(error_energy, min=torch.finfo(torch.float32).tiny)
        / torch.clamp(target_energy, min=torch.finfo(torch.float32).tiny)
    )
    spec_shape = reconstructed_spec.shape
    flat_spec = reconstructed_spec.reshape(-1, spec_shape[-2], spec_shape[-1])
    reconstructed_waveform = torch.istft(
        flat_spec,
        n_fft=transform.n_fft,
        hop_length=transform.hop_len,
        win_length=transform.win_len,
        window=transform.swin.to(reconstructed_spec.device),
        onesided=True,
        length=clean.shape[-1],
    ).reshape(*spec_shape[:-2], clean.shape[-1])
    reconstructed_waveform = reconstructed_waveform.detach().cpu().numpy()
    target_waveform = reverberant.detach().cpu().numpy()
    si_sdr = np.empty(reconstructed_waveform.shape[:2], dtype=np.float64)
    for batch_index in range(reconstructed_waveform.shape[0]):
        for source_index in range(reconstructed_waveform.shape[1]):
            si_sdr[batch_index, source_index] = scale_invariant_sdr(
                reconstructed_waveform[batch_index, source_index],
                target_waveform[batch_index, source_index],
            )
    return {
        "recon_rimag": rimag.detach().cpu().numpy(),
        "recon_rimag_normalized": rimag_normalized.detach().cpu().numpy(),
        "recon_complex_nmse_db": complex_nmse.detach().cpu().numpy(),
        "recon_si_sdr_db": si_sdr,
    }


def replace_path_component(path, old, new):
    path = Path(path)
    parts = list(path.parts)
    matches = [index for index, part in enumerate(parts) if part == old]
    if len(matches) != 1:
        raise ValueError("Expected one '{}' component in {}".format(old, path))
    parts[matches[0]] = new
    return Path(*parts)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def auto_distance_metadata(test_dir):
    test_dir = Path(test_dir).resolve()
    split = test_dir.name
    for parent in (test_dir,) + tuple(test_dir.parents):
        candidate = parent / "prepared" / "metadata" / "distance_{}.json".format(split)
        if candidate.exists():
            return candidate
    return None


def load_geometry(path):
    if path is None:
        return {}, None
    path = Path(path).expanduser().resolve()
    records = load_json(path)
    mapping = {}
    for record in records:
        key = Path(record["utterance_id"]).name
        mapping[key] = {
            "s1_distance_m": float(record["s1"]["distance_to_left_mic"]),
            "s2_distance_m": float(record["s2"]["distance_to_left_mic"]),
        }
    return mapping, path


def load_config_and_paths(args):
    conf_path = Path(args.conf_dir).expanduser().resolve()
    with conf_path.open("r", encoding="utf-8") as handle:
        conf = yaml.safe_load(handle)
    if args.exp_dir:
        exp_dir = Path(args.exp_dir).expanduser().resolve()
    else:
        saved_exp = conf.get("main_args", {}).get("exp_dir")
        if saved_exp and Path(saved_exp).exists():
            exp_dir = Path(saved_exp).resolve()
        else:
            released_exp = SCRIPT_DIR / "checkpoints" / conf["exp"]["exp_name"]
            training_exp = SCRIPT_DIR / "Experiments" / "checkpoint" / conf["exp"]["exp_name"]
            exp_dir = released_exp if released_exp.exists() else training_exp
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else exp_dir / "best.pth"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else exp_dir / "results" / "rir_metrics_composed"
    )
    if not checkpoint.exists():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint))
    return conf, conf_path, exp_dir, checkpoint, output_dir


def manifest_paths(test_dir):
    return {
        "mixture": Path(test_dir) / "mix_both_reverb.json",
        "anechoic_1": Path(test_dir) / "s1_anechoic.json",
        "anechoic_2": Path(test_dir) / "s2_anechoic.json",
        "reverberant_1": Path(test_dir) / "s1_reverb.json",
        "reverberant_2": Path(test_dir) / "s2_reverb.json",
        "full_rir": Path(test_dir) / "rir_reverb.json",
    }


def load_eval_items(conf, seed, distance_metadata):
    data_conf = conf["datamodule"]["data_config"]
    test_dir = Path(data_conf["test_dir"]).expanduser().resolve()
    paths = manifest_paths(test_dir)
    manifests = {name: load_json(path) for name, path in paths.items()}
    lengths = {name: len(entries) for name, entries in manifests.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError("Manifest lengths differ: {}".format(lengths))
    sample_rate = int(data_conf["sample_rate"])
    segment = data_conf.get("segment")
    segment_samples = None if segment is None else int(round(float(segment) * sample_rate))
    metadata_path = (
        Path(distance_metadata).expanduser().resolve()
        if distance_metadata
        else auto_distance_metadata(test_dir)
    )
    geometry, metadata_path = load_geometry(metadata_path)
    rng = np.random.RandomState(seed)
    items = []
    total = lengths["mixture"]
    names = tuple(manifests)
    for index in range(total):
        entries = {name: manifests[name][index] for name in names}
        basenames = {name: Path(entry[0]).name for name, entry in entries.items()}
        if len(set(basenames.values())) != 1:
            raise ValueError("Manifest basename mismatch at {}: {}".format(index, basenames))
        key = basenames["mixture"]
        num_samples = int(entries["mixture"][1])
        if segment_samples is not None and num_samples < segment_samples:
            continue
        if segment_samples is None or num_samples == segment_samples:
            crop_start = 0
        else:
            crop_start = int(rng.randint(0, num_samples - segment_samples))
        full_rir_path = Path(entries["full_rir"][0])
        direct_rir_path = replace_path_component(full_rir_path, "rir_reverb", "rir_anechoic")
        if not direct_rir_path.exists():
            raise FileNotFoundError("Direct-path RIR does not exist: {}".format(direct_rir_path))
        item = {
            "dataset_index": index,
            "key": key,
            "mix_path": entries["mixture"][0],
            "anechoic_paths": (entries["anechoic_1"][0], entries["anechoic_2"][0]),
            "reverberant_paths": (
                entries["reverberant_1"][0],
                entries["reverberant_2"][0],
            ),
            "full_rir_path": str(full_rir_path),
            "direct_rir_path": str(direct_rir_path),
            "crop_start": crop_start,
            "segment_samples": segment_samples,
        }
        item.update(geometry.get(key, {}))
        items.append(item)
    return items, sample_rate, total, metadata_path


def load_audio_segment(path, start, segment_samples):
    stop = None if segment_samples is None else start + segment_samples
    waveform, sample_rate = sf.read(path, start=start, stop=stop, dtype="float32")
    if waveform.ndim == 2 and waveform.shape[1] == 1:
        waveform = waveform[:, 0]
    if waveform.ndim != 1:
        raise ValueError("Expected mono waveform at {}, got {}".format(path, waveform.shape))
    return waveform, sample_rate


def load_item_audio(item, expected_sample_rate):
    mixture, sample_rate = load_audio_segment(
        item["mix_path"], item["crop_start"], item["segment_samples"]
    )
    if sample_rate != expected_sample_rate:
        raise ValueError("Mixture sample-rate mismatch for {}".format(item["key"]))
    clean = []
    reverberant = []
    for path in item["anechoic_paths"]:
        waveform, source_rate = load_audio_segment(path, item["crop_start"], item["segment_samples"])
        if source_rate != expected_sample_rate:
            raise ValueError("Anechoic sample-rate mismatch for {}".format(item["key"]))
        clean.append(waveform)
    for path in item["reverberant_paths"]:
        waveform, source_rate = load_audio_segment(path, item["crop_start"], item["segment_samples"])
        if source_rate != expected_sample_rate:
            raise ValueError("Reverberant sample-rate mismatch for {}".format(item["key"]))
        reverberant.append(waveform)
    lengths = {mixture.size} | {waveform.size for waveform in clean + reverberant}
    if len(lengths) != 1:
        raise ValueError("Waveform length mismatch for {}: {}".format(item["key"], lengths))
    return mixture, np.stack(clean), np.stack(reverberant)


def load_paired_rirs(item, expected_sample_rate):
    full, full_rate = sf.read(item["full_rir_path"], dtype="float32", always_2d=True)
    direct, direct_rate = sf.read(item["direct_rir_path"], dtype="float32", always_2d=True)
    if full_rate != expected_sample_rate or direct_rate != expected_sample_rate:
        raise ValueError("RIR sample-rate mismatch for {}".format(item["key"]))
    if full.shape[1] != 2 or direct.shape[1] != 2:
        raise ValueError(
            "Expected two source RIRs for {}, got full {} and direct {}".format(
                item["key"], full.shape, direct.shape
            )
        )
    return full.T.astype(np.float64), direct.T.astype(np.float64)


def load_model(conf, checkpoint, sample_rate, device):
    import look2hear.models

    model_class = getattr(look2hear.models, conf["audionet"]["audionet_name"])
    model = model_class.from_pretrain(
        str(checkpoint),
        sample_rate=sample_rate,
        **conf["audionet"]["audionet_config"],
    )
    model.to(device)
    model.eval()
    return model


def source_fieldnames(acoustic_parameters):
    fields = [
        "utterance",
        "dataset_index",
        "crop_start",
        "source",
        "role",
        "distance_m",
        "effective_peak_raw",
        "effective_peak_scale",
        "direct_path_peak",
    ]
    fields.extend(RECONSTRUCTION_METRICS)
    fields.extend(SHAPE_METRICS)
    for name in acoustic_parameters:
        fields.extend(("target_{}".format(name), "est_{}".format(name), "error_{}".format(name)))
    for name in ("edt_r2", "t20_r2", "t30_r2"):
        fields.extend(("target_{}".format(name), "est_{}".format(name)))
    return fields


def mixture_fieldnames():
    return [
        "utterance",
        "dataset_index",
        "crop_start",
        "geometry_near_source",
        "target_drr_near_source",
        "estimated_drr_near_source",
        "target_drr_tie",
        "estimated_drr_tie",
        "target_drr_geometry_agree",
        "near_far_correct",
        "drr_rank_fidelity",
        "target_drr_gap_db",
        "estimated_drr_gap_db",
        "drr_gap_error_db",
        "drr_gap_abs_error_db",
    ]


def completed_keys(source_path, mixture_path):
    if not source_path.exists() and not mixture_path.exists():
        return set()
    if not source_path.exists() or not mixture_path.exists():
        raise RuntimeError("Resume requires both per-source and per-mixture CSV files")
    source_rows = read_csv_rows(source_path)
    mixture_rows = read_csv_rows(mixture_path)
    source_ids = {}
    for row in source_rows:
        source_ids.setdefault(row["utterance"], set()).add(int(float(row["source"])))
    mixture_keys = {row["utterance"] for row in mixture_rows}
    complete = {
        key for key, identifiers in source_ids.items() if identifiers == {1, 2} and key in mixture_keys
    }
    if len(source_rows) != 2 * len(complete) or len(mixture_rows) != len(complete):
        source_by_pair = {}
        for row in source_rows:
            key = (row["utterance"], int(float(row["source"])))
            if row["utterance"] in complete and key not in source_by_pair:
                source_by_pair[key] = row
        mixture_by_key = {}
        for row in mixture_rows:
            if row["utterance"] in complete and row["utterance"] not in mixture_by_key:
                mixture_by_key[row["utterance"]] = row
        rewrite_csv(source_path, list(source_by_pair.values()))
        rewrite_csv(mixture_path, list(mixture_by_key.values()))
    return complete


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_csv_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def rewrite_csv(path, rows):
    with path.open("r", newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    if not fieldnames:
        raise RuntimeError("Cannot resume from CSV without a header: {}".format(path))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def resolve_assignment_mode(conf, requested):
    declared = conf.get("loss", {}).get("train", {}).get("config", {}).get("pit_from")
    if requested == "auto":
        if declared == "no_pit":
            return "fixed_identity"
        if declared in ("pw_mtx", "pit"):
            return "waveform_pit"
        raise ValueError(
            "Cannot infer source assignment from training pit_from={!r}; pass "
            "--assignment-mode explicitly".format(declared)
        )
    if requested == "fixed_identity" and declared != "no_pit":
        raise ValueError(
            "fixed_identity is incompatible with training pit_from={!r}".format(declared)
        )
    if requested == "waveform_pit" and declared == "no_pit":
        raise ValueError("waveform_pit is unnecessary for a no_pit checkpoint")
    return requested


def build_run_signature(
    args, conf, conf_path, checkpoint, sample_rate, distance_path, assignment_mode
):
    test_dir = Path(conf["datamodule"]["data_config"]["test_dir"]).resolve()
    manifests = {
        name: file_fingerprint(path) for name, path in manifest_paths(test_dir).items()
    }
    script_bytes = Path(__file__).resolve().read_bytes()
    return {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "config": file_fingerprint(conf_path),
        "checkpoint": file_fingerprint(checkpoint),
        "manifests": manifests,
        "distance_metadata": file_fingerprint(distance_path) if distance_path else None,
        "sample_rate": sample_rate,
        "segment": conf["datamodule"]["data_config"].get("segment"),
        "seed": args.seed,
        "max_examples": args.max_examples,
        "tail_seconds": args.tail_seconds,
        "effective_response_seconds": args.effective_response_seconds,
        "pre_direct_ms": args.pre_direct_ms,
        "rir50_ms": args.rir50_ms,
        "direct_half_ms": args.direct_half_ms,
        "drr_analysis_pre_ms": args.drr_analysis_pre_ms,
        "drr_sensitivity_half_ms": list(args.drr_sensitivity_half_ms),
        "edc_limit_db": args.edc_limit_db,
        "lsd_floor_db": args.lsd_floor_db,
        "clarity_min_late_db": args.clarity_min_late_db,
        "min_decay_r2": args.min_decay_r2,
        "drr_tie_tolerance_db": args.drr_tie_tolerance_db,
        "training_assignment": conf["loss"]["train"]["config"].get("pit_from"),
        "assignment_mode": assignment_mode,
        "response_composition": "decoded_effective_response_convolved_with_paired_direct_path_rir",
    }


def validate_or_write_run_signature(output_dir, signature, resume):
    path = output_dir / "run_config.json"
    if resume:
        if not path.exists():
            raise RuntimeError("Resume requested but run_config.json is missing in {}".format(output_dir))
        previous = load_json(path)
        if previous != signature:
            raise RuntimeError(
                "Resume configuration does not match the existing run; use a new output directory"
            )
    else:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(signature, handle, indent=2)


def write_status(output_dir, status, **details):
    path = output_dir / "STATUS.json"
    temporary = output_dir / ".STATUS.json.tmp"
    payload = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload.update(details)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def invalidate_derived_outputs(output_dir):
    for name in ("summary.json", "summary.csv", "REPORT.md"):
        path = output_dir / name
        if path.exists():
            os.replace(path, output_dir / (name + ".stale"))


def numeric_values(rows, name):
    values = [to_float(row.get(name)) for row in rows]
    return np.asarray([value for value in values if finite(value)], dtype=np.float64)


def distribution_summary(rows, name):
    values = numeric_values(rows, name)
    if values.size == 0:
        return {
            "n_valid": 0,
            "valid_rate": 0.0,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    return {
        "n_valid": int(values.size),
        "valid_rate": float(values.size / max(len(rows), 1)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def parameter_is_valid(row, side, parameter, min_decay_r2):
    if not finite(row.get("{}_{}".format(side, parameter))):
        return False
    decay_prefix = {"edt_s": "edt", "t20_s": "t20", "t30_s": "t30"}.get(parameter)
    if decay_prefix is None:
        return True
    r2 = row.get("{}_{}_r2".format(side, decay_prefix))
    return finite(r2) and float(r2) >= min_decay_r2


def paired_summary(rows, parameter, min_decay_r2):
    target_name = "target_{}".format(parameter)
    estimate_name = "est_{}".format(parameter)
    target_all = np.asarray(
        [to_float(row[target_name]) for row in rows if parameter_is_valid(row, "target", parameter, min_decay_r2)],
        dtype=np.float64,
    )
    estimate_all = np.asarray(
        [to_float(row[estimate_name]) for row in rows if parameter_is_valid(row, "est", parameter, min_decay_r2)],
        dtype=np.float64,
    )
    pairs = [
        (to_float(row.get(target_name)), to_float(row.get(estimate_name)))
        for row in rows
        if parameter_is_valid(row, "target", parameter, min_decay_r2)
        and parameter_is_valid(row, "est", parameter, min_decay_r2)
    ]
    result = {
        "n_target_valid": int(target_all.size),
        "target_valid_rate": float(target_all.size / max(len(rows), 1)),
        "n_estimate_valid": int(estimate_all.size),
        "estimate_valid_rate": float(estimate_all.size / max(len(rows), 1)),
        "n_valid": len(pairs),
        "valid_rate": float(len(pairs) / max(len(rows), 1)),
        "target_mean": None,
        "estimate_mean": None,
        "bias": None,
        "mae": None,
        "rmse": None,
        "median_abs_error": None,
        "pearson_r": None,
        "spearman_r": None,
    }
    if not pairs:
        return result
    target, estimate = np.asarray(pairs, dtype=np.float64).T
    error = estimate - target
    if target.size >= 2 and np.std(target) > 0.0 and np.std(estimate) > 0.0:
        pearson = float(np.corrcoef(target, estimate)[0, 1])
        spearman_result = stats.spearmanr(target, estimate)
        spearman = float(
            getattr(spearman_result, "statistic", getattr(spearman_result, "correlation", np.nan))
        )
    else:
        pearson = float("nan")
        spearman = float("nan")
    result.update(
        {
            "target_mean": float(np.mean(target)),
            "estimate_mean": float(np.mean(estimate)),
            "bias": float(np.mean(error)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "median_abs_error": float(np.median(np.abs(error))),
            "pearson_r": pearson if finite(pearson) else None,
            "spearman_r": spearman if finite(spearman) else None,
        }
    )
    return result


def paired_columns_summary(rows, target_name, estimate_name):
    pairs = [
        (to_float(row.get(target_name)), to_float(row.get(estimate_name)))
        for row in rows
        if finite(row.get(target_name)) and finite(row.get(estimate_name))
    ]
    result = {
        "n_valid": len(pairs),
        "valid_rate": float(len(pairs) / max(len(rows), 1)),
        "target_mean": None,
        "estimate_mean": None,
        "bias": None,
        "mae": None,
        "rmse": None,
        "median_abs_error": None,
        "pearson_r": None,
        "spearman_r": None,
    }
    if not pairs:
        return result
    target, estimate = np.asarray(pairs, dtype=np.float64).T
    error = estimate - target
    if target.size >= 2 and np.std(target) > 0.0 and np.std(estimate) > 0.0:
        pearson = float(np.corrcoef(target, estimate)[0, 1])
        spearman_result = stats.spearmanr(target, estimate)
        spearman = float(
            getattr(spearman_result, "statistic", getattr(spearman_result, "correlation", np.nan))
        )
    else:
        pearson = float("nan")
        spearman = float("nan")
    result.update(
        {
            "target_mean": float(np.mean(target)),
            "estimate_mean": float(np.mean(estimate)),
            "bias": float(np.mean(error)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "median_abs_error": float(np.median(np.abs(error))),
            "pearson_r": pearson if finite(pearson) else None,
            "spearman_r": spearman if finite(spearman) else None,
        }
    )
    return result


def add_wilson_interval(summary, rows, name):
    values = numeric_values(rows, name)
    if values.size == 0:
        summary["ci95_low"] = None
        summary["ci95_high"] = None
        return summary
    probability = float(np.mean(values))
    count = float(values.size)
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    half_width = z * math.sqrt(
        probability * (1.0 - probability) / count + z * z / (4.0 * count * count)
    ) / denominator
    summary["ci95_low"] = max(0.0, center - half_width)
    summary["ci95_high"] = min(1.0, center + half_width)
    return summary


def correlation_summary(source_rows, response_name):
    pairs = [
        (to_float(row.get("distance_m")), to_float(row.get(response_name)))
        for row in source_rows
        if finite(row.get("distance_m")) and finite(row.get(response_name))
    ]
    if len(pairs) < 2:
        return {"n_valid": len(pairs), "pearson_r": None, "spearman_r": None}
    distance, response = np.asarray(pairs, dtype=np.float64).T
    if np.std(distance) <= 0.0 or np.std(response) <= 0.0:
        return {"n_valid": len(pairs), "pearson_r": None, "spearman_r": None}
    spearman_result = stats.spearmanr(distance, response)
    spearman = float(
        getattr(spearman_result, "statistic", getattr(spearman_result, "correlation", np.nan))
    )
    return {
        "n_valid": len(pairs),
        "pearson_r": float(np.corrcoef(distance, response)[0, 1]),
        "spearman_r": spearman if finite(spearman) else None,
    }


def build_summary(source_rows, mixture_rows, acoustic_parameters, metadata, min_decay_r2):
    groups = {
        "overall": source_rows,
        "source_1": [row for row in source_rows if int(float(row["source"])) == 1],
        "source_2": [row for row in source_rows if int(float(row["source"])) == 2],
        "near": [row for row in source_rows if row.get("role") == "near"],
        "far": [row for row in source_rows if row.get("role") == "far"],
    }
    summary = {"metadata": metadata, "groups": {}}
    for group_name, rows in groups.items():
        summary["groups"][group_name] = {
            "n_rows": len(rows),
            "reconstruction": {
                name: distribution_summary(rows, name) for name in RECONSTRUCTION_METRICS
            },
            "response_shape": {name: distribution_summary(rows, name) for name in SHAPE_METRICS},
            "acoustic_parameters": {
                name: paired_summary(rows, name, min_decay_r2) for name in acoustic_parameters
            },
        }
    summary["mixture_metrics"] = {
        name: distribution_summary(mixture_rows, name)
        for name in (
            "target_drr_geometry_agree",
            "near_far_correct",
            "drr_rank_fidelity",
            "target_drr_tie",
            "estimated_drr_tie",
            "target_drr_gap_db",
            "estimated_drr_gap_db",
            "drr_gap_error_db",
            "drr_gap_abs_error_db",
        )
    }
    for name in (
        "target_drr_geometry_agree",
        "near_far_correct",
        "drr_rank_fidelity",
        "target_drr_tie",
        "estimated_drr_tie",
    ):
        add_wilson_interval(summary["mixture_metrics"][name], mixture_rows, name)
    summary["drr_gap"] = paired_columns_summary(
        mixture_rows, "target_drr_gap_db", "estimated_drr_gap_db"
    )
    summary["distance_relation"] = {
        "target_drr": correlation_summary(source_rows, "target_drr_db"),
        "estimated_drr": correlation_summary(source_rows, "est_drr_db"),
    }
    return summary


def summary_fieldnames():
    return [
        "group",
        "category",
        "metric",
        "n_valid",
        "valid_rate",
        "n_target_valid",
        "target_valid_rate",
        "n_estimate_valid",
        "estimate_valid_rate",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "target_mean",
        "estimate_mean",
        "bias",
        "mae",
        "rmse",
        "median_abs_error",
        "pearson_r",
        "spearman_r",
        "ci95_low",
        "ci95_high",
    ]


def write_summary_csv(summary, path):
    fields = summary_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_name, group in summary["groups"].items():
            for category in ("reconstruction", "response_shape", "acoustic_parameters"):
                for metric, values in group[category].items():
                    row = {"group": group_name, "category": category, "metric": metric}
                    row.update(values)
                    writer.writerow(row)
        for metric, values in summary["mixture_metrics"].items():
            row = {"group": "mixture", "category": "distance_order", "metric": metric}
            row.update(values)
            writer.writerow(row)
        row = {"group": "mixture", "category": "distance_gap", "metric": "drr_gap_db"}
        row.update(summary["drr_gap"])
        writer.writerow(row)


def fmt(value, digits=4):
    return "NA" if value is None or not finite(value) else ("{:.%df}" % digits).format(float(value))


def write_report(summary, path):
    metadata = summary["metadata"]
    overall = summary["groups"]["overall"]
    reconstruction = overall["reconstruction"]
    shape = overall["response_shape"]
    acoustic = overall["acoustic_parameters"]
    mixture = summary["mixture_metrics"]
    drr_gap = summary["drr_gap"]
    lines = [
        "# Composed room-response evaluation",
        "",
        "- Checkpoint: `{}`".format(metadata["checkpoint"]),
        "- Evaluated utterances: {} ({} source responses)".format(
            metadata["evaluated_utterances"], metadata["evaluated_source_rows"]
        ),
        "- Eligible/original manifest utterances: {}/{}".format(
            metadata["eligible_utterances"], metadata["manifest_utterances"]
        ),
        "- Native model output: complex CTF `[B*S, 2, F, L]`.",
        "- Conversion: the exponential-sweep/inverse-filter procedure from `inference_rir.py`.",
        "- Semantic calibration: decoded effective response convolved with paired `rir_anechoic`, then compared with `rir_reverb`.",
        "- Assignment mode: {}; the same source permutation is applied to waveform reconstruction and CTF/RIR metrics.".format(metadata["assignment_mode"]),
        "- C50/C80 validity: late energy must be at least {:.1f} dB relative to total energy.".format(metadata["clarity_min_late_db"]),
        "- DRR tie tolerance: {:.3g} dB; ties are excluded from rank accuracy.".format(metadata["drr_tie_tolerance_db"]),
        "- Main analysis pre-direct guard: {:.3f} ms; post-direct analysis tail: {:.3f} s.".format(metadata["analysis_pre_direct_ms"], metadata["tail_seconds"]),
        "- DRR analysis pre-direct guard: {:.3f} ms; every DRR window uses this same response support.".format(metadata["drr_analysis_pre_ms"]),
        "",
        "## Primary response metrics",
        "",
        "| Metric | Mean | Median | Valid |",
        "|---|---:|---:|---:|",
        "| Reconstruction complex NMSE (dB) | {} | {} | {} |".format(
            fmt(reconstruction["recon_complex_nmse_db"]["mean"]),
            fmt(reconstruction["recon_complex_nmse_db"]["median"]),
            reconstruction["recon_complex_nmse_db"]["n_valid"],
        ),
        "| Reconstruction SI-SDR (dB) | {} | {} | {} |".format(
            fmt(reconstruction["recon_si_sdr_db"]["mean"]),
            fmt(reconstruction["recon_si_sdr_db"]["median"]),
            reconstruction["recon_si_sdr_db"]["n_valid"],
        ),
        "| RIR-50 ms normalized RMSE | {} | {} | {} |".format(
            fmt(shape["rir50_rmse"]["mean"]),
            fmt(shape["rir50_rmse"]["median"]),
            shape["rir50_rmse"]["n_valid"],
        ),
        "| EDC RMSE (dB) | {} | {} | {} |".format(
            fmt(shape["edc_rmse_db"]["mean"]),
            fmt(shape["edc_rmse_db"]["median"]),
            shape["edc_rmse_db"]["n_valid"],
        ),
        "| SI-NMSE (dB) | {} | {} | {} |".format(
            fmt(shape["si_nmse_db"]["mean"]),
            fmt(shape["si_nmse_db"]["median"]),
            shape["si_nmse_db"]["n_valid"],
        ),
        "",
        "## Acoustic parameter estimation",
        "",
        "| Parameter | MAE | RMSE | Pearson | Spearman | Paired valid rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("t20_s", "edt_s", "drr_db", "c50_db", "c80_db"):
        values = acoustic[name]
        lines.append(
            "| {} | {} | {} | {} | {} | {:.1%} |".format(
                name,
                fmt(values["mae"]),
                fmt(values["rmse"]),
                fmt(values["pearson_r"]),
                fmt(values["spearman_r"]),
                values["valid_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Distance-related ordering",
            "",
            "| Metric | Value | Valid |",
            "|---|---:|---:|",
            "| Target DRR / geometry agreement | {} | {} |".format(
                fmt(mixture["target_drr_geometry_agree"]["mean"]),
                mixture["target_drr_geometry_agree"]["n_valid"],
            ),
            "| Estimated DRR near/far accuracy | {} | {} |".format(
                fmt(mixture["near_far_correct"]["mean"]),
                mixture["near_far_correct"]["n_valid"],
            ),
            "| DRR-gap MAE (dB) | {} | {} |".format(
                fmt(drr_gap["mae"]),
                drr_gap["n_valid"],
            ),
            "| Estimated DRR tie rate | {} | {} |".format(
                fmt(mixture["estimated_drr_tie"]["mean"]),
                mixture["estimated_drr_tie"]["n_valid"],
            ),
            "",
            "## Interpretation",
            "",
            "- `t20_s` is a -5 to -25 dB Schroeder EDC fit extrapolated to -60 dB.",
            "- EDT/T20/T30 summary pairs require target and estimate decay-fit R² >= {:.3f}.".format(metadata["min_decay_r2"]),
            "- Main DRR uses a window centered on the direct arrival; sensitivity windows are retained in the CSV.",
            "- Direct arrival is anchored by the paired direct-path RIR, not the largest peak of the full RIR.",
            "- Center time `ts_ms` starts at that direct-arrival anchor; pre-direct interpolation energy is excluded.",
            "- Shape metrics use direct-arrival alignment; RIR-50 ms is independently peak-normalized and polarity-aligned.",
            "- Absolute propagation delay, absolute response gain, and meter-level distance are not evaluated.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def pad_channels(channels):
    length = max(channel.size for channel in channels)
    return np.stack([np.pad(channel, (0, length - channel.size)) for channel in channels])


def save_example(output_dir, key, effective, estimated, target, direct, sample_rate):
    example_dir = output_dir / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_key(key)
    payload = {
        "effective_response": pad_channels(effective),
        "estimated_full_rir": pad_channels(estimated),
        "target_full_rir": pad_channels(target),
        "direct_path_rir": pad_channels(direct),
    }
    for suffix, waveform in payload.items():
        sf.write(
            example_dir / "{}_{}.wav".format(stem, suffix),
            waveform.T.astype(np.float32),
            sample_rate,
            subtype="FLOAT",
        )


def geometry_role(item, source_index):
    first = item.get("s1_distance_m")
    second = item.get("s2_distance_m")
    if first is None or second is None or first == second:
        return "unknown", float("nan")
    near_source = 1 if first < second else 2
    role = "near" if source_index == near_source else "far"
    distance = first if source_index == 1 else second
    return role, float(distance)


def ranked_source(values, tolerance):
    if not all(finite(value) for value in values):
        return None, None
    difference = float(values[0]) - float(values[1])
    if abs(difference) <= tolerance:
        return None, True
    return (1 if difference > 0.0 else 2), False


def build_mixture_row(item, source_rows, tie_tolerance_db):
    distances = [item.get("s1_distance_m"), item.get("s2_distance_m")]
    geometry_near = None
    if all(value is not None for value in distances) and distances[0] != distances[1]:
        geometry_near = 1 if distances[0] < distances[1] else 2
    target_drr = [to_float(row["target_drr_db"]) for row in source_rows]
    estimated_drr = [to_float(row["est_drr_db"]) for row in source_rows]
    target_near, target_tie = ranked_source(target_drr, tie_tolerance_db)
    estimated_near, estimated_tie = ranked_source(estimated_drr, tie_tolerance_db)
    row = {
        "utterance": item["key"],
        "dataset_index": item["dataset_index"],
        "crop_start": item["crop_start"],
        "geometry_near_source": geometry_near if geometry_near is not None else "",
        "target_drr_near_source": target_near if target_near is not None else "",
        "estimated_drr_near_source": estimated_near if estimated_near is not None else "",
        "target_drr_tie": int(target_tie) if target_tie is not None else float("nan"),
        "estimated_drr_tie": (
            int(estimated_tie) if estimated_tie is not None else float("nan")
        ),
        "target_drr_geometry_agree": (
            int(target_near == geometry_near)
            if target_near is not None and geometry_near is not None
            else float("nan")
        ),
        "near_far_correct": (
            int(estimated_near == geometry_near)
            if estimated_near is not None and geometry_near is not None
            else float("nan")
        ),
        "drr_rank_fidelity": (
            int(estimated_near == target_near)
            if estimated_near is not None and target_near is not None
            else float("nan")
        ),
        "target_drr_gap_db": float("nan"),
        "estimated_drr_gap_db": float("nan"),
        "drr_gap_error_db": float("nan"),
        "drr_gap_abs_error_db": float("nan"),
    }
    if geometry_near is not None and all(finite(value) for value in target_drr + estimated_drr):
        near_index = geometry_near - 1
        far_index = 1 - near_index
        target_gap = target_drr[near_index] - target_drr[far_index]
        estimate_gap = estimated_drr[near_index] - estimated_drr[far_index]
        error = estimate_gap - target_gap
        row.update(
            {
                "target_drr_gap_db": target_gap,
                "estimated_drr_gap_db": estimate_gap,
                "drr_gap_error_db": error,
                "drr_gap_abs_error_db": abs(error),
            }
        )
    return row


def evaluate_source(
    item,
    source_index,
    raw_effective,
    full_rir,
    direct_rir,
    reconstruction,
    sample_rate,
    args,
    acoustic_parameters,
):
    decode_pre_samples = int(round(args.pre_direct_ms * sample_rate / 1000.0))
    direct_half_samples = int(round(args.direct_half_ms * sample_rate / 1000.0))
    sensitivity_half_samples = [
        int(round(value * sample_rate / 1000.0))
        for value in args.drr_sensitivity_half_ms
    ]
    analysis_pre_samples = decode_pre_samples
    drr_pre_samples = int(round(args.drr_analysis_pre_ms * sample_rate / 1000.0))
    effective_samples = decode_pre_samples + int(
        round(args.effective_response_seconds * sample_rate)
    )
    output_samples = analysis_pre_samples + int(round(args.tail_seconds * sample_rate))
    effective, effective_direct, effective_peak, effective_scale = prepare_effective_response(
        raw_effective, decode_pre_samples, effective_samples
    )
    direct_peak = int(np.argmax(np.abs(direct_rir)))
    composed = compose_full_response(effective, direct_rir)
    estimate_anchor = direct_peak + effective_direct
    target_anchor = direct_peak
    estimated_aligned = crop_around_anchor(
        composed, estimate_anchor, analysis_pre_samples, output_samples
    )
    target_aligned = crop_around_anchor(
        full_rir, target_anchor, analysis_pre_samples, output_samples
    )
    direct_index = min(analysis_pre_samples, output_samples - 1)
    shape = response_shape_metrics(
        estimated_aligned,
        target_aligned,
        direct_index,
        sample_rate,
        args.rir50_ms,
        args.edc_limit_db,
        args.lsd_floor_db,
    )
    target_parameters = room_parameters(
        target_aligned,
        sample_rate,
        direct_index,
        direct_half_samples,
        (),
        args.clarity_min_late_db,
    )
    estimated_parameters = room_parameters(
        estimated_aligned,
        sample_rate,
        direct_index,
        direct_half_samples,
        (),
        args.clarity_min_late_db,
    )
    drr_output_samples = drr_pre_samples + int(
        round(args.tail_seconds * sample_rate)
    )
    drr_estimated = crop_around_anchor(
        composed,
        estimate_anchor,
        drr_pre_samples,
        drr_output_samples,
    )
    drr_target = crop_around_anchor(
        full_rir,
        target_anchor,
        drr_pre_samples,
        drr_output_samples,
    )
    target_parameters["drr_db"] = direct_to_reverberant_ratio(
        drr_target, drr_pre_samples, direct_half_samples
    )
    estimated_parameters["drr_db"] = direct_to_reverberant_ratio(
        drr_estimated, drr_pre_samples, direct_half_samples
    )
    for half_ms, half_samples in zip(
        args.drr_sensitivity_half_ms, sensitivity_half_samples
    ):
        name = drr_parameter_name(half_ms)
        target_parameters[name] = direct_to_reverberant_ratio(
            drr_target, drr_pre_samples, half_samples
        )
        estimated_parameters[name] = direct_to_reverberant_ratio(
            drr_estimated, drr_pre_samples, half_samples
        )
    role, distance = geometry_role(item, source_index + 1)
    row = {
        "utterance": item["key"],
        "dataset_index": item["dataset_index"],
        "crop_start": item["crop_start"],
        "source": source_index + 1,
        "role": role,
        "distance_m": distance,
        "effective_peak_raw": effective_peak,
        "effective_peak_scale": effective_scale,
        "direct_path_peak": direct_peak,
    }
    for name in RECONSTRUCTION_METRICS:
        row[name] = float(reconstruction[name])
    row.update(shape)
    for name in acoustic_parameters:
        target_value = target_parameters[name]
        estimated_value = estimated_parameters[name]
        row["target_{}".format(name)] = target_value
        row["est_{}".format(name)] = estimated_value
        row["error_{}".format(name)] = (
            estimated_value - target_value
            if finite(target_value) and finite(estimated_value)
            else float("nan")
        )
    for name in ("edt_r2", "t20_r2", "t30_r2"):
        row["target_{}".format(name)] = target_parameters[name]
        row["est_{}".format(name)] = estimated_parameters[name]
    return row, effective, estimated_aligned, target_aligned


def self_test():
    import torch

    sample_rate = 8000
    pre_samples = 20
    direct = np.zeros(128, dtype=np.float64)
    direct[30] = 0.5
    effective = np.zeros(8000, dtype=np.float64)
    effective[pre_samples] = 2.0
    tail = 0.2 * np.exp(-np.arange(1, 1200) / 300.0)
    effective[pre_samples + 1 : pre_samples + 1 + tail.size] = tail
    full = compose_full_response(effective, direct)
    estimated = crop_around_anchor(full, 30 + pre_samples, pre_samples, 8000)
    target = crop_around_anchor(full, 30 + pre_samples, pre_samples, 8000)
    shape = response_shape_metrics(2.5 * estimated, target, pre_samples, sample_rate, 50.0, -35.0, -80.0)
    if shape["rir50_rmse"] > 1e-10 or shape["si_nmse_db"] > -120.0:
        raise AssertionError("Scale-invariant response metrics failed: {}".format(shape))
    target_parameters = room_parameters(
        target, sample_rate, pre_samples, 20, (1.25, 2.5, 5.0), -80.0
    )
    estimated_parameters = room_parameters(
        3.0 * target, sample_rate, pre_samples, 20, (1.25, 2.5, 5.0), -80.0
    )
    for name in ("drr_db", "c50_db", "c80_db", "t20_s"):
        if finite(target_parameters[name]) and abs(target_parameters[name] - estimated_parameters[name]) > 1e-9:
            raise AssertionError("Scale-invariant acoustic parameter failed: {}".format(name))
    if scale_invariant_sdr(target, target) < 100.0:
        raise AssertionError("SI-SDR self-test failed")
    decoder = SweepRIRDecoder(sample_rate, torch.device("cpu"))
    identity_ctf = torch.zeros(2, 257, 60, dtype=torch.complex64)
    identity_ctf[0, :, 0] = 1.0
    identity_ctf[1, :, 1] = 1.0
    decoded = decoder.decode(identity_ctf)
    peaks = [int(np.argmax(np.abs(response))) for response in decoded]
    if peaks[1] - peaks[0] != 128:
        raise AssertionError("CTF tap direction/spacing self-test failed: {}".format(peaks))
    generator = torch.Generator().manual_seed(0)
    clean = torch.randn(1, 2, 4096, generator=generator)
    reconstruction = reconstruction_metrics(clean, clean, identity_ctf[:1].repeat(2, 1, 1).reshape(1, 2, 257, 60), decoder.tf)
    if not np.all(np.isfinite(reconstruction["recon_complex_nmse_db"])):
        raise AssertionError("Reconstruction self-test returned non-finite metrics")
    print("RIR metric self-test passed")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    conf, conf_path, _, checkpoint, output_dir = load_config_and_paths(args)
    train_loss_config = conf.get("loss", {}).get("train", {}).get("config", {})
    training_assignment = train_loss_config.get("pit_from")
    assignment_mode = resolve_assignment_mode(conf, args.assignment_mode)
    if conf.get("datamodule", {}).get("data_config", {}).get("normalize_audio", False):
        raise ValueError(
            "This evaluator currently requires normalize_audio=false so model input and "
            "teacher-forced reconstruction exactly match the tested checkpoint"
        )
    items, sample_rate, manifest_count, distance_path = load_eval_items(
        conf, args.seed, args.distance_metadata
    )
    sample_windows = {
        "tail_seconds": args.tail_seconds * sample_rate,
        "effective_response_seconds": args.effective_response_seconds * sample_rate,
        "rir50_ms": args.rir50_ms * sample_rate / 1000.0,
        "direct_half_ms": args.direct_half_ms * sample_rate / 1000.0,
        "drr_analysis_pre_ms": args.drr_analysis_pre_ms * sample_rate / 1000.0,
    }
    sample_windows.update(
        {
            "drr_sensitivity_half_ms={}".format(value): value * sample_rate / 1000.0
            for value in args.drr_sensitivity_half_ms
        }
    )
    if any(int(round(value)) < 1 for value in sample_windows.values()):
        raise ValueError("Metric windows must span at least one sample: {}".format(sample_windows))
    largest_drr_half_samples = max(
        int(round(args.direct_half_ms * sample_rate / 1000.0)),
        *(int(round(value * sample_rate / 1000.0)) for value in args.drr_sensitivity_half_ms),
    )
    if int(round(args.tail_seconds * sample_rate)) < largest_drr_half_samples:
        raise ValueError("Post-direct tail must cover the largest DRR half-window")
    eligible_count = len(items)
    if args.max_examples is not None:
        items = items[: args.max_examples]
    if not items:
        raise RuntimeError("No eligible evaluation items")
    if items[0]["segment_samples"] is None and args.batch_size != 1:
        raise ValueError("Variable-length full-utterance evaluation requires --batch-size 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    acoustic_parameters = acoustic_parameter_names(args.drr_sensitivity_half_ms)
    source_path = output_dir / "per_source_metrics.csv"
    mixture_path = output_dir / "per_mixture_metrics.csv"
    protected = (
        source_path,
        mixture_path,
        output_dir / "summary.json",
        output_dir / "summary.csv",
        output_dir / "run_config.json",
        output_dir / "REPORT.md",
        output_dir / "STATUS.json",
    )
    if any(path.exists() for path in protected) and not args.resume and not args.overwrite:
        raise FileExistsError(
            "RIR metric outputs already exist in {}; pass --resume or --overwrite".format(output_dir)
        )
    signature = build_run_signature(
        args, conf, conf_path, checkpoint, sample_rate, distance_path, assignment_mode
    )
    if args.overwrite:
        with source_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle, fieldnames=source_fieldnames(acoustic_parameters)
            ).writeheader()
        with mixture_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=mixture_fieldnames()).writeheader()
    validate_or_write_run_signature(output_dir, signature, args.resume)
    completed = completed_keys(source_path, mixture_path) if args.resume else set()
    pending = [item for item in items if item["key"] not in completed]
    if pending:
        invalidate_derived_outputs(output_dir)
        write_status(
            output_dir,
            "running",
            requested_utterances=len(items),
            completed_before_run=len(items) - len(pending),
            pending_utterances=len(pending),
        )
    device = torch.device(args.device)
    model = load_model(conf, checkpoint, sample_rate, device) if pending else None
    decoder = SweepRIRDecoder(sample_rate, device) if pending else None
    source_mode = "a" if args.resume and source_path.exists() else "w"
    mixture_mode = "a" if args.resume and mixture_path.exists() else "w"
    source_handle = source_path.open(source_mode, newline="", encoding="utf-8")
    mixture_handle = mixture_path.open(mixture_mode, newline="", encoding="utf-8")
    source_writer = csv.DictWriter(source_handle, fieldnames=source_fieldnames(acoustic_parameters))
    mixture_writer = csv.DictWriter(mixture_handle, fieldnames=mixture_fieldnames())
    if source_mode == "w":
        source_writer.writeheader()
    if mixture_mode == "w":
        mixture_writer.writeheader()
    started = time.time()
    processed = 0
    assignment_ties = 0
    try:
        with torch.inference_mode():
            for batch_start in range(0, len(pending), args.batch_size):
                batch_items = pending[batch_start : batch_start + args.batch_size]
                loaded = [load_item_audio(item, sample_rate) for item in batch_items]
                mixtures = np.stack([entry[0] for entry in loaded])
                clean = np.stack([entry[1] for entry in loaded])
                reverberant = np.stack([entry[2] for entry in loaded])
                mixture_tensor = torch.from_numpy(mixtures).to(device)
                clean_tensor = torch.from_numpy(clean).to(device)
                reverberant_tensor = torch.from_numpy(reverberant).to(device)
                output = model(mixture_tensor)
                ctf_flat = decode_complex_ctf(output["rir"])
                batch_size = len(batch_items)
                source_count = clean.shape[1]
                if ctf_flat.shape[0] != batch_size * source_count:
                    raise ValueError(
                        "Expected {} CTFs, got {}".format(
                            batch_size * source_count, ctf_flat.shape[0]
                        )
                    )
                ctf = ctf_flat.reshape(batch_size, source_count, *ctf_flat.shape[1:])
                if assignment_mode == "fixed_identity":
                    permutations = [(0, 1)] * batch_size
                else:
                    dereverberated = output["x_derev"]
                    if dereverberated.ndim == 2:
                        dereverberated = dereverberated.reshape(batch_size, source_count, -1)
                    elif dereverberated.ndim != 3:
                        raise ValueError(
                            "Expected x_derev with [B,S,T] or [B*S,T], got {}".format(
                                tuple(dereverberated.shape)
                            )
                        )
                    permutations, ties = waveform_pit_assignments(
                        dereverberated, clean_tensor
                    )
                    assignment_ties += ties
                    ctf = reorder_ctf_sources(ctf, permutations)
                reconstruction = reconstruction_metrics(
                    clean_tensor, reverberant_tensor, ctf, decoder.tf
                )
                raw_effective = decoder.decode(ctf.reshape(batch_size * source_count, *ctf.shape[2:]))
                raw_effective = raw_effective.reshape(batch_size, source_count, -1)
                for local_index, item in enumerate(batch_items):
                    full_rirs, direct_rirs = load_paired_rirs(item, sample_rate)
                    rows = []
                    effective_examples = []
                    estimated_examples = []
                    target_examples = []
                    for source_index in range(source_count):
                        source_reconstruction = {
                            name: reconstruction[name][local_index, source_index]
                            for name in RECONSTRUCTION_METRICS
                        }
                        row, effective, estimated, target = evaluate_source(
                            item,
                            source_index,
                            raw_effective[local_index, source_index],
                            full_rirs[source_index],
                            direct_rirs[source_index],
                            source_reconstruction,
                            sample_rate,
                            args,
                            acoustic_parameters,
                        )
                        rows.append(row)
                        effective_examples.append(effective)
                        estimated_examples.append(estimated)
                        target_examples.append(target)
                    mixture_row = build_mixture_row(
                        item, rows, args.drr_tie_tolerance_db
                    )
                    source_writer.writerows(rows)
                    mixture_writer.writerow(mixture_row)
                    source_handle.flush()
                    mixture_handle.flush()
                    processed += 1
                    if processed <= args.save_example_rirs:
                        save_example(
                            output_dir,
                            item["key"],
                            effective_examples,
                            estimated_examples,
                            target_examples,
                            [direct_rirs[index] for index in range(source_count)],
                            sample_rate,
                        )
                    if args.print_every > 0 and (
                        processed == 1
                        or processed % args.print_every == 0
                        or processed == len(pending)
                    ):
                        elapsed = time.time() - started
                        rate = processed / max(elapsed, EPS)
                        remaining = (len(pending) - processed) / max(rate, EPS)
                        print(
                            "[{}/{}] {:.3f} utterances/s, ETA {:.1f} min".format(
                                processed, len(pending), rate, remaining / 60.0
                            ),
                            flush=True,
                        )
    finally:
        source_handle.close()
        mixture_handle.close()
    source_rows = read_csv_rows(source_path)
    mixture_rows = read_csv_rows(mixture_path)
    evaluated_keys = {row["utterance"] for row in mixture_rows}
    metadata = {
        "checkpoint": str(checkpoint),
        "config": str(conf_path),
        "distance_metadata": str(distance_path) if distance_path else None,
        "sample_rate": sample_rate,
        "manifest_utterances": manifest_count,
        "eligible_utterances": eligible_count,
        "requested_utterances": len(items),
        "evaluated_utterances": len(evaluated_keys),
        "evaluated_source_rows": len(source_rows),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "tail_seconds": args.tail_seconds,
        "effective_response_seconds": args.effective_response_seconds,
        "pre_direct_ms": args.pre_direct_ms,
        "analysis_pre_direct_ms": args.pre_direct_ms,
        "drr_analysis_pre_ms": args.drr_analysis_pre_ms,
        "rir50_ms": args.rir50_ms,
        "direct_half_ms": args.direct_half_ms,
        "drr_sensitivity_half_ms": list(args.drr_sensitivity_half_ms),
        "edc_limit_db": args.edc_limit_db,
        "lsd_floor_db": args.lsd_floor_db,
        "clarity_min_late_db": args.clarity_min_late_db,
        "min_decay_r2": args.min_decay_r2,
        "drr_tie_tolerance_db": args.drr_tie_tolerance_db,
        "device": str(device),
        "cuda_visible_devices": args.cuda_visible_devices,
        "elapsed_seconds_this_run": time.time() - started,
        "training_assignment": training_assignment,
        "assignment_mode": assignment_mode,
        "assignment_ties": assignment_ties,
        "training_w_recon": train_loss_config.get("w_recon"),
        "training_w_rir": train_loss_config.get("w_rir"),
        "ctf_conversion": {
            "sweep_duration_seconds": 8.192,
            "f1_hz": 62.5,
            "f2_hz": sample_rate / 2.0,
            "n_fft": 512,
            "win_length": 256,
            "hop_length": 128,
            "window": "sqrthann",
        },
        "response_semantics": "decoded effective response convolved with paired direct-path RIR",
    }
    summary = build_summary(
        source_rows, mixture_rows, acoustic_parameters, metadata, args.min_decay_r2
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    write_summary_csv(summary, output_dir / "summary.csv")
    write_report(summary, output_dir / "REPORT.md")
    write_status(
        output_dir,
        "complete",
        requested_utterances=len(items),
        evaluated_utterances=len(evaluated_keys),
        evaluated_source_rows=len(source_rows),
    )
    print("Wrote composed RIR metrics to {}".format(output_dir))


if __name__ == "__main__":
    main()
