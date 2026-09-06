from __future__ import annotations

import csv
import itertools
import json
import warnings
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import fast_bss_eval
import numpy as np
import torch
from scipy.signal import resample_poly

try:
    from pesq import pesq as pesq_score
except ImportError:
    pesq_score = None

try:
    from pystoi import stoi as stoi_score
except ImportError:
    stoi_score = None


class DNSMOSP835:
    sample_rate = 16000
    input_length_seconds = 9.01

    def __init__(self, model_path: str, num_threads: int = 8) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "DNSMOS P.835 requires onnxruntime==1.19.2."
            ) from exc

        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.is_file():
            raise FileNotFoundError(
                "DNSMOS P.835 model not found: {}".format(resolved_model)
            )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(num_threads))
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(resolved_model),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_samples = int(
            round(self.input_length_seconds * self.sample_rate)
        )

    @staticmethod
    def _calibrate(raw_scores: np.ndarray) -> np.ndarray:
        calibrated = np.empty_like(raw_scores, dtype=np.float64)
        calibrated[:, 0] = np.polyval(
            [-0.08397278, 1.22083953, 0.0052439], raw_scores[:, 0]
        )
        calibrated[:, 1] = np.polyval(
            [-0.13166888, 1.60915514, -0.39604546], raw_scores[:, 1]
        )
        calibrated[:, 2] = np.polyval(
            [-0.06766283, 1.11546468, 0.04602535], raw_scores[:, 2]
        )
        return calibrated

    def _resample(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == self.sample_rate:
            return audio.astype(np.float32, copy=False)
        common_divisor = gcd(int(source_rate), self.sample_rate)
        return resample_poly(
            audio,
            self.sample_rate // common_divisor,
            int(source_rate) // common_divisor,
        ).astype(np.float32, copy=False)

    def _windows(self, audio: np.ndarray, source_rate: int) -> List[np.ndarray]:
        audio = self._resample(audio, source_rate)
        if audio.size == 0:
            raise ValueError("DNSMOS P.835 cannot score an empty waveform.")
        while audio.size < self.input_samples:
            audio = np.concatenate([audio, audio])

        num_hops = (
            int(np.floor(audio.size / self.sample_rate) - self.input_length_seconds)
            + 1
        )
        windows = []
        for hop_index in range(num_hops):
            start = hop_index * self.sample_rate
            segment = audio[start : start + self.input_samples]
            if segment.size == self.input_samples:
                windows.append(segment)
        if not windows:
            raise RuntimeError(
                "DNSMOS P.835 could not construct a valid 9.01-second window."
            )
        return windows

    def __call__(self, waveforms: np.ndarray, source_rate: int) -> np.ndarray:
        waveforms = np.asarray(waveforms, dtype=np.float32)
        if waveforms.ndim == 1:
            waveforms = waveforms[None, :]
        if waveforms.ndim != 2:
            raise ValueError("DNSMOS P.835 expects [source, time] waveforms.")
        if not np.isfinite(waveforms).all():
            raise ValueError(
                "DNSMOS P.835 received NaN or infinite samples."
            )

        batched_windows = []
        source_indices = []
        for source_index, waveform in enumerate(waveforms):
            windows = self._windows(waveform, int(source_rate))
            batched_windows.extend(windows)
            source_indices.extend([source_index] * len(windows))

        model_input = np.ascontiguousarray(
            np.stack(batched_windows), dtype=np.float32
        )
        raw_scores = np.asarray(
            self.session.run(None, {self.input_name: model_input})[0],
            dtype=np.float64,
        )
        if raw_scores.ndim != 2 or raw_scores.shape[1] != 3:
            raise RuntimeError(
                "Unexpected DNSMOS P.835 output shape: {}".format(
                    raw_scores.shape
                )
            )
        calibrated = self._calibrate(raw_scores)

        source_indices_array = np.asarray(source_indices)
        source_scores = []
        for source_index in range(waveforms.shape[0]):
            source_scores.append(
                calibrated[source_indices_array == source_index].mean(axis=0)
            )
        return np.stack(source_scores)


def _ensure_2d(waveforms: torch.Tensor) -> torch.Tensor:
    if waveforms.ndim == 1:
        return waveforms.unsqueeze(0)
    if waveforms.ndim != 2:
        raise ValueError(
            "Expected [source, time] waveform tensor, got {}".format(
                tuple(waveforms.shape)
            )
        )
    return waveforms


def si_sdr_sources(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    estimate = _ensure_2d(estimate)
    reference = _ensure_2d(reference)
    projection_scale = torch.sum(estimate * reference, dim=-1, keepdim=True)
    projection_scale = projection_scale / (
        torch.sum(reference**2, dim=-1, keepdim=True) + eps
    )
    projected = projection_scale * reference
    residual = estimate - projected
    ratio = torch.sum(projected**2, dim=-1) / (
        torch.sum(residual**2, dim=-1) + eps
    )
    return 10.0 * torch.log10(ratio + eps)


def align_by_si_sdr(
    estimate: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    num_sources = int(reference.shape[0])
    best_score = None
    best_permutation = tuple(range(num_sources))
    for permutation in itertools.permutations(range(num_sources)):
        candidate = estimate[list(permutation)]
        score = float(si_sdr_sources(candidate, reference).mean().item())
        if best_score is None or score > best_score:
            best_score = score
            best_permutation = tuple(int(index) for index in permutation)
    return estimate[list(best_permutation)], best_permutation


class ExtendedMetricsTracker:
    metric_columns = [
        "si_sdr",
        "si_sdr_i",
        "sdr",
        "sdr_i",
        "pesq_nb",
        "stoi",
        "estoi",
        "sir",
        "sar",
        "dnsmos_p835_sig",
        "dnsmos_p835_bak",
        "dnsmos_p835_ovrl",
    ]

    def __init__(
        self,
        save_file: str,
        sample_rate: int,
        alignment: str,
        dnsmos_model_path: str,
        dnsmos_threads: int = 8,
        bss_filter_length: int = 512,
    ) -> None:
        if pesq_score is None:
            raise RuntimeError("PESQ-NB requires pesq==0.0.4.")
        if stoi_score is None:
            raise RuntimeError("STOI/ESTOI requires pystoi==0.4.1.")
        if sample_rate not in (8000, 16000):
            raise ValueError(
                "PESQ-NB supports 8000 or 16000 Hz input, got {}.".format(
                    sample_rate
                )
            )
        if alignment not in ("pit", "fixed"):
            raise ValueError("Unknown alignment mode: {}".format(alignment))

        self.save_file = Path(save_file)
        self.sample_rate = int(sample_rate)
        self.alignment = alignment
        self.bss_filter_length = int(bss_filter_length)
        self.dnsmos = DNSMOSP835(
            dnsmos_model_path, num_threads=dnsmos_threads
        )
        self.rows: List[Dict[str, float]] = []
        self.results_csv = self.save_file.open(
            "x", newline="", encoding="utf-8"
        )
        self.writer = csv.DictWriter(
            self.results_csv,
            fieldnames=["snt_id", "alignment"] + self.metric_columns,
        )
        self.writer.writeheader()
        self.results_csv.flush()

    @staticmethod
    def _finite_mean(values: Iterable[float]) -> float:
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        return float(array.mean()) if array.size else float("nan")

    @staticmethod
    def _finite_std(values: Iterable[float]) -> float:
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        return float(array.std()) if array.size else float("nan")

    @staticmethod
    def _to_numpy(waveforms: torch.Tensor) -> np.ndarray:
        array = waveforms.detach().cpu().float().numpy()
        if not np.isfinite(array).all():
            raise ValueError("Metric input contains NaN or infinite samples.")
        return np.ascontiguousarray(array, dtype=np.float32)

    def _perceptual_scores(
        self,
        clean: np.ndarray,
        estimate: np.ndarray,
        key: str,
    ) -> Tuple[float, float, float]:
        pesq_values = []
        stoi_values = []
        estoi_values = []
        for source_index, (reference, degraded) in enumerate(
            zip(clean, estimate)
        ):
            try:
                pesq_values.append(
                    float(
                        pesq_score(
                            self.sample_rate,
                            reference,
                            degraded,
                            "nb",
                        )
                    )
                )
            except Exception as exc:
                warnings.warn(
                    "PESQ-NB failed for {} source {}: {}".format(
                        key, source_index, exc
                    )
                )
                pesq_values.append(float("nan"))
            try:
                stoi_values.append(
                    float(
                        stoi_score(
                            reference,
                            degraded,
                            self.sample_rate,
                            extended=False,
                        )
                    )
                )
                estoi_values.append(
                    float(
                        stoi_score(
                            reference,
                            degraded,
                            self.sample_rate,
                            extended=True,
                        )
                    )
                )
            except Exception as exc:
                warnings.warn(
                    "STOI/ESTOI failed for {} source {}: {}".format(
                        key, source_index, exc
                    )
                )
                stoi_values.append(float("nan"))
                estoi_values.append(float("nan"))
        return (
            self._finite_mean(pesq_values),
            self._finite_mean(stoi_values),
            self._finite_mean(estoi_values),
        )

    def __call__(
        self,
        mix: torch.Tensor,
        clean: torch.Tensor,
        estimate: torch.Tensor,
        key: str,
    ) -> Dict[str, float]:
        mix = mix.detach().float().cpu()
        clean = _ensure_2d(clean.detach().float().cpu())
        estimate = _ensure_2d(estimate.detach().float().cpu())
        common_length = min(
            mix.shape[-1], clean.shape[-1], estimate.shape[-1]
        )
        mix = mix[..., :common_length]
        clean = clean[..., :common_length]
        estimate = estimate[..., :common_length]

        if self.alignment == "pit":
            aligned_estimate, _ = align_by_si_sdr(estimate, clean)
        else:
            aligned_estimate = estimate
        mix_sources = torch.stack([mix] * clean.shape[0], dim=0)

        si_sdr = si_sdr_sources(aligned_estimate, clean).mean()
        si_sdr_baseline = si_sdr_sources(mix_sources, clean).mean()

        if self.alignment == "pit":
            sdr = -fast_bss_eval.sdr_pit_loss(estimate, clean).mean()
            sdr_baseline = -fast_bss_eval.sdr_pit_loss(
                mix_sources, clean
            ).mean()
        else:
            sdr = -fast_bss_eval.sdr_loss(estimate, clean).mean()
            sdr_baseline = -fast_bss_eval.sdr_loss(
                mix_sources, clean
            ).mean()

        bss_eval_output = fast_bss_eval.bss_eval_sources(
            clean,
            estimate,
            filter_length=self.bss_filter_length,
            zero_mean=False,
            compute_permutation=self.alignment == "pit",
        )
        _, sir, sar = bss_eval_output[:3]

        clean_numpy = self._to_numpy(clean)
        aligned_estimate_numpy = self._to_numpy(aligned_estimate)
        pesq_nb, stoi, estoi = self._perceptual_scores(
            clean_numpy, aligned_estimate_numpy, key
        )
        dnsmos_scores = self.dnsmos(
            aligned_estimate_numpy, self.sample_rate
        )

        row = {
            "snt_id": key,
            "alignment": self.alignment,
            "si_sdr": float(si_sdr.item()),
            "si_sdr_i": float((si_sdr - si_sdr_baseline).item()),
            "sdr": float(sdr.item()),
            "sdr_i": float((sdr - sdr_baseline).item()),
            "pesq_nb": pesq_nb,
            "stoi": stoi,
            "estoi": estoi,
            "sir": self._finite_mean(sir.detach().cpu().numpy()),
            "sar": self._finite_mean(sar.detach().cpu().numpy()),
            "dnsmos_p835_sig": self._finite_mean(dnsmos_scores[:, 0]),
            "dnsmos_p835_bak": self._finite_mean(dnsmos_scores[:, 1]),
            "dnsmos_p835_ovrl": self._finite_mean(dnsmos_scores[:, 2]),
        }
        self.writer.writerow(row)
        self.results_csv.flush()
        self.rows.append(row)
        return row

    def update(self) -> Dict[str, float]:
        return {
            "count": float(len(self.rows)),
            "si_sdr_i": self._finite_mean(
                row["si_sdr_i"] for row in self.rows
            ),
            "sdr_i": self._finite_mean(row["sdr_i"] for row in self.rows),
            "pesq_nb": self._finite_mean(
                row["pesq_nb"] for row in self.rows
            ),
            "stoi": self._finite_mean(row["stoi"] for row in self.rows),
            "dnsmos_p835_ovrl": self._finite_mean(
                row["dnsmos_p835_ovrl"] for row in self.rows
            ),
        }

    def final(self) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "alignment": self.alignment,
            "num_samples": len(self.rows),
            "mean": {},
            "std": {},
        }
        for row_name, reducer in (
            ("avg", self._finite_mean),
            ("std", self._finite_std),
        ):
            output_row: Dict[str, object] = {
                "snt_id": row_name,
                "alignment": self.alignment,
            }
            for metric_name in self.metric_columns:
                value = reducer(
                    row[metric_name] for row in self.rows
                )
                output_row[metric_name] = value
                summary["mean" if row_name == "avg" else "std"][
                    metric_name
                ] = value
            self.writer.writerow(output_row)
        self.results_csv.close()

        summary_path = self.save_file.with_suffix(".summary.json")
        with summary_path.open("x", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)
        return summary

    def close(self) -> None:
        if not self.results_csv.closed:
            self.results_csv.close()
