###
# Author: Kai Li
# Date: 2021-06-22 12:41:36
# LastEditors: Please set LastEditors
# LastEditTime: 2022-06-05 14:48:00
###
import csv
import logging
import warnings

import fast_bss_eval
import numpy as np
import torch

from ..losses import PITLossWrapper_test, PairwiseNegSDR
from .dnsmos_p835 import DNSMOSP835

try:
    from pesq import pesq as pesq_score
except ImportError:
    pesq_score = None

try:
    from pystoi import stoi as stoi_score
except ImportError:
    stoi_score = None

logger = logging.getLogger(__name__)


class MetricsTracker_test:
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
        save_file: str = "",
        sample_rate: int = 8000,
        dnsmos_model_path: str = None,
        dnsmos_threads: int = 8,
        bss_filter_length: int = 512,
    ):
        if pesq_score is None:
            raise RuntimeError("PESQ-NB evaluation requires pesq==0.0.4.")
        if stoi_score is None:
            raise RuntimeError("STOI/ESTOI evaluation requires pystoi==0.4.1.")
        if sample_rate not in [8000, 16000]:
            raise ValueError("PESQ-NB supports only 8000 or 16000 Hz audio.")
        if dnsmos_model_path is None:
            raise ValueError("dnsmos_model_path is required for DNSMOS P.835 evaluation.")

        self.sample_rate = int(sample_rate)
        self.bss_filter_length = int(bss_filter_length)
        self.dnsmos = DNSMOSP835(dnsmos_model_path, num_threads=dnsmos_threads)
        self.rows = []
        self.results_csv = open(save_file, "x", newline="")
        self.writer = csv.DictWriter(
            self.results_csv, fieldnames=["snt_id"] + self.metric_columns
        )
        self.writer.writeheader()
        self.pit_sisdr = PITLossWrapper_test(
            PairwiseNegSDR("sisdr", zero_mean=False), pit_from="pw_mtx"
        )

    @staticmethod
    def _finite_mean(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else np.nan

    @staticmethod
    def _finite_std(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(values.std()) if values.size else np.nan

    @staticmethod
    def _to_numpy(waveforms):
        waveforms = waveforms.detach().cpu().float().numpy()
        if not np.isfinite(waveforms).all():
            raise ValueError("Metric input contains NaN or infinite samples.")
        return np.ascontiguousarray(waveforms, dtype=np.float32)

    def _perceptual_scores(self, clean, estimate, key):
        pesq_values = []
        stoi_values = []
        estoi_values = []
        for source_index, (reference, degraded) in enumerate(zip(clean, estimate)):
            try:
                pesq_values.append(
                    float(pesq_score(self.sample_rate, reference, degraded, "nb"))
                )
            except Exception as exc:
                warnings.warn(
                    "PESQ-NB failed for {} source {}: {}".format(
                        key, source_index, exc
                    )
                )
                pesq_values.append(np.nan)
            try:
                stoi_values.append(
                    float(stoi_score(reference, degraded, self.sample_rate, extended=False))
                )
                estoi_values.append(
                    float(stoi_score(reference, degraded, self.sample_rate, extended=True))
                )
            except Exception as exc:
                warnings.warn(
                    "STOI/ESTOI failed for {} source {}: {}".format(
                        key, source_index, exc
                    )
                )
                stoi_values.append(np.nan)
                estoi_values.append(np.nan)
        return (
            self._finite_mean(pesq_values),
            self._finite_mean(stoi_values),
            self._finite_mean(estoi_values),
        )

    def __call__(self, mix, clean, estimate, key):
        mix = mix.detach().cpu()
        clean = clean.detach().cpu()
        estimate = estimate.detach().cpu()

        common_length = min(mix.shape[-1], clean.shape[-1], estimate.shape[-1])
        mix = mix[..., :common_length]
        clean = clean[..., :common_length]
        estimate = estimate[..., :common_length]

        sisdr_loss, aligned_estimate = self.pit_sisdr(
            estimate.unsqueeze(0), clean.unsqueeze(0), return_ests=True
        )
        mix = torch.stack([mix] * clean.shape[0], dim=0)
        sisdr_baseline_loss = self.pit_sisdr(mix.unsqueeze(0), clean.unsqueeze(0))
        si_sdr = -sisdr_loss
        si_sdr_baseline = -sisdr_baseline_loss

        sdr = -fast_bss_eval.sdr_pit_loss(estimate, clean).mean()
        sdr_baseline = -fast_bss_eval.sdr_pit_loss(mix, clean).mean()
        _, sir, sar, _ = fast_bss_eval.bss_eval_sources(
            clean,
            estimate,
            filter_length=self.bss_filter_length,
            zero_mean=False,
            compute_permutation=True,
        )

        clean_np = self._to_numpy(clean)
        aligned_estimate_np = self._to_numpy(aligned_estimate.squeeze(0))
        pesq_nb, stoi, estoi = self._perceptual_scores(
            clean_np, aligned_estimate_np, key
        )
        dnsmos_scores = self.dnsmos(aligned_estimate_np, self.sample_rate)

        row = {
            "snt_id": key,
            "si_sdr": si_sdr.item(),
            "si_sdr_i": (si_sdr - si_sdr_baseline).item(),
            "sdr": sdr.item(),
            "sdr_i": (sdr - sdr_baseline).item(),
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

    def update(self):
        return {
            "si_sdr_i": self._finite_mean([row["si_sdr_i"] for row in self.rows]),
            "sdr_i": self._finite_mean([row["sdr_i"] for row in self.rows]),
            "pesq_nb": self._finite_mean([row["pesq_nb"] for row in self.rows]),
            "stoi": self._finite_mean([row["stoi"] for row in self.rows]),
            "dnsmos_ovrl": self._finite_mean(
                [row["dnsmos_p835_ovrl"] for row in self.rows]
            ),
        }

    def final(self):
        for row_name, reducer in [
            ("avg", self._finite_mean),
            ("std", self._finite_std),
        ]:
            row = {"snt_id": row_name}
            for metric_name in self.metric_columns:
                row[metric_name] = reducer(
                    [metric_row[metric_name] for metric_row in self.rows]
                )
            self.writer.writerow(row)
        self.results_csv.close()
