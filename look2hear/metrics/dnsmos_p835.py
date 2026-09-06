from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


class DNSMOSP835:
    sample_rate = 16000
    input_length_seconds = 9.01

    def __init__(self, model_path, num_threads=8):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "DNSMOS P.835 requires onnxruntime. Install onnxruntime==1.19.2."
            ) from exc

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError("DNSMOS P.835 model not found: {}".format(model_path))

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, int(num_threads))
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_samples = int(round(self.input_length_seconds * self.sample_rate))

    @staticmethod
    def _calibrate(raw_scores):
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

    def _resample(self, audio, source_rate):
        if source_rate == self.sample_rate:
            return audio.astype(np.float32, copy=False)
        common_divisor = gcd(int(source_rate), self.sample_rate)
        return resample_poly(
            audio,
            self.sample_rate // common_divisor,
            int(source_rate) // common_divisor,
        ).astype(np.float32, copy=False)

    def _windows(self, audio, source_rate):
        audio = self._resample(audio, source_rate)
        if audio.size == 0:
            raise ValueError("DNSMOS P.835 cannot score an empty waveform.")
        while audio.size < self.input_samples:
            audio = np.concatenate([audio, audio])

        num_hops = int(np.floor(audio.size / self.sample_rate) - self.input_length_seconds) + 1
        windows = []
        for hop_index in range(num_hops):
            start = hop_index * self.sample_rate
            segment = audio[start : start + self.input_samples]
            if segment.size == self.input_samples:
                windows.append(segment)
        if not windows:
            raise RuntimeError("DNSMOS P.835 could not construct a valid 9.01-second window.")
        return windows

    def __call__(self, waveforms, source_rate):
        waveforms = np.asarray(waveforms, dtype=np.float32)
        if waveforms.ndim == 1:
            waveforms = waveforms[None, :]
        if waveforms.ndim != 2:
            raise ValueError("DNSMOS P.835 expects [source, time] waveforms.")
        if not np.isfinite(waveforms).all():
            raise ValueError("DNSMOS P.835 received NaN or infinite samples.")

        batched_windows = []
        source_indices = []
        for source_index, waveform in enumerate(waveforms):
            windows = self._windows(waveform, int(source_rate))
            batched_windows.extend(windows)
            source_indices.extend([source_index] * len(windows))

        model_input = np.ascontiguousarray(np.stack(batched_windows), dtype=np.float32)
        raw_scores = np.asarray(
            self.session.run(None, {self.input_name: model_input})[0], dtype=np.float64
        )
        if raw_scores.ndim != 2 or raw_scores.shape[1] != 3:
            raise RuntimeError(
                "Unexpected DNSMOS P.835 output shape: {}".format(raw_scores.shape)
            )
        calibrated = self._calibrate(raw_scores)

        source_indices = np.asarray(source_indices)
        source_scores = []
        for source_index in range(waveforms.shape[0]):
            source_scores.append(calibrated[source_indices == source_index].mean(axis=0))
        return np.stack(source_scores)
