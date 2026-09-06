import argparse
import csv
import os
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

try:
    import fast_bss_eval
except Exception:
    fast_bss_eval = None

try:
    from pesq import pesq as pesq_fn
except Exception:
    pesq_fn = None

try:
    from pystoi import stoi as stoi_fn
except Exception:
    stoi_fn = None


SCRIPT_DIR = Path(__file__).resolve().parent
EPS = 1e-8

METRIC_COLUMNS = [
    "snt_id",
    "sdr",
    "sdr_i",
    "si_sdr",
    "si_sdr_i",
    "snr",
    "snr_i",
    "sd_sdr",
    "sd_sdr_i",
    "sir",
    "sir_i",
    "sar",
    "sar_i",
    "stoi",
    "stoi_i",
    "estoi",
    "estoi_i",
    "pesq_nb",
    "pesq_nb_i",
    "rmse",
    "mae",
    "lsd",
    "spectral_convergence",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default=str(SCRIPT_DIR / "configs" / "dars.yml"),
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Override experiment directory. Defaults to Experiments/checkpoint/<exp_name>.",
    )
    parser.add_argument(
        "--output_name",
        default="metrics_more.csv",
        help="CSV name under the experiment results directory.",
    )
    parser.add_argument(
        "--save_file",
        default=None,
        help="Full CSV output path. Overrides --output_name.",
    )
    parser.add_argument(
        "--pit_from",
        choices=["pw_mtx", "no_pit"],
        default="no_pit",
        help="Use SI-SDR PIT alignment or keep the original source order.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Evaluate only the first N examples. Useful for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cuda, cuda:0, or cpu. Defaults to config/CUDA availability.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default="0",
        help="Value assigned to CUDA_VISIBLE_DEVICES before model execution.",
    )
    parser.add_argument(
        "--skip_stoi",
        action="store_true",
        help="Skip STOI and ESTOI even when pystoi is installed.",
    )
    parser.add_argument(
        "--skip_pesq",
        action="store_true",
        help="Skip PESQ even when pesq is installed.",
    )
    parser.add_argument("--bss_filter_length", type=int, default=512)
    parser.add_argument("--bss_use_cg_iter", type=int, default=None)
    parser.add_argument(
        "--clamp_db",
        type=float,
        default=100.0,
        help="Clamp dB-valued metrics to [-clamp_db, clamp_db]. Use a negative value to disable.",
    )
    parser.add_argument("--stft_n_fft", type=int, default=512)
    parser.add_argument("--stft_win_length", type=int, default=256)
    parser.add_argument("--stft_hop_length", type=int, default=128)
    return parser.parse_args()


def configure_runtime_env(cuda_visible_devices=None):
    cache_dir = SCRIPT_DIR / ".cache"
    numba_cache_dir = cache_dir / "numba"
    matplotlib_cache_dir = cache_dir / "matplotlib"
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def resolve_path(path_str):
    path = Path(path_str).expanduser()
    if path.exists():
        return path
    alt_path = SCRIPT_DIR / path_str
    if alt_path.exists():
        return alt_path
    raise FileNotFoundError("Could not find path: {}".format(path_str))


def safe_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean()
        return float(value.cpu().item())
    return float(value)


def finite_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def finite_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.std(arr))


def clamp_scores(scores, clamp_db):
    if clamp_db is None:
        return scores
    return torch.clamp(scores, min=-clamp_db, max=clamp_db)


def as_sources_2d(wav, name):
    if wav.ndim == 1:
        return wav.unsqueeze(0)
    if wav.ndim == 2:
        return wav
    if wav.ndim == 3 and wav.shape[0] == 1:
        return wav.squeeze(0)
    raise ValueError("{} must have shape [time], [src, time], or [1, src, time].".format(name))


def as_mix_1d(mix):
    if mix.ndim == 1:
        return mix
    if mix.ndim == 2 and mix.shape[0] == 1:
        return mix.squeeze(0)
    if mix.ndim == 2 and mix.shape[1] == 1:
        return mix.squeeze(1)
    raise ValueError("mix must be a single-channel waveform for this metric script.")


def crop_to_common_length(mix, clean, estimate):
    min_len = min(mix.shape[-1], clean.shape[-1], estimate.shape[-1])
    return mix[..., :min_len], clean[..., :min_len], estimate[..., :min_len]


def sourcewise_score(estimate, clean, metric, clamp_db):
    estimate = estimate.float()
    clean = clean.float()
    if metric == "snr":
        signal = clean.pow(2).sum(dim=-1)
        noise = (estimate - clean).pow(2).sum(dim=-1)
    elif metric in ["si_sdr", "sd_sdr"]:
        dot = (estimate * clean).sum(dim=-1, keepdim=True)
        clean_energy = clean.pow(2).sum(dim=-1, keepdim=True) + EPS
        scaled_clean = dot * clean / clean_energy
        signal = scaled_clean.pow(2).sum(dim=-1)
        if metric == "si_sdr":
            noise = (estimate - scaled_clean).pow(2).sum(dim=-1)
        else:
            noise = (estimate - clean).pow(2).sum(dim=-1)
    else:
        raise ValueError("Unsupported metric: {}".format(metric))

    scores = 10.0 * torch.log10(signal / (noise + EPS) + EPS)
    return clamp_scores(scores, clamp_db)


def pairwise_si_sdr(estimate, clean, clamp_db):
    estimate = estimate.float()[:, None, :]
    clean = clean.float()[None, :, :]
    dot = (estimate * clean).sum(dim=-1, keepdim=True)
    clean_energy = clean.pow(2).sum(dim=-1, keepdim=True) + EPS
    scaled_clean = dot * clean / clean_energy
    signal = scaled_clean.pow(2).sum(dim=-1)
    noise = (estimate - scaled_clean).pow(2).sum(dim=-1)
    scores = 10.0 * torch.log10(signal / (noise + EPS) + EPS)
    return clamp_scores(scores, clamp_db)


def best_si_sdr_permutation(estimate, clean, clamp_db):
    scores = pairwise_si_sdr(estimate, clean, clamp_db)
    n_src = clean.shape[0]
    best_perm = tuple(range(n_src))
    best_score = None

    for perm in permutations(range(n_src)):
        perm_idx = torch.tensor(perm, dtype=torch.long, device=scores.device)
        target_idx = torch.arange(n_src, device=scores.device)
        score = scores[perm_idx, target_idx].mean()
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm

    return list(best_perm)


def mean_source_score(estimate, clean, metric, clamp_db):
    return safe_float(sourcewise_score(estimate, clean, metric, clamp_db).mean())


def numpy_sources(wavs):
    wavs = wavs.detach().cpu().float().numpy()
    wavs = np.nan_to_num(wavs, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(wavs, dtype=np.float32)


def average_stoi(clean_np, estimate_np, sample_rate, extended):
    if stoi_fn is None:
        return np.nan
    values = []
    for clean, estimate in zip(clean_np, estimate_np):
        try:
            values.append(float(stoi_fn(clean, estimate, sample_rate, extended=extended)))
        except Exception:
            values.append(np.nan)
    return finite_mean(values)


def average_pesq_nb(clean_np, estimate_np, sample_rate):
    if pesq_fn is None or sample_rate not in [8000, 16000]:
        return np.nan
    values = []
    for clean, estimate in zip(clean_np, estimate_np):
        try:
            values.append(float(pesq_fn(sample_rate, clean, estimate, "nb")))
        except Exception:
            values.append(np.nan)
    return finite_mean(values)


def spectral_metrics(estimate, clean, n_fft, win_length, hop_length):
    estimate = estimate.float()
    clean = clean.float()
    window = torch.hann_window(win_length, dtype=estimate.dtype, device=estimate.device)
    est_spec = torch.stft(
        estimate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    clean_spec = torch.stft(
        clean,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    est_mag = est_spec.abs()
    clean_mag = clean_spec.abs()
    diff_db = 20.0 * (torch.log10(est_mag + EPS) - torch.log10(clean_mag + EPS))
    lsd = torch.sqrt(torch.mean(diff_db.pow(2), dim=1)).mean()
    spectral_convergence = torch.norm(est_mag - clean_mag) / (torch.norm(clean_mag) + EPS)
    return safe_float(lsd), safe_float(spectral_convergence)


class MoreMetricsTracker:
    def __init__(
        self,
        save_file,
        sample_rate,
        pit_from,
        skip_stoi,
        skip_pesq,
        bss_filter_length,
        bss_use_cg_iter,
        clamp_db,
        stft_n_fft,
        stft_win_length,
        stft_hop_length,
    ):
        self.sample_rate = sample_rate
        self.pit_from = pit_from
        self.skip_stoi = skip_stoi
        self.skip_pesq = skip_pesq
        self.bss_filter_length = bss_filter_length
        self.bss_use_cg_iter = bss_use_cg_iter
        self.clamp_db = None if clamp_db is not None and clamp_db < 0 else clamp_db
        self.stft_n_fft = stft_n_fft
        self.stft_win_length = stft_win_length
        self.stft_hop_length = stft_hop_length
        self.metric_columns = [col for col in METRIC_COLUMNS if col != "snt_id"]
        self.rows = []

        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=METRIC_COLUMNS)
        self.writer.writeheader()

    def _bss_metrics(self, clean, estimate):
        if fast_bss_eval is None:
            return np.nan, np.nan, np.nan
        kwargs = {
            "filter_length": self.bss_filter_length,
            "use_cg_iter": self.bss_use_cg_iter,
            "zero_mean": False,
            "clamp_db": self.clamp_db,
            "compute_permutation": False,
        }
        try:
            sdr, sir, sar = fast_bss_eval.bss_eval_sources(clean, estimate, **kwargs)
            if self.clamp_db is not None:
                sdr = clamp_scores(sdr, self.clamp_db)
                sir = clamp_scores(sir, self.clamp_db)
                sar = clamp_scores(sar, self.clamp_db)
            return safe_float(sdr.mean()), safe_float(sir.mean()), safe_float(sar.mean())
        except Exception:
            return np.nan, np.nan, np.nan

    def _write_row(self, row):
        self.writer.writerow(row)
        self.results_csv.flush()

    def __call__(self, mix, clean, estimate, key):
        mix = as_mix_1d(mix.detach())
        clean = as_sources_2d(clean.detach(), "clean")
        estimate = as_sources_2d(estimate.detach(), "estimate")
        mix, clean, estimate = crop_to_common_length(mix, clean, estimate)

        if clean.shape[0] != estimate.shape[0]:
            raise ValueError(
                "clean and estimate must have the same number of sources, got {} and {}.".format(
                    clean.shape[0], estimate.shape[0]
                )
            )

        if self.pit_from == "pw_mtx":
            perm = best_si_sdr_permutation(estimate, clean, self.clamp_db)
            estimate = estimate[perm]

        mix_sources = torch.stack([mix] * clean.shape[0], dim=0)

        sdr, sir, sar = self._bss_metrics(clean, estimate)
        mix_sdr, mix_sir, mix_sar = self._bss_metrics(clean, mix_sources)

        si_sdr = mean_source_score(estimate, clean, "si_sdr", self.clamp_db)
        si_sdr_baseline = mean_source_score(mix_sources, clean, "si_sdr", self.clamp_db)
        snr = mean_source_score(estimate, clean, "snr", self.clamp_db)
        snr_baseline = mean_source_score(mix_sources, clean, "snr", self.clamp_db)
        sd_sdr = mean_source_score(estimate, clean, "sd_sdr", self.clamp_db)
        sd_sdr_baseline = mean_source_score(mix_sources, clean, "sd_sdr", self.clamp_db)

        error = estimate - clean
        rmse = safe_float(torch.sqrt(torch.mean(error.pow(2), dim=-1)).mean())
        mae = safe_float(torch.mean(torch.abs(error), dim=-1).mean())
        lsd, spectral_convergence = spectral_metrics(
            estimate,
            clean,
            self.stft_n_fft,
            self.stft_win_length,
            self.stft_hop_length,
        )

        clean_np = numpy_sources(clean)
        estimate_np = numpy_sources(estimate)
        mix_np = numpy_sources(mix_sources)

        if self.skip_stoi:
            stoi = np.nan
            stoi_baseline = np.nan
            estoi = np.nan
            estoi_baseline = np.nan
        else:
            stoi = average_stoi(clean_np, estimate_np, self.sample_rate, extended=False)
            stoi_baseline = average_stoi(clean_np, mix_np, self.sample_rate, extended=False)
            estoi = average_stoi(clean_np, estimate_np, self.sample_rate, extended=True)
            estoi_baseline = average_stoi(clean_np, mix_np, self.sample_rate, extended=True)

        if self.skip_pesq:
            pesq_nb = np.nan
            pesq_nb_baseline = np.nan
        else:
            pesq_nb = average_pesq_nb(clean_np, estimate_np, self.sample_rate)
            pesq_nb_baseline = average_pesq_nb(clean_np, mix_np, self.sample_rate)

        row = {
            "snt_id": key,
            "sdr": sdr,
            "sdr_i": sdr - mix_sdr,
            "si_sdr": si_sdr,
            "si_sdr_i": si_sdr - si_sdr_baseline,
            "snr": snr,
            "snr_i": snr - snr_baseline,
            "sd_sdr": sd_sdr,
            "sd_sdr_i": sd_sdr - sd_sdr_baseline,
            "sir": sir,
            "sir_i": sir - mix_sir,
            "sar": sar,
            "sar_i": sar - mix_sar,
            "stoi": stoi,
            "stoi_i": stoi - stoi_baseline,
            "estoi": estoi,
            "estoi_i": estoi - estoi_baseline,
            "pesq_nb": pesq_nb,
            "pesq_nb_i": pesq_nb - pesq_nb_baseline,
            "rmse": rmse,
            "mae": mae,
            "lsd": lsd,
            "spectral_convergence": spectral_convergence,
        }
        self.rows.append(row)
        self._write_row(row)

    def update(self):
        return {
            "sdr_i": finite_mean([row["sdr_i"] for row in self.rows]),
            "si_sdr_i": finite_mean([row["si_sdr_i"] for row in self.rows]),
            "stoi_i": finite_mean([row["stoi_i"] for row in self.rows]),
            "pesq_i": finite_mean([row["pesq_nb_i"] for row in self.rows]),
        }

    def final(self):
        for row_name, reducer in [("avg", finite_mean), ("std", finite_std)]:
            row = {"snt_id": row_name}
            for col in self.metric_columns:
                row[col] = reducer([metric_row[col] for metric_row in self.rows])
            self._write_row(row)
        self.results_csv.close()


def pick_device(train_conf, requested_device):
    if requested_device is not None:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but CUDA is not available.")
        return device

    configured_gpus = train_conf.get("training", {}).get("gpus", None)
    if configured_gpus:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "This config requests GPU evaluation, but CUDA is not available in this session."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def build_progress():
    try:
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )
        from look2hear.utils import (
            BatchesProcessedColumn,
            MyMetricsTextColumn,
            RichProgressBarTheme,
        )

        metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
        progress = Progress(
            TextColumn("[bold blue]Testing", justify="right"),
            BarColumn(bar_width=None),
            "•",
            BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            "•",
            metricscolumn,
        )
        return progress, metricscolumn
    except Exception:
        return None, None


def tensors_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(tensors_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [tensors_to_device(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: tensors_to_device(value, device) for key, value in obj.items()}
    return obj


def run_one_example(model, test_set, idx, model_device, tracker):
    mix, sources, reverb_source, rir_target, key = tensors_to_device(
        test_set[idx], device=model_device
    )
    out = model(mix[None])
    est_sources = out["x_derev"].squeeze(0)
    tracker(mix=mix, clean=sources, estimate=est_sources, key=key)


def main(args):
    configure_runtime_env(args.cuda_visible_devices)

    import look2hear.datas
    import look2hear.models

    conf_path = resolve_path(args.conf_dir)
    with open(conf_path, "rb") as f:
        train_conf = yaml.safe_load(f)

    if args.exp_dir is None:
        released_exp = SCRIPT_DIR / "checkpoints" / train_conf["exp"]["exp_name"]
        training_exp = SCRIPT_DIR / "Experiments" / "checkpoint" / train_conf["exp"]["exp_name"]
        exp_dir = released_exp if released_exp.exists() else training_exp
    else:
        exp_dir = Path(args.exp_dir).expanduser()
    train_conf.setdefault("main_args", {})["exp_dir"] = str(exp_dir)

    model_path = exp_dir / "best.pth"
    if not model_path.exists():
        raise FileNotFoundError("Could not find model checkpoint: {}".format(model_path))

    device = pick_device(train_conf, args.device)
    model = getattr(look2hear.models, train_conf["audionet"]["audionet_name"]).from_pretrain(
        str(model_path),
        sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"],
        **train_conf["audionet"]["audionet_config"],
    )
    model.to(device)
    model.eval()
    model_device = next(model.parameters()).device

    datamodule = getattr(look2hear.datas, train_conf["datamodule"]["data_name"])(
        **train_conf["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    results_dir = exp_dir / "results"
    if args.save_file is None:
        save_file = results_dir / args.output_name
    else:
        save_file = Path(args.save_file).expanduser()

    tracker = MoreMetricsTracker(
        save_file=str(save_file),
        sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"],
        pit_from=args.pit_from,
        skip_stoi=args.skip_stoi,
        skip_pesq=args.skip_pesq,
        bss_filter_length=args.bss_filter_length,
        bss_use_cg_iter=args.bss_use_cg_iter,
        clamp_db=args.clamp_db,
        stft_n_fft=args.stft_n_fft,
        stft_win_length=args.stft_win_length,
        stft_hop_length=args.stft_hop_length,
    )

    num_examples = len(test_set)
    if args.max_examples is not None:
        num_examples = min(num_examples, args.max_examples)
    if num_examples == 0:
        tracker.final()
        print("Saved metrics to {}".format(save_file))
        return

    progress, metricscolumn = build_progress()
    with torch.no_grad():
        if progress is None:
            for idx in range(num_examples):
                run_one_example(model, test_set, idx, model_device, tracker)
                if idx % 50 == 0:
                    print("{}/{} {}".format(idx + 1, num_examples, tracker.update()))
        else:
            with progress:
                for idx in progress.track(range(num_examples)):
                    run_one_example(model, test_set, idx, model_device, tracker)
                    if idx % 50 == 0:
                        metricscolumn.update(tracker.update())
    tracker.final()
    print("Saved metrics to {}".format(save_file))


if __name__ == "__main__":
    main(parse_args())
