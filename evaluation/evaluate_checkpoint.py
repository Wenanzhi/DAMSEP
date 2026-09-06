from __future__ import annotations

import argparse
import importlib
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.extended_metrics import ExtendedMetricsTracker


LOGGER = logging.getLogger("cross_project_evaluation")


@dataclass
class EvaluationBundle:
    model: torch.nn.Module
    test_set: Any
    sample_rate: int
    alignment: str
    decode_sample: Callable[[Any], Tuple[torch.Tensor, torch.Tensor, str]]
    model_output_key: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate separation checkpoints with a common metric set."
    )
    parser.add_argument(
        "--adapter",
        choices=("dars", "spmamba", "tdanet", "tflocoformer"),
        required=True,
    )
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for metric CSV/JSON output (default: checkpoint/results).",
    )
    parser.add_argument("--output-name", default="metrics_extended_comparison.csv")
    parser.add_argument("--dnsmos-model", default=None)
    parser.add_argument("--dnsmos-threads", type=int, default=8)
    parser.add_argument("--bss-filter-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--alignment",
        choices=("auto", "fixed", "pit"),
        default="auto",
        help="Metric source matching. 'fixed' preserves raw output identity.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def resolve_file(path_value: str) -> Path:
    resolved = Path(path_value).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("File not found: {}".format(resolved))
    return resolved


def resolve_directory(path_value: str) -> Path:
    resolved = Path(path_value).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError("Directory not found: {}".format(resolved))
    return resolved


def resolve_dnsmos_model(requested_path: Optional[str]) -> Path:
    candidates = []
    if requested_path:
        candidates.append(Path(requested_path).expanduser())
    environment_path = os.environ.get("DNSMOS_P835_MODEL")
    if environment_path:
        candidates.append(Path(environment_path).expanduser())
    candidates.append(
        WORKSPACE_ROOT
        / "third_party"
        / "ClearerVoice-Studio"
        / "speechscore"
        / "scores"
        / "dnsmos"
        / "DNSMOS"
        / "sig_bak_ovr.onnx"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "DNSMOS P.835 model not found. Pass --dnsmos-model."
    )


def unique_output_path(path: Path) -> Path:
    if not path.exists() and not path.with_suffix(".summary.json").exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name(
            "{}_{}{}".format(path.stem, index, path.suffix)
        )
        if (
            not candidate.exists()
            and not candidate.with_suffix(".summary.json").exists()
        ):
            return candidate
    raise RuntimeError("Could not allocate output path for {}".format(path))


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Invalid YAML config: {}".format(path))
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_spmamba_family(
    config: Dict[str, Any], checkpoint: Path
) -> EvaluationBundle:
    look2hear_models = importlib.import_module("look2hear.models")
    look2hear_datas = importlib.import_module("look2hear.datas")
    model_config = config["audionet"]
    data_config = config["datamodule"]["data_config"]
    model_class = getattr(
        look2hear_models, model_config["audionet_name"]
    )
    model = model_class.from_pretrain(
        str(checkpoint),
        sample_rate=data_config["sample_rate"],
        **model_config["audionet_config"],
    )
    datamodule_class = getattr(
        look2hear_datas, config["datamodule"]["data_name"]
    )
    datamodule = datamodule_class(**data_config)
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    def decode_sample(sample: Any) -> Tuple[torch.Tensor, torch.Tensor, str]:
        mixture, sources, key = sample
        return mixture, sources, str(key)

    return EvaluationBundle(
        model=model,
        test_set=test_set,
        sample_rate=int(data_config["sample_rate"]),
        alignment="pit",
        decode_sample=decode_sample,
    )


def load_dars(
    config: Dict[str, Any], checkpoint: Path
) -> EvaluationBundle:
    look2hear_models = importlib.import_module("look2hear.models")
    look2hear_datas = importlib.import_module("look2hear.datas")
    model_config = config["audionet"]
    data_config = config["datamodule"]["data_config"]
    model_class = getattr(
        look2hear_models, model_config["audionet_name"]
    )
    model = model_class.from_pretrain(
        str(checkpoint),
        sample_rate=data_config["sample_rate"],
        **model_config["audionet_config"],
    )
    datamodule_class = getattr(
        look2hear_datas, config["datamodule"]["data_name"]
    )
    datamodule = datamodule_class(**data_config)
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    def decode_sample(sample: Any) -> Tuple[torch.Tensor, torch.Tensor, str]:
        mixture, sources, _, _, key = sample
        return mixture, sources, str(key)

    training_assignment = str(
        config["loss"]["train"]["config"].get("pit_from", "pw_mtx")
    )
    return EvaluationBundle(
        model=model,
        test_set=test_set,
        sample_rate=int(data_config["sample_rate"]),
        alignment="fixed" if training_assignment == "no_pit" else "pit",
        decode_sample=decode_sample,
        model_output_key="x_derev",
    )


def load_tdanet(
    config: Dict[str, Any], checkpoint: Path
) -> EvaluationBundle:
    seed_everything(int(config.get("seed", 2026)))
    project_audio_test = importlib.import_module("audio_test")
    look2hear_datas = importlib.import_module("look2hear.datas")
    model = project_audio_test._load_model(config, str(checkpoint))
    data_config = config["datamodule"]["data_config"]
    datamodule_class = getattr(
        look2hear_datas, config["datamodule"]["data_name"]
    )
    datamodule = datamodule_class(**data_config)
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    def decode_sample(sample: Any) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return sample["mixture"], sample["sources"], str(sample["key"])

    protocol = str(config["experiment"]["protocol"])
    return EvaluationBundle(
        model=model,
        test_set=test_set,
        sample_rate=int(data_config["sample_rate"]),
        alignment="pit" if protocol == "pit" else "fixed",
        decode_sample=decode_sample,
    )


def load_tflocoformer(
    config: Dict[str, Any], checkpoint: Path
) -> EvaluationBundle:
    seed_everything(int(config["experiment"].get("seed", 0)))
    experiment_utils = importlib.import_module("experiment_utils")
    project_test = importlib.import_module("test")
    datamodule = experiment_utils.build_datamodule(config)
    _, _, test_set = datamodule.make_sets
    model = experiment_utils.build_model(config)
    project_test.load_model_checkpoint(model, str(checkpoint))

    def decode_sample(sample: Any) -> Tuple[torch.Tensor, torch.Tensor, str]:
        mixture, sources, metadata = sample
        return mixture, sources, str(metadata.get("key", ""))

    protocol = str(config["experiment"]["protocol"])
    return EvaluationBundle(
        model=model,
        test_set=test_set,
        sample_rate=int(config["data"]["sample_rate"]),
        alignment="pit" if protocol == "pit" else "fixed",
        decode_sample=decode_sample,
    )


def build_bundle(
    adapter: str,
    config: Dict[str, Any],
    checkpoint: Path,
) -> EvaluationBundle:
    if adapter == "dars":
        return load_dars(config, checkpoint)
    if adapter == "spmamba":
        return load_spmamba_family(config, checkpoint)
    if adapter == "tdanet":
        return load_tdanet(config, checkpoint)
    if adapter == "tflocoformer":
        return load_tflocoformer(config, checkpoint)
    raise ValueError("Unsupported adapter: {}".format(adapter))


def main(args: argparse.Namespace) -> Path:
    project_dir = resolve_directory(args.project_dir)
    config_path = resolve_file(args.config)
    checkpoint_path = resolve_file(args.checkpoint)
    dnsmos_model = resolve_dnsmos_model(args.dnsmos_model)

    sys.path.insert(0, str(project_dir))
    os.chdir(project_dir)
    config = load_yaml(config_path)
    bundle = build_bundle(args.adapter, config, checkpoint_path)
    metric_alignment = (
        bundle.alignment if args.alignment == "auto" else args.alignment
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    bundle.model.to(device)
    bundle.model.eval()

    results_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint_path.parent / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = unique_output_path(results_dir / args.output_name)
    tracker = ExtendedMetricsTracker(
        save_file=str(output_path),
        sample_rate=bundle.sample_rate,
        alignment=metric_alignment,
        dnsmos_model_path=str(dnsmos_model),
        dnsmos_threads=args.dnsmos_threads,
        bss_filter_length=args.bss_filter_length,
    )

    dataset_size = len(bundle.test_set)
    start_index = max(0, int(args.start_index))
    end_index = dataset_size if args.end_index is None else int(args.end_index)
    end_index = min(dataset_size, max(start_index, end_index))
    if args.max_examples is not None:
        end_index = min(
            end_index,
            start_index + max(0, int(args.max_examples)),
        )
    num_examples = end_index - start_index
    LOGGER.info(
        "adapter=%s samples=%d range=[%d,%d) sample_rate=%d alignment=%s device=%s",
        args.adapter,
        num_examples,
        start_index,
        end_index,
        bundle.sample_rate,
        metric_alignment,
        device,
    )

    start_time = time.time()
    try:
        with torch.inference_mode():
            for processed_index, sample_index in enumerate(
                range(start_index, end_index), start=1
            ):
                mixture, sources, key = bundle.decode_sample(
                    bundle.test_set[sample_index]
                )
                model_input = mixture.unsqueeze(0).to(device)
                estimate = bundle.model(model_input)
                if bundle.model_output_key is not None:
                    if not isinstance(estimate, dict):
                        raise TypeError(
                            "Expected dict model output for key {}, got {}".format(
                                bundle.model_output_key,
                                type(estimate).__name__,
                            )
                        )
                    estimate = estimate[bundle.model_output_key]
                if not isinstance(estimate, torch.Tensor):
                    raise TypeError(
                        "Expected tensor model output, got {}".format(
                            type(estimate).__name__
                        )
                    )
                estimate = estimate.squeeze(0).detach().cpu()
                tracker(
                    mix=mixture,
                    clean=sources,
                    estimate=estimate,
                    key=key,
                )
                if (
                    processed_index == 1
                    or processed_index % args.progress_every == 0
                    or processed_index == num_examples
                ):
                    elapsed = time.time() - start_time
                    LOGGER.info(
                        "progress=%d/%d elapsed=%.1fs metrics=%s",
                        processed_index,
                        num_examples,
                        elapsed,
                        tracker.update(),
                    )
        summary = tracker.final()
    except Exception:
        tracker.close()
        raise

    LOGGER.info("summary=%s", summary)
    LOGGER.info("saved=%s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main(parse_args())
