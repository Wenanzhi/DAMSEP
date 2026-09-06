import argparse
import os
from pathlib import Path

import torch
import yaml
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


SCRIPT_DIR = Path(__file__).resolve().parent


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
        help="Experiment directory containing best.pth.",
    )
    parser.add_argument(
        "--output_name",
        default="metrics_extended.csv",
        help="New CSV filename under the experiment results directory.",
    )
    parser.add_argument(
        "--dnsmos_model",
        default=None,
        help="Path to the DNSMOS P.835 sig_bak_ovr.onnx model.",
    )
    parser.add_argument("--dnsmos_threads", type=int, default=8)
    parser.add_argument("--bss_filter_length", type=int, default=512)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--cuda_visible_devices", default="0")
    return parser.parse_args()


def resolve_path(path_value):
    path = Path(path_value).expanduser()
    if path.exists():
        return path.resolve()
    project_path = SCRIPT_DIR / path
    if project_path.exists():
        return project_path.resolve()
    raise FileNotFoundError("Path not found: {}".format(path_value))


def resolve_dnsmos_model(requested_path):
    candidates = []
    if requested_path:
        candidates.append(Path(requested_path).expanduser())
    environment_path = os.environ.get("DNSMOS_P835_MODEL")
    if environment_path:
        candidates.append(Path(environment_path).expanduser())
    candidates.append(
        SCRIPT_DIR
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
        "DNSMOS P.835 model not found. Pass --dnsmos_model or set DNSMOS_P835_MODEL."
    )


def unique_output_path(path):
    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name("{}_{}{}".format(path.stem, index, path.suffix))
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a unique output path for {}".format(path))


def build_progress():
    from look2hear.utils import (
        BatchesProcessedColumn,
        MyMetricsTextColumn,
        RichProgressBarTheme,
    )

    metrics_column = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
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
        metrics_column,
    )
    return progress, metrics_column


def main(args):
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import look2hear.datas
    import look2hear.models
    from look2hear.metrics import MetricsTracker_test
    from look2hear.utils import tensors_to_device

    conf_path = resolve_path(args.conf_dir)
    with open(conf_path, "rb") as config_file:
        train_conf = yaml.safe_load(config_file)

    if args.exp_dir is None:
        released_exp = SCRIPT_DIR / "checkpoints" / train_conf["exp"]["exp_name"]
        training_exp = (
            SCRIPT_DIR / "Experiments" / "checkpoint" / train_conf["exp"]["exp_name"]
        )
        exp_dir = released_exp if released_exp.exists() else training_exp
    else:
        exp_dir = resolve_path(args.exp_dir)
    model_path = exp_dir / "best.pth"
    if not model_path.is_file():
        raise FileNotFoundError("Model checkpoint not found: {}".format(model_path))

    model = getattr(
        look2hear.models, train_conf["audionet"]["audionet_name"]
    ).from_pretrain(
        str(model_path),
        sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"],
        **train_conf["audionet"]["audionet_config"],
    )
    if train_conf["training"]["gpus"]:
        if not torch.cuda.is_available():
            raise RuntimeError("The experiment requests CUDA, but CUDA is unavailable.")
        model.to("cuda")
    model_device = next(model.parameters()).device

    datamodule = getattr(
        look2hear.datas, train_conf["datamodule"]["data_name"]
    )(**train_conf["datamodule"]["data_config"])
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_file = unique_output_path(results_dir / args.output_name)
    dnsmos_model = resolve_dnsmos_model(args.dnsmos_model)
    metrics = MetricsTracker_test(
        save_file=str(save_file),
        sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"],
        dnsmos_model_path=str(dnsmos_model),
        dnsmos_threads=args.dnsmos_threads,
        bss_filter_length=args.bss_filter_length,
    )

    if args.start_index < 0 or args.start_index > len(test_set):
        raise ValueError("start_index must be between 0 and {}".format(len(test_set)))
    stop_index = len(test_set)
    if args.max_examples is not None:
        stop_index = min(stop_index, args.start_index + max(0, args.max_examples))
    test_indices = range(args.start_index, stop_index)

    progress, metrics_column = build_progress()
    with torch.no_grad(), progress:
        for idx in progress.track(test_indices):
            mix, sources, reverb_source, rir_target, key = tensors_to_device(
                test_set[idx], device=model_device
            )
            output = model(mix[None])
            metrics(
                mix=mix,
                clean=sources,
                estimate=output["x_derev"].squeeze(0),
                key=key,
            )
            if idx % 50 == 0:
                metrics_column.update(metrics.update())
    metrics.final()
    print("Saved extended metrics to {}".format(save_file))


if __name__ == "__main__":
    main(parse_args())
