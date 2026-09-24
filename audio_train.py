import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(SCRIPT_DIR / ".cache" / "numba"))

import sys
import torch
import math
from torch import Tensor
import argparse
import json
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.utils
from look2hear.system import make_optimizer
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from rich import print, reconfigure
from collections.abc import MutableMapping
from look2hear.utils import print_only, MyRichProgressBar, RichProgressBarTheme

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default=str(SCRIPT_DIR / "configs" / "dars.yml"),
    help="Path to the training configuration.",
)
parser.add_argument(
    "--resume_from_checkpoint",
    default=None,
    help="Path to a Lightning checkpoint to resume training from.",
)


class LossHistoryCallback(pl.Callback):
    def __init__(self, exp_dir, append_existing=False):
        super().__init__()
        self.exp_dir = exp_dir
        self.log_path = os.path.join(exp_dir, "loss.log")
        self.plot_path = os.path.join(exp_dir, "loss.png")
        self.append_existing = append_existing
        self.history = []

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        os.makedirs(self.exp_dir, exist_ok=True)
        self.history = []
        if self.append_existing and os.path.exists(self.log_path):
            self._load_history()
            return
        with open(self.log_path, "w") as f:
            f.write("epoch\ttrain_loss\teval_loss\n")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metrics = {}
        metrics.update(getattr(trainer, "callback_metrics", {}))
        metrics.update(getattr(trainer, "logged_metrics", {}))
        metrics.update(getattr(trainer, "progress_bar_metrics", {}))

        train_loss = self._get_metric(
            metrics,
            ("train_loss_epoch", "train_loss", "loss"),
        )
        eval_loss = self._get_metric(
            metrics,
            (
                "val_loss/dataloader_idx_0",
                "val_loss_epoch/dataloader_idx_0",
                "val_loss",
                "eval_loss",
            ),
        )

        if train_loss is None or eval_loss is None:
            return

        epoch = trainer.current_epoch + 1
        self.history.append((epoch, train_loss, eval_loss))
        with open(self.log_path, "a") as f:
            f.write(f"{epoch}\t{train_loss:.8f}\t{eval_loss:.8f}\n")

        print_only(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, eval_loss={eval_loss:.6f}"
        )
        self._save_plot()

    @staticmethod
    def _get_metric(metrics, names):
        for name in names:
            if name not in metrics:
                continue
            value = metrics[name]
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "item"):
                value = value.item()
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return None

    def _save_plot(self):
        os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".cache" / "matplotlib"))
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print_only("matplotlib is not installed; loss.png was not saved.")
            return

        epochs = [item[0] for item in self.history]
        train_losses = [item[1] for item in self.history]
        eval_losses = [item[2] for item in self.history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, marker="o", label="train_loss")
        plt.plot(epochs, eval_losses, marker="o", label="eval_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()

    def _load_history(self):
        with open(self.log_path) as f:
            next(f, None)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                try:
                    epoch = int(parts[0])
                    train_loss = float(parts[1])
                    eval_loss = float(parts[2])
                except ValueError:
                    continue
                self.history.append((epoch, train_loss, eval_loss))


def main(config):
    seed = int(config["training"].get("seed", 0))
    pl.seed_everything(seed, workers=True)

    print_only(
        "Instantiating datamodule <{}>".format(config["datamodule"]["data_name"])
    )
    datamodule: object = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader

    # Define model and optimizer
    print_only(
        "Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"])
    )
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    # import pdb; pdb.set_trace()
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"])
        )
        if config["scheduler"]["sche_name"] != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
                optimizer=optimizer, **config["scheduler"]["sche_config"]
            )
        else:
            scheduler = {
                "scheduler": getattr(look2hear.system.schedulers, config["scheduler"]["sche_name"])(
                    optimizer, len(train_loader) // config["datamodule"]["data_config"]["batch_size"], 64
                ),
                "interval": "step",
            }

    # Just after instantiating, save the args. Easy loading in the future.
    config.setdefault("main_args", {})["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)
    resume_from_checkpoint = config["main_args"].get("resume_from_checkpoint")
    if resume_from_checkpoint:
        resume_from_checkpoint = os.path.abspath(resume_from_checkpoint)
        if not os.path.exists(resume_from_checkpoint):
            raise FileNotFoundError(
                f"resume_from_checkpoint does not exist: {resume_from_checkpoint}"
            )

    # Define Loss function.
    print_only(
        "Instantiating Loss, Train <{}>, Val <{}>".format(
            config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]
        )
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    print_only("Instantiating System <{}>".format(config["training"]["system"]))
    system = getattr(look2hear.system, config["training"]["system"])(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=10,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)
    callbacks.append(
        LossHistoryCallback(exp_dir, append_existing=bool(resume_from_checkpoint))
    )

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    logger_name = str(config["training"].get("logger", "tensorboard")).lower()
    if logger_name == "tensorboard":
        experiment_logger = TensorBoardLogger(
            logger_dir, name=config["exp"]["exp_name"]
        )
    elif logger_name == "wandb":
        experiment_logger = WandbLogger(
            name=config["exp"]["exp_name"],
            save_dir=os.path.join(logger_dir, config["exp"]["exp_name"]),
            project=config["training"].get("wandb_project", "DARS"),
        )
    elif logger_name in ("none", "false"):
        experiment_logger = False
    else:
        raise ValueError("Unknown training logger: {}".format(logger_name))

    multiple_devices = isinstance(gpus, (list, tuple)) and len(gpus) > 1
    strategy = (
        DDPStrategy(
            find_unused_parameters=config["training"].get(
                "find_unused_parameters", False
            )
        )
        if multiple_devices
        else "auto"
    )
    precision = config["training"].get(
        "precision", "bf16-mixed" if torch.cuda.is_available() else "32-true"
    )

    trainer = pl.Trainer(
        # precision="16-mixed",
        precision=precision,
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus if torch.cuda.is_available() else 1,
        accelerator=distributed_backend or "cpu",
        # strategy=DDPStrategy(find_unused_parameters=True), # wenwen 0317 changed
        strategy=strategy,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=experiment_logger,
        sync_batchnorm=multiple_devices,
        # num_sanity_val_steps=0,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )

    # ckpt_path = os.path.join(exp_dir, "epoch=129.ckpt")
    # trainer.fit(system,ckpt_path=ckpt_path)
    trainer.fit(system, ckpt_path=resume_from_checkpoint)
    print_only("Finished Training")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from look2hear.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    # pprint(arg_dic)
    main(arg_dic)
