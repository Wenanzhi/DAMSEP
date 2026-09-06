import hashlib
import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "f03aa104551e2b8043f7c198b26c31e2497c55ac4bd3afa31391e4b921f39ebe"


def test_paper_configuration():
    with (ROOT / "configs" / "dars.yml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for split in ("train", "val"):
        loss = config["loss"][split]["config"]
        assert loss["pit_from"] == "no_pit"
        assert float(loss["w_rev"]) == 0.1
        assert float(loss["w_recon"]) == 0.5
        assert float(loss["w_rir"]) == 0.0

    assert config["scheduler"]["sche_config"]["patience"] == 5
    assert config["training"]["early_stop"]["patience"] == 5
    assert config["exp"]["exp_name"] == "dars_mixed_p5"
    for split in ("train_dir", "valid_dir", "test_dir"):
        assert not Path(config["datamodule"]["data_config"][split]).is_absolute()


def test_released_checkpoint_digest():
    checkpoint = ROOT / "checkpoints" / "dars_mixed_p10" / "best.pth"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == CHECKPOINT_SHA256

    with (checkpoint.parent / "conf.yml").open("r", encoding="utf-8") as handle:
        checkpoint_config = yaml.safe_load(handle)
    assert checkpoint_config["training"]["early_stop"]["patience"] == 10


def read_csv(name):
    with (ROOT / "results" / name).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_paper_result_scope():
    main_rows = read_csv("main_separation_metrics.csv")
    assert [row["model"] for row in main_rows] == [
        "DARS (ES patience 10)",
        "SPMamba",
        "TDANet-Large",
        "TF-Locoformer-M",
    ]
    assert {int(row["separation_num_samples"]) for row in main_rows} == {2801}
    assert {int(row["sample_rate_hz"]) for row in main_rows} == {8000}

    for name, expected_models in (
        (
            "measured_rir_metrics.csv",
            {"DARS (ES patience 10)", "Rec-RIR", "VINP (TCN+SA+S, epoch 120)", "BUDDy", "Speech2RIR", "FiNS (local epoch 250)"},
        ),
        (
            "measured_rir_oracle_metrics.csv",
            {"Rec-RIR", "VINP (TCN+SA+S, epoch 120)", "BUDDy", "Speech2RIR", "FiNS (local epoch 250)"},
        ),
    ):
        rows = read_csv(name)
        assert {row["panel"] for row in rows} == {"0716"}
        assert {row["model"] for row in rows} == expected_models
        assert {(int(row["n_mixtures"]), int(row["n_responses"])) for row in rows} == {
            (390, 780)
        }

    category_rows = read_csv("category_separation_metrics.csv")
    assert len(category_rows) == 80
    assert {row["model"] for row in category_rows} == {
        "DARS",
        "SPMamba",
        "TDANet-Large",
        "TF-Locoformer-M",
    }
    assert not (ROOT / "results" / "rir_outlier_sensitivity.csv").exists()


def test_ablation_configs_match_table():
    variants = {
        "SPMamba (L_sep)": ("dars_lsep.yml", 0.0, 0.0),
        "DARS + L_rev": ("dars_lsep_lrev.yml", 0.1, 0.0),
        "DARS + L_resp": ("dars_lsep_lresp.yml", 0.0, 0.5),
        "DARS full": ("dars_full.yml", 0.1, 0.5),
    }
    rows = read_csv("ablation_metrics.csv")
    assert [row["variant"] for row in rows] == list(variants)
    for row in rows:
        config_name, w_rev, w_recon = variants[row["variant"]]
        with (ROOT / "configs" / "ablation" / config_name).open(
            "r", encoding="utf-8"
        ) as handle:
            config = yaml.safe_load(handle)
        loss = config["loss"]["train"]["config"]
        assert (float(loss["w_rev"]), float(loss["w_recon"]), float(loss["w_rir"])) == (
            w_rev,
            w_recon,
            0.0,
        )
        assert float(row["w_rev"]) == w_rev
        assert float(row["w_recon"]) == w_recon


def test_no_machine_specific_paths_in_release_text():
    forbidden = (
        "/mnt/",
        "/home/",
        "/hpc_",
        "C:\\Users\\",
        "D:\\Users\\",
    )
    suffixes = {
        ".py",
        ".yml",
        ".yaml",
        ".csv",
        ".json",
        ".toml",
        ".sh",
        ".md",
        ".txt",
    }
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.suffix not in suffixes
            or "tests" in path.relative_to(ROOT).parts
        ):
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
