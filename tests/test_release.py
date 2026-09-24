import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "f03aa104551e2b8043f7c198b26c31e2497c55ac4bd3afa31391e4b921f39ebe"


def test_default_training_configuration():
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
