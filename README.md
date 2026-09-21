# DAMSEP: Distance-Aware Monaural Source Separation using Multi-RIR Estimation

Official implementation of **DAMSEP: Distance-Aware Monaural Source Separation
using Multi-RIR Estimation**.

DAMSEP separates a reverberant monaural mixture while jointly estimating a clean
source signal and a source-specific complex convolutive transfer function
(CTF) for each output. The CTF can be converted to a time-domain room impulse
response (RIR) for room-response analysis and ordinal near/far inference.

## Architecture

![DAMSEP architecture from Figure 1 of the paper: separation, shared dereverberation, and source-specific RIR estimation](assets/damsep_architecture.png)

**Figure 1. Architecture of DAMSEP.** The separation module estimates
source-specific reverberant spectra. A dereverberation module shared across
sources predicts clean-source spectra, and the RIR estimation module fuses
the clean and reverberant branches to estimate source-specific complex CTFs.
The three training objectives supervise reverberant-source separation,
clean-source estimation, and CTF-based reconstruction, respectively.

## Model outputs

For a waveform batch `mixture` with shape `[B, T]`, the model returns:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x_sep` | `[B, 2, T]` | separated reverberant source images |
| `x_derev` | `[B, 2, T]` | clean/reference-source estimates |
| `rir` | `[B*2, 2, 257, 60]` | real/imaginary parts of the estimated complex CTF |

The `rir` entry is a CTF, not a time-domain impulse response. Use
`inference_rir.py` or the decoder in `evaluate_rir_metrics.py` to obtain an RIR.

The released configuration uses distance-ordered targets: source 1 is nearer
to the reference microphone and source 2 is farther away.

The Python class remains named `SPMamba` for compatibility with the serialized
paper checkpoint; the DAMSEP response branch and objectives are implemented in
that class and `look2hear/models/RecRIR.py`.

## Installation

The reference environment uses Python 3.9, PyTorch 2.0, CUDA, and
`mamba-ssm==1.2.0.post1`:

```bash
conda env create -f environment.yml
conda activate dars
```

`mamba-ssm` is CUDA-dependent. Select PyTorch/CUDA builds compatible with your
driver if the pinned wheels are unavailable on your system.

## Data layout

DAMSEP expects the distance-ordered HETMIXR layout below. Audio is not bundled
with this repository.

```text
data/hetmixr/wav8k/min/
├── tr/
├── cv/
└── tt/
    ├── mix_both_reverb.json
    ├── s1_anechoic.json
    ├── s2_anechoic.json
    ├── s1_reverb.json
    ├── s2_reverb.json
    └── rir_reverb.json
```

Each JSON manifest is a list of `[audio_path, num_samples]` pairs. The paired
RIR WAV contains source 1 and source 2 in channels 1 and 2. See
[`data/README.md`](data/README.md) for preparation details and the released
test-set distance metadata.

## Pretrained checkpoint

The paper checkpoint is included at
`checkpoints/dars_mixed_p10/best.pth`. Its portable configuration is
`configs/dars.yml` and its SHA-256 digest is recorded in
[`checkpoints/README.md`](checkpoints/README.md).

```python
import yaml
from look2hear.models import SPMamba

with open("configs/dars.yml", "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

model = SPMamba.from_pretrain(
    "checkpoints/dars_mixed_p10/best.pth",
    sample_rate=config["datamodule"]["data_config"]["sample_rate"],
    **config["audionet"]["audionet_config"],
)
model.eval()
```

Run end-to-end inference on one or more 8 kHz monaural WAV files:

```bash
python inference_rir.py mixture.wav --device cuda:0
```

This writes two-channel reverberant estimates, clean-source estimates, decoded
responses, and a JSON manifest under `outputs/inference/`. Channel 1 is the
near source and channel 2 the far source.

## Training

Set the three dataset paths in `configs/dars.yml`, then run:

```bash
python audio_train.py --conf_dir configs/dars.yml
```

The checked-in configuration records the paper's eight-GPU setup. Adjust
`training.gpus` for the local machine; single-GPU training automatically avoids
DDP. TensorBoard is the default logger, while `training.logger: wandb` enables
Weights & Biases.

The public training configuration uses a scheduler patience of 5 and an
early-stopping patience of 5. The released checkpoint retains its original
patience-10 experiment metadata under `checkpoints/dars_mixed_p10/conf.yml`.

The paper configuration uses `w_rev=0.1`, `w_recon=0.5`, and `w_rir=0`. The
response branch is supervised indirectly by reconstructing the reverberant
source image. Loss ablations are in `configs/ablation/`.

## Evaluation

Run the metric implementation self-test without a checkpoint or dataset:

```bash
python evaluate_rir_metrics.py --self-test
```

Evaluate the released checkpoint after preparing HETMIXR:

```bash
python evaluate_rir_metrics.py \
  --conf-dir configs/dars.yml \
  --exp-dir checkpoints/dars_mixed_p10 \
  --distance-metadata data/metadata/distance_test.json \
  --device cuda:0
```

Separation, ablation, simulated-RIR, and measured-RIR table data are retained
under `results/`. Cross-model adapters and validation scripts are documented in
[`evaluation/README.md`](evaluation/README.md).

## Acknowledgements

The separation backbone is derived from
[SPMamba](https://github.com/JusperLee/SPMamba). The response-estimation design
and reconstruction objective build on
[Rec-RIR](https://github.com/Audio-WestlakeU/Rec-RIR). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and license
details.

## Citation

The paper citation and public preprint URL will be added when available.

## License

The repository is released under the Apache License 2.0. Components derived
from Rec-RIR remain subject to its MIT license, reproduced in
`licenses/Rec-RIR-LICENSE`.
