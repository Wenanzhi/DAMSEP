# DAMSEP: Distance-Aware Monaural Source Separation using Multi-RIR Estimation

**Wen Wen, Qiang Zhou, Yu Xi, Haoyu Li, Bohan Li, and Kai Yu**

DAMSEP jointly estimates source signals and source-specific acoustic responses
from a single-microphone mixture. A separation backbone is coupled with shared
dereverberation and response-estimation modules. Clean-source supervision,
reverberant-source supervision, and acoustic reconstruction train these modules
together. In the paper, the estimated responses are decoded into room impulse
responses (RIRs), whose direct-to-reverberant ratios (DRRs) provide relative
near/far ordering.

This release includes the training implementation and configuration, a
pretrained checkpoint, RIR analysis utilities, and test-set distance metadata.
Training audio must be prepared separately; see [Data preparation](#data-preparation).

## Architecture

![DAMSEP architecture: source separation, shared dereverberation, and source-specific response estimation](assets/damsep_architecture.png)

**Figure 1. Architecture of DAMSEP.** The separation module estimates
source-specific reverberant spectra. The shared dereverberation module predicts
clean-source spectra. The response branch combines the clean and reverberant
representations to estimate a complex convolutive transfer function (CTF) for
each source. Reconstruction supervision links the estimated CTFs to the
reference reverberant signals.

The Python model class is named `SPMamba` to preserve checkpoint compatibility.
The joint model is implemented in [SPMamba.py](look2hear/models/SPMamba.py) and
[RecRIR.py](look2hear/models/RecRIR.py).

## Results reported in the paper

| Evaluation setting | Metric | DAMSEP |
| --- | --- | --- |
| HETMIXR, 2,801 eligible test mixtures | SI-SDR improvement | 14.90 dB |
| HETMIXR, 2,801 eligible test mixtures | Distance-ordering accuracy | 99.11% |
| Unseen room, 390 mixtures rendered with measured RIRs | Distance-ordering accuracy | 99.74% |

The model has 7.2 million parameters. On HETMIXR, the paper reports a 0.65 dB
SI-SDRi improvement over TF-Locoformer. The measured-RIR experiment uses a model
trained on simulated acoustic environments, without fine-tuning on the unseen
room. See [Configuration notes](#configuration-notes) when using this release
to reproduce the experiments.

## Installation

The reference environment uses Python 3.9, PyTorch 2.0, and
`mamba-ssm==1.2.0.post1`. Use a CUDA-capable Linux environment with a compatible
NVIDIA driver and CUDA build toolchain.

```bash
conda env create -f environment.yml
conda activate dars
```

The environment name and existing `dars` file paths are retained for
compatibility with the released code and checkpoint.

## Data preparation

HETMIXR contains heterogeneous two-source mixtures with clean references,
reverberant source images, source-specific RIRs, and geometric distance labels.
The paper uses 20,000 training, 5,000 validation, and 3,000 generated test
mixtures; excluding test mixtures shorter than four seconds leaves 2,801.
Simulated reverberation times range from 0.1 to 1.0 s, with source distances
sampled from 1.0–1.9 m and 2.0–4.0 m.

Prepare 8 kHz audio and six aligned JSON manifests in **each** split directory:

```text
data/hetmixr/wav8k/min/
├── tr/  # training manifests
├── cv/  # validation manifests
└── tt/  # test manifests
```

The required manifests are `mix_both_reverb.json`, `s1_anechoic.json`,
`s2_anechoic.json`, `s1_reverb.json`, `s2_reverb.json`, and `rir_reverb.json`.
Every manifest contains `[audio_path, num_samples]` pairs in the same mixture
order. Source 1 is the nearer source and source 2 is the farther source for the
released fixed-order recipe.

See [data/README.md](data/README.md) for the format and channel conventions.
The included [distance_test.json](data/metadata/distance_test.json) contains
test-set geometry and ordering labels. It does not contain the audio or the
training manifests.

## Training

Edit [configs/dars.yml](configs/dars.yml):

1. Set `datamodule.data_config.train_dir`, `valid_dir`, and `test_dir` to the
   directories containing your manifests. All three splits are required by the
   current data loader; the test loader is used for periodic monitoring.
2. Set `training.gpus` for your machine, for example `[0]` for one GPU. The
   checked-in configuration lists eight GPUs; DDP is used for multiple GPUs.
3. Choose an experiment name with `exp.exp_name` and adjust the data-loading
   workers for your hardware. Keep the default per-device batch size of 1 for
   the current CTF reconstruction implementation.

Run from the repository root:

```bash
python audio_train.py --conf_dir configs/dars.yml
```

### Training objectives

The default recipe uses four-second segments at 8 kHz, Adam with an initial
learning rate of `1e-3`, and the following supervision:

| Objective | Configuration | Weight |
| --- | --- | --- |
| Clean-source estimation | `pairwise_neg_snr` during training | 1.0 |
| Reverberant-source estimation | `w_rev` | 0.1 |
| CTF-based reverberant reconstruction | `w_recon` | 0.5 |

Reconstruction uses the reference clean-source spectra and the estimated CTFs
to explain the corresponding reverberant sources. The response branch is
trained through reconstruction; direct RIR supervision is disabled (`w_rir=0`).
The implementation is in [pit_wrapper.py](look2hear/losses/pit_wrapper.py).

### Checkpoints, logs, and resuming

Training writes Lightning checkpoints, a resolved configuration, loss history,
and the exported `best.pth` to `Experiments/checkpoint/<exp_name>/`.
TensorBoard logs are written to `Experiments/tensorboard_logs/` by default.
Set `training.logger: wandb` to use Weights & Biases or
`training.logger: none` to disable the experiment logger.

Resume a run with its Lightning checkpoint:

```bash
python audio_train.py --conf_dir configs/dars.yml \
  --resume_from_checkpoint Experiments/checkpoint/dars_mixed_p5/last.ckpt
```

The released `best.pth` contains model weights and metadata; use a training
`.ckpt` file to restore optimizer and scheduler state when resuming.

### Configuration notes

- The released default configuration and pretrained checkpoint use fixed
  distance-ordered supervision (`pit_from: no_pit`). Section 2.3 of the
  manuscript describes permutation-invariant matching. These source-assignment
  settings differ and should be accounted for when reproducing the method.
- The default configuration uses a scheduler patience and early-stopping
  patience of 5. The retained checkpoint comes from an experiment with an
  early-stopping patience of 10.

## Pretrained checkpoint and model outputs

The pretrained weights are included at
[checkpoints/best.pth](checkpoints/best.pth).
Architecture arguments for loading the model are provided by
[configs/dars.yml](configs/dars.yml).

```python
import yaml
from look2hear.models import SPMamba

with open("configs/dars.yml", "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

model = SPMamba.from_pretrain(
    "checkpoints/best.pth",
    sample_rate=config["datamodule"]["data_config"]["sample_rate"],
    **config["audionet"]["audionet_config"],
)
model.eval()
```

For an input waveform batch of shape `[B, T]`, the default model returns:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x_sep` | `[B, 2, T]` | Separated reverberant source images |
| `x_derev` | `[B, 2, T]` | Clean-source estimates |
| `rir` | `[B*2, 2, 257, 60]` | Real and imaginary components of each source's CTF |

The `rir` output represents a complex CTF. The paper converts it to a
time-domain RIR using a fixed sine sweep and inverse filtering. This release
does not include that decoding pipeline.

## RIR analysis utilities

The retained [look2hear/eval/](look2hear/eval/) scripts operate on a directory
of already decoded, two-channel RIR WAV files:

```bash
python look2hear/eval/estimate_DRR.py -i path/to/rir_wavs --sr 8000
python look2hear/eval/estimate_T60_stereo.py -i path/to/rir_wavs --sr 8000
```

They estimate channel-wise DRR and/or reverberation time and save JSON
summaries beside the input files. Their inputs are time-domain RIRs rather
than the model's CTF tensors.

## Acknowledgements and license

The separation backbone and training framework build on
[SPMamba](https://github.com/JusperLee/SPMamba). The response-estimation design
and reconstruction objective build on
[Rec-RIR](https://github.com/Audio-WestlakeU/Rec-RIR).

The repository is released under the [Apache License 2.0](LICENSE).
Rec-RIR-derived components retain their [MIT license](licenses/Rec-RIR-LICENSE).

## Citation

```bibtex
@misc{wen2026damsep,
  title  = {DAMSEP: Distance-Aware Monaural Source Separation using Multi-RIR Estimation},
  author = {Wen, Wen and Zhou, Qiang and Xi, Yu and Li, Haoyu and Li, Bohan and Yu, Kai},
  year   = {2026}
}
```

The public preprint link and identifier will be added when available.
