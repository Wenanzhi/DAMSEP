# Evaluation

## DARS metrics

`evaluate_checkpoint.py` evaluates clean-source separation with the common
SI-SDR, SDR, PESQ-NB, STOI/ESTOI, SIR/SAR, and DNSMOS P.835 protocol.
`../evaluate_rir_metrics.py` decodes the complex CTF and evaluates the composed
source-specific room response.

DNSMOS requires the official `sig_bak_ovr.onnx` model. Pass its path with
`--dnsmos-model` or set `DNSMOS_P835_MODEL`; the model file is not redistributed.

Evaluate the released DARS checkpoint with fixed output identity:

```bash
python -m evaluation.evaluate_checkpoint \
  --adapter dars \
  --project-dir . \
  --config configs/dars.yml \
  --checkpoint checkpoints/dars_mixed_p10/best.pth \
  --output-dir outputs/main_table/dars \
  --alignment fixed \
  --dnsmos-model /path/to/sig_bak_ovr.onnx
```

For the paper's separation comparison, place separately obtained baseline
projects and checkpoints under `third_party/SPMamba`,
`third_party/TDANet-Large`, and `third_party/TF-Locoformer-M`. Then run
`evaluate_main_table.py`; use `--dry-run` first to inspect all resolved commands.
`build_main_table.py` validates complete per-example outputs and produces the
aggregate CSV/Markdown table.

The checked-in plot source can be rendered without the private per-example
outputs:

```bash
python evaluation/build_separation_quality_radar.py
python evaluation/build_ablation_results.py
```

## External RIR estimators

The `rir_baselines/` adapters support Rec-RIR, VINP, BUDDy, Speech2RIR, and
FiNS. Their repositories, environments, and checkpoints are intentionally not
vendored. Every adapter requires explicit local project/checkpoint paths.

The comparison protocol is:

1. export DARS-separated reverberant stems with
   `rir_baselines/export_dars_reverberant.py`;
2. run each external estimator on the same manifest;
3. evaluate predictions with `rir_baselines/evaluate_external_rirs.py`;
4. validate and aggregate with `rir_baselines/build_comparison_table.py`.

For example:

```bash
python evaluation/rir_baselines/export_dars_reverberant.py --device cuda:0

python evaluation/rir_baselines/run_recrir_vinp.py \
  --method recrir \
  --manifest outputs/rir_baselines/dars_stems/manifest.csv \
  --project-root /path/to/Rec-RIR \
  --config /path/to/Rec-RIR/config/Rec-RIR.toml \
  --checkpoint /path/to/Rec-RIR/ckpt/epoch35.tar \
  --output-dir outputs/rir_baselines/predictions/recrir
```

`measured_rir/run_matched_measured_rir.py` prepares and evaluates the measured
RIR test set. The paper reports only the `0716` panel. The measured RIR files are
not bundled and must be supplied with `--rir-root`.

```bash
python evaluation/measured_rir/run_matched_measured_rir.py \
  --rir-root /path/to/measured_rirs \
  --mode all --device cuda:0
```

## Retained results

The release includes compact, paper-facing CSVs in `../results/`; per-example
audio, predictions, logs, and intermediate checkpoints are excluded.
