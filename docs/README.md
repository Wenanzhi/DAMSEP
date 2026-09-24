# DAMSEP interactive demo

A static English project page containing the project introduction, six paired
test scenes, source-separation listening examples and source-specific RIR
comparisons. It uses local HTML/CSS/JavaScript, SVG plots and WAV audio, with
no build step, external font, analytics, backend or browser model inference.

The layout follows conventional academic project pages: a centered full paper
title and author list, resource links, an overview, a visible architecture
figure, sample tabs, native HTML audio controls in comparison tables, and
scientific plots with figure captions. Layout references include
[MMAudioSep](https://pontakahashi.github.io/MMAudioSep_Demo/),
[AnyRIR](https://kyungyunlee.github.io/anyRIR-demo/) and
[FlowSep](https://audio-agi.github.io/FlowSep_demo/).

## Preview and publish

Open `index.html` directly in a browser, keeping this entire directory together.
The bundled data use a script rather than `fetch`, so the demo also works
offline using `file://`. Alternatively, from the repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8765/`.

For GitHub Pages, publish `docs/` from the `main` branch using repository
**Settings → Pages → Build and deployment → Deploy from a branch**.
The intended project URL is `https://wenanzhi.github.io/DAMSEP/`.
The `.nojekyll` file keeps these assets as a plain static site. No generated
asset refers to a local filesystem path. Publishing requires the site files
to be in the remote branch and Pages to be configured; creating this directory
does not publish it. See the [GitHub Pages publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Samples and correspondence

The scenes are the six eligible test mixtures with the **smallest farther-source
distance to the reference microphone**. They are ordered by that distance;
ties use nearer-source distance, then utterance ID. Both sources are within
2.5 metres: near sources span 0.92–1.12 m and far sources span 2.34–2.48 m.
Selection is independent of model scores. Each scene uses the exact four-second
crop and source identities in the existing RIR baseline input manifest.
Only these six scenes are included in the page.

| Scene | Test entry | Crop start (samples at 8 kHz) |
| --- | --- | --- |
| 01 | `00004_2.26982_dialog_2_d3334_part001_-2.26982.wav` | 336 |
| 02 | `dialog_2_d6901_part000_2.49430_00041_-2.49430.wav` | 968 |
| 03 | `400o031b_1.97479_dialog_2_d2657_part000_-1.97479.wav` | 1507 |
| 04 | `dialog_2_d5178_part002_1.88300_00002_-1.88300.wav` | 344 |
| 05 | `dialog_2_d4116_part000_0.88970_22ha0112_-0.88970.wav` | 967 |
| 06 | `20ua010s_2.06999_dialog_2_d3813_part001_-2.06999.wav` | 1171 |

The geometry is an XY projection relative to the microphone, with one common
scale for the two axes. The labelled distances use all three coordinates.
Heights are absolute z coordinates; Δz is relative to the microphone. The
plot is fitted to each scene. Positions and metres are **ground truth**;
DAMSEP predicts relative near/far order through DRR.

There are 66 mono, PCM-16, 8 kHz WAV clips: one input mixture, two clean
references, two reverberant references, two DAMSEP clean estimates, two DAMSEP
reverberant estimates, and two SPMamba clean estimates per scene. All eleven
clips in a scene use the **same playback gain**; the highest absolute sample
across all eleven is scaled to 0.94. No per-method loudness normalization is
applied. Native browser audio controls provide playback, seeking and volume.
Switching scenes or audio modes stops playback; starting a clip pauses the
previous one. Source geometry and per-example DRR are shown in compact tables.

## RIR comparison protocol

- DAMSEP uses the unresolved two-source mixture. A sine sweep decodes its
  predicted complex CTF, and the effective response is convolved with the
  **paired ground-truth direct-path RIR**. The result is compared with the
  full reference response. This is reference-assisted response conversion.
- Rec-RIR, VINP, FiNS, BUDDy and Speech2RIR use the corresponding DAMSEP
  `x_sep` reverberant stem, with the existing estimator-specific preprocessing.
  These are the cached **estimated-stem** results, not oracle-input or
  single-speaker branch-selection results.
  FiNS uses the local epoch-250 checkpoint. Speech2RIR's native 0.25 s
  response is zero-padded to the common window, without extrapolation.
- Responses are resampled to 8 kHz and aligned at the direct arrival, using
  2.5 ms of preceding context and a 1 s tail. Waveform plots are individually
  peak-normalized and polarity-aligned to the reference. Absolute delay and
  gain are excluded from this comparison.
- The first 50 ms plot retains every sample. The full response uses ordered
  min/max pairs per 16-sample bin, preserving narrow peaks. EDCs are computed
  from the full aligned responses, normalized by total energy, sampled every
  16 samples, and clipped for display at −60 dB.
- DRR uses the original evaluator's ±2.5 ms direct window, 5 ms analysis
  guard, and 1 s post-direct tail. Higher DRR is interpreted as nearer.
  Each table row explicitly reports whether that inferred order agrees with the
  geometric label; ties within 1e-6 dB have no order.
- Original fixed source identities are retained for separation and RIRs.
  No RIR-based permutation or oracle branch selection is used here.

## Checkpoint provenance

The displayed DAMSEP checkpoint is the existing patience-10 experiment,
historically named DARS/JSRE. It is not a rerun of the current public
patience-5 training configuration.

| Model | Experiment | SHA-256 |
| --- | --- | --- |
| DAMSEP | `mix_rir0_p10_0719` | `f03aa104551e2b8043f7c198b26c31e2497c55ac4bd3afa31391e4b921f39ebe` |
| SPMamba | `mixed_g3090_p5_0227` | `ed2184ef0be6c98433aee24f657df1832176f3b1c0f072066e43e1e0d1a1acd6` |

External estimators reuse the existing runs: Rec-RIR `epoch35.tar`, VINP-TCNSAS
`epoch120.tar`, FiNS local run `m-251022-174032/epoch-250.pt`, BUDDy
`VCTK_16k_4s_time-190000.pt`, and Speech2RIR `checkpoint-1040000steps.pkl`.

The data bundle `data/demo-data.js` records checkpoint provenance, exact
test identities, crop starts, geometry, shared playback gains and curves.

## Rebuild from the experiment workspace

`scripts/build_demo.py` uses the existing experiment workspace containing
`distant-Separation`, `SPMamba-distant`, `mixed_dataset`, and
`evaluation/rir_baselines`. Run it in the existing `look2hear` environment:

```bash
python scripts/build_demo.py --workspace /path/to/Channel-based_Separation --device cuda:0
```

The script selects samples without consulting scores, reruns only six inputs
per separation model, checks DAMSEP output correspondence against the cached
baseline inputs, and exports existing RIR baseline predictions. Reference
DRR must agree between the new export and the cached metric records.
Intermediate arrays stay in the ignored `.demo-cache/` directory. Raw
corpora and checkpoints are not copied into the page.

To run the browser checks, install Playwright in a separate temporary tools
directory and provide it through `NODE_PATH`:

```bash
npm install --prefix /tmp/damsep-browser playwright
/tmp/damsep-browser/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/damsep-browser/node_modules node scripts/check_demo.cjs
```
