# HETMIXR data format

The training implementation expects two-source, monaural mixtures sampled at
8 kHz. The default recipe uses fixed distance order: source 1 is nearer to
the reference microphone and source 2 is farther away.

Prepare the audio and manifests separately. This repository includes test-set
distance metadata; it does not redistribute the source-corpus audio or the
rendered mixtures.

## Split directories

Set `train_dir`, `valid_dir`, and `test_dir` in
[configs/dars.yml](../configs/dars.yml). Each directory must contain all six
manifests below. The current data module creates loaders for all three splits,
including a test loader used for periodic monitoring during training.

| Manifest | Contents of each referenced WAV |
| --- | --- |
| `mix_both_reverb.json` | Monaural sum of the two reverberant sources |
| `s1_anechoic.json` | Source 1 clean reference |
| `s2_anechoic.json` | Source 2 clean reference |
| `s1_reverb.json` | Source 1 reverberant image |
| `s2_reverb.json` | Source 2 reverberant image |
| `rir_reverb.json` | Two-channel RIR, ordered as source 1 then source 2 |

Each manifest is a JSON list of `[audio_path, num_samples]` entries, for example:

```json
[
  ["data/hetmixr/audio/tr/mix_both_reverb/example_0001.wav", 40000],
  ["data/hetmixr/audio/tr/mix_both_reverb/example_0002.wav", 48000]
]
```

Replace the example paths and lengths with real files and their sample counts.
Paths are read as provided: relative audio paths are resolved against the
process working directory, so run training from the repository root.

Keep all six manifests in the same mixture order and align the source signals
in time with the mixture. Mixture and source WAVs are mono; RIR WAVs have two
channels. All audio must already have the configured sample rate because the
loader does not resample it. Keep RIR lengths consistent within a batch.

The default segment length is four seconds. Mixtures shorter than this are
filtered out by the loader. `rir_reverb.json` and its WAV files are still read
when `w_rir=0`, so they are required by the current loader.

## Released test metadata

[metadata/distance_test.json](metadata/distance_test.json) contains geometry
and near/far labels for 3,000 generated test mixtures. Each record includes
`utterance_id`, the reference-microphone position, source positions and
distances, and `closer_to_left_mic`. Match records by `utterance_id` when
working with a filtered subset.

The paper evaluates 2,801 test mixtures after excluding recordings shorter
than four seconds. This metadata is separate from the six training-loader
manifests and is not read by `MixDataModule`.
