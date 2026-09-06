# Distance-ordered HETMIXR preparation

The paper uses two-source, monaural, distance-ordered mixtures. Source 1 is
generated nearer to the reference microphone and source 2 farther away. The
audio corpora themselves are not redistributed.

## Utilities

- `gen_mix_list.py`: deterministically sample source pairs and relative gains
  from an SCP file.
- `make_2mix.py`: active-level normalization and clean two-source mixture
  preparation.
- `gen_meta_csv.py`: deterministic room, microphone, source-position, and T60
  sampling.
- `create_wham_from_scratch.py`: synthesize anechoic/reverberant source images,
  paired RIRs, and mixtures from the generated metadata.
- `wham_room.py`: Pyroomacoustics room wrapper.

Example source and pair formats are provided under `examples/`.

```bash
python data/gen_mix_list.py \
  --scp data/examples/sources.scp.example \
  --out outputs/pairs.txt \
  --num 100 --max_gain 2.5 --seed 0 --no_same

python data/make_2mix.py \
  --task_txt outputs/pairs.txt \
  --cache outputs/active_levels.json \
  --out8k outputs/wav8k \
  --out16k outputs/wav16k \
  --subset tr --minmax min --num_workers 8
```

The released `metadata/distance_test.json` contains geometry and near/far labels
for the 3,000 generated test entries. After 4-second eligibility filtering, the
paper evaluates 2,801 mixtures.

Generated split directories must contain these JSON manifests:

```text
mix_both_reverb.json
s1_anechoic.json
s2_anechoic.json
s1_reverb.json
s2_reverb.json
rir_reverb.json
```

Each entry has the form `[path, num_samples]`. Keep all manifests in identical
order and use a two-channel RIR file ordered as `[source_1, source_2]`. For RIR
evaluation, each `rir_reverb` WAV must have a matching direct-path WAV whose
path is obtained by replacing that path component with `rir_anechoic`.
