# DARS loss ablations

All configurations use the same distance-ordered HETMIXR protocol, DARS
architecture, optimizer, scheduler, seed, and early-stopping patience of 10.
The waveform separation loss is active in every row.

| Configuration | `w_rev` | `w_recon` | `w_rir` |
| --- | ---: | ---: | ---: |
| `dars_lsep.yml` | 0.0 | 0.0 | 0.0 |
| `dars_lsep_lrev.yml` | 0.1 | 0.0 | 0.0 |
| `dars_lsep_lresp.yml` | 0.0 | 0.5 | 0.0 |
| `dars_full.yml` | 0.1 | 0.5 | 0.0 |

`w_rir` remains zero because the direct-RIR loss branch is disabled. The
response estimate is trained indirectly through reverberant reconstruction.

Run another seed by overriding the second-level `seed` and `exp_name` fields:

```bash
python audio_train.py \
  --conf_dir configs/ablation/dars_full.yml \
  --seed 1 \
  --exp_name dars_ablation_full_s1
```

The `L_sep` and `L_sep + L_rev` variants enable DDP unused-parameter detection
because their CTF-only prediction layers receive no gradient.
