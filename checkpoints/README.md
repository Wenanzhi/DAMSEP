# Pretrained DAMSEP checkpoint

| Item | Value |
| --- | --- |
| File | [dars_mixed_p10/best.pth](dars_mixed_p10/best.pth) |
| Size | 29,069,782 bytes |
| SHA-256 | `f03aa104551e2b8043f7c198b26c31e2497c55ac4bd3afa31391e4b921f39ebe` |
| Sample rate | 8 kHz |
| Sources | 2 |
| Training assignment | Fixed distance order (`no_pit`) |
| Loss weights | `w_rev=0.1`, `w_recon=0.5`, `w_rir=0` |
| Early-stopping patience | 10 |

Use the architecture arguments in [configs/dars.yml](../configs/dars.yml)
to load these weights. See the
[loading example](../README.md#pretrained-checkpoint-and-model-outputs).
The model class remains named `SPMamba` for checkpoint compatibility.

The serialized object contains `model_name`, `state_dict`, `model_args`, and
software-version metadata. This is an exported model, not a Lightning training
checkpoint with optimizer and scheduler state.

The default training configuration uses an early-stopping patience of 5,
whereas this checkpoint was trained with a patience of 10. See the
[configuration notes](../README.md#configuration-notes) for the source-assignment
difference between the released recipe and the manuscript description.
