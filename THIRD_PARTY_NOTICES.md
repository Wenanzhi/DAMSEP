# Third-party notices

This repository contains adaptations of research software from the following
projects. Existing source-file notices have been retained.

## SPMamba

- Project: <https://github.com/JusperLee/SPMamba>
- Use in DARS: separation backbone, training framework, and Look2Hear utility
  code.
- License: Apache License 2.0. The repository-level `LICENSE` file contains the
  applicable text.

## Rec-RIR

- Project: <https://github.com/Audio-WestlakeU/Rec-RIR>
- Use in DARS: CTF/RIR response-estimation blocks and the reverberant
  reconstruction formulation, subsequently adapted for source-specific joint
  separation.
- License: MIT, Copyright (c) 2025 Audio-WestlakeU. The complete license is in
  `licenses/Rec-RIR-LICENSE`.

The Rec-RIR-derived support code also retains notices for its upstream
components, including Microsoft RetNet and the lucidrains Conformer
implementation.

## External comparison systems

Files under `evaluation/rir_baselines/` are adapters. They do not vendor the
external model implementations or checkpoints. Obtain Rec-RIR, VINP, BUDDy,
Speech2RIR, and FiNS separately and follow each project's license and model
terms.

## Dataset preparation

The data utilities follow WSJ0-2mix/WHAM-style mixture and room-simulation
conventions and use Pyroomacoustics. Before public release, the provenance and
redistribution terms of these utilities and every source corpus must be checked
against their upstream licenses. No source-corpus audio is included here.
