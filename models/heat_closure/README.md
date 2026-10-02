<!--
SPDX-FileCopyrightText: 2026 René Widera
SPDX-License-Identifier: MPL-2.0
-->

# Heat-closure models

This directory tracks only the small **metadata manifests** and this README.
The trained weight binaries are intentionally **not committed** (operator
decision: "do not check in the model binary; everyone can retrain it"); the
`weights*.bin` files are `.gitignore`d. Both models are retrainable with
`tools/train_heat_closure.py` (v1) or the variant-B training harness (v2).

## Which model to use

| model | file | contract | role |
| --- | --- | --- | --- |
| v1 | `weights.bin` + `weights.bin.metadata.json` | raw `[u,x,y]`, `alpakaNN-heat-closure-f32-v1`, 1792 B / 448 floats | original contract baseline (for comparison) |
| **v2 (variant B)** | `weights_v2.bin` + `weights_v2.bin.metadata.json` | Fourier `fourier_xy_k0_5`, 27 inputs, `alpakaNN-heat-closure-f32-v2`, 14080 B / 3520 floats | **recommended** |

Both are the same bias-free gated-SiLU MLP (`Wgate/Wup [in,64]`, `Wdown [64,1]`,
`alpha = alpha_min + (alpha_max - alpha_min)*sigmoid(z)`); only the input basis
changes. The loader in `example/heatEquationNn/src/ModelLoader.hpp` selects the
contract from the `format` / `feature_encoding` metadata fields, so the same
`--material nn --weights FILE` invocation serves either model.

## v2 (variant B) metrics

Trained checkpoint, documented by `weights_v2.bin.metadata.json`
(`weights_sha256 = 83c98b08abe90eb45fc3da1a4adfb599294a1e35332535590a6b4e445de958b0`):

- **Coefficient MAE**: validation 0.1695, **clean holdout 0.1323** — target
  `MAE <= 0.30` **PASS**.
- **Coefficient max abs**: validation 3.690, clean holdout 2.000 — target
  `max|err| <= 1.20` **NOT met**.
- **Field vs analytical preset** (grid 64, `tmax 0.1`): final `relL2` 0.050 /
  normalized `Linf` 0.192 — Section 7.2 targets 0.01 / 0.02 **NOT met** (both
  improved versus v1).

The `max|err|` target is **not met by any continuous model** and is provably
unreachable for this geometry: the inclusion/conductor interface is a
coefficient jump of `J = (4.0-0.02)*1.5 ≈ 5.97`, so a continuous approximant
pays at least `J/2 ≈ 2.99` there. See
`../../docs/heat_closure_report.md` and the experiments under
`/workspace/heat_closure_variantB` (MoE/region classifier, separated geometry,
symmetry features) for the evidence. The deliverable is an
**integration/portability demonstrator**, not an accurate physical closure.

## Regenerating the binaries

v1 (raw contract), from the repository root:

```sh
python3 tools/train_heat_closure.py dataset --output /tmp/heat_closure.csv
python3 tools/train_heat_closure.py train \
  --csv /tmp/heat_closure.csv \
  --output models/heat_closure/weights.bin --seed 0 --epochs 200 --batch-size 4096
```

v2 (variant B Fourier contract) uses the variant-B harness
`train_variantB_heat_closure.py` (recorded on the dev host; not committed):

```sh
python3 train_variantB_heat_closure.py gen-data --out /tmp/heat_closure_v2/data --seed 0
python3 train_variantB_heat_closure.py train --data /tmp/heat_closure_v2/data \
  --out models/heat_closure/weights_v2.bin --variant B \
  --epochs 1000 --patience 120 --batch-size 4096 --lr 1e-3 --seed 0
```

Both exporters write the binary plus its `.metadata.json`; the metadata
`weights_file` must match the binary name (`weights_v2.bin` here) and
`weights_sha256` must match its bytes. The C++ tests and the example build and
pass on a fresh checkout with **no** binary present: the tests generate
deterministic v1 and v2 fixtures.
