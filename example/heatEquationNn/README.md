# Heat-equation neural coefficient closure (`heatEquationNn`)

This example embeds an [alpakaNN](../../README.md) gated-SiLU MLP behind a
conservative finite-volume heat solver. The host orchestrates feature packing,
inference and the conservative stencil on **one device queue**; the solver
state, face coefficients and arithmetic are `double`, the network input/weights
are `float`. There is **no device-to-host copy of the temperature field on the
inference path**; host copies happen only for snapshots, CSV export and final
diagnostics.

Two learned models are documented in `models/heat_closure/`: **v1** (raw
`[u,x,y]` contract baseline) and **v2 / variant B** (Fourier-encoded, 27 inputs,
the recommended model). Both are **demonstrators, not validated accurate
closures**: the field targets are not met, and `max|err|` is provably out of
reach for a continuous model (see [Model contract](#model-contract) and
`../../docs/heat_closure_report.md`); v2 does meet the coefficient MAE target.

The trained binaries `models/heat_closure/weights.bin` and
`models/heat_closure/weights_v2.bin` are **intentionally not committed** (the
operator decision is "do not check in the model binary; everyone can retrain
it"). Only the small documentation manifests
(`weights.bin.metadata.json`, `weights_v2.bin.metadata.json`) and
`models/heat_closure/README.md` are tracked. The C++ tests and this example build
and pass on a fresh checkout without any pre-existing model: tests generate
deterministic v1 and v2 in-test fixtures, and `--material nn` runs whenever you
pass a retrained `--weights FILE`. See
[section 2](#2-train-or-regenerate-the-model-external-pytorch) for the exact
retrain commands.

Everything below was verified in the `device-resident-runtime` worktree; each
command is replayable verbatim from the repository root.

## 1. Build (host, pedantic)

Requires CMake >= 3.25, a C++20 compiler, OpenBLAS (host BLAS) and the network
access needed to fetch the pinned `alpakaVendor`/`alpaka3` dependencies. The
example is off by default:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dalpaka_COMPILE_PEDANTIC=ON \
  -DalpakaNN_BUILD_TESTS=ON \
  -DalpakaNN_BUILD_HEAT_CLOSURE_EXAMPLE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The host build uses the repository default alpaka configuration (host CPU +
OpenMP) and reports:

- GNU 13.3.0, alpaka 3.0.0, OpenMP 4.5, OpenBLAS 0.3.26, FFTW 3.3.10.
- `ctest` on the exact matrix commit (see `docs/heat_closure_report.md`)
  reported **57/57 passed** locally and **105/105** on the dev-hal host build.
- `alpaka_COMPILE_PEDANTIC=ON` is defaulted by the repository root and is
  repeated here so `-Werror` feedback matches CI.

The executable is written to
`build/example/heatEquationNn/heatEquationNn`.

## 2. Train or regenerate the model (external PyTorch)

**The model binaries are not committed.** `models/heat_closure/weights.bin` and
`models/heat_closure/weights_v2.bin` (and their generated `.inference_parity.csv`
/ `.history.json` sidecars) are `.gitignore`d and absent from a fresh checkout;
regenerate them with `tools/train_heat_closure.py` before running an NN
model-quality comparison that needs the trained checkpoint. The tests do not need
them (they build v1 and v2 fixtures), and `--material nn` only needs whatever
`--weights FILE` you pass. Committing a regenerated binary is not required; if
you deliberately want to track one, use
`git add -f models/heat_closure/weights.bin`.

Training is **outside** alpakaNN and is not part of the C++ build. The tool is
`tools/train_heat_closure.py` (dataset generation and metadata/serialization do
not require PyTorch; only `train` does).

PyTorch versions actually observed in the run logs:

- v1 final model fine-tuning: `torch 2.5.1+rocm6.2` on an AMD Radeon RX 7900 XTX
  (`device` in `models/heat_closure/weights.bin.metadata.json` is
  `cuda:AMD Radeon RX 7900 XTX`).
- v2 variant B training: `torch 2.5.1+rocm6.2`, `device = cuda`, 1000 epochs
  (recorded in `models/heat_closure/weights_v2.bin.metadata.json`).
- Host inference-parity recomputation: `torch 2.5.1+cpu`.
- No CUDA PyTorch install was used for any recorded run; the NVIDIA A30 was used
  only for the CUDA **C++/alpaka** backend.

The matching install commands for those versions are:

```sh
# AMD ROCm 6.2 (the GPU training environment)
python3 -m pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch==2.5.1

# CPU-only (dataset generation, parity recomputation, CPU training)
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
```

Verify what you have (`python3 -c "import torch; print(torch.__version__)"`)
before training; any different build will change the reported device label.

Generate a dataset and train the gated-SiLU model (defaults reproduce the
architecture the tracked `weights.bin.metadata.json` documents; the recorded
checkpoint was fine-tuned with a longer, region-weighted schedule, see that
manifest). The `train` subcommand runs on CPU (the recorded checkpoint was
fine-tuned on a ROCm GPU out-of-band; the C++ example does not care which device
trained it). This is exactly how to regenerate the uncommitted binary:

```sh
python3 tools/train_heat_closure.py dataset --output /tmp/heat_closure.csv
python3 tools/train_heat_closure.py train \
  --csv /tmp/heat_closure.csv \
  --output models/heat_closure/weights.bin \
  --seed 0 --epochs 200 --batch-size 4096 --lr 1e-3
```

The exporter writes `weights.bin` (float32, little-endian, row-major `[in,out]`,
`gate`/`up`/`down` concatenated), `weights.bin.metadata.json` and
`weights.history.json`, plus a fixed-input parity CSV. All of these except the
metadata manifest are `.gitignore`d so a retrain stays untracked. After
retraining, the runtime loader accepts the model only if the metadata contract
matches; a different `weights.bin` also requires the checksum in the metadata.

## 3. Run

The interface is (see `heatEquationNn --help`):

```text
--grid N --tmax T --steps N --material uniform|preset|nn --beta B
--alpha-min A --alpha-max A --weights FILE --output DIR --export-alpha CSV
--frames N --no-output --validate-strict --backend host|hip|cuda|oneapi|<device-index>
--dump-features CSV [--samples N --seed N --temperature-samples N]
```

Reference and dataset modes run without trained weights; `--material nn`
requires `--weights`. Timesteps are chosen automatically for stability unless
`--steps` is given (unstable requests are rejected). Each backend writes its own
output directory.

### Uniform, preset and NN runs (the Section 12 invocations)

```sh
HEAT=build/example/heatEquationNn/heatEquationNn

$HEAT --material uniform --grid 64 --tmax 0.1 --validate-strict
$HEAT --material preset  --grid 64 --tmax 0.1 --frames 121 --output results/preset
$HEAT --material nn --weights models/heat_closure/weights.bin \
  --grid 64 --tmax 0.1 --frames 121 --output results/nn
```

### `--export-alpha` (coefficient maps / region metrics)

Writes one streaming CSV over all saved snapshots, header
`x,y,u,alpha,mode,time,step`, including the `t = 0` state. It cannot be combined
with `--no-output` (rejected with a clear error):

```sh
$HEAT --material nn --weights models/heat_closure/weights.bin \
  --grid 64 --tmax 0.1 --frames 121 \
  --output results/nn --export-alpha results/nn/alpha.csv
```

### `--dump-features` (dataset generation from the authoritative C++ evaluator)

```sh
$HEAT --dump-features /tmp/heat_closure_cpp.csv \
  --samples 100000 --seed 0 --temperature-samples 11 --beta 0.5
```

This writes `/tmp/heat_closure_cpp.csv` with columns `u,x,y,alpha_true` and a
sidecar `/tmp/heat_closure_cpp.csv.metadata.json` recording `beta`, bounds, the
seed, the sample count and the sampling rule.

### Compute-only timing

`--no-output` disables frame export and prints `total_seconds`,
`seconds_per_step`, `steps` and `nn_per_step_d2h=no`; snapshot copies for
visualization are reported separately as `snapshots_d2h=yes`.

```sh
$HEAT --material nn --weights models/heat_closure/weights.bin \
  --grid 64 --tmax 0.1 --no-output
```

### Backend selection

The selector is `host` (default), `hip`/`amd`/`amdgpu`, `cuda`,
`oneapi`/`sycl`, or a bare `0`-based accelerator index. A backend that was not
compiled in is rejected at runtime with the list of compiled backends; a
compiled-in backend with no visible device is also rejected.

Exact known-working configure commands on the dev-hal host (`terok-dev`), all
with `-Dalpaka_COMPILE_PEDANTIC=ON ... -DalpakaNN_BUILD_TESTS=ON
-DalpakaNN_BUILD_HEAT_CLOSURE_EXAMPLE=ON -DCMAKE_BUILD_TYPE=Release` and
`cmake --build <dir> -j32`:

**HIP** (AMD Radeon RX 7900 XTX, gfx1100, ROCm 7.2.4; `ctest` 81/81):

```sh
cmake -S . -B build-rocm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -Dalpaka_COMPILE_PEDANTIC=ON \
  -DalpakaNN_BUILD_TESTS=ON -DalpakaNN_BUILD_HEAT_CLOSURE_EXAMPLE=ON \
  -Dalpaka_DEP_HIP=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_HIP_AmdGpu=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_PLATFORM=amd \
  -Dhip_DIR=/opt/rocm-7.2.4/lib/cmake/hip
$HEAT --backend hip --material nn --weights models/heat_closure/weights.bin \
  --grid 64 --tmax 0.1 --no-output
```

**CUDA** (NVIDIA A30, sm_80, CUDA 13.4.92; `ctest` 81/81):

```sh
PATH=/usr/local/cuda/bin:$PATH cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -Dalpaka_COMPILE_PEDANTIC=ON \
  -DalpakaNN_BUILD_TESTS=ON -DalpakaNN_BUILD_HEAT_CLOSURE_EXAMPLE=ON \
  -Dalpaka_DEP_CUDA=ON -Dalpaka_DEP_HIP=OFF -Dalpaka_DEP_OMP=OFF \
  -Dalpaka_CUDA_NvidiaGpu=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
$HEAT --backend cuda --material nn --weights models/heat_closure/weights.bin \
  --grid 64 --tmax 0.1 --no-output
```

**SYCL / oneAPI** (Intel Arc A770, `opencl:gpu`, oneAPI 2026.1.1; `ctest`
81/81 with the device pinned; the Arc has no native fp64 so the solver's
`double` path runs under IGC FP64 emulation):

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -Dalpaka_COMPILE_PEDANTIC=ON \
  -DalpakaNN_BUILD_TESTS=ON -DalpakaNN_BUILD_HEAT_CLOSURE_EXAMPLE=ON \
  -Dalpaka_DEP_ONEAPI=ON -Dalpaka_DEP_HIP=OFF -Dalpaka_DEP_CUDA=OFF -Dalpaka_DEP_OMP=OFF \
  -Dalpaka_ONEAPI_Cpu=OFF -Dalpaka_ONEAPI_IntelGpu=ON \
  -Dalpaka_ONEAPI_NvidiaGpu=OFF -Dalpaka_ONEAPI_AmdGpu=OFF \
  -DCMAKE_CXX_COMPILER=icpx
ONEAPI_DEVICE_SELECTOR=opencl:gpu \
IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1 \
  $HEAT --backend oneapi --material nn --weights models/heat_closure/weights.bin \
  --grid 64 --tmax 0.1 --no-output
```

`ONEAPI_DEVICE_SELECTOR=opencl:gpu` is **required**: default oneAPI device
discovery on that host crashes inside the broken level_zero enumeration (an
environment/vendor defect), and the A30 is not reachable from this oneAPI
install because no CUDA/level_zero UR adapter ships with it. See
`docs/heat_closure_report.md` for the measured backend table.

## 4. Model contract

A runtime model is a companion pair `weights.bin` + `weights.bin.metadata.json`.
Only the metadata manifest is tracked here; supply or regenerate the binary (see
section 2). The loader in `src/ModelLoader.hpp` reads the width, input
dimension, feature encoding and weight shapes from the metadata; nothing about
the architecture is hardcoded. Two metadata formats are accepted.

### v1 — raw contract (`alpakaNN-heat-closure-f32-v1`)

- `architecture = "gated_silu_bias_free_v1"`, `dtype = "float32"`.
- `feature_order = ["u","x","y"]`; output
  `alpha_min + (alpha_max - alpha_min) * sigmoid(z)`; weight layout
  `row-major [in,out], little-endian float32; gate,up,down concatenated`;
  shapes `[3,64]`, `[3,64]`, `[64,1]` (1792 bytes, 448 floats).
- `format` is the discriminator; `feature_encoding` is optional and defaults to
  `raw`.

### v2 — Fourier-encoded contract (`alpakaNN-heat-closure-f32-v2`)

- Same architecture, dtype, output equation and weight-layout string.
- `feature_encoding = "fourier_xy_k0_5"` (required for v2).
- `feature_order` is the fully expanded 27-name order produced by the training
  harness: `["u","x","y","sin1pi_x","cos1pi_x","sin1pi_y","cos1pi_y", ...]`
  with `k = 0..5` terms `sin2^k pi coord`, `cos2^k pi coord` for
  `coord in (x, y)`. The compact `["u","x","y"]` form is also accepted.
- shapes `[27,64]`, `[27,64]`, `[64,1]` (14080 bytes, 3520 floats):
  `Wgate,Wup [27,64]`, `Wdown [64,1]`.

Common validation (both formats): `beta` matching `--beta` exactly
(`beta_mismatch_policy = "reject"`), non-negative finite bounds with
`alpha_max >= 4*(1+beta)`, `feature_encoding ∈ {raw, fourier_xy_k0_5}`,
consistent `weight_shapes` and total byte size `4*sum(products)`, and a weight
payload whose SHA-256 equals `weights_sha256` (all floats finite). Unknown
formats or encodings are rejected.

The forward pass is bias-free and identical in PyTorch and C++:

```text
f = encode(u, x, y)    # raw: [u, x, y]; v2: 27 Fourier columns
g = f Wgate            Wgate: [in, 64]
p = f Wup              Wup:   [in, 64]
h = SiLU(g) ⊙ p
z = h Wdown            Wdown: [64, 1]
alpha_NN = alpha_min + (alpha_max - alpha_min) sigmoid(z)
```

`PackFeatures` (device) and `NeuralInference.hpp` (host parity) build the same
feature matrix; for v2 the `sin`/`cos` terms are evaluated with device math in
the solver kernel, so the inference path stays on the accelerator.

### Retraining the v2 (variant B) model

`tools/train_heat_closure.py train --variant B` emits the v2 metadata and
binary; `--variant raw` (default) emits the v1 contract:

```sh
python3 tools/train_heat_closure.py dataset --output /tmp/heat_closure.csv
python3 tools/train_heat_closure.py train --variant B \
  --csv /tmp/heat_closure.csv \
  --output models/heat_closure/weights_v2.bin \
  --seed 0 --epochs 200 --batch-size 4096 --lr 1e-3
```

**The trained binary is not committed** (the operator decision is "do not check
in the model binary; everyone can retrain it"); only the small documentation
manifests (`weights.bin.metadata.json`, `weights_v2.bin.metadata.json`) and
`models/heat_closure/README.md` are tracked; the binaries are `.gitignore`d. The
tracked `weights_v2.bin.metadata.json` documents the recommended trained variant-B
checkpoint described in `models/heat_closure/README.md` (it also carries the
recorded PyTorch-GPU training record; the committed tool reproduces the contract,
not necessarily bit-identical weights). The C++ tests self-generate a v1 and a
v2 fixture, so a fresh checkout builds and passes **without any model binary**.
When you pass a retrained `--weights FILE`, `--material nn` accepts either
format and takes the alpha bounds from that metadata.

The coefficient is computed from the current state **before the first update**
and recomputed every step; solver arithmetic stays `double` while network I/O
stays `float`.

## 5. Metrics and limitations

Two model contracts are documented. **v1** is the original raw `[u,x,y]`
contract baseline; **v2 (variant B)** is the recommended Fourier-encoded model.
Both miss the field targets; v2 meets the coefficient MAE target but not the
`max|err|` target, which is provably unreachable for any continuous model on
this geometry.

### v2 (variant B, recommended)

The tracked `models/heat_closure/weights_v2.bin.metadata.json` documents the
trained variant-B checkpoint (`weights_sha256 = 83c98b08…`); see
`models/heat_closure/README.md`.

| split | coefficient MAE | coefficient max abs | targets 0.30 / 1.20 |
|---|---|---|---|
| validation | 0.1695 | 3.6900 | MAE PASS, max FAIL |
| clean holdout | **0.1323** | 2.0000 | **MAE PASS**, max FAIL |
| test | 0.1693 | 3.5832 | MAE PASS, max FAIL |

Field vs analytical preset (grid 64, `tmax 0.1`, matched schedule):
final `relL2 0.0500` / normalized `Linf 0.1925` versus Section 7.2 targets
1 % / 2 % — **FAIL**, but both improved versus v1. The worst transient `Linf` is
0.4403 (early transients), i.e. not uniformly better.

### v1 (original contract baseline, for comparison)

| split | MAE | max abs | target |
|---|---|---|---|
| validation | 0.4527 | 4.4188 | FAIL |
| clean holdout | 0.4522 | 4.8138 | FAIL |
| test | 0.4673 | 4.4900 | FAIL |

On the same instantaneous NN final state, v1's exported `alpha.csv` gives overall
MAE 0.3181 / max abs 3.2761, with the conductor region (MAE 2.1184) dominating;
v1 same-grid final-field error is `relL2` 0.0911 / normalized Linf 0.2539.

### Why `max|err|` cannot be met by a continuous model

The ground-truth map is **discontinuous** (0.02 inclusion, 4.0 conductor,
0.5/0.9 stripe), while both models are small continuous gated-SiLU networks.
The inclusion/conductor interface is a coefficient jump of `J = (4.0-0.02)*1.5
≈ 5.97`, so any continuous approximant pays at least `J/2 ≈ 2.99` across that
interface — above the 1.20 target by construction. The separate experiments under
`/workspace/heat_closure_variantB` confirm this empirically: a
discontinuity-aware MoE / region classifier can meet both targets only by
hardening the classification (and only on the clean holdout), separated-geometry
and symmetry-feature variants still fail `max|err|` (3.4–3.7), and all tested
continuous variants fail it. These are recorded as integration evidence, not as
a production accuracy claim.

The correct consequence is to treat this as an integration/portability
demonstrator, not a physical closure. The regression test reports the measured
numbers and deliberately does not enforce the unmet targets.

Other limitations, stated plainly:

- This is an integration and portability demonstration. The analytical material
  formula is cheap; replacing it with a network does **not** by itself establish
  a speedup, and no speedup is claimed.
- Precision: solver state and face coefficients are `double`; network
  input/weights/arithmetic are `float` (bounded by
  `alpha_min + (alpha_max-alpha_min) sigmoid`). FP64-incapable devices (Intel
  Arc) require IGC emulation, which was used for the recorded SYCL runs.
- Synchronization: the public alpakaNN host MLP path used by this example
  enqueues its GEMM/SwiGLU/GEMM stages on the caller queue and the solver waits
  only at measurement/output boundaries; no per-step synchronization is claimed
  beyond that, and no per-step D2H copy exists (`nn_per_step_d2h=no`). Snapshot
  and CSV export copies are explicitly separate (`snapshots_d2h=yes`).
- Backends not tested by the recorded matrix: CUDA/oneAPI targeting the A30
  through oneAPI is unvalidated (no UR adapter in that install); any other
  device/OS is untested. Unavailable selectors are rejected, never silently
  substituted.
