# Heat-equation neural coefficient closure (`heatEquationNn`)

This example embeds an [alpakaNN](../../README.md) gated-SiLU MLP behind a
conservative finite-volume heat solver. The host orchestrates feature packing,
inference and the conservative stencil on **one device queue**; the solver
state, face coefficients and arithmetic are `double`, the network input/weights
are `float`. There is **no device-to-host copy of the temperature field on the
inference path**; host copies happen only for snapshots, CSV export and final
diagnostics.

The learned model in `models/heat_closure/` is a **demonstrator, not a validated
accurate closure**: the training pipeline's own acceptance verdict is FAIL (see
[Model contract](#model-contract) and `../../docs/heat_closure_report.md`).

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

Training is **outside** alpakaNN and is not part of the C++ build. The tool is
`tools/train_heat_closure.py` (dataset generation and metadata/serialization do
not require PyTorch; only `train` does).

PyTorch versions actually observed in the run logs:

- Final model fine-tuning: `torch 2.5.1+rocm6.2` on an AMD Radeon RX 7900 XTX
  (`device` in `models/heat_closure/weights.bin.metadata.json` is
  `cuda:AMD Radeon RX 7900 XTX`).
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

Generate a dataset and train the exact gated-SiLU model (defaults reproduce the
shipped architecture; the shipped checkpoint was fine-tuned with a longer,
region-weighted schedule, see `weights.bin.metadata.json`):

```sh
python3 tools/train_heat_closure.py dataset --output /tmp/heat_closure.csv
python3 tools/train_heat_closure.py train \
  --csv /tmp/heat_closure.csv \
  --output models/heat_closure/weights.bin \
  --device cpu --seed 0 --epochs 200 --batch-size 4096 --lr 1e-3
```

`--device cuda` uses CUDA/HIP when PyTorch can see it; `--device cpu` is always
available. The exporter writes `weights.bin` (float32, little-endian, row-major
`[in,out]`, `gate`/`up`/`down` concatenated), `weights.bin.metadata.json` and
`weights.history.json`, plus a fixed-input parity CSV. After retraining, the
runtime loader accepts the model only if the metadata contract matches; a
different `weights.bin` also requires the checksum in the metadata and the test
fixtures to be regenerated together.

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
The loader in `src/ModelLoader.hpp` accepts only:

- `format = "alpakaNN-heat-closure-f32-v1"`, `kind = "trained_model"`,
  `architecture = "gated_silu_bias_free_v1"`, `dtype = "float32"`.
- `feature_order = ["u","x","y"]`; output
  `alpha_min + (alpha_max - alpha_min) * sigmoid(z)`; weight layout
  `row-major [in,out], little-endian float32; gate,up,down concatenated`;
  shapes `[3,64]`, `[3,64]`, `[64,1]` (1792 bytes).
- `beta` matching `--beta` exactly (`beta_mismatch_policy = "reject"`), and
  non-negative finite bounds with `alpha_max >= 4*(1+beta)`.
- A weight payload whose size and SHA-256 equal the metadata
  (`weights_sha256`); the weights are checked for finiteness.

The forward pass is bias-free and identical in PyTorch and C++:

```text
f = [u, x, y]
g = f Wgate            Wgate: [3, 64]
p = f Wup              Wup:   [3, 64]
h = SiLU(g) ⊙ p
z = h Wdown            Wdown: [64, 1]
alpha_NN = alpha_min + (alpha_max - alpha_min) sigmoid(z)
```

The coefficient is computed from the current state **before the first update**
and recomputed every step; solver arithmetic stays `double` while network I/O
stays `float`.

## 5. Metrics and limitations

**Acceptance is FAIL.** The metadata block `acceptance` records
`verdict = "FAIL"` for both the validation and the clean-holdout splits against
the Section 7.4 targets `MAE <= 0.05*alpha_max = 0.30` and
`max_abs <= 0.20*alpha_max = 1.20`:

| split | MAE | max abs | target |
|---|---|---|---|
| validation | 0.4527 | 4.4188 | FAIL |
| clean holdout | 0.4522 | 4.8138 | FAIL |
| test | 0.4673 | 4.4900 | FAIL |
| spatial-eval | 0.3363 | 1.2519 | (diagnostic) |

On the same instantaneous NN final state, the exported `alpha.csv` gives
overall MAE 0.3181 / max abs 3.2761, with the conductor region (MAE 2.1184)
dominating; on the older model in the primary matrix the same-state MAE is
0.3904 / max 2.9996. Same-grid NN-vs-preset final-field error is `relL2` 0.0911
/ normalized Linf 0.2539 (final model) versus the Section 7.2 targets 1 % / 2 %
— also FAIL. The regression test reports these numbers and deliberately does
not enforce the unmet targets.

Why it fails: the ground-truth map is **discontinuous** (a 0.02 inclusion and a
4.0 conductor embedded in a 0.5/0.9 stripe), while the model is a small smooth
bias-free gated-SiLU network. A smooth, approximately bias-free regressor has a
global-MAE floor of roughly **0.343** on this random state distribution even
with a perfect conductor fit; the discontinuous interfaces cannot be represented
at network resolution. The correct consequence is to treat this as an
integration/portability demonstrator, not a physical closure.

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
