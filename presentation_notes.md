# Heat-closure presentation notes

Five-minute talk for the `heatEquationNn` demonstrator. Every number below is
measured and is read from
`/tmp/alpakaNN-results/nhc-20261001/T4/summary.json` and the exported CSVs; the
figure/video artifacts were rendered from those files only.

**Message to deliver:** a learned material model participates in every
simulation timestep, using device-resident data and the simulation queue — this
is an **integration and portability demonstration**, not an accurate closure and
not a speedup claim. The prescribed material law is cheap; replacing it with a
network is interesting only where the physical closure is genuinely expensive.

## Deliverable paths

- Figures directory:
  `/tmp/alpakaNN-results/nhc-20261001/T6/figures/` (10 PNGs; inventory in
  `manifest.json`).
- Video: `/tmp/alpakaNN-results/nhc-20261001/T7/heat_closure_demo.mp4`
  (55.0 s, 1650 frames, 1920×1080, H.264 `yuv420p`, 30 fps; full-decode OK).
- Video inspection: `/tmp/alpakaNN-results/nhc-20261001/T7/inspection/`
  (`contact_sheet.png`, `findings.json`, OCR samples).
- Source branch: `device-resident-runtime`; matrix commit
  `4cb3ac246f6c673a889460612c4e1da67958f485`; installed model sha256
  `b3a5f955482cef375ddef1995a9233c3c88bd96fe5748ca0d4f8ad15d572fa79`.

## Five-minute outline (Section 11)

| Part | What to show | What to say |
| --- | --- | --- |
| Problem (30 s) | `figures/setup_diagram.png` | “Our solver needs a local transport coefficient at every timestep.” |
| Baseline (60 s) | `figures/homogeneous_sequence.png` + `comparison_*.png` | “This prescribed material law gives us a controlled reference.” |
| Integration (60 s) | `figures/comparison_*.png`, `figures/final_comparison.png` | “alpakaNN consumes device-resident features and supplies the next solver update through the same queue.” |
| Evidence (90 s) | `figures/coefficient_comparison.png`, `figures/final_comparison.png` | State the measured field and coefficient errors, with their definitions, plus the failed targets. |
| Portability and cost (45 s) | `figures/backend_table.png` | Name the devices actually tested, the skipped backend, and the measured NN overhead. |
| Outlook (15 s) | `figures/final_comparison.png` | “The next application would replace a more expensive physical closure.” |

## Speaker notes

### 1. Problem — geometry and material map (30 s)

Show `figures/setup_diagram.png`. On the unit square we solve the conservative
nonlinear diffusion equation `∂u/∂t = ∇·(α(x,y,u)∇u)`, not `αΔu`: across
material jumps the conservative flux form is required. Left wall `u = 1`, right
wall `u = 0`, top/bottom insulated, interior starts at zero. The true
coefficient is `base(x,y)(1 + βu)`, β = 0.5, with a low-transport circular
inclusion (`base = 0.02`), a short central conductor (`base = 4.0`), and a
0.5/0.9 striped background. It is **discontinuous**.

Say: “Our solver needs a local transport coefficient at every cell, every
timestep. Here the law is prescribed and cheap — we are demonstrating the
integration path.”

### 2. Baseline — analytical reference (60 s)

Show `figures/homogeneous_sequence.png` (uniform coefficient) and one of
`figures/comparison_00000{1..4}.png`. The heated left wall drives a front from
left to right; the inclusion deflects it and the conductor enhances transport.
Describe only features visible in the data. The baseline uses the analytical
preset coefficient on a cell-centered finite-volume grid with shared harmonic
face coefficients, a half-cell Dirichlet wall flux, and automatic stable
timesteps (64², `tmax = 0.1`, 13654 steps, `dt = 7.32386e-06`).

### 3. Integration — feature → inference → coefficient → conservative update (60 s)

The host orchestrates on **one device queue**. Once per timestep: pack the
minimal feature vector `[u, x, y]` in a device kernel, run the billinear
bias-free gated-SiLU network in `float`, cast the bounded output
`α_min + (α_max−α_min)·sigmoid(z)` into the `double` coefficient buffer in a
device kernel, then advance `uⁿ → uⁿ⁺¹` with conservative fluxes and swap
buffers. Weights are uploaded once; buffers are allocated once. The inference
path performs **no per-step device-to-host copy** of the temperature field;
host copies exist only for snapshots/CSV export and final diagnostics
(`nn_per_step_d2h=no`, `snapshots_d2h=yes`). Training happens externally in
PyTorch; inference happens in C++ with alpakaNN.

### 4. Evidence — measured errors, with definitions (90 s)

Show `figures/coefficient_comparison.png` (true vs learned coefficient on the
same final NN state, plus absolute error; fixed display range `[0,4.5]`, fixed
error range `[0,3.5]`, true max error printed) and `figures/final_comparison.png`.

Definitions:

- `relL2 = sqrt(Σ(u_NN − u_ref)² / Σ u_ref²)` on the final field.
- `normalized Linf = max|u_NN − u_ref| / (u_hot − u_cold)`, here `/1.0`.
- coefficient MAE / max abs are `|α_NN − α_true|` evaluated on the **same**
  instantaneous `u_NN`.

Measured (grid 64², 13654 steps; final installed model `b3a5f955…`, host-only
supplementary run):

- Same-grid field vs preset: **relL2 0.091123, normalized Linf 0.253858**.
  Section 7.2 targets are 0.01 / 0.02 → **FAIL**.
- Same-state coefficient: **overall MAE 0.318084, max abs 3.276115**;
  conductor MAE 2.118381, max 3.276115; inclusion MAE 0.497027; stripe_high
  MAE 0.324676; background_low MAE 0.226603. Section 7.4 targets are
  MAE ≤ 0.30 and max ≤ 1.20 → **FAIL**.
- Mesh convergence with the analytical coefficient (separate from NN error):
  32/64 relL2 0.015234, 64/128 0.007430, 128/256 0.003351.
- Fixed-input PyTorch-vs-C++ parity: max abs 7.662e-06 (within atol 1e-5).

Say plainly: the coefficient targets and same-grid field targets **fail** for
the final model. The truth is discontinuous (0.02 inclusion, 4.0 conductor)
while the model is a small smooth bias-free gated-SiLU network; a smooth,
approximately bias-free regressor has a global-MAE floor of roughly **0.343**
on this state distribution even with a perfect conductor fit. This is a
demonstrator, not an accurate closure. Do not call it learned physics.

### 5. Portability and measured cost (45 s)

Show `figures/backend_table.png`. Devices actually tested, all with a pedantic
build and end-to-end NN+preset runs:

| backend | device | toolchain | tests | inference parity vs host |
| --- | --- | --- | --- | --- |
| host | x86_64 CPU | GNU 13.3.0 | 57/57 | — |
| hip | AMD RX 7900 XTX, gfx1100 | ROCm 7.2.4 | 81/81 | ≤ 3.0e-09 (NN) |
| cuda | NVIDIA A30, sm_80 | CUDA 13.4.92 | 81/81 | ≤ 5.1e-09 (NN) |
| oneapi | Intel Arc A770, `opencl:gpu`, IGC FP64 emulation | oneAPI 2026.1.1 | 81/81 | ≤ 1.4e-08 (NN) |

Compute-only timings (`--no-output`, 13654 steps, `per_step_d2h = no`), with
measured NN-overhead ratio vs the analytical preset on the same backend:

| backend | preset s | nn s | nn / preset |
| --- | --- | --- | --- |
| host | 2.040714 | 55.114960 | 27.0× |
| hip | 0.176850 | 14.144359 | 80.0× |
| cuda | 0.133986 | 15.758807 | 117.6× |
| sycl | 3.113811 | 9.961625 | 3.20× |

Skips: oneAPI targeting the A30 is **SKIP** — this oneAPI install has no
CUDA/level_zero UR adapter, and default oneAPI device discovery segfaults on
the host (an environment/vendor defect), so the Arc is selected with
`ONEAPI_DEVICE_SELECTOR=opencl:gpu`. The Arc has no native fp64, so the
`double` solver runs under `IGC_EnableDPEmulation=1
OverrideDefaultFP64Settings=1`.

Say: the NN path is 3–118× the cheap analytical formula here; that is the
honest measured overhead. No speedup is claimed. The demonstrated benefit is
integration with existing alpaka buffers, queue ordering, C++ kernels and the
tested backend ecosystem.

### 6. Outlook — final card (15 s)

Show the closing card at the end of the video. Next step: replace a genuinely
expensive physical closure with the same integration path, keeping the
conservative flux formulation and the positive bounded output. The video is
accelerated playback of simulation time built from real saved frames (nearest
saved timestamp per segment, frame holds, no numerical interpolation); it is
not real-time execution and not proof of speed.

## Running the artifacts yourself

The full build/train/run commands are in
`example/heatEquationNn/README.md`; the numerical evidence is in
`docs/heat_closure_report.md`. To re-render the figures and video from the
verified arrays:

```sh
python3 tools/render_heat_closure.py \
  --runs /tmp/alpakaNN-results/nhc-20261001/runs \
  --summary /tmp/alpakaNN-results/nhc-20261001/T4/summary.json \
  --final-nn /tmp/alpakaNN-results/nhc-20261001/runs-finalmodel/nn \
  --output /tmp/alpakaNN-results/nhc-20261001/T6/figures

python3 tools/make_heat_closure_video.py \
  --frames /tmp/alpakaNN-results/nhc-20261001/T6/figures \
  --summary /tmp/alpakaNN-results/nhc-20261001/T4/summary.json \
  --output /tmp/alpakaNN-results/nhc-20261001/T7/heat_closure_demo.mp4
```

The video uses 30 fps, H.264, `yuv420p`, even 1920×1080 dimensions; the
uncompressed PNG sequence is kept under `T7/presentation_frames/`.
