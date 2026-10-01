# Heat-closure validation report (`heatEquationNn`)

This report covers the `device-resident-runtime` worktree. It states what was
measured, on which device, and what is skipped. Values are read from the
machine-readable artifacts listed at the end; nothing is smoothed, re-scaled or
inferred.

**Verdict: the integration, portability and solver checks pass; the neural
accuracy targets FAIL.** The shipped model is a demonstrator, not an accurate
closure.

## Provenance / measured configuration

| item | value |
|---|---|
| branch | `device-resident-runtime` |
| primary matrix commit | `4cb3ac246f6c673a889460612c4e1da67958f485` (base `8ecc9e0`) |
| installed model commit | `2bea01d0edbb70962e353b0bd20fac4b0d44f96a` (weights sha256 `b3a5f955…`) |
| primary-matrix weights | sha256 `f075cffd50324e17434713d5f9479fb02375eefc712565e2dbfcfd54ac82bf01` |
| grid / tmax / frames / steps | 64² / 0.1 / 121 / 13654, dt 7.32386e-06 |
| host | x86_64, GNU 13.3.0, OpenMP 4.5, OpenBLAS 0.3.26, Release, pedantic |
| build gate | `-Dalpaka_COMPILE_PEDANTIC=ON` on every recorded configure |

The **primary** matrix (all backends) used model `f075cffd…`, the weights present
at run time; the retrained checkpoint `b3a5f955…` was not yet committed into the
matrix worktree. Because the numbers differ, both are reported. The
**final model** `b3a5f955…` is what `models/heat_closure/` now ships; its
same-grid and same-state numbers are the supplementary host-only run in
`summary.json` (`field_t2_final_model`, `coefficient_t2_final_model`).

## 7.1 Solver checks — PASS

Run by `heatEquationNn_strict_solver` and `heatEquationNn_device_solver` (logs
`T12/solver-stdout.log`, `T12/device-solver-stdout.log`); the HIP tree also
passed them (5/5 `heatEquationNn_*`, 81/81 total).

| check | measured | status |
|---|---|---|
| Uniform coefficient vs independent reference (same grid/IC/BC) | relative L2 `< 1e-14` | PASS (target `<= 1e-6`) |
| Constant-field equilibrium with compatible walls | drift `< 1e-14` | PASS |
| Closed (all-insulated) box total heat | change `< 1e-12` | PASS |
| Driven-wall test: Δ(total heat) = Σ boundary input | residual `< 1e-12` | PASS |
| Unstable explicit `--steps` rejected with the documented minimum | exact `minimum steps=` message; `minimum` accepted, `minimum-1` rejected | PASS |
| Half-cell Dirichlet wall factor | independent reference with factor 2 vs 1; relative error `< 1e-14`, wall row provably colder than centre | PASS |
| Heterogeneous spatial refinement (`N=8/16/32`, area-averaged) | `err(32<-16) < err(16<-8)` and both finite | PASS |

Scope note: the reported `<= 1e-6` and `<= 1e-14` are the test's own measured
thresholds (`solver_test.cpp` uses `< 1e-14` for the float-free double paths).
The heterogeneous refinement check asserts monotone decrease, not a convergence
order; at discontinuous interfaces no global second-order rate is claimed.

## 7.2 Same-grid neural comparison — FAIL

Same 64² grid, same 13654-step schedule, same zero initial state, same walls.

| quantity | primary `f075cffd…` | final `b3a5f955…` | Section 7.2 target |
|---|---|---|---|
| final `relL2` | 0.111712 | **0.091123** | <= 0.01 |
| final normalized Linf | 0.249030 | **0.253858** | <= 0.02 |
| max over 121 saved times `relL2` | 0.144207 | 0.106249 | — |
| mean over times `relL2` | 0.105238 | 0.086093 | — |

Both models **fail** the acceptance targets. The regression test
`heatEquationNn_device_solver` computes and prints these metrics on its small
fixed grid (`n=6`, `tmax=0.01`, 20 steps) and reports `relL2=0.31485`,
`normLinf=0.562251`, with `target_enforced=0`; it deliberately does not fake a
pass. The 64² matrix values above are the presentation-relevant numbers.

## 7.3 Independent mesh convergence — PASS (monotone, not order-2)

Analytical preset coefficient, automatic stable steps, fine-grid cell averages
area-restricted to the coarse grid:

| coarse/fine | `relL2` | normalized Linf |
|---|---|---|
| 32/64 | 0.015234 | 0.048766 |
| 64/128 | 0.007430 | 0.035729 |
| 128/256 | 0.003351 | 0.018935 |

The differences decrease under refinement. Because of the discontinuous
inclusion/conductor, no global convergence order is asserted, and this is kept
separate from the NN/preset comparison above.

## 7.4 Coefficient errors on the same instantaneous NN state — FAIL

`alpha_NN(x,y,u_NN)` versus `alpha_true(x,y,u_NN)` on the final `u_NN`, so
trajectory differences do not contaminate the model error. Exported with
`--export-alpha`; no clipping (0 clipped points).

| region | n | primary MAE | primary max | final MAE | final max |
|---|---|---|---|---|---|
| overall | 4096 | 0.390450 | 2.999605 | 0.318084 | 3.276115 |
| inclusion | 186 | 0.814557 | 1.514984 | 0.497027 | 1.640952 |
| conductor | 72 | 2.426298 | 2.999605 | 2.118381 | 3.276115 |
| stripe_high | 1919 | 0.395604 | 1.797766 | 0.324676 | 1.649389 |
| background_low | 1919 | 0.267804 | 1.421098 | 0.226603 | 1.645552 |

Section 7.4 targets are `MAE <= 0.05*alpha_max = 0.30` and
`max_abs <= 0.20*alpha_max = 1.20`. Both FAIL. The training pipeline's own
acceptance block agrees (`verdict: FAIL`): validation MAE 0.4527 / max 4.4188,
clean holdout MAE 0.4522 / max 4.8138, test MAE 0.4673 / max 4.4900. The
conductor region dominates the max error.

**Why it fails.** The ground truth is a discontinuous coefficient map (0.02
inclusion, 4.0 conductor, 0.5/0.9 stripe). The model is a small, smooth,
approximately bias-free gated-SiLU regressor; even with a perfect conductor fit
it cannot place discontinuous edges, and the global-MAE floor on this random
state distribution is roughly **0.343**. This cannot be removed by more epochs
without changing the architecture/target; it is recorded, not worked around.

## 7.5 Reproducibility and portability — PASS (with skips)

- Same-backend repeat: two identical host runs agree to `1e-12` (measured
  bitwise-equal) in `heatEquationNn_device_solver`.
- Host + HIP cross-backend field parity: `relL2 = 4.98306e-17`,
  `max_abs = 1.11022e-16`.
- Cross-backend final field vs host (matrix): preset `<= 2.22e-16`, NN
  `<= 1.409e-08`.
- Fixed-input PyTorch-vs-C++ inference parity: `max_abs = 7.662e-06`,
  `max_rel = 7.493e-06`, inside the declared `atol = 1e-5`, `rtol = 1e-4`.
- Tests: host 57/57; HIP 81/81; CUDA 81/81; SYCL 81/81 (device pinned). All
  builds pedantic, 0 warnings/errors.
- Skips: oneAPI targeting the NVIDIA A30 is **SKIP** (no CUDA/level_zero UR
  adapter in the install); default oneAPI device discovery segfaults and is an
  environment/vendor defect, so it is a **SKIP** and the Arc is selected with
  `ONEAPI_DEVICE_SELECTOR=opencl:gpu`. No other device/OS was tested.

## 8. Performance — measured compute-only, no speedup claimed

`--no-output`, grid 64², 13654 steps, same execution per mode. Timings are
wall-clock queue-event times as printed by the binary; host = CPU, hip =
RX 7900 XTX, cuda = A30, sycl = Arc A770. `per_step_d2h = no` for every row.

| backend/mode | total s | s/step | per-step D2H |
|---|---|---|---|
| host/uniform | 0.652943 | 4.782e-05 | no |
| host/preset | 2.040714 | 1.495e-04 | no |
| host/nn | 55.114960 | 4.037e-03 | no |
| hip/preset | 0.176850 | 1.295e-05 | no |
| hip/nn | 14.144359 | 1.036e-03 | no |
| cuda/preset | 0.133986 | 9.813e-06 | no |
| cuda/nn | 15.758807 | 1.154e-03 | no |
| sycl/preset | 3.113811 | 2.281e-04 | no |
| sycl/nn | 9.961625 | 7.296e-04 | no |

The NN path costs ~27× (host) to ~118× (CUDA) the analytical preset at this grid
and batch size. This is the **measured NN overhead**; it is not a recommendation
and not a speedup claim. The analytical closure is cheap by construction;
replacing it with a network is only interesting where the physical closure is
genuinely expensive. Timings are not compared across backends as a performance
ranking (different silicon and host clock behavior). Synchronization: the
compute-only path measures queue events; no per-step host copy exists, and
snapshot/CSV copies are disabled for these runs and reported separately for the
output runs.

## 9. Images and video — PASS

10 figures rendered from exported CSVs/JSON only (`T6/figures/manifest.json`
lists each figure's sources and fixed ranges); 55.0 s H.264 `yuv420p`
1920×1080 video at 30 fps, 1650 frames, full-decode OK and 15 inspected samples
with no blank panels (`T7/inspection/findings.json`). Paths are listed in
`../presentation_notes.md`.

## Acceptance checklist (Section 13)

| # | item | status | evidence |
|---|---|---|---|
| 1 | Builds through current dependency arrangement | PASS | host 57/57; dev-hal host 105/105 |
| 2 | `∇·(α∇u)`, shared harmonic faces, half-cell walls | PASS | 7.1 table |
| 3 | Coefficient initialized before first update, recomputed each step | PASS | solver/test + CLI order in `heatEquationNn.cpp` |
| 4 | Stable timesteps chosen, unstable requests rejected | PASS | 7.1 |
| 5 | Independent reference / equilibrium / heat-balance tests | PASS | 7.1 |
| 6 | Identical training/inference equations, layouts, bounds, metadata | PASS | model contract; parity 7.5 |
| 7 | Standalone PyTorch/alpakaNN inference parity | PASS | 7.5 (`max_abs 7.66e-06`) |
| 8 | Same-grid NN error separated from mesh error | PASS (measured) | 7.2 vs 7.3 |
| 9 | Coefficient errors on same state incl. regions | PASS (measured) | 7.4 |
| 10 | Meets numerical acceptance targets **or** reports failures | **FAIL (reported)** | 7.2 and 7.4 targets unmet; stated plainly |
| 11 | Reproducibility, tested backends, skipped devices recorded | PASS | 7.5 |
| 12 | Compute-only timings, honest NN overhead | PASS | 8 |
| 13 | Raw snapshots + labeled images, fixed scales, matched times | PASS | 9, `T6/figures/manifest.json` |
| 14 | Edited, verified MP4 from generated images | PASS | 9, `T7` |
| 15 | Presentation notes, measured tables, reproducible commands | PASS | `example/heatEquationNn/README.md`, `presentation_notes.md` |
| 16 | No unsupported speed/synchronization/portability claims | PASS | see scope notes |

Item 10 is intentionally the only FAIL. It is a model-quality failure, not an
integration failure; the example demonstrates the device-resident,
same-queue inference path and reports the unmet accuracy targets rather than
hiding them.

## Artifact paths (measurement sources)

All under `/tmp/alpakaNN-results/nhc-20261001/`:

- `T4/summary.json`, `T4/summary.md`, `T4/VALIDATION.md` — field/coefficient/
  mesh/timing/backends.
- `T4/logs/nhc-T4-{host,hip,cuda,sycl}-{configure,build,ctest}-final.log` —
  per-backend builds.
- `T4/runs/<backend>/<mode>/` and `T4/runs-finalmodel/nn/` — raw CSVs,
  `alpha.csv`, `manifest.csv`.
- `T3/T3.log`, `T3/remote/{torch,cpp}_parity.log` — inference parity.
- `hip/SUMMARY.md`, `hip/T5b-SUMMARY.md` — HIP/CUDA/SYCL device evidence;
  `hip/sycl_DEVICE-EVIDENCE.txt`, `hip/sycl_sycl-ls.txt`.
- `T12/SUMMARY.md`, `T12/{solver,device-solver,model-loader}-stdout.log` —
  regression-test evidence.
- `models/heat_closure/weights.bin.metadata.json` (in the worktree) —
  acceptance FAIL block.
- `T6/figures/`, `T6/figures/manifest.json`, `T6/render.log` — figures.
- `T7/heat_closure_demo.mp4`, `T7/logs/ffprobe_final.txt`,
  `T7/inspection/findings.json` — video.

## Deviations

- The primary all-backend matrix predates the installed retrained checkpoint;
  the final-model field/coefficient numbers are host-only by design and are
  labelled as such throughout. A full all-backend rerun on `b3a5f955…` was not
  performed.
- The Section 7.2 test statistic (`relL2 0.31485`) is on a tiny fixed grid and is
  not the same run as the 64² matrix values; both are reported with their grids.
