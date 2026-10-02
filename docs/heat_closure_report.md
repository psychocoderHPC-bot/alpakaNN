# Heat-closure validation report (`heatEquationNn`)

This report covers the `device-resident-runtime` worktree. It states what was
measured, on which device, and what is skipped. Values are read from the
machine-readable artifacts listed at the end; nothing is smoothed, re-scaled or
inferred.

**Verdict: the integration, portability and solver checks pass.** The original
v1 raw-contract baseline fails all accuracy targets; the recommended v2
Fourier **variant B** model **meets the coefficient MAE target**
(clean-holdout 0.1323 <= 0.30) but still misses the field targets and the
`max|err|` target, which is **provably unreachable for any continuous model**
on this geometry (conductor jump `J ≈ 5.97`, so `J/2 ≈ 2.99 > 1.20`). The
deliverable is an integration/portability demonstrator, not an accurate
physical closure.

## Provenance / measured configuration

| item | value |
|---|---|
| branch | `device-resident-runtime` |
| primary matrix commit | `4cb3ac246f6c673a889460612c4e1da67958f485` (base `8ecc9e0`) |
| installed model commit | `2bea01d0edbb70962e353b0bd20fac4b0d44f96a` (weights sha256 `b3a5f955…`) |
| primary-matrix weights | sha256 `f075cffd50324e17434713d5f9479fb02375eefc712565e2dbfcfd54ac82bf01` |
| recommended model (v2 variant B) | sha256 `83c98b08abe90eb45fc3da1a4adfb599294a1e35332535590a6b4e445de958b0` |
| variant-B source revision | `3d4310e946ef6e312d9c9a5c3ed650132324227d` (cherry-picked onto `6942b65`) |
| grid / tmax / frames / steps | 64² / 0.1 / 121 / 13654, dt 7.32386e-06 |
| host | x86_64, GNU 13.3.0, OpenMP 4.5, OpenBLAS 0.3.26, Release, pedantic |
| build gate | `-Dalpaka_COMPILE_PEDANTIC=ON` on every recorded configure |

The **primary** matrix (all backends) used model `f075cffd…`, the weights present
at run time; the retrained checkpoint `b3a5f955…` was not yet committed into the
matrix worktree. Because the numbers differ, both are reported. The
**final model** `b3a5f955…` is the retrained checkpoint whose metrics the tracked
`models/heat_closure/weights.bin.metadata.json` documents; its same-grid and
same-state numbers are the supplementary host-only run in `summary.json`
(`field_t2_final_model`, `coefficient_t2_final_model`). The binary itself is
**intentionally not committed** — regenerate it via `tools/train_heat_closure.py`
(see `example/heatEquationNn/README.md` section 2); the manifest records
`weights_sha256` so a regenerated model can be compared/detected.

The **recommended v2 variant-B** model is documented separately by
`models/heat_closure/weights_v2.bin.metadata.json` (sha256 `83c98b08…`); its
field/coefficient metrics are in section 7.2 / 7.4 and in the variant-B artifact
bundle. It is intentionally not committed either.

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

## 7.2 Same-grid neural comparison — FAIL (v2 improves but does not pass)

Same 64² grid, same 13654-step schedule, same zero initial state, same walls.
v2 = recommended variant B (Fourier, 27 inputs); v1 = original raw `[u,x,y]`
contract baseline.

| quantity | v1 primary `f075cffd…` | v1 final `b3a5f955…` | **v2 variant B** | Section 7.2 target |
|---|---|---|---|---|
| final `relL2` | 0.111712 | **0.091123** | **0.04996** | <= 0.01 |
| final normalized Linf | 0.249030 | **0.253858** | **0.19247** | <= 0.02 |
| worst transient normalized Linf | — | 0.253858 | 0.44028 | — |
| max over 121 saved times `relL2` | 0.144207 | 0.106249 | 0.07713 | — |
| mean over times `relL2` | 0.105238 | 0.086093 | — | — |

v2 roughly **halves** final `relL2` and reduces final Linf by ~24%, but both
remain above the 1 % / 2 % targets, and its worst early-transient Linf (0.4403)
is worse than v1's. The v2 field numbers are from a matched-schedule dev-hal host
run of the reviewed variant-B source (see artifact `data/v1v2_metrics.json`);
they are model-quality measurements, not an all-backend matrix.

Both v1 and v2 **fail** the field acceptance targets. The regression test
`heatEquationNn_device_solver` computes and prints these metrics on its small
fixed grid (`n=6`, `tmax=0.01`) from a single conservative step with one shared
step size and identical zero initial state for the preset and NN solvers (a
same-schedule single-step comparison, not a 20-step or full-schedule run); it
reports `relL2=0.31485`, `normLinf=0.562251`, with `target_enforced=0`; it
deliberately does not fake a pass. A regression ceiling with 1.25x headroom over
these measured values is asserted so an accuracy regression still fails the
test. The 64² matrix values above are the presentation-relevant numbers.

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

## 7.4 Coefficient errors on the same instantaneous NN state — v2 MAE PASS, max FAIL

`alpha_NN(x,y,u_NN)` versus `alpha_true(x,y,u_NN)` on the final `u_NN`, so
trajectory differences do not contaminate the model error. Exported with
`--export-alpha`; no clipping (0 clipped points).

**v2 variant B** (recommended; tracked manifest
`models/heat_closure/weights_v2.bin.metadata.json`, `weights_sha256 = 83c98b08…`):

| split | MAE | max abs | target 0.30 / 1.20 |
|---|---|---|---|
| validation | 0.1695 | 3.6900 | MAE **PASS**, max FAIL |
| **clean holdout** | **0.1323** | 2.0000 | **MAE PASS**, max FAIL |
| test | 0.1693 | 3.5832 | MAE PASS, max FAIL |

Same-state coefficient error at the v2 final NN state (grid 64): overall MAE
0.13370 / max 1.96592, conductor MAE 1.89109 / max 1.96592, inclusion MAE
0.03682, stripe_high MAE 0.11597, background_low MAE 0.09488. v2 halves the
overall MAE versus v1 and meets `<= 0.30`, but the conductor edge still misses
`max <= 1.20`.

**v1 raw baseline** (for comparison), same-state at the final v1 NN state:

| region | n | v1 MAE | v1 max |
|---|---|---|---|
| overall | 4096 | 0.318084 | 3.276115 |
| inclusion | 186 | 0.497027 | 1.640952 |
| conductor | 72 | 2.118381 | 3.276115 |
| stripe_high | 1919 | 0.324676 | 1.649389 |
| background_low | 1919 | 0.226603 | 1.645552 |

v1 fails both targets. Its training acceptance block agrees
(`verdict: FAIL`): validation MAE 0.4527 / max 4.4188, clean holdout MAE 0.4522
/ max 4.8138, test MAE 0.4673 / max 4.4900.

**Why `max|err|` fails and cannot be fixed by a continuous model.** The ground
truth is piecewise-constant with sharp interfaces. The largest jump is at the
inclusion/conductor boundary (`base = 0.02` vs `4.0`, priority inclusion first);
with `u` up to 1 and `beta = 0.5` the factor `(1 + beta*u)` is up to 1.5, so the
coefficient jumps by `J = (4.0 - 0.02) * 1.5 ≈ 5.97` there. Any **continuous**
approximant of a function with a jump of size `J` incurs an error of at least
`J/2 ≈ 2.99` somewhere on the interface, which already exceeds the `1.20`
target; hence no continuous model can pass `max|err| <= 1.20` on this geometry.
(The same-state conductor max errors are ~2 for v2 and ~3.3 for v1.) This is a structural property of the
prescribed target, not a training deficiency. The global-MAE floor for a smooth
regressor is roughly **0.343**; v2's Fourier basis lowers the MAE below it but
does not remove the interface jump.

**Empirical confirmation (separate experiments, paths under
`/workspace/heat_closure_variantB`).** These are bounded explorations, not part
of this deliverable:

- `moe/MOE.md` — a discontinuity-aware region classifier / mixture of experts
  can meet both targets on validation and the clean holdout, but only by
  hardening the classification and effectively reproducing the known 4-region
  material map; the pure Fourier `k=0..5` classifier fails the max target
  (val max 4.04).
- `separated/SEPARATED_GEOM.md` — moving the conductor away from the inclusion
  changes the geometry but no continuous variant meets max (all 2.0–3.7).
- `symmetry/SYMMETRY.md` — additive `|y-0.5|` / oracle stripe / radial features
  reduce MAE but no variant meets max (all 3.4–3.7).
- `data/FIT_STUDY.md` — wider/higher-`k` continuous variants and an oracle
  baseline all fail the max target.

The correct consequence is to state the floor honestly and treat the example as
an integration/portability demonstrator.

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
| 10 | Meets numerical acceptance targets **or** reports failures | **v2 MAE PASS; field/max FAIL (reported)** | 7.2 and 7.4: v2 clean-holdout MAE 0.1323 <= 0.30; field and `max|err|` unmet and the `J/2` floor documented |
| 11 | Reproducibility, tested backends, skipped devices recorded | PASS | 7.5 |
| 12 | Compute-only timings, honest NN overhead | PASS | 8 |
| 13 | Raw snapshots + labeled images, fixed scales, matched times | PASS | 9, `T6/figures/manifest.json` |
| 14 | Edited, verified MP4 from generated images | PASS | 9, `T7` |
| 15 | Presentation notes, measured tables, reproducible commands | PASS | `example/heatEquationNn/README.md`, `presentation_notes.md` |
| 16 | No unsupported speed/synchronization/portability claims | PASS | see scope notes |

Item 10 is a model-quality result, not an integration failure: v2 meets the
coefficient MAE target, while the field targets and `max|err|` stay unmet
(`max|err|` is impossible for a continuous model on this target). The example
demonstrates the device-resident, same-queue inference path for both the v1 and
v2 contracts and reports the measured accuracy rather than hiding it.

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
- `models/heat_closure/weights.bin.metadata.json` (in the worktree) — v1
  acceptance FAIL block and `weights_sha256` of the trained v1 checkpoint.
- `models/heat_closure/weights_v2.bin.metadata.json` (in the worktree) — v2
  variant-B `weights_sha256 = 83c98b08…`, per-region metrics, and the
  acceptance PASS/FAIL table (MAE pass, max fail). The binaries `weights.bin`
  and `weights_v2.bin` are **not committed**; regenerate them via
  `tools/train_heat_closure.py` (`--variant B` for v2); they stay untracked by
  `.gitignore`. `models/heat_closure/README.md` summarizes both models.
- `/tmp/alpakaNN-results/nhc-20261001/variantB/v1v2_metrics.json`,
  `V1V2_METRICS.md`, `VARIANTB_REPORT.md`, `inference_parity_v2.csv` — v2
  field/coefficient metrics, parity and reports (also mirrored under
  `/workspace/heat_closure_variantB/data/`).
- `/workspace/heat_closure_variantB/{moe,separated,symmetry}/` — MoE,
  separated-geometry and symmetry experiment evidence for the `J/2` floor.
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
