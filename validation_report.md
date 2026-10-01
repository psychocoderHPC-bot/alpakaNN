# Heat-closure validation report

This report describes the implementation as it currently exists; it does not
claim completion of the device-resident integration objective.

## Build and tests

- Local GNU C++20 Release build with
  `-DalpakaNN_BUILD_HEAT_CLOSURE_EXAMPLE=ON -Dalpaka_COMPILE_PEDANTIC=ON`:
  passed. Local CTest: 54/54 passed.
- On HAL `terok-dev`, the same pedantic host build passed and CTest passed
  102/102 (HAL had additional host test variants enabled).
- HAL A30 was visible with `nvidia-smi` (NVIDIA-SMI 610.57.04, driver 610.57.04,
  CUDA UMD 13.3). `nvcc` was not installed, so CUDA compilation/runtime could
  not be tested. The current solver and NN integration select the host CPU, not
  the A30.
- `pre-commit run --all-files`: passed after formatting, REUSE metadata, and
  executable-bit fixes.
- Python training tests: 3/3 passed. Presentation smoke test passed. No full
  MP4 from validated coefficient/field comparison data is claimed.

## Numerical smoke run

On HAL, preset, uniform, and NN modes each completed a strict `8 x 8`,
`tmax=0.1`, 214-step no-output run and stayed within the temperature maximum
principle. Preset and NN modes were also saved at the same final physical time
and step (0.1, 214). Their field difference was RMS/L2-per-cell
`0.06862476123011571` and Linf `0.19502713258336024`. This is a smoke comparison,
not evidence of an accurate closure. The executable does not currently export
coefficient maps or region-specific errors.

## Model and integration limitations

The included checkpoint's validation metrics are MSE `0.6541`, MAE `0.5190`,
and maximum absolute error `3.9792`; conductor-region validation MSE is `8.2455`.
The checkpoint was selected on validation at epoch 1989/2000. The final export
used `--skip-test`; however, test data were inadvertently evaluated in two
earlier exploratory tuner runs. Those results did not drive checkpoint selection
or model changes, and are not reported as a clean final-model estimate. No
fixed-input PyTorch/alpakaNN parity, final-test regional error,
rollout-distribution evaluation, or accepted coefficient/field-error target is
established here. Do not use this model as a validated physical closure.

The implementation is a host `std::vector<double>` solver and calls alpakaNN's
host MLP. It allocates/copies features and weights and synchronizes at each
inference call. It does not implement the requested device-resident solver,
same-queue inference/stencil execution, or allocation-free per-step NN path.
CUDA/HIP/SYCL support is not established. Dataset feature export via
`--dump-features`, structured strict validation of all model metadata fields,
coefficient-map visualization, robust comparison-time/error reporting, and the
full prescribed presentation evidence sequence remain incomplete.
