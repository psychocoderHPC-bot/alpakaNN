# Heat-equation coefficient closure

Build from the repository root (C++20; this example uses alpaka's public host MLP API):

```sh
cmake -S . -B build -DalpakaNN_BUILD_TESTS=ON \
  -DalpakaNN_BUILD_HEAT_CLOSURE_EXAMPLE=ON -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the reference material presets or trained closure:

```sh
build/example/heatEquationNn/heatEquationNn --material preset --grid 64
build/example/heatEquationNn/heatEquationNn --material uniform --grid 64
build/example/heatEquationNn/heatEquationNn --material nn \
  --weights models/heat_closure/weights.bin --beta 0.5 --alpha-min 0.01 --alpha-max 6 \
  --grid 16 --steps 900 --validate-strict --no-output
```

The NN model is a companion pair: `weights.bin` and `weights.bin.metadata.json`.
The loader accepts only `alpakaNN-heat-closure-f32-v1`, bias-free gated-SiLU
architecture, float32 little-endian row-major gate/up/down arrays of shapes
`[3,64]`, `[3,64]`, `[64,1]` (1792 bytes total), feature order `[u,x,y]`,
and a matching beta. Weight payload size and finiteness are checked; alpha
bounds must be positive and ordered. The current lightweight metadata reader is
format-sensitive and does not validate every declared semantic field, so it is
not a general JSON-schema validator. Outputs are mapped to physical coefficient
units as `alpha_min + (alpha_max-alpha_min)*sigmoid(z)`. Model bounds/beta must
match CLI options. Train/regenerate via
`python3 tools/train_heat_closure.py --help` (PyTorch required).

**This checkpoint is not a validated accurate closure; the training pipeline's
own acceptance verdict is FAIL.** The metadata's acceptance targets
(MAE <= 0.30 and maximum absolute error <= 1.20, from `0.05*alpha_max` and
`0.20*alpha_max`) are not met. Spatial-evaluation metrics: MAE 0.3363, maximum
absolute error 1.2519. Validation: MAE 0.4527, maximum absolute error 4.4188.
Test: MAE 0.4673, maximum absolute error 4.4900. Clean holdout: MAE 0.4522,
maximum absolute error 4.8138. The largest errors are concentrated in the
conductor region (validation/test/clean-holdout conductor maximum absolute
error ~4.4-4.8). These numbers must not be interpreted as model quality or a
production recommendation: the example demonstrates the inference path, not an
accurate closure.

Inference is deliberately host-only and batched once per time step. The solver
state and conservative stencil remain `std::vector<double>` host data; no
GPU-resident state, fused device stencil, same-queue stencil/inference, or
allocation-free/asynchronous execution is claimed. The public alpakaNN host MLP
allocates three hidden intermediates and waits between its stages; this path
copies each feature/weight batch to the selected host device and waits for the
result. CUDA/HIP/oneAPI inference backends are not enabled by this example.
