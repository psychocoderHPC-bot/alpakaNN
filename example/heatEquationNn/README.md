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

**This checkpoint is not a validated accurate closure.** Its metadata reports
validation coefficient MSE 0.6541, MAE 0.5190, and maximum absolute error
3.9792 (2000 epochs; checkpoint selected at epoch 1989). The conductor-region
validation MSE is 8.2455. These poor metrics must not be interpreted as model
quality or a production recommendation. The final export skips test evaluation;
test data were inadvertently evaluated in earlier exploratory tuner runs and
are not reported as a clean final-model estimate.

Inference is deliberately host-only and batched once per time step. The solver
state and conservative stencil remain `std::vector<double>` host data; no
GPU-resident state, fused device stencil, same-queue stencil/inference, or
allocation-free/asynchronous execution is claimed. The public alpakaNN host MLP
allocates three hidden intermediates and waits between its stages; this path
copies each feature/weight batch to the selected host device and waits for the
result. CUDA/HIP/oneAPI inference backends are not enabled by this example.
