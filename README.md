# alpakaNN

Header-only neural network building blocks on top of
[alpaka3](https://github.com/alpaka-group/alpaka3).

alpakaNN provides reusable, backend-agnostic primitive operations that can be
composed into larger models. It is not an application and does not ship a
model runner or an inference front end. All algorithms are exposed through the
umbrella header `<alpaka/nn/nn.hpp>` and the exported CMake target
`alpaka::nn`.

## Scope

- `core` - generic layout and shape helpers shared by host and accelerator code.
- `nn` - generic layout tags (`rope`, `attention`) usable on host and device.
- `onHost` - host-callable APIs in `alpaka::nn::onHost`:
  - `core/view.hpp` - padded-view and subview helpers.
  - `matrix` - `gemm` and `gemv`, thin wrappers over the alpakaVendor BLAS
    backend (no native kernel and no fallback; `float`/`double` only).
  - `ops` - elementwise and reduction launch APIs.
  - `nn` - `embeddingLookup`, `rmsNorm`, `softmax`/`causalSoftmax`, `rope`/
    `ropeInPlace`, `qkvProjection`, and `attentionScores`/`attentionApply`/`mlp`.
- `onAcc/internal` - accelerator-only kernel building blocks used by the host
  wrappers (attention, embedding, RMSNorm, RoPE, softmax).

## Requirements

- A C++20 compiler.
- CMake >= 3.25.
- [alpakaVendor](https://github.com/psychocoderHPC/alpakaVendor), which bundles
  the alpaka3 dependency and the BLAS/FFT backends. It is fetched automatically
  at configure time.
- A vendor BLAS backend matching each enabled alpaka backend: OpenBLAS for Host
  (always required), cuBLAS for CUDA, rocBLAS for HIP, and oneMKL
  (`MKL::MKL_SYCL::DFT`) for oneAPI. GEMM and GEMV always go through
  `alpaka::blas::onHost`; there is no native fallback, so a missing matching
  backend fails the configure.

## Build and test

The library is header-only; building the tests is the main local workflow:

```bash
cmake -S . -B build/host -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build/host -j
ctest --test-dir build/host --output-on-failure
```

Device backends (CUDA, HIP, oneAPI) are selected through the usual alpaka CMake
options, for example `-Dalpaka_DEP_CUDA=ON -Dalpaka_CUDA_NvidiaGpu=ON`. Ready-made
combinations are provided in `CMakePresets.json`.

The component tests exercise both `float` and `double`. Since fp64 is not
guaranteed on every oneAPI GPU (for example Intel Arc), the `double` cases are
guarded at compile time by `alpaka::nn::test::supportsFp64(device)`; fp64-less
devices still run all `float` cases instead of having the whole spec skipped.
When `alpaka_DEP_ONEAPI` is enabled, the CMake test configuration exports the
documented IGC fp64-emulation variables
(`IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1`) to the discovered
tests so the Intel GPU is exercised rather than excluded. These variables are
Intel-specific and ignored by other backends. Even with emulation active, the
`double` cases remain compile-time-guarded by `supportsFp64` for oneAPI GPUs,
so native-fp64 devices are unaffected.

## Consuming the library

```cmake
find_package(alpakaNN CONFIG REQUIRED)
target_link_libraries(myTarget PRIVATE alpaka::nn)
```

```cpp
#include <alpaka/nn/nn.hpp>
```

## License

MPL-2.0 - see `LICENSE`.
