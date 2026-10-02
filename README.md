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
  wrappers (attention, embedding, RMSNorm, RoPE, softmax). These are
  implementation details and are **not** part of the public API.

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

## Mental model

alpakaNN follows the alpaka3 execution model, so three ideas explain almost
every function signature.

1. **Views, not storage.** An operation never allocates its output: the caller
   allocates storage (`alpaka::onHost::allocHost` / `alpaka::onHost::alloc`)
   and passes views in. Views alias the underlying storage and can be padded or
   be sub-views of a larger buffer. The caller owns the lifetime of every view;
   nothing is copied implicitly.
2. **Asynchronous queues.** Operations only *enqueue* work on an
   `alpaka::onHost::Queue`. The call returns before the work has necessarily
   finished. Keep every input and output view alive until the work completes,
   and call `alpaka::onHost::wait(queue)` before reading results on the host
   (or use `view.keepAlive(queue)` to extend a view's lifetime). This is the
   one contract that is easy to get wrong; it is restated on every primitive.
3. **Layouts and shapes.** The `alpaka::nn::layout` tags (`TH`, `BTH`, `BTHD`,
   `BHTD`, `LBHTD`) name the axis order of a tensor and expose its compile-time
   rank and fastest (last) axis. `alpaka::nn::shape` provides the runtime
   helpers (`isContiguous`, `requireLayout`, ...). `BTHD`/`BHTD` select whether
   tokens or heads are the outer axis of attention/RoPE tensors.

Precision and portability:

- GEMM/GEMV support only `float` and `double` (vendor BLAS has no integer GEMM).
  The remaining kernels are templates over the element type.
- fp64 is **not** guaranteed on every oneAPI GPU (e.g. Intel Arc). Detect the
  capability from the alpaka backend/device-kind tags at compile time and fall
  back to `float` there. There is no installed capability helper; an inline
  `consteval` predicate works, mirroring the library's own tests:

  ```cpp
  #include <concepts>
  #include <utility>

  template<typename T_Device>
  consteval bool deviceSupportsFp64(T_Device const&)
  {
      if constexpr(std::same_as<
                       std::remove_cvref_t<decltype(alpaka::getApi(std::declval<T_Device>()))>,
                       alpaka::api::OneApi>)
          return std::same_as<
              std::remove_cvref_t<decltype(alpaka::getDeviceKind(std::declval<T_Device>()))>,
              alpaka::deviceKind::Cpu>;
      else
          return true;
  }
  ```

  Use it as `if constexpr(deviceSupportsFp64(device))` around `double`
  instantiations; a user-facing capability query is deliberately **not** part of
  the public API.
- The layout tags and `layout::rank`/`layout::fastestAxis` are compile-time
  (`consteval`) and can be used from host and device code. The `shape::`
  helpers and the `onHost` primitives are host-side entry points; the device
  kernels in `onAcc/internal` are not public.

## Quickstart

This example mirrors `examples/quickstart.cpp`, which is compiled as part of
the regular build (CMake option `alpakaNN_EXAMPLES`, on by default for a
top-level checkout). It allocates caller-owned views, runs elementwise
operations, synchronizes the queue and reads the result back.

```cpp
#include <alpaka/nn/nn.hpp>

#include <cstdint>

int main()
{
    auto device = alpaka::onHost::makeHostDevice();
    auto exec = alpaka::exec::cpuSerial;
    auto queue = device.makeQueue();

    constexpr uint32_t n = 8u;
    auto a = alpaka::onHost::allocHost<float>(alpaka::Vec{n});
    auto b = alpaka::onHost::allocHost<float>(alpaka::Vec{n});
    auto out = alpaka::onHost::allocHost<float>(alpaka::Vec{n});
    for(uint32_t i = 0u; i < n; ++i)
    {
        a[alpaka::Vec{i}] = static_cast<float>(i);
        b[alpaka::Vec{i}] = static_cast<float>(2u * i + 1u);
    }

    // Enqueue work: view shapes must match, and the views must outlive the work.
    alpaka::nn::onHost::ops::add<float>(queue, exec, a, b, out);
    alpaka::nn::onHost::ops::relu<float>(queue, exec, out, out);

    // Synchronize before consuming the result on the host.
    alpaka::onHost::wait(queue);

    // out[i] == 3 * i + 1
    return out[alpaka::Vec{0u}] == 1.0f ? 0 : 1;
}
```

On an accelerator, replace the host device with a device selected from the
enabled backends and use `alpaka::onHost::allocLike(device, hostView)` plus
`alpaka::onHost::memcpy` for the host/device transfers, as the component tests
do.

## API index

All symbols live in namespace `alpaka::nn` unless stated otherwise.

Host- and device-callable:

| Symbol | Header |
| --- | --- |
| `layout::TH`, `layout::BTH`, `layout::BTHD`, `layout::BHTD`, `layout::LBHTD` | `<alpaka/nn/core/layout.hpp>` |
| `layout::rank`, `layout::fastestAxis` | `<alpaka/nn/core/layout.hpp>` |
| `RopeLayout`, `AttentionKvLayout` | `<alpaka/nn/nn/rope.hpp>`, `<alpaka/nn/nn/attention.hpp>` |

`layout::requireRank` is host-only: it is plain `constexpr` (not `ALPAKA_FN_ACC`)
and throws `std::invalid_argument` on a rank mismatch, so it is not callable from
device code. It may still run in a constant expression when the ranks match.

Host-only core helpers (`alpaka::nn::shape`; plain `inline`, not `ALPAKA_FN_ACC`):

| Symbol | Header |
| --- | --- |
| `shape::ValueType`, `shape::extentAt`, `shape::pitchAtBytes`, `shape::elementPitchAt` | `<alpaka/nn/core/shape.hpp>` |
| `shape::requireLayout`, `shape::requireAxis`, `shape::isAxisContiguous`, `shape::isContiguous`, `shape::requireContiguousAxis` | `<alpaka/nn/core/shape.hpp>` |

Host view helpers (`alpaka::nn::onHost::view`):

| Symbol | Header |
| --- | --- |
| `makePaddedView`, `subView` | `<alpaka/nn/onHost/core/view.hpp>` |

Vendor-BLAS matrix wrappers (`alpaka::nn::onHost`):

| Symbol | Header |
| --- | --- |
| `gemm<T_Type>` | `<alpaka/nn/onHost/matrix/gemm.hpp>` |
| `gemv<T_Type>` | `<alpaka/nn/onHost/matrix/gemv.hpp>` |

Elementwise ops (`alpaka::nn::onHost::ops`):

| Symbol | Header |
| --- | --- |
| `fill`, `copy`, `unaryOp`, `binaryOp` | `<alpaka/nn/onHost/ops/elementwise.hpp>` |
| `add`, `sub`, `mul`, `div`, `scale`, `axpy`, `biasAdd`, `cast` | `<alpaka/nn/onHost/ops/elementwise.hpp>` |
| `exp`, `sqrt`, `rsqrt`, `relu`, `sigmoid`, `silu`, `gelu`, `swiglu` | `<alpaka/nn/onHost/ops/elementwise.hpp>` |

Reductions (`alpaka::nn::onHost::ops`):

| Symbol | Header |
| --- | --- |
| `reduceSum`, `reduceMax`, `reduceMean`, `reduceSumSquares`, `dot`, `makeReducedExtents` | `<alpaka/nn/onHost/ops/reduction.hpp>` |

Neural-network primitives (`alpaka::nn::onHost::nn`):

| Symbol | Header |
| --- | --- |
| `embeddingLookup` | `<alpaka/nn/onHost/nn/embedding.hpp>` |
| `rmsNorm` | `<alpaka/nn/onHost/nn/rms_norm.hpp>` |
| `softmax`, `maskedSoftmax`, `causalSoftmax` | `<alpaka/nn/onHost/nn/softmax.hpp>` |
| `rope`, `ropeInPlace` | `<alpaka/nn/onHost/nn/rope.hpp>` |
| `qkvProjection`, `attentionScores`, `attentionApply`, `outputProjection` | `<alpaka/nn/onHost/nn/attention.hpp>` |
| `linear`, `mlp` | `<alpaka/nn/onHost/nn/mlp.hpp>` |

The layout tags and `shape::` helpers above are pure compile-time/runtime
queries and enqueue nothing. The `onHost` primitives in the following tables
(vendor-BLAS matrix wrappers, elementwise ops, reductions, and NN primitives)
enqueue asynchronous work on the queue; see the inline Doxygen in the headers
for the exact shape, layout, dtype and lifetime preconditions.

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
compile-time guarded so fp64-less devices still run all `float` cases instead of
having the whole spec skipped. When `alpaka_DEP_ONEAPI` is enabled, the CMake
test configuration exports the documented IGC fp64-emulation variables
(`IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1`) to the discovered
tests. These variables are Intel-specific and ignored by other backends. They
only affect fp64 code paths that are actually built: for oneAPI the capability
guard is CPU-only, so `double` instantiations remain compile-time excluded on
oneAPI GPUs regardless of the emulation variables. Native-fp64 devices are
unaffected.

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
