# `include/alpaka/nn`

## Public structure

- `nn.hpp`
  - Canonical umbrella include.
- `core/layout.hpp`
  - Generic layout tags in `alpaka::nn::layout`.
- `core/shape.hpp`
  - Generic view/shape helpers in `alpaka::nn::shape`.
  - Rank queries use `ALPAKA_TYPEOF(extents)::dim()` as an nvcc-safe workaround for `static consteval Vec::dim()`.
- `nn/attention.hpp`
  - Generic `AttentionKvLayout`.
- `nn/rope.hpp`
  - Generic `RopeLayout`.

## Host APIs

- `onHost/core/view.hpp`
  - Host-only padded-view and subview helpers.
- `onHost/matrix`
  - `gemm`, `gemv`: thin wrappers over `alpaka::blas::onHost` (vendor BLAS only; no native kernel, no
    fallback, `float`/`double` only). The matching vendor backend is mandatory per enabled alpaka backend.
- `onHost/ops`
  - Elementwise and reduction launch APIs.
  - Shape validation and last-axis selection use `ALPAKA_TYPEOF(... )::dim()` instead of `.dim()`.
- `onHost/nn`
  - Embedding, RMSNorm, softmax, RoPE, attention, and MLP host wrappers.
  - Host-side shape checks cache extents locally before type-based rank queries.
  - `attention.hpp` launches `attentionScores` / `attentionApply` with native 4D extents.

## Acc internals

- `onAcc/internal/nn`
  - Embedding, RMSNorm, softmax, RoPE, and attention kernels.
  - Kernel axis selection uses `ALPAKA_TYPEOF(extents)::dim()` to avoid nvcc `consteval` call failures.
  - `AttentionScoresKernel` is currently mapped per score `(batch, head, query, key)`.
- `onAcc/internal/ops`
  - Operator kernels used by host-side launchers.
  - Kernel/validation rank queries use the same nvcc workaround as host code.

## Core tests

- `test/unit/nn/attention.cpp`
  - Pure-primitive attention coverage: scores/apply determinism, grouped-query head mapping, and
    BTHD/BHTD reference matches for prefill/decode shapes.
  - `qkvProjection` + `ropeInPlace` are covered against a host scalar reference (vendor-backed GEMM).
- `test/unit/matrix/{matmul,gemv}.cpp`
  - Dense float/double numerical coverage against a host scalar reference, vendor-backed only.
- `test/unit/nn/embedding.cpp`
  - `embeddingLookup` shape/rank validation, numerical correctness, and a non-contiguous subview case.
- `test/unit/nn/{rope,rms_norm,softmax,mlp}.cpp`
  - Host-wrapper numerical coverage against scalar references.
