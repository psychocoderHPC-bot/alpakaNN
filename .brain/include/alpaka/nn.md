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
  - `gemm`, `gemv`, `matrixMultiply`.
  - Matrix/vector rank validation now uses type-based `::dim()` queries for nvcc compatibility.
- `onHost/ops`
  - Elementwise and reduction launch APIs.
  - Shape validation and last-axis selection use `ALPAKA_TYPEOF(... )::dim()` instead of `.dim()`.
- `onHost/nn`
  - Embedding, RMSNorm, softmax, RoPE, attention, and MLP host wrappers.
  - Host-side shape checks cache extents locally before type-based rank queries.
- `onHost/inference`
  - KV cache, transformer block, and greedy generation.
- `onHost/model`
  - TinyLlama loader and decoder orchestration.

## Acc internals

- `onAcc/internal/matrix`
  - GEMM/GEMV kernels.
- `onAcc/internal/nn`
  - Embedding, RMSNorm, softmax, RoPE, and attention kernels.
  - Kernel axis selection uses `ALPAKA_TYPEOF(extents)::dim()` to avoid nvcc `consteval` call failures.
- `onAcc/internal/ops`
  - Operator kernels used by host-side launchers.
  - Kernel/validation rank queries use the same nvcc workaround as host code.
