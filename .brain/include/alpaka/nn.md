# `include/alpaka/nn`

## Public structure

- `nn.hpp`
  - Canonical umbrella include.
- `core/layout.hpp`
  - Generic layout tags in `alpaka::nn::layout`.
- `core/shape.hpp`
  - Generic view/shape helpers in `alpaka::nn::shape`.
- `nn/attention.hpp`
  - Generic `AttentionKvLayout`.
- `nn/rope.hpp`
  - Generic `RopeLayout`.

## Host APIs

- `onHost/core/view.hpp`
  - Host-only padded-view and subview helpers.
- `onHost/matrix`
  - `gemm`, `gemv`, `matrixMultiply`.
- `onHost/ops`
  - Elementwise and reduction launch APIs.
- `onHost/nn`
  - Embedding, RMSNorm, softmax, RoPE, attention, and MLP host wrappers.
- `onHost/inference`
  - KV cache, transformer block, and greedy generation.
- `onHost/model`
  - TinyLlama loader and decoder orchestration.

## Acc internals

- `onAcc/internal/matrix`
  - GEMM/GEMV kernels.
- `onAcc/internal/nn`
  - Embedding, RMSNorm, softmax, RoPE, and attention kernels.
- `onAcc/internal/ops`
  - Operator kernels used by host-side launchers.
