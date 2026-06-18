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
  - `attention.hpp` now launches `attentionScores` / `attentionApply` with native 4D extents instead of flattening to 1D.
  - This matches the rest of the kernel launch style and avoids decoder-only CUDA divergence where `scores raw` depended on spare cache capacity.
- `onHost/inference`
  - KV cache, transformer block, and greedy generation.
  - `generate.hpp` supports `ALPAKANN_DEBUG_TOPK=1` to print per-step top logits and selections during greedy decoding.
  - `transformer_block.hpp` supports `ALPAKANN_DEBUG_BLOCK_TRACE=1` for before/after-append block summaries and `ALPAKANN_SKIP_PREFILL_CACHE_APPEND=1` to isolate cache-append side effects during prefill.
- `onHost/model`
  - TinyLlama loader and decoder orchestration.
  - `decoder.hpp` supports `ALPAKANN_DEBUG_PREFILL_TRACE=1` to print per-stage tensor summaries through `prefill`, useful for CUDA divergence tracing between implicit and explicit cache paths.

## Acc internals

- `onAcc/internal/matrix`
  - GEMM/GEMV kernels.
- `onAcc/internal/nn`
  - Embedding, RMSNorm, softmax, RoPE, and attention kernels.
  - Kernel axis selection uses `ALPAKA_TYPEOF(extents)::dim()` to avoid nvcc `consteval` call failures.
- `onAcc/internal/ops`
  - Operator kernels used by host-side launchers.
  - Kernel/validation rank queries use the same nvcc workaround as host code.
