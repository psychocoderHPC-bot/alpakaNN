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
  - `attention.hpp` launches `attentionScores` / `attentionApply` with native 4D extents.
  - Current local decoder diagnostics indicate prefill parity is stable, while a CUDA-focused decode-only divergence remains isolated to kernel-produced `scores raw` over cache-backed `BHTD` views.
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
  - `AttentionScoresKernel` is currently mapped per score `(batch, head, query, key)`.
  - Local decoder substage tests now distinguish cache-content parity from kernel-score parity, which is the main remaining path to watch on CUDA.

## Decoder diagnostics

- `test/unit/model/decoder.cpp`
  - `StageSnapshot` now supports arbitrary-rank captured tensors by flattening values and storing the full shape vector.
  - Added `tiny llama prefill layer 0 substages are independent of cache capacity`.
  - Added `tiny llama decode layer 0 substages are independent of cache capacity`.
  - The decode substage test also captures:
    - cache keys
    - cache values
    - reference decode scores from copied host tensors
  - Current local signal:
    - prefill substage parity passes
    - decode cache keys/values parity passes
    - decode reference-score parity passes
    - decode kernel `scores raw` can still diverge across spare cache capacities
- `test/unit/nn/attention.cpp`
  - Added `decoder decode attention cache-backed kv views are independent of spare capacity`.
  - Added `decoder prefill attention flat packed views are independent of spare allocation`.
  - Added `decoder prefill projected attention is independent of spare allocation`.
  - Current local signal:
    - direct cache-backed decode attention parity passes across capacities
    - direct prefill attention parity also passes for:
      - synthetic flat packed `BTHD` views
      - projected `qkvProjection -> rope -> attention` path
    - if CUDA fails this direct test, the fault is inside attention/cache-view execution itself
    - if CUDA passes it but decoder substages still fail, the remaining bug is in decoder-specific interaction or memory corruption around that path
- `onAcc/internal/ops`
  - Operator kernels used by host-side launchers.
  - Kernel/validation rank queries use the same nvcc workaround as host code.
