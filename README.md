**alpaka** - GEMM example
=========================

This is a GEMM implementation with alpaka.
It shows two basic implementations:
- `gemm` is using global memory and no thread blocking
- `sgemm` is using shared memory for caching a tiles

Keep in mind, that both methods are not optimized for performance and will be much slower than optimized BLAS libraries.
