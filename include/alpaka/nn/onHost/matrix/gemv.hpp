/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/blas.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>

#include <concepts>
#include <stdexcept>
#include <type_traits>

namespace alpaka::nn::onHost
{
    /** Dense matrix-vector multiplication routed through the alpakaVendor BLAS backend.
     *
     * Computes ``y = alpha * W * x + beta * y`` with ``alpha == 1`` and ``beta == 0``. This is a thin wrapper over
     * ``alpaka::blas::onHost::gemv``; there is no native kernel and no layout fallback. The matching vendor BLAS
     * backend (OpenBLAS for Host, cuBLAS for CUDA, rocBLAS for HIP, oneMKL for oneAPI) is mandatory and enforced by
     * CMake. Operands must be row-major dense views: a non-multiple byte pitch is rejected by the vendor wrapper
     * itself (``std::invalid_argument``), which is the contract here.
     *
     * Only ``float`` and ``double`` are supported; integer operands are intentionally not supported.
     */
    template<typename T_Type = float>
    void gemv(auto& queue, auto const& W, auto const& x, auto& y)
    {
        static_assert(
            std::same_as<T_Type, float> || std::same_as<T_Type, double>,
            "alpaka::nn::onHost::gemv supports only float and double (vendor BLAS has no integer GEMV).");
        alpaka::blas::onHost::gemv(queue, T_Type{1}, W, x, T_Type{0}, y);
    }

    /** @brief Batched GEMV as one row-major GEMM: ``(S,K) coefficients x (K,N) matrix -> (S,N)``.
     *
     * This is the single-call path for applying one weight matrix to a batch of coefficient rows
     * (for example one row per source). It routes through the vendor GEMM once, with no per-row
     * handle creation and no host round-trip. The result is deterministic for fixed inputs and
     * device configuration because the summation order is defined by the vendor GEMM.
     *
     * @param queue queue the operation is enqueued on.
     * @param coefficients ``[S, K]`` row-major coefficient/activation matrix.
     * @param matrix ``[K, N]`` row-major weight matrix (model format ``[in, out]``).
     * @param output ``[S, N]`` destination.
     *
     * All views must stay alive until the enqueued work has completed.
     */
    template<typename T_Type = float>
    void gemvBatched(auto& queue, auto const& coefficients, auto const& matrix, auto& output)
    {
        static_assert(
            std::same_as<T_Type, float> || std::same_as<T_Type, double>,
            "alpaka::nn::onHost::gemvBatched supports only float and double.");
        auto const& coefficientExtents = coefficients.getExtents();
        auto const& matrixExtents = matrix.getExtents();
        auto const& outputExtents = output.getExtents();
        if(coefficientExtents[1] != matrixExtents[0])
            throw std::invalid_argument{"gemvBatched: coefficients [S,K] and matrix [K,N] disagree on K."};
        if(outputExtents[0] != coefficientExtents[0] || outputExtents[1] != matrixExtents[1])
            throw std::invalid_argument{"gemvBatched: output extent does not match [S,N]."};
        alpaka::nn::onHost::gemm<T_Type>(queue, coefficients, matrix, output);
    }
} // namespace alpaka::nn::onHost
