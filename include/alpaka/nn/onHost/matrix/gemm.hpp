/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/blas.hpp>

#include <concepts>
#include <type_traits>

namespace alpaka::nn::onHost
{
    /** Dense matrix-matrix multiplication routed through the alpakaVendor BLAS backend.
     *
     * Computes ``C = alpha * A * B + beta * C`` with ``alpha == 1`` and ``beta == 0``. This is a thin wrapper over
     * ``alpaka::blas::onHost::gemm``; there is no native kernel and no layout fallback. The matching vendor BLAS
     * backend (OpenBLAS for Host, cuBLAS for CUDA, rocBLAS for HIP, oneMKL for oneAPI) is mandatory and enforced by
     * CMake. Operands must be row-major dense views: a non-multiple byte pitch or a non-unit column stride is
     * rejected by the vendor wrapper itself (``std::invalid_argument``), which is the contract here.
     *
     * Only ``float`` and ``double`` are supported; integer operands are intentionally not supported.
     *
     * @tparam T_Type Scalar type; must be ``float`` or ``double`` (defaults to ``float``).
     * @param queue alpaka queue the work is enqueued on. The call is asynchronous with respect to the host.
     * @param A Left operand, a row-major dense 2D view of extents ``(M, K)``.
     * @param B Right operand, a row-major dense 2D view of extents ``(K, N)``.
     * @param C Output, a preallocated row-major dense 2D view of extents ``(M, N)``. It is overwritten
     *          (``beta == 0``) rather than accumulated.
     *
     * @throw std::invalid_argument if the vendor wrapper rejects an operand pitch or column stride.
     *
     * @note The operation is asynchronous and the caller owns every view. Keep @p A, @p B and @p C alive and valid
     *       until the enqueued work completes, and call ``alpaka::onHost::wait(queue)`` (or ``C.keepAlive(queue)``)
     *       before consuming @p C on the host.
     */
    template<typename T_Type = float>
    void gemm(auto& queue, auto const& A, auto const& B, auto& C)
    {
        static_assert(
            std::same_as<T_Type, float> || std::same_as<T_Type, double>,
            "alpaka::nn::onHost::gemm supports only float and double (vendor BLAS has no integer GEMM).");
        alpaka::blas::onHost::gemm(queue, T_Type{1}, A, B, T_Type{0}, C);
    }
} // namespace alpaka::nn::onHost
