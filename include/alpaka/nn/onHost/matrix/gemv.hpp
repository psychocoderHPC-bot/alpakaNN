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
    /** Dense matrix-vector multiplication routed through the alpakaVendor BLAS backend.
     *
     * Computes ``y = alpha * W * x + beta * y`` with ``alpha == 1`` and ``beta == 0``. This is a thin wrapper over
     * ``alpaka::blas::onHost::gemv``; there is no native kernel and no layout fallback. The matching vendor BLAS
     * backend (OpenBLAS for Host, cuBLAS for CUDA, rocBLAS for HIP, oneMKL for oneAPI) is mandatory and enforced by
     * CMake. Operands must be row-major dense views: a non-multiple byte pitch is rejected by the vendor wrapper
     * itself (``std::invalid_argument``), which is the contract here.
     *
     * Only ``float`` and ``double`` are supported; integer operands are intentionally not supported.
     *
     * @tparam T_Type Scalar type; must be ``float`` or ``double`` (defaults to ``float``).
     * @param queue alpaka queue the work is enqueued on. The call is asynchronous with respect to the host.
     * @param W Weight matrix, a row-major dense 2D view of extents ``(M, K)``.
     * @param x Input vector, a 1D view of extent ``(K,)``.
     * @param y Output vector, a preallocated 1D view of extent ``(M,)``. It is overwritten (``beta == 0``) rather
     *          than accumulated.
     *
     * @throw std::invalid_argument if the vendor wrapper rejects a weight pitch.
     *
     * @note The operation is asynchronous and the caller owns every view. Keep @p W, @p x and @p y alive and valid
     *       until the enqueued work completes, and call ``alpaka::onHost::wait(queue)`` (or ``y.keepAlive(queue)``)
     *       before consuming @p y on the host.
     */
    template<typename T_Type = float>
    void gemv(auto& queue, auto const& W, auto const& x, auto& y)
    {
        static_assert(
            std::same_as<T_Type, float> || std::same_as<T_Type, double>,
            "alpaka::nn::onHost::gemv supports only float and double (vendor BLAS has no integer GEMV).");
        alpaka::blas::onHost::gemv(queue, T_Type{1}, W, x, T_Type{0}, y);
    }
} // namespace alpaka::nn::onHost
