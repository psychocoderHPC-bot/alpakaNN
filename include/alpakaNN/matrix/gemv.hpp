/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/detail/launch.hpp"
#include "alpakaNN/matrix/blas.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpakaNN
{
    namespace detail
    {
        ALPAKA_FN_SYMBOL(GemvFn);

        template<typename T_Type>
        struct GemvKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto const W, auto const x, auto y) const
            {
                auto const matrixExtent = W.getExtents();
                for(auto idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{y.getExtents()}))
                {
                    T_Type sum{};
                    for(uint32_t inner = 0u; inner < matrixExtent.x(); ++inner)
                        sum += W[alpaka::Vec{idx[0], inner}] * x[alpaka::Vec{inner}];
                    y[idx] = sum;
                }
            }
        };

        template<typename T_Type>
        void fnDispatch(
            GemvFn,
            auto& queue,
            auto exec,
            std::type_identity<T_Type>,
            auto const& W,
            auto const& x,
            auto& y)
        {
            queue.enqueue(
                alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, y.getExtents()),
                alpaka::KernelBundle{GemvKernel<T_Type>{}, W, x, y});
        }

#if ALPAKANN_OPENBLAS_ENABLED
        template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Type>
        void fnDispatch(
            GemvFn::Spec<alpaka::api::Host, T_DeviceKind>,
            auto& queue,
            auto exec,
            std::type_identity<T_Type> dataType,
            auto const& W,
            auto const& x,
            auto& y) requires(matrix::detail::OpenBlasScalar<T_Type>)
        {
            // runtime dispatch to alpaka under wrong conditions
            if(!matrix::detail::hasBlasMatrixLayout(W) || !matrix::detail::hasBlasMatrixLayout(x)
               || !matrix::detail::hasBlasMatrixLayout(y))
            {
                fnDispatch(GemvFn{}, queue, exec, dataType, W, x, y);
                return;
            }
            alpaka::onHost::wait(queue);

            auto const m = matrix::detail::toBlasInt(W.getExtents()[0], "gemv M");
            auto const n = matrix::detail::toBlasInt(W.getExtents()[1], "gemv N");
            auto const lda = matrix::detail::leadingDimension(W, "gemv lda");
            auto const incx = matrix::detail::vectorIncrement(x, "gemv incx");
            auto const incy = matrix::detail::vectorIncrement(y, "gemv incy");
            if constexpr(std::same_as<T_Type, float>)
            {
                cblas_sgemv(
                    CblasRowMajor,
                    CblasNoTrans,
                    m,
                    n,
                    1.0f,
                    W.data(),
                    lda,
                    x.data(),
                    incx,
                    0.0f,
                    y.data(),
                    incy);
            }
            else
            {
                cblas_dgemv(
                    CblasRowMajor,
                    CblasNoTrans,
                    m,
                    n,
                    1.0,
                    W.data(),
                    lda,
                    x.data(),
                    incx,
                    0.0,
                    y.data(),
                    incy);
            }
        }
#endif
    } // namespace detail

    template<typename T_Type = float>
    void gemv(auto& queue, auto exec, auto const& W, auto const& x, auto& y)
    {
        auto const wExtent = W.getExtents();
        auto const xExtent = x.getExtents();
        auto const yExtent = y.getExtents();

        if(wExtent.dim() != 2u || xExtent.dim() != 1u || yExtent.dim() != 1u)
            throw std::invalid_argument{"gemv expects W as 2D and x/y as 1D views."};
        if(wExtent.x() != xExtent[0] || wExtent.y() != yExtent[0])
            throw std::invalid_argument{"gemv shape mismatch."};

        detail::GemvFn::call(queue, exec, std::type_identity<T_Type>{}, W, x, y);
    }
} // namespace alpakaNN
