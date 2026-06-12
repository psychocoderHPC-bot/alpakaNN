/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onAcc/internal/matrix/gemv.hpp>
#include <alpaka/nn/onHost/internal/launch.hpp>
#include <alpaka/nn/onHost/matrix/internal/blas.hpp>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace alpaka::nn::onHost
{
    namespace internal
    {
        ALPAKA_FN_SYMBOL(GemvFn);

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
                alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, y.getExtents()),
                alpaka::KernelBundle{alpaka::nn::onAcc::internal::matrix::GemvKernel<T_Type>{}, W, x, y});
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
            auto& y) requires(matrix::internal::OpenBlasScalar<T_Type>)
        {
            if(!matrix::internal::hasBlasMatrixLayout(W) || !matrix::internal::hasBlasVectorLayout(x)
               || !matrix::internal::hasBlasVectorLayout(y))
            {
                fnDispatch(GemvFn{}, queue, exec, dataType, W, x, y);
                return;
            }
            alpaka::onHost::wait(queue);

            auto const m = matrix::internal::toBlasInt(W.getExtents()[0], "gemv M");
            auto const n = matrix::internal::toBlasInt(W.getExtents()[1], "gemv N");
            auto const lda = matrix::internal::leadingDimension(W, "gemv lda");
            auto const incx = matrix::internal::vectorIncrement(x, "gemv incx");
            auto const incy = matrix::internal::vectorIncrement(y, "gemv incy");
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
    } // namespace internal

    template<typename T_Type = float>
    void gemv(auto& queue, auto exec, auto const& W, auto const& x, auto& y)
    {
        auto const wExtent = W.getExtents();
        auto const xExtent = x.getExtents();
        auto const yExtent = y.getExtents();

        if(ALPAKA_TYPEOF(wExtent)::dim() != 2u || ALPAKA_TYPEOF(xExtent)::dim() != 1u
           || ALPAKA_TYPEOF(yExtent)::dim() != 1u)
            throw std::invalid_argument{"gemv expects W as 2D and x/y as 1D views."};
        if(wExtent.x() != xExtent[0] || wExtent.y() != yExtent[0])
            throw std::invalid_argument{"gemv shape mismatch."};

        internal::GemvFn::call(queue, exec, std::type_identity<T_Type>{}, W, x, y);
    }
} // namespace alpaka::nn::onHost
