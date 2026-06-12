/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/nn/onAcc/internal/matrix/gemm.hpp>
#include <alpaka/nn/onHost/internal/launch.hpp>
#include <alpaka/nn/onHost/matrix/internal/blas.hpp>

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace alpaka::nn::onHost
{
    namespace internal
    {
        ALPAKA_FN_SYMBOL(GemmFn);

        template<typename T_Type>
        void fnDispatch(
            GemmFn,
            auto& queue,
            auto exec,
            std::type_identity<T_Type>,
            auto const& A,
            auto const& B,
            auto& C)
        {
            queue.enqueue(
                alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, C.getExtents()),
                alpaka::KernelBundle{alpaka::nn::onAcc::internal::matrix::GemmKernel<T_Type>{}, A, B, C});
        }

#if ALPAKANN_OPENBLAS_ENABLED
        template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Type>
        void fnDispatch(
            GemmFn::Spec<alpaka::api::Host, T_DeviceKind>,
            auto& queue,
            [[maybe_unused]] auto exec,
            std::type_identity<T_Type> dataType,
            auto const& A,
            auto const& B,
            auto& C) requires(matrix::internal::OpenBlasScalar<T_Type>)
        {
            if(!matrix::internal::hasBlasMatrixLayout(A) || !matrix::internal::hasBlasMatrixLayout(B)
               || !matrix::internal::hasBlasMatrixLayout(C))
            {
                fnDispatch(GemmFn{}, queue, exec, dataType, A, B, C);
                return;
            }

            alpaka::onHost::wait(queue);

            auto const m = matrix::internal::toBlasInt(A.getExtents()[0], "gemm M");
            auto const k = matrix::internal::toBlasInt(A.getExtents()[1], "gemm K");
            auto const n = matrix::internal::toBlasInt(B.getExtents()[1], "gemm N");
            auto const lda = matrix::internal::leadingDimension(A, "gemm lda");
            auto const ldb = matrix::internal::leadingDimension(B, "gemm ldb");
            auto const ldc = matrix::internal::leadingDimension(C, "gemm ldc");
            if constexpr(std::same_as<T_Type, float>)
            {
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,
                    n,
                    k,
                    1.0f,
                    A.data(),
                    lda,
                    B.data(),
                    ldb,
                    0.0f,
                    C.data(),
                    ldc);
            }
            else
            {
                cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,
                    n,
                    k,
                    1.0,
                    A.data(),
                    lda,
                    B.data(),
                    ldb,
                    0.0,
                    C.data(),
                    ldc);
            }
        }
#endif
    } // namespace internal

    template<typename T_Type = float>
    void gemm(auto& queue, auto exec, auto const& A, auto const& B, auto& C)
    {
        auto const aExtent = A.getExtents();
        auto const bExtent = B.getExtents();
        auto const cExtent = C.getExtents();

        if(aExtent.dim() != 2u || bExtent.dim() != 2u || cExtent.dim() != 2u)
            throw std::invalid_argument{"gemm expects 2D matrices."};
        if(aExtent.x() != bExtent.y())
            throw std::invalid_argument{"gemm requires A.cols == B.rows."};
        if(cExtent.y() != aExtent.y() || cExtent.x() != bExtent.x())
            throw std::invalid_argument{"gemm requires C to match A.rows x B.cols."};

        internal::GemmFn::call(queue, exec, std::type_identity<T_Type>{}, A, B, C);
    }
} // namespace alpaka::nn::onHost
