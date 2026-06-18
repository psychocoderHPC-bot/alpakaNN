/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onAcc/internal/matrix/gemm.hpp>
#include <alpaka/nn/onHost/internal/launch.hpp>
#include <alpaka/nn/onHost/matrix/internal/blas.hpp>

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

        if(ALPAKA_TYPEOF(aExtent)::dim() != 2u || ALPAKA_TYPEOF(bExtent)::dim() != 2u
           || ALPAKA_TYPEOF(cExtent)::dim() != 2u)
            throw std::invalid_argument{"gemm expects 2D matrices."};
        if(aExtent.x() != bExtent.y())
            throw std::invalid_argument{"gemm requires A.cols == B.rows."};
        if(cExtent.y() != aExtent.y() || cExtent.x() != bExtent.x())
            throw std::invalid_argument{"gemm requires C to match A.rows x B.cols."};

        internal::GemmFn::call(queue, exec, std::type_identity<T_Type>{}, A, B, C);
    }
} // namespace alpaka::nn::onHost

#if __has_include(<cublas_v2.h>)
#    include <cublas_v2.h>

namespace alpaka::nn::onHost
{
    namespace internal
    {
        template<typename T>
        struct CublasTraits;

        template<>
        struct CublasTraits<float>
        {
            static constexpr cudaDataType_t dataType = CUDA_R_32F;
            static constexpr cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_PEDANTIC;
        };

        template<>
        struct CublasTraits<double>
        {
            static constexpr cudaDataType_t dataType = CUDA_R_64F;
            static constexpr cublasComputeType_t computeType = CUBLAS_COMPUTE_64F_PEDANTIC;
        };

        template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Type>
        void fnDispatch(
            GemmFn::Spec<alpaka::api::Cuda, T_DeviceKind>,
            auto& queue,
            [[maybe_unused]] auto exec,
            std::type_identity<T_Type>,
            auto const& A,
            auto const& B,
            auto& C) requires(std::same_as<T_Type, float> || std::same_as<T_Type, double>)
        {
            cublasHandle_t handle;
            cublasStatus_t stat = cublasCreate(&handle);
            if(stat != CUBLAS_STATUS_SUCCESS)
            {
                throw std::invalid_argument(
                    "cublasCreate failed with error code: " + std::to_string(static_cast<int>(stat)));
            }
            cublasSetStream(handle, queue.getNativeHandle());

            int M = A.getExtents().y();
            int N = A.getExtents().x();
            int K = B.getExtents().x();
            T_Type alpha = 1;
            T_Type beta = 0;

            constexpr auto dataType = CublasTraits<T_Type>::dataType;
            constexpr auto computeType = CublasTraits<T_Type>::computeType;

            cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

            stat = cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                K,
                M,
                N,
                &alpha,
                B.data(),
                dataType,
                B.getPitches().y() / sizeof(T_Type),
                A.data(),
                dataType,
                A.getPitches().y() / sizeof(T_Type),
                &beta,
                C.data(),
                dataType,
                C.getPitches().y() / sizeof(T_Type),
                computeType,
                CUBLAS_GEMM_DEFAULT);

            cublasStatus_t destroyStat = cublasDestroy(handle);
            alpaka::unused(destroyStat);

            if(stat != CUBLAS_STATUS_SUCCESS)
            {
                throw std::invalid_argument(
                    "cublasGemmEx failed with error code: " + std::to_string(static_cast<int>(stat)));
            }
        }
    } // namespace internal
} // namespace alpaka::nn::onHost

#endif
