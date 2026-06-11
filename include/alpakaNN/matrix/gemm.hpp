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
        ALPAKA_FN_SYMBOL(GemmFn, alpaka::fn::Fallback::toAlpaka, alpaka::fn::Registration::enforced);

        template<typename T_Type, uint32_t T_TileSize>
        struct GemmKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto const A, auto const B, auto C) const
            {
                auto const aExtent = A.getExtents();
                for(auto idx : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{C.getExtents()}))
                {
                    T_Type sum{};
                    for(uint32_t inner = 0u; inner < aExtent.x(); ++inner)
                        sum += A[alpaka::Vec{idx[0], inner}] * B[alpaka::Vec{inner, idx[1]}];
                    C[idx] = sum;
                }
            }
        };

        template<alpaka::concepts::DeviceKind T_DeviceKind>
        constexpr void fnRegister(GemmFn::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>)
        {
        }

#if ALPAKANN_OPENBLAS_ENABLED
        template<alpaka::concepts::DeviceKind T_DeviceKind>
        constexpr void fnRegister(GemmFn::Spec<alpaka::api::Host, T_DeviceKind>)
        {
        }
#endif

        template<typename T_Type, uint32_t T_TileSize>
        void gemmAlpaka(auto& queue, auto exec, auto const& A, auto const& B, auto& C)
        {
            auto const cExtent = C.getExtents();
            constexpr auto blockExtent = alpaka::CVec<uint32_t, T_TileSize, T_TileSize>{};
            queue.enqueue(
                alpaka::onHost::FrameSpec{alpaka::divCeil(cExtent, blockExtent), blockExtent, exec},
                alpaka::KernelBundle{GemmKernel<T_Type, T_TileSize>{}, A, B, C});
        }

        template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Type, uint32_t T_TileSize>
        void fnDispatch(
            GemmFn::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>,
            auto const&,
            auto& queue,
            auto exec,
            std::type_identity<T_Type>,
            std::integral_constant<uint32_t, T_TileSize>,
            auto const& A,
            auto const& B,
            auto& C)
        {
            gemmAlpaka<T_Type, T_TileSize>(queue, exec, A, B, C);
        }

#if ALPAKANN_OPENBLAS_ENABLED
        template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Type, uint32_t T_TileSize>
        void fnDispatch(
            GemmFn::Spec<alpaka::api::Host, T_DeviceKind>,
            auto const&,
            auto& queue,
            auto exec,
            std::type_identity<T_Type>,
            std::integral_constant<uint32_t, T_TileSize>,
            auto const& A,
            auto const& B,
            auto& C)
        {
            if constexpr(!matrix::detail::OpenBlasScalar<T_Type>)
            {
                gemmAlpaka<T_Type, T_TileSize>(queue, exec, A, B, C);
            }
            else if(
                !matrix::detail::hasBlasMatrixLayout(A) || !matrix::detail::hasBlasMatrixLayout(B)
                || !matrix::detail::hasBlasMatrixLayout(C))
            {
                gemmAlpaka<T_Type, T_TileSize>(queue, exec, A, B, C);
            }
            else
            {
                alpaka::onHost::wait(queue);

                auto const m = matrix::detail::toBlasInt(A.getExtents()[0], "gemm M");
                auto const k = matrix::detail::toBlasInt(A.getExtents()[1], "gemm K");
                auto const n = matrix::detail::toBlasInt(B.getExtents()[1], "gemm N");
                auto const lda = matrix::detail::leadingDimension(A, "gemm lda");
                auto const ldb = matrix::detail::leadingDimension(B, "gemm ldb");
                auto const ldc = matrix::detail::leadingDimension(C, "gemm ldc");
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
        }
#endif
    } // namespace detail

    template<typename T_Type = float, uint32_t T_TileSize = 16u>
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

        detail::GemmFn::call(
            queue.getDevice(),
            queue,
            exec,
            std::type_identity<T_Type>{},
            std::integral_constant<uint32_t, T_TileSize>{},
            A,
            B,
            C);
    }
} // namespace alpakaNN
