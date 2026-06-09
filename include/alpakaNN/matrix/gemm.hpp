/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/detail/launch.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpakaNN
{
    namespace detail
    {
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

        constexpr auto blockExtent = alpaka::CVec<uint32_t, T_TileSize, T_TileSize>{};
        queue.enqueue(
            alpaka::onHost::FrameSpec{alpaka::divCeil(cExtent, blockExtent), blockExtent, exec},
            alpaka::KernelBundle{detail::GemmKernel<T_Type, T_TileSize>{}, A, B, C});
    }
} // namespace alpakaNN
