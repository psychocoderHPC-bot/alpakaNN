/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace alpaka::nn::onAcc::internal::matrix
{
    template<typename T_Type>
    struct GemmKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, auto const A, auto const B, auto C) const
        {
            auto const aExtent = A.getExtents();
            for(auto idx : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{C.getExtents()}))
            {
                T_Type sum{};
                for(uint32_t inner = 0u; inner < aExtent.x(); ++inner)
                    sum += A[alpaka::Vec{idx[0], inner}] * B[alpaka::Vec{inner, idx[1]}];
                C[idx] = sum;
            }
        }
    };
} // namespace alpaka::nn::onAcc::internal::matrix
