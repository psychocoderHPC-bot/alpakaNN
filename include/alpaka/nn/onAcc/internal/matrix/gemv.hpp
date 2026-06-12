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
    struct GemvKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, auto const W, auto const x, auto y) const
        {
            auto const matrixExtent = W.getExtents();
            for(auto idx :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{y.getExtents()}))
            {
                T_Type sum{};
                for(uint32_t inner = 0u; inner < matrixExtent.x(); ++inner)
                    sum += W[alpaka::Vec{idx[0], inner}] * x[alpaka::Vec{inner}];
                y[idx] = sum;
            }
        }
    };
} // namespace alpaka::nn::onAcc::internal::matrix
