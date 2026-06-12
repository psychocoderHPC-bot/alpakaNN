/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace alpaka::nn::onAcc::internal::nn
{
    template<typename T_Type>
    struct RmsNormKernel
    {
        T_Type epsilon;

        ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto input, auto weight) const
        {
            auto const extents = out.getExtents();
            auto const hiddenAxis = ALPAKA_TYPEOF(extents)::dim() - 1u;
            auto const hiddenExtent = static_cast<uint32_t>(extents[hiddenAxis]);

            for(auto idx :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{extents}))
            {
                auto reduceIdx = idx;
                T_Type sumSquares{};
                for(uint32_t h = 0u; h < hiddenExtent; ++h)
                {
                    reduceIdx[hiddenAxis] = h;
                    auto const value = input[reduceIdx];
                    sumSquares += value * value;
                }

                auto const rms = alpaka::math::rsqrt(sumSquares / static_cast<T_Type>(hiddenExtent) + epsilon);
                out[idx] = input[idx] * rms * weight[alpaka::Vec{idx[hiddenAxis]}];
            }
        }
    };
} // namespace alpaka::nn::onAcc::internal::nn
