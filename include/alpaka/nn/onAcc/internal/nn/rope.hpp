/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn/rope.hpp>

#include <cstdint>

namespace alpaka::nn::onAcc::internal::nn
{
    template<typename T_Type>
    struct RopeSingleKernel
    {
        uint32_t positionOffset;
        alpaka::nn::RopeLayout layout;

        template<typename T_View>
        ALPAKA_FN_ACC auto tokenIndex(T_View const& view, auto const& idx) const
        {
            alpaka::unused(view);
            return layout == alpaka::nn::RopeLayout::BTHD ? idx[1] : idx[2];
        }

        ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto in, auto cosTable, auto sinTable) const
        {
            auto const outExtents = out.getExtents();
            auto const headDimAxis = ALPAKA_TYPEOF(outExtents)::dim() - 1u;
            for(auto idx :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{outExtents}))
            {
                auto const component = static_cast<uint32_t>(idx[headDimAxis]);
                auto const pair = component / 2u;
                auto pairIdx = idx;
                pairIdx[headDimAxis] = (component / 2u) * 2u;
                auto const even = in[pairIdx];
                pairIdx[headDimAxis] += 1u;
                auto const odd = in[pairIdx];
                auto const position = positionOffset + static_cast<uint32_t>(tokenIndex(out, idx));
                auto const cosValue = cosTable[alpaka::Vec{position, pair}];
                auto const sinValue = sinTable[alpaka::Vec{position, pair}];
                out[idx]
                    = (component % 2u == 0u) ? (even * cosValue - odd * sinValue) : (even * sinValue + odd * cosValue);
            }
        }
    };
} // namespace alpaka::nn::onAcc::internal::nn
