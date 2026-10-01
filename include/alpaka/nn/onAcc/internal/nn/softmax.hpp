/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>

namespace alpaka::nn::onAcc::internal::nn
{
    template<typename T_Type>
    struct SoftmaxKernel
    {
        uint32_t axis;
        bool causal;
        uint32_t queryAxis;
        uint32_t keyAxis;

        ALPAKA_FN_ACC bool masked(auto const& idx, uint32_t axisIndex) const
        {
            if(!causal)
                return false;
            auto probe = idx;
            probe[axis] = axisIndex;
            return probe[queryAxis] < probe[keyAxis];
        }

        ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto input) const
        {
            auto const axisExtent = static_cast<uint32_t>(input.getExtents()[axis]);
            auto const extents = out.getExtents();
            auto iterExtents = extents;
            iterExtents[axis] = 1u;

            for(auto iterIdx :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{iterExtents}))
            {
                auto probe = iterIdx;

                T_Type maxValue = std::numeric_limits<T_Type>::lowest();
                for(uint32_t i = 0u; i < axisExtent; ++i)
                {
                    if(masked(iterIdx, i))
                        continue;
                    probe[axis] = i;
                    maxValue = alpaka::math::max(maxValue, input[probe]);
                }

                T_Type sum{};
                for(uint32_t i = 0u; i < axisExtent; ++i)
                {
                    probe[axis] = i;
                    if(masked(iterIdx, i))
                    {
                        out[probe] = T_Type{};
                        continue;
                    }
                    auto const value = alpaka::math::exp(input[probe] - maxValue);
                    out[probe] = value;
                    sum += value;
                }

                for(uint32_t i = 0u; i < axisExtent; ++i)
                {
                    probe[axis] = i;
                    if(!masked(iterIdx, i))
                        out[probe] = out[probe] / sum;
                }
            }
        }
    };

    template<typename T_Type>
    struct MaskedSoftmaxKernel
    {
        uint32_t axis;

        ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto input, auto mask) const
        {
            auto const axisExtent = static_cast<uint32_t>(input.getExtents()[axis]);
            auto const extents = out.getExtents();
            auto iterExtents = extents;
            iterExtents[axis] = 1u;

            for(auto iterIdx :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{iterExtents}))
            {
                auto probe = iterIdx;
                T_Type maxValue = std::numeric_limits<T_Type>::lowest();
                for(uint32_t i = 0u; i < axisExtent; ++i)
                {
                    probe[axis] = i;
                    maxValue = alpaka::math::max(maxValue, input[probe] + mask[probe]);
                }

                T_Type sum{};
                for(uint32_t i = 0u; i < axisExtent; ++i)
                {
                    probe[axis] = i;
                    auto const value = alpaka::math::exp(input[probe] + mask[probe] - maxValue);
                    out[probe] = value;
                    sum += value;
                }

                for(uint32_t i = 0u; i < axisExtent; ++i)
                {
                    probe[axis] = i;
                    out[probe] = out[probe] / sum;
                }
            }
        }
    };
} // namespace alpaka::nn::onAcc::internal::nn
