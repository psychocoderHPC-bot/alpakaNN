/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/detail/launch.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpakaNN::nn
{
    namespace detail
    {
        template<typename T_Type>
        struct RmsNormKernel
        {
            T_Type epsilon;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto input, auto weight) const
            {
                auto const extents = out.getExtents();
                auto const hiddenAxis = extents.dim() - 1u;
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
    } // namespace detail

    template<typename T_Type>
    void rmsNorm(auto& queue, auto exec, auto const& input, auto const& weight, auto& output, T_Type epsilon)
    {
        if(input.getExtents() != output.getExtents())
            throw std::invalid_argument{"rmsNorm shape mismatch."};
        if(weight.getExtents().dim() != 1u
           || weight.getExtents()[0] != input.getExtents()[input.getExtents().dim() - 1u])
            throw std::invalid_argument{"rmsNorm expects a 1D weight matching the last axis."};

        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{detail::RmsNormKernel<T_Type>{epsilon}, output, input, weight});
    }
} // namespace alpakaNN::nn
