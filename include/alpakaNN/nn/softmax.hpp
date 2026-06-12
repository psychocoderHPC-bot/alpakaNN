/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/detail/launch.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace alpakaNN::nn
{
    namespace detail
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
                for(auto idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{out.getExtents()}))
                {
                    auto probe = idx;
                    T_Type maxValue = std::numeric_limits<T_Type>::lowest();
                    for(uint32_t i = 0u; i < axisExtent; ++i)
                    {
                        if(masked(idx, i))
                            continue;
                        probe[axis] = i;
                        maxValue = alpaka::math::max(maxValue, input[probe]);
                    }

                    T_Type sum{};
                    for(uint32_t i = 0u; i < axisExtent; ++i)
                    {
                        probe[axis] = i;
                        if(masked(idx, i))
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
                        if(!masked(idx, i))
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
                for(auto idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{out.getExtents()}))
                {
                    auto probe = idx;
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
    } // namespace detail

    template<typename T_Type>
    void softmax(auto& queue, auto exec, auto const& input, auto& output, uint32_t axis)
    {
        if(input.getExtents() != output.getExtents())
            throw std::invalid_argument{"softmax shape mismatch."};
        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{detail::SoftmaxKernel<T_Type>{axis, false, 0u, 0u}, output, input});
    }

    template<typename T_Type>
    void maskedSoftmax(auto& queue, auto exec, auto const& input, auto const& mask, auto& output, uint32_t axis)
    {
        if(input.getExtents() != output.getExtents() || input.getExtents() != mask.getExtents())
            throw std::invalid_argument{"maskedSoftmax shape mismatch."};
        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{detail::MaskedSoftmaxKernel<T_Type>{axis}, output, input, mask});
    }

    template<typename T_Type>
    void causalSoftmax(
        auto& queue,
        auto exec,
        auto const& input,
        auto& output,
        uint32_t axis,
        uint32_t queryAxis,
        uint32_t keyAxis)
    {
        if(input.getExtents() != output.getExtents())
            throw std::invalid_argument{"causalSoftmax shape mismatch."};
        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{detail::SoftmaxKernel<T_Type>{axis, true, queryAxis, keyAxis}, output, input});
    }
} // namespace alpakaNN::nn
