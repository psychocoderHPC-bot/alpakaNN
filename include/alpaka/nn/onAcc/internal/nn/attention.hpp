/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn/attention.hpp>

#include <cstdint>

namespace alpaka::nn::onAcc::internal::nn
{
    template<typename T_Type>
    struct AttentionScoresKernel
    {
        uint32_t queriesPerKvGroup;
        alpaka::nn::AttentionKvLayout kvLayout;

        ALPAKA_FN_ACC void operator()(auto const& acc, auto scores, auto q, auto k) const
        {
            for(auto idx : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{scores.getExtents()}))
            {
                auto const queryHead = static_cast<uint32_t>(idx[1]);
                auto const kvHead = queryHead / queriesPerKvGroup;
                T_Type sum{};
                for(uint32_t d = 0u; d < q.getExtents()[3]; ++d)
                {
                    auto const key = kvLayout == alpaka::nn::AttentionKvLayout::BTHD
                                         ? k[alpaka::Vec{idx[0], idx[3], kvHead, d}]
                                         : k[alpaka::Vec{idx[0], kvHead, idx[3], d}];
                    sum += q[alpaka::Vec{idx[0], idx[2], queryHead, d}] * key;
                }
                scores[idx] = sum;
            }
        }
    };

    template<typename T_Type>
    struct AttentionApplyKernel
    {
        uint32_t queriesPerKvGroup;
        alpaka::nn::AttentionKvLayout kvLayout;

        ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto probs, auto values) const
        {
            for(auto idx : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{out.getExtents()}))
            {
                T_Type sum{};
                auto const queryHead = static_cast<uint32_t>(idx[2]);
                auto const kvHead = queryHead / queriesPerKvGroup;
                for(uint32_t key = 0u; key < probs.getExtents()[3]; ++key)
                {
                    auto const value = kvLayout == alpaka::nn::AttentionKvLayout::BTHD
                                           ? values[alpaka::Vec{idx[0], key, kvHead, idx[3]}]
                                           : values[alpaka::Vec{idx[0], kvHead, key, idx[3]}];
                    sum += probs[alpaka::Vec{idx[0], queryHead, idx[1], key}] * value;
                }
                out[idx] = sum;
            }
        }
    };
} // namespace alpaka::nn::onAcc::internal::nn
