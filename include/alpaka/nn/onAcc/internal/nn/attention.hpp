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
    namespace detail
    {
        template<typename T_Extents>
        ALPAKA_FN_ACC auto flattenExtent(T_Extents const& extents)
        {
            return static_cast<uint32_t>(extents[0]) * static_cast<uint32_t>(extents[1])
                   * static_cast<uint32_t>(extents[2]) * static_cast<uint32_t>(extents[3]);
        }

        template<typename T_Extents>
        ALPAKA_FN_ACC auto unflatten4D(uint32_t linearIdx, T_Extents const& extents)
        {
            auto idx3 = linearIdx % static_cast<uint32_t>(extents[3]);
            linearIdx /= static_cast<uint32_t>(extents[3]);
            auto idx2 = linearIdx % static_cast<uint32_t>(extents[2]);
            linearIdx /= static_cast<uint32_t>(extents[2]);
            auto idx1 = linearIdx % static_cast<uint32_t>(extents[1]);
            linearIdx /= static_cast<uint32_t>(extents[1]);
            auto idx0 = linearIdx;
            return alpaka::Vec{idx0, idx1, idx2, idx3};
        }
    } // namespace detail

    template<typename T_Type>
    struct AttentionScoresKernel
    {
        uint32_t queriesPerKvGroup;
        alpaka::nn::AttentionKvLayout kvLayout;

        ALPAKA_FN_ACC void operator()(auto const& acc, auto scores, auto q, auto k) const
        {
            auto const scoreExtents = scores.getExtents();
            auto const totalElements = detail::flattenExtent(scoreExtents);
            for(auto linearIdx : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{totalElements}}))
            {
                auto const idx = detail::unflatten4D(static_cast<uint32_t>(linearIdx[0]), scoreExtents);
                T_Type sum{};
                auto const queryHead = static_cast<uint32_t>(idx[1]);
                auto const kvHead = queryHead / queriesPerKvGroup;
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
            auto const outExtents = out.getExtents();
            auto const totalElements = detail::flattenExtent(outExtents);
            for(auto linearIdx : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{totalElements}}))
            {
                auto const idx = detail::unflatten4D(static_cast<uint32_t>(linearIdx[0]), outExtents);
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
