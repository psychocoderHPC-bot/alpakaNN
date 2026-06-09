/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/detail/launch.hpp"
#include "alpakaNN/ops/elementwise.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpakaNN::nn
{
    enum class RopeLayout
    {
        BTHD,
        BHTD
    };

    namespace detail
    {
        template<typename T_Type>
        struct RopeKernel
        {
            uint32_t positionOffset;
            RopeLayout layout;

            template<typename T_View>
            ALPAKA_FN_ACC auto tokenIndex(T_View const& view, auto const& idx) const
            {
                alpaka::unused(view);
                return layout == RopeLayout::BTHD ? idx[1] : idx[2];
            }

            ALPAKA_FN_ACC void operator()(auto const& acc, auto outQ, auto outK, auto inQ, auto inK, auto cosTable, auto sinTable) const
            {
                auto const headDimAxis = outQ.getExtents().dim() - 1u;
                for(auto idx : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{outQ.getExtents()}))
                {
                    auto const component = static_cast<uint32_t>(idx[headDimAxis]);
                    auto const pair = component / 2u;
                    auto pairIdx = idx;
                    pairIdx[headDimAxis] = (component / 2u) * 2u;
                    auto const even = inQ[pairIdx];
                    pairIdx[headDimAxis] += 1u;
                    auto const odd = inQ[pairIdx];
                    auto const position = positionOffset + static_cast<uint32_t>(tokenIndex(outQ, idx));
                    auto const cosValue = cosTable[alpaka::Vec{position, pair}];
                    auto const sinValue = sinTable[alpaka::Vec{position, pair}];
                    auto const rotatedQ = (component % 2u == 0u) ? (even * cosValue - odd * sinValue) : (even * sinValue + odd * cosValue);

                    pairIdx = idx;
                    pairIdx[headDimAxis] = (component / 2u) * 2u;
                    auto const kEven = inK[pairIdx];
                    pairIdx[headDimAxis] += 1u;
                    auto const kOdd = inK[pairIdx];
                    auto const rotatedK = (component % 2u == 0u) ? (kEven * cosValue - kOdd * sinValue) : (kEven * sinValue + kOdd * cosValue);

                    outQ[idx] = rotatedQ;
                    outK[idx] = rotatedK;
                }
            }
        };
    } // namespace detail

    template<typename T_Type>
    void rope(
        auto& queue,
        auto exec,
        auto const& q,
        auto const& k,
        auto const& cosTable,
        auto const& sinTable,
        auto& outQ,
        auto& outK,
        RopeLayout layout = RopeLayout::BTHD,
        uint32_t positionOffset = 0u)
    {
        if(q.getExtents() != k.getExtents() || q.getExtents() != outQ.getExtents() || q.getExtents() != outK.getExtents())
            throw std::invalid_argument{"rope shape mismatch."};
        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, outQ.getExtents()),
            alpaka::KernelBundle{
                detail::RopeKernel<T_Type>{positionOffset, layout},
                outQ,
                outK,
                q,
                k,
                cosTable,
                sinTable});
    }

    template<typename T_Type>
    void ropeInPlace(
        auto& queue,
        auto exec,
        auto& q,
        auto& k,
        auto const& cosTable,
        auto const& sinTable,
        RopeLayout layout = RopeLayout::BTHD,
        uint32_t positionOffset = 0u)
    {
        auto tmpQ = alpaka::onHost::alloc<typename std::remove_reference_t<decltype(q)>::value_type>(queue.getDevice(), q.getExtents());
        auto tmpK = alpaka::onHost::alloc<typename std::remove_reference_t<decltype(k)>::value_type>(queue.getDevice(), k.getExtents());
        rope<T_Type>(queue, exec, q, k, cosTable, sinTable, tmpQ, tmpK, layout, positionOffset);
        alpakaNN::ops::copy(queue, exec, tmpQ, q);
        alpakaNN::ops::copy(queue, exec, tmpK, k);
        alpaka::onHost::wait(queue);
    }
} // namespace alpakaNN::nn
