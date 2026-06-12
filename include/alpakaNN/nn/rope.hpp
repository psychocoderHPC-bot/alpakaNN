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
#include <type_traits>

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
        struct RopeSingleKernel
        {
            uint32_t positionOffset;
            RopeLayout layout;

            template<typename T_View>
            ALPAKA_FN_ACC auto tokenIndex(T_View const& view, auto const& idx) const
            {
                alpaka::unused(view);
                return layout == RopeLayout::BTHD ? idx[1] : idx[2];
            }

            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto in, auto cosTable, auto sinTable) const
            {
                auto const headDimAxis = out.getExtents().dim() - 1u;
                for(auto idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{out.getExtents()}))
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
                    out[idx] = (component % 2u == 0u) ? (even * cosValue - odd * sinValue)
                                                      : (even * sinValue + odd * cosValue);
                }
            }
        };

        inline void validateRopeShape(
            auto const& in,
            auto const& out,
            auto const& cosTable,
            auto const& sinTable,
            uint32_t positionOffset)
        {
            if(in.getExtents() != out.getExtents())
                throw std::invalid_argument{"rope shape mismatch."};
            auto const headDimAxis = in.getExtents().dim() - 1u;
            auto const headDim = static_cast<uint32_t>(in.getExtents()[headDimAxis]);
            if(headDim % 2u != 0u)
                throw std::invalid_argument{"rope requires an even head dimension."};
            if(cosTable.getExtents() != sinTable.getExtents())
                throw std::invalid_argument{"rope cosine/sine table shape mismatch."};
            if(static_cast<uint32_t>(cosTable.getExtents()[1]) != headDim / 2u)
                throw std::invalid_argument{"rope table pair count mismatch."};
            auto const tokenAxis = in.getExtents().dim() == 4u ? 1u : 0u;
            if(positionOffset + static_cast<uint32_t>(in.getExtents()[tokenAxis])
               > static_cast<uint32_t>(cosTable.getExtents()[0]))
                throw std::invalid_argument{"rope table does not cover requested positions."};
        }
    } // namespace detail

    template<typename T_Type>
    void rope(
        auto& queue,
        auto exec,
        auto const& in,
        auto const& cosTable,
        auto const& sinTable,
        auto& out,
        RopeLayout layout = RopeLayout::BTHD,
        uint32_t positionOffset = 0u)
    {
        detail::validateRopeShape(in, out, cosTable, sinTable, positionOffset);
        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, out.getExtents()),
            alpaka::KernelBundle{
                detail::RopeSingleKernel<T_Type>{positionOffset, layout},
                out,
                in,
                cosTable,
                sinTable});
    }

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
        rope<T_Type>(queue, exec, q, cosTable, sinTable, outQ, layout, positionOffset);
        rope<T_Type>(queue, exec, k, cosTable, sinTable, outK, layout, positionOffset);
    }

    template<typename T_Type>
    void ropeInPlace(
        auto& queue,
        auto exec,
        auto& tensor,
        auto const& cosTable,
        auto const& sinTable,
        RopeLayout layout = RopeLayout::BTHD,
        uint32_t positionOffset = 0u)
    {
        auto tmp = alpaka::onHost::alloc<typename std::remove_reference_t<decltype(tensor)>::value_type>(
            queue.getDevice(),
            tensor.getExtents());
        rope<T_Type>(queue, exec, tensor, cosTable, sinTable, tmp, layout, positionOffset);
        alpakaNN::ops::copy(queue, exec, tmp, tensor);
        alpaka::onHost::wait(queue);
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
        ropeInPlace<T_Type>(queue, exec, q, cosTable, sinTable, layout, positionOffset);
        ropeInPlace<T_Type>(queue, exec, k, cosTable, sinTable, layout, positionOffset);
    }
} // namespace alpakaNN::nn
