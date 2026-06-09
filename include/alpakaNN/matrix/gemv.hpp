/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/detail/launch.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpakaNN
{
    namespace detail
    {
        template<typename T_Type>
        struct GemvKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto const W, auto const x, auto y) const
            {
                auto const matrixExtent = W.getExtents();
                for(auto idx : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{y.getExtents()}))
                {
                    T_Type sum{};
                    for(uint32_t inner = 0u; inner < matrixExtent.x(); ++inner)
                        sum += W[alpaka::Vec{idx[0], inner}] * x[alpaka::Vec{inner}];
                    y[idx] = sum;
                }
            }
        };
    } // namespace detail

    template<typename T_Type = float>
    void gemv(auto& queue, auto exec, auto const& W, auto const& x, auto& y)
    {
        auto const wExtent = W.getExtents();
        auto const xExtent = x.getExtents();
        auto const yExtent = y.getExtents();

        if(wExtent.dim() != 2u || xExtent.dim() != 1u || yExtent.dim() != 1u)
            throw std::invalid_argument{"gemv expects W as 2D and x/y as 1D views."};
        if(wExtent.x() != xExtent[0] || wExtent.y() != yExtent[0])
            throw std::invalid_argument{"gemv shape mismatch."};

        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, yExtent),
            alpaka::KernelBundle{detail::GemvKernel<T_Type>{}, W, x, y});
    }
} // namespace alpakaNN
