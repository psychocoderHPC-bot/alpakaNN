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
        struct EmbeddingLookupKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto tokenIds, auto embedding) const
            {
                for(auto idx : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{out.getExtents()}))
                {
                    auto const token = tokenIds[alpaka::Vec{idx[0]}];
                    out[idx] = embedding[alpaka::Vec{token, idx[1]}];
                }
            }
        };
    } // namespace detail

    template<typename T_Type>
    void embeddingLookup(auto& queue, auto exec, auto const& tokenIds, auto const& embedding, auto& output)
    {
        if(tokenIds.getExtents().dim() != 1u || embedding.getExtents().dim() != 2u || output.getExtents().dim() != 2u)
            throw std::invalid_argument{"embeddingLookup expects 1D ids and 2D embedding/output."};
        if(output.getExtents()[0] != tokenIds.getExtents()[0] || output.getExtents()[1] != embedding.getExtents()[1])
            throw std::invalid_argument{"embeddingLookup shape mismatch."};

        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{detail::EmbeddingLookupKernel<T_Type>{}, output, tokenIds, embedding});
    }
} // namespace alpakaNN::nn
