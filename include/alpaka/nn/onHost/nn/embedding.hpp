/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onAcc/internal/nn/embedding.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpaka::nn::onHost::nn
{
    template<typename T_Type>
    void embeddingLookup(auto& queue, auto exec, auto const& tokenIds, auto const& embedding, auto& output)
    {
        auto const tokenExtents = tokenIds.getExtents();
        auto const embeddingExtents = embedding.getExtents();
        auto const outputExtents = output.getExtents();

        if(ALPAKA_TYPEOF(tokenExtents)::dim() != 1u || ALPAKA_TYPEOF(embeddingExtents)::dim() != 2u
           || ALPAKA_TYPEOF(outputExtents)::dim() != 2u)
            throw std::invalid_argument{"embeddingLookup expects 1D ids and 2D embedding/output."};
        if(outputExtents[0] != tokenExtents[0] || outputExtents[1] != embeddingExtents[1])
            throw std::invalid_argument{"embeddingLookup shape mismatch."};

        queue.enqueue(
            alpaka::onHost::getFrameSpec(queue.getDevice(), exec, outputExtents),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::EmbeddingLookupKernel<T_Type>{},
                output,
                tokenIds,
                embedding});
    }
} // namespace alpaka::nn::onHost::nn
