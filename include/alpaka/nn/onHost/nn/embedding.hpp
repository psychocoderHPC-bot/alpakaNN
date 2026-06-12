/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onAcc/internal/nn/embedding.hpp>
#include <alpaka/nn/onHost/internal/launch.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpaka::nn::onHost::nn
{
    template<typename T_Type>
    void embeddingLookup(auto& queue, auto exec, auto const& tokenIds, auto const& embedding, auto& output)
    {
        if(tokenIds.getExtents().dim() != 1u || embedding.getExtents().dim() != 2u || output.getExtents().dim() != 2u)
            throw std::invalid_argument{"embeddingLookup expects 1D ids and 2D embedding/output."};
        if(output.getExtents()[0] != tokenIds.getExtents()[0] || output.getExtents()[1] != embedding.getExtents()[1])
            throw std::invalid_argument{"embeddingLookup shape mismatch."};

        queue.enqueue(
            alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::EmbeddingLookupKernel<T_Type>{},
                output,
                tokenIds,
                embedding});
    }
} // namespace alpaka::nn::onHost::nn
