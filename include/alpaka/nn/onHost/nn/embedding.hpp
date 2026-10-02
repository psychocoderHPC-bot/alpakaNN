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
    /** @brief Gather rows of an embedding table by token id.
     *
     * `output[t, :] = embedding[tokenIds[t], :]` for every token @p t. The embedding table may be a non-contiguous
     * sub-view; its pitches are honored.
     *
     * @tparam T_Type Element type of @p embedding and @p output.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param tokenIds 1D view of token indices (typically `uint32_t`); values must be valid row indices into
     *                 @p embedding.
     * @param embedding 2D table of extents `(vocabulary, embeddingDim)`.
     * @param output Preallocated 2D output of extents `(tokenIds.extent(0), embeddingDim)`.
     *
     * @throw std::invalid_argument if the ranks are not 1D/2D/2D or the output extents do not derive from the
     *        inputs.
     * @note Asynchronous: the caller owns all three views and must keep them alive and call
     *       `alpaka::onHost::wait(queue)` before reading @p output.
     */
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
