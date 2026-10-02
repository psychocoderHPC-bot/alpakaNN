/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onAcc/internal/nn/rms_norm.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpaka::nn::onHost::nn
{
    /** @brief Root-mean-square layer normalization over the last axis.
     *
     * For each element `output[i] = input[i] / sqrt(mean(input[i]^2, last axis) + epsilon) * weight[last(i)]`.
     *
     * @tparam T_Type Element type of @p input, @p weight and @p output.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param input Input view of rank >= 1; the last axis is the normalized hidden dimension.
     * @param weight 1D weight view whose extent must equal the last axis of @p input.
     * @param output Preallocated output with exactly the same extents as @p input.
     * @param epsilon Small additive constant for numerical stability.
     *
     * @throw std::invalid_argument if @p input and @p output differ in shape, or if @p weight is not 1D or does not
     *        match the last axis.
     * @note Asynchronous: the caller owns all views and must keep them alive and call
     *       `alpaka::onHost::wait(queue)` before reading @p output.
     */
    template<typename T_Type>
    void rmsNorm(auto& queue, auto exec, auto const& input, auto const& weight, auto& output, T_Type epsilon)
    {
        auto const inputExtents = input.getExtents();
        auto const weightExtents = weight.getExtents();
        auto const outputExtents = output.getExtents();

        if(inputExtents != outputExtents)
            throw std::invalid_argument{"rmsNorm shape mismatch."};
        if(ALPAKA_TYPEOF(weightExtents)::dim() != 1u
           || weightExtents[0] != inputExtents[ALPAKA_TYPEOF(inputExtents)::dim() - 1u])
            throw std::invalid_argument{"rmsNorm expects a 1D weight matching the last axis."};

        queue.enqueue(
            alpaka::onHost::getFrameSpec(queue.getDevice(), exec, outputExtents),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::RmsNormKernel<T_Type>{epsilon},
                output,
                input,
                weight});
    }
} // namespace alpaka::nn::onHost::nn
