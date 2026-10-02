/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onAcc/internal/nn/softmax.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpaka::nn::onHost::nn
{
    namespace detail
    {
        auto softmaxLaunchExtents(auto const& outputExtents, uint32_t axis)
        {
            auto launchExtents = outputExtents;
            launchExtents[axis] = 1u;
            return launchExtents;
        }
    } // namespace detail

    /** @brief Numerically stable softmax over a single axis.
     *
     * For every index of the non-reduced axes the output is `exp(x - max) / sum(exp(x - max))` along @p axis.
     *
     * @tparam T_Type Element type of @p input and @p output.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param input Input view.
     * @param output Preallocated output with exactly the same extents as @p input.
     * @param axis Axis to normalize; must be in range.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous: the caller owns both views and must keep them alive and call
     *       `alpaka::onHost::wait(queue)` before reading @p output.
     */
    template<typename T_Type>
    void softmax(auto& queue, auto exec, auto const& input, auto& output, uint32_t axis)
    {
        if(input.getExtents() != output.getExtents())
            throw std::invalid_argument{"softmax shape mismatch."};
        queue.enqueue(
            alpaka::onHost::getFrameSpec(
                queue.getDevice(),
                exec,
                detail::softmaxLaunchExtents(output.getExtents(), axis)),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::SoftmaxKernel<T_Type>{axis, false, 0u, 0u},
                output,
                input});
    }

    /** @brief Softmax over @p axis with an additive mask of the same shape.
     *
     * The mask is added to the logits before the max-subtraction, so a large negative mask value effectively
     * removes an entry. The mask usually holds zeros and `-inf`/large negatives.
     *
     * @tparam T_Type Element type of @p input, @p mask and @p output.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param input Input logits.
     * @param mask Additive mask view with exactly the same extents as @p input.
     * @param output Preallocated output with exactly the same extents as @p input.
     * @param axis Axis to normalize; must be in range.
     *
     * @throw std::invalid_argument if the shapes of @p input, @p mask and @p output differ.
     * @note Asynchronous: the caller owns all views and must keep them alive and call
     *       `alpaka::onHost::wait(queue)` before reading @p output.
     */
    template<typename T_Type>
    void maskedSoftmax(auto& queue, auto exec, auto const& input, auto const& mask, auto& output, uint32_t axis)
    {
        if(input.getExtents() != output.getExtents() || input.getExtents() != mask.getExtents())
            throw std::invalid_argument{"maskedSoftmax shape mismatch."};
        queue.enqueue(
            alpaka::onHost::getFrameSpec(
                queue.getDevice(),
                exec,
                detail::softmaxLaunchExtents(output.getExtents(), axis)),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::MaskedSoftmaxKernel<T_Type>{axis},
                output,
                input,
                mask});
    }

    /** @brief Causal softmax: normalize over @p axis but zero and exclude entries with `query < key`.
     *
     * Used for decoder attention where position @p query may attend only to keys up to and including @p query.
     * Positions with `key > query` are written as zero; the remaining entries along @p axis sum to one.
     *
     * @tparam T_Type Element type of @p input and @p output.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param input Input logits.
     * @param output Preallocated output with exactly the same extents as @p input.
     * @param axis Axis to normalize (usually the key axis).
     * @param queryAxis Axis that holds the query position used in the causality test.
     * @param keyAxis Axis that holds the key position used in the causality test.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous: the caller owns both views and must keep them alive and call
     *       `alpaka::onHost::wait(queue)` before reading @p output.
     */
    template<typename T_Type>
    void causalSoftmax(
        auto& queue,
        auto exec,
        auto const& input,
        auto& output,
        uint32_t axis,
        uint32_t queryAxis,
        uint32_t keyAxis)
    {
        if(input.getExtents() != output.getExtents())
            throw std::invalid_argument{"softmax shape mismatch."};
        queue.enqueue(
            alpaka::onHost::getFrameSpec(
                queue.getDevice(),
                exec,
                detail::softmaxLaunchExtents(output.getExtents(), axis)),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::SoftmaxKernel<T_Type>{axis, true, queryAxis, keyAxis},
                output,
                input});
    }
} // namespace alpaka::nn::onHost::nn
