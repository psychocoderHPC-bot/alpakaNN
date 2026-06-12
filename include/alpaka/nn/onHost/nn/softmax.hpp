/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/nn/onAcc/internal/nn/softmax.hpp>
#include <alpaka/nn/onHost/internal/launch.hpp>

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpaka::nn::onHost::nn
{
    template<typename T_Type>
    void softmax(auto& queue, auto exec, auto const& input, auto& output, uint32_t axis)
    {
        if(input.getExtents() != output.getExtents())
            throw std::invalid_argument{"softmax shape mismatch."};
        queue.enqueue(
            alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::SoftmaxKernel<T_Type>{axis, false, 0u, 0u},
                output,
                input});
    }

    template<typename T_Type>
    void maskedSoftmax(auto& queue, auto exec, auto const& input, auto const& mask, auto& output, uint32_t axis)
    {
        if(input.getExtents() != output.getExtents() || input.getExtents() != mask.getExtents())
            throw std::invalid_argument{"maskedSoftmax shape mismatch."};
        queue.enqueue(
            alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{alpaka::nn::onAcc::internal::nn::MaskedSoftmaxKernel<T_Type>{axis}, output, input, mask});
    }

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
            throw std::invalid_argument{"causalSoftmax shape mismatch."};
        queue.enqueue(
            alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::SoftmaxKernel<T_Type>{axis, true, queryAxis, keyAxis},
                output,
                input});
    }
} // namespace alpaka::nn::onHost::nn
