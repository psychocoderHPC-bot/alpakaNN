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
