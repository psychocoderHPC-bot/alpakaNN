/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/nn/onAcc/internal/nn/rms_norm.hpp>
#include <alpaka/nn/onHost/internal/launch.hpp>

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpaka::nn::onHost::nn
{
    template<typename T_Type>
    void rmsNorm(auto& queue, auto exec, auto const& input, auto const& weight, auto& output, T_Type epsilon)
    {
        if(input.getExtents() != output.getExtents())
            throw std::invalid_argument{"rmsNorm shape mismatch."};
        if(weight.getExtents().dim() != 1u
           || weight.getExtents()[0] != input.getExtents()[input.getExtents().dim() - 1u])
            throw std::invalid_argument{"rmsNorm expects a 1D weight matching the last axis."};

        queue.enqueue(
            alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, output.getExtents()),
            alpaka::KernelBundle{alpaka::nn::onAcc::internal::nn::RmsNormKernel<T_Type>{epsilon}, output, input, weight});
    }
} // namespace alpaka::nn::onHost::nn
