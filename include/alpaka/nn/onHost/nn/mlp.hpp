/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>

namespace alpaka::nn::onHost::nn
{
    template<typename T_Type>
    void linear(auto& queue, auto exec, auto const& input, auto const& weight, auto& output)
    {
        alpaka::nn::onHost::gemm<T_Type>(queue, exec, input, weight, output);
    }

    template<typename T_Type>
    void mlp(
        auto& queue,
        auto exec,
        auto const& input,
        auto const& Wgate,
        auto const& Wup,
        auto const& Wdown,
        auto& output)
    {
        auto gate = alpaka::onHost::alloc<T_Type>(
            queue.getDevice(),
            alpaka::Vec{input.getExtents()[0], Wgate.getExtents()[1]});
        auto up = alpaka::onHost::alloc<T_Type>(
            queue.getDevice(),
            alpaka::Vec{input.getExtents()[0], Wup.getExtents()[1]});
        auto hidden = alpaka::onHost::alloc<T_Type>(queue.getDevice(), gate.getExtents());

        linear<T_Type>(queue, exec, input, Wgate, gate);
        alpaka::onHost::wait(queue);
        linear<T_Type>(queue, exec, input, Wup, up);
        alpaka::onHost::wait(queue);
        alpaka::nn::onHost::ops::swiglu<T_Type>(queue, exec, gate, up, hidden);
        alpaka::onHost::wait(queue);
        linear<T_Type>(queue, exec, hidden, Wdown, output);
        alpaka::onHost::wait(queue);
    }
} // namespace alpaka::nn::onHost::nn
