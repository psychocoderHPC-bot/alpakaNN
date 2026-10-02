/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>

namespace alpaka::nn::onHost::nn
{
    template<typename T_Type>
    void linear(auto& queue, auto const& input, auto const& weight, auto& output)
    {
        alpaka::nn::onHost::gemm<T_Type>(queue, input, weight, output);
    }

    /** @brief Run a gated MLP forward pass.
     *
     * The work is enqueued on @p queue and may complete asynchronously. The caller is responsible for the lifetime of
     * every view passed in (@p input, @p Wgate, @p Wup, @p Wdown, @p output): they must stay alive and valid until
     * the enqueued work has completed. Before consuming results on the host the caller must synchronize the queue, or
     * extend the lifetime of the involved views, in their own code as appropriate. The function's own internal scratch
     * buffers are kept alive internally.
     */
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

        linear<T_Type>(queue, input, Wgate, gate);
        linear<T_Type>(queue, input, Wup, up);
        alpaka::nn::onHost::ops::swiglu<T_Type>(queue, exec, gate, up, hidden);
        linear<T_Type>(queue, hidden, Wdown, output);
        // keep the internal temporaries alive until the enqueued pipeline has consumed them
        gate.keepAlive(queue);
        up.keepAlive(queue);
        hidden.keepAlive(queue);
    }
} // namespace alpaka::nn::onHost::nn
