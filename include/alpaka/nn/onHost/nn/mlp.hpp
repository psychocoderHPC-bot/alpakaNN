/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>

#include <stdexcept>

namespace alpaka::nn::onHost::nn
{
    template<typename T_Type>
    void linear(auto& queue, auto const& input, auto const& weight, auto& output)
    {
        alpaka::nn::onHost::gemm<T_Type>(queue, input, weight, output);
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

        linear<T_Type>(queue, input, Wgate, gate);
        alpaka::onHost::wait(queue);
        linear<T_Type>(queue, input, Wup, up);
        alpaka::onHost::wait(queue);
        alpaka::nn::onHost::ops::swiglu<T_Type>(queue, exec, gate, up, hidden);
        alpaka::onHost::wait(queue);
        linear<T_Type>(queue, hidden, Wdown, output);
        alpaka::onHost::wait(queue);
    }

    /** Enqueue the gated MLP using caller-owned intermediates.
     *
     * All tensors are rank-2 row-major matrices: input [batch,in], gate/up weights
     * [in,hidden], down weights [hidden,out], workspaces [batch,hidden], and output
     * [batch,out]. Operations are enqueued in order on the supplied queue. This
     * overload performs no allocation and does not wait; the caller owns workspace
     * lifetime and synchronization.
     */
    template<typename T_Type>
    void mlp(
        auto& queue,
        auto exec,
        auto const& input,
        auto const& Wgate,
        auto const& Wup,
        auto const& Wdown,
        auto& gate,
        auto& up,
        auto& hidden,
        auto& output)
    {
        static_assert(ALPAKA_TYPEOF(input.getExtents())::dim() == 2u, "MLP tensors must be rank 2");
        static_assert(ALPAKA_TYPEOF(Wgate.getExtents())::dim() == 2u, "MLP weights must be rank 2");
        static_assert(ALPAKA_TYPEOF(Wup.getExtents())::dim() == 2u, "MLP weights must be rank 2");
        static_assert(ALPAKA_TYPEOF(Wdown.getExtents())::dim() == 2u, "MLP weights must be rank 2");
        static_assert(ALPAKA_TYPEOF(gate.getExtents())::dim() == 2u, "MLP workspaces must be rank 2");
        static_assert(ALPAKA_TYPEOF(up.getExtents())::dim() == 2u, "MLP workspaces must be rank 2");
        static_assert(ALPAKA_TYPEOF(hidden.getExtents())::dim() == 2u, "MLP workspaces must be rank 2");
        static_assert(ALPAKA_TYPEOF(output.getExtents())::dim() == 2u, "MLP output must be rank 2");

        auto const batch = input.getExtents()[0];
        auto const inFeatures = input.getExtents()[1];
        auto const hiddenFeatures = Wgate.getExtents()[1];
        auto const outputFeatures = Wdown.getExtents()[1];
        if(Wgate.getExtents()[0] != inFeatures || Wup.getExtents()[0] != inFeatures
           || Wup.getExtents()[1] != hiddenFeatures || Wdown.getExtents()[0] != hiddenFeatures
           || gate.getExtents()[0] != batch || gate.getExtents()[1] != hiddenFeatures
           || up.getExtents() != gate.getExtents() || hidden.getExtents() != gate.getExtents()
           || output.getExtents()[0] != batch || output.getExtents()[1] != outputFeatures)
            throw std::invalid_argument{"MLP tensor shape mismatch."};

        linear<T_Type>(queue, input, Wgate, gate);
        linear<T_Type>(queue, input, Wup, up);
        alpaka::nn::onHost::ops::swiglu<T_Type>(queue, exec, gate, up, hidden);
        linear<T_Type>(queue, hidden, Wdown, output);
    }
} // namespace alpaka::nn::onHost::nn
