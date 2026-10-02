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
    /** @brief Dense linear layer `output = input * weight` via the vendor BLAS backend.
     *
     * Thin row-major dense wrapper over `alpaka::nn::onHost::gemm`; @p input is 2D `(M, K)`, @p weight is 2D
     * `(K, N)` and @p output is preallocated 2D `(M, N)`. Only `float` and `double` are supported.
     *
     * @tparam T_Type Scalar type; must be `float` or `double`.
     * @param queue alpaka queue the work is enqueued on.
     * @param input Input activations, 2D `(M, K)`.
     * @param weight Weight matrix, 2D `(K, N)`.
     * @param output Preallocated output, 2D `(M, N)`.
     *
     * @note Asynchronous and caller-owned: keep all views alive and call `alpaka::onHost::wait(queue)` before
     *       reading @p output.
     */
    template<typename T_Type>
    void linear(auto& queue, auto const& input, auto const& weight, auto& output)
    {
        alpaka::nn::onHost::gemm<T_Type>(queue, input, weight, output);
    }

    /** @brief Run a gated MLP forward pass.
     *
     * Executes `gate = input * Wgate`, `up = input * Wup`, `hidden = swiglu(gate, up)` and
     * `output = hidden * Wdown`, using internal scratch buffers for the intermediate activations. @p input is 2D
     * `(M, K)`, `Wgate`/`Wup` are `(K, H)`, `Wdown` is `(H, N)` and @p output is preallocated 2D `(M, N)`.
     *
     * The work is enqueued on @p queue and may complete asynchronously. The caller is responsible for the lifetime of
     * every view passed in (@p input, @p Wgate, @p Wup, @p Wdown, @p output): they must stay alive and valid until
     * the enqueued work has completed. Before consuming results on the host the caller must synchronize the queue, or
     * extend the lifetime of the involved views, in their own code as appropriate. The function's own internal scratch
     * buffers are kept alive internally.
     *
     * @tparam T_Type Scalar type; must be `float` or `double`.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue (used by the SwiGLU kernel).
     * @param input Input activations, 2D `(M, K)`.
     * @param Wgate Gate projection weight, 2D `(K, H)`.
     * @param Wup Up projection weight, 2D `(K, H)`.
     * @param Wdown Down projection weight, 2D `(H, N)`.
     * @param output Preallocated output, 2D `(M, N)`.
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
