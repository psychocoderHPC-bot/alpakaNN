/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace alpaka::nn::onHost::nn
{
    /** @brief Dense linear layer, row-major ``[in, out]`` weight.
     *
     * Computes ``output[batch, out] = input[batch, in] * weight[in, out]`` with a single
     * vendor-backed GEMM. The weight layout is the one required by the neural-bunch model
     * format (``order: row_major_in_out``), not the conventional ``[out, in]`` layout used by
     * some other frameworks. ``weight`` must therefore have extents ``[in, out]``; a caller that
     * stores ``[out, in]`` has to transpose it (or pass ``alpaka::blas::transposed(weight)`` to
     * the vendor BLAS directly) before calling this wrapper.
     *
     * The work is enqueued on @p queue; every view must stay alive until the queue work has
     * completed.
     */
    template<typename T_Type>
    void linear(auto& queue, auto const& input, auto const& weight, auto& output)
    {
        alpaka::nn::onHost::gemm<T_Type>(queue, input, weight, output);
    }

    /** @brief Dense linear layer followed by a per-column bias add.
     *
     * Computes ``output = input * weight + bias`` with the same row-major ``[in, out]`` weight
     * contract as linear(). @p bias is a one-dimensional view with extent ``out``.
     */
    template<typename T_Type>
    void linearBias(auto& queue, auto exec, auto const& input, auto const& weight, auto const& bias, auto& output)
    {
        linear<T_Type>(queue, input, weight, output);
        alpaka::nn::onHost::ops::biasAdd(queue, exec, output, bias, output);
    }

    /** @brief Elementwise activation supported by mlpForward(). */
    enum class Activation
    {
        none,
        relu,
        silu,
        gelu
    };

    /** @brief Thin wrapper around a weight view so a homogeneous span of layer weights can be built.
     *
     * The wrapped view is not copied into device memory and is not kept alive by this struct.
     */
    template<typename T_View>
    struct WeightView
    {
        T_View view;
    };

    /** @brief Thin wrapper around a bias view so a homogeneous span of layer biases can be built. */
    template<typename T_View>
    struct BiasView
    {
        T_View view;
    };

    namespace detail
    {
        template<typename T_Type>
        void applyActivation(auto& queue, auto exec, Activation activation, auto const& in, auto& out)
        {
            switch(activation)
            {
            case Activation::none:
                alpaka::nn::onHost::ops::copy(queue, exec, in, out);
                break;
            case Activation::relu:
                alpaka::nn::onHost::ops::relu<T_Type>(queue, exec, in, out);
                break;
            case Activation::silu:
                alpaka::nn::onHost::ops::silu<T_Type>(queue, exec, in, out);
                break;
            case Activation::gelu:
                alpaka::nn::onHost::ops::gelu<T_Type>(queue, exec, in, out);
                break;
            default:
                throw std::invalid_argument{"mlpForward: unsupported activation."};
            }
        }
    } // namespace detail

    template<typename T_Type, typename T_WeightView, typename T_BiasView>
    void mlpForward(
        auto& queue,
        auto exec,
        auto const& input,
        std::span<WeightView<T_WeightView> const> weights,
        std::span<BiasView<T_BiasView> const> biases,
        Activation hiddenActivation,
        Activation lastActivation,
        auto& output);

    template<typename T_Type, typename T_WeightView, typename T_BiasView, std::size_t T_Depth>
    void mlpForward(
        auto& queue,
        auto exec,
        auto const& input,
        std::array<WeightView<T_WeightView>, T_Depth> const& weights,
        std::array<BiasView<T_BiasView>, T_Depth> const& biases,
        Activation hiddenActivation,
        Activation lastActivation,
        auto& output);

    /** @brief Plain additive MLP forward pass (no gating), one GEMM plus bias and activation per layer.
     *
     * For every layer ``i`` the operation is
     * ``hidden = activation(input * weight[i] + bias[i])`` where each weight is stored row-major
     * as ``[in, out]`` and each bias is ``[out]``. The last layer uses @p output; intermediate
     * activations are applied layer by layer, so a trailing ``Activation::none`` implements the
     * usual regression head.
     *
     * @param queue queue the work is enqueued on.
     * @param exec executor used by the elementwise bias/activation kernels.
     * @param input ``[batch, in]`` activation.
     * @param weights span of per-layer weight views, each ``[in, out]`` (model format
     *        ``row_major_in_out``).
     * @param biases span of per-layer one-dimensional bias views, each ``[out]``.
     * @param activation applied after adding the bias on every layer.
     * @param output ``[batch, out_last]`` destination.
     *
     * The work is enqueued on @p queue and may complete asynchronously. Every intermediate
     * activation is allocated on ``queue.getDevice()`` and consumed by the next queued GEMM or
     * elementwise kernel: there is no host copy and no host synchronization between layers, so a
     * device-resident MLP tail stays on the accelerator for the whole forward pass. The caller
     * owns the lifetime of @p input, every wrapped view in @p weights / @p biases and @p output:
     * they must stay alive and valid until the enqueued work has completed. Internal scratch
     * buffers are kept alive with ``keepAlive(queue)``.
     */
    template<typename T_Type, typename T_WeightView, typename T_BiasView>
    void mlpForward(
        auto& queue,
        auto exec,
        auto const& input,
        std::span<WeightView<T_WeightView> const> weights,
        std::span<BiasView<T_BiasView> const> biases,
        Activation activation,
        auto& output)
    {
        mlpForward<T_Type>(queue, exec, input, weights, biases, activation, activation, output);
    }

    /** @brief Plain additive MLP forward pass with a separate activation for the last layer.
     *
     * The neural-bunch encoder uses ``silu`` on its hidden layers and ``none`` on its output
     * layer (``last_layer_activation: none``). This overload expresses that without changing the
     * required single-activation signature, which applies @p hiddenActivation to every layer.
     */
    template<typename T_Type, typename T_WeightView, typename T_BiasView>
    void mlpForward(
        auto& queue,
        auto exec,
        auto const& input,
        std::span<WeightView<T_WeightView> const> weights,
        std::span<BiasView<T_BiasView> const> biases,
        Activation hiddenActivation,
        Activation lastActivation,
        auto& output)
    {
        if(weights.empty())
            throw std::invalid_argument{"mlpForward: at least one layer is required."};
        if(weights.size() != biases.size())
            throw std::invalid_argument{"mlpForward: weights and biases must have the same length."};

        auto const batchExtent = input.getExtents()[0];
        auto const inputExtent = input.getExtents()[1];
        auto const batch = static_cast<std::size_t>(batchExtent);
        auto const inputDim = static_cast<std::size_t>(inputExtent);

        auto expectedInput = inputDim;
        for(std::size_t layer = 0u; layer < weights.size(); ++layer)
        {
            auto const& weightExtents = weights[layer].view.getExtents();
            auto const& biasExtents = biases[layer].view.getExtents();
            if(static_cast<std::size_t>(weightExtents[0]) != expectedInput)
                throw std::invalid_argument{
                    "mlpForward: layer " + std::to_string(layer) + " weight expects in="
                    + std::to_string(weightExtents[0]) + " but previous out=" + std::to_string(expectedInput) + "."};
            if(static_cast<std::size_t>(biasExtents[0]) != static_cast<std::size_t>(weightExtents[1]))
                throw std::invalid_argument{
                    "mlpForward: layer " + std::to_string(layer)
                    + " bias extent does not match the weight out extent."};
            expectedInput = static_cast<std::size_t>(weightExtents[1]);
        }
        if(static_cast<std::size_t>(output.getExtents()[0]) != batch
           || static_cast<std::size_t>(output.getExtents()[1]) != expectedInput)
            throw std::invalid_argument{"mlpForward: output extent does not match batch x last-out."};

        using Buffer
            = decltype(alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{batchExtent, inputExtent}));
        std::vector<Buffer> scratch;
        scratch.reserve(2u * weights.size() + 1u);

        scratch.push_back(alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{batchExtent, inputExtent}));
        std::size_t current = 0u;
        alpaka::nn::onHost::ops::copy(queue, exec, input, scratch[current]);

        for(std::size_t layer = 0u; layer < weights.size(); ++layer)
        {
            auto const outExtent = weights[layer].view.getExtents()[1];
            auto const last = layer + 1u == weights.size();

            scratch.push_back(alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{batchExtent, outExtent}));
            auto const pre = scratch.size() - 1u;
            alpaka::nn::onHost::gemm<T_Type>(queue, scratch[current], weights[layer].view, scratch[pre]);

            if(last)
            {
                alpaka::nn::onHost::ops::biasAdd(queue, exec, scratch[pre], biases[layer].view, output);
                if(lastActivation != Activation::none)
                    detail::applyActivation<T_Type>(queue, exec, lastActivation, output, output);
            }
            else
            {
                scratch.push_back(
                    alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{batchExtent, outExtent}));
                auto const post = scratch.size() - 1u;
                alpaka::nn::onHost::ops::biasAdd(queue, exec, scratch[pre], biases[layer].view, scratch[post]);
                if(hiddenActivation != Activation::none)
                    detail::applyActivation<T_Type>(queue, exec, hiddenActivation, scratch[post], scratch[post]);
                current = post;
            }
        }

        for(auto& buffer : scratch)
            buffer.keepAlive(queue);
    }

    /** @brief Convenience overload for a fixed compile-time depth.
     *
     * Identical to the span overload, but takes ``std::array`` so the layer count is part of the
     * type.
     */
    template<typename T_Type, typename T_WeightView, typename T_BiasView, std::size_t T_Depth>
    void mlpForward(
        auto& queue,
        auto exec,
        auto const& input,
        std::array<WeightView<T_WeightView>, T_Depth> const& weights,
        std::array<BiasView<T_BiasView>, T_Depth> const& biases,
        Activation activation,
        auto& output)
    {
        mlpForward<T_Type>(queue, exec, input, weights, biases, activation, activation, output);
    }

    /** @brief Fixed-depth overload with a separate last-layer activation. */
    template<typename T_Type, typename T_WeightView, typename T_BiasView, std::size_t T_Depth>
    void mlpForward(
        auto& queue,
        auto exec,
        auto const& input,
        std::array<WeightView<T_WeightView>, T_Depth> const& weights,
        std::array<BiasView<T_BiasView>, T_Depth> const& biases,
        Activation hiddenActivation,
        Activation lastActivation,
        auto& output)
    {
        mlpForward<T_Type>(
            queue,
            exec,
            input,
            std::span<WeightView<T_WeightView> const>{weights},
            std::span<BiasView<T_BiasView> const>{biases},
            hiddenActivation,
            lastActivation,
            output);
    }

    /** @brief Apply the linear decoder to a single latent row.
     *
     * The neural-bunch decoder stores its matrix as ``[latent, 1536]`` (``row_major_in_out``) and
     * evaluates ``F[1, 1536] = z[1, latent] * W[latent, 1536]``. This is a single-row GEMM; the
     * vendor gemv wrapper covers the other orientation only (``y = W * x`` with a ``[rows, cols]``
     * matrix), so a transposed gemv would need a [1536, latent] weight that the artifact does not
     * store. The wrapper exists to make that difference explicit at the call site.
     *
     * The decoder has no bias and no activation. @p exec is accepted for symmetry with the other
     * layer helpers; the vendor GEMM itself is driven by @p queue.
     */
    template<typename T_Type>
    void decodeLatent(auto& queue, auto exec, auto const& latent, auto const& weight, auto& output)
    {
        (void) exec;
        alpaka::nn::onHost::gemm<T_Type>(queue, latent, weight, output);
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
