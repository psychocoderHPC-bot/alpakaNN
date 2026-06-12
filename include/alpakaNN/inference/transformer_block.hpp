/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/inference/kv_cache.hpp"
#include "alpakaNN/nn/attention.hpp"
#include "alpakaNN/nn/mlp.hpp"
#include "alpakaNN/nn/rms_norm.hpp"
#include "alpakaNN/nn/rope.hpp"
#include "alpakaNN/nn/softmax.hpp"
#include "alpakaNN/ops/elementwise.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace alpakaNN::inference
{
    template<typename T_Type, typename T_VectorBuffer, typename T_MatrixBuffer>
    struct TransformerBlockWeights
    {
        T_VectorBuffer rms1Weight;
        T_VectorBuffer rms2Weight;
        T_MatrixBuffer Wq;
        T_MatrixBuffer Wk;
        T_MatrixBuffer Wv;
        T_MatrixBuffer Wo;
        T_MatrixBuffer Wgate;
        T_MatrixBuffer Wup;
        T_MatrixBuffer Wdown;
        uint32_t numHeads;
        uint32_t numKeyValueHeads;
        uint32_t headDim;
        T_Type epsilon;
    };

    template<typename T_Type>
    void transformerBlock(
        auto& queue,
        auto exec,
        auto const& input,
        auto const& weights,
        auto& cache,
        uint32_t layer,
        auto const& ropeCos,
        auto const& ropeSin,
        auto& output)
    {
        auto norm1 = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto q = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto const tokens = static_cast<uint32_t>(input.getExtents()[0]);
        auto const kvWidth = weights.numKeyValueHeads * weights.headDim;
        auto const queriesPerKvGroup = weights.numHeads / weights.numKeyValueHeads;
        auto k = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{tokens, kvWidth});
        auto v = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{tokens, kvWidth});
        auto attn = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto proj = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto norm2 = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto mlpOut = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());

        nn::rmsNorm<T_Type>(queue, exec, input, weights.rms1Weight, norm1, weights.epsilon);
        nn::qkvProjection<T_Type>(queue, exec, norm1, weights.Wq, weights.Wk, weights.Wv, q, k, v);

        auto q4 = alpaka::makeView(
            queue.getDevice(),
            q.data(),
            alpaka::Vec{1u, tokens, weights.numHeads, weights.headDim});
        auto k4 = alpaka::makeView(
            queue.getDevice(),
            k.data(),
            alpaka::Vec{1u, tokens, weights.numKeyValueHeads, weights.headDim});
        auto v4 = alpaka::makeView(
            queue.getDevice(),
            v.data(),
            alpaka::Vec{1u, tokens, weights.numKeyValueHeads, weights.headDim});
        nn::ropeInPlace<T_Type>(queue, exec, q4, ropeCos, ropeSin);
        nn::ropeInPlace<T_Type>(queue, exec, k4, ropeCos, ropeSin);

        auto scores
            = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, weights.numHeads, tokens, tokens});
        auto probs = alpaka::onHost::alloc<T_Type>(queue.getDevice(), scores.getExtents());
        auto attn4 = alpaka::makeView(
            queue.getDevice(),
            attn.data(),
            alpaka::Vec{1u, tokens, weights.numHeads, weights.headDim});

        nn::attentionScores<T_Type>(queue, exec, q4, k4, scores, queriesPerKvGroup, nn::AttentionKvLayout::BTHD);
        alpakaNN::ops::scale<T_Type>(
            queue,
            exec,
            scores,
            static_cast<T_Type>(1) / alpaka::math::sqrt(static_cast<T_Type>(weights.headDim)),
            scores);
        nn::causalSoftmax<T_Type>(queue, exec, scores, probs, 3u, 2u, 3u);
        nn::attentionApply<T_Type>(queue, exec, probs, v4, attn4, queriesPerKvGroup, nn::AttentionKvLayout::BTHD);
        nn::outputProjection<T_Type>(queue, exec, attn, weights.Wo, proj);
        alpakaNN::ops::add<T_Type>(queue, exec, input, proj, output);
        nn::rmsNorm<T_Type>(queue, exec, output, weights.rms2Weight, norm2, weights.epsilon);
        nn::mlp<T_Type>(queue, exec, norm2, weights.Wgate, weights.Wup, weights.Wdown, mlpOut);
        alpakaNN::ops::add<T_Type>(queue, exec, output, mlpOut, output);

        for(uint32_t token = 0u; token < tokens; ++token)
        {
            auto kToken = k4.getSubView(
                alpaka::Vec{0u, token, 0u, 0u},
                alpaka::Vec{1u, 1u, weights.numKeyValueHeads, weights.headDim});
            auto vToken = v4.getSubView(
                alpaka::Vec{0u, token, 0u, 0u},
                alpaka::Vec{1u, 1u, weights.numKeyValueHeads, weights.headDim});
            cache.append(queue, exec, layer, 0u, token, kToken, vToken);
        }
        alpaka::onHost::wait(queue);
    }

    template<typename T_Type>
    void transformerBlockDecodeStep(
        auto& queue,
        auto exec,
        auto const& input,
        auto const& weights,
        auto& cache,
        uint32_t layer,
        auto const& ropeCos,
        auto const& ropeSin,
        auto& output)
    {
        auto norm1 = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto q = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto const kvWidth = weights.numKeyValueHeads * weights.headDim;
        auto const queriesPerKvGroup = weights.numHeads / weights.numKeyValueHeads;
        auto k = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, kvWidth});
        auto v = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, kvWidth});
        auto attn = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto proj = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto norm2 = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto mlpOut = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());

        nn::rmsNorm<T_Type>(queue, exec, input, weights.rms1Weight, norm1, weights.epsilon);
        nn::qkvProjection<T_Type>(queue, exec, norm1, weights.Wq, weights.Wk, weights.Wv, q, k, v);

        auto q4
            = alpaka::makeView(queue.getDevice(), q.data(), alpaka::Vec{1u, 1u, weights.numHeads, weights.headDim});
        auto k4 = alpaka::makeView(
            queue.getDevice(),
            k.data(),
            alpaka::Vec{1u, 1u, weights.numKeyValueHeads, weights.headDim});
        auto v4 = alpaka::makeView(
            queue.getDevice(),
            v.data(),
            alpaka::Vec{1u, 1u, weights.numKeyValueHeads, weights.headDim});
        auto const tokenPosition = cache.length(layer, 0u);
        nn::ropeInPlace<T_Type>(queue, exec, q4, ropeCos, ropeSin, nn::RopeLayout::BTHD, tokenPosition);
        nn::ropeInPlace<T_Type>(queue, exec, k4, ropeCos, ropeSin, nn::RopeLayout::BTHD, tokenPosition);

        cache.append(queue, exec, layer, 0u, tokenPosition, k4, v4);
        auto const contextTokens = cache.length(layer, 0u);
        auto keys = cache.getKeys(layer, 0u, contextTokens);
        auto values = cache.getValues(layer, 0u, contextTokens);

        auto scores
            = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, weights.numHeads, 1u, contextTokens});
        auto probs = alpaka::onHost::alloc<T_Type>(queue.getDevice(), scores.getExtents());
        auto attn4
            = alpaka::makeView(queue.getDevice(), attn.data(), alpaka::Vec{1u, 1u, weights.numHeads, weights.headDim});

        nn::attentionScores<T_Type>(queue, exec, q4, keys, scores, queriesPerKvGroup, nn::AttentionKvLayout::BHTD);
        alpakaNN::ops::scale<T_Type>(
            queue,
            exec,
            scores,
            static_cast<T_Type>(1) / alpaka::math::sqrt(static_cast<T_Type>(weights.headDim)),
            scores);
        nn::softmax<T_Type>(queue, exec, scores, probs, 3u);
        nn::attentionApply<T_Type>(queue, exec, probs, values, attn4, queriesPerKvGroup, nn::AttentionKvLayout::BHTD);
        nn::outputProjection<T_Type>(queue, exec, attn, weights.Wo, proj);
        alpakaNN::ops::add<T_Type>(queue, exec, input, proj, output);
        nn::rmsNorm<T_Type>(queue, exec, output, weights.rms2Weight, norm2, weights.epsilon);
        nn::mlp<T_Type>(queue, exec, norm2, weights.Wgate, weights.Wup, weights.Wdown, mlpOut);
        alpakaNN::ops::add<T_Type>(queue, exec, output, mlpOut, output);
        alpaka::onHost::wait(queue);
    }
} // namespace alpakaNN::inference
