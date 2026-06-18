/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/inference/kv_cache.hpp>
#include <alpaka/nn/onHost/nn/attention.hpp>
#include <alpaka/nn/onHost/nn/mlp.hpp>
#include <alpaka/nn/onHost/nn/rms_norm.hpp>
#include <alpaka/nn/onHost/nn/rope.hpp>
#include <alpaka/nn/onHost/nn/softmax.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

namespace alpaka::nn::onHost::inference
{
    namespace detail
    {
        inline bool blockTraceEnabled()
        {
            auto const* env = std::getenv("ALPAKANN_DEBUG_BLOCK_TRACE");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }

        inline bool skipPrefillCacheAppend()
        {
            auto const* env = std::getenv("ALPAKANN_SKIP_PREFILL_CACHE_APPEND");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }

        inline bool detailedBlockTraceEnabled()
        {
            auto const* env = std::getenv("ALPAKANN_DEBUG_BLOCK_DETAILS");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }

        inline void printTrace(std::string const& message)
        {
            if(blockTraceEnabled())
                std::fprintf(stderr, "%s\n", message.c_str());
        }

        template<typename T_Type>
        void traceTensor(auto& queue, auto const& tensor, std::string_view label, uint32_t sampleCount = 8u)
        {
            if(!blockTraceEnabled())
                return;

            auto host = alpaka::onHost::allocHost<T_Type>(tensor.getExtents());
            alpaka::onHost::memcpy(queue, host, tensor);
            alpaka::onHost::wait(queue);

            std::ostringstream os;
            os << label << " extents=" << tensor.getExtents();

            auto const extents = host.getExtents();
            if constexpr(ALPAKA_TYPEOF(extents)::dim() == 2u)
            {
                if(extents[0] > 0u)
                {
                    auto const width = static_cast<uint32_t>(extents[1]);
                    auto const keep = std::min(sampleCount, width);
                    auto traceRow = [&](uint32_t row, std::string_view rowLabel)
                    {
                        T_Type sum{};
                        T_Type maxAbs{};
                        for(uint32_t col = 0u; col < width; ++col)
                        {
                            auto const value = host[alpaka::Vec{row, col}];
                            sum += value;
                            auto const absValue = value < T_Type{} ? -value : value;
                            if(absValue > maxAbs)
                                maxAbs = absValue;
                        }
                        os << ' ' << rowLabel << '=' << row << " first" << keep << '=';
                        for(uint32_t col = 0u; col < keep; ++col)
                            os << ' ' << host[alpaka::Vec{row, col}];
                        os << " sum=" << sum << " maxAbs=" << maxAbs;
                    };

                    traceRow(0u, "row0");
                    if(extents[0] > 2u)
                        traceRow(static_cast<uint32_t>(extents[0] / 2u), "rowMid");
                    if(extents[0] > 1u)
                        traceRow(static_cast<uint32_t>(extents[0] - 1u), "rowLast");
                }
            }
            else
            {
                auto it = host.begin();
                auto const end = host.end();
                T_Type sum{};
                T_Type maxAbs{};
                uint32_t count = 0u;
                os << " first" << sampleCount << '=';
                for(; it != end; ++it)
                {
                    auto const value = *it;
                    sum += value;
                    auto const absValue = value < T_Type{} ? -value : value;
                    if(absValue > maxAbs)
                        maxAbs = absValue;
                    if(count < sampleCount)
                        os << ' ' << value;
                    ++count;
                }
                os << " sum=" << sum << " maxAbs=" << maxAbs;
            }
            printTrace(os.str());
        }

        template<typename T_Type>
        void traceCompare(auto& queue, auto const& lhs, auto const& rhs, std::string const& label, T_Type tolerance)
        {
            if(!detailedBlockTraceEnabled())
                return;

            auto hostLhs = alpaka::onHost::allocHost<T_Type>(lhs.getExtents());
            auto hostRhs = alpaka::onHost::allocHost<T_Type>(rhs.getExtents());
            alpaka::onHost::memcpy(queue, hostLhs, lhs);
            alpaka::onHost::memcpy(queue, hostRhs, rhs);
            alpaka::onHost::wait(queue);

            uint32_t mismatchCount = 0u;
            std::string firstIdx = "n/a";
            T_Type firstLhs{};
            T_Type firstRhs{};
            float maxAbsDiff = 0.0f;
            for(auto idx : alpaka::IdxRange{hostLhs.getExtents()})
            {
                auto const lhsValue = hostLhs[idx];
                auto const rhsValue = hostRhs[idx];
                auto const absDiff = static_cast<float>(std::fabs(static_cast<double>(lhsValue - rhsValue)));
                maxAbsDiff = std::max(maxAbsDiff, absDiff);
                if(absDiff > static_cast<float>(tolerance))
                {
                    if(mismatchCount == 0u)
                    {
                        std::ostringstream idxStream;
                        idxStream << idx;
                        firstIdx = idxStream.str();
                        firstLhs = lhsValue;
                        firstRhs = rhsValue;
                    }
                    ++mismatchCount;
                }
            }

            std::ostringstream os;
            os << label << " mismatchCount=" << mismatchCount << " maxAbsDiff=" << maxAbsDiff;
            if(mismatchCount != 0u)
                os << " firstIdx=" << firstIdx << " lhs=" << firstLhs << " rhs=" << firstRhs;
            printTrace(os.str());
        }
    } // namespace detail

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
        auto const tokens = static_cast<uint32_t>(input.getExtents()[0]);
        auto const hiddenWidth = static_cast<uint32_t>(input.getExtents()[1]);
        auto const kvWidth = weights.numKeyValueHeads * weights.headDim;
        auto const queriesPerKvGroup = weights.numHeads / weights.numKeyValueHeads;
        auto qStorage = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{tokens * hiddenWidth});
        auto kStorage = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{tokens * kvWidth});
        auto vStorage = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{tokens * kvWidth});
        auto attnStorage = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{tokens * hiddenWidth});
        auto q = alpaka::makeView(queue.getDevice(), qStorage.data(), alpaka::Vec{tokens, hiddenWidth});
        auto k = alpaka::makeView(queue.getDevice(), kStorage.data(), alpaka::Vec{tokens, kvWidth});
        auto v = alpaka::makeView(queue.getDevice(), vStorage.data(), alpaka::Vec{tokens, kvWidth});
        auto attn = alpaka::makeView(queue.getDevice(), attnStorage.data(), alpaka::Vec{tokens, hiddenWidth});
        auto proj = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto residual1 = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto norm2 = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto mlpOut = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());

        alpaka::nn::onHost::nn::rmsNorm<T_Type>(queue, exec, input, weights.rms1Weight, norm1, weights.epsilon);
        alpaka::nn::onHost::nn::qkvProjection<T_Type>(queue, exec, norm1, weights.Wq, weights.Wk, weights.Wv, q, k, v);
        detail::traceTensor<T_Type>(queue, norm1, "transformerBlock layer " + std::to_string(layer) + " norm1");
        detail::traceTensor<T_Type>(queue, q, "transformerBlock layer " + std::to_string(layer) + " q");
        detail::traceTensor<T_Type>(queue, k, "transformerBlock layer " + std::to_string(layer) + " k");
        detail::traceTensor<T_Type>(queue, v, "transformerBlock layer " + std::to_string(layer) + " v");

        auto q4 = alpaka::makeView(
            queue.getDevice(),
            qStorage.data(),
            alpaka::Vec{1u, tokens, weights.numHeads, weights.headDim});
        auto k4 = alpaka::makeView(
            queue.getDevice(),
            kStorage.data(),
            alpaka::Vec{1u, tokens, weights.numKeyValueHeads, weights.headDim});
        auto v4 = alpaka::makeView(
            queue.getDevice(),
            vStorage.data(),
            alpaka::Vec{1u, tokens, weights.numKeyValueHeads, weights.headDim});
        alpaka::nn::onHost::nn::ropeInPlace<T_Type>(queue, exec, q4, ropeCos, ropeSin);
        alpaka::nn::onHost::nn::ropeInPlace<T_Type>(queue, exec, k4, ropeCos, ropeSin);
        detail::traceTensor<T_Type>(queue, q, "transformerBlock layer " + std::to_string(layer) + " q rope");
        detail::traceTensor<T_Type>(queue, k, "transformerBlock layer " + std::to_string(layer) + " k rope");

        auto scores
            = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, weights.numHeads, tokens, tokens});
        auto probs = alpaka::onHost::alloc<T_Type>(queue.getDevice(), scores.getExtents());
        auto attn4 = alpaka::makeView(
            queue.getDevice(),
            attnStorage.data(),
            alpaka::Vec{1u, tokens, weights.numHeads, weights.headDim});

        alpaka::nn::onHost::nn::attentionScores<T_Type>(
            queue,
            exec,
            q4,
            k4,
            scores,
            queriesPerKvGroup,
            alpaka::nn::AttentionKvLayout::BTHD);
        detail::traceTensor<T_Type>(queue, scores, "transformerBlock layer " + std::to_string(layer) + " scores raw");
        alpaka::nn::onHost::ops::scale<T_Type>(
            queue,
            exec,
            scores,
            static_cast<T_Type>(1) / alpaka::math::sqrt(static_cast<T_Type>(weights.headDim)),
            scores);
        detail::traceTensor<T_Type>(queue, scores, "transformerBlock layer " + std::to_string(layer) + " scores");
        alpaka::nn::onHost::nn::causalSoftmax<T_Type>(queue, exec, scores, probs, 3u, 2u, 3u);
        detail::traceTensor<T_Type>(queue, probs, "transformerBlock layer " + std::to_string(layer) + " probs");
        alpaka::nn::onHost::nn::attentionApply<T_Type>(
            queue,
            exec,
            probs,
            v4,
            attn4,
            queriesPerKvGroup,
            alpaka::nn::AttentionKvLayout::BTHD);
        detail::traceTensor<T_Type>(queue, attn, "transformerBlock layer " + std::to_string(layer) + " attn");
        alpaka::nn::onHost::nn::outputProjection<T_Type>(queue, exec, attn, weights.Wo, proj);
        detail::traceTensor<T_Type>(queue, proj, "transformerBlock layer " + std::to_string(layer) + " proj");
        alpaka::nn::onHost::ops::add<T_Type>(queue, exec, input, proj, residual1);
        detail::traceTensor<T_Type>(
            queue,
            residual1,
            "transformerBlock layer " + std::to_string(layer) + " residual1");
        alpaka::nn::onHost::nn::rmsNorm<T_Type>(queue, exec, residual1, weights.rms2Weight, norm2, weights.epsilon);
        detail::traceTensor<T_Type>(queue, norm2, "transformerBlock layer " + std::to_string(layer) + " norm2");
        alpaka::nn::onHost::nn::mlp<T_Type>(queue, exec, norm2, weights.Wgate, weights.Wup, weights.Wdown, mlpOut);
        detail::traceTensor<T_Type>(queue, mlpOut, "transformerBlock layer " + std::to_string(layer) + " mlp");
        alpaka::nn::onHost::ops::add<T_Type>(queue, exec, residual1, mlpOut, output);
        detail::traceTensor<T_Type>(
            queue,
            output,
            "transformerBlock layer " + std::to_string(layer) + " output before append");
        auto beforeAppend = alpaka::onHost::alloc<T_Type>(queue.getDevice(), output.getExtents());
        alpaka::nn::onHost::ops::copy(queue, exec, output, beforeAppend);

        if(!detail::skipPrefillCacheAppend())
        {
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
        }
        detail::traceTensor<T_Type>(
            queue,
            output,
            "transformerBlock layer " + std::to_string(layer) + " output after append");
        detail::traceCompare<T_Type>(
            queue,
            beforeAppend,
            output,
            "transformerBlock layer " + std::to_string(layer) + " output append delta",
            static_cast<T_Type>(1.0e-5));
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
        auto const hiddenWidth = static_cast<uint32_t>(input.getExtents()[1]);
        auto const kvWidth = weights.numKeyValueHeads * weights.headDim;
        auto const queriesPerKvGroup = weights.numHeads / weights.numKeyValueHeads;
        auto qStorage = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{hiddenWidth});
        auto kStorage = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{kvWidth});
        auto vStorage = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{kvWidth});
        auto attnStorage = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{hiddenWidth});
        auto q = alpaka::makeView(queue.getDevice(), qStorage.data(), alpaka::Vec{1u, hiddenWidth});
        auto k = alpaka::makeView(queue.getDevice(), kStorage.data(), alpaka::Vec{1u, kvWidth});
        auto v = alpaka::makeView(queue.getDevice(), vStorage.data(), alpaka::Vec{1u, kvWidth});
        auto attn = alpaka::makeView(queue.getDevice(), attnStorage.data(), alpaka::Vec{1u, hiddenWidth});
        auto proj = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto residual1 = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto norm2 = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());
        auto mlpOut = alpaka::onHost::alloc<T_Type>(queue.getDevice(), input.getExtents());

        alpaka::nn::onHost::nn::rmsNorm<T_Type>(queue, exec, input, weights.rms1Weight, norm1, weights.epsilon);
        alpaka::nn::onHost::nn::qkvProjection<T_Type>(queue, exec, norm1, weights.Wq, weights.Wk, weights.Wv, q, k, v);
        detail::traceTensor<T_Type>(queue, norm1, "decodeStep layer " + std::to_string(layer) + " norm1");
        detail::traceTensor<T_Type>(queue, q, "decodeStep layer " + std::to_string(layer) + " q");
        detail::traceTensor<T_Type>(queue, k, "decodeStep layer " + std::to_string(layer) + " k");
        detail::traceTensor<T_Type>(queue, v, "decodeStep layer " + std::to_string(layer) + " v");

        auto q4
            = alpaka::makeView(queue.getDevice(), qStorage.data(), alpaka::Vec{1u, 1u, weights.numHeads, weights.headDim});
        auto k4 = alpaka::makeView(
            queue.getDevice(),
            kStorage.data(),
            alpaka::Vec{1u, 1u, weights.numKeyValueHeads, weights.headDim});
        auto v4 = alpaka::makeView(
            queue.getDevice(),
            vStorage.data(),
            alpaka::Vec{1u, 1u, weights.numKeyValueHeads, weights.headDim});
        auto const tokenPosition = cache.length(layer, 0u);
        alpaka::nn::onHost::nn::ropeInPlace<T_Type>(
            queue,
            exec,
            q4,
            ropeCos,
            ropeSin,
            alpaka::nn::RopeLayout::BTHD,
            tokenPosition);
        alpaka::nn::onHost::nn::ropeInPlace<T_Type>(
            queue,
            exec,
            k4,
            ropeCos,
            ropeSin,
            alpaka::nn::RopeLayout::BTHD,
            tokenPosition);
        detail::traceTensor<T_Type>(queue, q, "decodeStep layer " + std::to_string(layer) + " q rope");
        detail::traceTensor<T_Type>(queue, k, "decodeStep layer " + std::to_string(layer) + " k rope");

        cache.append(queue, exec, layer, 0u, tokenPosition, k4, v4);
        auto const contextTokens = cache.length(layer, 0u);
        auto keys = cache.getKeys(layer, 0u, contextTokens);
        auto values = cache.getValues(layer, 0u, contextTokens);
        detail::traceTensor<T_Type>(queue, keys, "decodeStep layer " + std::to_string(layer) + " cache keys");
        detail::traceTensor<T_Type>(queue, values, "decodeStep layer " + std::to_string(layer) + " cache values");

        auto scores
            = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, weights.numHeads, 1u, contextTokens});
        auto probs = alpaka::onHost::alloc<T_Type>(queue.getDevice(), scores.getExtents());
        auto attn4
            = alpaka::makeView(queue.getDevice(), attnStorage.data(), alpaka::Vec{1u, 1u, weights.numHeads, weights.headDim});

        alpaka::nn::onHost::nn::attentionScores<T_Type>(
            queue,
            exec,
            q4,
            keys,
            scores,
            queriesPerKvGroup,
            alpaka::nn::AttentionKvLayout::BHTD);
        detail::traceTensor<T_Type>(queue, scores, "decodeStep layer " + std::to_string(layer) + " scores raw");
        alpaka::nn::onHost::ops::scale<T_Type>(
            queue,
            exec,
            scores,
            static_cast<T_Type>(1) / alpaka::math::sqrt(static_cast<T_Type>(weights.headDim)),
            scores);
        detail::traceTensor<T_Type>(queue, scores, "decodeStep layer " + std::to_string(layer) + " scores");
        alpaka::nn::onHost::nn::softmax<T_Type>(queue, exec, scores, probs, 3u);
        detail::traceTensor<T_Type>(queue, probs, "decodeStep layer " + std::to_string(layer) + " probs");
        alpaka::nn::onHost::nn::attentionApply<T_Type>(
            queue,
            exec,
            probs,
            values,
            attn4,
            queriesPerKvGroup,
            alpaka::nn::AttentionKvLayout::BHTD);
        detail::traceTensor<T_Type>(queue, attn, "decodeStep layer " + std::to_string(layer) + " attn");
        alpaka::nn::onHost::nn::outputProjection<T_Type>(queue, exec, attn, weights.Wo, proj);
        detail::traceTensor<T_Type>(queue, proj, "decodeStep layer " + std::to_string(layer) + " proj");
        alpaka::nn::onHost::ops::add<T_Type>(queue, exec, input, proj, residual1);
        detail::traceTensor<T_Type>(queue, residual1, "decodeStep layer " + std::to_string(layer) + " residual1");
        alpaka::nn::onHost::nn::rmsNorm<T_Type>(queue, exec, residual1, weights.rms2Weight, norm2, weights.epsilon);
        detail::traceTensor<T_Type>(queue, norm2, "decodeStep layer " + std::to_string(layer) + " norm2");
        alpaka::nn::onHost::nn::mlp<T_Type>(queue, exec, norm2, weights.Wgate, weights.Wup, weights.Wdown, mlpOut);
        detail::traceTensor<T_Type>(queue, mlpOut, "decodeStep layer " + std::to_string(layer) + " mlp");
        alpaka::nn::onHost::ops::add<T_Type>(queue, exec, residual1, mlpOut, output);
        detail::traceTensor<T_Type>(queue, output, "decodeStep layer " + std::to_string(layer) + " output");
        alpaka::onHost::wait(queue);
    }
} // namespace alpaka::nn::onHost::inference
