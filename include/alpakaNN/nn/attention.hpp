/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/matrix/gemm.hpp"
#include "alpakaNN/ops/elementwise.hpp"
#include "alpakaNN/nn/softmax.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>

namespace alpakaNN::nn
{
    enum class AttentionKvLayout
    {
        BTHD,
        BHTD
    };

    namespace detail
    {
        template<typename T_Type>
        struct AttentionScoresKernel
        {
            uint32_t queriesPerKvGroup;
            AttentionKvLayout kvLayout;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto scores, auto q, auto k) const
            {
                for(auto idx : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{scores.getExtents()}))
                {
                    T_Type sum{};
                    auto const queryHead = static_cast<uint32_t>(idx[1]);
                    auto const kvHead = queryHead / queriesPerKvGroup;
                    for(uint32_t d = 0u; d < q.getExtents()[3]; ++d)
                    {
                        auto const key = kvLayout == AttentionKvLayout::BTHD
                            ? k[alpaka::Vec{idx[0], idx[3], kvHead, d}]
                            : k[alpaka::Vec{idx[0], kvHead, idx[3], d}];
                        sum += q[alpaka::Vec{idx[0], idx[2], queryHead, d}] * key;
                    }
                    scores[idx] = sum;
                }
            }
        };

        template<typename T_Type>
        struct AttentionApplyKernel
        {
            uint32_t queriesPerKvGroup;
            AttentionKvLayout kvLayout;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto probs, auto values) const
            {
                for(auto idx : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{out.getExtents()}))
                {
                    T_Type sum{};
                    auto const queryHead = static_cast<uint32_t>(idx[2]);
                    auto const kvHead = queryHead / queriesPerKvGroup;
                    for(uint32_t key = 0u; key < probs.getExtents()[3]; ++key)
                    {
                        auto const value = kvLayout == AttentionKvLayout::BTHD
                            ? values[alpaka::Vec{idx[0], key, kvHead, idx[3]}]
                            : values[alpaka::Vec{idx[0], kvHead, key, idx[3]}];
                        sum += probs[alpaka::Vec{idx[0], queryHead, idx[1], key}] * value;
                    }
                    out[idx] = sum;
                }
            }
        };
    } // namespace detail

    template<typename T_Type>
    void qkvProjection(auto& queue, auto exec, auto const& input, auto const& Wq, auto const& Wk, auto const& Wv, auto& Q, auto& K, auto& V)
    {
        alpakaNN::gemm<T_Type>(queue, exec, input, Wq, Q);
        alpakaNN::gemm<T_Type>(queue, exec, input, Wk, K);
        alpakaNN::gemm<T_Type>(queue, exec, input, Wv, V);
    }

    template<typename T_Type>
    void attentionScores(
        auto& queue,
        auto exec,
        auto const& Q,
        auto const& K,
        auto& scores,
        uint32_t queriesPerKvGroup = 1u,
        AttentionKvLayout kvLayout = AttentionKvLayout::BTHD)
    {
        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, scores.getExtents()),
            alpaka::KernelBundle{
                detail::AttentionScoresKernel<T_Type>{queriesPerKvGroup, kvLayout},
                scores,
                Q,
                K});
    }

    template<typename T_Type>
    void attentionApply(
        auto& queue,
        auto exec,
        auto const& probs,
        auto const& values,
        auto& out,
        uint32_t queriesPerKvGroup = 1u,
        AttentionKvLayout kvLayout = AttentionKvLayout::BTHD)
    {
        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, out.getExtents()),
            alpaka::KernelBundle{
                detail::AttentionApplyKernel<T_Type>{queriesPerKvGroup, kvLayout},
                out,
                probs,
                values});
    }

    template<typename T_Type>
    void outputProjection(auto& queue, auto exec, auto const& input, auto const& Wo, auto& output)
    {
        alpakaNN::gemm<T_Type>(queue, exec, input, Wo, output);
    }
} // namespace alpakaNN::nn
