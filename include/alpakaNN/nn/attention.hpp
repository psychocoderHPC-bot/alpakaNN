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
    namespace detail
    {
        template<typename T_Type>
        struct AttentionScoresKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto scores, auto q, auto k) const
            {
                for(auto idx : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{scores.getExtents()}))
                {
                    T_Type sum{};
                    for(uint32_t d = 0u; d < q.getExtents()[3]; ++d)
                        sum += q[alpaka::Vec{idx[0], idx[2], idx[1], d}] * k[alpaka::Vec{idx[0], idx[3], idx[1], d}];
                    scores[idx] = sum;
                }
            }
        };

        template<typename T_Type>
        struct AttentionApplyKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto probs, auto values) const
            {
                for(auto idx : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{out.getExtents()}))
                {
                    T_Type sum{};
                    for(uint32_t key = 0u; key < probs.getExtents()[3]; ++key)
                        sum += probs[alpaka::Vec{idx[0], idx[2], idx[1], key}] * values[alpaka::Vec{idx[0], key, idx[2], idx[3]}];
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
    void attentionScores(auto& queue, auto exec, auto const& Q, auto const& K, auto& scores)
    {
        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, scores.getExtents()),
            alpaka::KernelBundle{detail::AttentionScoresKernel<T_Type>{}, scores, Q, K});
    }

    template<typename T_Type>
    void attentionApply(auto& queue, auto exec, auto const& probs, auto const& values, auto& out)
    {
        queue.enqueue(
            alpakaNN::detail::makeFrameSpec(queue.getDevice(), exec, out.getExtents()),
            alpaka::KernelBundle{detail::AttentionApplyKernel<T_Type>{}, out, probs, values});
    }

    template<typename T_Type>
    void outputProjection(auto& queue, auto exec, auto const& input, auto const& Wo, auto& output)
    {
        alpakaNN::gemm<T_Type>(queue, exec, input, Wo, output);
    }
} // namespace alpakaNN::nn
