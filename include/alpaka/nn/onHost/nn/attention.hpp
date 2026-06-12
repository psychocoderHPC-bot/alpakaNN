/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn/attention.hpp>
#include <alpaka/nn/onAcc/internal/nn/attention.hpp>
#include <alpaka/nn/onHost/internal/launch.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/nn/softmax.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>

#include <cstdint>

namespace alpaka::nn::onHost::nn
{
    template<typename T_Type>
    void qkvProjection(
        auto& queue,
        auto exec,
        auto const& input,
        auto const& Wq,
        auto const& Wk,
        auto const& Wv,
        auto& Q,
        auto& K,
        auto& V)
    {
        alpaka::nn::onHost::gemm<T_Type>(queue, exec, input, Wq, Q);
        alpaka::nn::onHost::gemm<T_Type>(queue, exec, input, Wk, K);
        alpaka::nn::onHost::gemm<T_Type>(queue, exec, input, Wv, V);
    }

    template<typename T_Type>
    void attentionScores(
        auto& queue,
        auto exec,
        auto const& Q,
        auto const& K,
        auto& scores,
        uint32_t queriesPerKvGroup = 1u,
        alpaka::nn::AttentionKvLayout kvLayout = alpaka::nn::AttentionKvLayout::BTHD)
    {
        queue.enqueue(
            alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, scores.getExtents()),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::AttentionScoresKernel<T_Type>{queriesPerKvGroup, kvLayout},
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
        alpaka::nn::AttentionKvLayout kvLayout = alpaka::nn::AttentionKvLayout::BTHD)
    {
        queue.enqueue(
            alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, out.getExtents()),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::AttentionApplyKernel<T_Type>{queriesPerKvGroup, kvLayout},
                out,
                probs,
                values});
    }

    template<typename T_Type>
    void outputProjection(auto& queue, auto exec, auto const& input, auto const& Wo, auto& output)
    {
        alpaka::nn::onHost::gemm<T_Type>(queue, exec, input, Wo, output);
    }
} // namespace alpaka::nn::onHost::nn
