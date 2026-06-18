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

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <cstdint>

namespace alpaka::nn::onHost::nn
{
    namespace detail
    {
        inline bool attentionTraceEnabled()
        {
            auto const* env = std::getenv("ALPAKANN_DEBUG_ATTENTION_TRACE");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }

        inline void printTrace(std::string const& message)
        {
            if(attentionTraceEnabled())
                std::fprintf(stderr, "%s\n", message.c_str());
        }
    } // namespace detail

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
        if(queriesPerKvGroup == 0u)
            throw std::invalid_argument{"attentionScores requires queriesPerKvGroup > 0."};
        if(detail::attentionTraceEnabled())
        {
            std::ostringstream os;
            os << "attentionScores q=" << Q.getExtents() << " k=" << K.getExtents() << " scores=" << scores.getExtents()
               << " queriesPerKvGroup=" << queriesPerKvGroup << " layout="
               << (kvLayout == alpaka::nn::AttentionKvLayout::BTHD ? "BTHD" : "BHTD");
            detail::printTrace(os.str());
        }
        auto const scoreExtents = scores.getExtents();
        auto const totalElements
            = static_cast<uint32_t>(scoreExtents[0]) * static_cast<uint32_t>(scoreExtents[1])
              * static_cast<uint32_t>(scoreExtents[2]) * static_cast<uint32_t>(scoreExtents[3]);
        queue.enqueue(
            alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, alpaka::Vec{totalElements}),
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
        if(queriesPerKvGroup == 0u)
            throw std::invalid_argument{"attentionApply requires queriesPerKvGroup > 0."};
        if(detail::attentionTraceEnabled())
        {
            std::ostringstream os;
            os << "attentionApply probs=" << probs.getExtents() << " values=" << values.getExtents()
               << " out=" << out.getExtents() << " queriesPerKvGroup=" << queriesPerKvGroup << " layout="
               << (kvLayout == alpaka::nn::AttentionKvLayout::BTHD ? "BTHD" : "BHTD");
            detail::printTrace(os.str());
        }
        auto const outExtents = out.getExtents();
        auto const totalElements = static_cast<uint32_t>(outExtents[0]) * static_cast<uint32_t>(outExtents[1])
                                   * static_cast<uint32_t>(outExtents[2]) * static_cast<uint32_t>(outExtents[3]);
        queue.enqueue(
            alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, alpaka::Vec{totalElements}),
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
