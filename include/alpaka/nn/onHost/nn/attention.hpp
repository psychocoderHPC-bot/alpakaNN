/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn/attention.hpp>
#include <alpaka/nn/onAcc/internal/nn/attention.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/nn/softmax.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>

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

    /** @brief Fused query/key/value input projection: `Q = input * Wq`, `K = input * Wk`, `V = input * Wv`.
     *
     * A convenience wrapper that issues three vendor-BLAS GEMMs. All operands are row-major dense 2D views:
     * @p input is `(M, K)`, each weight is `(K, N)` and each output is preallocated `(M, N)`.
     *
     * @tparam T_Type Scalar type; must be `float` or `double`.
     * @param queue alpaka queue the work is enqueued on.
     * @param input Input activations, 2D `(M, K)`.
     * @param Wq Query projection weight, 2D `(K, Nq)`.
     * @param Wk Key projection weight, 2D `(K, Nk)`.
     * @param Wv Value projection weight, 2D `(K, Nv)`.
     * @param Q Preallocated query output, 2D `(M, Nq)`.
     * @param K Preallocated key output, 2D `(M, Nk)`.
     * @param V Preallocated value output, 2D `(M, Nv)`.
     *
     * @note Asynchronous and caller-owned: see `alpaka::nn::onHost::gemm`. Keep all views alive and call
     *       `alpaka::onHost::wait(queue)` before reading @p Q, @p K or @p V.
     */
    template<typename T_Type>
    void qkvProjection(
        auto& queue,
        auto const& input,
        auto const& Wq,
        auto const& Wk,
        auto const& Wv,
        auto& Q,
        auto& K,
        auto& V)
    {
        alpaka::nn::onHost::gemm<T_Type>(queue, input, Wq, Q);
        alpaka::nn::onHost::gemm<T_Type>(queue, input, Wk, K);
        alpaka::nn::onHost::gemm<T_Type>(queue, input, Wv, V);
    }

    /** @brief Dot-product attention scores without the softmax (unnormalized `sum_d Q . K`).
     *
     * Computes `scores[b, h, tq, tk] = sum_d Q[b, h, tq, d] * K[...]` where the key head is
     * `kvHead = h / queriesPerKvGroup`. Queries are always batch x query x head x head-dim (`BTHD`, i.e.
     * token-major); the keys are batch x token x kv-head x head-dim (`BTHD`, the default) or batch x kv-head x
     * token x head-dim (`BHTD`). @p scores must have extents `(B, numQueryHeads, numQueries, numKeys)`.
     *
     * @tparam T_Type Scalar type.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param Q Query tensor of extents `(B, numQueries, numQueryHeads, headDim)` (`BTHD`, token-major).
     * @param K Key tensor in the layout selected by @p kvLayout.
     * @param scores Preallocated output of extents `(B, numQueryHeads, numQueries, numKeys)`.
     * @param queriesPerKvGroup Number of query heads sharing one key/value head; must be greater than zero (grouped
     *        query attention). Use 1 for ordinary multi-head attention.
     * @param kvLayout Layout of the key tensor.
     *
     * @throw std::invalid_argument if @p queriesPerKvGroup is zero.
     * @note Asynchronous: the caller owns all views and must keep them alive and call
     *       `alpaka::onHost::wait(queue)` before reading @p scores.
     */
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
            os << "attentionScores q=" << Q.getExtents() << " k=" << K.getExtents()
               << " scores=" << scores.getExtents() << " queriesPerKvGroup=" << queriesPerKvGroup
               << " layout=" << (kvLayout == alpaka::nn::AttentionKvLayout::BTHD ? "BTHD" : "BHTD");
            detail::printTrace(os.str());
        }
        queue.enqueue(
            alpaka::onHost::getFrameSpec(queue.getDevice(), exec, scores.getExtents()),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::AttentionScoresKernel<T_Type>{queriesPerKvGroup, kvLayout},
                scores,
                Q,
                K});
    }

    /** @brief Apply attention probabilities to the values: `out = probs * V`.
     *
     * Computes `out[b, tq, h, d] = sum_tk probs[b, h, tq, tk] * V[...]` where the value head is
     * `kvHead = h / queriesPerKvGroup`. Probabilities are batch x head x query x key (`BHTD`); the values are in
     * the layout selected by @p kvLayout. @p out must have extents
     * `(B, numQueries, numQueryHeads, headDim)` (`BTHD`, token-major).
     *
     * @tparam T_Type Scalar type.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param probs Attention probabilities of extents `(B, numQueryHeads, numQueries, numKeys)` (`BHTD`).
     * @param values Value tensor in the layout selected by @p kvLayout.
     * @param out Preallocated output of extents `(B, numQueries, numQueryHeads, headDim)` (`BTHD`, token-major).
     * @param queriesPerKvGroup Number of query heads sharing one value head; must be greater than zero.
     * @param kvLayout Layout of the value tensor.
     *
     * @throw std::invalid_argument if @p queriesPerKvGroup is zero.
     * @note Asynchronous: the caller owns all views and must keep them alive and call
     *       `alpaka::onHost::wait(queue)` before reading @p out.
     */
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
               << " out=" << out.getExtents() << " queriesPerKvGroup=" << queriesPerKvGroup
               << " layout=" << (kvLayout == alpaka::nn::AttentionKvLayout::BTHD ? "BTHD" : "BHTD");
            detail::printTrace(os.str());
        }
        queue.enqueue(
            alpaka::onHost::getFrameSpec(queue.getDevice(), exec, out.getExtents()),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::AttentionApplyKernel<T_Type>{queriesPerKvGroup, kvLayout},
                out,
                probs,
                values});
    }

    /** @brief Output projection of an attention block: `output = input * Wo`.
     *
     * A thin row-major dense vendor-BLAS GEMM; see `alpaka::nn::onHost::gemm` for the operand contract.
     *
     * @tparam T_Type Scalar type; must be `float` or `double`.
     * @param queue alpaka queue the work is enqueued on.
     * @param input Attention result, 2D `(M, K)`.
     * @param Wo Output projection weight, 2D `(K, N)`.
     * @param output Preallocated output, 2D `(M, N)`.
     *
     * @note Asynchronous and caller-owned: keep all views alive and call `alpaka::onHost::wait(queue)` before
     *       reading @p output.
     */
    template<typename T_Type>
    void outputProjection(auto& queue, auto const& input, auto const& Wo, auto& output)
    {
        alpaka::nn::onHost::gemm<T_Type>(queue, input, Wo, output);
    }
} // namespace alpaka::nn::onHost::nn
