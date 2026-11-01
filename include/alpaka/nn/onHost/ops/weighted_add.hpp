/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace alpaka::nn::onHost::ops
{
    namespace detail
    {
        template<typename T_Type>
        struct WeightedAddKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto inputs, auto weights) const
            {
                auto const rows = inputs.getExtents()[0];
                using IndexType = std::remove_cvref_t<decltype(rows)>;
                for(auto idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{out.getExtents()}))
                {
                    auto const column = idx[0];
                    T_Type sum{};
                    // Fixed ascending row order: the accumulation is deterministic and independent
                    // of the grid/thread mapping.
                    for(IndexType row = 0u; row < rows; ++row)
                    {
                        sum += static_cast<T_Type>(weights[alpaka::Vec{row}])
                               * static_cast<T_Type>(inputs[alpaka::Vec{row, column}]);
                    }
                    out[idx] += sum;
                }
            }
        };
    } // namespace detail

    /** @brief Accumulate a weighted sum of N-tensors into an existing output tensor.
     *
     * Computes ``out[i] += sum_s weights[s] * inputs[s, i]`` for every element. This is the
     * additive accumulation used when combining per-source contributions. Each output element
     * sums the rows in a fixed ascending order, so the result is deterministic.
     *
     * @param queue queue the work is enqueued on.
     * @param exec executor used by the kernel.
     * @param inputs ``[S, N]`` row-major per-source tensors.
     * @param weights ``[S]`` per-row scalar weights.
     * @param output ``[N]`` accumulator; it is read and written in place.
     *
     * All views must stay alive until the enqueued work has completed.
     */
    template<typename T_Type>
    void weightedAdd(auto& queue, auto exec, auto const& inputs, auto const& weights, auto& output)
    {
        auto const& inputExtents = inputs.getExtents();
        auto const& weightExtents = weights.getExtents();
        auto const& outputExtents = output.getExtents();
        if(weightExtents[0] != inputExtents[0])
            throw std::invalid_argument{"weightedAdd: weights extent must match the input row count."};
        if(outputExtents[0] != inputExtents[1])
            throw std::invalid_argument{"weightedAdd: output extent must match the input column count."};
        detail::enqueue(
            queue,
            exec,
            output.getExtents(),
            alpaka::KernelBundle{detail::WeightedAddKernel<T_Type>{}, output, inputs, weights});
    }
} // namespace alpaka::nn::onHost::ops
