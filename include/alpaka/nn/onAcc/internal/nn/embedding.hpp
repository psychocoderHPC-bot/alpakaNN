/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

namespace alpaka::nn::onAcc::internal::nn
{
    template<typename T_Type>
    struct EmbeddingLookupKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto tokenIds, auto embedding) const
        {
            for(auto idx : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{out.getExtents()}))
            {
                auto const token = tokenIds[alpaka::Vec{idx[0]}];
                out[idx] = embedding[alpaka::Vec{token, idx[1]}];
            }
        }
    };
} // namespace alpaka::nn::onAcc::internal::nn
