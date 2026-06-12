/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/nn/onHost/matrix/gemm.hpp>

namespace alpaka::nn::onHost
{
    template<typename T_Type = float, uint32_t T_TileSize = 16u>
    void matrixMultiply(auto& queue, auto exec, auto const& A, auto const& B, auto& C)
    {
        gemm<T_Type>(queue, exec, A, B, C);
    }
} // namespace alpaka::nn::onHost
