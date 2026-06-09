/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpakaNN/matrix/gemm.hpp>

namespace alpakaNN
{
    template<typename T_Type = float, uint32_t T_TileSize = 16u>
    void matrixMultiply(auto& queue, auto exec, auto const& A, auto const& B, auto& C)
    {
        gemm<T_Type, T_TileSize>(queue, exec, A, B, C);
    }
} // namespace alpakaNN
