/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::nn
{
    /** @brief Memory layout of the tensors passed to the RoPE host operations.
     *
     * The head dimension must be the last axis in both cases; the enum selects where the token axis lives.
     */
    enum class RopeLayout
    {
        /** @brief Batch x token x head x head-dimension: tokens are axis 1, heads axis 2. */
        BTHD,
        /** @brief Batch x head x token x head-dimension: tokens are axis 2, heads axis 1. */
        BHTD
    };
} // namespace alpaka::nn
