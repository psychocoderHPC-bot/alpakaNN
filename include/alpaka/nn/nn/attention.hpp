/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::nn
{
    /** @brief Memory layout of the key/value tensors consumed by the attention host operations.
     *
     * This enum only selects how the keys and values are laid out in memory; the query, score, probability and
     * output layouts are documented on the respective operations.
     */
    enum class AttentionKvLayout
    {
        /** @brief Batch x token x head x head-dimension: key/value tokens are axis 1, heads axis 2. */
        BTHD,
        /** @brief Batch x head x token x head-dimension: key/value tokens are axis 2, heads axis 1. */
        BHTD
    };
} // namespace alpaka::nn
