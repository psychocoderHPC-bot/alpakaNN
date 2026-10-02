/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace alpaka::nn::layout
{
    /** @brief Row-major 2D token x hidden layout tag.
     *
     * Rank-2 (`T`, `H`) tag with the fastest axis last, as expected by the host GEMM/GEMV wrappers.
     */
    struct TH
    {
        static constexpr std::string_view name = "TH";
        static constexpr std::array<uint32_t, 2u> axes{0u, 1u};
    };

    /** @brief Row-major 3D batch x token x hidden layout tag. */
    struct BTH
    {
        static constexpr std::string_view name = "BTH";
        static constexpr std::array<uint32_t, 3u> axes{0u, 1u, 2u};
    };

    /** @brief Batch x token x head x head-dimension layout tag (token-major) for attention and RoPE tensors. */
    struct BTHD
    {
        static constexpr std::string_view name = "BTHD";
        static constexpr std::array<uint32_t, 4u> axes{0u, 1u, 2u, 3u};
    };

    /** @brief Batch x head x token x head-dimension layout tag (head-major); the alternative KV cache layout. */
    struct BHTD
    {
        static constexpr std::string_view name = "BHTD";
        static constexpr std::array<uint32_t, 4u> axes{0u, 1u, 2u, 3u};
    };

    /** @brief Layer x batch x head x token x head-dimension layout tag for stacked transformer layers. */
    struct LBHTD
    {
        static constexpr std::string_view name = "LBHTD";
        static constexpr std::array<uint32_t, 5u> axes{0u, 1u, 2u, 3u, 4u};
    };

    /** @brief Compile-time rank (number of axes) of a layout tag.
     *
     * @tparam T_Layout One of the layout tags in this namespace.
     * @return Number of axes, for example 4 for `BTHD`.
     */
    template<typename T_Layout>
    consteval uint32_t rank()
    {
        return static_cast<uint32_t>(T_Layout::axes.size());
    }

    /** @brief Index of the contiguous/fastest axis of a layout tag (always the last axis).
     *
     * @tparam T_Layout One of the layout tags in this namespace.
     * @return `rank<T_Layout>() - 1u`.
     */
    template<typename T_Layout>
    consteval uint32_t fastestAxis()
    {
        return rank<T_Layout>() - 1u;
    }

    /** @brief Validate at runtime that a view rank matches a layout tag.
     *
     * @tparam T_Layout Expected layout tag.
     * @param actualRank Rank of the view that is being checked.
     *
     * @throw std::invalid_argument if @p actualRank differs from `rank<T_Layout>()`.
     */
    template<typename T_Layout>
    constexpr void requireRank(uint32_t actualRank)
    {
        if(actualRank != rank<T_Layout>())
            throw std::invalid_argument{std::string(T_Layout::name) + " layout rank mismatch."};
    }
} // namespace alpaka::nn::layout
