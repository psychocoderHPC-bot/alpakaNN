/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace alpakaNN::layout
{
    struct TH
    {
        static constexpr std::string_view name = "TH";
        static constexpr std::array<uint32_t, 2u> axes{0u, 1u};
    };

    struct BTH
    {
        static constexpr std::string_view name = "BTH";
        static constexpr std::array<uint32_t, 3u> axes{0u, 1u, 2u};
    };

    struct BTHD
    {
        static constexpr std::string_view name = "BTHD";
        static constexpr std::array<uint32_t, 4u> axes{0u, 1u, 2u, 3u};
    };

    struct BHTD
    {
        static constexpr std::string_view name = "BHTD";
        static constexpr std::array<uint32_t, 4u> axes{0u, 1u, 2u, 3u};
    };

    struct LBHTD
    {
        static constexpr std::string_view name = "LBHTD";
        static constexpr std::array<uint32_t, 5u> axes{0u, 1u, 2u, 3u, 4u};
    };

    template<typename T_Layout>
    consteval uint32_t rank()
    {
        return static_cast<uint32_t>(T_Layout::axes.size());
    }

    template<typename T_Layout>
    consteval uint32_t fastestAxis()
    {
        return rank<T_Layout>() - 1u;
    }

    template<typename T_Layout>
    constexpr void requireRank(uint32_t actualRank)
    {
        if(actualRank != rank<T_Layout>())
            throw std::invalid_argument{std::string(T_Layout::name) + " layout rank mismatch."};
    }
} // namespace alpakaNN::layout
