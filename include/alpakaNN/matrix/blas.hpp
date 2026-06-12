/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/fn.hpp>

#include <alpakaNN/core/shape.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if ALPAKANN_OPENBLAS_ENABLED
#    include <cblas.h>
#endif

namespace alpakaNN::matrix::detail
{
    template<typename T_Value>
    constexpr bool OpenBlasScalar = std::same_as<T_Value, float> || std::same_as<T_Value, double>;

    inline auto toBlasInt(auto value, char const* what) -> int
    {
        using ValueType = decltype(value);
        static_assert(std::is_integral_v<ValueType>);
        if(value > static_cast<ValueType>(std::numeric_limits<int>::max()))
            throw std::invalid_argument{std::string(what) + " exceeds OpenBLAS integer range."};
        return static_cast<int>(value);
    }

    template<typename T_View>
    inline auto leadingDimension(T_View const& view, char const* what) -> int
    {
        return toBlasInt(shape::elementPitchAt(view, 0u), what);
    }

    template<typename T_View>
    inline auto vectorIncrement(T_View const& view, char const* what) -> int
    {
        return toBlasInt(shape::elementPitchAt(view, 0u), what);
    }

    template<typename T_View>
    inline bool hasBlasMatrixLayout(T_View const& view)
    {
        return view.getExtents().dim() == 2u && shape::isAxisContiguous(view, 1u);
    }

    template<typename T_View>
    inline bool hasBlasVectorLayout(T_View const& view)
    {
        return view.getExtents().dim() == 1u;
    }
} // namespace alpakaNN::matrix::detail
