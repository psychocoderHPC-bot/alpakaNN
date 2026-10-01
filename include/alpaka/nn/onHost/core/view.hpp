/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/core/shape.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace alpaka::nn::onHost::view
{
    template<typename T_Type, std::size_t T_Size>
    inline auto makePaddedView(
        T_Type* data,
        std::array<std::size_t, T_Size> const& extents,
        std::array<std::size_t, T_Size> const& pitchesInElements)
    {
        using Vec = alpaka::Vec<std::size_t, static_cast<uint32_t>(T_Size)>;
        auto extentVec = Vec{};
        auto pitchVec = Vec{};
        for(std::size_t i = 0; i < T_Size; ++i)
        {
            extentVec[static_cast<uint32_t>(i)] = extents[i];
            pitchVec[static_cast<uint32_t>(i)] = pitchesInElements[i] * sizeof(T_Type);
        }
        return alpaka::makeView(alpaka::api::host, data, extentVec, pitchVec);
    }

    inline auto subView(
        alpaka::concepts::IMdSpan auto&& source,
        alpaka::concepts::VectorOrScalar auto const& offset,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        return source.getSubView(offset, extents);
    }
} // namespace alpaka::nn::onHost::view
