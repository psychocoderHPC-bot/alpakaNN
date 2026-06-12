/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/core/layout.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace alpaka::nn::shape
{
    template<typename T_View>
    using ValueType = std::remove_cv_t<typename std::remove_reference_t<T_View>::value_type>;

    inline auto extentAt(alpaka::concepts::Vector auto const& extents, uint32_t axis)
    {
        return extents[axis];
    }

    inline auto pitchAtBytes(alpaka::concepts::Vector auto const& pitches, uint32_t axis)
    {
        return pitches[axis];
    }

    template<typename T_View>
    inline auto elementPitchAt(T_View const& view, uint32_t axis)
    {
        return pitchAtBytes(view.getPitches(), axis) / sizeof(ValueType<T_View>);
    }

    template<typename T_Layout>
    inline void requireLayout(alpaka::concepts::IMdSpan auto const& view)
    {
        layout::requireRank<T_Layout>(view.getExtents().dim());
    }

    inline void requireAxis(alpaka::concepts::IMdSpan auto const& view, uint32_t axis, std::string_view what)
    {
        if(axis >= view.getExtents().dim())
            throw std::invalid_argument{std::string(what) + " axis out of range."};
    }

    inline bool isAxisContiguous(alpaka::concepts::IMdSpan auto const& view, uint32_t axis)
    {
        requireAxis(view, axis, "contiguous");
        return elementPitchAt(view, axis) == 1u;
    }

    inline bool isContiguous(alpaka::concepts::IMdSpan auto const& view)
    {
        auto expectedPitch = std::size_t{1u};
        for(uint32_t axis = view.getExtents().dim(); axis-- > 0u;)
        {
            if(elementPitchAt(view, axis) != expectedPitch)
                return false;
            expectedPitch *= static_cast<std::size_t>(extentAt(view.getExtents(), axis));
        }
        return true;
    }

    inline void requireContiguousAxis(alpaka::concepts::IMdSpan auto const& view, uint32_t axis, std::string_view what)
    {
        if(!isAxisContiguous(view, axis))
            throw std::invalid_argument{std::string(what) + " requires a contiguous axis."};
    }
} // namespace alpaka::nn::shape
