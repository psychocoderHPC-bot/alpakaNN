/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
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
    /** @brief Value type of a view, with references and cv-qualifiers removed.
     *
     * @tparam T_View A view type exposing a `value_type` member.
     */
    template<typename T_View>
    using ValueType = std::remove_cv_t<typename std::remove_reference_t<T_View>::value_type>;

    /** @brief Extent (number of elements) of a view along one axis.
     *
     * @param extents Extent vector as returned by `view.getExtents()`.
     * @param axis Axis index; the caller must ensure it is in range.
     * @return `extents[axis]`.
     */
    inline auto extentAt(alpaka::concepts::Vector auto const& extents, uint32_t axis)
    {
        return extents[axis];
    }

    /** @brief Byte pitch of a view along one axis.
     *
     * @param pitches Pitch vector as returned by `view.getPitches()`; entries are byte offsets.
     * @param axis Axis index; the caller must ensure it is in range.
     * @return `pitches[axis]`.
     */
    inline auto pitchAtBytes(alpaka::concepts::Vector auto const& pitches, uint32_t axis)
    {
        return pitches[axis];
    }

    /** @brief Pitch of a view along one axis expressed in elements of the view's value type.
     *
     * @tparam T_View View type; its `value_type` determines the element size.
     * @param view View to query.
     * @param axis Axis index; the caller must ensure it is in range.
     * @return Byte pitch divided by `sizeof(ValueType<T_View>)`.
     */
    template<typename T_View>
    inline auto elementPitchAt(T_View const& view, uint32_t axis)
    {
        return pitchAtBytes(view.getPitches(), axis) / sizeof(ValueType<T_View>);
    }

    /** @brief Throw unless a view has the rank required by a layout tag.
     *
     * @tparam T_Layout Expected layout tag (see `alpaka::nn::layout`).
     * @param view View whose rank is checked.
     *
     * @throw std::invalid_argument if the view rank differs from `layout::rank<T_Layout>()`.
     */
    template<typename T_Layout>
    inline void requireLayout(alpaka::concepts::IMdSpan auto const& view)
    {
        auto const extents = view.getExtents();
        layout::requireRank<T_Layout>(ALPAKA_TYPEOF(extents)::dim());
    }

    /** @brief Throw unless @p axis is a valid axis of a view.
     *
     * @param view View whose rank bounds the axis.
     * @param axis Axis index to validate.
     * @param what Operation name used in the exception message.
     *
     * @throw std::invalid_argument if `axis >= rank(view)`.
     */
    inline void requireAxis(alpaka::concepts::IMdSpan auto const& view, uint32_t axis, std::string_view what)
    {
        auto const extents = view.getExtents();
        if(axis >= ALPAKA_TYPEOF(extents)::dim())
            throw std::invalid_argument{std::string(what) + " axis out of range."};
    }

    /** @brief Whether a single view axis is contiguous in memory (element pitch equal to 1).
     *
     * @param view View to query.
     * @param axis Axis index; must be in range.
     * @return `true` if the axis has an element pitch of one.
     *
     * @throw std::invalid_argument if @p axis is out of range.
     */
    inline bool isAxisContiguous(alpaka::concepts::IMdSpan auto const& view, uint32_t axis)
    {
        requireAxis(view, axis, "contiguous");
        return elementPitchAt(view, axis) == 1u;
    }

    /** @brief Whether an entire view is row-major contiguous.
     *
     * Walks the axes from the fastest (last) to the slowest (first) and checks that each pitch equals the product
     * of the extents already visited; a padded view is therefore not contiguous.
     *
     * @param view View to query.
     * @return `true` if the view is densely packed in row-major order.
     */
    inline bool isContiguous(alpaka::concepts::IMdSpan auto const& view)
    {
        auto const extents = view.getExtents();
        auto expectedPitch = std::size_t{1u};
        for(uint32_t axis = ALPAKA_TYPEOF(extents)::dim(); axis-- > 0u;)
        {
            if(elementPitchAt(view, axis) != expectedPitch)
                return false;
            expectedPitch *= static_cast<std::size_t>(extentAt(extents, axis));
        }
        return true;
    }

    /** @brief Throw unless a single view axis is contiguous.
     *
     * @param view View to check.
     * @param axis Axis index; must be in range.
     * @param what Operation name used in the exception message.
     *
     * @throw std::invalid_argument if @p axis is out of range or not contiguous.
     */
    inline void requireContiguousAxis(alpaka::concepts::IMdSpan auto const& view, uint32_t axis, std::string_view what)
    {
        if(!isAxisContiguous(view, axis))
            throw std::invalid_argument{std::string(what) + " requires a contiguous axis."};
    }
} // namespace alpaka::nn::shape
