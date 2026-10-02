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
    /** @brief Create a host view over caller-owned memory with an explicit padded layout.
     *
     * Unlike a plain `alpaka::makeView`, the pitches are given in elements and can describe padding: a row stride
     * larger than the fastest extent leaves gaps between rows. The storage is not allocated or owned here, so
     * @p data must outlive every use of the returned view. Row-major operations that require contiguity (for
     * example the vendor GEMM/GEMV wrappers) reject a padded view; elementwise and reduction ops honor the pitches.
     *
     * @tparam T_Type Element type.
     * @tparam T_Size Number of dimensions.
     * @param data Pointer to the caller-owned storage; not copied or freed.
     * @param extents Extent of the view per axis, in elements.
     * @param pitchesInElements Distance in elements between consecutive elements along each axis.
     * @return A host view aliasing @p data with the requested extents and pitches.
     */
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

    /** @brief Create a sub-view of an existing view without copying.
     *
     * The result aliases @p source: writes through either view are visible through the other, and the original
     * storage must stay alive for as long as the sub-view is used.
     *
     * @param source View to slice.
     * @param offset Per-axis start offset (a vector, or a scalar broadcast to all axes).
     * @param extents Per-axis extent of the slice (a vector, or a scalar broadcast to all axes).
     * @return Sub-view of @p source starting at @p offset with extent @p extents.
     */
    inline auto subView(
        alpaka::concepts::IMdSpan auto&& source,
        alpaka::concepts::VectorOrScalar auto const& offset,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        return source.getSubView(offset, extents);
    }
} // namespace alpaka::nn::onHost::view
