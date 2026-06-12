/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

TEST_CASE("shape utilities detect contiguous and strided layouts", "[core][shape][view]")
{
    std::vector<float> data(32u, 0.0f);
    auto contiguous = alpaka::makeView(alpaka::api::host, data.data(), alpaka::Vec{2u, 4u});

    REQUIRE(alpaka::nn::shape::isContiguous(contiguous));
    REQUIRE(alpaka::nn::shape::isAxisContiguous(contiguous, 1u));
    REQUIRE_FALSE(alpaka::nn::shape::isAxisContiguous(contiguous, 0u));

    auto padded = alpaka::nn::onHost::view::makePaddedView<float>(
        data.data(),
        std::array<std::size_t, 2u>{2u, 4u},
        std::array<std::size_t, 2u>{6u, 1u});
    REQUIRE_FALSE(alpaka::nn::shape::isContiguous(padded));
    REQUIRE(alpaka::nn::shape::isAxisContiguous(padded, 1u));
}

TEST_CASE("subviews preserve aliasing without copies", "[core][view][subview]")
{
    auto buffer = alpaka::onHost::allocHost<int>(alpaka::Vec{2u, 6u});

    for(auto idx : alpaka::IdxRange{buffer.getExtents()})
        buffer[idx] = static_cast<int>(idx.y() * 10u + idx.x());

    auto slice = alpaka::nn::onHost::view::subView(buffer, alpaka::Vec{1u, 2u}, alpaka::Vec{1u, 3u});
    REQUIRE(slice.getExtents() == alpaka::Vec{1u, 3u});
    REQUIRE(slice[alpaka::Vec{0u, 0u}] == 12);
    REQUIRE(slice[alpaka::Vec{0u, 2u}] == 14);

    slice[alpaka::Vec{0u, 1u}] = 999;
    REQUIRE(buffer[alpaka::Vec{1u, 3u}] == 999);
}
