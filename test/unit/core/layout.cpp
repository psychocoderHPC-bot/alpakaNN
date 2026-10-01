/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("layout tags expose stable ranks and fastest axis", "[core][layout]")
{
    STATIC_CHECK(alpaka::nn::layout::rank<alpaka::nn::layout::TH>() == 2u);
    STATIC_CHECK(alpaka::nn::layout::rank<alpaka::nn::layout::BTH>() == 3u);
    STATIC_CHECK(alpaka::nn::layout::rank<alpaka::nn::layout::BTHD>() == 4u);
    STATIC_CHECK(alpaka::nn::layout::rank<alpaka::nn::layout::BHTD>() == 4u);
    STATIC_CHECK(alpaka::nn::layout::rank<alpaka::nn::layout::LBHTD>() == 5u);
    STATIC_CHECK(alpaka::nn::layout::fastestAxis<alpaka::nn::layout::LBHTD>() == 4u);
}
