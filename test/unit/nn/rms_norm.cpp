/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpakaNN/alpakaNN.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using TestApis = alpakaNN::test::TestApis;

TEMPLATE_LIST_TEST_CASE("rmsNorm matches reference", "[nn][rmsnorm]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    auto input = alpaka::onHost::allocHost<float>(alpaka::Vec{2u, 4u});
    auto weight = alpaka::onHost::allocHost<float>(4u);
    auto output = alpaka::onHost::allocHost<float>(input.getExtents());
    for(auto idx : alpaka::IdxRange{input.getExtents()})
        input[idx] = static_cast<float>(idx[0] * 4u + idx[1] + 1u);
    for(uint32_t i = 0u; i < 4u; ++i)
        weight[alpaka::Vec{i}] = 0.5f + static_cast<float>(i);

    auto devIn = alpaka::onHost::allocLike(device, input);
    auto devWeight = alpaka::onHost::allocLike(device, weight);
    auto devOut = alpaka::onHost::allocLike(device, output);
    alpaka::onHost::memcpy(queue, devIn, input);
    alpaka::onHost::memcpy(queue, devWeight, weight);
    alpakaNN::nn::rmsNorm<float>(queue, exec, devIn, devWeight, devOut, 1.0e-5f);
    alpaka::onHost::memcpy(queue, output, devOut);
    alpaka::onHost::wait(queue);

    for(uint32_t row = 0u; row < 2u; ++row)
    {
        float sumSquares{};
        for(uint32_t col = 0u; col < 4u; ++col)
            sumSquares += input[alpaka::Vec{row, col}] * input[alpaka::Vec{row, col}];
        auto const invRms = 1.0f / std::sqrt(sumSquares / 4.0f + 1.0e-5f);
        for(uint32_t col = 0u; col < 4u; ++col)
            alpakaNN::test::checkValue(
                output[alpaka::Vec{row, col}],
                input[alpaka::Vec{row, col}] * invRms * weight[alpaka::Vec{col}]);
    }
}
