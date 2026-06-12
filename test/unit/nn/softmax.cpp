/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpakaNN/alpakaNN.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using TestApis = alpakaNN::test::TestApis;

TEMPLATE_LIST_TEST_CASE("softmax and causalSoftmax are stable", "[nn][softmax]", TestApis)
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

    auto input = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 2u, 3u});
    auto output = alpaka::onHost::allocHost<float>(input.getExtents());
    input[alpaka::Vec{0u, 0u, 0u, 0u}] = 1000.0f;
    input[alpaka::Vec{0u, 0u, 0u, 1u}] = 1001.0f;
    input[alpaka::Vec{0u, 0u, 0u, 2u}] = 999.0f;
    input[alpaka::Vec{0u, 0u, 1u, 0u}] = -1.0f;
    input[alpaka::Vec{0u, 0u, 1u, 1u}] = 0.0f;
    input[alpaka::Vec{0u, 0u, 1u, 2u}] = 1.0f;

    auto devIn = alpaka::onHost::allocLike(device, input);
    auto devOut = alpaka::onHost::allocLike(device, output);
    alpaka::onHost::memcpy(queue, devIn, input);
    alpakaNN::nn::causalSoftmax<float>(queue, exec, devIn, devOut, 3u, 2u, 3u);
    alpaka::onHost::memcpy(queue, output, devOut);
    alpaka::onHost::wait(queue);

    CHECK(output[alpaka::Vec{0u, 0u, 0u, 1u}] == 0.0f);
    CHECK(output[alpaka::Vec{0u, 0u, 0u, 2u}] == 0.0f);
    auto sum0 = output[alpaka::Vec{0u, 0u, 0u, 0u}];
    auto sum1 = output[alpaka::Vec{0u, 0u, 1u, 0u}] + output[alpaka::Vec{0u, 0u, 1u, 1u}]
                + output[alpaka::Vec{0u, 0u, 1u, 2u}];
    alpakaNN::test::checkValue(sum0, 1.0f);
    alpakaNN::test::checkValue(sum1, 1.0f);
}
