/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpakaNN/alpakaNN.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using TestApis = alpakaNN::test::TestApis;

TEMPLATE_LIST_TEST_CASE("kv cache append and read", "[inference][kv-cache]", TestApis)
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

    auto cache = alpakaNN::inference::makeKvCache<float>(device, 2u, 1u, 2u, 4u, 3u);
    auto token = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 2u, 3u});
    for(auto idx : alpaka::IdxRange{token.getExtents()})
        token[idx] = static_cast<float>(idx[2] * 10u + idx[3] + 1u);
    auto devToken = alpaka::onHost::allocLike(device, token);
    alpaka::onHost::memcpy(queue, devToken, token);

    cache.append(queue, exec, 1u, 0u, 0u, devToken, devToken);
    auto keys = cache.getKeys(1u, 0u, 1u);
    auto hostKeys = alpaka::onHost::allocHost<float>(keys.getExtents());
    alpaka::onHost::memcpy(queue, hostKeys, keys);
    alpaka::onHost::wait(queue);

    REQUIRE(cache.length(1u, 0u) == 1u);
    for(auto idx : alpaka::IdxRange{hostKeys.getExtents()})
        alpakaNN::test::checkValue(hostKeys[idx], token[alpaka::Vec{0u, 0u, idx[1], idx[3]}]);
}
