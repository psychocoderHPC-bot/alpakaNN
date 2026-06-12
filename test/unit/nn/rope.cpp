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

TEMPLATE_LIST_TEST_CASE("rope applies pairwise rotations", "[nn][rope]", TestApis)
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

    auto q = alpaka::onHost::allocHost<double>(alpaka::Vec{1u, 2u, 1u, 4u});
    auto k = alpaka::onHost::allocHost<double>(q.getExtents());
    auto cosTable = alpaka::onHost::allocHost<double>(alpaka::Vec{2u, 2u});
    auto sinTable = alpaka::onHost::allocHost<double>(alpaka::Vec{2u, 2u});
    auto outQ = alpaka::onHost::allocHost<double>(q.getExtents());
    auto outK = alpaka::onHost::allocHost<double>(q.getExtents());

    for(auto idx : alpaka::IdxRange{q.getExtents()})
    {
        q[idx] = static_cast<double>(idx[1] * 10u + idx[3] + 1u);
        k[idx] = q[idx] + 1.0;
    }
    for(uint32_t pos = 0u; pos < 2u; ++pos)
    {
        for(uint32_t pair = 0u; pair < 2u; ++pair)
        {
            auto angle = static_cast<double>(pos + pair + 1u) * 0.25;
            cosTable[alpaka::Vec{pos, pair}] = std::cos(angle);
            sinTable[alpaka::Vec{pos, pair}] = std::sin(angle);
        }
    }

    auto devQ = alpaka::onHost::allocLike(device, q);
    auto devK = alpaka::onHost::allocLike(device, k);
    auto devCos = alpaka::onHost::allocLike(device, cosTable);
    auto devSin = alpaka::onHost::allocLike(device, sinTable);
    auto devOutQ = alpaka::onHost::allocLike(device, outQ);
    auto devOutK = alpaka::onHost::allocLike(device, outK);
    alpaka::onHost::memcpy(queue, devQ, q);
    alpaka::onHost::memcpy(queue, devK, k);
    alpaka::onHost::memcpy(queue, devCos, cosTable);
    alpaka::onHost::memcpy(queue, devSin, sinTable);
    alpakaNN::nn::rope<double>(queue, exec, devQ, devK, devCos, devSin, devOutQ, devOutK);
    alpaka::onHost::memcpy(queue, outQ, devOutQ);
    alpaka::onHost::memcpy(queue, outK, devOutK);
    alpaka::onHost::wait(queue);

    for(uint32_t token = 0u; token < 2u; ++token)
    {
        for(uint32_t pair = 0u; pair < 2u; ++pair)
        {
            auto const c = cosTable[alpaka::Vec{token, pair}];
            auto const s = sinTable[alpaka::Vec{token, pair}];
            auto const baseQ0 = q[alpaka::Vec{0u, token, 0u, pair * 2u}];
            auto const baseQ1 = q[alpaka::Vec{0u, token, 0u, pair * 2u + 1u}];
            alpakaNN::test::checkValue(
                outQ[alpaka::Vec{0u, token, 0u, pair * 2u}],
                baseQ0 * c - baseQ1 * s,
                1.0e-12,
                1.0e-12);
            alpakaNN::test::checkValue(
                outQ[alpaka::Vec{0u, token, 0u, pair * 2u + 1u}],
                baseQ0 * s + baseQ1 * c,
                1.0e-12,
                1.0e-12);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("rope rotates q and k with different head counts", "[nn][rope]", TestApis)
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

    auto q = alpaka::onHost::allocHost<double>(alpaka::Vec{1u, 2u, 2u, 4u});
    auto k = alpaka::onHost::allocHost<double>(alpaka::Vec{1u, 2u, 1u, 4u});
    auto cosTable = alpaka::onHost::allocHost<double>(alpaka::Vec{2u, 2u});
    auto sinTable = alpaka::onHost::allocHost<double>(alpaka::Vec{2u, 2u});
    auto outQ = alpaka::onHost::allocHost<double>(q.getExtents());
    auto outK = alpaka::onHost::allocHost<double>(k.getExtents());

    for(auto idx : alpaka::IdxRange{q.getExtents()})
        q[idx] = static_cast<double>(idx[1] * 100u + idx[2] * 10u + idx[3] + 1u);
    for(auto idx : alpaka::IdxRange{k.getExtents()})
        k[idx] = static_cast<double>(idx[1] * 100u + idx[3] + 2u);
    for(uint32_t pos = 0u; pos < 2u; ++pos)
    {
        for(uint32_t pair = 0u; pair < 2u; ++pair)
        {
            auto angle = static_cast<double>(pos + pair + 1u) * 0.2;
            cosTable[alpaka::Vec{pos, pair}] = std::cos(angle);
            sinTable[alpaka::Vec{pos, pair}] = std::sin(angle);
        }
    }

    auto devQ = alpaka::onHost::allocLike(device, q);
    auto devK = alpaka::onHost::allocLike(device, k);
    auto devCos = alpaka::onHost::allocLike(device, cosTable);
    auto devSin = alpaka::onHost::allocLike(device, sinTable);
    auto devOutQ = alpaka::onHost::allocLike(device, outQ);
    auto devOutK = alpaka::onHost::allocLike(device, outK);
    alpaka::onHost::memcpy(queue, devQ, q);
    alpaka::onHost::memcpy(queue, devK, k);
    alpaka::onHost::memcpy(queue, devCos, cosTable);
    alpaka::onHost::memcpy(queue, devSin, sinTable);
    alpakaNN::nn::rope<double>(queue, exec, devQ, devK, devCos, devSin, devOutQ, devOutK);
    alpaka::onHost::memcpy(queue, outQ, devOutQ);
    alpaka::onHost::memcpy(queue, outK, devOutK);
    alpaka::onHost::wait(queue);

    REQUIRE(std::isfinite(outQ[alpaka::Vec{0u, 1u, 1u, 3u}]));
    REQUIRE(std::isfinite(outK[alpaka::Vec{0u, 1u, 0u, 3u}]));
}
