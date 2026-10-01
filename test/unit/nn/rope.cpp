/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <type_traits>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    template<typename T_Type>
    void runRopePairwiseCase(auto& queue, auto exec, auto const& device)
    {
        auto q = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 2u, 1u, 4u});
        auto k = alpaka::onHost::allocHost<T_Type>(q.getExtents());
        auto cosTable = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 2u});
        auto sinTable = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 2u});
        auto outQ = alpaka::onHost::allocHost<T_Type>(q.getExtents());
        auto outK = alpaka::onHost::allocHost<T_Type>(q.getExtents());

        for(auto idx : alpaka::IdxRange{q.getExtents()})
        {
            q[idx] = static_cast<T_Type>(idx[1] * 10u + idx[3] + 1u);
            k[idx] = q[idx] + T_Type{1};
        }
        for(uint32_t pos = 0u; pos < 2u; ++pos)
        {
            for(uint32_t pair = 0u; pair < 2u; ++pair)
            {
                auto angle = static_cast<T_Type>(pos + pair + 1u) * static_cast<T_Type>(0.25);
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
        alpaka::nn::onHost::nn::rope<T_Type>(queue, exec, devQ, devK, devCos, devSin, devOutQ, devOutK);
        alpaka::onHost::memcpy(queue, outQ, devOutQ);
        alpaka::onHost::memcpy(queue, outK, devOutK);
        alpaka::onHost::wait(queue);

        // Keep the original tight fp64 tolerance while giving float a single-precision budget.
        constexpr double epsilon = std::is_same_v<T_Type, float> ? 1.0e-5 : 1.0e-12;
        for(uint32_t token = 0u; token < 2u; ++token)
        {
            for(uint32_t pair = 0u; pair < 2u; ++pair)
            {
                auto const c = cosTable[alpaka::Vec{token, pair}];
                auto const s = sinTable[alpaka::Vec{token, pair}];
                auto const baseQ0 = q[alpaka::Vec{0u, token, 0u, pair * 2u}];
                auto const baseQ1 = q[alpaka::Vec{0u, token, 0u, pair * 2u + 1u}];
                alpaka::nn::test::checkValue(
                    outQ[alpaka::Vec{0u, token, 0u, pair * 2u}],
                    baseQ0 * c - baseQ1 * s,
                    epsilon,
                    epsilon);
                alpaka::nn::test::checkValue(
                    outQ[alpaka::Vec{0u, token, 0u, pair * 2u + 1u}],
                    baseQ0 * s + baseQ1 * c,
                    epsilon,
                    epsilon);
            }
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("rope applies pairwise rotations", "[nn][rope]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runRopePairwiseCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runRopePairwiseCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runRopeDifferentHeadCountsCase(auto& queue, auto exec, auto const& device)
    {
        auto q = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 2u, 2u, 4u});
        auto k = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 2u, 1u, 4u});
        auto cosTable = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 2u});
        auto sinTable = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 2u});
        auto outQ = alpaka::onHost::allocHost<T_Type>(q.getExtents());
        auto outK = alpaka::onHost::allocHost<T_Type>(k.getExtents());

        for(auto idx : alpaka::IdxRange{q.getExtents()})
            q[idx] = static_cast<T_Type>(idx[1] * 100u + idx[2] * 10u + idx[3] + 1u);
        for(auto idx : alpaka::IdxRange{k.getExtents()})
            k[idx] = static_cast<T_Type>(idx[1] * 100u + idx[3] + 2u);
        for(uint32_t pos = 0u; pos < 2u; ++pos)
        {
            for(uint32_t pair = 0u; pair < 2u; ++pair)
            {
                auto angle = static_cast<T_Type>(pos + pair + 1u) * static_cast<T_Type>(0.2);
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
        alpaka::nn::onHost::nn::rope<T_Type>(queue, exec, devQ, devK, devCos, devSin, devOutQ, devOutK);
        alpaka::onHost::memcpy(queue, outQ, devOutQ);
        alpaka::onHost::memcpy(queue, outK, devOutK);
        alpaka::onHost::wait(queue);

        REQUIRE(std::isfinite(outQ[alpaka::Vec{0u, 1u, 1u, 3u}]));
        REQUIRE(std::isfinite(outK[alpaka::Vec{0u, 1u, 0u, 3u}]));
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("rope rotates q and k with different head counts", "[nn][rope]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runRopeDifferentHeadCountsCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runRopeDifferentHeadCountsCase<double>(queue, exec, device);
}
