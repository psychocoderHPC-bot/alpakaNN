/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpaka/alpaka.hpp>

#include <alpakaNN/alpakaNN.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using TestApis = alpakaNN::test::TestApis;

TEMPLATE_LIST_TEST_CASE("gemv supports contiguous and subviews", "[matrix][gemv]", TestApis)
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

    auto hostW = alpaka::onHost::allocHost<float>(alpaka::Vec{3u, 4u});
    auto hostX = alpaka::onHost::allocHost<float>(4u);
    auto hostY = alpaka::onHost::allocHost<float>(3u);

    for(auto idx : alpaka::IdxRange{hostW.getExtents()})
        hostW[idx] = static_cast<float>(idx[0] * 10u + idx[1] + 1u);
    for(uint32_t i = 0u; i < hostX.getExtents()[0]; ++i)
        hostX[alpaka::Vec{i}] = static_cast<float>(i + 1u);

    auto devW = alpaka::onHost::allocLike(device, hostW);
    auto devX = alpaka::onHost::allocLike(device, hostX);
    auto devY = alpaka::onHost::allocLike(device, hostY);
    alpaka::onHost::memcpy(queue, devW, hostW);
    alpaka::onHost::memcpy(queue, devX, hostX);

    alpakaNN::gemv<float>(queue, exec, devW, devX, devY);
    alpaka::onHost::memcpy(queue, hostY, devY);
    alpaka::onHost::wait(queue);

    for(uint32_t row = 0u; row < hostY.getExtents()[0]; ++row)
    {
        float expected{};
        for(uint32_t col = 0u; col < hostX.getExtents()[0]; ++col)
            expected += hostW[alpaka::Vec{row, col}] * hostX[alpaka::Vec{col}];
        alpakaNN::test::checkValue(hostY[alpaka::Vec{row}], expected);
    }
}
