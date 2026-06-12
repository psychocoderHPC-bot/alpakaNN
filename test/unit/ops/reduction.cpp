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

template<typename T_Type>
void initReductionInput(auto& in)
{
    T_Type value = static_cast<T_Type>(1);
    for(auto idx : alpaka::IdxRange{in.getExtents()})
    {
        in[idx] = value;
        value += static_cast<T_Type>(1);
    }
}

TEMPLATE_LIST_TEST_CASE("axis reductions", "[ops][reduction]", TestApis)
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

    auto hostIn = alpaka::onHost::allocHost<double>(alpaka::Vec{2u, 3u, 4u});
    initReductionInput<double>(hostIn);
    auto hostOut = alpaka::onHost::allocHost<double>(alpakaNN::ops::makeReducedExtents(hostIn.getExtents(), 2u));
    auto devIn = alpaka::onHost::allocLike(device, hostIn);
    auto devOut = alpaka::onHost::allocLike(device, hostOut);

    alpaka::onHost::memcpy(queue, devIn, hostIn);

    SECTION("sum mean max sumSquares over fastest axis")
    {
        alpakaNN::ops::reduceSum<double>(queue, exec, devIn, devOut, 2u);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
        {
            auto base = idx;
            double sum{};
            for(uint32_t r = 0u; r < hostIn.getExtents()[2]; ++r)
            {
                base[2] = r;
                sum += hostIn[base];
            }
            alpakaNN::test::checkValue(hostOut[idx], sum, 1.0e-12, 1.0e-12);
        }

        alpakaNN::ops::reduceMean<double>(queue, exec, devIn, devOut, 2u);
        alpakaNN::ops::reduceMax<double>(queue, exec, devIn, devOut, 2u);
        alpakaNN::ops::reduceSumSquares<double>(queue, exec, devIn, devOut, 2u);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
        {
            auto base = idx;
            double expected{};
            for(uint32_t r = 0u; r < hostIn.getExtents()[2]; ++r)
            {
                base[2] = r;
                expected += hostIn[base] * hostIn[base];
            }
            alpakaNN::test::checkValue(hostOut[idx], expected, 1.0e-12, 1.0e-12);
        }
    }

    SECTION("dot over middle axis with padded views")
    {
        auto lhsStorage = std::vector<float>(48u, 0.0f);
        auto rhsStorage = std::vector<float>(48u, 0.0f);
        auto lhs = alpakaNN::view::makePaddedView<float>(
            lhsStorage.data(),
            std::array<std::size_t, 3u>{2u, 3u, 4u},
            std::array<std::size_t, 3u>{24u, 8u, 1u});
        auto rhs = alpakaNN::view::makePaddedView<float>(
            rhsStorage.data(),
            std::array<std::size_t, 3u>{2u, 3u, 4u},
            std::array<std::size_t, 3u>{24u, 8u, 1u});
        initReductionInput<float>(lhs);
        initReductionInput<float>(rhs);
        auto hostDot = alpaka::onHost::allocHost<float>(alpakaNN::ops::makeReducedExtents(lhs.getExtents(), 1u));
        auto devLhs = alpaka::onHost::allocLike(device, lhs);
        auto devRhs = alpaka::onHost::allocLike(device, rhs);
        auto devDot = alpaka::onHost::allocLike(device, hostDot);
        alpaka::onHost::memcpy(queue, devLhs, lhs);
        alpaka::onHost::memcpy(queue, devRhs, rhs);

        alpakaNN::ops::dot<float>(queue, exec, devLhs, devRhs, devDot, 1u);
        alpaka::onHost::memcpy(queue, hostDot, devDot);
        alpaka::onHost::wait(queue);

        for(auto idx : alpaka::IdxRange{hostDot.getExtents()})
        {
            auto base = idx;
            float expected{};
            for(uint32_t r = 0u; r < lhs.getExtents()[1]; ++r)
            {
                base[1] = r;
                expected += lhs[base] * rhs[base];
            }
            alpakaNN::test::checkValue(hostDot[idx], expected);
        }
    }
}
