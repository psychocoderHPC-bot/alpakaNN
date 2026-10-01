/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <type_traits>
#include <vector>

using TestApis = alpaka::nn::test::TestApis;

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

namespace
{
    template<typename T_Type>
    void runAxisReductionsCase(auto& queue, auto exec, auto const& device)
    {
        auto hostIn = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 3u, 4u});
        initReductionInput<T_Type>(hostIn);
        auto hostOut
            = alpaka::onHost::allocHost<T_Type>(alpaka::nn::onHost::ops::makeReducedExtents(hostIn.getExtents(), 2u));
        auto devIn = alpaka::onHost::allocLike(device, hostIn);
        auto devOut = alpaka::onHost::allocLike(device, hostOut);

        alpaka::onHost::memcpy(queue, devIn, hostIn);

        // Keep the original tight fp64 tolerance while giving float a single-precision budget.
        constexpr double epsilon = std::is_same_v<T_Type, float> ? 1.0e-5 : 1.0e-12;

        // sum over fastest axis
        alpaka::nn::onHost::ops::reduceSum<T_Type>(queue, exec, devIn, devOut, 2u);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
        {
            auto base = idx;
            T_Type sum{};
            for(uint32_t r = 0u; r < hostIn.getExtents()[2]; ++r)
            {
                base[2] = r;
                sum += hostIn[base];
            }
            alpaka::nn::test::checkValue(hostOut[idx], sum, epsilon, epsilon);
        }

        // mean, max and sumSquares
        alpaka::nn::onHost::ops::reduceMean<T_Type>(queue, exec, devIn, devOut, 2u);
        alpaka::nn::onHost::ops::reduceMax<T_Type>(queue, exec, devIn, devOut, 2u);
        alpaka::nn::onHost::ops::reduceSumSquares<T_Type>(queue, exec, devIn, devOut, 2u);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
        {
            auto base = idx;
            T_Type expected{};
            for(uint32_t r = 0u; r < hostIn.getExtents()[2]; ++r)
            {
                base[2] = r;
                expected += hostIn[base] * hostIn[base];
            }
            alpaka::nn::test::checkValue(hostOut[idx], expected, epsilon, epsilon);
        }
    }

    template<typename T_Type>
    void runDotCase(auto& queue, auto exec, auto const& device)
    {
        auto lhsStorage = std::vector<T_Type>(48u, T_Type{});
        auto rhsStorage = std::vector<T_Type>(48u, T_Type{});
        auto lhs = alpaka::nn::onHost::view::makePaddedView<T_Type>(
            lhsStorage.data(),
            std::array<std::size_t, 3u>{2u, 3u, 4u},
            std::array<std::size_t, 3u>{24u, 8u, 1u});
        auto rhs = alpaka::nn::onHost::view::makePaddedView<T_Type>(
            rhsStorage.data(),
            std::array<std::size_t, 3u>{2u, 3u, 4u},
            std::array<std::size_t, 3u>{24u, 8u, 1u});
        initReductionInput<T_Type>(lhs);
        initReductionInput<T_Type>(rhs);
        auto hostDot
            = alpaka::onHost::allocHost<T_Type>(alpaka::nn::onHost::ops::makeReducedExtents(lhs.getExtents(), 1u));
        auto devLhs = alpaka::onHost::allocLike(device, lhs);
        auto devRhs = alpaka::onHost::allocLike(device, rhs);
        auto devDot = alpaka::onHost::allocLike(device, hostDot);
        alpaka::onHost::memcpy(queue, devLhs, lhs);
        alpaka::onHost::memcpy(queue, devRhs, rhs);

        constexpr double epsilon = std::is_same_v<T_Type, float> ? 1.0e-5 : 1.0e-12;

        alpaka::nn::onHost::ops::dot<T_Type>(queue, exec, devLhs, devRhs, devDot, 1u);
        alpaka::onHost::memcpy(queue, hostDot, devDot);
        alpaka::onHost::wait(queue);

        for(auto idx : alpaka::IdxRange{hostDot.getExtents()})
        {
            auto base = idx;
            T_Type expected{};
            for(uint32_t r = 0u; r < lhs.getExtents()[1]; ++r)
            {
                base[1] = r;
                expected += lhs[base] * rhs[base];
            }
            alpaka::nn::test::checkValue(hostDot[idx], expected, epsilon, epsilon);
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("axis reductions", "[ops][reduction]", TestApis)
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

    runAxisReductionsCase<float>(queue, exec, device);
    runDotCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
    {
        runAxisReductionsCase<double>(queue, exec, device);
        runDotCase<double>(queue, exec, device);
    }
}
