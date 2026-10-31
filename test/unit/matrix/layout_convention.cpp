/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "test.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    // A 2x3 times 3x4 product with hand-computed values. The right operand is stored
    // row-major [in, out], which is the neural-bunch model-format convention.
    constexpr std::array<float, 6u> aValues{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    constexpr std::array<float, 12u> bValues{
        1.0f, 0.0f, -1.0f, 2.0f,
        0.0f, 1.0f, 2.0f, -1.0f,
        3.0f, 1.0f, 0.0f, 1.0f};
    constexpr std::array<float, 8u> expectedValues{
        10.0f, 5.0f, 3.0f, 3.0f,
        22.0f, 11.0f, 6.0f, 9.0f};

    template<typename T_Type>
    void runConvention(auto& queue, auto const& device, bool viaLinear)
    {
        auto hostA = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 3u});
        auto hostB = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{3u, 4u});
        auto hostC = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 4u});

        for(std::uint32_t i = 0u; i < aValues.size(); ++i)
            hostA[alpaka::Vec{i / 3u, i % 3u}] = static_cast<T_Type>(aValues[i]);
        for(std::uint32_t i = 0u; i < bValues.size(); ++i)
            hostB[alpaka::Vec{i / 4u, i % 4u}] = static_cast<T_Type>(bValues[i]);

        auto devA = alpaka::onHost::allocLike(device, hostA);
        auto devB = alpaka::onHost::allocLike(device, hostB);
        auto devC = alpaka::onHost::allocLike(device, hostC);
        alpaka::onHost::memcpy(queue, devA, hostA);
        alpaka::onHost::memcpy(queue, devB, hostB);

        if(viaLinear)
            alpaka::nn::onHost::nn::linear<T_Type>(queue, devA, devB, devC);
        else
            alpaka::nn::onHost::gemm<T_Type>(queue, devA, devB, devC);

        alpaka::onHost::memcpy(queue, hostC, devC);
        alpaka::onHost::wait(queue);

        for(std::uint32_t i = 0u; i < expectedValues.size(); ++i)
        {
            INFO("viaLinear=" << viaLinear << " element=" << i);
            alpaka::nn::test::checkValue(
                hostC[alpaka::Vec{i / 4u, i % 4u}],
                static_cast<T_Type>(expectedValues[i]),
                1.0e-6,
                1.0e-6);
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("gemm and linear use row-major [in, out] weights", "[matrix][convention]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    runConvention<float>(queue, device, false);
    runConvention<float>(queue, device, true);
    if constexpr(alpaka::nn::test::supportsFp64(device))
    {
        runConvention<double>(queue, device, false);
        runConvention<double>(queue, device, true);
    }
}
