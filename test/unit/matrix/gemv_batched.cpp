/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "test.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <type_traits>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    template<typename T_Type>
    void fill(auto& view, T_Type scale, T_Type shift)
    {
        auto index = 0u;
        for(auto idx : alpaka::IdxRange{view.getExtents()})
        {
            view[idx] = scale * static_cast<T_Type>(index + 1u) + shift;
            ++index;
        }
    }

    template<typename T_Type>
    void runCase(auto& queue, auto const& device, std::uint32_t s, std::uint32_t k, std::uint32_t n)
    {
        auto hostCoefficients = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{s, k});
        auto hostMatrix = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{k, n});
        auto hostOutput = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{s, n});
        auto hostReference = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{s, n});
        auto hostOutput2 = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{s, n});
        fill<T_Type>(hostCoefficients, static_cast<T_Type>(0.13), static_cast<T_Type>(-0.7));
        fill<T_Type>(hostMatrix, static_cast<T_Type>(0.07), static_cast<T_Type>(-0.3));

        auto devCoefficients = alpaka::onHost::allocLike(device, hostCoefficients);
        auto devMatrix = alpaka::onHost::allocLike(device, hostMatrix);
        auto devOutput = alpaka::onHost::allocLike(device, hostOutput);
        auto devReference = alpaka::onHost::allocLike(device, hostReference);
        auto devOutput2 = alpaka::onHost::allocLike(device, hostOutput2);
        alpaka::onHost::memcpy(queue, devCoefficients, hostCoefficients);
        alpaka::onHost::memcpy(queue, devMatrix, hostMatrix);

        alpaka::nn::onHost::gemvBatched<T_Type>(queue, devCoefficients, devMatrix, devOutput);
        alpaka::nn::onHost::gemm<T_Type>(queue, devCoefficients, devMatrix, devReference);
        alpaka::nn::onHost::gemvBatched<T_Type>(queue, devCoefficients, devMatrix, devOutput2);

        alpaka::onHost::memcpy(queue, hostOutput, devOutput);
        alpaka::onHost::memcpy(queue, hostReference, devReference);
        alpaka::onHost::memcpy(queue, hostOutput2, devOutput2);
        alpaka::onHost::wait(queue);

        for(std::uint32_t row = 0u; row < s; ++row)
        {
            for(std::uint32_t col = 0u; col < n; ++col)
            {
                auto const idx = alpaka::Vec{row, col};
                // gemvBatched is the same single GEMM call, so it agrees bit-exactly.
                CHECK(hostOutput[idx] == hostReference[idx]);
                CHECK(hostOutput[idx] == hostOutput2[idx]);

                T_Type expected{};
                for(std::uint32_t inner = 0u; inner < k; ++inner)
                    expected += hostCoefficients[alpaka::Vec{row, inner}] * hostMatrix[alpaka::Vec{inner, col}];
                alpaka::nn::test::checkValue(
                    hostOutput[idx],
                    expected,
                    std::is_same_v<T_Type, float> ? 2.0e-5 : 1.0e-12,
                    1.0e-6);
            }
        }
    }

    template<typename T_Type>
    void runCrossBatch(auto& queue, auto const& device)
    {
        constexpr std::uint32_t k = 5u;
        constexpr std::uint32_t n = 7u;
        auto hostOne = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, k});
        auto hostMany = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{8u, k});
        auto hostMatrix = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{k, n});
        auto hostOutOne = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, n});
        auto hostOutMany = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{8u, n});
        fill<T_Type>(hostOne, static_cast<T_Type>(0.11), static_cast<T_Type>(-0.2));
        fill<T_Type>(hostMatrix, static_cast<T_Type>(0.09), static_cast<T_Type>(-0.4));
        for(std::uint32_t row = 0u; row < 8u; ++row)
            for(std::uint32_t col = 0u; col < k; ++col)
                hostMany[alpaka::Vec{row, col}] = hostOne[alpaka::Vec{0u, col}];

        auto devOne = alpaka::onHost::allocLike(device, hostOne);
        auto devMany = alpaka::onHost::allocLike(device, hostMany);
        auto devMatrix = alpaka::onHost::allocLike(device, hostMatrix);
        auto devOutOne = alpaka::onHost::allocLike(device, hostOutOne);
        auto devOutMany = alpaka::onHost::allocLike(device, hostOutMany);
        alpaka::onHost::memcpy(queue, devOne, hostOne);
        alpaka::onHost::memcpy(queue, devMany, hostMany);
        alpaka::onHost::memcpy(queue, devMatrix, hostMatrix);
        alpaka::nn::onHost::gemvBatched<T_Type>(queue, devOne, devMatrix, devOutOne);
        alpaka::nn::onHost::gemvBatched<T_Type>(queue, devMany, devMatrix, devOutMany);
        alpaka::onHost::memcpy(queue, hostOutOne, devOutOne);
        alpaka::onHost::memcpy(queue, hostOutMany, devOutMany);
        alpaka::onHost::wait(queue);

        for(std::uint32_t col = 0u; col < n; ++col)
            alpaka::nn::test::checkValue(
                hostOutMany[alpaka::Vec{0u, col}],
                hostOutOne[alpaka::Vec{0u, col}],
                1.0e-5,
                1.0e-6);
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("gemvBatched matches the single GEMM path", "[matrix][gemv][batched]", TestApis)
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

    runCase<float>(queue, device, 1u, 3u, 4u);
    runCase<float>(queue, device, 8u, 5u, 7u);
    runCase<float>(queue, device, 64u, 6u, 9u);
    runCrossBatch<float>(queue, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
    {
        runCase<double>(queue, device, 1u, 3u, 4u);
        runCase<double>(queue, device, 8u, 5u, 7u);
        runCrossBatch<double>(queue, device);
    }
}
