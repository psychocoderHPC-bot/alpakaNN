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
    void runCase(auto& queue, auto exec, auto const& device, std::uint32_t s, std::uint32_t n)
    {
        auto hostInputs = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{s, n});
        auto hostWeights = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{s});
        auto hostOutput = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{n});
        auto hostOutput2 = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{n});
        fill<T_Type>(hostInputs, static_cast<T_Type>(0.17), static_cast<T_Type>(-0.9));
        fill<T_Type>(hostWeights, static_cast<T_Type>(0.23), static_cast<T_Type>(-0.4));
        for(std::uint32_t i = 0u; i < n; ++i)
        {
            hostOutput[alpaka::Vec{i}] = static_cast<T_Type>(0.25) * static_cast<T_Type>(i + 1u);
            hostOutput2[alpaka::Vec{i}] = hostOutput[alpaka::Vec{i}];
        }

        auto devInputs = alpaka::onHost::allocLike(device, hostInputs);
        auto devWeights = alpaka::onHost::allocLike(device, hostWeights);
        auto devOutput = alpaka::onHost::allocLike(device, hostOutput);
        auto devOutput2 = alpaka::onHost::allocLike(device, hostOutput2);
        alpaka::onHost::memcpy(queue, devInputs, hostInputs);
        alpaka::onHost::memcpy(queue, devWeights, hostWeights);
        alpaka::onHost::memcpy(queue, devOutput, hostOutput);
        alpaka::onHost::memcpy(queue, devOutput2, hostOutput2);

        alpaka::nn::onHost::ops::weightedAdd<T_Type>(queue, exec, devInputs, devWeights, devOutput);
        alpaka::nn::onHost::ops::weightedAdd<T_Type>(queue, exec, devInputs, devWeights, devOutput2);

        alpaka::onHost::memcpy(queue, hostOutput, devOutput);
        alpaka::onHost::memcpy(queue, hostOutput2, devOutput2);
        alpaka::onHost::wait(queue);

        for(std::uint32_t i = 0u; i < n; ++i)
        {
            auto const idx = alpaka::Vec{i};
            CHECK(hostOutput[idx] == hostOutput2[idx]);
            T_Type expected = hostOutput2[idx];
            // Recompute from the original accumulator value to keep the reference independent.
            expected = static_cast<T_Type>(0.25) * static_cast<T_Type>(i + 1u);
            for(std::uint32_t row = 0u; row < s; ++row)
                expected += hostWeights[alpaka::Vec{row}] * hostInputs[alpaka::Vec{row, i}];
            alpaka::nn::test::checkValue(
                hostOutput[idx],
                expected,
                std::is_same_v<T_Type, float> ? 2.0e-5 : 1.0e-12,
                1.0e-6);
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("weightedAdd accumulates deterministic weighted rows", "[ops][weightedAdd]", TestApis)
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

    runCase<float>(queue, exec, device, 1u, 8u);
    runCase<float>(queue, exec, device, 5u, 11u);
    if constexpr(alpaka::nn::test::supportsFp64(device))
    {
        runCase<double>(queue, exec, device, 1u, 8u);
        runCase<double>(queue, exec, device, 5u, 11u);
    }
}
