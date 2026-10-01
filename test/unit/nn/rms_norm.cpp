/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    template<typename T_Type>
    void runRmsNormCase(auto& queue, auto exec, auto const& device)
    {
        auto input = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 4u});
        auto weight = alpaka::onHost::allocHost<T_Type>(4u);
        auto output = alpaka::onHost::allocHost<T_Type>(input.getExtents());
        for(auto idx : alpaka::IdxRange{input.getExtents()})
            input[idx] = static_cast<T_Type>(idx[0] * 4u + idx[1] + 1u);
        for(uint32_t i = 0u; i < 4u; ++i)
            weight[alpaka::Vec{i}] = static_cast<T_Type>(0.5) + static_cast<T_Type>(i);

        auto devIn = alpaka::onHost::allocLike(device, input);
        auto devWeight = alpaka::onHost::allocLike(device, weight);
        auto devOut = alpaka::onHost::allocLike(device, output);
        alpaka::onHost::memcpy(queue, devIn, input);
        alpaka::onHost::memcpy(queue, devWeight, weight);
        alpaka::nn::onHost::nn::rmsNorm<T_Type>(queue, exec, devIn, devWeight, devOut, static_cast<T_Type>(1.0e-5));
        alpaka::onHost::memcpy(queue, output, devOut);
        alpaka::onHost::wait(queue);

        for(uint32_t row = 0u; row < 2u; ++row)
        {
            T_Type sumSquares{};
            for(uint32_t col = 0u; col < 4u; ++col)
                sumSquares += input[alpaka::Vec{row, col}] * input[alpaka::Vec{row, col}];
            auto const invRms = T_Type{1} / std::sqrt(sumSquares / T_Type{4} + static_cast<T_Type>(1.0e-5));
            for(uint32_t col = 0u; col < 4u; ++col)
                alpaka::nn::test::checkValue(
                    output[alpaka::Vec{row, col}],
                    input[alpaka::Vec{row, col}] * invRms * weight[alpaka::Vec{col}]);
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("rmsNorm matches reference", "[nn][rmsnorm]", TestApis)
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

    runRmsNormCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runRmsNormCase<double>(queue, exec, device);
}
