/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    template<typename T_Type>
    auto referenceCausalSoftmax(auto const& input)
    {
        auto output = alpaka::onHost::allocHost<T_Type>(input.getExtents());
        auto const extents = input.getExtents();
        for(uint32_t batch = 0u; batch < extents[0]; ++batch)
        {
            for(uint32_t head = 0u; head < extents[1]; ++head)
            {
                for(uint32_t query = 0u; query < extents[2]; ++query)
                {
                    T_Type maxValue = -std::numeric_limits<T_Type>::infinity();
                    for(uint32_t key = 0u; key <= query; ++key)
                        maxValue = std::max(maxValue, input[alpaka::Vec{batch, head, query, key}]);

                    T_Type sum{};
                    for(uint32_t key = 0u; key < extents[3]; ++key)
                    {
                        if(key > query)
                        {
                            output[alpaka::Vec{batch, head, query, key}] = T_Type{};
                            continue;
                        }
                        auto const value = std::exp(input[alpaka::Vec{batch, head, query, key}] - maxValue);
                        output[alpaka::Vec{batch, head, query, key}] = value;
                        sum += value;
                    }
                    for(uint32_t key = 0u; key <= query; ++key)
                        output[alpaka::Vec{batch, head, query, key}] /= sum;
                }
            }
        }
        return output;
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("softmax and causalSoftmax are stable", "[nn][softmax]", TestApis)
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

    auto input = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 2u, 3u});
    auto output = alpaka::onHost::allocHost<float>(input.getExtents());
    input[alpaka::Vec{0u, 0u, 0u, 0u}] = 1000.0f;
    input[alpaka::Vec{0u, 0u, 0u, 1u}] = 1001.0f;
    input[alpaka::Vec{0u, 0u, 0u, 2u}] = 999.0f;
    input[alpaka::Vec{0u, 0u, 1u, 0u}] = -1.0f;
    input[alpaka::Vec{0u, 0u, 1u, 1u}] = 0.0f;
    input[alpaka::Vec{0u, 0u, 1u, 2u}] = 1.0f;

    auto devIn = alpaka::onHost::allocLike(device, input);
    auto devOut = alpaka::onHost::allocLike(device, output);
    alpaka::onHost::memcpy(queue, devIn, input);
    alpaka::nn::onHost::nn::causalSoftmax<float>(queue, exec, devIn, devOut, 3u, 2u, 3u);
    alpaka::onHost::memcpy(queue, output, devOut);
    alpaka::onHost::wait(queue);

    CHECK(output[alpaka::Vec{0u, 0u, 0u, 1u}] == 0.0f);
    CHECK(output[alpaka::Vec{0u, 0u, 0u, 2u}] == 0.0f);
    auto sum0 = output[alpaka::Vec{0u, 0u, 0u, 0u}];
    auto sum1 = output[alpaka::Vec{0u, 0u, 1u, 0u}] + output[alpaka::Vec{0u, 0u, 1u, 1u}]
                + output[alpaka::Vec{0u, 0u, 1u, 2u}];
    alpaka::nn::test::checkValue(sum0, 1.0f);
    alpaka::nn::test::checkValue(sum1, 1.0f);
}

TEMPLATE_LIST_TEST_CASE("causalSoftmax matches decoder prefill reference shape", "[nn][softmax][decoder]", TestApis)
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

    auto input = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    auto output = alpaka::onHost::allocHost<float>(input.getExtents());
    for(auto idx : alpaka::IdxRange{input.getExtents()})
        input[idx] = static_cast<float>(idx[0] * 97u + idx[1] * 23u + idx[2] * 11u + idx[3] * 5u) * 0.03125f;
    for(auto idx : alpaka::IdxRange{output.getExtents()})
        output[idx] = -123.0f;

    auto expected = referenceCausalSoftmax<float>(input);
    auto devIn = alpaka::onHost::allocLike(device, input);
    auto devOut = alpaka::onHost::allocLike(device, output);
    alpaka::onHost::memcpy(queue, devIn, input);
    alpaka::onHost::memcpy(queue, devOut, output);

    alpaka::nn::onHost::nn::causalSoftmax<float>(queue, exec, devIn, devOut, 3u, 2u, 3u);
    alpaka::onHost::memcpy(queue, output, devOut);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{output.getExtents()})
        alpaka::nn::test::checkValue(output[idx], expected[idx], 1.0e-5f, 1.0e-5f);
}
