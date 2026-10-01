/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
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

    template<typename T_Type>
    auto referenceSoftmax(auto const& input)
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
                    for(uint32_t key = 0u; key < extents[3]; ++key)
                        maxValue = std::max(maxValue, input[alpaka::Vec{batch, head, query, key}]);

                    T_Type sum{};
                    for(uint32_t key = 0u; key < extents[3]; ++key)
                    {
                        auto const value = std::exp(input[alpaka::Vec{batch, head, query, key}] - maxValue);
                        output[alpaka::Vec{batch, head, query, key}] = value;
                        sum += value;
                    }
                    for(uint32_t key = 0u; key < extents[3]; ++key)
                        output[alpaka::Vec{batch, head, query, key}] /= sum;
                }
            }
        }
        return output;
    }
} // namespace

namespace
{
    template<typename T_Type>
    void runSoftmaxStabilityCase(auto& queue, auto exec, auto const& device)
    {
        auto input = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 2u, 3u});
        auto output = alpaka::onHost::allocHost<T_Type>(input.getExtents());
        input[alpaka::Vec{0u, 0u, 0u, 0u}] = static_cast<T_Type>(1000);
        input[alpaka::Vec{0u, 0u, 0u, 1u}] = static_cast<T_Type>(1001);
        input[alpaka::Vec{0u, 0u, 0u, 2u}] = static_cast<T_Type>(999);
        input[alpaka::Vec{0u, 0u, 1u, 0u}] = static_cast<T_Type>(-1);
        input[alpaka::Vec{0u, 0u, 1u, 1u}] = T_Type{};
        input[alpaka::Vec{0u, 0u, 1u, 2u}] = T_Type{1};

        auto devIn = alpaka::onHost::allocLike(device, input);
        auto devOut = alpaka::onHost::allocLike(device, output);
        alpaka::onHost::memcpy(queue, devIn, input);
        alpaka::nn::onHost::nn::causalSoftmax<T_Type>(queue, exec, devIn, devOut, 3u, 2u, 3u);
        alpaka::onHost::memcpy(queue, output, devOut);
        alpaka::onHost::wait(queue);

        CHECK(output[alpaka::Vec{0u, 0u, 0u, 1u}] == T_Type{});
        CHECK(output[alpaka::Vec{0u, 0u, 0u, 2u}] == T_Type{});
        auto sum0 = output[alpaka::Vec{0u, 0u, 0u, 0u}];
        auto sum1 = output[alpaka::Vec{0u, 0u, 1u, 0u}] + output[alpaka::Vec{0u, 0u, 1u, 1u}]
                    + output[alpaka::Vec{0u, 0u, 1u, 2u}];
        alpaka::nn::test::checkValue(sum0, T_Type{1});
        alpaka::nn::test::checkValue(sum1, T_Type{1});
    }

    template<typename T_Type>
    void runCausalSoftmaxPrefillCase(auto& queue, auto exec, auto const& device)
    {
        auto input = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        auto output = alpaka::onHost::allocHost<T_Type>(input.getExtents());
        for(auto idx : alpaka::IdxRange{input.getExtents()})
            input[idx] = static_cast<T_Type>(idx[0] * 97u + idx[1] * 23u + idx[2] * 11u + idx[3] * 5u)
                         * static_cast<T_Type>(0.03125);
        for(auto idx : alpaka::IdxRange{output.getExtents()})
            output[idx] = static_cast<T_Type>(-123);

        auto expected = referenceCausalSoftmax<T_Type>(input);
        auto devIn = alpaka::onHost::allocLike(device, input);
        auto devOut = alpaka::onHost::allocLike(device, output);
        alpaka::onHost::memcpy(queue, devIn, input);
        alpaka::onHost::memcpy(queue, devOut, output);

        alpaka::nn::onHost::nn::causalSoftmax<T_Type>(queue, exec, devIn, devOut, 3u, 2u, 3u);
        alpaka::onHost::memcpy(queue, output, devOut);
        alpaka::onHost::wait(queue);

        for(auto idx : alpaka::IdxRange{output.getExtents()})
            alpaka::nn::test::checkValue(output[idx], expected[idx], 1.0e-5f, 1.0e-5f);
    }

    template<typename T_Type>
    void runSoftmaxDecodeCase(auto& queue, auto exec, auto const& device)
    {
        auto input = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 1u, 6u});
        auto output = alpaka::onHost::allocHost<T_Type>(input.getExtents());
        for(auto idx : alpaka::IdxRange{input.getExtents()})
            input[idx] = static_cast<T_Type>(idx[0] * 97u + idx[1] * 29u + idx[2] * 13u + idx[3] * 3u)
                             * static_cast<T_Type>(0.0625)
                         - static_cast<T_Type>(0.5);
        for(auto idx : alpaka::IdxRange{output.getExtents()})
            output[idx] = static_cast<T_Type>(-321);

        auto expected = referenceSoftmax<T_Type>(input);
        auto devIn = alpaka::onHost::allocLike(device, input);
        auto devOut = alpaka::onHost::allocLike(device, output);
        alpaka::onHost::memcpy(queue, devIn, input);
        alpaka::onHost::memcpy(queue, devOut, output);

        alpaka::nn::onHost::nn::softmax<T_Type>(queue, exec, devIn, devOut, 3u);
        alpaka::onHost::memcpy(queue, output, devOut);
        alpaka::onHost::wait(queue);

        for(auto idx : alpaka::IdxRange{output.getExtents()})
            alpaka::nn::test::checkValue(output[idx], expected[idx], 1.0e-5f, 1.0e-5f);
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("softmax and causalSoftmax are stable", "[nn][softmax]", TestApis)
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

    runSoftmaxStabilityCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runSoftmaxStabilityCase<double>(queue, exec, device);
}

TEMPLATE_LIST_TEST_CASE("causalSoftmax matches decoder prefill reference shape", "[nn][softmax][decoder]", TestApis)
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

    runCausalSoftmaxPrefillCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runCausalSoftmaxPrefillCase<double>(queue, exec, device);
}

TEMPLATE_LIST_TEST_CASE("softmax matches decoder decode reference shape", "[nn][softmax][decoder]", TestApis)
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

    runSoftmaxDecodeCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runSoftmaxDecodeCase<double>(queue, exec, device);
}
