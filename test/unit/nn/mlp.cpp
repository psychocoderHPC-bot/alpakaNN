/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using TestApis = alpaka::nn::test::TestApis;

TEMPLATE_LIST_TEST_CASE("mlp matches reference", "[nn][mlp]", TestApis)
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

    auto input = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u});
    auto gate = alpaka::onHost::allocHost<float>(alpaka::Vec{4u, 6u});
    auto up = alpaka::onHost::allocHost<float>(alpaka::Vec{4u, 6u});
    auto down = alpaka::onHost::allocHost<float>(alpaka::Vec{6u, 4u});
    auto output = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u});
    for(auto idx : alpaka::IdxRange{input.getExtents()})
        input[idx] = static_cast<float>(idx[1] + 1u);
    for(auto idx : alpaka::IdxRange{gate.getExtents()})
        gate[idx] = 0.05f * static_cast<float>(idx[0] + idx[1] + 1u);
    for(auto idx : alpaka::IdxRange{up.getExtents()})
        up[idx] = 0.04f * static_cast<float>(idx[0] + idx[1] + 1u);
    for(auto idx : alpaka::IdxRange{down.getExtents()})
        down[idx] = 0.03f * static_cast<float>(idx[0] + idx[1] + 1u);

    auto devInput = alpaka::onHost::allocLike(device, input);
    auto devGate = alpaka::onHost::allocLike(device, gate);
    auto devUp = alpaka::onHost::allocLike(device, up);
    auto devDown = alpaka::onHost::allocLike(device, down);
    auto devOut = alpaka::onHost::allocLike(device, output);
    alpaka::onHost::memcpy(queue, devInput, input);
    alpaka::onHost::memcpy(queue, devGate, gate);
    alpaka::onHost::memcpy(queue, devUp, up);
    alpaka::onHost::memcpy(queue, devDown, down);
    alpaka::nn::onHost::nn::mlp<float>(queue, exec, devInput, devGate, devUp, devDown, devOut);
    alpaka::onHost::memcpy(queue, output, devOut);
    alpaka::onHost::wait(queue);

    float hidden[6]{};
    for(uint32_t col = 0u; col < 6u; ++col)
    {
        float a{};
        float b{};
        for(uint32_t k = 0u; k < 4u; ++k)
        {
            a += input[alpaka::Vec{0u, k}] * gate[alpaka::Vec{k, col}];
            b += input[alpaka::Vec{0u, k}] * up[alpaka::Vec{k, col}];
        }
        hidden[col] = a / (1.0f + std::exp(-a)) * b;
    }
    for(uint32_t col = 0u; col < 4u; ++col)
    {
        float expected{};
        for(uint32_t k = 0u; k < 6u; ++k)
            expected += hidden[k] * down[alpaka::Vec{k, col}];
        alpaka::nn::test::checkValue(output[alpaka::Vec{0u, col}], expected);
    }
}
