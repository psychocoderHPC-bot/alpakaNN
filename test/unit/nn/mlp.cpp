/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    template<typename T_Type>
    void runMlpCase(auto& queue, auto exec, auto const& device)
    {
        auto input = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u});
        auto gate = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{4u, 6u});
        auto up = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{4u, 6u});
        auto down = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{6u, 4u});
        auto output = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u});
        for(auto idx : alpaka::IdxRange{input.getExtents()})
            input[idx] = static_cast<T_Type>(idx[1] + 1u);
        for(auto idx : alpaka::IdxRange{gate.getExtents()})
            gate[idx] = static_cast<T_Type>(0.05) * static_cast<T_Type>(idx[0] + idx[1] + 1u);
        for(auto idx : alpaka::IdxRange{up.getExtents()})
            up[idx] = static_cast<T_Type>(0.04) * static_cast<T_Type>(idx[0] + idx[1] + 1u);
        for(auto idx : alpaka::IdxRange{down.getExtents()})
            down[idx] = static_cast<T_Type>(0.03) * static_cast<T_Type>(idx[0] + idx[1] + 1u);

        auto devInput = alpaka::onHost::allocLike(device, input);
        auto devGate = alpaka::onHost::allocLike(device, gate);
        auto devUp = alpaka::onHost::allocLike(device, up);
        auto devDown = alpaka::onHost::allocLike(device, down);
        auto devOut = alpaka::onHost::allocLike(device, output);
        auto devOutWorkspace = alpaka::onHost::allocLike(device, output);
        auto gateWorkspace = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{1u, 6u});
        auto upWorkspace = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{1u, 6u});
        auto hiddenWorkspace = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{1u, 6u});
        alpaka::onHost::memcpy(queue, devInput, input);
        alpaka::onHost::memcpy(queue, devGate, gate);
        alpaka::onHost::memcpy(queue, devUp, up);
        alpaka::onHost::memcpy(queue, devDown, down);
        alpaka::nn::onHost::nn::mlp<T_Type>(queue, exec, devInput, devGate, devUp, devDown, devOut);
        auto legacyOutput = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u});
        alpaka::onHost::memcpy(queue, legacyOutput, devOut);
        alpaka::nn::onHost::nn::mlp<T_Type>(
            queue,
            exec,
            devInput,
            devGate,
            devUp,
            devDown,
            gateWorkspace,
            upWorkspace,
            hiddenWorkspace,
            devOutWorkspace);
        // Queue a dependent read immediately; synchronization is the caller's responsibility.
        alpaka::onHost::memcpy(queue, output, devOutWorkspace);
        alpaka::onHost::wait(queue);
        auto invalidWorkspace = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{1u, 5u});
        CHECK_THROWS(
            alpaka::nn::onHost::nn::mlp<T_Type>(
                queue,
                exec,
                devInput,
                devGate,
                devUp,
                devDown,
                invalidWorkspace,
                upWorkspace,
                hiddenWorkspace,
                devOutWorkspace));

        T_Type hidden[6]{};
        for(uint32_t col = 0u; col < 6u; ++col)
        {
            T_Type a{};
            T_Type b{};
            for(uint32_t k = 0u; k < 4u; ++k)
            {
                a += input[alpaka::Vec{0u, k}] * gate[alpaka::Vec{k, col}];
                b += input[alpaka::Vec{0u, k}] * up[alpaka::Vec{k, col}];
            }
            hidden[col] = a / (T_Type{1} + std::exp(-a)) * b;
        }
        for(uint32_t col = 0u; col < 4u; ++col)
        {
            T_Type expected{};
            for(uint32_t k = 0u; k < 6u; ++k)
                expected += hidden[k] * down[alpaka::Vec{k, col}];
            alpaka::nn::test::checkValue(output[alpaka::Vec{0u, col}], expected);
            alpaka::nn::test::checkValue(output[alpaka::Vec{0u, col}], legacyOutput[alpaka::Vec{0u, col}]);
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("mlp matches reference", "[nn][mlp]", TestApis)
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

    runMlpCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runMlpCase<double>(queue, exec, device);
}
