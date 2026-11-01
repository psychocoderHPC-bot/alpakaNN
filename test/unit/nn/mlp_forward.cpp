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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    template<typename T_Type>
    T_Type activate(alpaka::nn::onHost::nn::Activation activation, T_Type value)
    {
        switch(activation)
        {
        case alpaka::nn::onHost::nn::Activation::none:
            return value;
        case alpaka::nn::onHost::nn::Activation::relu:
            return value > T_Type{} ? value : T_Type{};
        case alpaka::nn::onHost::nn::Activation::silu:
            return value / (T_Type{1} + std::exp(-value));
        case alpaka::nn::onHost::nn::Activation::gelu:
            return T_Type{0.5} * value * (T_Type{1} + std::erf(value * static_cast<T_Type>(0.70710678118654752440)));
        }
        return value;
    }

    template<typename T_Type>
    void fillDeterministic(auto& view, T_Type scale, T_Type shift)
    {
        auto index = 0u;
        for(auto idx : alpaka::IdxRange{view.getExtents()})
        {
            view[idx] = scale * static_cast<T_Type>(index + 1u) + shift;
            ++index;
        }
    }

    template<typename T_Device>
    auto copyToDevice(auto& queue, T_Device const& device, auto const& host)
    {
        auto deviceView = alpaka::onHost::allocLike(device, host);
        alpaka::onHost::memcpy(queue, deviceView, host);
        return deviceView;
    }

    template<typename T_Type>
    void runSingleLayer(auto& queue, auto exec, auto const& device)
    {
        using namespace alpaka::nn::onHost;
        std::uint32_t const batch = 2u;
        std::uint32_t const inDim = 3u;
        std::uint32_t const outDim = 4u;

        auto hostInput = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{batch, inDim});
        auto hostWeight = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{inDim, outDim});
        auto hostBias = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{outDim});
        auto hostOutput = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{batch, outDim});
        fillDeterministic<T_Type>(hostInput, static_cast<T_Type>(0.3), static_cast<T_Type>(-0.2));
        fillDeterministic<T_Type>(hostWeight, static_cast<T_Type>(0.11), static_cast<T_Type>(-0.4));
        fillDeterministic<T_Type>(hostBias, static_cast<T_Type>(0.07), static_cast<T_Type>(0.01));

        auto devInput = copyToDevice(queue, device, hostInput);
        auto devWeight = copyToDevice(queue, device, hostWeight);
        auto devBias = copyToDevice(queue, device, hostBias);
        auto devOutput = copyToDevice(queue, device, hostOutput);

        using Weight = alpaka::nn::onHost::nn::WeightView<decltype(devWeight)>;
        using Bias = alpaka::nn::onHost::nn::BiasView<decltype(devBias)>;
        std::array<Weight, 1u> weights{Weight{devWeight}};
        std::array<Bias, 1u> biases{Bias{devBias}};

        alpaka::nn::onHost::nn::mlpForward<T_Type>(
            queue,
            exec,
            devInput,
            weights,
            biases,
            alpaka::nn::onHost::nn::Activation::silu,
            devOutput);

        alpaka::onHost::memcpy(queue, hostOutput, devOutput);
        alpaka::onHost::wait(queue);

        for(std::uint32_t row = 0u; row < batch; ++row)
        {
            for(std::uint32_t col = 0u; col < outDim; ++col)
            {
                T_Type expected{};
                for(std::uint32_t inner = 0u; inner < inDim; ++inner)
                    expected += hostInput[alpaka::Vec{row, inner}] * hostWeight[alpaka::Vec{inner, col}];
                expected += hostBias[alpaka::Vec{col}];
                expected = activate(alpaka::nn::onHost::nn::Activation::silu, expected);
                alpaka::nn::test::checkValue(hostOutput[alpaka::Vec{row, col}], expected);
            }
        }
    }

    template<typename T_Type>
    void runThreeLayersSiluBias(auto& queue, auto exec, auto const& device)
    {
        using namespace alpaka::nn::onHost;
        std::uint32_t const batch = 2u;
        std::array<std::uint32_t, 4u> const dims{3u, 5u, 4u, 2u};

        auto hostInput = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{batch, dims[0]});
        fillDeterministic<T_Type>(hostInput, static_cast<T_Type>(0.2), static_cast<T_Type>(-0.3));

        auto hW0 = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{dims[0], dims[1]});
        auto hW1 = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{dims[1], dims[2]});
        auto hW2 = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{dims[2], dims[3]});
        auto hB0 = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{dims[1]});
        auto hB1 = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{dims[2]});
        auto hB2 = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{dims[3]});
        fillDeterministic<T_Type>(hW0, static_cast<T_Type>(0.09), static_cast<T_Type>(-0.2));
        fillDeterministic<T_Type>(hW1, static_cast<T_Type>(0.07), static_cast<T_Type>(-0.1));
        fillDeterministic<T_Type>(hW2, static_cast<T_Type>(0.05), static_cast<T_Type>(-0.05));
        fillDeterministic<T_Type>(hB0, static_cast<T_Type>(0.03), static_cast<T_Type>(0.01));
        fillDeterministic<T_Type>(hB1, static_cast<T_Type>(0.02), static_cast<T_Type>(-0.01));
        fillDeterministic<T_Type>(hB2, static_cast<T_Type>(0.01), static_cast<T_Type>(0.02));

        auto hostOutput = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{batch, dims[3]});

        auto devInput = copyToDevice(queue, device, hostInput);
        auto dW0 = copyToDevice(queue, device, hW0);
        auto dW1 = copyToDevice(queue, device, hW1);
        auto dW2 = copyToDevice(queue, device, hW2);
        auto dB0 = copyToDevice(queue, device, hB0);
        auto dB1 = copyToDevice(queue, device, hB1);
        auto dB2 = copyToDevice(queue, device, hB2);
        auto devOutput = copyToDevice(queue, device, hostOutput);

        using Weight = alpaka::nn::onHost::nn::WeightView<decltype(dW0)>;
        using Bias = alpaka::nn::onHost::nn::BiasView<decltype(dB0)>;
        std::array<Weight, 3u> weights{Weight{dW0}, Weight{dW1}, Weight{dW2}};
        std::array<Bias, 3u> biases{Bias{dB0}, Bias{dB1}, Bias{dB2}};

        using Activation = alpaka::nn::onHost::nn::Activation;
        alpaka::nn::onHost::nn::mlpForward<
            T_Type>(queue, exec, devInput, weights, biases, Activation::silu, Activation::none, devOutput);

        alpaka::onHost::memcpy(queue, hostOutput, devOutput);
        alpaka::onHost::wait(queue);

        for(std::uint32_t row = 0u; row < batch; ++row)
        {
            std::array<std::array<T_Type, 5u>, 2u> layer0{};
            std::array<std::array<T_Type, 4u>, 2u> layer1{};
            for(std::uint32_t col = 0u; col < dims[1]; ++col)
            {
                T_Type value = hB0[alpaka::Vec{col}];
                for(std::uint32_t inner = 0u; inner < dims[0]; ++inner)
                    value += hostInput[alpaka::Vec{row, inner}] * hW0[alpaka::Vec{inner, col}];
                layer0[row][col] = activate(Activation::silu, value);
            }
            for(std::uint32_t col = 0u; col < dims[2]; ++col)
            {
                T_Type value = hB1[alpaka::Vec{col}];
                for(std::uint32_t inner = 0u; inner < dims[1]; ++inner)
                    value += layer0[row][inner] * hW1[alpaka::Vec{inner, col}];
                layer1[row][col] = activate(Activation::silu, value);
            }
            for(std::uint32_t col = 0u; col < dims[3]; ++col)
            {
                T_Type value = hB2[alpaka::Vec{col}];
                for(std::uint32_t inner = 0u; inner < dims[2]; ++inner)
                    value += layer1[row][inner] * hW2[alpaka::Vec{inner, col}];
                alpaka::nn::test::checkValue(hostOutput[alpaka::Vec{row, col}], value);
            }
        }
    }

    template<typename T_Type>
    void runDecodeLatent(auto& queue, auto exec, auto const& device)
    {
        using namespace alpaka::nn::onHost;
        std::uint32_t const rank = 3u;
        std::uint32_t const outDim = 5u;

        auto hostLatent = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, rank});
        auto hostWeight = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{rank, outDim});
        auto hostOutput = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, outDim});
        fillDeterministic<T_Type>(hostLatent, static_cast<T_Type>(0.4), static_cast<T_Type>(-0.6));
        fillDeterministic<T_Type>(hostWeight, static_cast<T_Type>(0.08), static_cast<T_Type>(-0.3));

        auto devLatent = copyToDevice(queue, device, hostLatent);
        auto devWeight = copyToDevice(queue, device, hostWeight);
        auto devOutput = copyToDevice(queue, device, hostOutput);

        alpaka::nn::onHost::nn::decodeLatent<T_Type>(queue, exec, devLatent, devWeight, devOutput);

        alpaka::onHost::memcpy(queue, hostOutput, devOutput);
        alpaka::onHost::wait(queue);

        for(std::uint32_t col = 0u; col < outDim; ++col)
        {
            T_Type expected{};
            for(std::uint32_t inner = 0u; inner < rank; ++inner)
                expected += hostLatent[alpaka::Vec{0u, inner}] * hostWeight[alpaka::Vec{inner, col}];
            alpaka::nn::test::checkValue(hostOutput[alpaka::Vec{0u, col}], expected);
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("mlpForward single layer matches reference", "[nn][mlp][forward]", TestApis)
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

    runSingleLayer<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runSingleLayer<double>(queue, exec, device);
}

TEMPLATE_LIST_TEST_CASE("mlpForward three layers silu+bias matches reference", "[nn][mlp][forward]", TestApis)
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

    runThreeLayersSiluBias<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runThreeLayersSiluBias<double>(queue, exec, device);
}

TEMPLATE_LIST_TEST_CASE("decodeLatent matches reference", "[nn][mlp][decoder]", TestApis)
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

    runDecodeLatent<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runDecodeLatent<double>(queue, exec, device);
}
