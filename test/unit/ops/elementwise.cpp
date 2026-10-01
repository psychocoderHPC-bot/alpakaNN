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
#include <type_traits>
#include <vector>

using TestApis = alpaka::nn::test::TestApis;

template<typename T_Type>
void fillSequence(auto& view)
{
    T_Type value = static_cast<T_Type>(-3);
    for(auto idx : alpaka::IdxRange{view.getExtents()})
    {
        view[idx] = value;
        value += static_cast<T_Type>(0.5);
    }
}

namespace
{
    template<typename T_Type>
    void runElementwiseCase(auto& queue, auto exec, auto const& device)
    {
        auto hostA = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 4u});
        auto hostB = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 4u});
        auto hostOut = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{2u, 4u});
        fillSequence<T_Type>(hostA);
        fillSequence<T_Type>(hostB);
        for(auto idx : alpaka::IdxRange{hostB.getExtents()})
            hostB[idx] += T_Type{1};

        auto devA = alpaka::onHost::allocLike(device, hostA);
        auto devB = alpaka::onHost::allocLike(device, hostB);
        auto devOut = alpaka::onHost::allocLike(device, hostOut);

        alpaka::onHost::memcpy(queue, devA, hostA);
        alpaka::onHost::memcpy(queue, devB, hostB);

        // fill copy add sub mul div scale axpy
        alpaka::nn::onHost::ops::fill(queue, exec, devOut, T_Type{2});
        alpaka::nn::onHost::ops::copy(queue, exec, devA, devOut);
        alpaka::nn::onHost::ops::add<T_Type>(queue, exec, devA, devB, devOut);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
            alpaka::nn::test::checkValue(hostOut[idx], hostA[idx] + hostB[idx]);

        alpaka::nn::onHost::ops::sub<T_Type>(queue, exec, devB, devA, devOut);
        alpaka::nn::onHost::ops::mul<T_Type>(queue, exec, devA, devB, devOut);
        alpaka::nn::onHost::ops::div<T_Type>(queue, exec, devB, devA, devOut);
        alpaka::nn::onHost::ops::scale<T_Type>(queue, exec, devA, T_Type{3}, devOut);
        alpaka::nn::onHost::ops::axpy<T_Type>(queue, exec, T_Type{2}, devA, devB, devOut);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
            alpaka::nn::test::checkValue(hostOut[idx], T_Type{2} * hostA[idx] + hostB[idx]);

        // bias add and cast
        auto hostBias = alpaka::onHost::allocHost<T_Type>(4u);
        for(uint32_t i = 0u; i < 4u; ++i)
            hostBias[alpaka::Vec{i}] = static_cast<T_Type>(i);
        auto devBias = alpaka::onHost::allocLike(device, hostBias);
        alpaka::onHost::memcpy(queue, devBias, hostBias);

        alpaka::nn::onHost::ops::biasAdd(queue, exec, devA, devBias, devOut);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
            alpaka::nn::test::checkValue(hostOut[idx], hostA[idx] + hostBias[alpaka::Vec{idx[1]}]);

        auto hostInt = alpaka::onHost::allocHost<int>(hostOut.getExtents());
        auto devInt = alpaka::onHost::allocLike(device, hostInt);
        alpaka::nn::onHost::ops::cast<int>(queue, exec, devA, devInt);
        alpaka::onHost::memcpy(queue, hostInt, devInt);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostInt.getExtents()})
            CHECK(hostInt[idx] == static_cast<int>(hostA[idx]));

        // transcendentals and activations on padded views
        std::vector<T_Type> paddedStorage(24u, T_Type{});
        std::vector<T_Type> paddedOutStorage(24u, T_Type{});
        auto paddedIn = alpaka::nn::onHost::view::makePaddedView<T_Type>(
            paddedStorage.data(),
            std::array<std::size_t, 2u>{2u, 4u},
            std::array<std::size_t, 2u>{8u, 1u});
        auto paddedOut = alpaka::nn::onHost::view::makePaddedView<T_Type>(
            paddedOutStorage.data(),
            std::array<std::size_t, 2u>{2u, 4u},
            std::array<std::size_t, 2u>{8u, 1u});
        fillSequence<T_Type>(paddedIn);

        auto devIn = alpaka::onHost::allocLike(device, paddedIn);
        auto devOutLocal = alpaka::onHost::allocLike(device, paddedOut);
        alpaka::onHost::memcpy(queue, devIn, paddedIn);

        alpaka::nn::onHost::ops::exp<T_Type>(queue, exec, devIn, devOutLocal);
        alpaka::nn::onHost::ops::sqrt<T_Type>(queue, exec, devOutLocal, devOutLocal);
        alpaka::nn::onHost::ops::rsqrt<T_Type>(queue, exec, devOutLocal, devOutLocal);
        alpaka::nn::onHost::ops::relu<T_Type>(queue, exec, devIn, devOutLocal);
        alpaka::nn::onHost::ops::sigmoid<T_Type>(queue, exec, devIn, devOutLocal);
        alpaka::nn::onHost::ops::silu<T_Type>(queue, exec, devIn, devOutLocal);
        alpaka::nn::onHost::ops::gelu<T_Type>(queue, exec, devIn, devOutLocal);
        alpaka::nn::onHost::ops::swiglu<T_Type>(queue, exec, devIn, devIn, devOutLocal);
        alpaka::onHost::memcpy(queue, paddedOut, devOutLocal);
        alpaka::onHost::wait(queue);

        // Keep the tight double tolerance; float gets a single-precision budget.
        constexpr double epsilon = std::is_same_v<T_Type, float> ? 1.0e-5 : 1.0e-9;
        for(auto idx : alpaka::IdxRange{paddedOut.getExtents()})
        {
            auto const x = paddedIn[idx];
            auto const sigmoid = T_Type{1} / (T_Type{1} + std::exp(-x));
            auto const expected = x * sigmoid * x;
            alpaka::nn::test::checkValue(paddedOut[idx], expected, epsilon, epsilon);
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("elementwise kernels", "[ops][elementwise]", TestApis)
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

    runElementwiseCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runElementwiseCase<double>(queue, exec, device);
}
