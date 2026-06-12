/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpaka/alpaka.hpp>

#include <alpakaNN/alpakaNN.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <vector>

using TestApis = alpakaNN::test::TestApis;

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

TEMPLATE_LIST_TEST_CASE("elementwise kernels", "[ops][elementwise]", TestApis)
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

    auto hostA = alpaka::onHost::allocHost<float>(alpaka::Vec{2u, 4u});
    auto hostB = alpaka::onHost::allocHost<float>(alpaka::Vec{2u, 4u});
    auto hostOut = alpaka::onHost::allocHost<float>(alpaka::Vec{2u, 4u});
    fillSequence<float>(hostA);
    fillSequence<float>(hostB);
    for(auto idx : alpaka::IdxRange{hostB.getExtents()})
        hostB[idx] += 1.0f;

    auto devA = alpaka::onHost::allocLike(device, hostA);
    auto devB = alpaka::onHost::allocLike(device, hostB);
    auto devOut = alpaka::onHost::allocLike(device, hostOut);

    alpaka::onHost::memcpy(queue, devA, hostA);
    alpaka::onHost::memcpy(queue, devB, hostB);

    SECTION("fill copy add sub mul div scale axpy")
    {
        alpakaNN::ops::fill(queue, exec, devOut, 2.0f);
        alpakaNN::ops::copy(queue, exec, devA, devOut);
        alpakaNN::ops::add<float>(queue, exec, devA, devB, devOut);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
            alpakaNN::test::checkValue(hostOut[idx], hostA[idx] + hostB[idx]);

        alpakaNN::ops::sub<float>(queue, exec, devB, devA, devOut);
        alpakaNN::ops::mul<float>(queue, exec, devA, devB, devOut);
        alpakaNN::ops::div<float>(queue, exec, devB, devA, devOut);
        alpakaNN::ops::scale<float>(queue, exec, devA, 3.0f, devOut);
        alpakaNN::ops::axpy<float>(queue, exec, 2.0f, devA, devB, devOut);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
            alpakaNN::test::checkValue(hostOut[idx], 2.0f * hostA[idx] + hostB[idx]);
    }

    SECTION("bias add and cast")
    {
        auto hostBias = alpaka::onHost::allocHost<float>(4u);
        for(uint32_t i = 0u; i < 4u; ++i)
            hostBias[alpaka::Vec{i}] = static_cast<float>(i);
        auto devBias = alpaka::onHost::allocLike(device, hostBias);
        alpaka::onHost::memcpy(queue, devBias, hostBias);

        alpakaNN::ops::biasAdd(queue, exec, devA, devBias, devOut);
        alpaka::onHost::memcpy(queue, hostOut, devOut);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostOut.getExtents()})
            alpakaNN::test::checkValue(hostOut[idx], hostA[idx] + hostBias[alpaka::Vec{idx[1]}]);

        auto hostInt = alpaka::onHost::allocHost<int>(hostOut.getExtents());
        auto devInt = alpaka::onHost::allocLike(device, hostInt);
        alpakaNN::ops::cast<int>(queue, exec, devA, devInt);
        alpaka::onHost::memcpy(queue, hostInt, devInt);
        alpaka::onHost::wait(queue);
        for(auto idx : alpaka::IdxRange{hostInt.getExtents()})
            CHECK(hostInt[idx] == static_cast<int>(hostA[idx]));
    }

    SECTION("transcendentals and activations on padded views")
    {
        std::vector<double> paddedStorage(24u, 0.0);
        std::vector<double> paddedOutStorage(24u, 0.0);
        auto paddedIn = alpakaNN::view::makePaddedView<double>(
            paddedStorage.data(),
            std::array<std::size_t, 2u>{2u, 4u},
            std::array<std::size_t, 2u>{8u, 1u});
        auto paddedOut = alpakaNN::view::makePaddedView<double>(
            paddedOutStorage.data(),
            std::array<std::size_t, 2u>{2u, 4u},
            std::array<std::size_t, 2u>{8u, 1u});
        fillSequence<double>(paddedIn);

        auto devIn = alpaka::onHost::allocLike(device, paddedIn);
        auto devOutLocal = alpaka::onHost::allocLike(device, paddedOut);
        alpaka::onHost::memcpy(queue, devIn, paddedIn);

        alpakaNN::ops::exp<double>(queue, exec, devIn, devOutLocal);
        alpakaNN::ops::sqrt<double>(queue, exec, devOutLocal, devOutLocal);
        alpakaNN::ops::rsqrt<double>(queue, exec, devOutLocal, devOutLocal);
        alpakaNN::ops::relu<double>(queue, exec, devIn, devOutLocal);
        alpakaNN::ops::sigmoid<double>(queue, exec, devIn, devOutLocal);
        alpakaNN::ops::silu<double>(queue, exec, devIn, devOutLocal);
        alpakaNN::ops::gelu<double>(queue, exec, devIn, devOutLocal);
        alpakaNN::ops::swiglu<double>(queue, exec, devIn, devIn, devOutLocal);
        alpaka::onHost::memcpy(queue, paddedOut, devOutLocal);
        alpaka::onHost::wait(queue);

        for(auto idx : alpaka::IdxRange{paddedOut.getExtents()})
        {
            auto const x = paddedIn[idx];
            auto const sigmoid = 1.0 / (1.0 + std::exp(-x));
            auto const expected = x * sigmoid * x;
            alpakaNN::test::checkValue(paddedOut[idx], expected, 1.0e-9, 1.0e-9);
        }
    }
}
