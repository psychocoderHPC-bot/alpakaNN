/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

using namespace alpaka;

using TestApis = alpaka::nn::test::TestApis;

// Dense vendor-backed GEMM numerical case for float/double operands only. The result must match a host
// scalar reference.
template<typename T_Type>
void runMatmulCase(auto& queue, auto const& device, uint32_t m, uint32_t k, uint32_t n)
{
    auto const aExtent = Vec{m, k};
    auto const bExtent = Vec{k, n};
    auto const cExtent = Vec{m, n};

    auto hostA = onHost::allocHost<T_Type>(aExtent);
    auto hostB = onHost::allocHost<T_Type>(bExtent);
    auto hostC = onHost::allocHost<T_Type>(cExtent);
    auto expectedC = onHost::allocHostLike(hostC);

    for(auto idx : IdxRange{aExtent})
    {
        auto const value = (static_cast<int>(idx.y()) * 3 + static_cast<int>(idx.x()) * 5 + 2) % 11 - 5;
        hostA[idx] = static_cast<T_Type>(value);
    }
    for(auto idx : IdxRange{bExtent})
    {
        auto const value = (static_cast<int>(idx.y()) * 5 + static_cast<int>(idx.x()) * 7 + 1) % 13 - 6;
        hostB[idx] = static_cast<T_Type>(value);
    }
    for(auto idx : IdxRange{cExtent})
    {
        auto const value = (static_cast<int>(idx.y()) * 7 + static_cast<int>(idx.x()) * 3 + 4) % 9 - 4;
        hostC[idx] = static_cast<T_Type>(value);
        expectedC[idx] = hostC[idx];
    }

    for(auto row = 0u; row < m; ++row)
    {
        for(auto col = 0u; col < n; ++col)
        {
            T_Type sum{};
            for(auto inner = 0u; inner < k; ++inner)
                sum += hostA[Vec{row, inner}] * hostB[Vec{inner, col}];
            expectedC[Vec{row, col}] = sum;
        }
    }

    auto devA = onHost::allocLike(device, hostA);
    auto devB = onHost::allocLike(device, hostB);
    auto devC = onHost::allocLike(device, hostC);

    onHost::memcpy(queue, devA, hostA);
    onHost::memcpy(queue, devB, hostB);
    onHost::memcpy(queue, devC, hostC);

    alpaka::nn::onHost::gemm<T_Type>(queue, devA, devB, devC);

    onHost::memcpy(queue, hostC, devC);
    onHost::wait(queue);

    INFO("type=" << onHost::demangledName<T_Type>() << " extents=" << cExtent << " inner=" << k);
    for(auto idx : IdxRange{cExtent})
        alpaka::nn::test::checkValue(hostC[idx], expectedC[idx]);
}

TEMPLATE_LIST_TEST_CASE("dense vendor-backed matrix multiply", "[matrix][matmul]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = onHost::makeDeviceSpec(cfg);
    auto exec = cfg[object::exec];

    auto selector = onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
    {
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    auto device = selector.makeDevice(0);
    INFO("api=" << deviceSpec.getApi().getName());
    INFO("device=" << device.getName());
    INFO("exec=" << onHost::demangledName(exec));

    auto queue = device.makeQueue();

    auto const cases = std::array{
        std::array{1u, 1u, 1u},
        std::array{2u, 3u, 4u},
        std::array{5u, 7u, 3u},
        std::array{16u, 16u, 16u},
        std::array{17u, 8u, 33u},
        std::array{31u, 19u, 7u},
        std::array{9u, 37u, 23u}};

    for(auto const& dims : cases)
    {
        runMatmulCase<float>(queue, device, dims[0], dims[1], dims[2]);
        // Guarded double instantiation: fp64-less oneAPI GPU devices must not build a double SYCL kernel.
        if constexpr(alpaka::nn::test::supportsFp64(device))
            runMatmulCase<double>(queue, device, dims[0], dims[1], dims[2]);
    }
}
