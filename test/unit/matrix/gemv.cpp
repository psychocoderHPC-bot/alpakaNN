/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "test.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    // Deterministic (integer-valued) operands keep the reference exact for a 1 ulp free scalar comparison.
    // Dense vendor-backed GEMV numerical case for float/double operands only.
    template<typename T_Type>
    void runGemvCase(auto& queue, auto const& device, uint32_t rows, uint32_t cols)
    {
        auto hostW = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{rows, cols});
        auto hostX = alpaka::onHost::allocHost<T_Type>(cols);
        auto hostY = alpaka::onHost::allocHost<T_Type>(rows);

        for(auto idx : alpaka::IdxRange{hostW.getExtents()})
            hostW[idx] = static_cast<T_Type>((idx.y() * 5u + idx.x() * 3u + 1u) % 13u - 6u);
        for(uint32_t i = 0u; i < cols; ++i)
            hostX[alpaka::Vec{i}] = static_cast<T_Type>(i % 7u - 3u);

        std::vector<T_Type> expected(rows, T_Type{});
        for(uint32_t row = 0u; row < rows; ++row)
            for(uint32_t col = 0u; col < cols; ++col)
                expected[row] += hostW[alpaka::Vec{row, col}] * hostX[alpaka::Vec{col}];

        auto devW = alpaka::onHost::allocLike(device, hostW);
        auto devX = alpaka::onHost::allocLike(device, hostX);
        auto devY = alpaka::onHost::allocLike(device, hostY);
        alpaka::onHost::memcpy(queue, devW, hostW);
        alpaka::onHost::memcpy(queue, devX, hostX);

        alpaka::nn::onHost::gemv<T_Type>(queue, devW, devX, devY);
        alpaka::onHost::memcpy(queue, hostY, devY);
        alpaka::onHost::wait(queue);

        for(uint32_t row = 0u; row < rows; ++row)
            alpaka::nn::test::checkValue(hostY[alpaka::Vec{row}], expected[row]);
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("dense vendor-backed gemv", "[matrix][gemv]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }

    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    runGemvCase<float>(queue, device, 3u, 4u);
    runGemvCase<float>(queue, device, 5u, 7u);
    runGemvCase<float>(queue, device, 1u, 1u);
    runGemvCase<float>(queue, device, 11u, 13u);

    // double gemv reference cases (the vendor oneMKL/cuBLAS path is fp64-sensitive, so cover them explicitly).
    // The double instantiation is guarded so fp64-less devices (e.g. oneAPI GPU) only build float kernels.
    if constexpr(alpaka::nn::test::supportsFp64(device))
    {
        runGemvCase<double>(queue, device, 3u, 4u);
        runGemvCase<double>(queue, device, 5u, 7u);
        runGemvCase<double>(queue, device, 1u, 1u);
        runGemvCase<double>(queue, device, 11u, 13u);
    }
}
