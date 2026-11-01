/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn.hpp>

#include <cmath>

int main()
{
    auto device = alpaka::onHost::makeHostDevice();
    auto queue = device.makeQueue();

    auto input = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 2u});
    auto weight = alpaka::onHost::allocHost<float>(alpaka::Vec{2u, 2u});
    auto output = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 2u});

    input[alpaka::Vec{0u, 0u}] = 1.0f;
    input[alpaka::Vec{0u, 1u}] = 2.0f;
    weight[alpaka::Vec{0u, 0u}] = 1.0f;
    weight[alpaka::Vec{0u, 1u}] = 2.0f;
    weight[alpaka::Vec{1u, 0u}] = 3.0f;
    weight[alpaka::Vec{1u, 1u}] = 4.0f;

    // [1,2] * [[1,2],[3,4]] = [7,10]
    alpaka::nn::onHost::nn::linear<float>(queue, input, weight, output);
    alpaka::onHost::wait(queue);

    auto const ok = std::fabs(output[alpaka::Vec{0u, 0u}] - 7.0f) < 1.0e-6f
                    && std::fabs(output[alpaka::Vec{0u, 1u}] - 10.0f) < 1.0e-6f;
    return ok ? 0 : 1;
}
