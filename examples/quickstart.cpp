/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

// Minimal alpakaNN quickstart: include the umbrella header, pick a device,
// allocate caller-owned views, enqueue an operation, synchronize the queue and
// read the result back. Build it through CMake with the `alpaka::nn` target:

//     find_package(alpakaNN CONFIG REQUIRED)
//     target_link_libraries(myTarget PRIVATE alpaka::nn)

#include <alpaka/nn/nn.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

int main()
{
    // 1. Pick a device and create a queue. This example uses the host device.
    auto device = alpaka::onHost::makeHostDevice();
    auto exec = alpaka::exec::cpuSerial;
    auto queue = device.makeQueue();

    constexpr uint32_t n = 8u;

    // 2. Allocate caller-owned views. alpakaNN operations never allocate their
    //    output storage; the caller provides it.
    auto a = alpaka::onHost::allocHost<float>(alpaka::Vec{n});
    auto b = alpaka::onHost::allocHost<float>(alpaka::Vec{n});
    auto out = alpaka::onHost::allocHost<float>(alpaka::Vec{n});

    for(uint32_t i = 0u; i < n; ++i)
    {
        a[alpaka::Vec{i}] = static_cast<float>(i);
        b[alpaka::Vec{i}] = static_cast<float>(2u * i + 1u);
    }

    // 3. Enqueue the work. Every operation is asynchronous with respect to the
    //    host; the views must stay alive until the enqueued work has completed.
    alpaka::nn::onHost::ops::add<float>(queue, exec, a, b, out);
    alpaka::nn::onHost::ops::relu<float>(queue, exec, out, out);

    // 4. Synchronize the queue before consuming the result on the host.
    alpaka::onHost::wait(queue);

    // 5. Read the result back and check it.
    bool ok = true;
    for(uint32_t i = 0u; i < n; ++i)
    {
        float const expected = static_cast<float>(3u * i + 1u);
        if(out[alpaka::Vec{i}] != expected)
            ok = false;
    }

    if(!ok)
    {
        std::cerr << "alpakaNN quickstart: unexpected result\n";
        return EXIT_FAILURE;
    }

    std::cout << "alpakaNN quickstart: OK\n";
    return EXIT_SUCCESS;
}
