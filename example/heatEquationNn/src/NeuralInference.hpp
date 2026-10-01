// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "ModelLoader.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/nn/mlp.hpp>

#include <array>
#include <vector>

namespace heatclosure
{
    // This public host MLP currently allocates three hidden buffers and waits between
    // stages; each call synchronizes. This is intentionally not an asynchronous solver kernel.
    template<class TQueue, class TExec, class TDevice>
    std::vector<double> infer(
        TQueue& queue,
        TExec exec,
        TDevice const& device,
        Model const& model,
        std::vector<std::array<double, 3>> const& features)
    {
        using namespace alpaka;
        auto x = onHost::allocHost<float>(Vec{static_cast<uint32_t>(features.size()), 3u});
        auto wg = onHost::allocHost<float>(Vec{3u, 64u});
        auto wu = onHost::allocHost<float>(Vec{3u, 64u});
        auto wd = onHost::allocHost<float>(Vec{64u, 1u});
        auto y = onHost::allocHost<float>(Vec{static_cast<uint32_t>(features.size()), 1u});
        for(size_t i = 0; i < features.size(); ++i)
            for(size_t j = 0; j < 3; ++j)
                x[Vec{static_cast<uint32_t>(i), static_cast<uint32_t>(j)}] = static_cast<float>(features[i][j]);
        for(size_t i = 0; i < 3; ++i)
            for(size_t j = 0; j < 64; ++j)
            {
                wg[Vec{static_cast<uint32_t>(i), static_cast<uint32_t>(j)}] = model.gate[i * 64 + j];
                wu[Vec{static_cast<uint32_t>(i), static_cast<uint32_t>(j)}] = model.up[i * 64 + j];
            }
        for(size_t i = 0; i < 64; ++i)
            wd[Vec{static_cast<uint32_t>(i), 0u}] = model.down[i];
        auto dx = onHost::allocLike(device, x);
        auto dgate = onHost::allocLike(device, wg);
        auto dup = onHost::allocLike(device, wu);
        auto ddown = onHost::allocLike(device, wd);
        auto dy = onHost::allocLike(device, y);
        onHost::memcpy(queue, dx, x);
        onHost::memcpy(queue, dgate, wg);
        onHost::memcpy(queue, dup, wu);
        onHost::memcpy(queue, ddown, wd);
        nn::onHost::nn::mlp<float>(queue, exec, dx, dgate, dup, ddown, dy);
        onHost::memcpy(queue, y, dy);
        onHost::wait(queue);
        std::vector<double> out(features.size());
        for(size_t i = 0; i < out.size(); ++i)
        {
            double z = y[Vec{static_cast<uint32_t>(i), 0u}];
            double s = 1.0 / (1.0 + std::exp(-z));
            out[i] = model.alphaMin + (model.alphaMax - model.alphaMin) * s;
        }
        return out;
    }
} // namespace heatclosure
