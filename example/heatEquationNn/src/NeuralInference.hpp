// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "ModelLoader.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/nn/mlp.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace heatclosure
{
    namespace detail
    {
        /** Expand one raw `(u, x, y)` sample into row-major feature order.
         *
         * v1 (`raw`) emits `[u, x, y]`. v2 (`fourier_xy_k0_5`) appends, for each
         * coordinate in `(x, y)` and `k = 0..5`, `sin(2^k*pi*coord)` then
         * `cos(2^k*pi*coord)`. The trigonometric arguments are evaluated in double,
         * matching the device kernel, and only the stored value is narrowed to `T`.
         */
        template<class T>
        inline void emitFeatures(std::vector<T>& out, FeatureEncoding encoding, double u, double x, double y)
        {
            out.push_back(static_cast<T>(u));
            out.push_back(static_cast<T>(x));
            out.push_back(static_cast<T>(y));
            if(encoding == FeatureEncoding::fourier_xy_k0_5)
            {
                constexpr double pi = 3.14159265358979323846;
                for(int k = 0; k < 6; ++k)
                {
                    auto const w = static_cast<double>(1 << k) * pi;
                    out.push_back(static_cast<T>(std::sin(w * x)));
                    out.push_back(static_cast<T>(std::cos(w * x)));
                    out.push_back(static_cast<T>(std::sin(w * y)));
                    out.push_back(static_cast<T>(std::cos(w * y)));
                }
            }
        }
    } // namespace detail

    // This public host MLP currently allocates three hidden buffers and waits between
    // stages; each call synchronizes. This is intentionally not an asynchronous solver kernel.
    template<class TQueue, class TExec, class TDevice>
    std::vector<double> infer(
        TQueue& queue,
        TExec exec,
        TDevice const& device,
        Model const& model,
        std::vector<std::array<double, 3>> const& points)
    {
        using namespace alpaka;
        auto const inputDim = static_cast<uint32_t>(model.inputDim);
        auto const width = static_cast<uint32_t>(model.width);
        auto const batch = static_cast<uint32_t>(points.size());
        auto x = onHost::allocHost<float>(Vec{batch, inputDim});
        auto wg = onHost::allocHost<float>(Vec{inputDim, width});
        auto wu = onHost::allocHost<float>(Vec{inputDim, width});
        auto wd = onHost::allocHost<float>(Vec{width, 1u});
        auto y = onHost::allocHost<float>(Vec{batch, 1u});
        std::vector<float> featureRow;
        featureRow.reserve(inputDim);
        for(size_t i = 0; i < points.size(); ++i)
        {
            featureRow.clear();
            detail::emitFeatures(featureRow, model.encoding, points[i][0], points[i][1], points[i][2]);
            for(size_t j = 0; j < inputDim; ++j)
                x[Vec{static_cast<uint32_t>(i), static_cast<uint32_t>(j)}] = featureRow[j];
        }
        for(size_t i = 0; i < inputDim; ++i)
            for(size_t j = 0; j < width; ++j)
            {
                wg[Vec{static_cast<uint32_t>(i), static_cast<uint32_t>(j)}] = model.gate[i * model.width + j];
                wu[Vec{static_cast<uint32_t>(i), static_cast<uint32_t>(j)}] = model.up[i * model.width + j];
            }
        for(size_t i = 0; i < width; ++i)
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
        std::vector<double> out(points.size());
        for(size_t i = 0; i < out.size(); ++i)
        {
            double z = y[Vec{static_cast<uint32_t>(i), 0u}];
            double s = 1.0 / (1.0 + std::exp(-z));
            out[i] = model.alphaMin + (model.alphaMax - model.alphaMin) * s;
        }
        return out;
    }
} // namespace heatclosure
