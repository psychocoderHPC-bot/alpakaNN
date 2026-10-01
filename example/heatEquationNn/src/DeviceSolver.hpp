// SPDX-License-Identifier: MPL-2.0
#pragma once

#include "ConservativeSolver.hpp"
#include "ModelLoader.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/nn/mlp.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace heatclosure
{
    enum class CoefficientMode
    {
        preset,
        uniform,
        neural
    };

    namespace device_detail
    {
        struct PackFeatures
        {
            double dx;
            uint32_t n;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto features, auto u) const
            {
                for(auto i : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{u.getExtents()}))
                {
                    auto const row = i[0];
                    features[alpaka::Vec{row, 0u}] = static_cast<float>(u[i]);
                    features[alpaka::Vec{row, 1u}] = static_cast<float>((i[0] % n + 0.5) * dx);
                    features[alpaka::Vec{row, 2u}] = static_cast<float>((i[0] / n + 0.5) * dx);
                }
            }
        };

        struct NeuralAlpha
        {
            double lo, hi;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto alpha, auto logits) const
            {
                for(auto i : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpha.getExtents()}))
                {
                    auto const z = static_cast<double>(logits[alpaka::Vec{i[0], 0u}]);
                    auto const s = 1.0 / (1.0 + alpaka::math::exp(-z));
                    alpha[i] = lo + (hi - lo) * s;
                }
            }
        };

        ALPAKA_FN_ACC inline double materialBase(double x, double y)
        {
            auto const pi = 3.14159265358979323846;
            if((x - 0.35) * (x - 0.35) + (y - 0.5) * (y - 0.5) < 0.12 * 0.12)
                return 0.02;
            if(alpaka::math::abs(y - 0.5) < 0.05 && x > 0.45 && x < 0.65)
                return 4.0;
            return 0.5 + 0.4 * (alpaka::math::sin(6.0 * pi * y) >= 0.0 ? 1.0 : 0.0);
        }

        struct PresetAlpha
        {
            double dx, beta;
            uint32_t n;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto alpha, auto u) const
            {
                for(auto i : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{u.getExtents()}))
                {
                    auto const x = (i[0] % n + 0.5) * dx;
                    auto const y = (i[0] / n + 0.5) * dx;
                    alpha[i] = materialBase(x, y) * (1.0 + beta * u[i]);
                }
            }
        };

        struct UniformAlpha
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto alpha) const
            {
                for(auto i : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpha.getExtents()}))
                    alpha[i] = 0.5;
            }
        };

        struct Stencil
        {
            double dx, dt, left, right;
            uint32_t n;
            bool insulatedX;

            ALPAKA_FN_ACC static double face(double a, double b)
            {
                return 2.0 * a * b / (a + b);
            }

            ALPAKA_FN_ACC void operator()(auto const& acc, auto next, auto old, auto alpha) const
            {
                auto const inv = 1.0 / (dx * dx);
                for(auto i : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{old.getExtents()}))
                {
                    auto const row = i[0];
                    auto const x = row % n;
                    auto const y = row / n;
                    auto div = 0.0;
                    if(x + 1 < n)
                        div += face(alpha[alpaka::Vec{row}], alpha[alpaka::Vec{row + 1}])
                               * (old[alpaka::Vec{row + 1}] - old[alpaka::Vec{row}]) * inv;
                    else if(!insulatedX)
                        div += 2.0 * alpha[alpaka::Vec{row}] * (right - old[alpaka::Vec{row}]) * inv;
                    if(x > 0)
                        div -= face(alpha[alpaka::Vec{row}], alpha[alpaka::Vec{row - 1}])
                               * (old[alpaka::Vec{row}] - old[alpaka::Vec{row - 1}]) * inv;
                    else if(!insulatedX)
                        div += 2.0 * alpha[alpaka::Vec{row}] * (left - old[alpaka::Vec{row}]) * inv;
                    if(y + 1 < n)
                        div += face(alpha[alpaka::Vec{row}], alpha[alpaka::Vec{row + n}])
                               * (old[alpaka::Vec{row + n}] - old[alpaka::Vec{row}]) * inv;
                    if(y > 0)
                        div -= face(alpha[alpaka::Vec{row}], alpha[alpaka::Vec{row - n}])
                               * (old[alpaka::Vec{row}] - old[alpaka::Vec{row - n}]) * inv;
                    next[alpaka::Vec{row}] = old[alpaka::Vec{row}] + dt * div;
                }
            }
        };
    } // namespace device_detail

    /** Queue-backed device-buffer solver. All step operations remain on the supplied queue;
     * snapshots are the explicit synchronization boundary for host consumers.
     */
    template<class TQueue, class TDevice>
    class DeviceSolver
    {
    public:
        using Field = decltype(alpaka::onHost::alloc<double>(std::declval<TDevice const&>(), alpaka::Vec{1u}));
        Config const cfg;
        std::size_t const steps;
        double const dx, dt;

        DeviceSolver(
            TQueue& queue,
            TDevice const& device,
            Config config,
            CoefficientMode mode,
            std::vector<double> const& initial,
            Model const* model = nullptr)
            : cfg(config)
            , steps(checkedSteps(config))
            , dx(1.0 / static_cast<double>(config.n))
            , dt(config.tmax / static_cast<double>(steps))
            , m_queue(&queue)
            , m_mode(mode)
            , m_u(alpaka::onHost::alloc<double>(device, alpaka::Vec{static_cast<uint32_t>(config.n * config.n)}))
            , m_next(alpaka::onHost::alloc<double>(device, alpaka::Vec{static_cast<uint32_t>(config.n * config.n)}))
            , m_alpha(alpaka::onHost::alloc<double>(device, alpaka::Vec{static_cast<uint32_t>(config.n * config.n)}))
            , m_features(
                  alpaka::onHost::alloc<float>(device, alpaka::Vec{static_cast<uint32_t>(config.n * config.n), 3u}))
            , m_logits(
                  alpaka::onHost::alloc<float>(device, alpaka::Vec{static_cast<uint32_t>(config.n * config.n), 1u}))
            , m_gate(
                  alpaka::onHost::alloc<float>(device, alpaka::Vec{static_cast<uint32_t>(config.n * config.n), 64u}))
            , m_up(alpaka::onHost::alloc<float>(device, alpaka::Vec{static_cast<uint32_t>(config.n * config.n), 64u}))
            , m_hidden(
                  alpaka::onHost::alloc<float>(device, alpaka::Vec{static_cast<uint32_t>(config.n * config.n), 64u}))
            , m_wgate(alpaka::onHost::alloc<float>(device, alpaka::Vec{3u, 64u}))
            , m_wup(alpaka::onHost::alloc<float>(device, alpaka::Vec{3u, 64u}))
            , m_wdown(alpaka::onHost::alloc<float>(device, alpaka::Vec{64u, 1u}))
        {
            if(initial.size() != config.n * config.n)
                throw std::invalid_argument("initial field size mismatch");
            for(double v : initial)
                if(!std::isfinite(v))
                    throw std::invalid_argument("initial field must be finite");
            auto host = alpaka::onHost::allocHost<double>(alpaka::Vec{static_cast<uint32_t>(initial.size())});
            for(std::size_t i = 0; i < initial.size(); ++i)
                host[alpaka::Vec{static_cast<uint32_t>(i)}] = initial[i];
            alpaka::onHost::memcpy(queue, m_u, host);
            alpaka::onHost::wait(queue); // staging memory must stay alive through the initial transfer
            if(mode == CoefficientMode::neural)
            {
                if(model == nullptr || !(model->alphaMin > 0.0) || !(model->alphaMax > model->alphaMin)
                   || model->beta != config.beta || model->alphaMin != config.alphaMin
                   || model->alphaMax != config.alphaMax)
                    throw std::invalid_argument("neural mode requires matching model metadata");
                allocateModel(queue, *model);
            }
        }

        void step(TQueue& queue, auto exec)
        {
            requireQueue(queue);
            if(m_mode == CoefficientMode::neural)
            {
                queue.enqueue(
                    alpaka::onHost::getFrameSpec(queue.getDevice(), exec, m_u.getExtents()),
                    alpaka::KernelBundle{
                        device_detail::PackFeatures{dx, static_cast<uint32_t>(cfg.n)},
                        m_features,
                        m_u});
                alpaka::nn::onHost::nn::mlp<
                    float>(queue, exec, m_features, m_wgate, m_wup, m_wdown, m_gate, m_up, m_hidden, m_logits);
                queue.enqueue(
                    alpaka::onHost::getFrameSpec(queue.getDevice(), exec, m_u.getExtents()),
                    alpaka::KernelBundle{device_detail::NeuralAlpha{cfg.alphaMin, cfg.alphaMax}, m_alpha, m_logits});
            }
            else if(m_mode == CoefficientMode::preset)
                queue.enqueue(
                    alpaka::onHost::getFrameSpec(queue.getDevice(), exec, m_u.getExtents()),
                    alpaka::KernelBundle{
                        device_detail::PresetAlpha{dx, cfg.beta, static_cast<uint32_t>(cfg.n)},
                        m_alpha,
                        m_u});
            else
                queue.enqueue(
                    alpaka::onHost::getFrameSpec(queue.getDevice(), exec, m_u.getExtents()),
                    alpaka::KernelBundle{device_detail::UniformAlpha{}, m_alpha});
            queue.enqueue(
                alpaka::onHost::getFrameSpec(queue.getDevice(), exec, m_u.getExtents()),
                alpaka::KernelBundle{
                    device_detail::Stencil{
                        dx,
                        dt,
                        cfg.leftWall,
                        cfg.rightWall,
                        static_cast<uint32_t>(cfg.n),
                        cfg.insulatedX},
                    m_next,
                    m_u,
                    m_alpha});
            std::swap(m_u, m_next);
        }

        std::vector<double> snapshot(TQueue& queue) const
        {
            requireQueue(queue);
            auto host = alpaka::onHost::allocHost<double>(alpaka::Vec{static_cast<uint32_t>(cfg.n * cfg.n)});
            alpaka::onHost::memcpy(queue, host, m_u);
            alpaka::onHost::wait(queue);
            std::vector<double> result(cfg.n * cfg.n);
            for(std::size_t i = 0; i < result.size(); ++i)
            {
                result[i] = host[alpaka::Vec{static_cast<uint32_t>(i)}];
                if(!std::isfinite(result[i]))
                    throw std::runtime_error("non-finite device temperature");
            }
            return result;
        }

        /** Coefficients for the current field state (including after the most recent step). */
        std::vector<double> coefficientSnapshot(TQueue& queue, auto exec)
        {
            requireQueue(queue);
            // step() leaves alpha for the state it consumed. Recompute for the present
            // temperature so coefficient diagnostics are aligned with snapshots of u.
            updateAlpha(queue, exec);
            auto host = alpaka::onHost::allocHost<double>(alpaka::Vec{static_cast<uint32_t>(cfg.n * cfg.n)});
            alpaka::onHost::memcpy(queue, host, m_alpha);
            alpaka::onHost::wait(queue);
            std::vector<double> result(cfg.n * cfg.n);
            for(std::size_t i = 0; i < result.size(); ++i)
            {
                result[i] = host[alpaka::Vec{static_cast<uint32_t>(i)}];
                if(!(result[i] > 0.0) || !std::isfinite(result[i]))
                    throw std::runtime_error("device coefficient must be finite and positive");
            }
            return result;
        }

    private:
        TQueue* m_queue;
        CoefficientMode m_mode;
        Field m_u, m_next, m_alpha;
        decltype(alpaka::onHost::alloc<float>(std::declval<TDevice const&>(), alpaka::Vec{1u, 3u})) m_features;
        decltype(alpaka::onHost::alloc<float>(std::declval<TDevice const&>(), alpaka::Vec{1u, 1u})) m_logits;
        decltype(alpaka::onHost::alloc<float>(std::declval<TDevice const&>(), alpaka::Vec{1u, 64u})) m_gate, m_up,
            m_hidden;
        decltype(alpaka::onHost::alloc<float>(std::declval<TDevice const&>(), alpaka::Vec{3u, 64u})) m_wgate, m_wup;
        decltype(alpaka::onHost::alloc<float>(std::declval<TDevice const&>(), alpaka::Vec{64u, 1u})) m_wdown;

        static std::size_t checkedSteps(Config const& c)
        {
            if(c.n < 2 || c.n > std::numeric_limits<uint32_t>::max() / c.n || !(c.tmax > 0.0) || !std::isfinite(c.tmax)
               || !(c.alphaMax > c.alphaMin) || !(c.alphaMin > 0.0) || !std::isfinite(c.alphaMin)
               || !std::isfinite(c.alphaMax) || !std::isfinite(c.beta) || c.beta < 0.0 || !std::isfinite(c.leftWall)
               || !std::isfinite(c.rightWall))
                throw std::invalid_argument("invalid device solver configuration");
            auto const dx = 1.0 / static_cast<double>(c.n);
            auto const bound = std::max(c.alphaMax, 4.0 * (1.0 + c.beta));
            auto const limit = (0.9 * dx * dx) / (5.0 * bound);
            auto const required = c.tmax / limit;
            if(!(limit > 0.0) || !std::isfinite(limit) || !std::isfinite(required)
               || required >= static_cast<double>(std::numeric_limits<std::size_t>::max()))
                throw std::invalid_argument("required timestep count is out of range");
            auto const minimum = static_cast<std::size_t>(std::ceil(required));
            auto const count = c.steps == 0 ? minimum : c.steps;
            if(count < minimum)
                throw std::invalid_argument("unstable explicit timestep; minimum steps=" + std::to_string(minimum));
            return count;
        }

        void requireQueue(TQueue& queue) const
        {
            if(&queue != m_queue)
                throw std::invalid_argument("DeviceSolver operations must use its construction queue");
        }

        void updateAlpha(TQueue& queue, auto exec)
        {
            if(m_mode == CoefficientMode::neural)
            {
                queue.enqueue(
                    alpaka::onHost::getFrameSpec(queue.getDevice(), exec, m_u.getExtents()),
                    alpaka::KernelBundle{
                        device_detail::PackFeatures{dx, static_cast<uint32_t>(cfg.n)},
                        m_features,
                        m_u});
                alpaka::nn::onHost::nn::mlp<
                    float>(queue, exec, m_features, m_wgate, m_wup, m_wdown, m_gate, m_up, m_hidden, m_logits);
                queue.enqueue(
                    alpaka::onHost::getFrameSpec(queue.getDevice(), exec, m_u.getExtents()),
                    alpaka::KernelBundle{device_detail::NeuralAlpha{cfg.alphaMin, cfg.alphaMax}, m_alpha, m_logits});
            }
            else if(m_mode == CoefficientMode::preset)
                queue.enqueue(
                    alpaka::onHost::getFrameSpec(queue.getDevice(), exec, m_u.getExtents()),
                    alpaka::KernelBundle{
                        device_detail::PresetAlpha{dx, cfg.beta, static_cast<uint32_t>(cfg.n)},
                        m_alpha,
                        m_u});
            else
                queue.enqueue(
                    alpaka::onHost::getFrameSpec(queue.getDevice(), exec, m_u.getExtents()),
                    alpaka::KernelBundle{device_detail::UniformAlpha{}, m_alpha});
        }

        void allocateModel(TQueue& queue, Model const& model)
        {
            auto gate = alpaka::onHost::allocHost<float>(alpaka::Vec{3u, 64u});
            auto up = alpaka::onHost::allocHost<float>(alpaka::Vec{3u, 64u});
            auto down = alpaka::onHost::allocHost<float>(alpaka::Vec{64u, 1u});
            for(std::size_t i = 0; i < 192; ++i)
            {
                gate[alpaka::Vec{static_cast<uint32_t>(i / 64), static_cast<uint32_t>(i % 64)}] = model.gate[i];
                up[alpaka::Vec{static_cast<uint32_t>(i / 64), static_cast<uint32_t>(i % 64)}] = model.up[i];
            }
            for(std::size_t i = 0; i < 64; ++i)
                down[alpaka::Vec{static_cast<uint32_t>(i), 0u}] = model.down[i];
            alpaka::onHost::memcpy(queue, m_wgate, gate);
            alpaka::onHost::memcpy(queue, m_wup, up);
            alpaka::onHost::memcpy(queue, m_wdown, down);
            alpaka::onHost::wait(queue); // weight staging buffers are local to construction
        }
    };
} // namespace heatclosure
