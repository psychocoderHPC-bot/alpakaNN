// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace heatclosure
{
    struct Config
    {
        std::size_t n = 64;
        double tmax = 0.1;
        double beta = 0.5;
        double alphaMin = 0.01;
        double alphaMax = 6.0;
        bool validateStrict = false;
        std::size_t steps = 0; // zero selects the conservative automatic count
        bool uniformMaterial = false;
        bool insulatedX = false;
        double leftWall = 1.0, rightWall = 0.0;
    };

    inline double baseAlpha(double x, double y)
    {
        if((x - 0.35) * (x - 0.35) + (y - 0.5) * (y - 0.5) < 0.12 * 0.12)
            return 0.02;
        if(std::abs(y - 0.5) < 0.05 && x > 0.45 && x < 0.65)
            return 4.0;
        return 0.5 + 0.4 * (std::sin(6.0 * std::acos(-1.0) * y) >= 0.0 ? 1.0 : 0.0);
    }

    inline double alphaTrue(double x, double y, double u, double beta)
    {
        return baseAlpha(x, y) * (1.0 + beta * u);
    }

    inline double harmonic(double a, double b)
    {
        return 2.0 * a * b / (a + b);
    }

    struct Solver
    {
        Config cfg;
        std::size_t steps;
        double dx, dt;
        std::vector<double> u, alpha;

        double coefficient(double x, double y, double temp) const
        {
            return cfg.uniformMaterial ? 0.5 : alphaTrue(x, y, temp, cfg.beta);
        }

        explicit Solver(Config c)
            : cfg(c)
            , steps(c.steps)
            , dx(1.0 / static_cast<double>(c.n))
            , dt(0.0)
            , u(c.n * c.n, 0.0)
            , alpha(c.n * c.n)
        {
            if(c.n < 2 || !(c.tmax > 0.0) || !std::isfinite(c.tmax) || !(c.alphaMax > c.alphaMin)
               || !(c.alphaMin > 0.0) || !std::isfinite(c.alphaMin) || !std::isfinite(c.alphaMax)
               || !std::isfinite(c.beta) || c.beta < 0.0)
                throw std::invalid_argument("grid>=2, finite tmax>0, finite 0<alphaMin<alphaMax and beta>=0 required");
            auto const bound = std::max(c.alphaMax, 4.0 * (1.0 + c.beta));
            auto const limit = 0.9 / (bound * (5.0 / (dx * dx)));
            auto const minimum = static_cast<std::size_t>(std::ceil(c.tmax / limit));
            if(steps == 0)
                steps = minimum;
            if(steps < minimum)
                throw std::invalid_argument("unstable explicit timestep; minimum steps=" + std::to_string(minimum));
            dt = c.tmax / static_cast<double>(steps);
            refreshAlpha();
        }

        std::size_t index(std::size_t x, std::size_t y) const
        {
            return y * cfg.n + x;
        }

        void refreshAlpha()
        {
            for(std::size_t y = 0; y < cfg.n; ++y)
                for(std::size_t x = 0; x < cfg.n; ++x)
                {
                    auto const a = coefficient((x + 0.5) * dx, (y + 0.5) * dx, u[index(x, y)]);
                    if(!(a > 0.0) || !std::isfinite(a))
                        throw std::runtime_error("coefficient must be finite and positive");
                    alpha[index(x, y)] = a;
                }
        }

        void advance(std::vector<double>& state, double stepDt, std::vector<double> const* supplied = nullptr)
        {
            auto old = state;
            std::vector<double> a(old.size());
            if(supplied)
                a = *supplied;
            else
                for(std::size_t y = 0; y < cfg.n; ++y)
                    for(std::size_t x = 0; x < cfg.n; ++x)
                        a[index(x, y)] = coefficient((x + 0.5) * dx, (y + 0.5) * dx, old[index(x, y)]);
            auto next = old;
            auto const inv = 1.0 / (dx * dx);
            for(std::size_t y = 0; y < cfg.n; ++y)
                for(std::size_t x = 0; x < cfg.n; ++x)
                {
                    auto k = index(x, y);
                    double div = 0.0;
                    if(x + 1 < cfg.n)
                        div += harmonic(a[k], a[index(x + 1, y)]) * (old[index(x + 1, y)] - old[k]) * inv;
                    else if(!cfg.insulatedX)
                        div += 2.0 * a[k] * (cfg.rightWall - old[k]) * inv;
                    if(x > 0)
                        div -= harmonic(a[k], a[index(x - 1, y)]) * (old[k] - old[index(x - 1, y)]) * inv;
                    else if(!cfg.insulatedX)
                        div += 2.0 * a[k] * (cfg.leftWall - old[k]) * inv;
                    if(y + 1 < cfg.n)
                        div += harmonic(a[k], a[index(x, y + 1)]) * (old[index(x, y + 1)] - old[k]) * inv;
                    if(y > 0)
                        div -= harmonic(a[k], a[index(x, y - 1)]) * (old[k] - old[index(x, y - 1)]) * inv;
                    next[k] = old[k] + stepDt * div;
                    if(!std::isfinite(next[k]))
                        throw std::runtime_error("non-finite temperature");
                }
            state.swap(next);
        }

        void step()
        {
            advance(u, dt);
            refreshAlpha();
        }

        void stepWithAlpha(std::vector<double> const& currentAlpha)
        {
            if(currentAlpha.size() != u.size())
                throw std::invalid_argument("coefficient field size mismatch");
            for(double a : currentAlpha)
                if(!(a > 0.0) || !std::isfinite(a))
                    throw std::invalid_argument("coefficient must be finite and positive");
            advance(u, dt, &currentAlpha);
            alpha = currentAlpha;
        }
    };
} // namespace heatclosure
