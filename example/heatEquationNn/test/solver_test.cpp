// SPDX-License-Identifier: MPL-2.0
#include "../src/ConservativeSolver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

void require(bool ok, char const* message)
{
    if(!ok)
        throw std::runtime_error(message);
}

double sum(std::vector<double> const& v)
{
    return std::accumulate(v.begin(), v.end(), 0.0);
}

int main()
try
{
    constexpr std::size_t n = 6;
    heatclosure::Config c;
    c.n = n;
    c.tmax = 0.01;
    c.steps = 20;
    c.uniformMaterial = true;
    heatclosure::Solver s(c);
    for(std::size_t y = 0; y < n; ++y)
        for(std::size_t x = 0; x < n; ++x)
            s.u[s.index(x, y)] = 0.1 + 0.03 * x + 0.02 * y;
    auto old = s.u;
    auto expected = old;
    double const d = s.dx, inv = 1 / (d * d), a = 0.5;
    for(std::size_t y = 0; y < n; ++y)
        for(std::size_t x = 0; x < n; ++x)
        {
            auto k = y * n + x;
            double rhs = 0;
            rhs += x + 1 < n ? a * (old[k + 1] - old[k]) * inv : 2 * a * (0 - old[k]) * inv;
            rhs += x > 0 ? -a * (old[k] - old[k - 1]) * inv : 2 * a * (1 - old[k]) * inv;
            if(y + 1 < n)
                rhs += a * (old[k + n] - old[k]) * inv;
            if(y > 0)
                rhs -= a * (old[k] - old[k - n]) * inv;
            expected[k] = old[k] + s.dt * rhs;
        }
    s.step();
    double error = 0, norm = 0;
    for(std::size_t k = 0; k < s.u.size(); ++k)
    {
        error += (s.u[k] - expected[k]) * (s.u[k] - expected[k]);
        norm += expected[k] * expected[k];
    }
    require(std::sqrt(error / norm) < 1e-14, "uniform independent reference mismatch");

    heatclosure::Config equilibrium = c;
    equilibrium.leftWall = equilibrium.rightWall = 0.3;
    heatclosure::Solver e(equilibrium);
    std::fill(e.u.begin(), e.u.end(), 0.3);
    e.step();
    for(double v : e.u)
        require(std::abs(v - 0.3) < 1e-14, "compatible equilibrium drift");

    heatclosure::Config closed = c;
    closed.insulatedX = true;
    heatclosure::Solver box(closed);
    for(std::size_t k = 0; k < box.u.size(); ++k)
        box.u[k] = 0.2 + 0.01 * (k % 11);
    auto before = sum(box.u);
    box.step();
    require(std::abs(sum(box.u) - before) < 1e-12, "insulated heat conservation");

    heatclosure::Solver driven(c);
    std::fill(driven.u.begin(), driven.u.end(), 0.2);
    double boundary = 0;
    for(std::size_t y = 0; y < n; ++y)
        boundary += 2 * a * (1 - 0.2) * inv + 2 * a * (0 - 0.2) * inv;
    driven.step();
    require(std::abs(sum(driven.u) - n * n * 0.2 - driven.dt * boundary) < 1e-12, "driven-wall balance");

    bool rejected = false;
    try
    {
        heatclosure::Config unstable = c;
        unstable.tmax = 0.1;
        unstable.steps = 1;
        heatclosure::Solver bad(unstable);
    }
    catch(std::invalid_argument const& ex)
    {
        rejected = std::string(ex.what()).find("minimum steps=") != std::string::npos;
    }
    require(rejected, "unstable requested timestep not rejected with minimum");
    heatclosure::Solver supplied(c);
    std::vector<double> coefficients(supplied.u.size(), 0.5);
    supplied.stepWithAlpha(coefficients);
    for(double a : supplied.alpha)
        require(a == 0.5, "supplied coefficient not retained");
    bool badField = false;
    try
    {
        auto bad = coefficients;
        bad[0] = std::numeric_limits<double>::quiet_NaN();
        supplied.stepWithAlpha(bad);
    }
    catch(std::invalid_argument const&)
    {
        badField = true;
    }
    require(badField, "invalid supplied coefficients not rejected");

    std::cout << "strict solver checks passed\n";
}
catch(std::exception const& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
