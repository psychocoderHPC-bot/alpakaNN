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

    // Stability accept/reject exactly at the documented conservative bound. The
    // bound is limit = 0.9*dx^2/(5*bound), minimum = ceil(tmax/limit). Choosing
    // tmax = limit*(minimum - 0.5) makes the required count floor to exactly
    // `minimum`, so `steps == minimum` must be accepted and `minimum - 1` rejected.
    {
        heatclosure::Config boundary = c;
        boundary.uniformMaterial = true;
        boundary.leftWall = 1.0;
        boundary.rightWall = 0.0;
        auto const dxB = 1.0 / static_cast<double>(n);
        auto const boundB = std::max(boundary.alphaMax, 4.0 * (1.0 + boundary.beta));
        auto const limitB = 0.9 * dxB * dxB / (5.0 * boundB);
        auto const minimum = std::size_t{10};
        boundary.tmax = limitB * (static_cast<double>(minimum) - 0.5);
        boundary.steps = 0;
        heatclosure::Solver automatic(boundary);
        require(automatic.steps == minimum, "automatic step count differs from the documented bound");
        boundary.steps = minimum;
        heatclosure::Solver accepted(boundary);
        require(accepted.steps == minimum, "minimum stable step count was not retained");
        boundary.steps = minimum - 1;
        bool belowMinimumRejected = false;
        std::string minimumMessage;
        try
        {
            heatclosure::Solver tooFew(boundary);
        }
        catch(std::invalid_argument const& ex)
        {
            belowMinimumRejected = true;
            minimumMessage = ex.what();
        }
        require(belowMinimumRejected, "one step below the documented minimum was accepted");
        require(
            minimumMessage.find("minimum steps=" + std::to_string(minimum)) != std::string::npos,
            "rejection message does not report the documented minimum");
    }

    // Half-cell Dirichlet wall contribution: a uniform-coefficient boundary cell
    // exchanges with the wall through a half cell, i.e. the wall face contributes
    // 2*alpha/dx^2 rather than alpha/dx^2. The independent reference below uses the
    // factor 2; a comparison against a deliberately wrong factor 1 reference shows
    // the test actually exercises that term. The documented conservative bound
    // (5*bound) is left unchanged.
    {
        heatclosure::Config wallCfg = c;
        wallCfg.uniformMaterial = true;
        wallCfg.leftWall = 0.0;
        wallCfg.rightWall = 0.0;
        wallCfg.steps = 20;
        heatclosure::Solver wallSolver(wallCfg);
        std::fill(wallSolver.u.begin(), wallSolver.u.end(), 1.0);
        auto const inv = 1.0 / (wallSolver.dx * wallSolver.dx);
        auto const a = 0.5;
        auto const old = wallSolver.u;
        auto reference = [&](double wallFactor)
        {
            auto expected = old;
            for(std::size_t y = 0; y < n; ++y)
                for(std::size_t x = 0; x < n; ++x)
                {
                    auto const k = y * n + x;
                    double rhs = 0.0;
                    rhs += x + 1 < n ? a * (old[k + 1] - old[k]) * inv
                                     : wallFactor * a * (wallCfg.rightWall - old[k]) * inv;
                    rhs += x > 0 ? -a * (old[k] - old[k - 1]) * inv
                                 : wallFactor * a * (wallCfg.leftWall - old[k]) * inv;
                    if(y + 1 < n)
                        rhs += a * (old[k + n] - old[k]) * inv;
                    if(y > 0)
                        rhs -= a * (old[k] - old[k - n]) * inv;
                    expected[k] = old[k] + wallSolver.dt * rhs;
                }
            return expected;
        };
        auto const correct = reference(2.0);
        auto const wrong = reference(1.0);
        wallSolver.step();
        double error = 0.0, norm = 0.0, factorGap = 0.0;
        for(std::size_t k = 0; k < wallSolver.u.size(); ++k)
        {
            error += (wallSolver.u[k] - correct[k]) * (wallSolver.u[k] - correct[k]);
            norm += correct[k] * correct[k];
            factorGap = std::max(factorGap, std::abs(correct[k] - wrong[k]));
        }
        require(std::sqrt(error / norm) < 1e-14, "half-cell Dirichlet wall reference mismatch");
        require(factorGap > 1e-6, "wall factor is not observable; half-cell term is not exercised");
        for(double v : wallSolver.u)
        {
            require(std::isfinite(v), "wall relaxation produced a non-finite value");
            require(v >= -1e-12 && v <= 1.0 + 1e-12, "wall relaxation overshot its physical bounds");
        }
        // The boundary row must lose more heat than the same row's centre cell
        // because of the extra wall flux.
        require(
            wallSolver.u[n * (n / 2) + 0] < wallSolver.u[n * (n / 2) + n / 2],
            "wall flux does not dominate the boundary row");
    }

    // Refinement of a heterogeneous (analytical alpha_true) solve: the only direct
    // exercise of heterogeneous face fluxes. Area-average the fine field onto the
    // coarse grid at a fixed tmax and require the difference to shrink under
    // refinement while staying finite.
    {
        auto const heterogeneousSolve = [](std::size_t grid, double tmax)
        {
            heatclosure::Config hc;
            hc.n = grid;
            hc.tmax = tmax;
            hc.steps = 0; // automatic stable count
            hc.uniformMaterial = false; // analytical alphaTrue(x,y,u,beta)
            heatclosure::Solver solver(hc);
            for(std::size_t y = 0; y < grid; ++y)
                for(std::size_t x = 0; x < grid; ++x)
                {
                    auto const cx = (static_cast<double>(x) + 0.5) / static_cast<double>(grid);
                    auto const cy = (static_cast<double>(y) + 0.5) / static_cast<double>(grid);
                    solver.u[y * grid + x]
                        = 0.5 + 0.2 * std::sin(3.14159265358979323846 * cx) * std::sin(3.14159265358979323846 * cy);
                }
            for(std::size_t s = 0; s < solver.steps; ++s)
                solver.step();
            return std::make_pair(solver.u, solver.alpha);
        };
        auto const areaAverage = [](std::vector<double> const& fine, std::size_t coarse)
        {
            auto const fineGrid = coarse * 2;
            std::vector<double> result(coarse * coarse, 0.0);
            for(std::size_t Y = 0; Y < coarse; ++Y)
                for(std::size_t X = 0; X < coarse; ++X)
                {
                    double sum = 0.0;
                    for(std::size_t dy = 0; dy < 2; ++dy)
                        for(std::size_t dx = 0; dx < 2; ++dx)
                            sum += fine[(2 * Y + dy) * fineGrid + (2 * X + dx)];
                    result[Y * coarse + X] = 0.25 * sum;
                }
            return result;
        };
        auto const relativeDifference = [](std::vector<double> const& coarse, std::vector<double> const& fineAveraged)
        {
            double numerator = 0.0, denominator = 0.0;
            for(std::size_t i = 0; i < coarse.size(); ++i)
            {
                numerator += (coarse[i] - fineAveraged[i]) * (coarse[i] - fineAveraged[i]);
                denominator += coarse[i] * coarse[i];
            }
            return std::sqrt(numerator / denominator);
        };
        auto const tmax = 0.002;
        auto const [u8, alpha8] = heterogeneousSolve(8, tmax);
        auto const [u16, alpha16] = heterogeneousSolve(16, tmax);
        auto const [u32, alpha32] = heterogeneousSolve(32, tmax);
        for(double v : u32)
            require(std::isfinite(v), "heterogeneous refinement produced a non-finite field");
        for(double v : alpha32)
            require(std::isfinite(v) && v > 0.0, "heterogeneous coefficient invalid");
        // alpha_true must actually vary spatially, otherwise heterogeneous face
        // fluxes are not being exercised.
        auto const [minAlpha, maxAlpha] = std::minmax_element(alpha32.begin(), alpha32.end());
        require(*maxAlpha - *minAlpha > 0.1, "heterogeneous case has an effectively uniform coefficient");
        auto const err1 = relativeDifference(u8, areaAverage(u16, 8));
        auto const err2 = relativeDifference(u16, areaAverage(u32, 16));
        require(std::isfinite(err1) && std::isfinite(err2), "heterogeneous refinement error is non-finite");
        require(err2 < err1, "heterogeneous refinement difference did not decrease");
        require(err1 > 1e-9, "heterogeneous refinement difference is suspiciously small");
    }

    std::cout << "strict solver checks passed\n";
}
catch(std::exception const& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
