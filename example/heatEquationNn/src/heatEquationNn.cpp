// SPDX-License-Identifier: MPL-2.0
#include "ConservativeSolver.hpp"
#include "ModelLoader.hpp"
#include "NeuralInference.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace
{
    std::string value(int& i, int argc, char** argv)
    {
        if(++i >= argc)
            throw std::invalid_argument("missing option value");
        return argv[i];
    }
} // namespace

int main(int argc, char** argv)
try
{
    heatclosure::Config c;
    std::string material = "preset", output = "results/heat_closure", weights = "models/heat_closure/weights.bin";
    std::size_t frames = 121;
    bool noOutput = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg == "--grid")
            c.n = std::stoull(value(i, argc, argv));
        else if(arg == "--tmax")
            c.tmax = std::stod(value(i, argc, argv));
        else if(arg == "--steps")
            c.steps = std::stoull(value(i, argc, argv));
        else if(arg == "--beta")
            c.beta = std::stod(value(i, argc, argv));
        else if(arg == "--alpha-max")
            c.alphaMax = std::stod(value(i, argc, argv));
        else if(arg == "--alpha-min")
            c.alphaMin = std::stod(value(i, argc, argv));
        else if(arg == "--material")
            material = value(i, argc, argv);
        else if(arg == "--weights")
            weights = value(i, argc, argv);
        else if(arg == "--output")
            output = value(i, argc, argv);
        else if(arg == "--frames")
            frames = std::stoull(value(i, argc, argv));
        else if(arg == "--no-output")
            noOutput = true;
        else if(arg == "--validate-strict")
            c.validateStrict = true;
        else if(arg == "--help")
        {
            std::cout << "--grid N --tmax T --steps N --material uniform|preset|nn --beta B --alpha-min A --alpha-max "
                         "A --weights FILE --output DIR --frames N --no-output --validate-strict\n";
            return 0;
        }
        else
            throw std::invalid_argument("unknown option: " + arg);
    }
    if(material != "uniform" && material != "preset" && material != "nn")
        throw std::invalid_argument("invalid material");
    c.uniformMaterial = material == "uniform";
    if(frames < 2)
        throw std::invalid_argument("frames must be >= 2");
    heatclosure::Solver solver(c);
    heatclosure::Model model;
    auto selector = alpaka::onHost::makeDeviceSelector(
        alpaka::onHost::makeDeviceSpec(alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu}));
    if(material == "nn")
    {
        if(!selector.isAvailable())
            throw std::runtime_error("host alpaka device unavailable");
        model = heatclosure::loadModel(weights, c.beta);
        if(model.alphaMin != c.alphaMin || model.alphaMax != c.alphaMax)
            throw std::invalid_argument("CLI alpha bounds must match model metadata");
    }
    auto inferAlpha = [&]()
    {
        std::vector<std::array<double, 3>> features;
        features.reserve(solver.u.size());
        for(std::size_t y = 0; y < c.n; ++y)
            for(std::size_t x = 0; x < c.n; ++x)
                features.push_back({solver.u[solver.index(x, y)], (x + 0.5) * solver.dx, (y + 0.5) * solver.dx});
        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();
        auto values = heatclosure::infer(queue, alpaka::exec::cpuSerial, device, model, features);
        std::vector<double> a(values.begin(), values.end());
        for(double v : a)
        {
            if(!(v >= model.alphaMin && v <= model.alphaMax) || !std::isfinite(v))
                throw std::runtime_error("NN produced invalid alpha");
        }
        return a;
    };
    if(!noOutput)
    {
        std::filesystem::create_directories(output);
        std::ofstream manifest(output + "/manifest.csv");
        manifest << "frame,time,step,material,grid,beta\n";
        auto save = [&](std::size_t frame, std::size_t step)
        {
            double time = step * solver.dt;
            std::ostringstream name;
            name << output << "/frame_" << std::setw(6) << std::setfill('0') << frame << ".csv";
            std::ofstream f(name.str());
            f << "x,y,u\n" << std::setprecision(17);
            for(std::size_t y = 0; y < c.n; ++y)
                for(std::size_t x = 0; x < c.n; ++x)
                    f << (x + 0.5) * solver.dx << ',' << (y + 0.5) * solver.dx << ',' << solver.u[solver.index(x, y)]
                      << '\n';
            manifest << frame << ',' << std::setprecision(17) << time << ',' << step << ',' << material << ',' << c.n
                     << ',' << c.beta << '\n';
        };
        save(0, 0);
        std::size_t frame = 1;
        for(std::size_t s = 1; s <= solver.steps; ++s)
        {
            if(material == "nn")
                solver.stepWithAlpha(inferAlpha());
            else
                solver.step();
            auto target = std::min(solver.steps, (frame * (solver.steps - 1) + (frames - 2) / 2) / (frames - 1));
            if(frame < frames - 1 && s >= target)
            {
                save(frame, s);
                ++frame;
            }
        }
        while(frame < frames)
        {
            save(frame, solver.steps);
            ++frame;
        }
    }
    else
        for(std::size_t s = 0; s < solver.steps; ++s)
        {
            if(material == "nn")
                solver.stepWithAlpha(inferAlpha());
            else
                solver.step();
        }
    double minU = *std::min_element(solver.u.begin(), solver.u.end());
    double maxU = *std::max_element(solver.u.begin(), solver.u.end());
    std::cout << "material=" << material << " grid=" << c.n << " steps=" << solver.steps << " dt=" << solver.dt
              << " final_range=[" << minU << ',' << maxU << "]\n";
    if(!std::isfinite(minU) || !std::isfinite(maxU) || (c.validateStrict && (minU < -1e-10 || maxU > 1.0 + 1e-10)))
        throw std::runtime_error("temperature violates finite-value/maximum-principle check");
    return 0;
}
catch(std::exception const& e)
{
    std::cerr << "heatEquationNn: " << e.what() << '\n';
    return 2;
}
