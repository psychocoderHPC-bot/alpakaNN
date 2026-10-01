// SPDX-License-Identifier: MPL-2.0
#include "DeviceSolver.hpp"
#include "NeuralInference.hpp"

#include <alpaka/alpaka.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

void require(bool condition, char const* message)
{
    if(!condition)
        throw std::runtime_error(message);
}

double sum(std::vector<double> const& values)
{
    return std::accumulate(values.begin(), values.end(), 0.0);
}

namespace
{
    int runCommand(std::string const& command)
    {
        return std::system(command.c_str());
    }

    std::string quote(std::filesystem::path const& path)
    {
        return "\"" + path.string() + "\"";
    }

    /** Read a streaming `x,y,u,alpha,mode,time,step` export, validating the header. */
    std::vector<std::array<std::string, 7>> readAlphaCsv(std::filesystem::path const& path)
    {
        std::ifstream file(path);
        require(static_cast<bool>(file), "cannot open exported alpha CSV");
        std::string header;
        std::getline(file, header);
        require(header == "x,y,u,alpha,mode,time,step", "alpha CSV header mismatch");
        std::vector<std::array<std::string, 7>> rows;
        std::string line;
        while(std::getline(file, line))
        {
            if(line.empty())
                continue;
            std::array<std::string, 7> columns;
            std::size_t start = 0;
            std::size_t count = 0;
            while(count < columns.size())
            {
                auto const comma = line.find(',', start);
                auto const end = comma == std::string::npos ? line.size() : comma;
                columns[count++] = line.substr(start, end - start);
                if(end == line.size())
                    break;
                start = end + 1;
            }
            require(count == columns.size(), "alpha CSV row has the wrong column count");
            rows.push_back(columns);
        }
        return rows;
    }

    std::string readFile(std::filesystem::path const& path)
    {
        std::ifstream file(path);
        require(static_cast<bool>(file), "cannot read command output");
        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    /** Extract the numeric value following `key` in a report line. */
    double fieldValue(std::string const& text, std::string const& key)
    {
        auto const position = text.find(key);
        require(position != std::string::npos, "compute-only report is missing a field");
        auto const begin = position + key.size();
        auto const end = text.find_first_of(" \n", begin);
        return std::stod(text.substr(begin, end - begin));
    }
} // namespace

struct FakeDevice
{
    int id;
    friend bool operator==(FakeDevice const&, FakeDevice const&) = default;
};

struct FakeQueue
{
    FakeDevice device;

    FakeDevice const& getDevice() const
    {
        return device;
    }
};

int main()
try
{
    // Exercise the exact production queue/device predicate without depending on
    // the host selector exposing two physical devices.
    FakeQueue matchingQueue{{1}};
    require(
        &heatclosure::device_detail::checkedQueue(matchingQueue, FakeDevice{1}) == &matchingQueue,
        "matching queue/device was rejected");
    bool mismatchedDeviceRejected = false;
    try
    {
        (void) heatclosure::device_detail::checkedQueue(matchingQueue, FakeDevice{2});
    }
    catch(std::invalid_argument const&)
    {
        mismatchedDeviceRejected = true;
    }
    require(mismatchedDeviceRejected, "queue associated with a different device was accepted");

    auto selector = alpaka::onHost::makeDeviceSelector(
        alpaka::onHost::makeDeviceSpec(alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu}));
    require(selector.isAvailable(), "host device unavailable");
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    constexpr std::size_t n = 6;
    heatclosure::Config c;
    c.n = n;
    c.tmax = 0.01;
    c.steps = 20;
    std::vector<double> initial(n * n);
    for(std::size_t y = 0; y < n; ++y)
        for(std::size_t x = 0; x < n; ++x)
            initial[y * n + x] = 0.1 + 0.03 * static_cast<double>(x) + 0.02 * static_cast<double>(y);
    heatclosure::DeviceSolver solver(queue, device, c, heatclosure::CoefficientMode::uniform, initial);
    auto expected = initial;
    auto const dx = 1.0 / static_cast<double>(n);
    auto const inv = 1.0 / (dx * dx);
    auto const a = 0.5;
    for(std::size_t y = 0; y < n; ++y)
        for(std::size_t x = 0; x < n; ++x)
        {
            auto const k = y * n + x;
            double rhs = 0.0;
            rhs += x + 1 < n ? a * (initial[k + 1] - initial[k]) * inv : 2.0 * a * (0.0 - initial[k]) * inv;
            rhs += x > 0 ? -a * (initial[k] - initial[k - 1]) * inv : 2.0 * a * (1.0 - initial[k]) * inv;
            if(y + 1 < n)
                rhs += a * (initial[k + n] - initial[k]) * inv;
            if(y > 0)
                rhs -= a * (initial[k] - initial[k - n]) * inv;
            expected[k] += solver.dt * rhs;
        }
    solver.step(queue, alpaka::exec::cpuSerial);
    auto got = solver.snapshot(queue);
    double error = 0.0;
    for(std::size_t i = 0; i < got.size(); ++i)
        error = std::max(error, std::abs(got[i] - expected[i]));
    require(error < 1e-13, "uniform stencil disagrees with independent reference");

    // Repeated swaps must continue to agree with the independent host implementation.
    auto hostConfig = c;
    hostConfig.uniformMaterial = true;
    heatclosure::Solver hostSolver(hostConfig);
    hostSolver.u = initial;
    heatclosure::DeviceSolver repeated(queue, device, c, heatclosure::CoefficientMode::uniform, initial);
    for(int step = 0; step < 5; ++step)
    {
        repeated.step(queue, alpaka::exec::cpuSerial);
        hostSolver.step();
        auto const deviceState = repeated.snapshot(queue);
        for(std::size_t i = 0; i < deviceState.size(); ++i)
            require(std::abs(deviceState[i] - hostSolver.u[i]) < 1e-13, "multi-step ping-pong mismatch");
    }

    auto otherQueue = device.makeQueue();
    bool wrongQueueRejected = false;
    try
    {
        (void) repeated.snapshot(otherQueue);
    }
    catch(std::invalid_argument const&)
    {
        wrongQueueRejected = true;
    }
    require(wrongQueueRejected, "snapshot on a different queue was accepted");

    heatclosure::Config equilibrium = c;
    equilibrium.leftWall = equilibrium.rightWall = 0.3;
    std::fill(initial.begin(), initial.end(), 0.3);
    heatclosure::DeviceSolver eq(queue, device, equilibrium, heatclosure::CoefficientMode::uniform, initial);
    eq.step(queue, alpaka::exec::cpuSerial);
    for(double value : eq.snapshot(queue))
        require(std::abs(value - 0.3) < 1e-13, "equilibrium drift");

    heatclosure::Config closed = c;
    closed.insulatedX = true;
    for(std::size_t i = 0; i < initial.size(); ++i)
        initial[i] = 0.2 + 0.01 * static_cast<double>(i % 11);
    auto before = sum(initial);
    heatclosure::DeviceSolver box(queue, device, closed, heatclosure::CoefficientMode::uniform, initial);
    box.step(queue, alpaka::exec::cpuSerial);
    require(std::abs(sum(box.snapshot(queue)) - before) < 1e-12, "closed-box conservation");

    heatclosure::Config driven = c;
    std::fill(initial.begin(), initial.end(), 0.2);
    double boundary = 0.0;
    for(std::size_t y = 0; y < n; ++y)
        boundary += 2.0 * a * (1.0 - 0.2) * inv + 2.0 * a * (0.0 - 0.2) * inv;
    heatclosure::DeviceSolver wall(queue, device, driven, heatclosure::CoefficientMode::uniform, initial);
    wall.step(queue, alpaka::exec::cpuSerial);
    require(
        std::abs(sum(wall.snapshot(queue)) - n * n * 0.2 - wall.dt * boundary) < 1e-12,
        "driven-wall flux balance");

    bool unstableRejected = false;
    try
    {
        auto unstable = c;
        unstable.steps = 1;
        heatclosure::DeviceSolver bad(queue, device, unstable, heatclosure::CoefficientMode::uniform, initial);
    }
    catch(std::invalid_argument const&)
    {
        unstableRejected = true;
    }
    require(unstableRejected, "unstable explicit count accepted");

    bool overflowRejected = false;
    try
    {
        auto enormous = c;
        enormous.tmax = std::numeric_limits<double>::max();
        heatclosure::DeviceSolver bad(queue, device, enormous, heatclosure::CoefficientMode::uniform, initial);
    }
    catch(std::invalid_argument const&)
    {
        overflowRejected = true;
    }
    require(overflowRejected, "out-of-range required step count accepted");

    bool wrongStepQueueRejected = false;
    try
    {
        repeated.step(otherQueue, alpaka::exec::cpuSerial);
    }
    catch(std::invalid_argument const&)
    {
        wrongStepQueueRejected = true;
    }
    require(wrongStepQueueRejected, "step on a different queue was accepted");

    heatclosure::Config preset = c;
    std::fill(initial.begin(), initial.end(), 0.0);
    heatclosure::DeviceSolver presetSolver(queue, device, preset, heatclosure::CoefficientMode::preset, initial);
    presetSolver.step(queue, alpaka::exec::cpuSerial);
    auto presetState = presetSolver.snapshot(queue);
    auto presetAlpha = presetSolver.coefficientSnapshot(queue, alpaka::exec::cpuSerial);
    for(std::size_t y = 0; y < n; ++y)
        for(std::size_t x = 0; x < n; ++x)
        {
            auto const k = y * n + x;
            auto const expectedAlpha
                = heatclosure::baseAlpha((x + 0.5) * dx, (y + 0.5) * dx) * (1.0 + preset.beta * presetState[k]);
            require(std::isfinite(presetAlpha[k]) && presetAlpha[k] > 0.0, "preset coefficient invalid");
            require(std::abs(presetAlpha[k] - expectedAlpha) < 1e-14, "preset coefficient mismatch");
        }

    auto const modelPath = std::string{HEAT_CLOSURE_MODEL_DIR} + "/weights.bin";
    auto model = heatclosure::loadModel(modelPath, c.beta);
    auto nnConfig = c;
    nnConfig.alphaMin = model.alphaMin;
    nnConfig.alphaMax = model.alphaMax;
    heatclosure::DeviceSolver nnSolver(queue, device, nnConfig, heatclosure::CoefficientMode::neural, initial, &model);
    nnSolver.step(queue, alpaka::exec::cpuSerial);
    auto nnState = nnSolver.snapshot(queue);
    auto neuralAlpha = nnSolver.coefficientSnapshot(queue, alpaka::exec::cpuSerial);
    std::vector<std::array<double, 3>> features(nnState.size());
    for(std::size_t i = 0; i < nnState.size(); ++i)
        features[i] = {nnState[i], ((i % n) + 0.5) / n, ((i / n) + 0.5) / n};
    auto expectedNeuralAlpha = heatclosure::infer(queue, alpaka::exec::cpuSerial, device, model, features);
    for(std::size_t i = 0; i < neuralAlpha.size(); ++i)
        require(std::abs(neuralAlpha[i] - expectedNeuralAlpha[i]) < 2e-6, "NN post-step coefficient parity mismatch");
    for(double value : neuralAlpha)
        require(std::isfinite(value) && value >= model.alphaMin && value <= model.alphaMax, "NN coefficient invalid");

    // The device step path must not copy anything to the host; only explicit
    // snapshots may synchronize.
    require(!heatclosure::DeviceSolver<decltype(queue), decltype(device)>::perStepHostCopy, "step path performs a D2H copy");

    // Exercise the CLI `--export-alpha` / compute-only timing paths end-to-end for
    // the preset and neural coefficient modes on the host backend.
    auto const base = std::filesystem::temp_directory_path()
                      / ("heatEquationNn_export_" + std::to_string(static_cast<unsigned long long>(std::rand())));
    auto const presetDir = base / "preset";
    auto const nnDir = base / "nn";
    std::filesystem::create_directories(base);
    auto const grid = std::string{"16"};
    auto const tmax = std::string{"0.01"};

    // preset: exported alpha must equal alphaTrue(x,y,u) for the exported u.
    auto const presetCsv = base / "preset_alpha.csv";
    auto const presetCommand = quote(HEAT_CLOSURE_EXAMPLE_BIN) + " --backend host --material preset --grid " + grid
                               + " --tmax " + tmax + " --frames 3 --output " + quote(presetDir)
                               + " --export-alpha " + quote(presetCsv) + " > " + quote(base / "preset.log") + " 2>&1";
    require(runCommand(presetCommand) == 0, "preset export-alpha run failed");
    auto const presetRows = readAlphaCsv(presetCsv);
    require(presetRows.size() == 3 * static_cast<std::size_t>(16 * 16), "preset export should hold three frames");
    for(auto const& row : presetRows)
    {
        require(row[4] == "preset", "preset export carries the wrong mode");
        auto const x = std::stod(row[0]);
        auto const y = std::stod(row[1]);
        auto const u = std::stod(row[2]);
        auto const alpha = std::stod(row[3]);
        auto const expected = heatclosure::alphaTrue(x, y, u, 0.5);
        require(std::abs(alpha - expected) < 1e-14, "preset export alpha does not match alphaTrue");
    }

    // nn: exported alpha must stay within the model bounds and use the nn mode.
    auto const nnCsv = base / "nn_alpha.csv";
    auto const nnCommand = quote(HEAT_CLOSURE_EXAMPLE_BIN) + " --backend host --material nn --grid " + grid + " --tmax "
                           + tmax + " --frames 3 --weights " + quote(modelPath) + " --output " + quote(nnDir)
                           + " --export-alpha " + quote(nnCsv) + " > " + quote(base / "nn.log") + " 2>&1";
    require(runCommand(nnCommand) == 0, "nn export-alpha run failed");
    auto const nnRows = readAlphaCsv(nnCsv);
    require(nnRows.size() == 3 * static_cast<std::size_t>(16 * 16), "nn export should hold three frames");
    for(auto const& row : nnRows)
    {
        require(row[4] == "nn", "nn export carries the wrong mode");
        auto const alpha = std::stod(row[3]);
        require(std::isfinite(alpha) && alpha >= model.alphaMin && alpha <= model.alphaMax, "exported NN alpha out of bounds");
    }

    // compute-only timing: `--no-output` must still exit 0 and report steps plus
    // the no-per-step-D2H property.
    auto const timingCommand = quote(HEAT_CLOSURE_EXAMPLE_BIN) + " --backend host --material nn --grid " + grid
                               + " --tmax " + tmax + " --weights " + quote(modelPath) + " --no-output > "
                               + quote(base / "timing.log") + " 2>&1";
    require(runCommand(timingCommand) == 0, "compute-only timing run failed");
    auto const timingReport = readFile(base / "timing.log");
    require(timingReport.find("compute_only=1") != std::string::npos, "compute-only report missing");
    require(timingReport.find("nn_per_step_d2h=no") != std::string::npos, "NN path reported a per-step D2H copy");
    require(fieldValue(timingReport, "total_seconds=") >= 0.0, "compute-only total time is invalid");
    require(fieldValue(timingReport, "seconds_per_step=") >= 0.0, "compute-only seconds-per-step is invalid");
    require(fieldValue(timingReport, "steps=") > 0.0, "compute-only step count is invalid");

    std::filesystem::remove_all(base);
    std::cout << "device solver checks passed\n";
}
catch(std::exception const& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
