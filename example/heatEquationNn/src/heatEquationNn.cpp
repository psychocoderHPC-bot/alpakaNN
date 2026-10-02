// SPDX-License-Identifier: MPL-2.0
#include "DeviceSolver.hpp"
#include "ModelLoader.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace
{
    std::string value(int& i, int argc, char** argv)
    {
        if(++i >= argc)
            throw std::invalid_argument("missing option value");
        return argv[i];
    }

    std::size_t parseCount(std::string const& text, char const* option)
    {
        std::size_t used = 0;
        auto const result = std::stoull(text, &used);
        if(used != text.size() || text.empty() || text.front() == '-')
            throw std::invalid_argument(std::string(option) + " must be a non-negative integer");
        if(result > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument(std::string(option) + " is out of range");
        return static_cast<std::size_t>(result);
    }

    double parseReal(std::string const& text, char const* option)
    {
        std::size_t used = 0;
        auto const result = std::stod(text, &used);
        if(used != text.size() || !std::isfinite(result))
            throw std::invalid_argument(std::string(option) + " must be a finite number");
        return result;
    }

    void writeCsv(std::filesystem::path const& path, std::vector<double> const& u, std::size_t n)
    {
        std::ofstream file(path);
        if(!file)
            throw std::runtime_error("cannot create snapshot: " + path.string());
        file << "x,y,u\n" << std::setprecision(17);
        auto const dx = 1.0 / static_cast<double>(n);
        for(std::size_t y = 0; y < n; ++y)
            for(std::size_t x = 0; x < n; ++x)
                file << (static_cast<double>(x) + 0.5) * dx << ',' << (static_cast<double>(y) + 0.5) * dx << ','
                     << u[y * n + x] << '\n';
        if(!file)
            throw std::runtime_error("failed writing snapshot: " + path.string());
    }

    char const* materialName(heatclosure::CoefficientMode mode)
    {
        return mode == heatclosure::CoefficientMode::neural
                   ? "nn"
                   : (mode == heatclosure::CoefficientMode::preset ? "preset" : "uniform");
    }

    /** Streaming `x,y,u,alpha,mode,time,step` coefficient export.
     *
     * A single CSV collects every snapshot frame (`--frames` times) in file order.
     * The `u` and `alpha` columns always describe the same instantaneous state.
     */
    class AlphaCsvWriter
    {
    public:
        AlphaCsvWriter(std::filesystem::path const& path, std::size_t n) : m_file(path), m_n(n)
        {
            if(!m_file)
                throw std::runtime_error("cannot create alpha export: " + path.string());
            m_file << "x,y,u,alpha,mode,time,step\n" << std::setprecision(17);
            if(!m_file)
                throw std::runtime_error("failed writing alpha export header: " + path.string());
        }

        void frame(
            std::vector<double> const& u,
            std::vector<double> const& alpha,
            heatclosure::CoefficientMode mode,
            double time,
            std::size_t step)
        {
            if(u.size() != m_n * m_n || alpha.size() != m_n * m_n)
                throw std::invalid_argument("alpha export requires aligned u/alpha fields");
            auto const dx = 1.0 / static_cast<double>(m_n);
            auto const name = materialName(mode);
            for(std::size_t y = 0; y < m_n; ++y)
                for(std::size_t x = 0; x < m_n; ++x)
                    m_file << (static_cast<double>(x) + 0.5) * dx << ',' << (static_cast<double>(y) + 0.5) * dx << ','
                           << u[y * m_n + x] << ',' << alpha[y * m_n + x] << ',' << name << ',' << time << ',' << step
                           << '\n';
            if(!m_file)
                throw std::runtime_error("failed writing alpha export");
        }

    private:
        std::ofstream m_file;
        std::size_t m_n;
    };

    void dumpFeatures(
        std::filesystem::path const& path,
        std::size_t count,
        std::size_t seed,
        std::size_t temperatureSamples,
        double beta)
    {
        if(count == 0 || temperatureSamples < 2)
            throw std::invalid_argument("feature sample count must be positive and temperature samples >= 2");
        std::ofstream file(path);
        if(!file)
            throw std::runtime_error("cannot create feature dump: " + path.string());
        file << "u,x,y,alpha_true\n" << std::setprecision(17);
        std::mt19937_64 random(static_cast<std::mt19937_64::result_type>(seed));
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        auto const gridCount = count / 2;
        auto emit = [&](double u, double x, double y)
        { file << u << ',' << x << ',' << y << ',' << heatclosure::alphaTrue(x, y, u, beta) << '\n'; };
        for(std::size_t i = 0; i < gridCount; ++i)
        {
            auto const x = (static_cast<double>(i % 256) + 0.5) / 256.0;
            auto const y = (static_cast<double>((i / 256) % 256) + 0.5) / 256.0;
            auto const t = i % temperatureSamples;
            emit(static_cast<double>(t) / static_cast<double>(temperatureSamples - 1), x, y);
        }
        // The remaining samples are reproducible random domain points, with half
        // concentrated immediately around the circular and conductor interfaces.
        for(std::size_t i = gridCount; i < count; ++i)
        {
            double x = unit(random), y = unit(random);
            switch((i - gridCount) % 4)
            {
            case 0:
                {
                    auto const theta = unit(random) * 2.0 * std::acos(-1.0);
                    auto const radius = 0.12 + (unit(random) < 0.5 ? -1.0 : 1.0) * 1.0e-5;
                    x = std::clamp(0.35 + radius * std::cos(theta), 0.0, 1.0);
                    y = std::clamp(0.5 + radius * std::sin(theta), 0.0, 1.0);
                    break;
                }
            case 1:
                x = 0.45 + (unit(random) < 0.5 ? -1.0 : 1.0) * 1.0e-5;
                y = 0.45 + 0.1 * unit(random);
                break;
            case 2:
                x = 0.65 + (unit(random) < 0.5 ? -1.0 : 1.0) * 1.0e-5;
                y = 0.45 + 0.1 * unit(random);
                break;
            default:
                x = 0.45 + 0.2 * unit(random);
                y = 0.45 + (unit(random) < 0.5 ? -1.0 : 1.0) * 1.0e-5;
                break;
            }
            auto const t = i % temperatureSamples;
            emit(static_cast<double>(t) / static_cast<double>(temperatureSamples - 1), x, y);
        }
        if(!file)
            throw std::runtime_error("failed writing feature dump: " + path.string());
        std::ofstream metadata(path.string() + ".metadata.json");
        metadata << "{\n  \"beta\": " << beta
                 << ",\n  \"alpha_min\": 0.01,\n  \"alpha_max\": 6.0,\n"
                    "  \"seed\": "
                 << seed << ",\n  \"sample_count\": " << count
                 << ",\n  \"temperature_samples\": " << temperatureSamples
                 << ",\n  \"sampling\": \"uniform temperature levels; regular domain samples plus seeded "
                    "interface-biased samples\"\n}\n";
        if(!metadata)
            throw std::runtime_error("failed writing feature metadata");
    }

    /** True when this translation unit was compiled with an OpenMP runtime. */
#if defined(_OPENMP)
    constexpr bool ompCompiled = true;
#else
    constexpr bool ompCompiled = false;
#endif

    /** Pick the executor that matches the selected alpaka API.
     *
     * The host path keeps `cpuSerial` so `--backend host` stays byte-for-byte equivalent
     * to the previous default; the accelerator paths use the matching GPU executor.
     * The separate `omp` selector (handled by dispatchBackend) maps the host device to
     * `cpuOmpBlocks`, keeping the default host result unchanged.
     */
    template<class TDevice>
    constexpr auto executorFor(TDevice const& device)
    {
        using Api = std::decay_t<decltype(device.getApi())>;
        if constexpr(std::is_same_v<Api, alpaka::api::Cuda>)
            return alpaka::exec::gpuCuda;
        else if constexpr(std::is_same_v<Api, alpaka::api::Hip>)
            return alpaka::exec::gpuHip;
        else if constexpr(std::is_same_v<Api, alpaka::api::OneApi>)
            return alpaka::exec::oneApi;
        else
            return alpaka::exec::cpuSerial;
    }

    template<class TDevice, class TExec>
    int run(
        TDevice device,
        TExec exec,
        std::string const& backendName,
        heatclosure::Config const& config,
        heatclosure::CoefficientMode mode,
        heatclosure::Model const* model,
        std::size_t frames,
        bool noOutput,
        bool strict,
        std::filesystem::path const& output,
        std::filesystem::path const& alphaExport)
    {
        // The compute-only timing path uses queue events so accelerator work is
        // measured without host-side synchronization inside the loop. Host timing
        // is enabled too, so a plain steady_clock fallback is unnecessary.
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking, alpaka::timing::enabled);
        std::vector<double> initial(config.n * config.n, 0.0);
        heatclosure::DeviceSolver solver(queue, device, config, mode, initial, model);
        auto const exportAlpha = !alphaExport.empty();
        if(noOutput && exportAlpha)
            throw std::invalid_argument("--export-alpha cannot be combined with --no-output");
        std::ofstream manifest;
        std::optional<AlphaCsvWriter> alphaWriter;
        if(!noOutput)
        {
            std::filesystem::create_directories(output);
            manifest.open(output / "manifest.csv");
            if(!manifest)
                throw std::runtime_error("cannot create output manifest");
            manifest << "frame,time,step,material,grid,beta\n";
            if(exportAlpha)
                alphaWriter.emplace(alphaExport, config.n);
        }
        std::size_t frame = 0;
        auto save = [&](std::size_t step)
        {
            auto const state = solver.snapshot(queue);
            auto const time = static_cast<double>(step) * solver.dt;
            if(!noOutput)
            {
                std::ostringstream name;
                name << "frame_" << std::setw(6) << std::setfill('0') << frame << ".csv";
                writeCsv(output / name.str(), state, config.n);
                if(alphaWriter)
                {
                    // coefficientSnapshot recomputes alpha from the current device
                    // state, so the exported alpha matches this exact u snapshot.
                    auto const alpha = solver.coefficientSnapshot(queue, exec);
                    alphaWriter->frame(state, alpha, mode, time, step);
                }
                manifest << frame << ',' << std::setprecision(17) << time << ',' << step << ',' << materialName(mode)
                         << ',' << config.n << ',' << config.beta << '\n';
                if(!manifest)
                    throw std::runtime_error("failed writing output manifest");
            }
            else if(alphaWriter)
            {
                auto const alpha = solver.coefficientSnapshot(queue, exec);
                alphaWriter->frame(state, alpha, mode, time, step);
            }
            ++frame;
        };
        // The initial state is saved (and its coefficient computed) before the
        // first update so the export includes the t=0 field.
        save(0);
        auto const start = queue.makeEvent();
        queue.enqueue(start);
        std::size_t nextFrame = 1;
        auto const outputs = frames > 1 ? frames - 1 : 1;
        for(std::size_t step = 1; step <= solver.steps; ++step)
        {
            solver.step(queue, exec);
            if(!noOutput)
            {
                while(nextFrame < frames)
                {
                    auto const targetTime
                        = config.tmax * static_cast<double>(nextFrame) / static_cast<double>(outputs);
                    auto const targetStep = static_cast<std::size_t>(std::ceil(targetTime / solver.dt));
                    if(step < targetStep)
                        break;
                    save(step);
                    ++nextFrame;
                }
            }
        }
        auto const end = queue.makeEvent();
        queue.enqueue(end);
        alpaka::onHost::wait(queue);
        std::optional<double> computeSeconds;
        if constexpr(std::same_as<ALPAKA_TYPEOF(end.getTiming()), alpaka::timing::Enabled>)
            computeSeconds = std::chrono::duration<double>(alpaka::onHost::getElapsedTime(start, end)).count();
        auto finalState = solver.snapshot(queue);
        if(!noOutput && frame < frames)
        {
            while(frame < frames)
                save(solver.steps);
            finalState = solver.snapshot(queue);
        }
        auto const [minIt, maxIt] = std::minmax_element(finalState.begin(), finalState.end());
        auto const minU = *minIt, maxU = *maxIt;
        if(!std::isfinite(minU) || !std::isfinite(maxU) || (strict && (minU < -1e-10 || maxU > 1.0 + 1e-10)))
            throw std::runtime_error("temperature violates finite-value/maximum-principle check");
        std::cout << "backend=" << backendName << " material=" << materialName(mode) << " grid=" << config.n
                  << " steps=" << solver.steps << " dt=" << solver.dt << " final_range=[" << minU << ',' << maxU
                  << "]\n";
        if(noOutput)
        {
            std::cout << std::setprecision(17);
            if(computeSeconds)
            {
                auto const total = *computeSeconds;
                std::cout << "compute_only=1 total_seconds=" << total
                          << " seconds_per_step=" << total / static_cast<double>(solver.steps)
                          << " steps=" << solver.steps << '\n';
            }
            else
                std::cout << "compute_only=1 total_seconds=unavailable steps=" << solver.steps << '\n';
            std::cout << "nn_per_step_d2h=" << (solver.perStepHostCopy ? "yes" : "no") << " snapshots_d2h=yes\n";
        }
        return 0;
    }

    struct BackendSelection
    {
        /// Canonical lowercase API name, or empty for a numeric accelerator index.
        std::string name;
        /// Device index within the selected device specification.
        std::uint32_t deviceIndex = 0;
        /// A bare device index addresses the first available accelerator.
        bool firstAccelerator = false;
        /// The `omp` selector uses the host device with the OpenMP executor.
        bool omp = false;
    };

    bool isNumeric(std::string const& text)
    {
        return !text.empty()
               && std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
    }

    /** Resolve the `--backend` selector into a canonical name plus device index.
     *
     * Accepted selectors: `host`, `omp`, `hip`/`amd`/`amdgpu`, `cuda`,
     * `oneapi`/`sycl`, and bare `0`-based accelerator device indices. A selector
     * that is not compiled into this build is rejected by the caller with the
     * exact missing build option.
     */
    BackendSelection resolveBackend(std::string const& selector)
    {
        if(selector == "host")
            return {"host", 0, false};
        if(selector == "omp")
            return {"omp", 0, false, true};
        if(selector == "hip" || selector == "amd" || selector == "amdgpu")
            return {"hip", 0, false};
        if(selector == "cuda")
            return {"cuda", 0, false};
        if(selector == "oneapi" || selector == "sycl")
            return {"oneapi", 0, false};
        if(isNumeric(selector))
        {
            auto const index = std::stoull(selector);
            if(index > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("device index out of range for --backend: " + selector);
            return {"", static_cast<std::uint32_t>(index), true};
        }
        throw std::invalid_argument(
            "unknown --backend selector '" + selector
            + "' (expected host, omp, hip, cuda, oneapi, or a device index)");
    }

    std::string lowerName(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    /** Dispatch a resolved selection to a concrete compiled device specification.
     *
     * Iterates the alpaka device specifications enabled at compile time (host is
     * always present), matches the requested canonical name or accelerator index,
     * checks availability, and constructs the device/queue pair passed to
     * `DeviceSolver`. Unavailable or unknown requests throw an actionable error.
     */
    template<class TFunction>
    int dispatchBackend(BackendSelection const& selection, TFunction&& launch)
    {
        if constexpr(!ompCompiled)
        {
            if(selection.omp)
                throw std::invalid_argument(
                    "--backend omp is not compiled in: this build has no OpenMP runtime "
                    "(configure with -Dalpaka_DEP_OMP=ON and an OpenMP-enabled compiler)");
        }
        bool matched = false;
        bool compiledButUnavailable = false;
        std::string available;
        int result = 2;
        std::apply(
            [&](auto... spec)
            {
                // Each spec is visited exactly once; the fold short-circuits via `matched`.
                (
                    [&]
                    {
                        auto const apiName = lowerName(spec.getApi().getName());
                        if(available.find(apiName) == std::string::npos)
                        {
                            if(!available.empty())
                                available += ", ";
                            available += apiName;
                        }
                        if(matched)
                            return;
                        auto const isHost = spec.getApi() == alpaka::api::host;
                        // Named selectors match one exact API; a numeric selector takes the
                        // first available accelerator (never host, which is always present).
                        // The `omp` selector targets the host device with the OpenMP executor.
                        auto const nameMatches
                            = selection.omp ? isHost
                                            : (selection.firstAccelerator ? !isHost : (selection.name == apiName));
                        if(!nameMatches)
                            return;
                        auto selector = alpaka::onHost::makeDeviceSelector(spec);
                        if(!selector.isAvailable())
                        {
                            compiledButUnavailable = true;
                            return;
                        }
                        if(selection.deviceIndex >= selector.getDeviceCount())
                            throw std::invalid_argument(
                                "--backend "
                                + (selection.name.empty() ? std::to_string(selection.deviceIndex) : selection.name)
                                + ": device index out of range (available: "
                                + std::to_string(selector.getDeviceCount()) + ")");
                        auto device = selector.makeDevice(selection.deviceIndex);
#if defined(_OPENMP)
                        if(selection.omp)
                            result = launch(device, alpaka::exec::cpuOmpBlocks, selection.name);
                        else
#endif
                            result = launch(device, executorFor(device), apiName);
                        matched = true;
                    }(),
                    ...);
            },
            alpaka::onHost::enabledDeviceSpecs);
        if(matched)
            return result;
        auto const requested = selection.name.empty() ? std::to_string(selection.deviceIndex) : selection.name;
        if(compiledButUnavailable)
            throw std::runtime_error(
                "--backend " + requested
                + ": the backend is compiled in but no device is available at runtime "
                  "(compiled backends: "
                + available + ")");
        throw std::invalid_argument(
            "--backend " + requested + " is not compiled in (compiled backends: " + available
            + "); rebuild with the matching alpaka_DEP_* and device-kind options to enable it");
    }
} // namespace

int main(int argc, char** argv)
try
{
    heatclosure::Config config;
    std::string material = "preset", backend = "host", weights;
    std::filesystem::path output = "results/heat_closure";
    std::filesystem::path featurePath;
    std::filesystem::path alphaExport;
    std::size_t frames = 121, sampleCount = 100000, seed = 0, temperatureSamples = 11;
    bool noOutput = false;
    for(int i = 1; i < argc; ++i)
    {
        auto const arg = std::string{argv[i]};
        if(arg == "--grid")
            config.n = parseCount(value(i, argc, argv), "--grid");
        else if(arg == "--tmax")
            config.tmax = parseReal(value(i, argc, argv), "--tmax");
        else if(arg == "--steps")
            config.steps = parseCount(value(i, argc, argv), "--steps");
        else if(arg == "--beta")
            config.beta = parseReal(value(i, argc, argv), "--beta");
        else if(arg == "--alpha-max")
            config.alphaMax = parseReal(value(i, argc, argv), "--alpha-max");
        else if(arg == "--alpha-min")
            config.alphaMin = parseReal(value(i, argc, argv), "--alpha-min");
        else if(arg == "--material")
            material = value(i, argc, argv);
        else if(arg == "--backend")
            backend = value(i, argc, argv);
        else if(arg == "--weights")
            weights = value(i, argc, argv);
        else if(arg == "--output")
            output = value(i, argc, argv);
        else if(arg == "--export-alpha")
            alphaExport = value(i, argc, argv);
        else if(arg == "--frames")
            frames = parseCount(value(i, argc, argv), "--frames");
        else if(arg == "--dump-features")
            featurePath = value(i, argc, argv);
        else if(arg == "--samples")
            sampleCount = parseCount(value(i, argc, argv), "--samples");
        else if(arg == "--seed")
            seed = parseCount(value(i, argc, argv), "--seed");
        else if(arg == "--temperature-samples")
            temperatureSamples = parseCount(value(i, argc, argv), "--temperature-samples");
        else if(arg == "--no-output")
            noOutput = true;
        else if(arg == "--validate-strict")
            config.validateStrict = true;
        else if(arg == "--help")
        {
            std::cout
                << "--grid N --tmax T --steps N --material uniform|preset|nn --beta B --alpha-min A --alpha-max A "
                   "--weights FILE --output DIR --export-alpha CSV --frames N --no-output --validate-strict "
                   "--backend host|omp|hip|cuda|oneapi|<device-index> "
                   "--dump-features CSV [--samples N --seed N --temperature-samples N]\n";
            return 0;
        }
        else
            throw std::invalid_argument("unknown option: " + arg);
    }
    if(material != "uniform" && material != "preset" && material != "nn")
        throw std::invalid_argument("invalid material");
    if(noOutput && !alphaExport.empty())
        throw std::invalid_argument("--export-alpha cannot be combined with --no-output");
    auto const selection = resolveBackend(backend);
    // Validate the grid before any host/device allocation. Building the initial
    // field first would turn an out-of-range grid into an opaque bad_alloc.
    heatclosure::device_detail::validateGrid(config.n, "--grid");
    if(!featurePath.empty())
    {
        dumpFeatures(featurePath, sampleCount, seed, temperatureSamples, config.beta);
        return 0;
    }
    if(frames < 2)
        throw std::invalid_argument("frames must be >= 2");
    config.uniformMaterial = material == "uniform";
    heatclosure::Model model;
    heatclosure::Model const* modelPtr = nullptr;
    auto mode = material == "nn"       ? heatclosure::CoefficientMode::neural
                : material == "preset" ? heatclosure::CoefficientMode::preset
                                       : heatclosure::CoefficientMode::uniform;
    if(material == "nn")
    {
        if(weights.empty())
            throw std::invalid_argument("--weights is required for --material nn");
        model = heatclosure::loadModel(weights, config.beta);
        if(model.alphaMin != config.alphaMin || model.alphaMax != config.alphaMax)
            throw std::invalid_argument("CLI alpha bounds must match model metadata");
        modelPtr = &model;
    }
    return dispatchBackend(
        selection,
        [&](auto device, auto exec, std::string const& backendName)
        {
            return run(
                device,
                exec,
                backendName,
                config,
                mode,
                modelPtr,
                frames,
                noOutput,
                config.validateStrict,
                output,
                alphaExport);
        });
}
catch(std::exception const& e)
{
    std::cerr << "heatEquationNn: " << e.what() << '\n';
    return 2;
}
