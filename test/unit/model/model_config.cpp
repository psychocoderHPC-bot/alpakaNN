/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpakaNN/alpakaNN.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace
{
    fs::path writeModelFile(uint32_t numHeads, uint32_t numKeyValueHeads)
    {
        auto const path = fs::temp_directory_path()
                          / ("alpakaNN-model-config-" + std::to_string(numHeads) + "-"
                             + std::to_string(numKeyValueHeads) + ".bin");
        std::ofstream output(path, std::ios::binary);
        REQUIRE(output);

        output.write("ANN1", 4);
        uint32_t values[] = {
            16u, // hiddenSize
            64u, // intermediateSize
            2u, // numLayers
            numHeads,
            numKeyValueHeads,
            32000u, // vocabSize
            1u, // bos
            2u, // eos
            2048u // maxPositionEmbeddings
        };
        float floats[] = {1.0e-6f, 10000.0f};
        output.write(reinterpret_cast<char*>(values), sizeof(values));
        output.write(reinterpret_cast<char*>(floats), sizeof(floats));
        REQUIRE(output.good());
        return path;
    }
} // namespace

TEST_CASE("loadTinyLlama accepts valid grouped-query attention configs", "[model][decoder]")
{
    auto backends = alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors);
    auto cfg = std::get<0>(backends);
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }

    auto const path = writeModelFile(4u, 2u);
    auto device = selector.makeDevice(0);
    REQUIRE_THROWS_WITH(
        alpakaNN::model::loadTinyLlama<float>(device, path.string()),
        Catch::Matchers::ContainsSubstring("Unexpected end of file"));
    fs::remove(path);
}

TEST_CASE("loadTinyLlama rejects invalid grouped-query attention ratios", "[model][decoder]")
{
    auto backends = alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors);
    auto cfg = std::get<0>(backends);
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }

    auto const path = writeModelFile(4u, 3u);
    auto device = selector.makeDevice(0);
    REQUIRE_THROWS_WITH(
        alpakaNN::model::loadTinyLlama<float>(device, path.string()),
        Catch::Matchers::ContainsSubstring("divisible by numKeyValueHeads"));
    fs::remove(path);
}
