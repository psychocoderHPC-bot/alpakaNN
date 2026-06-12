/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpakaNN/alpakaNN.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using TestApis = alpakaNN::test::TestApis;

namespace
{
    std::string ensureTinyModel()
    {
#ifdef TINY_LLAMA_MODEL_PATH
        auto const modelPath = std::string{TINY_LLAMA_MODEL_PATH};
#else
        auto const outDir = std::string{"/workspace/testdata/tiny_llama"};
        auto const modelPath = outDir + "/tiny_llama.bin";
#endif
        if(auto* file = std::fopen(modelPath.c_str(), "rb"))
        {
            std::fclose(file);
            return modelPath;
        }

#ifdef TINY_LLAMA_MODEL_PATH
        // When built with TINY_LLAMA_MODEL_PATH, model should already exist
        throw std::runtime_error("Model not found at: " + modelPath);
#else
        // Fallback to downloading at runtime
        auto const command = std::string{"python3 /workspace/tools/download_tiny_llama.py "} + outDir;
        REQUIRE(std::system(command.c_str()) == 0);
#endif
        return modelPath;
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("tiny llama model loads and generates deterministically", "[model][decoder]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }

    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    auto model = alpakaNN::model::loadTinyLlama<float>(device, ensureTinyModel());
    std::vector<uint32_t> prompt{1u, 2u, 3u, 4u};
    auto logits = alpakaNN::model::prefill(queue, exec, model, prompt);
    auto hostLogits = alpaka::onHost::allocHost<float>(logits.getExtents());
    alpaka::onHost::memcpy(queue, hostLogits, logits);
    alpaka::onHost::wait(queue);

    REQUIRE(hostLogits.getExtents()[0] == 1u);
    REQUIRE(hostLogits.getExtents()[1] == model.config.vocabSize);

    auto generated = alpakaNN::inference::generateGreedy(queue, exec, model, prompt, 2u);
    REQUIRE(generated.size() == 6u);
    CHECK(generated[4] == 25190u);
    CHECK(generated[5] == 6074u);
    alpakaNN::test::checkValue(hostLogits[alpaka::Vec{0u, 0u}], -0.0607535f, 1.0e-4, 1.0e-4);
    alpakaNN::test::checkValue(hostLogits[alpaka::Vec{0u, 1u}], 0.00580135f, 1.0e-4, 1.0e-4);
    alpakaNN::test::checkValue(hostLogits[alpaka::Vec{0u, 2u}], 0.0907773f, 1.0e-4, 1.0e-4);
}

TEMPLATE_LIST_TEST_CASE("tiny llama decodeStep matches greedy generation", "[model][decoder]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }

    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    auto model = alpakaNN::model::loadTinyLlama<float>(device, ensureTinyModel());
    std::vector<uint32_t> prompt{1u, 2u, 3u, 4u};
    auto cache = alpakaNN::inference::makeKvCache<float>(
        device,
        model.config.numLayers,
        1u,
        model.config.numKeyValueHeads,
        static_cast<uint32_t>(prompt.size() + 2u),
        model.config.hiddenSize / model.config.numHeads);
    auto logits = alpakaNN::model::prefill(queue, exec, model, prompt, cache);
    auto hostLogits = alpaka::onHost::allocHost<float>(logits.getExtents());
    alpaka::onHost::memcpy(queue, hostLogits, logits);
    alpaka::onHost::wait(queue);
    auto next = alpakaNN::inference::argmax<float>(hostLogits, 0u);
    REQUIRE(next == 25190u);

    auto stepLogits = alpakaNN::model::decodeStep(queue, exec, model, cache, next);
    auto hostStepLogits = alpaka::onHost::allocHost<float>(stepLogits.getExtents());
    alpaka::onHost::memcpy(queue, hostStepLogits, stepLogits);
    alpaka::onHost::wait(queue);
    REQUIRE(alpakaNN::inference::argmax<float>(hostStepLogits, 0u) == 6074u);
    REQUIRE(cache.length(0u, 0u) == prompt.size() + 1u);
}
