/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    template<typename T_Type>
    std::string formatTopLogits(std::string_view label, auto const& hostLogits, uint32_t row, uint32_t topK = 10u)
    {
        auto const vocabSize = static_cast<uint32_t>(hostLogits.getExtents()[1]);
        std::vector<std::pair<T_Type, uint32_t>> ranked;
        ranked.reserve(std::min(topK, vocabSize));

        for(uint32_t token = 0u; token < vocabSize; ++token)
        {
            auto const value = hostLogits[alpaka::Vec{row, token}];
            ranked.emplace_back(value, token);
        }

        auto const keep = std::min<uint32_t>(topK, vocabSize);
        std::partial_sort(
            ranked.begin(),
            ranked.begin() + static_cast<std::ptrdiff_t>(keep),
            ranked.end(),
            [](auto const& lhs, auto const& rhs)
            {
                if(lhs.first == rhs.first)
                    return lhs.second < rhs.second;
                return lhs.first > rhs.first;
            });
        ranked.resize(keep);

        std::ostringstream os;
        os << label << " top-" << keep << ':';
        for(auto const& [value, token] : ranked)
            os << " (" << token << ", " << value << ')';
        return os.str();
    }

    void maybePrintDiagnostic(std::string const& message)
    {
        auto const* env = std::getenv("ALPAKANN_DEBUG_TOPK");
        if(env == nullptr || env[0] == '\0' || env[0] == '0')
            return;
        std::fprintf(stderr, "%s\n", message.c_str());
    }

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

    auto model = alpaka::nn::onHost::model::loadTinyLlama<float>(device, ensureTinyModel());
    std::vector<uint32_t> prompt{1u, 2u, 3u, 4u};
    auto logits = alpaka::nn::onHost::model::prefill(queue, exec, model, prompt);
    auto hostLogits = alpaka::onHost::allocHost<float>(logits.getExtents());
    alpaka::onHost::memcpy(queue, hostLogits, logits);
    alpaka::onHost::wait(queue);

    REQUIRE(hostLogits.getExtents()[0] == 1u);
    REQUIRE(hostLogits.getExtents()[1] == model.config.vocabSize);
    auto const prefillTopLogits = formatTopLogits<float>("prefill logits", hostLogits, 0u);
    INFO(prefillTopLogits);
    maybePrintDiagnostic(prefillTopLogits);

    auto generated = alpaka::nn::onHost::inference::generateGreedy(queue, exec, model, prompt, 2u);
    REQUIRE(generated.size() == 6u);
    CHECK(generated[4] == 25190u);
    CHECK(generated[5] == 6074u);
    alpaka::nn::test::checkValue(hostLogits[alpaka::Vec{0u, 0u}], -0.0607535f, 1.0e-4, 1.0e-4);
    alpaka::nn::test::checkValue(hostLogits[alpaka::Vec{0u, 1u}], 0.00580135f, 1.0e-4, 1.0e-4);
    alpaka::nn::test::checkValue(hostLogits[alpaka::Vec{0u, 2u}], 0.0907773f, 1.0e-4, 1.0e-4);
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

    auto model = alpaka::nn::onHost::model::loadTinyLlama<float>(device, ensureTinyModel());
    std::vector<uint32_t> prompt{1u, 2u, 3u, 4u};
    auto cache = alpaka::nn::onHost::inference::makeKvCache<float>(
        device,
        model.config.numLayers,
        1u,
        model.config.numKeyValueHeads,
        static_cast<uint32_t>(prompt.size() + 2u),
        model.config.hiddenSize / model.config.numHeads);
    auto logits = alpaka::nn::onHost::model::prefill(queue, exec, model, prompt, cache);
    auto hostLogits = alpaka::onHost::allocHost<float>(logits.getExtents());
    alpaka::onHost::memcpy(queue, hostLogits, logits);
    alpaka::onHost::wait(queue);
    auto const prefillTopLogits = formatTopLogits<float>("prefill logits", hostLogits, 0u);
    INFO(prefillTopLogits);
    maybePrintDiagnostic(prefillTopLogits);
    auto next = alpaka::nn::onHost::inference::argmax<float>(hostLogits, 0u);
    REQUIRE(next == 25190u);

    auto stepLogits = alpaka::nn::onHost::model::decodeStep(queue, exec, model, cache, next);
    auto hostStepLogits = alpaka::onHost::allocHost<float>(stepLogits.getExtents());
    alpaka::onHost::memcpy(queue, hostStepLogits, stepLogits);
    alpaka::onHost::wait(queue);
    auto const stepTopLogits = formatTopLogits<float>("decodeStep logits", hostStepLogits, 0u);
    INFO(stepTopLogits);
    maybePrintDiagnostic(stepTopLogits);
    REQUIRE(alpaka::nn::onHost::inference::argmax<float>(hostStepLogits, 0u) == 6074u);
    REQUIRE(cache.length(0u, 0u) == prompt.size() + 1u);
}

TEMPLATE_LIST_TEST_CASE("tiny llama implicit and explicit prefill agree", "[model][decoder]", TestApis)
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

    auto model = alpaka::nn::onHost::model::loadTinyLlama<float>(device, ensureTinyModel());
    std::vector<uint32_t> prompt{1u, 2u, 3u, 4u};

    auto implicitLogits = alpaka::nn::onHost::model::prefill(queue, exec, model, prompt);
    auto explicitCache = alpaka::nn::onHost::inference::makeKvCache<float>(
        device,
        model.config.numLayers,
        1u,
        model.config.numKeyValueHeads,
        static_cast<uint32_t>(prompt.size() + 2u),
        model.config.hiddenSize / model.config.numHeads);
    auto explicitLogits = alpaka::nn::onHost::model::prefill(queue, exec, model, prompt, explicitCache);

    auto hostImplicit = alpaka::onHost::allocHost<float>(implicitLogits.getExtents());
    auto hostExplicit = alpaka::onHost::allocHost<float>(explicitLogits.getExtents());
    alpaka::onHost::memcpy(queue, hostImplicit, implicitLogits);
    alpaka::onHost::memcpy(queue, hostExplicit, explicitLogits);
    alpaka::onHost::wait(queue);

    auto const implicitTopLogits = formatTopLogits<float>("implicit prefill logits", hostImplicit, 0u);
    auto const explicitTopLogits = formatTopLogits<float>("explicit prefill logits", hostExplicit, 0u);
    INFO(implicitTopLogits);
    INFO(explicitTopLogits);
    maybePrintDiagnostic(implicitTopLogits);
    maybePrintDiagnostic(explicitTopLogits);

    REQUIRE(hostImplicit.getExtents() == hostExplicit.getExtents());
    for(uint32_t token = 0u; token < model.config.vocabSize; ++token)
    {
        alpaka::nn::test::checkValue(
            hostImplicit[alpaka::Vec{0u, token}],
            hostExplicit[alpaka::Vec{0u, token}],
            1.0e-4,
            1.0e-4);
    }
}
