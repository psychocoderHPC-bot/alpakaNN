/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

    bool prefillCompareEnabled()
    {
        auto const* env = std::getenv("ALPAKANN_DEBUG_PREFILL_COMPARE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }

    bool fullStageCompareEnabled()
    {
        auto const* env = std::getenv("ALPAKANN_DEBUG_PREFILL_FULL_COMPARE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }

    template<typename T_Type>
    struct StageSnapshot
    {
        std::string label;
        uint32_t rows{};
        uint32_t cols{};
        std::vector<T_Type> values;
    };

    template<typename T_Type>
    StageSnapshot<T_Type> captureStage(auto& queue, auto const& tensor, std::string label)
    {
        auto host = alpaka::onHost::allocHost<T_Type>(tensor.getExtents());
        alpaka::onHost::memcpy(queue, host, tensor);
        alpaka::onHost::wait(queue);

        StageSnapshot<T_Type> snapshot;
        snapshot.label = std::move(label);
        snapshot.rows = static_cast<uint32_t>(host.getExtents()[0]);
        snapshot.cols = static_cast<uint32_t>(host.getExtents()[1]);
        snapshot.values.reserve(static_cast<std::size_t>(snapshot.rows) * static_cast<std::size_t>(snapshot.cols));
        for(uint32_t row = 0u; row < snapshot.rows; ++row)
            for(uint32_t col = 0u; col < snapshot.cols; ++col)
                snapshot.values.push_back(host[alpaka::Vec{row, col}]);
        return snapshot;
    }

    template<typename T_Type>
    std::vector<StageSnapshot<T_Type>> runPrefillStages(
        auto& queue,
        auto exec,
        auto const& model,
        std::vector<uint32_t> const& tokenIds,
        auto& cache,
        std::string const& labelPrefix)
    {
        std::vector<StageSnapshot<T_Type>> stages;

        auto hostTokens = alpaka::onHost::allocHost<uint32_t>(static_cast<uint32_t>(tokenIds.size()));
        for(uint32_t i = 0u; i < tokenIds.size(); ++i)
            hostTokens[alpaka::Vec{i}] = tokenIds[i];
        auto devTokens = alpaka::onHost::allocLike(queue.getDevice(), hostTokens);
        alpaka::onHost::memcpy(queue, devTokens, hostTokens);

        auto hidden = alpaka::onHost::alloc<T_Type>(
            queue.getDevice(),
            alpaka::Vec{static_cast<uint32_t>(tokenIds.size()), model.config.hiddenSize});
        alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, devTokens, model.embedding, hidden);
        stages.push_back(captureStage<T_Type>(queue, hidden, labelPrefix + " embedding"));

        auto ropeTables = alpaka::nn::onHost::model::makeRopeTables<T_Type>(
            queue.getDevice(),
            static_cast<uint32_t>(tokenIds.size()),
            model.config.hiddenSize / model.config.numHeads / 2u,
            static_cast<T_Type>(model.config.ropeTheta));

        for(uint32_t layer = 0u; layer < model.config.numLayers; ++layer)
        {
            auto next = alpaka::onHost::alloc<T_Type>(queue.getDevice(), hidden.getExtents());
            alpaka::nn::onHost::inference::transformerBlock<T_Type>(
                queue,
                exec,
                hidden,
                model.layers[layer],
                cache,
                layer,
                ropeTables.first,
                ropeTables.second,
                next);
            hidden = next;
            stages.push_back(
                captureStage<T_Type>(queue, hidden, labelPrefix + " layer " + std::to_string(layer) + " output"));
        }

        auto norm = alpaka::onHost::alloc<T_Type>(queue.getDevice(), hidden.getExtents());
        alpaka::nn::onHost::nn::rmsNorm<T_Type>(
            queue,
            exec,
            hidden,
            model.finalNorm,
            norm,
            static_cast<T_Type>(model.config.rmsNormEpsilon));
        stages.push_back(captureStage<T_Type>(queue, norm, labelPrefix + " final norm"));

        auto logits = alpaka::onHost::alloc<T_Type>(
            queue.getDevice(),
            alpaka::Vec{static_cast<uint32_t>(tokenIds.size()), model.config.vocabSize});
        alpaka::nn::onHost::gemm<T_Type>(queue, exec, norm, model.lmHead, logits);
        stages.push_back(captureStage<T_Type>(queue, logits, labelPrefix + " full logits"));

        auto lastLogits = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, model.config.vocabSize});
        alpaka::onHost::memcpy(
            queue,
            lastLogits,
            logits.getSubView(
                alpaka::Vec{static_cast<uint32_t>(tokenIds.size() - 1u), 0u},
                alpaka::Vec{1u, model.config.vocabSize}));
        alpaka::onHost::wait(queue);
        stages.push_back(captureStage<T_Type>(queue, lastLogits, labelPrefix + " last logits"));
        return stages;
    }

    template<typename T_Type>
    std::vector<StageSnapshot<T_Type>> runDecodeStages(
        auto& queue,
        auto exec,
        auto const& model,
        std::vector<uint32_t> const& prompt,
        uint32_t nextToken,
        uint32_t cacheCapacity,
        std::string const& labelPrefix)
    {
        std::vector<StageSnapshot<T_Type>> stages;
        auto cache = alpaka::nn::onHost::inference::makeKvCache<T_Type>(
            queue.getDevice(),
            model.config.numLayers,
            1u,
            model.config.numKeyValueHeads,
            cacheCapacity,
            model.config.hiddenSize / model.config.numHeads);
        (void)alpaka::nn::onHost::model::prefill(queue, exec, model, prompt, cache, labelPrefix + " prefill");

        auto hostToken = alpaka::onHost::allocHost<uint32_t>(1u);
        hostToken[alpaka::Vec{0u}] = nextToken;
        auto devToken = alpaka::onHost::allocLike(queue.getDevice(), hostToken);
        alpaka::onHost::memcpy(queue, devToken, hostToken);

        auto hidden = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, model.config.hiddenSize});
        alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, devToken, model.embedding, hidden);
        stages.push_back(captureStage<T_Type>(queue, hidden, labelPrefix + " embedding"));

        auto const currentContext = cache.length(0u, 0u);
        auto ropeTables = alpaka::nn::onHost::model::makeRopeTables<T_Type>(
            queue.getDevice(),
            currentContext + 1u,
            model.config.hiddenSize / model.config.numHeads / 2u,
            static_cast<T_Type>(model.config.ropeTheta));

        for(uint32_t layer = 0u; layer < model.config.numLayers; ++layer)
        {
            auto next = alpaka::onHost::alloc<T_Type>(queue.getDevice(), hidden.getExtents());
            alpaka::nn::onHost::inference::transformerBlockDecodeStep<T_Type>(
                queue,
                exec,
                hidden,
                model.layers[layer],
                cache,
                layer,
                ropeTables.first,
                ropeTables.second,
                next);
            hidden = next;
            stages.push_back(
                captureStage<T_Type>(queue, hidden, labelPrefix + " layer " + std::to_string(layer) + " output"));
        }

        auto norm = alpaka::onHost::alloc<T_Type>(queue.getDevice(), hidden.getExtents());
        alpaka::nn::onHost::nn::rmsNorm<T_Type>(
            queue,
            exec,
            hidden,
            model.finalNorm,
            norm,
            static_cast<T_Type>(model.config.rmsNormEpsilon));
        stages.push_back(captureStage<T_Type>(queue, norm, labelPrefix + " final norm"));

        auto logits = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, model.config.vocabSize});
        alpaka::nn::onHost::gemm<T_Type>(queue, exec, norm, model.lmHead, logits);
        stages.push_back(captureStage<T_Type>(queue, logits, labelPrefix + " logits"));
        return stages;
    }

    template<typename T_Type>
    std::string summarizeStageDiff(StageSnapshot<T_Type> const& lhs, StageSnapshot<T_Type> const& rhs, T_Type tolerance)
    {
        std::ostringstream os;
        os << lhs.label << " vs " << rhs.label;
        if(lhs.rows != rhs.rows || lhs.cols != rhs.cols)
        {
            os << " shapeMismatch lhs=(" << lhs.rows << "," << lhs.cols << ") rhs=(" << rhs.rows << "," << rhs.cols
               << ")";
            return os.str();
        }

        uint32_t mismatchCount = 0u;
        uint32_t firstRow = 0u;
        uint32_t firstCol = 0u;
        T_Type firstLhs{};
        T_Type firstRhs{};
        double maxAbsDiff = 0.0;
        double meanAbsDiff = 0.0;
        for(std::size_t idx = 0; idx < lhs.values.size(); ++idx)
        {
            auto const left = lhs.values[idx];
            auto const right = rhs.values[idx];
            auto const absDiff = std::fabs(static_cast<double>(left - right));
            meanAbsDiff += absDiff;
            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            if(absDiff > static_cast<double>(tolerance))
            {
                if(mismatchCount == 0u)
                {
                    firstRow = static_cast<uint32_t>(idx / lhs.cols);
                    firstCol = static_cast<uint32_t>(idx % lhs.cols);
                    firstLhs = left;
                    firstRhs = right;
                }
                ++mismatchCount;
            }
        }
        if(!lhs.values.empty())
            meanAbsDiff /= static_cast<double>(lhs.values.size());
        os << " mismatchCount=" << mismatchCount << " maxAbsDiff=" << maxAbsDiff << " meanAbsDiff=" << meanAbsDiff;
        if(mismatchCount != 0u)
            os << " firstMismatch=(" << firstRow << "," << firstCol << ") lhs=" << firstLhs << " rhs=" << firstRhs;
        return os.str();
    }

    template<typename T_Type>
    std::string comparePrefillStages(
        auto& queue,
        auto exec,
        auto const& model,
        std::vector<uint32_t> const& prompt)
    {
        auto implicitCache = alpaka::nn::onHost::inference::makeKvCache<T_Type>(
            queue.getDevice(),
            model.config.numLayers,
            1u,
            model.config.numKeyValueHeads,
            static_cast<uint32_t>(prompt.size()),
            model.config.hiddenSize / model.config.numHeads);
        auto explicitCache = alpaka::nn::onHost::inference::makeKvCache<T_Type>(
            queue.getDevice(),
            model.config.numLayers,
            1u,
            model.config.numKeyValueHeads,
            static_cast<uint32_t>(prompt.size() + 2u),
            model.config.hiddenSize / model.config.numHeads);

        auto implicitStages = runPrefillStages<T_Type>(queue, exec, model, prompt, implicitCache, "implicit");
        auto explicitStages = runPrefillStages<T_Type>(queue, exec, model, prompt, explicitCache, "explicit");
        auto const stageCount = std::min(implicitStages.size(), explicitStages.size());
        for(std::size_t stage = 0; stage < stageCount; ++stage)
        {
            auto summary = summarizeStageDiff(implicitStages[stage], explicitStages[stage], static_cast<T_Type>(1.0e-4));
            if(summary.find("mismatchCount=0") == std::string::npos)
                return "first divergent stage: " + summary;
        }
        if(implicitStages.size() != explicitStages.size())
        {
            std::ostringstream os;
            os << "stage count mismatch implicit=" << implicitStages.size() << " explicit=" << explicitStages.size();
            return os.str();
        }
        return "all captured prefill stages matched";
    }

    template<typename T_Type>
    std::string fullPrefillStageReport(
        auto& queue,
        auto exec,
        auto const& model,
        std::vector<uint32_t> const& prompt,
        uint32_t capacityA,
        uint32_t capacityB)
    {
        auto cacheA = alpaka::nn::onHost::inference::makeKvCache<T_Type>(
            queue.getDevice(),
            model.config.numLayers,
            1u,
            model.config.numKeyValueHeads,
            capacityA,
            model.config.hiddenSize / model.config.numHeads);
        auto cacheB = alpaka::nn::onHost::inference::makeKvCache<T_Type>(
            queue.getDevice(),
            model.config.numLayers,
            1u,
            model.config.numKeyValueHeads,
            capacityB,
            model.config.hiddenSize / model.config.numHeads);
        auto stagesA = runPrefillStages<T_Type>(queue, exec, model, prompt, cacheA, "capacity " + std::to_string(capacityA));
        auto stagesB = runPrefillStages<T_Type>(queue, exec, model, prompt, cacheB, "capacity " + std::to_string(capacityB));

        std::ostringstream os;
        auto const stageCount = std::min(stagesA.size(), stagesB.size());
        for(std::size_t stage = 0; stage < stageCount; ++stage)
            os << summarizeStageDiff(stagesA[stage], stagesB[stage], static_cast<T_Type>(1.0e-4)) << '\n';
        if(stagesA.size() != stagesB.size())
            os << "stage count mismatch lhs=" << stagesA.size() << " rhs=" << stagesB.size() << '\n';
        return os.str();
    }

    template<typename T_Type>
    std::string compareDecodeStages(
        auto& queue,
        auto exec,
        auto const& model,
        std::vector<uint32_t> const& prompt,
        uint32_t nextToken)
    {
        auto compactStages = runDecodeStages<T_Type>(
            queue,
            exec,
            model,
            prompt,
            nextToken,
            static_cast<uint32_t>(prompt.size() + 2u),
            "decode compact");
        auto roomyStages = runDecodeStages<T_Type>(
            queue,
            exec,
            model,
            prompt,
            nextToken,
            static_cast<uint32_t>(prompt.size() + 4u),
            "decode roomy");
        auto const stageCount = std::min(compactStages.size(), roomyStages.size());
        for(std::size_t stage = 0; stage < stageCount; ++stage)
        {
            auto summary = summarizeStageDiff(compactStages[stage], roomyStages[stage], static_cast<T_Type>(1.0e-4));
            if(summary.find("mismatchCount=0") == std::string::npos)
                return "first divergent decode stage: " + summary;
        }
        if(compactStages.size() != roomyStages.size())
        {
            std::ostringstream os;
            os << "decode stage count mismatch compact=" << compactStages.size() << " roomy=" << roomyStages.size();
            return os.str();
        }
        return "all captured decode stages matched";
    }

    template<typename T_Type>
    std::string fullDecodeStageReport(
        auto& queue,
        auto exec,
        auto const& model,
        std::vector<uint32_t> const& prompt,
        uint32_t nextToken,
        uint32_t capacityA,
        uint32_t capacityB)
    {
        auto stagesA = runDecodeStages<T_Type>(
            queue,
            exec,
            model,
            prompt,
            nextToken,
            capacityA,
            "decode capacity " + std::to_string(capacityA));
        auto stagesB = runDecodeStages<T_Type>(
            queue,
            exec,
            model,
            prompt,
            nextToken,
            capacityB,
            "decode capacity " + std::to_string(capacityB));

        std::ostringstream os;
        auto const stageCount = std::min(stagesA.size(), stagesB.size());
        for(std::size_t stage = 0; stage < stageCount; ++stage)
            os << summarizeStageDiff(stagesA[stage], stagesB[stage], static_cast<T_Type>(1.0e-4)) << '\n';
        if(stagesA.size() != stagesB.size())
            os << "decode stage count mismatch lhs=" << stagesA.size() << " rhs=" << stagesB.size() << '\n';
        return os.str();
    }

    template<typename T_Type>
    std::vector<std::vector<StageSnapshot<T_Type>>> collectPrefillStagesByCapacity(
        auto& queue,
        auto exec,
        auto const& model,
        std::vector<uint32_t> const& prompt,
        std::vector<uint32_t> const& capacities)
    {
        std::vector<std::vector<StageSnapshot<T_Type>>> allStages;
        allStages.reserve(capacities.size());
        for(auto capacity : capacities)
        {
            auto cache = alpaka::nn::onHost::inference::makeKvCache<T_Type>(
                queue.getDevice(),
                model.config.numLayers,
                1u,
                model.config.numKeyValueHeads,
                capacity,
                model.config.hiddenSize / model.config.numHeads);
            allStages.push_back(runPrefillStages<T_Type>(
                queue,
                exec,
                model,
                prompt,
                cache,
                "capacity " + std::to_string(capacity)));
        }
        return allStages;
    }

    template<typename T_Type>
    void requireStageParity(
        std::vector<std::vector<StageSnapshot<T_Type>>> const& allStages,
        std::vector<uint32_t> const& capacities,
        std::size_t stageIndex,
        std::string_view context)
    {
        auto const& reference = allStages.front().at(stageIndex);
        for(std::size_t idx = 1; idx < allStages.size(); ++idx)
        {
            auto const& candidate = allStages[idx].at(stageIndex);
            INFO(
                std::string{context} + " stage=" + reference.label + " capacityRef="
                + std::to_string(capacities.front()) + " capacityCmp=" + std::to_string(capacities[idx]));
            INFO(summarizeStageDiff(reference, candidate, static_cast<T_Type>(1.0e-4)));
            REQUIRE(reference.rows == candidate.rows);
            REQUIRE(reference.cols == candidate.cols);

            uint32_t mismatchCount = 0u;
            for(std::size_t valueIdx = 0; valueIdx < reference.values.size(); ++valueIdx)
            {
                auto const absDiff = std::fabs(
                    static_cast<double>(reference.values[valueIdx] - candidate.values[valueIdx]));
                if(absDiff > 1.0e-4)
                    ++mismatchCount;
            }
            REQUIRE(mismatchCount == 0u);
        }
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
    if(generated[4] != 25190u)
        INFO(comparePrefillStages<float>(queue, exec, model, prompt));
    if(fullStageCompareEnabled())
        INFO(fullPrefillStageReport<float>(queue, exec, model, prompt, 4u, 5u));
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
    if(next != 25190u)
        INFO(comparePrefillStages<float>(queue, exec, model, prompt));
    if(fullStageCompareEnabled())
        INFO(fullPrefillStageReport<float>(queue, exec, model, prompt, 4u, 6u));
    REQUIRE(next == 25190u);

    auto stepLogits = alpaka::nn::onHost::model::decodeStep(queue, exec, model, cache, next);
    auto hostStepLogits = alpaka::onHost::allocHost<float>(stepLogits.getExtents());
    alpaka::onHost::memcpy(queue, hostStepLogits, stepLogits);
    alpaka::onHost::wait(queue);
    auto const stepTopLogits = formatTopLogits<float>("decodeStep logits", hostStepLogits, 0u);
    INFO(stepTopLogits);
    maybePrintDiagnostic(stepTopLogits);
    auto const next2 = alpaka::nn::onHost::inference::argmax<float>(hostStepLogits, 0u);
    if(next2 != 6074u)
        INFO(compareDecodeStages<float>(queue, exec, model, prompt, next));
    if(fullStageCompareEnabled())
        INFO(fullDecodeStageReport<float>(queue, exec, model, prompt, next, 6u, 7u));
    REQUIRE(next2 == 6074u);

    auto step2Logits = alpaka::nn::onHost::model::decodeStep(queue, exec, model, cache, next2);
    auto hostStep2Logits = alpaka::onHost::allocHost<float>(step2Logits.getExtents());
    alpaka::onHost::memcpy(queue, hostStep2Logits, step2Logits);
    alpaka::onHost::wait(queue);
    auto const step2TopLogits = formatTopLogits<float>("decodeStep 2 logits", hostStep2Logits, 0u);
    INFO(step2TopLogits);
    maybePrintDiagnostic(step2TopLogits);

    auto generated = alpaka::nn::onHost::inference::generateGreedy(queue, exec, model, prompt, 3u);
    REQUIRE(generated.size() == prompt.size() + 3u);
    REQUIRE(generated[4] == next);
    REQUIRE(generated[5] == next2);
    REQUIRE(generated[6] == alpaka::nn::onHost::inference::argmax<float>(hostStep2Logits, 0u));
    REQUIRE(cache.length(0u, 0u) == prompt.size() + 2u);
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
    uint32_t mismatchCount = 0u;
    uint32_t firstMismatchToken = 0u;
    float firstImplicitValue = 0.0f;
    float firstExplicitValue = 0.0f;
    float maxAbsDiff = 0.0f;
    for(uint32_t token = 0u; token < model.config.vocabSize; ++token)
    {
        auto const implicitValue = hostImplicit[alpaka::Vec{0u, token}];
        auto const explicitValue = hostExplicit[alpaka::Vec{0u, token}];
        auto const absDiff = std::fabs(implicitValue - explicitValue);
        maxAbsDiff = std::max(maxAbsDiff, absDiff);
        if(absDiff > 1.0e-4f)
        {
            if(mismatchCount == 0u)
            {
                firstMismatchToken = token;
                firstImplicitValue = implicitValue;
                firstExplicitValue = explicitValue;
            }
            ++mismatchCount;
        }
    }
    INFO(
        "mismatchCount=" << mismatchCount << " firstMismatchToken=" << firstMismatchToken
                         << " implicit=" << firstImplicitValue << " explicit=" << firstExplicitValue
                         << " maxAbsDiff=" << maxAbsDiff);
    if(prefillCompareEnabled() || mismatchCount != 0u)
        INFO(comparePrefillStages<float>(queue, exec, model, prompt));
    if(fullStageCompareEnabled())
        INFO(fullPrefillStageReport<float>(queue, exec, model, prompt, 4u, 6u));
    REQUIRE(mismatchCount == 0u);
}

TEMPLATE_LIST_TEST_CASE("tiny llama prefill is independent of cache capacity", "[model][decoder]", TestApis)
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
    std::vector<uint32_t> capacities{4u, 5u, 6u, 7u};
    std::vector<decltype(alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u}))> hostLogitsByCapacity;
    hostLogitsByCapacity.reserve(capacities.size());

    for(auto capacity : capacities)
    {
        auto cache = alpaka::nn::onHost::inference::makeKvCache<float>(
            device,
            model.config.numLayers,
            1u,
            model.config.numKeyValueHeads,
            capacity,
            model.config.hiddenSize / model.config.numHeads);
        auto logits = alpaka::nn::onHost::model::prefill(queue, exec, model, prompt, cache);
        auto hostLogits = alpaka::onHost::allocHost<float>(logits.getExtents());
        alpaka::onHost::memcpy(queue, hostLogits, logits);
        alpaka::onHost::wait(queue);

        INFO("capacity=" << capacity << ' ' << formatTopLogits<float>("prefill logits", hostLogits, 0u));
        hostLogitsByCapacity.push_back(std::move(hostLogits));
    }

    auto const& reference = hostLogitsByCapacity.front();
    for(std::size_t idx = 1; idx < hostLogitsByCapacity.size(); ++idx)
    {
        auto const& candidate = hostLogitsByCapacity[idx];
        REQUIRE(reference.getExtents() == candidate.getExtents());

        uint32_t mismatchCount = 0u;
        uint32_t firstMismatchToken = 0u;
        float referenceValue = 0.0f;
        float candidateValue = 0.0f;
        float maxAbsDiff = 0.0f;
        for(uint32_t token = 0u; token < model.config.vocabSize; ++token)
        {
            auto const left = reference[alpaka::Vec{0u, token}];
            auto const right = candidate[alpaka::Vec{0u, token}];
            auto const absDiff = std::fabs(left - right);
            maxAbsDiff = std::max(maxAbsDiff, absDiff);
            if(absDiff > 1.0e-4f)
            {
                if(mismatchCount == 0u)
                {
                    firstMismatchToken = token;
                    referenceValue = left;
                    candidateValue = right;
                }
                ++mismatchCount;
            }
        }
        INFO(
            "capacity " << capacities.front() << " vs " << capacities[idx] << " mismatchCount=" << mismatchCount
                        << " firstMismatchToken=" << firstMismatchToken << " reference=" << referenceValue
                        << " candidate=" << candidateValue << " maxAbsDiff=" << maxAbsDiff);
        if(fullStageCompareEnabled())
            INFO(fullPrefillStageReport<float>(queue, exec, model, prompt, capacities.front(), capacities[idx]));
        REQUIRE(mismatchCount == 0u);
    }
}

TEMPLATE_LIST_TEST_CASE("tiny llama prefill stages are independent of cache capacity", "[model][decoder]", TestApis)
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
    std::vector<uint32_t> capacities{4u, 5u, 6u, 7u};
    auto allStages = collectPrefillStagesByCapacity<float>(queue, exec, model, prompt, capacities);

    auto const stageCount = allStages.front().size();
    for(std::size_t stageIndex = 0; stageIndex < stageCount; ++stageIndex)
        requireStageParity<float>(allStages, capacities, stageIndex, "prefill stage parity");
}

TEMPLATE_LIST_TEST_CASE("tiny llama embedding stage is independent of cache capacity", "[model][decoder]", TestApis)
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
    std::vector<uint32_t> capacities{4u, 5u, 6u, 7u};
    auto allStages = collectPrefillStagesByCapacity<float>(queue, exec, model, prompt, capacities);
    requireStageParity<float>(allStages, capacities, 0u, "embedding parity");
}

TEMPLATE_LIST_TEST_CASE("tiny llama per-layer outputs are independent of cache capacity", "[model][decoder]", TestApis)
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
    std::vector<uint32_t> capacities{4u, 5u, 6u, 7u};
    auto allStages = collectPrefillStagesByCapacity<float>(queue, exec, model, prompt, capacities);

    for(uint32_t layer = 0u; layer < model.config.numLayers; ++layer)
        requireStageParity<float>(allStages, capacities, static_cast<std::size_t>(1u + layer), "layer output parity");
}

TEMPLATE_LIST_TEST_CASE("tiny llama final norm is independent of cache capacity", "[model][decoder]", TestApis)
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
    std::vector<uint32_t> capacities{4u, 5u, 6u, 7u};
    auto allStages = collectPrefillStagesByCapacity<float>(queue, exec, model, prompt, capacities);
    requireStageParity<float>(
        allStages,
        capacities,
        static_cast<std::size_t>(1u + model.config.numLayers),
        "final norm parity");
}

TEMPLATE_LIST_TEST_CASE("tiny llama lm head logits are independent of cache capacity", "[model][decoder]", TestApis)
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
    std::vector<uint32_t> capacities{4u, 5u, 6u, 7u};
    auto allStages = collectPrefillStagesByCapacity<float>(queue, exec, model, prompt, capacities);
    requireStageParity<float>(
        allStages,
        capacities,
        static_cast<std::size_t>(2u + model.config.numLayers),
        "lm head full logits parity");
    requireStageParity<float>(
        allStages,
        capacities,
        static_cast<std::size_t>(3u + model.config.numLayers),
        "lm head last logits parity");
}

TEMPLATE_LIST_TEST_CASE("tiny llama decode stages are independent of cache capacity", "[model][decoder]", TestApis)
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
    auto hostImplicit = alpaka::onHost::allocHost<float>(implicitLogits.getExtents());
    alpaka::onHost::memcpy(queue, hostImplicit, implicitLogits);
    alpaka::onHost::wait(queue);
    auto nextToken = alpaka::nn::onHost::inference::argmax<float>(hostImplicit, 0u);

    std::vector<uint32_t> capacities{6u, 7u, 8u};
    std::vector<std::vector<StageSnapshot<float>>> allStages;
    allStages.reserve(capacities.size());
    for(auto capacity : capacities)
        allStages.push_back(runDecodeStages<float>(
            queue,
            exec,
            model,
            prompt,
            nextToken,
            capacity,
            "decode capacity " + std::to_string(capacity)));

    auto const stageCount = allStages.front().size();
    for(std::size_t stageIndex = 0; stageIndex < stageCount; ++stageIndex)
        requireStageParity<float>(allStages, capacities, stageIndex, "decode stage parity");
}
