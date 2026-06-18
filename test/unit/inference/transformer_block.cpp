/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    auto makeMatrix(uint32_t rows, uint32_t cols, float scale)
    {
        auto host = alpaka::onHost::allocHost<float>(alpaka::Vec{rows, cols});
        for(auto idx : alpaka::IdxRange{host.getExtents()})
            host[idx] = scale * static_cast<float>(idx[0] + idx[1] + 1u);
        return host;
    }

    auto makeInput(uint32_t tokens, uint32_t hidden)
    {
        auto input = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens, hidden});
        for(auto idx : alpaka::IdxRange{input.getExtents()})
            input[idx] = 0.1f * static_cast<float>(idx[0] * hidden + idx[1] + 1u);
        return input;
    }

    auto makeUnitVector(uint32_t hidden)
    {
        auto weight = alpaka::onHost::allocHost<float>(hidden);
        for(uint32_t i = 0u; i < hidden; ++i)
            weight[alpaka::Vec{i}] = 1.0f;
        return weight;
    }

    auto makeRopeTables(uint32_t tokens, uint32_t halfDim)
    {
        auto cosTable = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens, halfDim});
        auto sinTable = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens, halfDim});
        for(uint32_t pos = 0u; pos < tokens; ++pos)
        {
            for(uint32_t d = 0u; d < halfDim; ++d)
            {
                auto const angle = 0.1f * static_cast<float>((pos + 1u) * (d + 1u));
                cosTable[alpaka::Vec{pos, d}] = std::cos(angle);
                sinTable[alpaka::Vec{pos, d}] = std::sin(angle);
            }
        }
        return std::pair{cosTable, sinTable};
    }
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "transformer block runs end to end on tiny dimensions",
    "[inference][transformer-block]",
    TestApis)
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

    constexpr uint32_t tokens = 2u;
    constexpr uint32_t hidden = 8u;
    constexpr uint32_t heads = 4u;
    constexpr uint32_t kvHeads = 2u;
    constexpr uint32_t headDim = 2u;
    constexpr uint32_t kvWidth = kvHeads * headDim;
    constexpr uint32_t intermediate = 10u;

    auto input = makeInput(tokens, hidden);
    auto rms1 = makeUnitVector(hidden);
    auto rms2 = makeUnitVector(hidden);
    auto Wq = makeMatrix(hidden, hidden, 0.02f);
    auto Wk = makeMatrix(hidden, kvWidth, 0.03f);
    auto Wv = makeMatrix(hidden, kvWidth, 0.04f);
    auto Wo = makeMatrix(hidden, hidden, 0.05f);
    auto Wgate = makeMatrix(hidden, intermediate, 0.02f);
    auto Wup = makeMatrix(hidden, intermediate, 0.03f);
    auto Wdown = makeMatrix(intermediate, hidden, 0.01f);
    auto [cosTable, sinTable] = makeRopeTables(tokens, headDim / 2u);

    auto devInput = alpaka::onHost::allocLike(device, input);
    auto devOutput = alpaka::onHost::allocLike(device, input);
    auto devRms1 = alpaka::onHost::allocLike(device, rms1);
    auto devRms2 = alpaka::onHost::allocLike(device, rms2);
    auto devWq = alpaka::onHost::allocLike(device, Wq);
    auto devWk = alpaka::onHost::allocLike(device, Wk);
    auto devWv = alpaka::onHost::allocLike(device, Wv);
    auto devWo = alpaka::onHost::allocLike(device, Wo);
    auto devWgate = alpaka::onHost::allocLike(device, Wgate);
    auto devWup = alpaka::onHost::allocLike(device, Wup);
    auto devWdown = alpaka::onHost::allocLike(device, Wdown);
    auto devCos = alpaka::onHost::allocLike(device, cosTable);
    auto devSin = alpaka::onHost::allocLike(device, sinTable);
    alpaka::onHost::memcpy(queue, devInput, input);
    alpaka::onHost::memcpy(queue, devRms1, rms1);
    alpaka::onHost::memcpy(queue, devRms2, rms2);
    alpaka::onHost::memcpy(queue, devWq, Wq);
    alpaka::onHost::memcpy(queue, devWk, Wk);
    alpaka::onHost::memcpy(queue, devWv, Wv);
    alpaka::onHost::memcpy(queue, devWo, Wo);
    alpaka::onHost::memcpy(queue, devWgate, Wgate);
    alpaka::onHost::memcpy(queue, devWup, Wup);
    alpaka::onHost::memcpy(queue, devWdown, Wdown);
    alpaka::onHost::memcpy(queue, devCos, cosTable);
    alpaka::onHost::memcpy(queue, devSin, sinTable);

    alpaka::nn::onHost::inference::TransformerBlockWeights<float, decltype(devRms1), decltype(devWq)> weights{
        devRms1,
        devRms2,
        devWq,
        devWk,
        devWv,
        devWo,
        devWgate,
        devWup,
        devWdown,
        heads,
        kvHeads,
        headDim,
        1.0e-5f};
    auto cache = alpaka::nn::onHost::inference::makeKvCache<float>(device, 1u, 1u, kvHeads, tokens, headDim);
    alpaka::nn::onHost::inference::transformerBlock<
        float>(queue, exec, devInput, weights, cache, 0u, devCos, devSin, devOutput);
    alpaka::onHost::memcpy(queue, input, devOutput);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{input.getExtents()})
        REQUIRE(std::isfinite(input[idx]));
    REQUIRE(cache.length(0u, 0u) == tokens);
    REQUIRE(cache.getKeys(0u, 0u, tokens).getExtents()[1] == kvHeads);
}

TEMPLATE_LIST_TEST_CASE(
    "transformer block output is independent of spare cache capacity",
    "[inference][transformer-block][decoder]",
    TestApis)
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

    constexpr uint32_t tokens = 4u;
    constexpr uint32_t hidden = 16u;
    constexpr uint32_t heads = 4u;
    constexpr uint32_t kvHeads = 4u;
    constexpr uint32_t headDim = 4u;
    constexpr uint32_t kvWidth = kvHeads * headDim;
    constexpr uint32_t intermediate = 24u;

    auto inputA = makeInput(tokens, hidden);
    auto inputB = makeInput(tokens, hidden);
    auto outputA = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens, hidden});
    auto outputB = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens, hidden});
    auto rms1 = makeUnitVector(hidden);
    auto rms2 = makeUnitVector(hidden);
    auto Wq = makeMatrix(hidden, hidden, 0.0075f);
    auto Wk = makeMatrix(hidden, kvWidth, 0.011f);
    auto Wv = makeMatrix(hidden, kvWidth, -0.009f);
    auto Wo = makeMatrix(hidden, hidden, 0.0065f);
    auto Wgate = makeMatrix(hidden, intermediate, 0.004f);
    auto Wup = makeMatrix(hidden, intermediate, -0.003f);
    auto Wdown = makeMatrix(intermediate, hidden, 0.0025f);
    auto [cosTable, sinTable] = makeRopeTables(tokens, headDim / 2u);

    auto devInputA = alpaka::onHost::allocLike(device, inputA);
    auto devInputB = alpaka::onHost::allocLike(device, inputB);
    auto devOutputA = alpaka::onHost::allocLike(device, outputA);
    auto devOutputB = alpaka::onHost::allocLike(device, outputB);
    auto devRms1 = alpaka::onHost::allocLike(device, rms1);
    auto devRms2 = alpaka::onHost::allocLike(device, rms2);
    auto devWq = alpaka::onHost::allocLike(device, Wq);
    auto devWk = alpaka::onHost::allocLike(device, Wk);
    auto devWv = alpaka::onHost::allocLike(device, Wv);
    auto devWo = alpaka::onHost::allocLike(device, Wo);
    auto devWgate = alpaka::onHost::allocLike(device, Wgate);
    auto devWup = alpaka::onHost::allocLike(device, Wup);
    auto devWdown = alpaka::onHost::allocLike(device, Wdown);
    auto devCos = alpaka::onHost::allocLike(device, cosTable);
    auto devSin = alpaka::onHost::allocLike(device, sinTable);
    alpaka::onHost::memcpy(queue, devInputA, inputA);
    alpaka::onHost::memcpy(queue, devInputB, inputB);
    alpaka::onHost::memcpy(queue, devRms1, rms1);
    alpaka::onHost::memcpy(queue, devRms2, rms2);
    alpaka::onHost::memcpy(queue, devWq, Wq);
    alpaka::onHost::memcpy(queue, devWk, Wk);
    alpaka::onHost::memcpy(queue, devWv, Wv);
    alpaka::onHost::memcpy(queue, devWo, Wo);
    alpaka::onHost::memcpy(queue, devWgate, Wgate);
    alpaka::onHost::memcpy(queue, devWup, Wup);
    alpaka::onHost::memcpy(queue, devWdown, Wdown);
    alpaka::onHost::memcpy(queue, devCos, cosTable);
    alpaka::onHost::memcpy(queue, devSin, sinTable);

    alpaka::nn::onHost::inference::TransformerBlockWeights<float, decltype(devRms1), decltype(devWq)> weights{
        devRms1,
        devRms2,
        devWq,
        devWk,
        devWv,
        devWo,
        devWgate,
        devWup,
        devWdown,
        heads,
        kvHeads,
        headDim,
        1.0e-5f};
    auto compactCache = alpaka::nn::onHost::inference::makeKvCache<float>(device, 1u, 1u, kvHeads, tokens, headDim);
    auto spareCache = alpaka::nn::onHost::inference::makeKvCache<float>(device, 1u, 1u, kvHeads, tokens + 2u, headDim);

    alpaka::nn::onHost::inference::transformerBlock<float>(
        queue,
        exec,
        devInputA,
        weights,
        compactCache,
        0u,
        devCos,
        devSin,
        devOutputA);
    alpaka::nn::onHost::inference::transformerBlock<float>(
        queue,
        exec,
        devInputB,
        weights,
        spareCache,
        0u,
        devCos,
        devSin,
        devOutputB);
    alpaka::onHost::memcpy(queue, outputA, devOutputA);
    alpaka::onHost::memcpy(queue, outputB, devOutputB);
    alpaka::onHost::wait(queue);

    REQUIRE(compactCache.length(0u, 0u) == tokens);
    REQUIRE(spareCache.length(0u, 0u) == tokens);
    for(auto idx : alpaka::IdxRange{outputA.getExtents()})
        alpaka::nn::test::checkValue(outputA[idx], outputB[idx], 1.0e-5f, 1.0e-5f);
}
