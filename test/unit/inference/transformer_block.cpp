/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpakaNN/alpakaNN.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using TestApis = alpakaNN::test::TestApis;

TEMPLATE_LIST_TEST_CASE("transformer block runs end to end on tiny dimensions", "[inference][transformer-block]", TestApis)
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

    auto makeMatrix = [&](uint32_t rows, uint32_t cols, float scale)
    {
        auto host = alpaka::onHost::allocHost<float>(alpaka::Vec{rows, cols});
        for(auto idx : alpaka::IdxRange{host.getExtents()})
            host[idx] = scale * static_cast<float>(idx[0] + idx[1] + 1u);
        return host;
    };

    auto input = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens, hidden});
    auto rms1 = alpaka::onHost::allocHost<float>(hidden);
    auto rms2 = alpaka::onHost::allocHost<float>(hidden);
    for(auto idx : alpaka::IdxRange{input.getExtents()})
        input[idx] = 0.1f * static_cast<float>(idx[0] * hidden + idx[1] + 1u);
    for(uint32_t i = 0u; i < hidden; ++i)
    {
        rms1[alpaka::Vec{i}] = 1.0f;
        rms2[alpaka::Vec{i}] = 1.0f;
    }
    auto Wq = makeMatrix(hidden, hidden, 0.02f);
    auto Wk = makeMatrix(hidden, kvWidth, 0.03f);
    auto Wv = makeMatrix(hidden, kvWidth, 0.04f);
    auto Wo = makeMatrix(hidden, hidden, 0.05f);
    auto Wgate = makeMatrix(hidden, intermediate, 0.02f);
    auto Wup = makeMatrix(hidden, intermediate, 0.03f);
    auto Wdown = makeMatrix(intermediate, hidden, 0.01f);
    auto cosTable = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens, headDim / 2u});
    auto sinTable = alpaka::onHost::allocHost<float>(cosTable.getExtents());
    for(uint32_t pos = 0u; pos < tokens; ++pos)
    {
        cosTable[alpaka::Vec{pos, 0u}] = std::cos(0.1f * static_cast<float>(pos + 1u));
        sinTable[alpaka::Vec{pos, 0u}] = std::sin(0.1f * static_cast<float>(pos + 1u));
    }

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

    alpakaNN::inference::TransformerBlockWeights<float, decltype(devRms1), decltype(devWq)> weights{
        devRms1, devRms2, devWq, devWk, devWv, devWo, devWgate, devWup, devWdown, heads, kvHeads, headDim, 1.0e-5f};
    auto cache = alpakaNN::inference::makeKvCache<float>(device, 1u, 1u, kvHeads, tokens, headDim);
    alpakaNN::inference::transformerBlock<float>(queue, exec, devInput, weights, cache, 0u, devCos, devSin, devOutput);
    alpaka::onHost::memcpy(queue, input, devOutput);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{input.getExtents()})
        REQUIRE(std::isfinite(input[idx]));
    REQUIRE(cache.length(0u, 0u) == tokens);
    REQUIRE(cache.getKeys(0u, 0u, tokens).getExtents()[1] == kvHeads);
}
