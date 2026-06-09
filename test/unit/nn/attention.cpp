/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpakaNN/alpakaNN.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using TestApis = alpakaNN::test::TestApis;

TEMPLATE_LIST_TEST_CASE("attention scores and apply are deterministic", "[nn][attention]", TestApis)
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

    auto q = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 2u, 1u, 2u});
    auto k = alpaka::onHost::allocHost<float>(q.getExtents());
    auto v = alpaka::onHost::allocHost<float>(q.getExtents());
    auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 2u, 2u});
    auto probs = alpaka::onHost::allocHost<float>(scores.getExtents());
    auto out = alpaka::onHost::allocHost<float>(q.getExtents());
    for(auto idx : alpaka::IdxRange{q.getExtents()})
    {
        q[idx] = static_cast<float>(idx[1] + idx[3] + 1u);
        k[idx] = static_cast<float>(idx[1] + idx[3] + 2u);
        v[idx] = static_cast<float>(idx[1] * 2u + idx[3] + 1u);
    }

    auto devQ = alpaka::onHost::allocLike(device, q);
    auto devK = alpaka::onHost::allocLike(device, k);
    auto devV = alpaka::onHost::allocLike(device, v);
    auto devScores = alpaka::onHost::allocLike(device, scores);
    auto devProbs = alpaka::onHost::allocLike(device, probs);
    auto devOut = alpaka::onHost::allocLike(device, out);
    alpaka::onHost::memcpy(queue, devQ, q);
    alpaka::onHost::memcpy(queue, devK, k);
    alpaka::onHost::memcpy(queue, devV, v);

    alpakaNN::nn::attentionScores<float>(queue, exec, devQ, devK, devScores);
    alpakaNN::nn::softmax<float>(queue, exec, devScores, devProbs, 3u);
    alpakaNN::nn::attentionApply<float>(queue, exec, devProbs, devV, devOut);
    alpaka::onHost::memcpy(queue, scores, devScores);
    alpaka::onHost::memcpy(queue, probs, devProbs);
    alpaka::onHost::memcpy(queue, out, devOut);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{scores.getExtents()})
        REQUIRE(std::isfinite(scores[idx]));
    for(auto idx : alpaka::IdxRange{out.getExtents()})
        REQUIRE(std::isfinite(out[idx]));
}
