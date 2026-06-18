/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    template<typename T_Type>
    void fill4D(auto& tensor, T_Type base)
    {
        for(auto idx : alpaka::IdxRange{tensor.getExtents()})
            tensor[idx] = base + static_cast<T_Type>(idx[0] * 101u + idx[1] * 17u + idx[2] * 7u + idx[3] * 3u);
    }

    template<typename T_Type>
    auto referenceScoresBTHD(auto const& q, auto const& k, uint32_t queriesPerKvGroup)
    {
        auto expected = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{
            static_cast<uint32_t>(q.getExtents()[0]),
            static_cast<uint32_t>(q.getExtents()[2]),
            static_cast<uint32_t>(q.getExtents()[1]),
            static_cast<uint32_t>(k.getExtents()[1])});
        for(auto idx : alpaka::IdxRange{expected.getExtents()})
        {
            T_Type sum{};
            auto const kvHead = static_cast<uint32_t>(idx[1]) / queriesPerKvGroup;
            for(uint32_t d = 0u; d < q.getExtents()[3]; ++d)
                sum += q[alpaka::Vec{idx[0], idx[2], idx[1], d}] * k[alpaka::Vec{idx[0], idx[3], kvHead, d}];
            expected[idx] = sum;
        }
        return expected;
    }

    template<typename T_Type>
    auto referenceScoresBHTD(auto const& q, auto const& k, uint32_t queriesPerKvGroup)
    {
        auto expected = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{
            static_cast<uint32_t>(q.getExtents()[0]),
            static_cast<uint32_t>(q.getExtents()[2]),
            static_cast<uint32_t>(q.getExtents()[1]),
            static_cast<uint32_t>(k.getExtents()[2])});
        for(auto idx : alpaka::IdxRange{expected.getExtents()})
        {
            T_Type sum{};
            auto const kvHead = static_cast<uint32_t>(idx[1]) / queriesPerKvGroup;
            for(uint32_t d = 0u; d < q.getExtents()[3]; ++d)
                sum += q[alpaka::Vec{idx[0], idx[2], idx[1], d}] * k[alpaka::Vec{idx[0], kvHead, idx[3], d}];
            expected[idx] = sum;
        }
        return expected;
    }

    template<typename T_Type>
    auto referenceApplyBHTD(auto const& probs, auto const& values, uint32_t queriesPerKvGroup)
    {
        auto expected = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{
            static_cast<uint32_t>(probs.getExtents()[0]),
            static_cast<uint32_t>(probs.getExtents()[2]),
            static_cast<uint32_t>(probs.getExtents()[1]),
            static_cast<uint32_t>(values.getExtents()[3])});
        for(auto idx : alpaka::IdxRange{expected.getExtents()})
        {
            T_Type sum{};
            auto const kvHead = static_cast<uint32_t>(idx[2]) / queriesPerKvGroup;
            for(uint32_t key = 0u; key < probs.getExtents()[3]; ++key)
                sum += probs[alpaka::Vec{idx[0], idx[2], idx[1], key}] * values[alpaka::Vec{idx[0], kvHead, key, idx[3]}];
            expected[idx] = sum;
        }
        return expected;
    }
} // namespace

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

    alpaka::nn::onHost::nn::attentionScores<float>(queue, exec, devQ, devK, devScores);
    alpaka::nn::onHost::nn::softmax<float>(queue, exec, devScores, devProbs, 3u);
    alpaka::nn::onHost::nn::attentionApply<float>(queue, exec, devProbs, devV, devOut);
    alpaka::onHost::memcpy(queue, scores, devScores);
    alpaka::onHost::memcpy(queue, probs, devProbs);
    alpaka::onHost::memcpy(queue, out, devOut);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{scores.getExtents()})
        REQUIRE(std::isfinite(scores[idx]));
    for(auto idx : alpaka::IdxRange{out.getExtents()})
        REQUIRE(std::isfinite(out[idx]));
}

TEMPLATE_LIST_TEST_CASE("attention maps grouped query heads onto fewer kv heads", "[nn][attention]", TestApis)
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

    auto q = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 4u, 2u});
    auto k = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 2u, 2u});
    auto v = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 2u, 2u});
    auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 1u, 1u});
    auto probs = alpaka::onHost::allocHost<float>(scores.getExtents());
    auto out = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 4u, 2u});

    for(uint32_t head = 0u; head < 4u; ++head)
    {
        q[alpaka::Vec{0u, 0u, head, 0u}] = static_cast<float>(head + 1u);
        q[alpaka::Vec{0u, 0u, head, 1u}] = static_cast<float>(head + 2u);
    }
    k[alpaka::Vec{0u, 0u, 0u, 0u}] = 1.0f;
    k[alpaka::Vec{0u, 0u, 0u, 1u}] = 2.0f;
    k[alpaka::Vec{0u, 0u, 1u, 0u}] = 3.0f;
    k[alpaka::Vec{0u, 0u, 1u, 1u}] = 4.0f;
    v[alpaka::Vec{0u, 0u, 0u, 0u}] = 10.0f;
    v[alpaka::Vec{0u, 0u, 0u, 1u}] = 20.0f;
    v[alpaka::Vec{0u, 0u, 1u, 0u}] = 30.0f;
    v[alpaka::Vec{0u, 0u, 1u, 1u}] = 40.0f;

    auto devQ = alpaka::onHost::allocLike(device, q);
    auto devK = alpaka::onHost::allocLike(device, k);
    auto devV = alpaka::onHost::allocLike(device, v);
    auto devScores = alpaka::onHost::allocLike(device, scores);
    auto devProbs = alpaka::onHost::allocLike(device, probs);
    auto devOut = alpaka::onHost::allocLike(device, out);
    alpaka::onHost::memcpy(queue, devQ, q);
    alpaka::onHost::memcpy(queue, devK, k);
    alpaka::onHost::memcpy(queue, devV, v);

    alpaka::nn::onHost::nn::attentionScores<float>(
        queue,
        exec,
        devQ,
        devK,
        devScores,
        2u,
        alpaka::nn::AttentionKvLayout::BTHD);
    alpaka::nn::onHost::nn::softmax<float>(queue, exec, devScores, devProbs, 3u);
    alpaka::nn::onHost::nn::attentionApply<float>(
        queue,
        exec,
        devProbs,
        devV,
        devOut,
        2u,
        alpaka::nn::AttentionKvLayout::BTHD);
    alpaka::onHost::memcpy(queue, scores, devScores);
    alpaka::onHost::memcpy(queue, out, devOut);
    alpaka::onHost::wait(queue);

    alpaka::nn::test::checkValue(scores[alpaka::Vec{0u, 0u, 0u, 0u}], 5.0f);
    alpaka::nn::test::checkValue(scores[alpaka::Vec{0u, 1u, 0u, 0u}], 8.0f);
    alpaka::nn::test::checkValue(scores[alpaka::Vec{0u, 2u, 0u, 0u}], 25.0f);
    alpaka::nn::test::checkValue(scores[alpaka::Vec{0u, 3u, 0u, 0u}], 32.0f);
    for(uint32_t head = 0u; head < 2u; ++head)
    {
        alpaka::nn::test::checkValue(out[alpaka::Vec{0u, 0u, head, 0u}], 10.0f);
        alpaka::nn::test::checkValue(out[alpaka::Vec{0u, 0u, head, 1u}], 20.0f);
    }
    for(uint32_t head = 2u; head < 4u; ++head)
    {
        alpaka::nn::test::checkValue(out[alpaka::Vec{0u, 0u, head, 0u}], 30.0f);
        alpaka::nn::test::checkValue(out[alpaka::Vec{0u, 0u, head, 1u}], 40.0f);
    }
}

TEMPLATE_LIST_TEST_CASE("attention scores match reference for decoder prefill shape", "[nn][attention][decoder]", TestApis)
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

    auto q = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    auto k = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    fill4D(q, 0.25f);
    fill4D(k, -0.5f);
    for(auto idx : alpaka::IdxRange{scores.getExtents()})
        scores[idx] = -999.0f;

    auto expected = referenceScoresBTHD<float>(q, k, 1u);
    auto devQ = alpaka::onHost::allocLike(device, q);
    auto devK = alpaka::onHost::allocLike(device, k);
    auto devScores = alpaka::onHost::allocLike(device, scores);
    alpaka::onHost::memcpy(queue, devQ, q);
    alpaka::onHost::memcpy(queue, devK, k);
    alpaka::onHost::memcpy(queue, devScores, scores);

    alpaka::nn::onHost::nn::attentionScores<float>(
        queue,
        exec,
        devQ,
        devK,
        devScores,
        1u,
        alpaka::nn::AttentionKvLayout::BTHD);
    alpaka::onHost::memcpy(queue, scores, devScores);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{scores.getExtents()})
        alpaka::nn::test::checkValue(scores[idx], expected[idx], 1.0e-5f, 1.0e-5f);
}

TEMPLATE_LIST_TEST_CASE("attention scores match reference for decoder decode layout", "[nn][attention][decoder]", TestApis)
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

    auto q = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 4u, 4u});
    auto k = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 5u, 4u});
    auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 1u, 5u});
    fill4D(q, 0.75f);
    fill4D(k, -0.125f);
    for(auto idx : alpaka::IdxRange{scores.getExtents()})
        scores[idx] = -777.0f;

    auto expected = referenceScoresBHTD<float>(q, k, 1u);
    auto devQ = alpaka::onHost::allocLike(device, q);
    auto devK = alpaka::onHost::allocLike(device, k);
    auto devScores = alpaka::onHost::allocLike(device, scores);
    alpaka::onHost::memcpy(queue, devQ, q);
    alpaka::onHost::memcpy(queue, devK, k);
    alpaka::onHost::memcpy(queue, devScores, scores);

    alpaka::nn::onHost::nn::attentionScores<float>(
        queue,
        exec,
        devQ,
        devK,
        devScores,
        1u,
        alpaka::nn::AttentionKvLayout::BHTD);
    alpaka::onHost::memcpy(queue, scores, devScores);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{scores.getExtents()})
        alpaka::nn::test::checkValue(scores[idx], expected[idx], 1.0e-5f, 1.0e-5f);
}

TEMPLATE_LIST_TEST_CASE("attention apply matches reference for decoder decode layout", "[nn][attention][decoder]", TestApis)
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

    auto probs = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 1u, 5u});
    auto values = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 5u, 4u});
    auto out = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 4u, 4u});
    fill4D(values, 1.25f);
    for(uint32_t head = 0u; head < 4u; ++head)
    {
        float norm = 0.0f;
        for(uint32_t key = 0u; key < 5u; ++key)
        {
            auto value = static_cast<float>((head + 1u) * (key + 2u));
            probs[alpaka::Vec{0u, head, 0u, key}] = value;
            norm += value;
        }
        for(uint32_t key = 0u; key < 5u; ++key)
            probs[alpaka::Vec{0u, head, 0u, key}] /= norm;
    }
    for(auto idx : alpaka::IdxRange{out.getExtents()})
        out[idx] = -555.0f;

    auto expected = referenceApplyBHTD<float>(probs, values, 1u);
    auto devProbs = alpaka::onHost::allocLike(device, probs);
    auto devValues = alpaka::onHost::allocLike(device, values);
    auto devOut = alpaka::onHost::allocLike(device, out);
    alpaka::onHost::memcpy(queue, devProbs, probs);
    alpaka::onHost::memcpy(queue, devValues, values);
    alpaka::onHost::memcpy(queue, devOut, out);

    alpaka::nn::onHost::nn::attentionApply<float>(
        queue,
        exec,
        devProbs,
        devValues,
        devOut,
        1u,
        alpaka::nn::AttentionKvLayout::BHTD);
    alpaka::onHost::memcpy(queue, out, devOut);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{out.getExtents()})
        alpaka::nn::test::checkValue(out[idx], expected[idx], 1.0e-5f, 1.0e-5f);
}
