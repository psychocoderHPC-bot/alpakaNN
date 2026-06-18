/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>
#include <alpaka/nn/onHost/inference/kv_cache.hpp>
#include <alpaka/nn/onHost/model/decoder.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
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

    template<typename T_Type>
    auto referenceApplyBTHD(auto const& probs, auto const& values, uint32_t queriesPerKvGroup)
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
                sum += probs[alpaka::Vec{idx[0], idx[2], idx[1], key}] * values[alpaka::Vec{idx[0], key, kvHead, idx[3]}];
            expected[idx] = sum;
        }
        return expected;
    }

    template<typename T_Type>
    void scaleTensor(auto& tensor, T_Type factor)
    {
        for(auto idx : alpaka::IdxRange{tensor.getExtents()})
            tensor[idx] *= factor;
    }

    void referenceCausalSoftmaxInPlace(auto& tensor)
    {
        auto const extents = tensor.getExtents();
        using T_Type = std::remove_cvref_t<decltype(tensor[alpaka::Vec{0u, 0u, 0u, 0u}])>;
        for(uint32_t batch = 0u; batch < extents[0]; ++batch)
        {
            for(uint32_t head = 0u; head < extents[1]; ++head)
            {
                for(uint32_t query = 0u; query < extents[2]; ++query)
                {
                    T_Type maxValue = -std::numeric_limits<T_Type>::infinity();
                    for(uint32_t key = 0u; key <= query; ++key)
                        maxValue = std::max(maxValue, tensor[alpaka::Vec{batch, head, query, key}]);

                    T_Type sum{};
                    for(uint32_t key = 0u; key < extents[3]; ++key)
                    {
                        if(key > query)
                        {
                            tensor[alpaka::Vec{batch, head, query, key}] = T_Type{};
                            continue;
                        }
                        auto const value = std::exp(tensor[alpaka::Vec{batch, head, query, key}] - maxValue);
                        tensor[alpaka::Vec{batch, head, query, key}] = value;
                        sum += value;
                    }
                    for(uint32_t key = 0u; key <= query; ++key)
                        tensor[alpaka::Vec{batch, head, query, key}] /= sum;
                }
            }
        }
    }

    void referenceSoftmaxInPlace(auto& tensor)
    {
        auto const extents = tensor.getExtents();
        using T_Type = std::remove_cvref_t<decltype(tensor[alpaka::Vec{0u, 0u, 0u, 0u}])>;
        for(uint32_t batch = 0u; batch < extents[0]; ++batch)
        {
            for(uint32_t head = 0u; head < extents[1]; ++head)
            {
                for(uint32_t query = 0u; query < extents[2]; ++query)
                {
                    T_Type maxValue = -std::numeric_limits<T_Type>::infinity();
                    for(uint32_t key = 0u; key < extents[3]; ++key)
                        maxValue = std::max(maxValue, tensor[alpaka::Vec{batch, head, query, key}]);

                    T_Type sum{};
                    for(uint32_t key = 0u; key < extents[3]; ++key)
                    {
                        auto const value = std::exp(tensor[alpaka::Vec{batch, head, query, key}] - maxValue);
                        tensor[alpaka::Vec{batch, head, query, key}] = value;
                        sum += value;
                    }
                    for(uint32_t key = 0u; key < extents[3]; ++key)
                        tensor[alpaka::Vec{batch, head, query, key}] /= sum;
                }
            }
        }
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

TEMPLATE_LIST_TEST_CASE("attention apply matches reference for decoder prefill layout", "[nn][attention][decoder]", TestApis)
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

    auto probs = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    auto values = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    auto out = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    fill4D(values, 1.0f);
    for(uint32_t query = 0u; query < 4u; ++query)
    {
        for(uint32_t head = 0u; head < 4u; ++head)
        {
            float norm = 0.0f;
            for(uint32_t key = 0u; key < 4u; ++key)
            {
                auto value = key <= query ? static_cast<float>((query + 1u) * (head + 2u) * (key + 1u)) : 0.0f;
                probs[alpaka::Vec{0u, head, query, key}] = value;
                norm += value;
            }
            for(uint32_t key = 0u; key < 4u; ++key)
                probs[alpaka::Vec{0u, head, query, key}] = norm > 0.0f ? probs[alpaka::Vec{0u, head, query, key}] / norm : 0.0f;
        }
    }
    for(auto idx : alpaka::IdxRange{out.getExtents()})
        out[idx] = -444.0f;

    auto expected = referenceApplyBTHD<float>(probs, values, 1u);
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
        alpaka::nn::AttentionKvLayout::BTHD);
    alpaka::onHost::memcpy(queue, out, devOut);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{out.getExtents()})
        alpaka::nn::test::checkValue(out[idx], expected[idx], 1.0e-5f, 1.0e-5f);
}

TEMPLATE_LIST_TEST_CASE("decoder prefill attention pipeline matches reference", "[nn][attention][decoder]", TestApis)
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
    auto v = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    auto probs = alpaka::onHost::allocHost<float>(scores.getExtents());
    auto out = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 4u, 4u});
    fill4D(q, 0.125f);
    fill4D(k, -0.25f);
    fill4D(v, 0.5f);

    auto expectedScores = referenceScoresBTHD<float>(q, k, 1u);
    scaleTensor(expectedScores, 0.5f);
    auto expectedProbs = alpaka::onHost::allocHost<float>(expectedScores.getExtents());
    for(auto idx : alpaka::IdxRange{expectedScores.getExtents()})
        expectedProbs[idx] = expectedScores[idx];
    referenceCausalSoftmaxInPlace(expectedProbs);
    auto expectedOut = referenceApplyBTHD<float>(expectedProbs, v, 1u);

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
        1u,
        alpaka::nn::AttentionKvLayout::BTHD);
    alpaka::nn::onHost::ops::scale<float>(queue, exec, devScores, 0.5f, devScores);
    alpaka::nn::onHost::nn::causalSoftmax<float>(queue, exec, devScores, devProbs, 3u, 2u, 3u);
    alpaka::nn::onHost::nn::attentionApply<float>(
        queue,
        exec,
        devProbs,
        devV,
        devOut,
        1u,
        alpaka::nn::AttentionKvLayout::BTHD);
    alpaka::onHost::memcpy(queue, scores, devScores);
    alpaka::onHost::memcpy(queue, probs, devProbs);
    alpaka::onHost::memcpy(queue, out, devOut);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{scores.getExtents()})
        alpaka::nn::test::checkValue(scores[idx], expectedScores[idx], 1.0e-5f, 1.0e-5f);
    for(auto idx : alpaka::IdxRange{probs.getExtents()})
        alpaka::nn::test::checkValue(probs[idx], expectedProbs[idx], 1.0e-5f, 1.0e-5f);
    for(auto idx : alpaka::IdxRange{out.getExtents()})
        alpaka::nn::test::checkValue(out[idx], expectedOut[idx], 1.0e-5f, 1.0e-5f);
}

TEMPLATE_LIST_TEST_CASE("decoder decode attention pipeline matches reference", "[nn][attention][decoder]", TestApis)
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
    auto v = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 5u, 4u});
    auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 1u, 5u});
    auto probs = alpaka::onHost::allocHost<float>(scores.getExtents());
    auto out = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 4u, 4u});
    fill4D(q, 0.375f);
    fill4D(k, -0.0625f);
    fill4D(v, 0.875f);

    auto expectedScores = referenceScoresBHTD<float>(q, k, 1u);
    scaleTensor(expectedScores, 0.5f);
    auto expectedProbs = alpaka::onHost::allocHost<float>(expectedScores.getExtents());
    for(auto idx : alpaka::IdxRange{expectedScores.getExtents()})
        expectedProbs[idx] = expectedScores[idx];
    referenceSoftmaxInPlace(expectedProbs);
    auto expectedOut = referenceApplyBHTD<float>(expectedProbs, v, 1u);

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
        1u,
        alpaka::nn::AttentionKvLayout::BHTD);
    alpaka::nn::onHost::ops::scale<float>(queue, exec, devScores, 0.5f, devScores);
    alpaka::nn::onHost::nn::softmax<float>(queue, exec, devScores, devProbs, 3u);
    alpaka::nn::onHost::nn::attentionApply<float>(
        queue,
        exec,
        devProbs,
        devV,
        devOut,
        1u,
        alpaka::nn::AttentionKvLayout::BHTD);
    alpaka::onHost::memcpy(queue, scores, devScores);
    alpaka::onHost::memcpy(queue, probs, devProbs);
    alpaka::onHost::memcpy(queue, out, devOut);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{scores.getExtents()})
        alpaka::nn::test::checkValue(scores[idx], expectedScores[idx], 1.0e-5f, 1.0e-5f);
    for(auto idx : alpaka::IdxRange{probs.getExtents()})
        alpaka::nn::test::checkValue(probs[idx], expectedProbs[idx], 1.0e-5f, 1.0e-5f);
    for(auto idx : alpaka::IdxRange{out.getExtents()})
        alpaka::nn::test::checkValue(out[idx], expectedOut[idx], 1.0e-5f, 1.0e-5f);
}

TEMPLATE_LIST_TEST_CASE(
    "decoder decode attention pipeline matches reference for cache-backed kv views",
    "[nn][attention][decoder]",
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

    auto q = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 4u, 4u});
    auto kTokens = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 5u, 4u, 4u});
    auto vTokens = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 5u, 4u, 4u});
    auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 1u, 5u});
    auto probs = alpaka::onHost::allocHost<float>(scores.getExtents());
    auto out = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 4u, 4u});
    fill4D(q, 0.375f);
    fill4D(kTokens, -0.0625f);
    fill4D(vTokens, 0.875f);

    auto expectedK = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 5u, 4u});
    auto expectedV = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 5u, 4u});
    for(uint32_t token = 0u; token < 5u; ++token)
    {
        for(uint32_t head = 0u; head < 4u; ++head)
        {
            for(uint32_t d = 0u; d < 4u; ++d)
            {
                expectedK[alpaka::Vec{0u, head, token, d}] = kTokens[alpaka::Vec{0u, token, head, d}];
                expectedV[alpaka::Vec{0u, head, token, d}] = vTokens[alpaka::Vec{0u, token, head, d}];
            }
        }
    }

    auto expectedScores = referenceScoresBHTD<float>(q, expectedK, 1u);
    scaleTensor(expectedScores, 0.5f);
    auto expectedProbs = alpaka::onHost::allocHost<float>(expectedScores.getExtents());
    for(auto idx : alpaka::IdxRange{expectedScores.getExtents()})
        expectedProbs[idx] = expectedScores[idx];
    referenceSoftmaxInPlace(expectedProbs);
    auto expectedOut = referenceApplyBHTD<float>(expectedProbs, expectedV, 1u);

    auto devQ = alpaka::onHost::allocLike(device, q);
    auto devKTokens = alpaka::onHost::allocLike(device, kTokens);
    auto devVTokens = alpaka::onHost::allocLike(device, vTokens);
    auto devScores = alpaka::onHost::allocLike(device, scores);
    auto devProbs = alpaka::onHost::allocLike(device, probs);
    auto devOut = alpaka::onHost::allocLike(device, out);
    alpaka::onHost::memcpy(queue, devQ, q);
    alpaka::onHost::memcpy(queue, devKTokens, kTokens);
    alpaka::onHost::memcpy(queue, devVTokens, vTokens);

    auto cache = alpaka::nn::onHost::inference::makeKvCache<float>(device, 1u, 1u, 4u, 7u, 4u);
    for(uint32_t token = 0u; token < 5u; ++token)
    {
        auto kToken = devKTokens.getSubView(alpaka::Vec{0u, token, 0u, 0u}, alpaka::Vec{1u, 1u, 4u, 4u});
        auto vToken = devVTokens.getSubView(alpaka::Vec{0u, token, 0u, 0u}, alpaka::Vec{1u, 1u, 4u, 4u});
        cache.append(queue, exec, 0u, 0u, token, kToken, vToken);
    }
    auto keys = cache.getKeys(0u, 0u, 5u);
    auto values = cache.getValues(0u, 0u, 5u);

    alpaka::nn::onHost::nn::attentionScores<float>(
        queue,
        exec,
        devQ,
        keys,
        devScores,
        1u,
        alpaka::nn::AttentionKvLayout::BHTD);
    alpaka::nn::onHost::ops::scale<float>(queue, exec, devScores, 0.5f, devScores);
    alpaka::nn::onHost::nn::softmax<float>(queue, exec, devScores, devProbs, 3u);
    alpaka::nn::onHost::nn::attentionApply<float>(
        queue,
        exec,
        devProbs,
        values,
        devOut,
        1u,
        alpaka::nn::AttentionKvLayout::BHTD);
    alpaka::onHost::memcpy(queue, scores, devScores);
    alpaka::onHost::memcpy(queue, probs, devProbs);
    alpaka::onHost::memcpy(queue, out, devOut);
    alpaka::onHost::wait(queue);

    for(auto idx : alpaka::IdxRange{scores.getExtents()})
        alpaka::nn::test::checkValue(scores[idx], expectedScores[idx], 1.0e-5f, 1.0e-5f);
    for(auto idx : alpaka::IdxRange{probs.getExtents()})
        alpaka::nn::test::checkValue(probs[idx], expectedProbs[idx], 1.0e-5f, 1.0e-5f);
    for(auto idx : alpaka::IdxRange{out.getExtents()})
        alpaka::nn::test::checkValue(out[idx], expectedOut[idx], 1.0e-5f, 1.0e-5f);
}

TEMPLATE_LIST_TEST_CASE(
    "decoder decode attention cache-backed kv views are independent of spare capacity",
    "[nn][attention][decoder]",
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

    auto q = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 4u, 4u});
    auto kTokens = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 5u, 4u, 4u});
    auto vTokens = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 5u, 4u, 4u});
    fill4D(q, 0.375f);
    fill4D(kTokens, -0.0625f);
    fill4D(vTokens, 0.875f);

    auto devQ = alpaka::onHost::allocLike(device, q);
    auto devKTokens = alpaka::onHost::allocLike(device, kTokens);
    auto devVTokens = alpaka::onHost::allocLike(device, vTokens);
    alpaka::onHost::memcpy(queue, devQ, q);
    alpaka::onHost::memcpy(queue, devKTokens, kTokens);
    alpaka::onHost::memcpy(queue, devVTokens, vTokens);

    auto runWithCapacity = [&](uint32_t capacity)
    {
        auto cache = alpaka::nn::onHost::inference::makeKvCache<float>(device, 1u, 1u, 4u, capacity, 4u);
        for(uint32_t token = 0u; token < 5u; ++token)
        {
            auto kToken = devKTokens.getSubView(alpaka::Vec{0u, token, 0u, 0u}, alpaka::Vec{1u, 1u, 4u, 4u});
            auto vToken = devVTokens.getSubView(alpaka::Vec{0u, token, 0u, 0u}, alpaka::Vec{1u, 1u, 4u, 4u});
            cache.append(queue, exec, 0u, 0u, token, kToken, vToken);
        }

        auto keys = cache.getKeys(0u, 0u, 5u);
        auto values = cache.getValues(0u, 0u, 5u);

        auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 4u, 1u, 5u});
        auto probs = alpaka::onHost::allocHost<float>(scores.getExtents());
        auto out = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, 1u, 4u, 4u});
        auto devScores = alpaka::onHost::allocLike(device, scores);
        auto devProbs = alpaka::onHost::allocLike(device, probs);
        auto devOut = alpaka::onHost::allocLike(device, out);

        alpaka::nn::onHost::nn::attentionScores<float>(
            queue,
            exec,
            devQ,
            keys,
            devScores,
            1u,
            alpaka::nn::AttentionKvLayout::BHTD);
        alpaka::nn::onHost::ops::scale<float>(queue, exec, devScores, 0.5f, devScores);
        alpaka::nn::onHost::nn::softmax<float>(queue, exec, devScores, devProbs, 3u);
        alpaka::nn::onHost::nn::attentionApply<float>(
            queue,
            exec,
            devProbs,
            values,
            devOut,
            1u,
            alpaka::nn::AttentionKvLayout::BHTD);
        alpaka::onHost::memcpy(queue, scores, devScores);
        alpaka::onHost::memcpy(queue, probs, devProbs);
        alpaka::onHost::memcpy(queue, out, devOut);
        alpaka::onHost::wait(queue);
        return std::tuple{scores, probs, out};
    };

    auto [scores6, probs6, out6] = runWithCapacity(6u);
    auto [scores7, probs7, out7] = runWithCapacity(7u);

    for(auto idx : alpaka::IdxRange{scores6.getExtents()})
        alpaka::nn::test::checkValue(scores6[idx], scores7[idx], 1.0e-5f, 1.0e-5f);
    for(auto idx : alpaka::IdxRange{probs6.getExtents()})
        alpaka::nn::test::checkValue(probs6[idx], probs7[idx], 1.0e-5f, 1.0e-5f);
    for(auto idx : alpaka::IdxRange{out6.getExtents()})
        alpaka::nn::test::checkValue(out6[idx], out7[idx], 1.0e-5f, 1.0e-5f);
}

TEMPLATE_LIST_TEST_CASE(
    "decoder prefill attention flat packed views are independent of spare allocation",
    "[nn][attention][decoder]",
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
    constexpr uint32_t heads = 4u;
    constexpr uint32_t headDim = 4u;
    constexpr uint32_t hidden = heads * headDim;

    auto runWithSpare = [&](uint32_t spareCapacity)
    {
        auto dummy = alpaka::onHost::alloc<float>(device, alpaka::Vec{1u, heads, spareCapacity, headDim});
        alpaka::unused(dummy);

        auto qStorage = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens * hidden});
        auto kStorage = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens * hidden});
        auto vStorage = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens * hidden});
        auto attnStorage = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens * hidden});

        for(uint32_t i = 0u; i < tokens * hidden; ++i)
        {
            qStorage[alpaka::Vec{i}] = 0.125f + static_cast<float>(i) * 0.01f;
            kStorage[alpaka::Vec{i}] = -0.25f + static_cast<float>(i) * 0.015f;
            vStorage[alpaka::Vec{i}] = 0.5f - static_cast<float>(i) * 0.0075f;
        }

        auto q = alpaka::makeView(device, qStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto k = alpaka::makeView(device, kStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto v = alpaka::makeView(device, vStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});

        auto expectedScores = referenceScoresBTHD<float>(q, k, 1u);
        scaleTensor(expectedScores, 0.5f);
        auto expectedProbs = alpaka::onHost::allocHost<float>(expectedScores.getExtents());
        for(auto idx : alpaka::IdxRange{expectedScores.getExtents()})
            expectedProbs[idx] = expectedScores[idx];
        referenceCausalSoftmaxInPlace(expectedProbs);
        auto expectedOut = referenceApplyBTHD<float>(expectedProbs, v, 1u);

        auto devQStorage = alpaka::onHost::allocLike(device, qStorage);
        auto devKStorage = alpaka::onHost::allocLike(device, kStorage);
        auto devVStorage = alpaka::onHost::allocLike(device, vStorage);
        auto devAttnStorage = alpaka::onHost::allocLike(device, attnStorage);
        alpaka::onHost::memcpy(queue, devQStorage, qStorage);
        alpaka::onHost::memcpy(queue, devKStorage, kStorage);
        alpaka::onHost::memcpy(queue, devVStorage, vStorage);

        auto devQ = alpaka::makeView(device, devQStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto devK = alpaka::makeView(device, devKStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto devV = alpaka::makeView(device, devVStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto devAttn = alpaka::makeView(device, devAttnStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});

        auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, heads, tokens, tokens});
        auto probs = alpaka::onHost::allocHost<float>(scores.getExtents());
        auto devScores = alpaka::onHost::allocLike(device, scores);
        auto devProbs = alpaka::onHost::allocLike(device, probs);

        alpaka::nn::onHost::nn::attentionScores<float>(
            queue,
            exec,
            devQ,
            devK,
            devScores,
            1u,
            alpaka::nn::AttentionKvLayout::BTHD);
        alpaka::nn::onHost::ops::scale<float>(queue, exec, devScores, 0.5f, devScores);
        alpaka::nn::onHost::nn::causalSoftmax<float>(queue, exec, devScores, devProbs, 3u, 2u, 3u);
        alpaka::nn::onHost::nn::attentionApply<float>(
            queue,
            exec,
            devProbs,
            devV,
            devAttn,
            1u,
            alpaka::nn::AttentionKvLayout::BTHD);

        alpaka::onHost::memcpy(queue, scores, devScores);
        alpaka::onHost::memcpy(queue, probs, devProbs);
        alpaka::onHost::memcpy(queue, attnStorage, devAttnStorage);
        alpaka::onHost::wait(queue);

        for(auto idx : alpaka::IdxRange{scores.getExtents()})
            alpaka::nn::test::checkValue(scores[idx], expectedScores[idx], 1.0e-5f, 1.0e-5f);
        for(auto idx : alpaka::IdxRange{probs.getExtents()})
            alpaka::nn::test::checkValue(probs[idx], expectedProbs[idx], 1.0e-5f, 1.0e-5f);

        auto attn2d = alpaka::makeView(device, attnStorage.data(), alpaka::Vec{tokens, hidden});
        return std::tuple{attnStorage, expectedOut, attn2d};
    };

    auto [attn4, expected4, attn2d4] = runWithSpare(4u);
    auto [attn5, expected5, attn2d5] = runWithSpare(5u);
    alpaka::unused(attn2d4);
    alpaka::unused(attn2d5);

    for(auto idx : alpaka::IdxRange{expected4.getExtents()})
        alpaka::nn::test::checkValue(expected4[idx], expected5[idx], 1.0e-5f, 1.0e-5f);

    for(uint32_t i = 0u; i < tokens * hidden; ++i)
        alpaka::nn::test::checkValue(attn4[alpaka::Vec{i}], attn5[alpaka::Vec{i}], 1.0e-5f, 1.0e-5f);
}

TEMPLATE_LIST_TEST_CASE(
    "decoder prefill projected attention is independent of spare allocation",
    "[nn][attention][decoder]",
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
    constexpr uint32_t heads = 4u;
    constexpr uint32_t headDim = 4u;
    constexpr uint32_t hidden = heads * headDim;

    auto inputHost = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens, hidden});
    auto wqHost = alpaka::onHost::allocHost<float>(alpaka::Vec{hidden, hidden});
    auto wkHost = alpaka::onHost::allocHost<float>(alpaka::Vec{hidden, hidden});
    auto wvHost = alpaka::onHost::allocHost<float>(alpaka::Vec{hidden, hidden});

    for(auto idx : alpaka::IdxRange{inputHost.getExtents()})
        inputHost[idx] = -0.25f + static_cast<float>(idx[0] * 13u + idx[1] * 3u) * 0.01f;
    for(auto idx : alpaka::IdxRange{wqHost.getExtents()})
    {
        auto const flat = static_cast<float>(idx[0] * hidden + idx[1]);
        wqHost[idx] = -0.1f + flat * 0.001f;
        wkHost[idx] = 0.2f - flat * 0.0007f;
        wvHost[idx] = -0.05f + flat * 0.0009f;
    }

    auto inputDev = alpaka::onHost::allocLike(device, inputHost);
    auto wqDev = alpaka::onHost::allocLike(device, wqHost);
    auto wkDev = alpaka::onHost::allocLike(device, wkHost);
    auto wvDev = alpaka::onHost::allocLike(device, wvHost);
    alpaka::onHost::memcpy(queue, inputDev, inputHost);
    alpaka::onHost::memcpy(queue, wqDev, wqHost);
    alpaka::onHost::memcpy(queue, wkDev, wkHost);
    alpaka::onHost::memcpy(queue, wvDev, wvHost);

    auto ropeTables = alpaka::nn::onHost::model::makeRopeTables<float>(device, tokens, headDim / 2u, 10000.0f);

    auto runWithSpare = [&](uint32_t spareCapacity)
    {
        auto dummy = alpaka::onHost::alloc<float>(device, alpaka::Vec{1u, heads, spareCapacity, headDim});
        alpaka::unused(dummy);

        auto qStorage = alpaka::onHost::alloc<float>(device, alpaka::Vec{tokens * hidden});
        auto kStorage = alpaka::onHost::alloc<float>(device, alpaka::Vec{tokens * hidden});
        auto vStorage = alpaka::onHost::alloc<float>(device, alpaka::Vec{tokens * hidden});
        auto attnStorage = alpaka::onHost::alloc<float>(device, alpaka::Vec{tokens * hidden});
        auto q2 = alpaka::makeView(device, qStorage.data(), alpaka::Vec{tokens, hidden});
        auto k2 = alpaka::makeView(device, kStorage.data(), alpaka::Vec{tokens, hidden});
        auto v2 = alpaka::makeView(device, vStorage.data(), alpaka::Vec{tokens, hidden});
        auto q4 = alpaka::makeView(device, qStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto k4 = alpaka::makeView(device, kStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto v4 = alpaka::makeView(device, vStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto attn4 = alpaka::makeView(device, attnStorage.data(), alpaka::Vec{1u, tokens, heads, headDim});

        alpaka::nn::onHost::nn::qkvProjection<float>(queue, exec, inputDev, wqDev, wkDev, wvDev, q2, k2, v2);
        alpaka::nn::onHost::nn::ropeInPlace<float>(queue, exec, q4, ropeTables.first, ropeTables.second);
        alpaka::nn::onHost::nn::ropeInPlace<float>(queue, exec, k4, ropeTables.first, ropeTables.second);

        auto scores = alpaka::onHost::allocHost<float>(alpaka::Vec{1u, heads, tokens, tokens});
        auto probs = alpaka::onHost::allocHost<float>(scores.getExtents());
        auto devScores = alpaka::onHost::allocLike(device, scores);
        auto devProbs = alpaka::onHost::allocLike(device, probs);

        alpaka::nn::onHost::nn::attentionScores<float>(
            queue,
            exec,
            q4,
            k4,
            devScores,
            1u,
            alpaka::nn::AttentionKvLayout::BTHD);
        alpaka::nn::onHost::ops::scale<float>(queue, exec, devScores, 0.5f, devScores);
        alpaka::nn::onHost::nn::causalSoftmax<float>(queue, exec, devScores, devProbs, 3u, 2u, 3u);
        alpaka::nn::onHost::nn::attentionApply<float>(
            queue,
            exec,
            devProbs,
            v4,
            attn4,
            1u,
            alpaka::nn::AttentionKvLayout::BTHD);

        auto qHost = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens * hidden});
        auto kHost = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens * hidden});
        auto vHost = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens * hidden});
        auto attnHost = alpaka::onHost::allocHost<float>(alpaka::Vec{tokens * hidden});
        alpaka::onHost::memcpy(queue, qHost, qStorage);
        alpaka::onHost::memcpy(queue, kHost, kStorage);
        alpaka::onHost::memcpy(queue, vHost, vStorage);
        alpaka::onHost::memcpy(queue, attnHost, attnStorage);
        alpaka::onHost::wait(queue);
        return std::tuple{qHost, kHost, vHost, attnHost};
    };

    auto [q4a, k4a, v4a, attn4a] = runWithSpare(4u);
    auto [q4b, k4b, v4b, attn4b] = runWithSpare(5u);

    for(uint32_t i = 0u; i < tokens * hidden; ++i)
    {
        alpaka::nn::test::checkValue(q4a[alpaka::Vec{i}], q4b[alpaka::Vec{i}], 1.0e-5f, 1.0e-5f);
        alpaka::nn::test::checkValue(k4a[alpaka::Vec{i}], k4b[alpaka::Vec{i}], 1.0e-5f, 1.0e-5f);
        alpaka::nn::test::checkValue(v4a[alpaka::Vec{i}], v4b[alpaka::Vec{i}], 1.0e-5f, 1.0e-5f);
        alpaka::nn::test::checkValue(attn4a[alpaka::Vec{i}], attn4b[alpaka::Vec{i}], 1.0e-5f, 1.0e-5f);
    }
}
