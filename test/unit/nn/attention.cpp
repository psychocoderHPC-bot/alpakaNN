/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <tuple>
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
                sum += probs[alpaka::Vec{idx[0], idx[2], idx[1], key}]
                       * values[alpaka::Vec{idx[0], kvHead, key, idx[3]}];
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
                sum += probs[alpaka::Vec{idx[0], idx[2], idx[1], key}]
                       * values[alpaka::Vec{idx[0], key, kvHead, idx[3]}];
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

namespace
{
    template<typename T_Type>
    void runAttentionDeterminismCase(auto& queue, auto exec, auto const& device)
    {
        auto q = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 2u, 1u, 2u});
        auto k = alpaka::onHost::allocHost<T_Type>(q.getExtents());
        auto v = alpaka::onHost::allocHost<T_Type>(q.getExtents());
        auto scores = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 2u, 2u});
        auto probs = alpaka::onHost::allocHost<T_Type>(scores.getExtents());
        auto out = alpaka::onHost::allocHost<T_Type>(q.getExtents());
        for(auto idx : alpaka::IdxRange{q.getExtents()})
        {
            q[idx] = static_cast<T_Type>(idx[1] + idx[3] + 1u);
            k[idx] = static_cast<T_Type>(idx[1] + idx[3] + 2u);
            v[idx] = static_cast<T_Type>(idx[1] * 2u + idx[3] + 1u);
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

        alpaka::nn::onHost::nn::attentionScores<T_Type>(queue, exec, devQ, devK, devScores);
        alpaka::nn::onHost::nn::softmax<T_Type>(queue, exec, devScores, devProbs, 3u);
        alpaka::nn::onHost::nn::attentionApply<T_Type>(queue, exec, devProbs, devV, devOut);
        alpaka::onHost::memcpy(queue, scores, devScores);
        alpaka::onHost::memcpy(queue, probs, devProbs);
        alpaka::onHost::memcpy(queue, out, devOut);
        alpaka::onHost::wait(queue);

        for(auto idx : alpaka::IdxRange{scores.getExtents()})
            REQUIRE(std::isfinite(scores[idx]));
        for(auto idx : alpaka::IdxRange{out.getExtents()})
            REQUIRE(std::isfinite(out[idx]));
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("attention scores and apply are deterministic", "[nn][attention]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runAttentionDeterminismCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runAttentionDeterminismCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runAttentionGroupedHeadsCase(auto& queue, auto exec, auto const& device)
    {
        auto q = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 4u, 2u});
        auto k = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 2u, 2u});
        auto v = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 2u, 2u});
        auto scores = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 1u, 1u});
        auto probs = alpaka::onHost::allocHost<T_Type>(scores.getExtents());
        auto out = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 4u, 2u});

        for(uint32_t head = 0u; head < 4u; ++head)
        {
            q[alpaka::Vec{0u, 0u, head, 0u}] = static_cast<T_Type>(head + 1u);
            q[alpaka::Vec{0u, 0u, head, 1u}] = static_cast<T_Type>(head + 2u);
        }
        k[alpaka::Vec{0u, 0u, 0u, 0u}] = T_Type{1};
        k[alpaka::Vec{0u, 0u, 0u, 1u}] = T_Type{2};
        k[alpaka::Vec{0u, 0u, 1u, 0u}] = T_Type{3};
        k[alpaka::Vec{0u, 0u, 1u, 1u}] = T_Type{4};
        v[alpaka::Vec{0u, 0u, 0u, 0u}] = T_Type{10};
        v[alpaka::Vec{0u, 0u, 0u, 1u}] = T_Type{20};
        v[alpaka::Vec{0u, 0u, 1u, 0u}] = T_Type{30};
        v[alpaka::Vec{0u, 0u, 1u, 1u}] = T_Type{40};

        auto devQ = alpaka::onHost::allocLike(device, q);
        auto devK = alpaka::onHost::allocLike(device, k);
        auto devV = alpaka::onHost::allocLike(device, v);
        auto devScores = alpaka::onHost::allocLike(device, scores);
        auto devProbs = alpaka::onHost::allocLike(device, probs);
        auto devOut = alpaka::onHost::allocLike(device, out);
        alpaka::onHost::memcpy(queue, devQ, q);
        alpaka::onHost::memcpy(queue, devK, k);
        alpaka::onHost::memcpy(queue, devV, v);

        alpaka::nn::onHost::nn::attentionScores<T_Type>(
            queue,
            exec,
            devQ,
            devK,
            devScores,
            2u,
            alpaka::nn::AttentionKvLayout::BTHD);
        alpaka::nn::onHost::nn::softmax<T_Type>(queue, exec, devScores, devProbs, 3u);
        alpaka::nn::onHost::nn::attentionApply<T_Type>(
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

        alpaka::nn::test::checkValue(scores[alpaka::Vec{0u, 0u, 0u, 0u}], T_Type{5});
        alpaka::nn::test::checkValue(scores[alpaka::Vec{0u, 1u, 0u, 0u}], T_Type{8});
        alpaka::nn::test::checkValue(scores[alpaka::Vec{0u, 2u, 0u, 0u}], T_Type{25});
        alpaka::nn::test::checkValue(scores[alpaka::Vec{0u, 3u, 0u, 0u}], T_Type{32});
        for(uint32_t head = 0u; head < 2u; ++head)
        {
            alpaka::nn::test::checkValue(out[alpaka::Vec{0u, 0u, head, 0u}], T_Type{10});
            alpaka::nn::test::checkValue(out[alpaka::Vec{0u, 0u, head, 1u}], T_Type{20});
        }
        for(uint32_t head = 2u; head < 4u; ++head)
        {
            alpaka::nn::test::checkValue(out[alpaka::Vec{0u, 0u, head, 0u}], T_Type{30});
            alpaka::nn::test::checkValue(out[alpaka::Vec{0u, 0u, head, 1u}], T_Type{40});
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("attention maps grouped query heads onto fewer kv heads", "[nn][attention]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runAttentionGroupedHeadsCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runAttentionGroupedHeadsCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runAttentionScoresPrefillCase(auto& queue, auto exec, auto const& device)
    {
        auto q = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        auto k = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        auto scores = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        fill4D(q, static_cast<T_Type>(0.25));
        fill4D(k, static_cast<T_Type>(-0.5));
        for(auto idx : alpaka::IdxRange{scores.getExtents()})
            scores[idx] = static_cast<T_Type>(-999);

        auto expected = referenceScoresBTHD<T_Type>(q, k, 1u);
        auto devQ = alpaka::onHost::allocLike(device, q);
        auto devK = alpaka::onHost::allocLike(device, k);
        auto devScores = alpaka::onHost::allocLike(device, scores);
        alpaka::onHost::memcpy(queue, devQ, q);
        alpaka::onHost::memcpy(queue, devK, k);
        alpaka::onHost::memcpy(queue, devScores, scores);

        alpaka::nn::onHost::nn::attentionScores<T_Type>(
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
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "attention scores match reference for decoder prefill shape",
    "[nn][attention][decoder]",
    TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runAttentionScoresPrefillCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runAttentionScoresPrefillCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runAttentionScoresDecodeCase(auto& queue, auto exec, auto const& device)
    {
        auto q = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 4u, 4u});
        auto k = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 5u, 4u});
        auto scores = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 1u, 5u});
        fill4D(q, static_cast<T_Type>(0.75));
        fill4D(k, static_cast<T_Type>(-0.125));
        for(auto idx : alpaka::IdxRange{scores.getExtents()})
            scores[idx] = static_cast<T_Type>(-777);

        auto expected = referenceScoresBHTD<T_Type>(q, k, 1u);
        auto devQ = alpaka::onHost::allocLike(device, q);
        auto devK = alpaka::onHost::allocLike(device, k);
        auto devScores = alpaka::onHost::allocLike(device, scores);
        alpaka::onHost::memcpy(queue, devQ, q);
        alpaka::onHost::memcpy(queue, devK, k);
        alpaka::onHost::memcpy(queue, devScores, scores);

        alpaka::nn::onHost::nn::attentionScores<T_Type>(
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
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "attention scores match reference for decoder decode layout",
    "[nn][attention][decoder]",
    TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runAttentionScoresDecodeCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runAttentionScoresDecodeCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runAttentionApplyDecodeCase(auto& queue, auto exec, auto const& device)
    {
        auto probs = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 1u, 5u});
        auto values = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 5u, 4u});
        auto out = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 4u, 4u});
        fill4D(values, static_cast<T_Type>(1.25));
        for(uint32_t head = 0u; head < 4u; ++head)
        {
            T_Type norm{};
            for(uint32_t key = 0u; key < 5u; ++key)
            {
                auto value = static_cast<T_Type>((head + 1u) * (key + 2u));
                probs[alpaka::Vec{0u, head, 0u, key}] = value;
                norm += value;
            }
            for(uint32_t key = 0u; key < 5u; ++key)
                probs[alpaka::Vec{0u, head, 0u, key}] /= norm;
        }
        for(auto idx : alpaka::IdxRange{out.getExtents()})
            out[idx] = static_cast<T_Type>(-555);

        auto expected = referenceApplyBHTD<T_Type>(probs, values, 1u);
        auto devProbs = alpaka::onHost::allocLike(device, probs);
        auto devValues = alpaka::onHost::allocLike(device, values);
        auto devOut = alpaka::onHost::allocLike(device, out);
        alpaka::onHost::memcpy(queue, devProbs, probs);
        alpaka::onHost::memcpy(queue, devValues, values);
        alpaka::onHost::memcpy(queue, devOut, out);

        alpaka::nn::onHost::nn::attentionApply<T_Type>(
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
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "attention apply matches reference for decoder decode layout",
    "[nn][attention][decoder]",
    TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runAttentionApplyDecodeCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runAttentionApplyDecodeCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runAttentionApplyPrefillCase(auto& queue, auto exec, auto const& device)
    {
        auto probs = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        auto values = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        auto out = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        fill4D(values, T_Type{1});
        for(uint32_t query = 0u; query < 4u; ++query)
        {
            for(uint32_t head = 0u; head < 4u; ++head)
            {
                T_Type norm{};
                for(uint32_t key = 0u; key < 4u; ++key)
                {
                    auto value
                        = key <= query ? static_cast<T_Type>((query + 1u) * (head + 2u) * (key + 1u)) : T_Type{};
                    probs[alpaka::Vec{0u, head, query, key}] = value;
                    norm += value;
                }
                for(uint32_t key = 0u; key < 4u; ++key)
                    probs[alpaka::Vec{0u, head, query, key}]
                        = norm > T_Type{} ? probs[alpaka::Vec{0u, head, query, key}] / norm : T_Type{};
            }
        }
        for(auto idx : alpaka::IdxRange{out.getExtents()})
            out[idx] = static_cast<T_Type>(-444);

        auto expected = referenceApplyBTHD<T_Type>(probs, values, 1u);
        auto devProbs = alpaka::onHost::allocLike(device, probs);
        auto devValues = alpaka::onHost::allocLike(device, values);
        auto devOut = alpaka::onHost::allocLike(device, out);
        alpaka::onHost::memcpy(queue, devProbs, probs);
        alpaka::onHost::memcpy(queue, devValues, values);
        alpaka::onHost::memcpy(queue, devOut, out);

        alpaka::nn::onHost::nn::attentionApply<T_Type>(
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
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "attention apply matches reference for decoder prefill layout",
    "[nn][attention][decoder]",
    TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runAttentionApplyPrefillCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runAttentionApplyPrefillCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runAttentionPrefillPipelineCase(auto& queue, auto exec, auto const& device)
    {
        auto q = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        auto k = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        auto v = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        auto scores = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        auto probs = alpaka::onHost::allocHost<T_Type>(scores.getExtents());
        auto out = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 4u, 4u});
        fill4D(q, static_cast<T_Type>(0.125));
        fill4D(k, static_cast<T_Type>(-0.25));
        fill4D(v, static_cast<T_Type>(0.5));

        auto expectedScores = referenceScoresBTHD<T_Type>(q, k, 1u);
        scaleTensor(expectedScores, static_cast<T_Type>(0.5));
        auto expectedProbs = alpaka::onHost::allocHost<T_Type>(expectedScores.getExtents());
        for(auto idx : alpaka::IdxRange{expectedScores.getExtents()})
            expectedProbs[idx] = expectedScores[idx];
        referenceCausalSoftmaxInPlace(expectedProbs);
        auto expectedOut = referenceApplyBTHD<T_Type>(expectedProbs, v, 1u);

        auto devQ = alpaka::onHost::allocLike(device, q);
        auto devK = alpaka::onHost::allocLike(device, k);
        auto devV = alpaka::onHost::allocLike(device, v);
        auto devScores = alpaka::onHost::allocLike(device, scores);
        auto devProbs = alpaka::onHost::allocLike(device, probs);
        auto devOut = alpaka::onHost::allocLike(device, out);
        alpaka::onHost::memcpy(queue, devQ, q);
        alpaka::onHost::memcpy(queue, devK, k);
        alpaka::onHost::memcpy(queue, devV, v);

        alpaka::nn::onHost::nn::attentionScores<T_Type>(
            queue,
            exec,
            devQ,
            devK,
            devScores,
            1u,
            alpaka::nn::AttentionKvLayout::BTHD);
        alpaka::nn::onHost::ops::scale<T_Type>(queue, exec, devScores, static_cast<T_Type>(0.5), devScores);
        alpaka::nn::onHost::nn::causalSoftmax<T_Type>(queue, exec, devScores, devProbs, 3u, 2u, 3u);
        alpaka::nn::onHost::nn::attentionApply<T_Type>(
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
} // namespace

TEMPLATE_LIST_TEST_CASE("decoder prefill attention pipeline matches reference", "[nn][attention][decoder]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runAttentionPrefillPipelineCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runAttentionPrefillPipelineCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runAttentionDecodePipelineCase(auto& queue, auto exec, auto const& device)
    {
        auto q = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 4u, 4u});
        auto k = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 5u, 4u});
        auto v = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 5u, 4u});
        auto scores = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 1u, 5u});
        auto probs = alpaka::onHost::allocHost<T_Type>(scores.getExtents());
        auto out = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 4u, 4u});
        fill4D(q, static_cast<T_Type>(0.375));
        fill4D(k, static_cast<T_Type>(-0.0625));
        fill4D(v, static_cast<T_Type>(0.875));

        auto expectedScores = referenceScoresBHTD<T_Type>(q, k, 1u);
        scaleTensor(expectedScores, static_cast<T_Type>(0.5));
        auto expectedProbs = alpaka::onHost::allocHost<T_Type>(expectedScores.getExtents());
        for(auto idx : alpaka::IdxRange{expectedScores.getExtents()})
            expectedProbs[idx] = expectedScores[idx];
        referenceSoftmaxInPlace(expectedProbs);
        auto expectedOut = referenceApplyBHTD<T_Type>(expectedProbs, v, 1u);

        auto devQ = alpaka::onHost::allocLike(device, q);
        auto devK = alpaka::onHost::allocLike(device, k);
        auto devV = alpaka::onHost::allocLike(device, v);
        auto devScores = alpaka::onHost::allocLike(device, scores);
        auto devProbs = alpaka::onHost::allocLike(device, probs);
        auto devOut = alpaka::onHost::allocLike(device, out);
        alpaka::onHost::memcpy(queue, devQ, q);
        alpaka::onHost::memcpy(queue, devK, k);
        alpaka::onHost::memcpy(queue, devV, v);

        alpaka::nn::onHost::nn::attentionScores<T_Type>(
            queue,
            exec,
            devQ,
            devK,
            devScores,
            1u,
            alpaka::nn::AttentionKvLayout::BHTD);
        alpaka::nn::onHost::ops::scale<T_Type>(queue, exec, devScores, static_cast<T_Type>(0.5), devScores);
        alpaka::nn::onHost::nn::softmax<T_Type>(queue, exec, devScores, devProbs, 3u);
        alpaka::nn::onHost::nn::attentionApply<T_Type>(
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
} // namespace

TEMPLATE_LIST_TEST_CASE("decoder decode attention pipeline matches reference", "[nn][attention][decoder]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runAttentionDecodePipelineCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runAttentionDecodePipelineCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runAttentionExplicitBhtdCase(auto& queue, auto exec, auto const& device)
    {
        auto q = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 4u, 4u});
        auto kTokens = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 5u, 4u, 4u});
        auto vTokens = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 5u, 4u, 4u});
        auto scores = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 1u, 5u});
        auto probs = alpaka::onHost::allocHost<T_Type>(scores.getExtents());
        auto out = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 1u, 4u, 4u});
        fill4D(q, static_cast<T_Type>(0.375));
        fill4D(kTokens, static_cast<T_Type>(-0.0625));
        fill4D(vTokens, static_cast<T_Type>(0.875));

        // Assemble the key/value buffers explicitly in BHTD layout from token-major (BTHD) inputs.
        auto expectedK = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 5u, 4u});
        auto expectedV = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{1u, 4u, 5u, 4u});
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

        auto expectedScores = referenceScoresBHTD<T_Type>(q, expectedK, 1u);
        scaleTensor(expectedScores, static_cast<T_Type>(0.5));
        auto expectedProbs = alpaka::onHost::allocHost<T_Type>(expectedScores.getExtents());
        for(auto idx : alpaka::IdxRange{expectedScores.getExtents()})
            expectedProbs[idx] = expectedScores[idx];
        referenceSoftmaxInPlace(expectedProbs);
        auto expectedOut = referenceApplyBHTD<T_Type>(expectedProbs, expectedV, 1u);

        auto devQ = alpaka::onHost::allocLike(device, q);
        auto devK = alpaka::onHost::allocLike(device, expectedK);
        auto devV = alpaka::onHost::allocLike(device, expectedV);
        auto devScores = alpaka::onHost::allocLike(device, scores);
        auto devProbs = alpaka::onHost::allocLike(device, probs);
        auto devOut = alpaka::onHost::allocLike(device, out);
        alpaka::onHost::memcpy(queue, devQ, q);
        alpaka::onHost::memcpy(queue, devK, expectedK);
        alpaka::onHost::memcpy(queue, devV, expectedV);

        alpaka::nn::onHost::nn::attentionScores<T_Type>(
            queue,
            exec,
            devQ,
            devK,
            devScores,
            1u,
            alpaka::nn::AttentionKvLayout::BHTD);
        alpaka::nn::onHost::ops::scale<T_Type>(queue, exec, devScores, static_cast<T_Type>(0.5), devScores);
        alpaka::nn::onHost::nn::softmax<T_Type>(queue, exec, devScores, devProbs, 3u);
        alpaka::nn::onHost::nn::attentionApply<T_Type>(
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
} // namespace

TEMPLATE_LIST_TEST_CASE("attention pipeline matches reference for explicit BHTD kv views", "[nn][attention]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runAttentionExplicitBhtdCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runAttentionExplicitBhtdCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runQkvProjectionRopeCase(auto& queue, auto exec, auto const& device)
    {
        constexpr uint32_t tokens = 3u;
        constexpr uint32_t heads = 1u;
        constexpr uint32_t headDim = 4u;
        constexpr uint32_t hidden = heads * headDim;
        constexpr uint32_t pairs = headDim / 2u;

        auto input = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{tokens, hidden});
        auto wq = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{hidden, hidden});
        auto wk = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{hidden, hidden});
        auto wv = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{hidden, hidden});
        for(auto idx : alpaka::IdxRange{input.getExtents()})
            input[idx] = static_cast<T_Type>(-0.25)
                         + static_cast<T_Type>(idx[0] * 13u + idx[1] * 3u) * static_cast<T_Type>(0.05);
        for(auto idx : alpaka::IdxRange{wq.getExtents()})
        {
            auto const flat = static_cast<T_Type>(idx[0] * hidden + idx[1]);
            wq[idx] = static_cast<T_Type>(-0.1) + flat * static_cast<T_Type>(0.01);
            wk[idx] = static_cast<T_Type>(0.2) - flat * static_cast<T_Type>(0.007);
            wv[idx] = static_cast<T_Type>(-0.05) + flat * static_cast<T_Type>(0.009);
        }

        // Host reference for qkvProjection: Q/K/V = input * W.
        auto referenceProjection = [&](auto const& weight)
        {
            auto expected = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{tokens, hidden});
            for(uint32_t row = 0u; row < tokens; ++row)
            {
                for(uint32_t col = 0u; col < hidden; ++col)
                {
                    T_Type sum{};
                    for(uint32_t inner = 0u; inner < hidden; ++inner)
                        sum += input[alpaka::Vec{row, inner}] * weight[alpaka::Vec{inner, col}];
                    expected[alpaka::Vec{row, col}] = sum;
                }
            }
            return expected;
        };
        auto referenceQ = referenceProjection(wq);
        auto referenceK = referenceProjection(wk);
        auto referenceV = referenceProjection(wv);

        // Host reference for ropeInPlace using explicit tables.
        auto cosTable = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{tokens, pairs});
        auto sinTable = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{tokens, pairs});
        for(uint32_t pos = 0u; pos < tokens; ++pos)
        {
            for(uint32_t pair = 0u; pair < pairs; ++pair)
            {
                auto const angle
                    = static_cast<T_Type>(pos + 1u) * static_cast<T_Type>(0.25) * static_cast<T_Type>(pair + 1u);
                cosTable[alpaka::Vec{pos, pair}] = std::cos(angle);
                sinTable[alpaka::Vec{pos, pair}] = std::sin(angle);
            }
        }
        auto referenceRope = [&](auto const& projected)
        {
            auto expected = alpaka::onHost::allocHost<T_Type>(projected.getExtents());
            for(uint32_t token = 0u; token < tokens; ++token)
            {
                for(uint32_t pair = 0u; pair < pairs; ++pair)
                {
                    auto const c = cosTable[alpaka::Vec{token, pair}];
                    auto const s = sinTable[alpaka::Vec{token, pair}];
                    auto const x0 = projected[alpaka::Vec{token, pair * 2u}];
                    auto const x1 = projected[alpaka::Vec{token, pair * 2u + 1u}];
                    expected[alpaka::Vec{token, pair * 2u}] = x0 * c - x1 * s;
                    expected[alpaka::Vec{token, pair * 2u + 1u}] = x0 * s + x1 * c;
                }
            }
            return expected;
        };
        auto expectedQ = referenceRope(referenceQ);
        auto expectedK = referenceRope(referenceK);

        auto inputDev = alpaka::onHost::allocLike(device, input);
        auto wqDev = alpaka::onHost::allocLike(device, wq);
        auto wkDev = alpaka::onHost::allocLike(device, wk);
        auto wvDev = alpaka::onHost::allocLike(device, wv);
        alpaka::onHost::memcpy(queue, inputDev, input);
        alpaka::onHost::memcpy(queue, wqDev, wq);
        alpaka::onHost::memcpy(queue, wkDev, wk);
        alpaka::onHost::memcpy(queue, wvDev, wv);

        // Q/K/V are also accessed through higher-rank rope views. A 2-D allocation can carry a padded row
        // pitch on some backends while makeView() assumes contiguous memory; mixing the two would address
        // different elements. Use flat storage and build both the 2-D and 4-D views over it.
        auto qFlat = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{tokens * hidden});
        auto kFlat = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{tokens * hidden});
        auto vFlat = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{tokens * hidden});
        auto q2 = alpaka::makeView(device, qFlat.data(), alpaka::Vec{tokens, hidden});
        auto k2 = alpaka::makeView(device, kFlat.data(), alpaka::Vec{tokens, hidden});
        auto v2 = alpaka::makeView(device, vFlat.data(), alpaka::Vec{tokens, hidden});
        alpaka::nn::onHost::nn::qkvProjection<T_Type>(queue, inputDev, wqDev, wkDev, wvDev, q2, k2, v2);

        auto q4 = alpaka::makeView(device, qFlat.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto k4 = alpaka::makeView(device, kFlat.data(), alpaka::Vec{1u, tokens, heads, headDim});
        auto cosDev = alpaka::onHost::allocLike(device, cosTable);
        auto sinDev = alpaka::onHost::allocLike(device, sinTable);
        alpaka::onHost::memcpy(queue, cosDev, cosTable);
        alpaka::onHost::memcpy(queue, sinDev, sinTable);
        alpaka::nn::onHost::nn::ropeInPlace<T_Type>(queue, exec, q4, k4, cosDev, sinDev);

        auto qHost = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{tokens, hidden});
        auto kHost = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{tokens, hidden});
        auto vHost = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{tokens, hidden});
        alpaka::onHost::memcpy(queue, qHost, q2);
        alpaka::onHost::memcpy(queue, kHost, k2);
        alpaka::onHost::memcpy(queue, vHost, v2);
        alpaka::onHost::wait(queue);

        for(auto idx : alpaka::IdxRange{vHost.getExtents()})
        {
            alpaka::nn::test::checkValue(qHost[idx], expectedQ[idx], 1.0e-5f, 1.0e-5f);
            alpaka::nn::test::checkValue(kHost[idx], expectedK[idx], 1.0e-5f, 1.0e-5f);
            alpaka::nn::test::checkValue(vHost[idx], referenceV[idx], 1.0e-5f, 1.0e-5f);
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("qkvProjection and ropeInPlace match a host reference", "[nn][attention]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto exec = cfg[alpaka::object::exec];
    auto queue = device.makeQueue();

    runQkvProjectionRopeCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runQkvProjectionRopeCase<double>(queue, exec, device);
}
