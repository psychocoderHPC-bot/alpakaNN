/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    template<typename T_Type>
    void runEmbeddingGatherCase(auto& queue, auto exec, auto const& device)
    {
        auto embedding = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{4u, 3u});
        for(auto idx : alpaka::IdxRange{embedding.getExtents()})
            embedding[idx] = static_cast<T_Type>(idx[0] * 10u + idx[1]);

        auto tokenIds = alpaka::onHost::allocHost<uint32_t>(alpaka::Vec{5u});
        tokenIds[alpaka::Vec{0u}] = 3u;
        tokenIds[alpaka::Vec{1u}] = 0u;
        tokenIds[alpaka::Vec{2u}] = 2u;
        tokenIds[alpaka::Vec{3u}] = 1u;
        tokenIds[alpaka::Vec{4u}] = 3u;

        auto output = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{5u, 3u});
        auto devEmbedding = alpaka::onHost::allocLike(device, embedding);
        auto devIds = alpaka::onHost::allocLike(device, tokenIds);
        auto devOutput = alpaka::onHost::allocLike(device, output);
        alpaka::onHost::memcpy(queue, devEmbedding, embedding);
        alpaka::onHost::memcpy(queue, devIds, tokenIds);

        alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, devIds, devEmbedding, devOutput);
        alpaka::onHost::memcpy(queue, output, devOutput);
        alpaka::onHost::wait(queue);

        for(uint32_t token = 0u; token < 5u; ++token)
        {
            for(uint32_t col = 0u; col < 3u; ++col)
            {
                auto const expected = embedding[alpaka::Vec{tokenIds[alpaka::Vec{token}], col}];
                alpaka::nn::test::checkValue(output[alpaka::Vec{token, col}], expected);
            }
        }
    }

    template<typename T_Type>
    void runEmbeddingSubViewCase(auto& queue, auto exec, auto const& device)
    {
        // The embedding lives in a wider storage buffer; the subview is pointed at a column offset so that
        // its rows are no longer contiguous. The gather must honor the subview pitches.
        constexpr uint32_t vocab = 3u;
        constexpr uint32_t hidden = 2u;
        constexpr uint32_t pitch = hidden + 3u;
        constexpr uint32_t colOffset = 2u;
        auto storage = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{vocab, pitch});
        for(auto idx : alpaka::IdxRange{storage.getExtents()})
            storage[idx] = static_cast<T_Type>(idx[0] * 10u + idx[1]);
        auto embedding = storage.getSubView(alpaka::Vec{0u, colOffset}, alpaka::Vec{vocab, hidden});

        auto tokenIds = alpaka::onHost::allocHost<uint32_t>(alpaka::Vec{4u});
        tokenIds[alpaka::Vec{0u}] = 2u;
        tokenIds[alpaka::Vec{1u}] = 0u;
        tokenIds[alpaka::Vec{2u}] = 1u;
        tokenIds[alpaka::Vec{3u}] = 2u;

        auto output = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{4u, hidden});
        auto devStorage = alpaka::onHost::allocLike(device, storage);
        auto devIds = alpaka::onHost::allocLike(device, tokenIds);
        auto devOutput = alpaka::onHost::allocLike(device, output);
        alpaka::onHost::memcpy(queue, devStorage, storage);
        alpaka::onHost::memcpy(queue, devIds, tokenIds);
        auto devEmbedding = devStorage.getSubView(alpaka::Vec{0u, colOffset}, alpaka::Vec{vocab, hidden});

        alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, devIds, devEmbedding, devOutput);
        alpaka::onHost::memcpy(queue, output, devOutput);
        alpaka::onHost::wait(queue);

        for(uint32_t token = 0u; token < 4u; ++token)
        {
            for(uint32_t col = 0u; col < hidden; ++col)
            {
                auto const expected = embedding[alpaka::Vec{tokenIds[alpaka::Vec{token}], col}];
                alpaka::nn::test::checkValue(output[alpaka::Vec{token, col}], expected);
            }
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("embeddingLookup gathers rows from the embedding table", "[nn][embedding]", TestApis)
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

    runEmbeddingGatherCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runEmbeddingGatherCase<double>(queue, exec, device);
}

TEMPLATE_LIST_TEST_CASE("embeddingLookup supports a non-contiguous embedding subview", "[nn][embedding]", TestApis)
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

    runEmbeddingSubViewCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runEmbeddingSubViewCase<double>(queue, exec, device);
}

namespace
{
    template<typename T_Type>
    void runEmbeddingShapeValidationCase(auto& queue, auto exec, auto const& device)
    {
        auto hostIds = alpaka::onHost::allocHost<uint32_t>(alpaka::Vec{2u});
        for(auto idx : alpaka::IdxRange{hostIds.getExtents()})
            hostIds[idx] = 0u;
        auto hostEmbedding = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{3u, 4u});
        for(auto idx : alpaka::IdxRange{hostEmbedding.getExtents()})
            hostEmbedding[idx] = T_Type{1};

        auto tokenIds = alpaka::onHost::alloc<uint32_t>(device, alpaka::Vec{2u});
        auto embedding = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{3u, 4u});
        alpaka::onHost::memcpy(queue, tokenIds, hostIds);
        alpaka::onHost::memcpy(queue, embedding, hostEmbedding);
        alpaka::onHost::wait(queue);

        auto validOutput = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{2u, 4u});
        auto wrongRowOutput = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{3u, 4u});
        auto wrongColOutput = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{2u, 5u});
        auto wrongRankOutput = alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{2u, 4u, 1u});

        REQUIRE_NOTHROW(
            alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, tokenIds, embedding, validOutput));
        alpaka::onHost::wait(queue);
        REQUIRE_THROWS_AS(
            alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, tokenIds, embedding, wrongRowOutput),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, tokenIds, embedding, wrongColOutput),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, tokenIds, embedding, wrongRankOutput),
            std::invalid_argument);
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("embeddingLookup validates shape and rank", "[nn][embedding]", TestApis)
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

    runEmbeddingShapeValidationCase<float>(queue, exec, device);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runEmbeddingShapeValidationCase<double>(queue, exec, device);
}
