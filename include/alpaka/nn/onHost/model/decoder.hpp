/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/inference/kv_cache.hpp>
#include <alpaka/nn/onHost/inference/transformer_block.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/nn/embedding.hpp>
#include <alpaka/nn/onHost/nn/rms_norm.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace alpaka::nn::onHost::model
{
    namespace detail
    {
        inline bool prefillTraceEnabled()
        {
            auto const* env = std::getenv("ALPAKANN_DEBUG_PREFILL_TRACE");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }

        inline void printTrace(std::string const& message)
        {
            if(prefillTraceEnabled())
                std::fprintf(stderr, "%s\n", message.c_str());
        }

        template<typename T_Type>
        void traceTensor(auto& queue, auto const& tensor, std::string_view label, uint32_t sampleCount = 8u)
        {
            if(!prefillTraceEnabled())
                return;

            auto host = alpaka::onHost::allocHost<T_Type>(tensor.getExtents());
            alpaka::onHost::memcpy(queue, host, tensor);
            alpaka::onHost::wait(queue);

            std::ostringstream os;
            os << label << " extents=" << tensor.getExtents();

            auto const extents = host.getExtents();
            if(ALPAKA_TYPEOF(extents)::dim() == 2u && extents[0] > 0u)
            {
                auto const row = static_cast<uint32_t>(extents[0] - 1u);
                auto const width = static_cast<uint32_t>(extents[1]);
                auto const keep = std::min(sampleCount, width);
                T_Type sum{};
                T_Type maxAbs{};
                for(uint32_t col = 0u; col < width; ++col)
                {
                    auto const value = host[alpaka::Vec{row, col}];
                    sum += value;
                    auto const absValue = value < T_Type{} ? -value : value;
                    if(absValue > maxAbs)
                        maxAbs = absValue;
                }
                os << " row=" << row << " first" << keep << '=';
                for(uint32_t col = 0u; col < keep; ++col)
                    os << ' ' << host[alpaka::Vec{row, col}];
                os << " sum=" << sum << " maxAbs=" << maxAbs;
            }
            else
            {
                auto it = host.begin();
                auto const end = host.end();
                T_Type sum{};
                T_Type maxAbs{};
                uint32_t count = 0u;
                os << " first" << sampleCount << '=';
                for(; it != end; ++it)
                {
                    auto const value = *it;
                    sum += value;
                    auto const absValue = value < T_Type{} ? -value : value;
                    if(absValue > maxAbs)
                        maxAbs = absValue;
                    if(count < sampleCount)
                        os << ' ' << value;
                    ++count;
                }
                os << " sum=" << sum << " maxAbs=" << maxAbs;
            }

            printTrace(os.str());
        }
    } // namespace detail

    struct ModelConfig
    {
        uint32_t hiddenSize{};
        uint32_t intermediateSize{};
        uint32_t numLayers{};
        uint32_t numHeads{};
        uint32_t numKeyValueHeads{};
        uint32_t vocabSize{};
        uint32_t bosTokenId{};
        uint32_t eosTokenId{};
        uint32_t maxPositionEmbeddings{};
        float rmsNormEpsilon{};
        float ropeTheta{};
    };

    template<typename T_Type, typename TVectorBuffer, typename TMatrixBuffer>
    struct DecoderModel
    {
        using value_type = T_Type;

        ModelConfig config;
        TMatrixBuffer embedding;
        std::vector<alpaka::nn::onHost::inference::TransformerBlockWeights<T_Type, TVectorBuffer, TMatrixBuffer>>
            layers;
        TVectorBuffer finalNorm;
        TMatrixBuffer lmHead;
    };

    inline ModelConfig readConfig(std::istream& input)
    {
        char magic[4];
        input.read(magic, 4);
        if(std::string_view(magic, 4) != "ANN1")
            throw std::runtime_error{"Invalid model file magic."};

        ModelConfig cfg{};
        input.read(reinterpret_cast<char*>(&cfg.hiddenSize), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&cfg.intermediateSize), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&cfg.numLayers), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&cfg.numHeads), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&cfg.numKeyValueHeads), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&cfg.vocabSize), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&cfg.bosTokenId), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&cfg.eosTokenId), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&cfg.maxPositionEmbeddings), sizeof(uint32_t));
        input.read(reinterpret_cast<char*>(&cfg.rmsNormEpsilon), sizeof(float));
        input.read(reinterpret_cast<char*>(&cfg.ropeTheta), sizeof(float));
        return cfg;
    }

    inline void validateSupportedConfig(ModelConfig const& cfg)
    {
        if(cfg.numHeads == 0u)
            throw std::runtime_error{"Model config is invalid: numHeads must be non-zero."};
        if(cfg.numKeyValueHeads == 0u)
            throw std::runtime_error{"Model config is invalid: numKeyValueHeads must be non-zero."};
        if(cfg.hiddenSize % cfg.numHeads != 0u)
            throw std::runtime_error{"Model config is invalid: hiddenSize must be divisible by numHeads."};
        if(cfg.numHeads % cfg.numKeyValueHeads != 0u)
            throw std::runtime_error{"Model config is invalid: numHeads must be divisible by numKeyValueHeads."};
    }

    template<typename T_Type, typename T_Device>
    auto loadTinyLlama(T_Device const& device, std::string const& path)
    {
        using VectorBuffer = decltype(alpaka::onHost::alloc<T_Type>(device, uint32_t{1}));
        using MatrixBuffer = decltype(alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{1u, 1u}));

        std::ifstream input(path, std::ios::binary);
        if(!input)
            throw std::runtime_error{"Failed to open model file: " + path};

        auto cfg = readConfig(input);
        validateSupportedConfig(cfg);
        auto deviceCopy = device;
        auto queue = deviceCopy.makeQueue();

        auto load1D = [&](uint32_t size)
        {
            auto host = alpaka::onHost::allocHost<T_Type>(size);
            input.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(size * sizeof(T_Type)));
            if(!input)
                throw std::runtime_error{"Unexpected end of file while reading 1D tensor."};
            auto dev = alpaka::onHost::allocLike(device, host);
            alpaka::onHost::memcpy(queue, dev, host);
            alpaka::onHost::wait(queue);
            return dev;
        };

        auto load2D = [&](uint32_t rows, uint32_t cols)
        {
            auto host = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{rows, cols});
            input.read(
                reinterpret_cast<char*>(host.data()),
                static_cast<std::streamsize>(rows * cols * sizeof(T_Type)));
            if(!input)
                throw std::runtime_error{"Unexpected end of file while reading 2D tensor."};
            auto dev = alpaka::onHost::allocLike(device, host);
            alpaka::onHost::memcpy(queue, dev, host);
            alpaka::onHost::wait(queue);
            return dev;
        };

        auto embedding = load2D(cfg.vocabSize, cfg.hiddenSize);
        std::vector<alpaka::nn::onHost::inference::TransformerBlockWeights<T_Type, VectorBuffer, MatrixBuffer>> layers;
        layers.reserve(cfg.numLayers);
        for(uint32_t layer = 0u; layer < cfg.numLayers; ++layer)
        {
            auto const headDim = cfg.hiddenSize / cfg.numHeads;
            layers.push_back(
                alpaka::nn::onHost::inference::TransformerBlockWeights<T_Type, VectorBuffer, MatrixBuffer>{
                    load1D(cfg.hiddenSize),
                    load1D(cfg.hiddenSize),
                    load2D(cfg.hiddenSize, cfg.hiddenSize),
                    load2D(cfg.hiddenSize, cfg.numKeyValueHeads * headDim),
                    load2D(cfg.hiddenSize, cfg.numKeyValueHeads * headDim),
                    load2D(cfg.hiddenSize, cfg.hiddenSize),
                    load2D(cfg.hiddenSize, cfg.intermediateSize),
                    load2D(cfg.hiddenSize, cfg.intermediateSize),
                    load2D(cfg.intermediateSize, cfg.hiddenSize),
                    cfg.numHeads,
                    cfg.numKeyValueHeads,
                    headDim,
                    cfg.rmsNormEpsilon});
        }
        auto finalNorm = load1D(cfg.hiddenSize);
        auto lmHead = load2D(cfg.hiddenSize, cfg.vocabSize);
        alpaka::onHost::wait(queue);

        return DecoderModel<T_Type, VectorBuffer, MatrixBuffer>{cfg, embedding, std::move(layers), finalNorm, lmHead};
    }

    template<typename T_Type, typename T_Device>
    auto makeRopeTables(T_Device const& device, uint32_t positions, uint32_t pairCount, T_Type theta)
    {
        auto hostCos = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{positions, pairCount});
        auto hostSin = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{positions, pairCount});
        for(uint32_t pos = 0u; pos < positions; ++pos)
        {
            for(uint32_t pair = 0u; pair < pairCount; ++pair)
            {
                auto const exponent = static_cast<T_Type>(2u * pair) / static_cast<T_Type>(pairCount * 2u);
                auto const angle = static_cast<T_Type>(pos) / std::pow(theta, exponent);
                hostCos[alpaka::Vec{pos, pair}] = std::cos(angle);
                hostSin[alpaka::Vec{pos, pair}] = std::sin(angle);
            }
        }

        auto deviceCopy = device;
        auto queue = deviceCopy.makeQueue();
        auto devCos = alpaka::onHost::allocLike(device, hostCos);
        auto devSin = alpaka::onHost::allocLike(device, hostSin);
        alpaka::onHost::memcpy(queue, devCos, hostCos);
        alpaka::onHost::memcpy(queue, devSin, hostSin);
        alpaka::onHost::wait(queue);
        return std::pair{devCos, devSin};
    }

    template<typename T_Model>
    auto prefill(
        auto& queue,
        auto exec,
        T_Model const& model,
        std::vector<uint32_t> const& tokenIds,
        auto& cache,
        std::string_view traceLabel = "prefill")
    {
        using T_Type = typename T_Model::value_type;
        if(tokenIds.empty())
            throw std::invalid_argument{"prefill requires at least one token."};

        auto hostTokens = alpaka::onHost::allocHost<uint32_t>(static_cast<uint32_t>(tokenIds.size()));
        for(uint32_t i = 0u; i < tokenIds.size(); ++i)
            hostTokens[alpaka::Vec{i}] = tokenIds[i];
        auto devTokens = alpaka::onHost::allocLike(queue.getDevice(), hostTokens);
        alpaka::onHost::memcpy(queue, devTokens, hostTokens);

        auto hidden = alpaka::onHost::alloc<T_Type>(
            queue.getDevice(),
            alpaka::Vec{static_cast<uint32_t>(tokenIds.size()), model.config.hiddenSize});
        alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, devTokens, model.embedding, hidden);
        detail::traceTensor<T_Type>(queue, hidden, std::string{traceLabel} + " embedding");

        auto ropeTables = makeRopeTables<T_Type>(
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
            detail::traceTensor<T_Type>(queue, hidden, std::string{traceLabel} + " layer " + std::to_string(layer));
        }

        auto norm = alpaka::onHost::alloc<T_Type>(queue.getDevice(), hidden.getExtents());
        alpaka::nn::onHost::nn::rmsNorm<T_Type>(
            queue,
            exec,
            hidden,
            model.finalNorm,
            norm,
            static_cast<T_Type>(model.config.rmsNormEpsilon));
        detail::traceTensor<T_Type>(queue, norm, std::string{traceLabel} + " final norm");
        auto logits = alpaka::onHost::alloc<T_Type>(
            queue.getDevice(),
            alpaka::Vec{static_cast<uint32_t>(tokenIds.size()), model.config.vocabSize});
        alpaka::nn::onHost::gemm<T_Type>(queue, exec, norm, model.lmHead, logits);
        detail::traceTensor<T_Type>(queue, logits, std::string{traceLabel} + " full logits");
        auto lastLogits = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, model.config.vocabSize});
        alpaka::onHost::memcpy(
            queue,
            lastLogits,
            logits.getSubView(
                alpaka::Vec{static_cast<uint32_t>(tokenIds.size() - 1u), 0u},
                alpaka::Vec{1u, model.config.vocabSize}));
        alpaka::onHost::wait(queue);
        detail::traceTensor<T_Type>(queue, lastLogits, std::string{traceLabel} + " last logits");
        return lastLogits;
    }

    template<typename T_Model>
    auto prefill(auto& queue, auto exec, T_Model const& model, std::vector<uint32_t> const& tokenIds)
    {
        using T_Type = typename T_Model::value_type;
        auto cache = alpaka::nn::onHost::inference::makeKvCache<T_Type>(
            queue.getDevice(),
            model.config.numLayers,
            1u,
            model.config.numKeyValueHeads,
            static_cast<uint32_t>(tokenIds.size()),
            model.config.hiddenSize / model.config.numHeads);
        return prefill(queue, exec, model, tokenIds, cache, "implicit prefill");
    }

    template<typename T_Model>
    auto decodeStep(auto& queue, auto exec, T_Model const& model, auto& cache, uint32_t tokenId)
    {
        using T_Type = typename T_Model::value_type;

        auto hostToken = alpaka::onHost::allocHost<uint32_t>(1u);
        hostToken[alpaka::Vec{0u}] = tokenId;
        auto devToken = alpaka::onHost::allocLike(queue.getDevice(), hostToken);
        alpaka::onHost::memcpy(queue, devToken, hostToken);

        auto hidden = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, model.config.hiddenSize});
        alpaka::nn::onHost::nn::embeddingLookup<T_Type>(queue, exec, devToken, model.embedding, hidden);

        auto const currentContext = cache.length(0u, 0u);
        auto ropeTables = makeRopeTables<T_Type>(
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
        }

        auto norm = alpaka::onHost::alloc<T_Type>(queue.getDevice(), hidden.getExtents());
        alpaka::nn::onHost::nn::rmsNorm<T_Type>(
            queue,
            exec,
            hidden,
            model.finalNorm,
            norm,
            static_cast<T_Type>(model.config.rmsNormEpsilon));
        auto logits = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{1u, model.config.vocabSize});
        alpaka::nn::onHost::gemm<T_Type>(queue, exec, norm, model.lmHead, logits);
        alpaka::onHost::wait(queue);
        return logits;
    }
} // namespace alpaka::nn::onHost::model
