/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/model/decoder.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace alpaka::nn::onHost::inference
{
    namespace detail
    {
        template<typename T_Type>
        std::string formatTopLogits(std::string_view label, auto const& hostLogits, uint32_t row, uint32_t topK = 10u)
        {
            auto const vocabSize = static_cast<uint32_t>(hostLogits.getExtents()[1]);
            std::vector<std::pair<T_Type, uint32_t>> ranked;
            ranked.reserve(vocabSize);

            for(uint32_t token = 0u; token < vocabSize; ++token)
                ranked.emplace_back(hostLogits[alpaka::Vec{row, token}], token);

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

        inline bool debugTopKEnabled()
        {
            auto const* env = std::getenv("ALPAKANN_DEBUG_TOPK");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }

        inline void printDiagnostic(std::string const& message)
        {
            if(debugTopKEnabled())
                std::fprintf(stderr, "%s\n", message.c_str());
        }
    } // namespace detail

    template<typename T_Type>
    uint32_t argmax(auto const& hostLogits, uint32_t row)
    {
        auto const vocab = static_cast<uint32_t>(hostLogits.getExtents()[1]);
        uint32_t best = 0u;
        T_Type bestValue = hostLogits[alpaka::Vec{row, 0u}];
        for(uint32_t col = 1u; col < vocab; ++col)
        {
            auto const value = hostLogits[alpaka::Vec{row, col}];
            if(value > bestValue)
            {
                bestValue = value;
                best = col;
            }
        }
        return best;
    }

    template<typename T_Model>
    std::vector<uint32_t> generateGreedy(
        auto& queue,
        auto exec,
        T_Model const& model,
        std::vector<uint32_t> tokens,
        uint32_t maxNewTokens)
    {
        if(maxNewTokens == 0u)
            return tokens;

        auto cache = alpaka::nn::onHost::inference::makeKvCache<typename T_Model::value_type>(
            queue.getDevice(),
            model.config.numLayers,
            1u,
            model.config.numKeyValueHeads,
            static_cast<uint32_t>(tokens.size() + maxNewTokens),
            model.config.hiddenSize / model.config.numHeads);
        auto logits = alpaka::nn::onHost::model::prefill(queue, exec, model, tokens, cache, "generate prefill");
        for(uint32_t step = 0u; step < maxNewTokens; ++step)
        {
            auto hostLogits = alpaka::onHost::allocHost<typename T_Model::value_type>(logits.getExtents());
            alpaka::onHost::memcpy(queue, hostLogits, logits);
            alpaka::onHost::wait(queue);
            auto next = argmax<typename T_Model::value_type>(hostLogits, 0u);
            if(detail::debugTopKEnabled())
            {
                auto message = detail::formatTopLogits<typename T_Model::value_type>(
                    "generate step " + std::to_string(step) + " logits",
                    hostLogits,
                    0u);
                message += " selected=(" + std::to_string(next) + ", "
                           + std::to_string(hostLogits[alpaka::Vec{0u, next}]) + ")";
                detail::printDiagnostic(message);
            }
            tokens.push_back(next);
            if(next == model.config.eosTokenId)
                break;
            logits = alpaka::nn::onHost::model::decodeStep(queue, exec, model, cache, next);
        }
        return tokens;
    }
} // namespace alpaka::nn::onHost::inference
