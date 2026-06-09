/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/model/decoder.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <vector>

namespace alpakaNN::inference
{
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
    std::vector<uint32_t> generateGreedy(auto& queue, auto exec, T_Model const& model, std::vector<uint32_t> tokens, uint32_t maxNewTokens)
    {
        for(uint32_t step = 0u; step < maxNewTokens; ++step)
        {
            auto logits = alpakaNN::model::prefill(queue, exec, model, tokens);
            auto hostLogits = alpaka::onHost::allocHost<typename T_Model::value_type>(logits.getExtents());
            alpaka::onHost::memcpy(queue, hostLogits, logits);
            alpaka::onHost::wait(queue);
            auto next = argmax<typename T_Model::value_type>(hostLogits, static_cast<uint32_t>(tokens.size() - 1u));
            tokens.push_back(next);
            if(next == model.config.eosTokenId)
                break;
        }
        return tokens;
    }
} // namespace alpakaNN::inference
