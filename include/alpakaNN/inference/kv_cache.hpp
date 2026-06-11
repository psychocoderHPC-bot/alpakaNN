/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpakaNN/detail/launch.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace alpakaNN::inference
{
    template<typename T_Type, typename T_Buffer>
    struct KvCache
    {
        uint32_t layers;
        uint32_t batchSize;
        uint32_t numKeyValueHeads;
        uint32_t maxContext;
        uint32_t headDim;
        std::vector<T_Buffer> keys;
        std::vector<T_Buffer> values;
        std::vector<uint32_t> lengths;

        template<typename T_Device>
        KvCache(T_Device const& device, uint32_t numLayers, uint32_t batch, uint32_t kvHeads, uint32_t context, uint32_t dim)
            : layers(numLayers)
            , batchSize(batch)
            , numKeyValueHeads(kvHeads)
            , maxContext(context)
            , headDim(dim)
            , lengths(numLayers * batch, 0u)
        {
            auto const extent = alpaka::Vec{batch, kvHeads, context, dim};
            for(uint32_t layer = 0u; layer < numLayers; ++layer)
            {
                keys.emplace_back(alpaka::onHost::alloc<T_Type>(device, extent));
                values.emplace_back(alpaka::onHost::alloc<T_Type>(device, extent));
            }
        }

        auto keyView(uint32_t layer) const
        {
            return keys.at(layer).getView();
        }

        auto valueView(uint32_t layer) const
        {
            return values.at(layer).getView();
        }

        uint32_t length(uint32_t layer, uint32_t batch) const
        {
            return lengths.at(layer * batchSize + batch);
        }

        void setLength(uint32_t layer, uint32_t batch, uint32_t value)
        {
            lengths.at(layer * batchSize + batch) = value;
        }

        void append(auto& queue, auto exec, uint32_t layer, uint32_t batch, uint32_t token, auto const& keyToken, auto const& valueToken)
        {
            alpaka::unused(exec);
            if(keyToken.getExtents() != valueToken.getExtents())
                throw std::invalid_argument{"KvCache append shape mismatch."};
            if(token >= maxContext)
                throw std::out_of_range{"KvCache append exceeded max context."};
            for(uint32_t head = 0u; head < numKeyValueHeads; ++head)
            {
                alpaka::onHost::memcpy(
                    queue,
                    keys.at(layer).getSubView(alpaka::Vec{batch, head, token, 0u}, alpaka::Vec{1u, 1u, 1u, headDim}),
                    keyToken.getSubView(alpaka::Vec{0u, 0u, head, 0u}, alpaka::Vec{1u, 1u, 1u, headDim}));
                alpaka::onHost::memcpy(
                    queue,
                    values.at(layer).getSubView(alpaka::Vec{batch, head, token, 0u}, alpaka::Vec{1u, 1u, 1u, headDim}),
                    valueToken.getSubView(alpaka::Vec{0u, 0u, head, 0u}, alpaka::Vec{1u, 1u, 1u, headDim}));
            }
            setLength(layer, batch, token + 1u);
        }

        auto getKeys(uint32_t layer, uint32_t batch, uint32_t tokenCount) const
        {
            return keys.at(layer).getSubView(
                alpaka::Vec{batch, 0u, 0u, 0u},
                alpaka::Vec{1u, numKeyValueHeads, tokenCount, headDim});
        }

        auto getValues(uint32_t layer, uint32_t batch, uint32_t tokenCount) const
        {
            return values.at(layer).getSubView(
                alpaka::Vec{batch, 0u, 0u, 0u},
                alpaka::Vec{1u, numKeyValueHeads, tokenCount, headDim});
        }
    };

    template<typename T_Type, typename T_Device>
    auto makeKvCache(T_Device const& device, uint32_t layers, uint32_t batchSize, uint32_t numKeyValueHeads, uint32_t maxContext, uint32_t headDim)
    {
        using Buffer = decltype(alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{1u, 1u, 1u, 1u}));
        return KvCache<T_Type, Buffer>{device, layers, batchSize, numKeyValueHeads, maxContext, headDim};
    }
} // namespace alpakaNN::inference
