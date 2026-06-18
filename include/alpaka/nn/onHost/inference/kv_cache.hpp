/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/internal/launch.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace alpaka::nn::onHost::inference
{
    namespace detail
    {
        inline bool cacheTraceEnabled()
        {
            auto const* env = std::getenv("ALPAKANN_DEBUG_CACHE_TRACE");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }

        inline bool cacheVerifyEnabled()
        {
            auto const* env = std::getenv("ALPAKANN_DEBUG_CACHE_VERIFY");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }

        inline void cachePrintTrace(std::string const& message)
        {
            if(cacheTraceEnabled() || cacheVerifyEnabled())
                std::fprintf(stderr, "%s\n", message.c_str());
        }

        template<typename T_Type>
        void traceTokenCompare(
            auto& queue,
            auto const& source,
            auto const& target,
            std::string const& label,
            T_Type tolerance = static_cast<T_Type>(1.0e-5))
        {
            if(!cacheVerifyEnabled())
                return;

            auto hostSource = alpaka::onHost::allocHost<T_Type>(source.getExtents());
            auto hostTarget = alpaka::onHost::allocHost<T_Type>(target.getExtents());
            alpaka::onHost::memcpy(queue, hostSource, source);
            alpaka::onHost::memcpy(queue, hostTarget, target);
            alpaka::onHost::wait(queue);

            uint32_t mismatchCount = 0u;
            float maxAbsDiff = 0.0f;
            std::string firstIdx = "n/a";
            T_Type firstSource{};
            T_Type firstTarget{};
            for(auto idx : alpaka::IdxRange{hostSource.getExtents()})
            {
                auto const lhs = hostSource[idx];
                auto const rhs = hostTarget[idx];
                auto const diff = static_cast<float>(std::fabs(static_cast<double>(lhs - rhs)));
                maxAbsDiff = std::max(maxAbsDiff, diff);
                if(diff > static_cast<float>(tolerance))
                {
                    if(mismatchCount == 0u)
                    {
                        std::ostringstream idxStream;
                        idxStream << idx;
                        firstIdx = idxStream.str();
                        firstSource = lhs;
                        firstTarget = rhs;
                    }
                    ++mismatchCount;
                }
            }

            std::ostringstream os;
            os << label << " mismatchCount=" << mismatchCount << " maxAbsDiff=" << maxAbsDiff;
            if(mismatchCount != 0u)
                os << " firstIdx=" << firstIdx << " src=" << firstSource << " dst=" << firstTarget;
            cachePrintTrace(os.str());
        }

        template<typename T_Type>
        struct CacheAppendTokenKernel
        {
            uint32_t batch;
            uint32_t token;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto dst, auto src) const
            {
                auto const extents = src.getExtents();
                for(auto idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{extents}))
                {
                    dst[alpaka::Vec{batch, idx[2], token, idx[3]}] = src[idx];
                }
            }
        };
    } // namespace detail

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
        KvCache(
            T_Device const& device,
            uint32_t numLayers,
            uint32_t batch,
            uint32_t kvHeads,
            uint32_t context,
            uint32_t dim)
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

        void append(
            auto& queue,
            auto exec,
            uint32_t layer,
            uint32_t batch,
            uint32_t token,
            auto const& keyToken,
            auto const& valueToken)
        {
            alpaka::unused(exec);
            if(keyToken.getExtents() != valueToken.getExtents())
                throw std::invalid_argument{"KvCache append shape mismatch."};
            if(token >= maxContext)
                throw std::out_of_range{"KvCache append exceeded max context."};
            if(detail::cacheTraceEnabled() || detail::cacheVerifyEnabled())
            {
                std::ostringstream os;
                os << "kv append layer=" << layer << " batch=" << batch << " token=" << token
                   << " keyExtents=" << keyToken.getExtents() << " cacheExtents=" << keys.at(layer).getExtents();
                detail::cachePrintTrace(os.str());
            }
            queue.enqueue(
                alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, keyToken.getExtents()),
                alpaka::KernelBundle{detail::CacheAppendTokenKernel<T_Type>{batch, token}, keys.at(layer), keyToken});
            queue.enqueue(
                alpaka::nn::onHost::internal::makeFrameSpec(queue.getDevice(), exec, valueToken.getExtents()),
                alpaka::KernelBundle{
                    detail::CacheAppendTokenKernel<T_Type>{batch, token},
                    values.at(layer),
                    valueToken});
            setLength(layer, batch, token + 1u);
            if(detail::cacheVerifyEnabled())
            {
                auto keySlice = keys.at(layer).getSubView(
                    alpaka::Vec{batch, 0u, token, 0u},
                    alpaka::Vec{1u, numKeyValueHeads, 1u, headDim});
                auto valueSlice = values.at(layer).getSubView(
                    alpaka::Vec{batch, 0u, token, 0u},
                    alpaka::Vec{1u, numKeyValueHeads, 1u, headDim});
                detail::traceTokenCompare<T_Type>(
                    queue,
                    keyToken,
                    keySlice,
                    "kv append verify keys layer=" + std::to_string(layer) + " token=" + std::to_string(token));
                detail::traceTokenCompare<T_Type>(
                    queue,
                    valueToken,
                    valueSlice,
                    "kv append verify values layer=" + std::to_string(layer) + " token=" + std::to_string(token));
            }
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
    auto makeKvCache(
        T_Device const& device,
        uint32_t layers,
        uint32_t batchSize,
        uint32_t numKeyValueHeads,
        uint32_t maxContext,
        uint32_t headDim)
    {
        using Buffer = decltype(alpaka::onHost::alloc<T_Type>(device, alpaka::Vec{1u, 1u, 1u, 1u}));
        return KvCache<T_Type, Buffer>{device, layers, batchSize, numKeyValueHeads, maxContext, headDim};
    }
} // namespace alpaka::nn::onHost::inference
