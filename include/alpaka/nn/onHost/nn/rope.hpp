/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn/rope.hpp>
#include <alpaka/nn/onAcc/internal/nn/rope.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace alpaka::nn::onHost::nn
{
    namespace internal
    {
        inline void validateRopeShape(
            auto const& in,
            auto const& out,
            auto const& cosTable,
            auto const& sinTable,
            uint32_t positionOffset)
        {
            auto const inExtents = in.getExtents();
            auto const outExtents = out.getExtents();
            auto const cosExtents = cosTable.getExtents();
            auto const sinExtents = sinTable.getExtents();

            if(inExtents != outExtents)
                throw std::invalid_argument{"rope shape mismatch."};
            auto const headDimAxis = ALPAKA_TYPEOF(inExtents)::dim() - 1u;
            auto const headDim = static_cast<uint32_t>(inExtents[headDimAxis]);
            if(headDim % 2u != 0u)
                throw std::invalid_argument{"rope requires an even head dimension."};
            if(cosExtents != sinExtents)
                throw std::invalid_argument{"rope cosine/sine table shape mismatch."};
            if(static_cast<uint32_t>(cosExtents[1]) != headDim / 2u)
                throw std::invalid_argument{"rope table pair count mismatch."};
            auto const tokenAxis = ALPAKA_TYPEOF(inExtents)::dim() == 4u ? 1u : 0u;
            if(positionOffset + static_cast<uint32_t>(inExtents[tokenAxis]) > static_cast<uint32_t>(cosExtents[0]))
                throw std::invalid_argument{"rope table does not cover requested positions."};
        }
    } // namespace internal

    template<typename T_Type>
    void rope(
        auto& queue,
        auto exec,
        auto const& in,
        auto const& cosTable,
        auto const& sinTable,
        auto& out,
        alpaka::nn::RopeLayout layout = alpaka::nn::RopeLayout::BTHD,
        uint32_t positionOffset = 0u)
    {
        internal::validateRopeShape(in, out, cosTable, sinTable, positionOffset);
        queue.enqueue(
            alpaka::onHost::getFrameSpec(queue.getDevice(), exec, out.getExtents()),
            alpaka::KernelBundle{
                alpaka::nn::onAcc::internal::nn::RopeSingleKernel<T_Type>{positionOffset, layout},
                out,
                in,
                cosTable,
                sinTable});
    }

    template<typename T_Type>
    void rope(
        auto& queue,
        auto exec,
        auto const& q,
        auto const& k,
        auto const& cosTable,
        auto const& sinTable,
        auto& outQ,
        auto& outK,
        alpaka::nn::RopeLayout layout = alpaka::nn::RopeLayout::BTHD,
        uint32_t positionOffset = 0u)
    {
        rope<T_Type>(queue, exec, q, cosTable, sinTable, outQ, layout, positionOffset);
        rope<T_Type>(queue, exec, k, cosTable, sinTable, outK, layout, positionOffset);
    }

    template<typename T_Type>
    void ropeInPlace(
        auto& queue,
        auto exec,
        auto& tensor,
        auto const& cosTable,
        auto const& sinTable,
        alpaka::nn::RopeLayout layout = alpaka::nn::RopeLayout::BTHD,
        uint32_t positionOffset = 0u)
    {
        auto tmp = alpaka::onHost::alloc<typename std::remove_reference_t<decltype(tensor)>::value_type>(
            queue.getDevice(),
            tensor.getExtents());
        rope<T_Type>(queue, exec, tensor, cosTable, sinTable, tmp, layout, positionOffset);
        alpaka::nn::onHost::ops::copy(queue, exec, tmp, tensor);
        alpaka::onHost::wait(queue);
    }

    template<typename T_Type>
    void ropeInPlace(
        auto& queue,
        auto exec,
        auto& q,
        auto& k,
        auto const& cosTable,
        auto const& sinTable,
        alpaka::nn::RopeLayout layout = alpaka::nn::RopeLayout::BTHD,
        uint32_t positionOffset = 0u)
    {
        ropeInPlace<T_Type>(queue, exec, q, cosTable, sinTable, layout, positionOffset);
        ropeInPlace<T_Type>(queue, exec, k, cosTable, sinTable, layout, positionOffset);
    }
} // namespace alpaka::nn::onHost::nn
