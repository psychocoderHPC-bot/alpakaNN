/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "alpaka/nn/core/shape.hpp"
#include "alpaka/nn/detail/launch.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace alpaka::nn::ops
{
    namespace detail
    {
        template<typename T_Type>
        struct IdentityTransform
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type value) const
            {
                return value;
            }
        };

        template<typename T_Type>
        struct SquareTransform
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type value) const
            {
                return value * value;
            }
        };

        template<typename T_Type>
        struct AddReducer
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type lhs, T_Type rhs) const
            {
                return lhs + rhs;
            }
        };

        template<typename T_Type>
        struct MaxReducer
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type lhs, T_Type rhs) const
            {
                return lhs > rhs ? lhs : rhs;
            }
        };

        template<typename T_Type>
        struct MeanFinalize
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type value, uint32_t count) const
            {
                return value / static_cast<T_Type>(count);
            }
        };

        template<typename T_Type>
        struct IdentityFinalize
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type value, uint32_t) const
            {
                return value;
            }
        };

        template<typename T_Type>
        struct MulTransform
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type lhs, T_Type rhs) const
            {
                return lhs * rhs;
            }
        };

        template<typename T_Transform, typename T_Reducer, typename T_Finalize>
        struct ReduceAxisKernel
        {
            uint32_t axis;
            T_Transform transform;
            T_Reducer reducer;
            T_Finalize finalize;
            typename T_Finalize::value_type init;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto in) const
            {
                auto const reduceExtent = static_cast<uint32_t>(in.getExtents()[axis]);
                for(auto outIdx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{out.getExtents()}))
                {
                    auto inIdx = outIdx;
                    auto accum = init;
                    for(uint32_t r = 0u; r < reduceExtent; ++r)
                    {
                        inIdx[axis] = r;
                        accum = reducer(accum, transform(in[inIdx]));
                    }
                    out[outIdx] = finalize(accum, reduceExtent);
                }
            }
        };

        template<typename T_Transform, typename T_Reducer, typename T_Finalize>
        struct ReduceAxisBinaryKernel
        {
            uint32_t axis;
            T_Transform transform;
            T_Reducer reducer;
            T_Finalize finalize;
            typename T_Finalize::value_type init;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto lhs, auto rhs) const
            {
                auto const reduceExtent = static_cast<uint32_t>(lhs.getExtents()[axis]);
                for(auto outIdx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{out.getExtents()}))
                {
                    auto inIdx = outIdx;
                    auto accum = init;
                    for(uint32_t r = 0u; r < reduceExtent; ++r)
                    {
                        inIdx[axis] = r;
                        accum = reducer(accum, transform(lhs[inIdx], rhs[inIdx]));
                    }
                    out[outIdx] = finalize(accum, reduceExtent);
                }
            }
        };

        template<typename T_Type>
        struct FinalizeWithType
        {
            using value_type = T_Type;
        };

        template<typename T_Type>
        struct IdentityFinalizeTyped : FinalizeWithType<T_Type>
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type value, uint32_t) const
            {
                return value;
            }
        };

        template<typename T_Type>
        struct MeanFinalizeTyped : FinalizeWithType<T_Type>
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type value, uint32_t count) const
            {
                return value / static_cast<T_Type>(count);
            }
        };

        inline auto makeReducedExtents(alpaka::concepts::Vector auto extents, uint32_t axis)
        {
            extents[axis] = 1u;
            return extents;
        }

        inline void validateReduction(
            alpaka::concepts::IMdSpan auto const& in,
            alpaka::concepts::IMdSpan auto const& out,
            uint32_t axis,
            char const* what)
        {
            if(in.getExtents().dim() != out.getExtents().dim())
                throw std::invalid_argument{std::string(what) + " requires output rank to match input rank."};
            if(axis >= in.getExtents().dim())
                throw std::invalid_argument{std::string(what) + " axis out of range."};
            if(out.getExtents() != makeReducedExtents(in.getExtents(), axis))
                throw std::invalid_argument{std::string(what) + " output extents mismatch."};
        }

        template<typename T_Kernel, typename... T_Args>
        void enqueueReduction(auto& queue, auto exec, auto const& extents, T_Kernel kernel, T_Args&&... args)
        {
            queue.enqueue(
                alpaka::nn::detail::makeFrameSpec(queue.getDevice(), exec, extents),
                alpaka::KernelBundle{kernel, ALPAKA_FORWARD(args)...});
        }
    } // namespace detail

    template<typename T_Type>
    void reduceSum(auto& queue, auto exec, auto const& in, auto& out, uint32_t axis)
    {
        detail::validateReduction(in, out, axis, "reduceSum");
        detail::enqueueReduction(
            queue,
            exec,
            out.getExtents(),
            detail::ReduceAxisKernel<
                detail::IdentityTransform<T_Type>,
                detail::AddReducer<T_Type>,
                detail::IdentityFinalizeTyped<T_Type>>{
                axis,
                detail::IdentityTransform<T_Type>{},
                detail::AddReducer<T_Type>{},
                detail::IdentityFinalizeTyped<T_Type>{},
                T_Type{}},
            out,
            in);
    }

    template<typename T_Type>
    void reduceMax(auto& queue, auto exec, auto const& in, auto& out, uint32_t axis)
    {
        detail::validateReduction(in, out, axis, "reduceMax");
        detail::enqueueReduction(
            queue,
            exec,
            out.getExtents(),
            detail::ReduceAxisKernel<
                detail::IdentityTransform<T_Type>,
                detail::MaxReducer<T_Type>,
                detail::IdentityFinalizeTyped<T_Type>>{
                axis,
                detail::IdentityTransform<T_Type>{},
                detail::MaxReducer<T_Type>{},
                detail::IdentityFinalizeTyped<T_Type>{},
                std::numeric_limits<T_Type>::lowest()},
            out,
            in);
    }

    template<typename T_Type>
    void reduceMean(auto& queue, auto exec, auto const& in, auto& out, uint32_t axis)
    {
        detail::validateReduction(in, out, axis, "reduceMean");
        detail::enqueueReduction(
            queue,
            exec,
            out.getExtents(),
            detail::ReduceAxisKernel<
                detail::IdentityTransform<T_Type>,
                detail::AddReducer<T_Type>,
                detail::MeanFinalizeTyped<T_Type>>{
                axis,
                detail::IdentityTransform<T_Type>{},
                detail::AddReducer<T_Type>{},
                detail::MeanFinalizeTyped<T_Type>{},
                T_Type{}},
            out,
            in);
    }

    template<typename T_Type>
    void reduceSumSquares(auto& queue, auto exec, auto const& in, auto& out, uint32_t axis)
    {
        detail::validateReduction(in, out, axis, "reduceSumSquares");
        detail::enqueueReduction(
            queue,
            exec,
            out.getExtents(),
            detail::ReduceAxisKernel<
                detail::SquareTransform<T_Type>,
                detail::AddReducer<T_Type>,
                detail::IdentityFinalizeTyped<T_Type>>{
                axis,
                detail::SquareTransform<T_Type>{},
                detail::AddReducer<T_Type>{},
                detail::IdentityFinalizeTyped<T_Type>{},
                T_Type{}},
            out,
            in);
    }

    template<typename T_Type>
    void dot(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out, uint32_t axis)
    {
        detail::validateReduction(lhs, out, axis, "dot");
        if(lhs.getExtents() != rhs.getExtents())
            throw std::invalid_argument{"dot shape mismatch."};
        detail::enqueueReduction(
            queue,
            exec,
            out.getExtents(),
            detail::ReduceAxisBinaryKernel<
                detail::MulTransform<T_Type>,
                detail::AddReducer<T_Type>,
                detail::IdentityFinalizeTyped<T_Type>>{
                axis,
                detail::MulTransform<T_Type>{},
                detail::AddReducer<T_Type>{},
                detail::IdentityFinalizeTyped<T_Type>{},
                T_Type{}},
            out,
            lhs,
            rhs);
    }

    inline auto makeReducedExtents(alpaka::concepts::Vector auto extents, uint32_t axis)
    {
        return detail::makeReducedExtents(extents, axis);
    }
} // namespace alpaka::nn::ops
