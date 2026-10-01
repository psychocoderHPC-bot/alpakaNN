/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace alpaka::nn::onHost::ops
{
    namespace detail
    {
        template<typename T_Type>
        struct IdentityOp
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type value) const
            {
                return value;
            }
        };

        template<typename T_Type>
        struct AddOp
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type lhs, T_Type rhs) const
            {
                return lhs + rhs;
            }
        };

        template<typename T_Type>
        struct SubOp
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type lhs, T_Type rhs) const
            {
                return lhs - rhs;
            }
        };

        template<typename T_Type>
        struct MulOp
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type lhs, T_Type rhs) const
            {
                return lhs * rhs;
            }
        };

        template<typename T_Type>
        struct DivOp
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type lhs, T_Type rhs) const
            {
                return lhs / rhs;
            }
        };

        template<typename T_Type>
        struct ScaleOp
        {
            T_Type scalar;

            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type value) const
            {
                return value * scalar;
            }
        };

        template<typename T_Type>
        struct AxpyOp
        {
            T_Type alpha;

            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type x, T_Type y) const
            {
                return alpha * x + y;
            }
        };

        template<typename T_Out, typename T_In>
        struct CastOp
        {
            ALPAKA_FN_ACC constexpr T_Out operator()(T_In value) const
            {
                return static_cast<T_Out>(value);
            }
        };

        template<typename T_Type>
        struct ExpOp
        {
            ALPAKA_FN_ACC T_Type operator()(T_Type value) const
            {
                return alpaka::math::exp(value);
            }
        };

        template<typename T_Type>
        struct SqrtOp
        {
            ALPAKA_FN_ACC T_Type operator()(T_Type value) const
            {
                return alpaka::math::sqrt(value);
            }
        };

        template<typename T_Type>
        struct RsqrtOp
        {
            ALPAKA_FN_ACC T_Type operator()(T_Type value) const
            {
                return alpaka::math::rsqrt(value);
            }
        };

        template<typename T_Type>
        struct ReluOp
        {
            ALPAKA_FN_ACC constexpr T_Type operator()(T_Type value) const
            {
                return value > T_Type{} ? value : T_Type{};
            }
        };

        template<typename T_Type>
        struct SigmoidOp
        {
            ALPAKA_FN_ACC T_Type operator()(T_Type value) const
            {
                return T_Type{1} / (T_Type{1} + alpaka::math::exp(-value));
            }
        };

        template<typename T_Type>
        struct SiluOp
        {
            ALPAKA_FN_ACC T_Type operator()(T_Type value) const
            {
                auto const sigmoid = T_Type{1} / (T_Type{1} + alpaka::math::exp(-value));
                return value * sigmoid;
            }
        };

        template<typename T_Type>
        struct GeluOp
        {
            ALPAKA_FN_ACC T_Type operator()(T_Type value) const
            {
                constexpr auto invSqrt2 = static_cast<T_Type>(0.70710678118654752440);
                return static_cast<T_Type>(0.5) * value
                       * (static_cast<T_Type>(1) + alpaka::math::erf(value * invSqrt2));
            }
        };

        template<typename T_Type>
        struct SwiGluOp
        {
            ALPAKA_FN_ACC T_Type operator()(T_Type lhs, T_Type rhs) const
            {
                auto const sigmoid = T_Type{1} / (T_Type{1} + alpaka::math::exp(-lhs));
                return lhs * sigmoid * rhs;
            }
        };

        template<typename T_Op>
        struct UnaryKernel
        {
            T_Op op;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto in) const
            {
                for(auto idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{out.getExtents()}))
                    out[idx] = op(in[idx]);
            }
        };

        template<typename T_Op>
        struct BinaryKernel
        {
            T_Op op;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto lhs, auto rhs) const
            {
                for(auto idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{out.getExtents()}))
                    out[idx] = op(lhs[idx], rhs[idx]);
            }
        };

        template<typename T_Value>
        struct FillKernel
        {
            T_Value value;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
            {
                for(auto idx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{out.getExtents()}))
                    out[idx] = value;
            }
        };

        template<typename T_BiasValue>
        struct BiasAddKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto input, auto bias) const
            {
                auto const outExtents = out.getExtents();
                auto const axis = ALPAKA_TYPEOF(outExtents)::dim() - 1u;
                for(auto idx :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{outExtents}))
                    out[idx] = input[idx] + bias[alpaka::Vec{idx[axis]}];
            }
        };

        inline void requireSameShape(
            alpaka::concepts::IMdSpan auto const& lhs,
            alpaka::concepts::IMdSpan auto const& rhs,
            char const* what)
        {
            if(lhs.getExtents() != rhs.getExtents())
                throw std::invalid_argument{std::string(what) + " shape mismatch."};
        }

        inline void requireBiasShape(
            alpaka::concepts::IMdSpan auto const& input,
            alpaka::concepts::IMdSpan auto const& bias)
        {
            auto const inputExtents = input.getExtents();
            auto const biasExtents = bias.getExtents();

            if(ALPAKA_TYPEOF(biasExtents)::dim() != 1u)
                throw std::invalid_argument{"biasAdd expects a 1D bias view."};
            if(biasExtents[0] != inputExtents[ALPAKA_TYPEOF(inputExtents)::dim() - 1u])
                throw std::invalid_argument{"biasAdd expects bias extent to match the last axis."};
        }

        inline void enqueue(auto& queue, auto exec, alpaka::concepts::Vector auto const& extents, auto const& bundle)
        {
            queue.enqueue(alpaka::onHost::getFrameSpec(queue.getDevice(), exec, extents), bundle);
        }
    } // namespace detail

    template<typename T_Value>
    void fill(auto& queue, auto exec, alpaka::concepts::IMdSpan auto& out, T_Value value)
    {
        detail::enqueue(queue, exec, out.getExtents(), alpaka::KernelBundle{detail::FillKernel<T_Value>{value}, out});
    }

    void copy(auto& queue, auto exec, alpaka::concepts::IMdSpan auto const& in, alpaka::concepts::IMdSpan auto& out)
    {
        detail::requireSameShape(in, out, "copy");
        detail::enqueue(
            queue,
            exec,
            out.getExtents(),
            alpaka::KernelBundle{
                detail::UnaryKernel{detail::IdentityOp<typename std::remove_reference_t<decltype(out)>::value_type>{}},
                out,
                in});
    }

    template<typename T_Op>
    void unaryOp(
        auto& queue,
        auto exec,
        alpaka::concepts::IMdSpan auto const& in,
        alpaka::concepts::IMdSpan auto& out,
        T_Op op)
    {
        detail::requireSameShape(in, out, "unaryOp");
        detail::enqueue(queue, exec, out.getExtents(), alpaka::KernelBundle{detail::UnaryKernel<T_Op>{op}, out, in});
    }

    template<typename T_Op>
    void binaryOp(
        auto& queue,
        auto exec,
        alpaka::concepts::IMdSpan auto const& lhs,
        alpaka::concepts::IMdSpan auto const& rhs,
        alpaka::concepts::IMdSpan auto& out,
        T_Op op)
    {
        detail::requireSameShape(lhs, rhs, "binaryOp");
        detail::requireSameShape(lhs, out, "binaryOp");
        detail::enqueue(
            queue,
            exec,
            out.getExtents(),
            alpaka::KernelBundle{detail::BinaryKernel<T_Op>{op}, out, lhs, rhs});
    }

    template<typename T_Type>
    void add(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::AddOp<T_Type>{});
    }

    template<typename T_Type>
    void sub(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::SubOp<T_Type>{});
    }

    template<typename T_Type>
    void mul(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::MulOp<T_Type>{});
    }

    template<typename T_Type>
    void div(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::DivOp<T_Type>{});
    }

    template<typename T_Type>
    void scale(auto& queue, auto exec, auto const& in, T_Type scalar, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::ScaleOp<T_Type>{scalar});
    }

    template<typename T_Type>
    void axpy(auto& queue, auto exec, T_Type alpha, auto const& x, auto const& y, auto& out)
    {
        binaryOp(queue, exec, x, y, out, detail::AxpyOp<T_Type>{alpha});
    }

    void biasAdd(
        auto& queue,
        auto exec,
        alpaka::concepts::IMdSpan auto const& input,
        alpaka::concepts::IMdSpan auto const& bias,
        alpaka::concepts::IMdSpan auto& out)
    {
        detail::requireSameShape(input, out, "biasAdd");
        detail::requireBiasShape(input, bias);
        detail::enqueue(
            queue,
            exec,
            out.getExtents(),
            alpaka::KernelBundle{
                detail::BiasAddKernel<typename std::remove_reference_t<decltype(bias)>::value_type>{},
                out,
                input,
                bias});
    }

    template<typename T_Out>
    void cast(auto& queue, auto exec, auto const& in, auto& out)
    {
        detail::requireSameShape(in, out, "cast");
        using InType = typename std::remove_reference_t<decltype(in)>::value_type;
        unaryOp(queue, exec, in, out, detail::CastOp<T_Out, InType>{});
    }

    template<typename T_Type>
    void exp(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::ExpOp<T_Type>{});
    }

    template<typename T_Type>
    void sqrt(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::SqrtOp<T_Type>{});
    }

    template<typename T_Type>
    void rsqrt(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::RsqrtOp<T_Type>{});
    }

    template<typename T_Type>
    void relu(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::ReluOp<T_Type>{});
    }

    template<typename T_Type>
    void sigmoid(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::SigmoidOp<T_Type>{});
    }

    template<typename T_Type>
    void silu(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::SiluOp<T_Type>{});
    }

    template<typename T_Type>
    void gelu(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::GeluOp<T_Type>{});
    }

    template<typename T_Type>
    void swiglu(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::SwiGluOp<T_Type>{});
    }
} // namespace alpaka::nn::onHost::ops
