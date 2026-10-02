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

    /** @brief Fill a view with a constant value.
     *
     * @tparam T_Value Value type; must be assignable to the view element type.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor (for example the accelerator executor selected for @p queue).
     * @param out View to write; every element is set to @p value.
     * @param value Constant to write.
     *
     * @note Asynchronous: work is enqueued on @p queue, the caller owns @p out and must keep it alive and call
     *       `alpaka::onHost::wait(queue)` (or `out.keepAlive(queue)`) before reading it back.
     */
    template<typename T_Value>
    void fill(auto& queue, auto exec, alpaka::concepts::IMdSpan auto& out, T_Value value)
    {
        detail::enqueue(queue, exec, out.getExtents(), alpaka::KernelBundle{detail::FillKernel<T_Value>{value}, out});
    }

    /** @brief Copy one view into another with identical extents.
     *
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Source view; not modified.
     * @param out Destination view; must have exactly the same extents as @p in.
     *
     * @throw std::invalid_argument if the shapes differ.
     *
     * @note Asynchronous: work is enqueued on @p queue, the caller owns both views and must keep them alive and
     *       call `alpaka::onHost::wait(queue)` before reading @p out back.
     */
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

    /** @brief Apply a caller-provided unary functor elementwise.
     *
     * @tparam T_Op Functor type callable as `op(value)` from the device.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Input view; not modified.
     * @param out Output view; must have exactly the same extents as @p in.
     * @param op Unary functor invoked as `op(in[idx])`.
     *
     * @throw std::invalid_argument if the shapes differ.
     *
     * @note Asynchronous, caller-owned views (see `fill`).
     */
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

    /** @brief Apply a caller-provided binary functor elementwise.
     *
     * @tparam T_Op Functor type callable as `op(lhs, rhs)` from the device.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param lhs First input view; not modified.
     * @param rhs Second input view; not modified.
     * @param out Output view; must have the same extents as @p lhs and @p rhs.
     * @param op Binary functor invoked as `op(lhs[idx], rhs[idx])`.
     *
     * @throw std::invalid_argument if any of the three shapes differ.
     *
     * @note Asynchronous, caller-owned views (see `fill`).
     */
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

    /** @brief Elementwise addition `out = lhs + rhs`.
     *
     * @tparam T_Type Scalar type of the operation; must match the view value type.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param lhs First input view.
     * @param rhs Second input view.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void add(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::AddOp<T_Type>{});
    }

    /** @brief Elementwise subtraction `out = lhs - rhs`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param lhs Minuend input view.
     * @param rhs Subtrahend input view.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void sub(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::SubOp<T_Type>{});
    }

    /** @brief Elementwise multiplication `out = lhs * rhs`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param lhs First input view.
     * @param rhs Second input view.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void mul(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::MulOp<T_Type>{});
    }

    /** @brief Elementwise division `out = lhs / rhs`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param lhs Dividend input view.
     * @param rhs Divisor input view; the caller must guarantee non-zero entries.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void div(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::DivOp<T_Type>{});
    }

    /** @brief Multiply a view by a scalar: `out = in * scalar`.
     *
     * @tparam T_Type Scalar type.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Input view.
     * @param scalar Factor applied to every element.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void scale(auto& queue, auto exec, auto const& in, T_Type scalar, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::ScaleOp<T_Type>{scalar});
    }

    /** @brief BLAS-style `out = alpha * x + y`.
     *
     * @tparam T_Type Scalar type.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param alpha Scalar multiplier for @p x.
     * @param x First input view.
     * @param y Second input view; added unchanged.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void axpy(auto& queue, auto exec, T_Type alpha, auto const& x, auto const& y, auto& out)
    {
        binaryOp(queue, exec, x, y, out, detail::AxpyOp<T_Type>{alpha});
    }

    /** @brief Add a per-last-axis bias vector to every row of a view: `out = input + bias`.
     *
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param input Input view of rank >= 1.
     * @param bias 1D bias view whose extent must equal the last axis of @p input; `bias[last]` is added to every
     *             element with that last-axis coordinate.
     * @param out Output view; must have the same extents as @p input.
     *
     * @throw std::invalid_argument if @p bias is not 1D, if its extent does not match the last axis, or if
     *        @p input and @p out differ in shape.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
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

    /** @brief Elementwise type conversion `out = static_cast<T_Out>(in)`.
     *
     * @tparam T_Out Destination element type.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Input view.
     * @param out Output view; identical extents required, element type may differ from @p in.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Out>
    void cast(auto& queue, auto exec, auto const& in, auto& out)
    {
        detail::requireSameShape(in, out, "cast");
        using InType = typename std::remove_reference_t<decltype(in)>::value_type;
        unaryOp(queue, exec, in, out, detail::CastOp<T_Out, InType>{});
    }

    /** @brief Elementwise exponential `out = exp(in)`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Input view.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void exp(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::ExpOp<T_Type>{});
    }

    /** @brief Elementwise square root `out = sqrt(in)`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Non-negative input view.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void sqrt(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::SqrtOp<T_Type>{});
    }

    /** @brief Elementwise reciprocal square root `out = 1 / sqrt(in)`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Positive input view; the caller must guard against zero.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void rsqrt(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::RsqrtOp<T_Type>{});
    }

    /** @brief Elementwise rectified linear unit `out = max(in, 0)`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Input view.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void relu(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::ReluOp<T_Type>{});
    }

    /** @brief Elementwise logistic sigmoid `out = 1 / (1 + exp(-in))`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Input view.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void sigmoid(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::SigmoidOp<T_Type>{});
    }

    /** @brief Elementwise SiLU / swish `out = in * sigmoid(in)`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Input view.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void silu(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::SiluOp<T_Type>{});
    }

    /** @brief Elementwise exact GELU using the erf formulation: `0.5 * x * (1 + erf(x / sqrt(2)))`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param in Input view.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void gelu(auto& queue, auto exec, auto const& in, auto& out)
    {
        unaryOp(queue, exec, in, out, detail::GeluOp<T_Type>{});
    }

    /** @brief Elementwise SwiGLU gate: `out = lhs * sigmoid(lhs) * rhs`.
     *
     * @tparam T_Type Scalar type of the operation.
     * @param queue alpaka queue the work is enqueued on.
     * @param exec Executor selected for @p queue.
     * @param lhs Gate input view.
     * @param rhs Value input view; multiplied by the gated term.
     * @param out Output view; identical extents required.
     *
     * @throw std::invalid_argument if the shapes differ.
     * @note Asynchronous, caller-owned views (see `fill`).
     */
    template<typename T_Type>
    void swiglu(auto& queue, auto exec, auto const& lhs, auto const& rhs, auto& out)
    {
        binaryOp(queue, exec, lhs, rhs, out, detail::SwiGluOp<T_Type>{});
    }
} // namespace alpaka::nn::onHost::ops
