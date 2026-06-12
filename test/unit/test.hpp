/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <type_traits>

namespace alpaka::nn::test
{
    using TestApis = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

    template<typename T_Type>
    inline void checkValue(T_Type actual, T_Type expected, double epsilon = 1.0e-5, double margin = 1.0e-6)
    {
        if constexpr(std::floating_point<T_Type>)
            CHECK(actual == Catch::Approx(expected).epsilon(epsilon).margin(margin));
        else
            CHECK(actual == expected);
    }
} // namespace alpaka::nn::test
