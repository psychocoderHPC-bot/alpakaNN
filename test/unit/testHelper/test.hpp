/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace alpaka::nn::test
{
    using TestApis = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;

    //! Value types exercised by the component tests. Every test must cover both single and double precision.
    using TestValueTypes = std::tuple<float, double>;

    //! Compile-time double-precision capability for a device.
    //!
    //! oneAPI GPU devices can lack native fp64 (e.g. Intel Arc), so double kernels must not be instantiated for
    //! them; otherwise alpaka3's SYCL kernel-bundle build fails and the whole spec is skipped. The guard is used
    //! as `if constexpr(supportsFp64(device))` around the double instantiation so fp64-less devices can still run
    //! the float cases. Mirrors alpakaVendor's supportsDoubleGemm() in test/integr/blas_gemm.cpp.
    //!
    //! Double precision can be enabled on Intel Arc through fp64 emulation by exporting
    //! `IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1`; alpakaNN does not enable this in CMake and relies
    //! on this capability guard instead.
    template<typename T_Device>
    consteval bool supportsFp64(T_Device const&)
    {
        using Api = std::remove_cvref_t<decltype(alpaka::getApi(std::declval<T_Device>()))>;
        using DeviceKind = std::remove_cvref_t<decltype(alpaka::getDeviceKind(std::declval<T_Device>()))>;
        if constexpr(std::same_as<Api, alpaka::api::OneApi>)
            return std::same_as<DeviceKind, alpaka::deviceKind::Cpu>;
        else
            return true;
    }

    template<typename T_Type>
    inline void checkValue(T_Type actual, T_Type expected, double epsilon = 1.0e-5, double margin = 1.0e-6)
    {
        if constexpr(std::floating_point<T_Type>)
            CHECK(actual == Catch::Approx(expected).epsilon(epsilon).margin(margin));
        else
            CHECK(actual == expected);
    }
} // namespace alpaka::nn::test
