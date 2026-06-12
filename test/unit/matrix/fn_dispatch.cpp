/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "../test.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdio>

namespace
{
    ALPAKA_FN_SYMBOL(OpenBlasFnProbe, alpaka::fn::Fallback::toAlpaka, alpaka::fn::Registration::enforced);

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    constexpr void fnRegister(OpenBlasFnProbe::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>)
    {
    }

#if ALPAKANN_OPENBLAS_ENABLED
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    constexpr void fnRegister(OpenBlasFnProbe::Spec<alpaka::api::Host, T_DeviceKind>)
    {
    }
#endif

    template<alpaka::concepts::DeviceKind T_DeviceKind>
    int fnDispatch(OpenBlasFnProbe::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>, auto const&)
    {
        std::printf("OpenBlasFnProbe: alpaka fallback dispatch\n");
        return 202;
    }

#if ALPAKANN_OPENBLAS_ENABLED
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    int fnDispatch(OpenBlasFnProbe::Spec<alpaka::api::Host, T_DeviceKind>, auto const&)
    {
        std::printf("OpenBlasFnProbe: host specialization dispatch\n");
        return 101;
    }
#endif
} // namespace

using TestApis = alpakaNN::test::TestApis;

TEMPLATE_LIST_TEST_CASE(
    "alpaka fn dispatch selects host specialization when OpenBLAS is enabled",
    "[matrix][fn]",
    TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }

    auto device = selector.makeDevice(0);

    if constexpr(device.getApi() == alpaka::api::host)
    {
#if ALPAKANN_OPENBLAS_ENABLED
        STATIC_CHECK(OpenBlasFnProbe::isRegistered(device));
        CHECK(OpenBlasFnProbe::call(device) == 101);
#else
        STATIC_CHECK_FALSE(OpenBlasFnProbe::isRegistered(device));
        STATIC_CHECK(OpenBlasFnProbe::hasRegisteredFallback(device));
        CHECK(OpenBlasFnProbe::call(device) == 202);
#endif
    }
    else
    {
        STATIC_CHECK_FALSE(OpenBlasFnProbe::isRegistered(device));
        STATIC_CHECK(OpenBlasFnProbe::hasRegisteredFallback(device));
        CHECK(OpenBlasFnProbe::call(device) == 202);
    }
}
