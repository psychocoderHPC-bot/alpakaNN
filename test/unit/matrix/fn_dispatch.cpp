/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "test.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/fn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdio>

namespace
{
    ALPAKA_FN_SYMBOL(OpenBlasFnProbe);

    int fnDispatch(OpenBlasFnProbe, auto const&)
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

using TestApis = alpaka::nn::test::TestApis;

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
        CHECK(OpenBlasFnProbe::call(device) == 101);
#else
        CHECK(OpenBlasFnProbe::call(device) == 202);
#endif
    }
    else
    {
        CHECK(OpenBlasFnProbe::call(device) == 202);
    }
}
