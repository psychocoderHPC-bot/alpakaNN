/*
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

namespace alpaka::nn::onHost::internal
{
    inline auto makeFrameSpec(
        auto const& device,
        alpaka::concepts::Executor auto exec,
        alpaka::concepts::Vector auto const& extents)
    {
        return alpaka::onHost::getFrameSpec(device, exec, extents);
    }
} // namespace alpaka::nn::onHost::internal
