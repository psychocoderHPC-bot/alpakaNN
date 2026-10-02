/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file nn.hpp
 *
 * Umbrella header for the entire public human-facing alpakaNN API. Include this single header to get the
 * generic layout/shape helpers (`alpaka::nn::layout`, `alpaka::nn::shape`), the shared layout tags
 * (`alpaka::nn::RopeLayout`, `alpaka::nn::AttentionKvLayout`), the host view helpers, the vendor-BLAS matrix
 * wrappers and every host-callable neural-network primitive in `alpaka::nn::onHost`.
 *
 * All algorithms are header-only and enqueue work on a caller-provided alpaka queue. Unless stated otherwise an
 * operation is asynchronous with respect to the host: the caller must keep every view alive until completion and
 * must synchronize (for example with `alpaka::onHost::wait(queue)`) before reading results back.
 */

#include <alpaka/nn/core/layout.hpp>
#include <alpaka/nn/core/shape.hpp>
#include <alpaka/nn/nn/attention.hpp>
#include <alpaka/nn/nn/rope.hpp>
#include <alpaka/nn/onHost/core/view.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/matrix/gemv.hpp>
#include <alpaka/nn/onHost/nn/attention.hpp>
#include <alpaka/nn/onHost/nn/embedding.hpp>
#include <alpaka/nn/onHost/nn/mlp.hpp>
#include <alpaka/nn/onHost/nn/rms_norm.hpp>
#include <alpaka/nn/onHost/nn/rope.hpp>
#include <alpaka/nn/onHost/nn/softmax.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>
#include <alpaka/nn/onHost/ops/reduction.hpp>
