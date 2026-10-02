// SPDX-FileCopyrightText: René Widera
// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "../src/ModelLoader.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace heatclosure::test
{
    /** Feature-column count of the two supported fixture encodings. */
    inline std::size_t fixtureInputDim(FeatureEncoding encoding)
    {
        return encoding == FeatureEncoding::fourier_xy_k0_5 ? 27u : 3u;
    }

    /** Deterministic, finite float32 fixture payload in gate/up/down order.
     *
     * The trained checkpoint is intentionally not committed (regenerate with
     * tools/train_heat_closure.py), so tests build their own valid model instead of
     * depending on a repository binary. The values only need to be finite and match
     * the metadata checksum; they are not a trained closure.
     */
    inline std::vector<unsigned char> fixtureWeights(FeatureEncoding encoding = FeatureEncoding::raw)
    {
        auto const inputDim = fixtureInputDim(encoding);
        auto const width = std::size_t{64};
        auto const count = 2 * inputDim * width + width;
        std::vector<unsigned char> bytes(count * 4);
        for(std::size_t i = 0; i < count; ++i)
        {
            float const value = (static_cast<float>((i * 29) % 61) - 30.0f) * 0.05f;
            uint32_t raw = 0;
            std::memcpy(&raw, &value, 4);
            bytes[4 * i + 0] = static_cast<unsigned char>(raw & 0xffu);
            bytes[4 * i + 1] = static_cast<unsigned char>((raw >> 8) & 0xffu);
            bytes[4 * i + 2] = static_cast<unsigned char>((raw >> 16) & 0xffu);
            bytes[4 * i + 3] = static_cast<unsigned char>((raw >> 24) & 0xffu);
        }
        return bytes;
    }

    /** Minimal metadata manifest accepted by heatclosure::loadModel.
     *
     * Pretty-printed with two-space indentation so the loader test can mutate exact
     * substrings without changing the manifest contract.
     */
    inline std::string fixtureMetadata(
        std::string const& weightsFile,
        std::string const& sha256,
        FeatureEncoding encoding = FeatureEncoding::raw)
    {
        auto const inputDim = fixtureInputDim(encoding);
        auto const isFourier = encoding == FeatureEncoding::fourier_xy_k0_5;
        std::ostringstream o;
        o << "{\n";
        o << "  \"alpha_max\": 6.0,\n";
        o << "  \"alpha_min\": 0.01,\n";
        o << "  \"architecture\": \"gated_silu_bias_free_v1\",\n";
        o << "  \"beta\": 0.5,\n";
        o << "  \"beta_mismatch_policy\": \"reject\",\n";
        o << "  \"coordinate_domain\": [\n    0.0,\n    1.0\n  ],\n";
        o << "  \"dtype\": \"float32\",\n";
        if(isFourier)
        {
            o << "  \"feature_encoding\": \"fourier_xy_k0_5\",\n";
            o << "  \"feature_order\": [\n    \"u\",\n    \"x\",\n    \"y\"";
            for(int k = 0; k < 6; ++k)
            {
                auto const prefix = std::to_string(1 << k);
                o << ",\n    \"sin" << prefix << "pi_x\",\n    \"cos" << prefix << "pi_x\",\n    \"sin" << prefix
                  << "pi_y\",\n    \"cos" << prefix << "pi_y\"";
            }
            o << "\n  ],\n";
        }
        else
        {
            o << "  \"feature_order\": [\n    \"u\",\n    \"x\",\n    \"y\"\n  ],\n";
        }
        o << "  \"format\": \"alpakaNN-heat-closure-f32-" << (isFourier ? "v2" : "v1") << "\",\n";
        o << "  \"kind\": \"trained_model\",\n";
        o << "  \"material_formula\": \"base(x,y) * (1 + beta*u); inclusion if r2 < 0.12^2 => 0.02; else conductor "
             "if abs(y-0.5)<0.05 and 0.45<x<0.65 => 4.0; else 0.5+0.4*H(sin(6*pi*y)), H(z)=1 iff z>=0\",\n";
        o << "  \"output\": \"alpha_min + (alpha_max - alpha_min) * sigmoid(z)\",\n";
        o << "  \"temperature_domain\": [\n    0.0,\n    1.0\n  ],\n";
        o << "  \"weight_layout\": \"row-major [in,out], little-endian float32; gate,up,down concatenated\",\n";
        o << "  \"weight_shapes\": [\n    [\n      " << inputDim << ",\n      64\n    ],\n    [\n      " << inputDim
          << ",\n      64\n    ],\n";
        o << "    [\n      64,\n      1\n    ]\n  ],\n";
        o << "  \"weights_file\": \"" << weightsFile << "\",\n";
        o << "  \"weights_sha256\": \"" << sha256 << "\",\n";
        o << "  \"width\": 64\n";
        o << "}\n";
        return o.str();
    }

    /** Write `weights.bin` plus `weights.bin.metadata.json` into dir and return the weights path. */
    inline std::filesystem::path writeFixture(
        std::filesystem::path const& dir,
        FeatureEncoding encoding = FeatureEncoding::raw)
    {
        std::filesystem::create_directories(dir);
        auto const weights = dir / "weights.bin";
        auto const bytes = fixtureWeights(encoding);
        std::ofstream out(weights, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<char const*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        out.close();
        std::ofstream meta(weights.string() + ".metadata.json", std::ios::trunc);
        meta << fixtureMetadata("weights.bin", heatclosure::detail::sha256(bytes), encoding);
        meta.close();
        return weights;
    }
} // namespace heatclosure::test
