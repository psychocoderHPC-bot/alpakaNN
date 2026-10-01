// SPDX-License-Identifier: MPL-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace heatclosure
{
    struct Model
    {
        double alphaMin{}, alphaMax{}, beta{};
        std::array<float, 192> gate{}, up{};
        std::array<float, 64> down{};
    };

    inline Model loadModel(std::filesystem::path const& weights, double beta)
    {
        auto path = weights;
        path += ".metadata.json";
        std::ifstream mf(path);
        if(!mf)
            throw std::runtime_error("cannot open model metadata: " + path.string());
        std::string m((std::istreambuf_iterator<char>(mf)), {});
        auto str = [&](char const* k)
        {
            std::smatch x;
            std::regex r("\\\"" + std::string(k) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
            if(!std::regex_search(m, x, r))
                throw std::runtime_error(std::string("missing metadata: ") + k);
            return x[1].str();
        };
        auto num = [&](char const* k)
        {
            std::smatch x;
            std::regex r("\\\"" + std::string(k) + "\\\"\\s*:\\s*([-+0-9.eE]+)");
            if(!std::regex_search(m, x, r))
                throw std::runtime_error(std::string("missing metadata: ") + k);
            double v = std::stod(x[1]);
            if(!std::isfinite(v))
                throw std::runtime_error("non-finite metadata");
            return v;
        };
        if(str("format") != "alpakaNN-heat-closure-f32-v1" || str("architecture") != "gated_silu_bias_free_v1"
           || str("dtype") != "float32"
           || str("weight_layout") != "row-major [in,out], little-endian float32; gate,up,down concatenated")
            throw std::runtime_error("unsupported model format");
        if(m.find("\"feature_order\": [\n    \"u\",\n    \"x\",\n    \"y\"") == std::string::npos
           || m.find(
                  "\"weight_shapes\": [\n    [\n      3,\n      64\n    ],\n    [\n      3,\n      64\n    ],\n    "
                  "[\n      64,\n      1")
                  == std::string::npos)
            throw std::runtime_error("unsupported features/shapes");
        Model o;
        o.alphaMin = num("alpha_min");
        o.alphaMax = num("alpha_max");
        o.beta = num("beta");
        if(!(o.alphaMin > 0 && o.alphaMax > o.alphaMin) || beta != o.beta)
            throw std::runtime_error("invalid bounds or model beta mismatch");
        std::ifstream f(weights, std::ios::binary);
        if(!f)
            throw std::runtime_error("cannot open model weights: " + weights.string());
        std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), {});
        if(b.size() != 1792)
            throw std::runtime_error("weights must contain exactly 1792 bytes");
        std::vector<float> w(448);
        for(size_t i = 0; i < w.size(); ++i)
        {
            uint32_t u = uint32_t(b[4 * i]) | uint32_t(b[4 * i + 1]) << 8 | uint32_t(b[4 * i + 2]) << 16
                         | uint32_t(b[4 * i + 3]) << 24;
            std::memcpy(&w[i], &u, 4);
            if(!std::isfinite(w[i]))
                throw std::runtime_error("non-finite model weight");
        }
        std::copy_n(w.begin(), 192, o.gate.begin());
        std::copy_n(w.begin() + 192, 192, o.up.begin());
        std::copy_n(w.begin() + 384, 64, o.down.begin());
        return o;
    }
} // namespace heatclosure
