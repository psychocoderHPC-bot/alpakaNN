// SPDX-License-Identifier: MPL-2.0
#pragma once
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace heatclosure
{
    namespace detail
    {
        // Small strict JSON reader: metadata is untrusted input and must be parsed
        // structurally rather than searched as text. It intentionally supports the
        // complete JSON value grammar used by the model manifest.
        struct Json
        {
            using Array = std::vector<Json>;
            using Object = std::map<std::string, Json>;
            std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;

            Object const& object() const
            {
                return std::get<Object>(value);
            }

            Array const& array() const
            {
                return std::get<Array>(value);
            }

            std::string const& string() const
            {
                return std::get<std::string>(value);
            }

            double number() const
            {
                return std::get<double>(value);
            }
        };

        class Parser
        {
            std::string const& s;
            size_t p = 0;

            void ws()
            {
                while(p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t'))
                    ++p;
            }

            char take()
            {
                if(p == s.size())
                    fail();
                return s[p++];
            }

            [[noreturn]] void fail() const
            {
                throw std::runtime_error("malformed model JSON metadata");
            }

            uint16_t hex4()
            {
                uint16_t value = 0;
                for(int i = 0; i < 4; ++i)
                {
                    char c = take();
                    value = static_cast<uint16_t>(value << 4);
                    if(c >= '0' && c <= '9')
                        value = static_cast<uint16_t>(value + c - '0');
                    else if(c >= 'a' && c <= 'f')
                        value = static_cast<uint16_t>(value + c - 'a' + 10);
                    else if(c >= 'A' && c <= 'F')
                        value = static_cast<uint16_t>(value + c - 'A' + 10);
                    else
                        fail();
                }
                return value;
            }

            static void appendUtf8(std::string& out, uint32_t codepoint)
            {
                if(codepoint <= 0x7f)
                    out += static_cast<char>(codepoint);
                else if(codepoint <= 0x7ff)
                {
                    out += static_cast<char>(0xc0 | (codepoint >> 6));
                    out += static_cast<char>(0x80 | (codepoint & 0x3f));
                }
                else if(codepoint <= 0xffff)
                {
                    out += static_cast<char>(0xe0 | (codepoint >> 12));
                    out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
                    out += static_cast<char>(0x80 | (codepoint & 0x3f));
                }
                else
                {
                    out += static_cast<char>(0xf0 | (codepoint >> 18));
                    out += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
                    out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
                    out += static_cast<char>(0x80 | (codepoint & 0x3f));
                }
            }

            std::string str()
            {
                if(take() != '"')
                    fail();
                std::string out;
                while(p < s.size())
                {
                    char c = take();
                    if(c == '"')
                        return out;
                    if(static_cast<unsigned char>(c) < 0x20)
                        fail();
                    // JSON text is UTF-8. Preserve valid raw sequences, but reject
                    // malformed encodings (including overlongs and surrogate codepoints).
                    if(static_cast<unsigned char>(c) >= 0x80)
                    {
                        auto const lead = static_cast<unsigned char>(c);
                        int continuationCount = 0;
                        uint32_t codepoint = 0;
                        if(lead >= 0xc2 && lead <= 0xdf)
                        {
                            continuationCount = 1;
                            codepoint = lead & 0x1f;
                        }
                        else if(lead >= 0xe0 && lead <= 0xef)
                        {
                            continuationCount = 2;
                            codepoint = lead & 0x0f;
                        }
                        else if(lead >= 0xf0 && lead <= 0xf4)
                        {
                            continuationCount = 3;
                            codepoint = lead & 0x07;
                        }
                        else
                            fail();
                        out += c;
                        for(int i = 0; i < continuationCount; ++i)
                        {
                            auto const next = static_cast<unsigned char>(take());
                            if((next & 0xc0) != 0x80)
                                fail();
                            codepoint = (codepoint << 6) | (next & 0x3f);
                            out += static_cast<char>(next);
                        }
                        if((continuationCount == 2 && codepoint < 0x800)
                           || (continuationCount == 3 && codepoint < 0x1'0000)
                           || (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10'ffff)
                            fail();
                        continue;
                    }
                    if(c == '\\')
                    {
                        c = take();
                        if(c == '"' || c == '\\' || c == '/')
                            out += c;
                        else if(c == 'b')
                            out += '\b';
                        else if(c == 'f')
                            out += '\f';
                        else if(c == 'n')
                            out += '\n';
                        else if(c == 'r')
                            out += '\r';
                        else if(c == 't')
                            out += '\t';
                        else if(c == 'u')
                        {
                            uint32_t cp = hex4();
                            if(cp >= 0xd800 && cp <= 0xdbff)
                            {
                                if(take() != '\\' || take() != 'u')
                                    fail();
                                uint32_t low = hex4();
                                if(low < 0xdc00 || low > 0xdfff)
                                    fail();
                                cp = 0x1'0000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                            }
                            else if(cp >= 0xdc00 && cp <= 0xdfff)
                                fail();
                            appendUtf8(out, cp);
                        }
                        else
                            fail();
                    }
                    else
                        out += c;
                }
                fail();
            }

            Json val()
            {
                ws();
                if(p == s.size())
                    fail();
                if(s[p] == '"')
                    return Json{str()};
                if(s[p] == '{')
                {
                    ++p;
                    Json::Object o;
                    ws();
                    if(p < s.size() && s[p] == '}')
                    {
                        ++p;
                        return Json{o};
                    }
                    for(;;)
                    {
                        ws();
                        if(p == s.size() || s[p] != '"')
                            fail();
                        auto k = str();
                        ws();
                        if(take() != ':')
                            fail();
                        if(!o.emplace(k, val()).second)
                            fail();
                        ws();
                        char c = take();
                        if(c == '}')
                            break;
                        if(c != ',')
                            fail();
                    }
                    return Json{o};
                }
                if(s[p] == '[')
                {
                    ++p;
                    Json::Array a;
                    ws();
                    if(p < s.size() && s[p] == ']')
                    {
                        ++p;
                        return Json{a};
                    }
                    for(;;)
                    {
                        a.push_back(val());
                        ws();
                        char c = take();
                        if(c == ']')
                            break;
                        if(c != ',')
                            fail();
                    }
                    return Json{a};
                }
                if(s.compare(p, 4, "true") == 0)
                {
                    p += 4;
                    return Json{true};
                }
                if(s.compare(p, 5, "false") == 0)
                {
                    p += 5;
                    return Json{false};
                }
                if(s.compare(p, 4, "null") == 0)
                {
                    p += 4;
                    return Json{nullptr};
                }
                size_t b = p;
                if(s[p] == '-')
                    ++p;
                if(p == s.size())
                    fail();
                if(s[p] == '0')
                    ++p;
                else
                {
                    if(s[p] < '1' || s[p] > '9')
                        fail();
                    while(p < s.size() && s[p] >= '0' && s[p] <= '9')
                        ++p;
                }
                if(p < s.size() && s[p] == '.')
                {
                    ++p;
                    size_t d = p;
                    while(p < s.size() && s[p] >= '0' && s[p] <= '9')
                        ++p;
                    if(d == p)
                        fail();
                }
                if(p < s.size() && (s[p] == 'e' || s[p] == 'E'))
                {
                    ++p;
                    if(p < s.size() && (s[p] == '+' || s[p] == '-'))
                        ++p;
                    size_t d = p;
                    while(p < s.size() && s[p] >= '0' && s[p] <= '9')
                        ++p;
                    if(d == p)
                        fail();
                }
                try
                {
                    double number = 0.0;
                    auto const* first = s.data() + b;
                    auto const* last = s.data() + p;
                    auto const result = std::from_chars(first, last, number, std::chars_format::general);
                    if(result.ec != std::errc{} || result.ptr != last || !std::isfinite(number))
                        fail();
                    return Json{number};
                }
                catch(...)
                {
                    fail();
                }
            }

        public:
            explicit Parser(std::string const& input) : s(input)
            {
            }

            Json parse()
            {
                auto r = val();
                ws();
                if(p != s.size())
                    fail();
                return r;
            }
        };

        inline std::string sha256(std::vector<unsigned char> const& data)
        {
            // SHA-256, kept local to avoid imposing a crypto-library dependency.
            static constexpr uint32_t k[64]
                = {0x428a'2f98, 0x7137'4491, 0xb5c0'fbcf, 0xe9b5'dba5, 0x3956'c25b, 0x59f1'11f1, 0x923f'82a4,
                   0xab1c'5ed5, 0xd807'aa98, 0x1283'5b01, 0x2431'85be, 0x550c'7dc3, 0x72be'5d74, 0x80de'b1fe,
                   0x9bdc'06a7, 0xc19b'f174, 0xe49b'69c1, 0xefbe'4786, 0x0fc1'9dc6, 0x240c'a1cc, 0x2de9'2c6f,
                   0x4a74'84aa, 0x5cb0'a9dc, 0x76f9'88da, 0x983e'5152, 0xa831'c66d, 0xb003'27c8, 0xbf59'7fc7,
                   0xc6e0'0bf3, 0xd5a7'9147, 0x06ca'6351, 0x1429'2967, 0x27b7'0a85, 0x2e1b'2138, 0x4d2c'6dfc,
                   0x5338'0d13, 0x650a'7354, 0x766a'0abb, 0x81c2'c92e, 0x9272'2c85, 0xa2bf'e8a1, 0xa81a'664b,
                   0xc24b'8b70, 0xc76c'51a3, 0xd192'e819, 0xd699'0624, 0xf40e'3585, 0x106a'a070, 0x19a4'c116,
                   0x1e37'6c08, 0x2748'774c, 0x34b0'bcb5, 0x391c'0cb3, 0x4ed8'aa4a, 0x5b9c'ca4f, 0x682e'6ff3,
                   0x748f'82ee, 0x78a5'636f, 0x84c8'7814, 0x8cc7'0208, 0x90be'fffa, 0xa450'6ceb, 0xbef9'a3f7,
                   0xc671'78f2};
            std::vector<unsigned char> m = data;
            uint64_t bits = uint64_t(m.size()) * 8;
            m.push_back(0x80);
            while(m.size() % 64 != 56)
                m.push_back(0);
            for(int i = 7; i >= 0; --i)
                m.push_back(static_cast<unsigned char>(bits >> (i * 8)));
            uint32_t h[8]
                = {0x6a09'e667,
                   0xbb67'ae85,
                   0x3c6e'f372,
                   0xa54f'f53a,
                   0x510e'527f,
                   0x9b05'688c,
                   0x1f83'd9ab,
                   0x5be0'cd19};
            auto r = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
            for(size_t off = 0; off < m.size(); off += 64)
            {
                uint32_t w[64];
                for(int i = 0; i < 16; ++i)
                    w[i] = (uint32_t(m[off + 4 * i]) << 24) | (uint32_t(m[off + 4 * i + 1]) << 16)
                           | (uint32_t(m[off + 4 * i + 2]) << 8) | m[off + 4 * i + 3];
                for(int i = 16; i < 64; ++i)
                {
                    auto a = r(w[i - 15], 7) ^ r(w[i - 15], 18) ^ (w[i - 15] >> 3);
                    auto b = r(w[i - 2], 17) ^ r(w[i - 2], 19) ^ (w[i - 2] >> 10);
                    w[i] = w[i - 16] + a + w[i - 7] + b;
                }
                uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], z = h[7];
                for(int i = 0; i < 64; ++i)
                {
                    uint32_t s1 = r(e, 6) ^ r(e, 11) ^ r(e, 25), ch = (e & f) ^ (~e & g),
                             t1 = z + s1 + ch + k[i] + w[i], s0 = r(a, 2) ^ r(a, 13) ^ r(a, 22),
                             maj = (a & b) ^ (a & c) ^ (b & c), t2 = s0 + maj;
                    z = g;
                    g = f;
                    f = e;
                    e = d + t1;
                    d = c;
                    c = b;
                    b = a;
                    a = t1 + t2;
                }
                h[0] += a;
                h[1] += b;
                h[2] += c;
                h[3] += d;
                h[4] += e;
                h[5] += f;
                h[6] += g;
                h[7] += z;
            }
            static char const hex[] = "0123456789abcdef";
            std::string out;
            for(auto x : h)
                for(int j = 7; j >= 0; --j)
                {
                    auto q = static_cast<unsigned char>(x >> (j * 4));
                    out += hex[q & 15];
                }
            return out;
        }

        inline Json const& get(Json const& j, std::string const& key)
        {
            auto const& o = j.object();
            auto i = o.find(key);
            if(i == o.end())
                throw std::runtime_error("missing model metadata: " + key);
            return i->second;
        }

        inline void equal(Json const& j, std::string const& k, char const* v)
        {
            if(get(j, k).string() != v)
                throw std::runtime_error("unsupported model metadata: " + k);
        }

        inline void array(Json const& j, std::string const& k, std::initializer_list<double> v)
        {
            auto const& a = get(j, k).array();
            if(a.size() != v.size())
                throw std::runtime_error("unsupported model metadata: " + k);
            size_t i = 0;
            for(double x : v)
                if(a[i++].number() != x)
                    throw std::runtime_error("unsupported model metadata: " + k);
        }
    } // namespace detail

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
        std::string text((std::istreambuf_iterator<char>(mf)), {});
        auto m = detail::Parser(text).parse();
        detail::equal(m, "format", "alpakaNN-heat-closure-f32-v1");
        detail::equal(m, "architecture", "gated_silu_bias_free_v1");
        detail::equal(m, "dtype", "float32");
        detail::equal(m, "weight_layout", "row-major [in,out], little-endian float32; gate,up,down concatenated");
        detail::equal(m, "beta_mismatch_policy", "reject");
        detail::equal(m, "output", "alpha_min + (alpha_max - alpha_min) * sigmoid(z)");
        detail::equal(
            m,
            "material_formula",
            "base(x,y) * (1 + beta*u); inclusion if r2 < 0.12^2 => 0.02; else conductor if abs(y-0.5)<0.05 and "
            "0.45<x<0.65 => 4.0; else 0.5+0.4*H(sin(6*pi*y)), H(z)=1 iff z>=0");
        detail::equal(m, "weights_file", weights.filename().string().c_str());
        auto featureNode = detail::get(m, "feature_order");
        auto const& features = featureNode.array();
        if(features.size() != 3 || features[0].string() != "u" || features[1].string() != "x"
           || features[2].string() != "y")
            throw std::runtime_error("unsupported feature order");
        auto shapeNode = detail::get(m, "weight_shapes");
        auto const& shapes = shapeNode.array();
        if(shapes.size() != 3)
            throw std::runtime_error("unsupported weight shapes");
        for(size_t i = 0; i < 3; ++i)
        {
            auto const& a = shapes[i].array();
            std::array<double, 2> expected = i == 2 ? std::array<double, 2>{64, 1} : std::array<double, 2>{3, 64};
            if(a.size() != 2 || a[0].number() != expected[0] || a[1].number() != expected[1])
                throw std::runtime_error("unsupported weight shapes");
        }
        if(detail::get(m, "width").number() != 64)
            throw std::runtime_error("unsupported model width");
        detail::array(m, "coordinate_domain", {0, 1});
        detail::array(m, "temperature_domain", {0, 1});
        Model out;
        out.alphaMin = detail::get(m, "alpha_min").number();
        out.alphaMax = detail::get(m, "alpha_max").number();
        out.beta = detail::get(m, "beta").number();
        // The training and solver contract requires beta >= 0. The reference
        // material has base coefficient in [0.02, 4.0] and temperature u in
        // [0, 1]. Output bounds may be wider, but may not clip this envelope.
        auto temperatureFactor = 1.0 + out.beta;
        auto requiredMinimum = 0.02 * std::min(1.0, temperatureFactor);
        auto requiredMaximum = 4.0 * std::max(1.0, temperatureFactor);
        if(!std::isfinite(out.alphaMin) || !std::isfinite(out.alphaMax) || !std::isfinite(out.beta) || out.beta < 0.0
           || !(out.alphaMin > 0 && out.alphaMax > out.alphaMin) || out.alphaMin > requiredMinimum
           || out.alphaMax < requiredMaximum || !std::isfinite(beta) || beta != out.beta)
            throw std::runtime_error("invalid model bounds or beta mismatch");
        std::ifstream f(weights, std::ios::binary);
        if(!f)
            throw std::runtime_error("cannot open model weights: " + weights.string());
        std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), {});
        if(b.size() != 1792 || detail::sha256(b) != detail::get(m, "weights_sha256").string())
            throw std::runtime_error("model weight size or checksum mismatch");
        std::vector<float> w(448);
        for(size_t i = 0; i < w.size(); ++i)
        {
            uint32_t u = uint32_t(b[4 * i]) | uint32_t(b[4 * i + 1]) << 8 | uint32_t(b[4 * i + 2]) << 16
                         | uint32_t(b[4 * i + 3]) << 24;
            std::memcpy(&w[i], &u, 4);
            if(!std::isfinite(w[i]))
                throw std::runtime_error("non-finite model weight");
        }
        std::copy_n(w.begin(), 192, out.gate.begin());
        std::copy_n(w.begin() + 192, 192, out.up.begin());
        std::copy_n(w.begin() + 384, 64, out.down.begin());
        return out;
    }
} // namespace heatclosure
