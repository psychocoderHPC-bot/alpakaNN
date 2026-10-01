// SPDX-License-Identifier: MPL-2.0
#include "../src/ModelLoader.hpp"

#include <chrono>
#include <iostream>
#include <sstream>

namespace
{
    std::string read(std::filesystem::path const& p)
    {
        std::ifstream f(p);
        return {(std::istreambuf_iterator<char>(f)), {}};
    }

    void write(std::filesystem::path const& p, std::string const& s)
    {
        std::ofstream f(p);
        f << s;
    }

    void rejected(std::filesystem::path const& weights, std::string const& caseName)
    {
        bool failed = false;
        try
        {
            (void) heatclosure::loadModel(weights, 0.5);
        }
        catch(std::exception const&)
        {
            failed = true;
        }
        if(!failed)
            throw std::runtime_error("invalid model accepted: " + caseName);
    }

    void rejectedWithBeta(std::filesystem::path const& weights, double beta, std::string const& caseName)
    {
        bool failed = false;
        try
        {
            (void) heatclosure::loadModel(weights, beta);
        }
        catch(std::exception const&)
        {
            failed = true;
        }
        if(!failed)
            throw std::runtime_error("invalid model accepted: " + caseName);
    }

    std::string compact(std::string const& src)
    {
        std::string out;
        bool quoted = false, escape = false;
        for(char c : src)
        {
            if(quoted)
            {
                out += c;
                if(escape)
                    escape = false;
                else if(c == '\\')
                    escape = true;
                else if(c == '"')
                    quoted = false;
            }
            else if(c == '"')
            {
                quoted = true;
                out += c;
            }
            else if(c != ' ' && c != '\n' && c != '\r' && c != '\t')
                out += c;
        }
        return out;
    }
} // namespace
int main()
try
{
    auto source = std::filesystem::path(HEAT_CLOSURE_MODEL_DIR) / "weights.bin";
    auto valid = heatclosure::loadModel(source, 0.5);
    if(valid.alphaMin != 0.01 || valid.alphaMax != 6.0 || valid.gate.size() != 192 || valid.down.size() != 64)
        throw std::runtime_error("model metadata mismatch");
    bool mismatch = false;
    try
    {
        (void) heatclosure::loadModel(source, 0.6);
    }
    catch(std::runtime_error const&)
    {
        mismatch = true;
    }
    if(!mismatch)
        throw std::runtime_error("beta mismatch accepted");

    auto temp = std::filesystem::temp_directory_path()
                / ("heat-model-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(temp);
    auto weights = temp / "weights.bin", metadata = std::filesystem::path(weights.string() + ".metadata.json");
    std::filesystem::copy_file(source, weights);
    auto original = read(source.string() + ".metadata.json");
    auto run = [&](std::string const& text, std::string const& label)
    {
        write(metadata, text);
        rejected(weights, label);
    };
    write(metadata, compact(original));
    (void) heatclosure::loadModel(weights, 0.5); // legal whitespace variation
    auto escaped = original;
    auto escapedPos = escaped.find("alpakaNN-heat-closure-f32-v1");
    if(escapedPos == std::string::npos)
        throw std::runtime_error("Unicode escape test target missing");
    escaped.replace(escapedPos, 1, "\\u0061");
    write(metadata, escaped);
    (void) heatclosure::loadModel(weights, 0.5); // JSON Unicode escape decodes to the same manifest value
    auto unicode = heatclosure::detail::Parser(R"({"text":"\u00e9\uD83D\uDE00"})").parse();
    if(heatclosure::detail::get(unicode, "text").string() != "\xc3\xa9\xf0\x9f\x98\x80")
        throw std::runtime_error("JSON BMP/surrogate Unicode decoding mismatch");
    auto mutation = [&](std::string const& before, std::string const& after, std::string const& label)
    {
        auto altered = original;
        auto pos = altered.find(before);
        if(pos == std::string::npos)
            throw std::runtime_error("test mutation target missing: " + label);
        altered.replace(pos, before.size(), after);
        run(altered, label);
    };
    mutation("\"format\": \"alpakaNN-heat-closure-f32-v1\"", "\"format\": \"other\"", "format");
    mutation("\"architecture\": \"gated_silu_bias_free_v1\"", "\"architecture\": \"other\"", "architecture");
    mutation("\"dtype\": \"float32\"", "\"dtype\": \"float64\"", "dtype");
    mutation(
        "\"feature_order\": [\n    \"u\",\n    \"x\",\n    \"y\"",
        "\"feature_order\": [\n    \"x\",\n    \"u\",\n    \"y\"",
        "feature order");
    mutation(
        "\"weight_shapes\": [",
        "\"weight_shapes\": [[3,63],[3,64],[64,1]], \"ignored_shapes\": [",
        "weight shapes");
    mutation(
        "\"weight_layout\": \"row-major [in,out], little-endian float32; gate,up,down concatenated\"",
        "\"weight_layout\": \"column major\"",
        "layout");
    mutation(
        "\"output\": \"alpha_min + (alpha_max - alpha_min) * sigmoid(z)\"",
        "\"output\": \"z\"",
        "output equation");
    mutation(
        "\"material_formula\": \"base(x,y) * (1 + beta*u); inclusion if r2 < 0.12^2 => 0.02; else conductor if "
        "abs(y-0.5)<0.05 and 0.45<x<0.65 => 4.0; else 0.5+0.4*H(sin(6*pi*y)), H(z)=1 iff z>=0\"",
        "\"material_formula\": \"constant\"",
        "material formula");
    mutation(
        "\"coordinate_domain\": [\n    0.0,\n    1.0\n  ]",
        "\"coordinate_domain\": [\n    -1.0,\n    1.0\n  ]",
        "coordinate domain");
    mutation(
        "\"temperature_domain\": [\n    0.0,\n    1.0\n  ]",
        "\"temperature_domain\": [\n    0.0,\n    2.0\n  ]",
        "temperature domain");
    mutation("\"beta_mismatch_policy\": \"reject\"", "\"beta_mismatch_policy\": \"ignore\"", "beta policy");
    mutation("\"beta\": 0.5", "\"beta\": 0.6", "beta");
    auto negativeBeta = original;
    auto betaPos = negativeBeta.find("\"beta\": 0.5");
    if(betaPos == std::string::npos)
        throw std::runtime_error("test mutation target missing: negative beta");
    negativeBeta.replace(betaPos, std::string("\"beta\": 0.5").size(), "\"beta\": -0.5");
    write(metadata, negativeBeta);
    rejectedWithBeta(weights, -0.5, "negative beta");
    mutation("\"alpha_min\": 0.01", "\"alpha_min\": -0.01", "alpha minimum");
    mutation("\"alpha_max\": 6.0", "\"alpha_max\": 0.0", "alpha maximum");
    mutation("\"alpha_max\": 6.0", "\"alpha_max\": 5.0", "alpha bounds clip true coefficient range");
    mutation("\"width\": 64", "\"width\": 32", "width");
    mutation("\"weights_file\": \"weights.bin\"", "\"weights_file\": \"different.bin\"", "declared weights name");
    mutation(
        "\"weights_sha256\": \"f075cffd50324e17434713d5f9479fb02375eefc712565e2dbfcfd54ac82bf01\"",
        "\"weights_sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"",
        "checksum metadata");
    run(original.substr(0, original.size() / 2), "truncated JSON");
    auto bytes = read(weights);
    bytes[0] ^= 1;
    write(weights, bytes);
    write(metadata, original);
    rejected(weights, "weights checksum mismatch");
    std::filesystem::remove_all(temp);
    std::cout << "model loader checks passed\n";
}
catch(std::exception const& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
