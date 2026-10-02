// SPDX-License-Identifier: MPL-2.0
#include "ModelFixture.hpp"

#include <chrono>
#include <clocale>
#include <iostream>
#include <sstream>

namespace
{
    std::string read(std::filesystem::path const& p)
    {
        std::ifstream f(p);
        return {(std::istreambuf_iterator<char>(f)), {}};
    }

    std::vector<unsigned char> readBytes(std::filesystem::path const& p)
    {
        std::ifstream f(p, std::ios::binary);
        if(!f)
            throw std::runtime_error("cannot read " + p.string());
        return {(std::istreambuf_iterator<char>(f)), {}};
    }

    void write(std::filesystem::path const& p, std::string const& s)
    {
        std::ofstream f(p);
        f << s;
    }

    /** Reject an invalid model and, when supplied, assert the failure-reason class. */
    void rejected(
        std::filesystem::path const& weights,
        std::string const& caseName,
        std::string const& expectedReason = {})
    {
        bool failed = false;
        std::string reason;
        try
        {
            (void) heatclosure::loadModel(weights, 0.5);
        }
        catch(std::exception const& e)
        {
            failed = true;
            reason = e.what();
        }
        if(!failed)
            throw std::runtime_error("invalid model accepted: " + caseName);
        if(!expectedReason.empty() && reason.find(expectedReason) == std::string::npos)
            throw std::runtime_error(
                "wrong rejection reason for " + caseName + ": got '" + reason + "', expected fragment '"
                + expectedReason + "'");
    }

    void rejectedWithBeta(
        std::filesystem::path const& weights,
        double beta,
        std::string const& caseName,
        std::string const& expectedReason = {})
    {
        bool failed = false;
        std::string reason;
        try
        {
            (void) heatclosure::loadModel(weights, beta);
        }
        catch(std::exception const& e)
        {
            failed = true;
            reason = e.what();
        }
        if(!failed)
            throw std::runtime_error("invalid model accepted: " + caseName);
        if(!expectedReason.empty() && reason.find(expectedReason) == std::string::npos)
            throw std::runtime_error(
                "wrong rejection reason for " + caseName + ": got '" + reason + "', expected fragment '"
                + expectedReason + "'");
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
    // The trained checkpoint is intentionally not committed. Build a deterministic
    // valid fixture here instead of requiring models/heat_closure/weights.bin.
    auto temp = std::filesystem::temp_directory_path()
                / ("heat-model-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(temp);
    auto weights = temp / "weights.bin", metadata = std::filesystem::path(weights.string() + ".metadata.json");
    auto source = heatclosure::test::writeFixture(temp);
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

    auto original = read(source.string() + ".metadata.json");
    // Derive the declared checksum from the weights file actually under test rather
    // than hardcoding a revision-specific digest. This keeps the mutation-rejection
    // test meaningful across model updates while still requiring a wrong checksum
    // to be rejected (the tamper case below flips a byte with matching metadata).
    auto const actualSha = heatclosure::detail::sha256(readBytes(source));
    auto const expectedChecksumLiteral = std::string("\"weights_sha256\": \"") + actualSha + "\"";
    if(original.find(expectedChecksumLiteral) == std::string::npos)
        throw std::runtime_error("metadata does not declare the actual weights sha256: " + actualSha);
    auto run = [&](std::string const& text, std::string const& label, std::string const& expectedReason = {})
    {
        write(metadata, text);
        rejected(weights, label, expectedReason);
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
    auto rawUnicode
        = heatclosure::detail::Parser(std::string("{\"text\":\"") + "\xc3\xa9\xf0\x9f\x98\x80" + "\"}").parse();
    if(heatclosure::detail::get(rawUnicode, "text").string() != "\xc3\xa9\xf0\x9f\x98\x80")
        throw std::runtime_error("valid raw UTF-8 JSON string mismatch");
    for(auto const& invalidUtf8 :
        {std::string("\xc0\xaf", 2), // overlong
         std::string("\xed\xa0\x80", 3), // encoded surrogate
         std::string("\xe2\x82", 2), // truncated
         std::string(
             "\xe2"
             "A"
             "\xa1",
             3), // invalid continuation
         std::string("\xf4\x90\x80\x80", 4)}) // above U+10FFFF
    {
        bool rejectedUtf8 = false;
        try
        {
            (void) heatclosure::detail::Parser(std::string("{\"text\":\"") + invalidUtf8 + "\"}").parse();
        }
        catch(std::runtime_error const&)
        {
            rejectedUtf8 = true;
        }
        if(!rejectedUtf8)
            throw std::runtime_error("invalid raw UTF-8 accepted");
    }
    bool duplicateRejected = false;
    try
    {
        (void) heatclosure::detail::Parser(R"({"x":1,"x":2})").parse();
    }
    catch(std::runtime_error const&)
    {
        duplicateRejected = true;
    }
    if(!duplicateRejected)
        throw std::runtime_error("duplicate JSON object key accepted");
    auto const* oldLocale = std::setlocale(LC_NUMERIC, nullptr);
    auto savedLocale = oldLocale ? std::string(oldLocale) : std::string("C");
    // Exercise under a comma-decimal locale when installed; parsing must remain JSON-locale invariant.
    char const* commaLocale = nullptr;
    for(auto candidate : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8"})
        if(std::setlocale(LC_NUMERIC, candidate))
        {
            commaLocale = candidate;
            break;
        }
    if(commaLocale == nullptr)
    {
        // The container may only ship the C locale. This is recorded explicitly and
        // the parser is still exercised for locale invariance under C, rather than
        // silently passing without checking anything.
        std::cout << "locale test: no comma-decimal locale installed; verified locale invariance under C only\n";
        std::setlocale(LC_NUMERIC, savedLocale.c_str());
    }
    auto const decimalLocale = commaLocale != nullptr ? std::string(commaLocale) : savedLocale;
    if(!std::setlocale(LC_NUMERIC, decimalLocale.c_str()))
        throw std::runtime_error("failed to select a decimal locale for the locale-invariance check");
    auto localeNumber = heatclosure::detail::Parser(R"({"n":1.5})").parse();
    if(heatclosure::detail::get(localeNumber, "n").number() != 1.5)
        throw std::runtime_error("JSON decimal parsing depends on locale");
    (void) std::setlocale(LC_NUMERIC, savedLocale.c_str());
    auto mutation = [&](std::string const& before,
                        std::string const& after,
                        std::string const& label,
                        std::string const& expectedReason = {})
    {
        auto altered = original;
        auto pos = altered.find(before);
        if(pos == std::string::npos)
            throw std::runtime_error("test mutation target missing: " + label);
        altered.replace(pos, before.size(), after);
        run(altered, label, expectedReason);
    };
    mutation(
        "\"format\": \"alpakaNN-heat-closure-f32-v1\"",
        "\"format\": \"other\"",
        "format",
        "unsupported model metadata: format");
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
    mutation(
        "\"weights_file\": \"weights.bin\"",
        "\"weights_file\": \"different.bin\"",
        "declared weights name",
        "unsupported model metadata: weights_file");
    mutation(
        expectedChecksumLiteral,
        "\"weights_sha256\": \"0000000000000000000000000000000000000000000000000000000000000000\"",
        "checksum metadata",
        "checksum mismatch");
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
