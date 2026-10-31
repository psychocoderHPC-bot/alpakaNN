/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "test.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/nn.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using TestApis = alpaka::nn::test::TestApis;

namespace
{
    std::filesystem::path makeTempPath(std::string const& tag)
    {
        static std::atomic<std::uint64_t> counter{0u};
        auto const name = "alpakann_binary_" + tag + "_" + std::to_string(counter.fetch_add(1u)) + ".bin";
        return std::filesystem::temp_directory_path() / name;
    }

    class TempFile
    {
    public:
        explicit TempFile(std::string const& tag) : path{makeTempPath(tag)}
        {
        }

        ~TempFile()
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }

        TempFile(TempFile const&) = delete;
        TempFile& operator=(TempFile const&) = delete;

        void write(std::vector<std::uint8_t> const& bytes) const
        {
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            REQUIRE(file);
            file.write(reinterpret_cast<char const*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            REQUIRE(file);
        }

        [[nodiscard]] std::string string() const
        {
            return path.string();
        }

    private:
        std::filesystem::path path;
    };

    std::vector<std::uint8_t> toBytes(std::vector<float> const& values)
    {
        std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }

    std::vector<std::uint8_t> toBytes(std::vector<double> const& values)
    {
        std::vector<std::uint8_t> bytes(values.size() * sizeof(double));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }

    template<typename T_Type>
    void runRoundtrip(auto& queue)
    {
        constexpr std::size_t count = 6u;
        std::vector<T_Type> values(count);
        for(std::size_t i = 0u; i < count; ++i)
            values[i] = static_cast<T_Type>(i) * static_cast<T_Type>(0.5) - static_cast<T_Type>(1.25);

        auto bytes = toBytes(values);
        std::vector<std::uint8_t> fileBytes(64u, 0xabu); // 64-byte aligned payload with a poison prefix
        fileBytes.insert(fileBytes.end(), bytes.begin(), bytes.end());

        TempFile file{"roundtrip"};
        file.write(fileBytes);

        alpaka::nn::onHost::TensorSpec const spec{
            .name = "t",
            .offsetBytes = 64u,
            .count = count,
            .shape = {2u, 3u}};

        auto const expectedSha = alpaka::nn::onHost::sha256File(file.string());
        auto deviceBuffer = alpaka::nn::onHost::loadBinaryTensor<T_Type>(queue, file.string(), spec, expectedSha);

        auto hostResult = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{count});
        alpaka::onHost::memcpy(queue, hostResult, deviceBuffer);
        alpaka::onHost::wait(queue);

        for(std::size_t i = 0u; i < count; ++i)
        {
            auto const expected = values[i];
            alpaka::nn::test::checkValue(
                hostResult[alpaka::Vec{i}],
                expected,
                std::is_same_v<T_Type, float> ? 1.0e-6 : 1.0e-14,
                0.0);
        }
    }

    alpaka::nn::onHost::TensorSpec makeSpec(std::size_t offset, std::size_t count, std::vector<std::size_t> shape = {})
    {
        return alpaka::nn::onHost::TensorSpec{
            .name = "spec",
            .offsetBytes = offset,
            .count = count,
            .shape = std::move(shape)};
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("sha256File matches known vectors", "[core][binary]", TestApis)
{
    TempFile emptyFile{"sha_empty"};
    emptyFile.write({});
    CHECK(
        alpaka::nn::onHost::sha256File(emptyFile.string())
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    TempFile abcFile{"sha_abc"};
    abcFile.write({'a', 'b', 'c'});
    CHECK(
        alpaka::nn::onHost::sha256File(abcFile.string())
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // Long input exercises the 64-byte block padding edge cases.
    std::vector<std::uint8_t> longInput(1000u);
    for(std::size_t i = 0u; i < longInput.size(); ++i)
        longInput[i] = static_cast<std::uint8_t>(i & 0xffu);
    TempFile longFile{"sha_long"};
    longFile.write(longInput);
    CHECK(alpaka::nn::onHost::sha256File(longFile.string()).size() == 64u);
}

TEMPLATE_LIST_TEST_CASE("binary tensor loader roundtrip", "[core][binary]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    runRoundtrip<float>(queue);
    if constexpr(alpaka::nn::test::supportsFp64(device))
        runRoundtrip<double>(queue);
}

TEMPLATE_LIST_TEST_CASE("binary tensor loader rejects invalid layouts", "[core][binary]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(alpaka::onHost::makeDeviceSpec(cfg));
    if(!selector.isAvailable())
    {
        SUCCEED("No device available");
        return;
    }
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    std::vector<float> const values(16u, 1.0f);
    auto const payload = toBytes(values);

    TempFile file{"reject"};
    file.write(payload);

    SECTION("offset is not 64-byte aligned")
    {
        CHECK_THROWS_AS(
            alpaka::nn::onHost::loadBinaryTensor<float>(queue, file.string(), makeSpec(8u, 4u)),
            std::invalid_argument);
    }

    SECTION("shape product does not match count")
    {
        CHECK_THROWS_AS(
            alpaka::nn::onHost::loadBinaryTensor<float>(queue, file.string(), makeSpec(0u, 6u, {2u, 4u})),
            std::invalid_argument);
    }

    SECTION("truncated payload")
    {
        CHECK_THROWS_AS(
            alpaka::nn::onHost::loadBinaryTensor<float>(queue, file.string(), makeSpec(0u, 1000u)),
            std::invalid_argument);
    }

    SECTION("count times element size overflows")
    {
        auto const huge = std::numeric_limits<std::size_t>::max() / sizeof(float) + 1u;
        CHECK_THROWS_AS(
            alpaka::nn::onHost::loadBinaryTensor<float>(queue, file.string(), makeSpec(0u, huge)),
            std::invalid_argument);
    }

    SECTION("zero-element tensor")
    {
        CHECK_THROWS_AS(
            alpaka::nn::onHost::loadBinaryTensor<float>(queue, file.string(), makeSpec(0u, 0u)),
            std::invalid_argument);
    }

    SECTION("corrupt expected sha256")
    {
        auto const corrupt = std::string(64u, '0');
        CHECK_THROWS_AS(
            alpaka::nn::onHost::loadBinaryTensor<float>(queue, file.string(), makeSpec(0u, 4u), corrupt),
            std::runtime_error);
    }

    SECTION("missing file")
    {
        auto const missing = (std::filesystem::temp_directory_path() / "alpakann_does_not_exist.bin").string();
        CHECK_THROWS_AS(
            alpaka::nn::onHost::loadBinaryTensor<float>(queue, missing, makeSpec(0u, 4u)),
            std::runtime_error);
    }
}
