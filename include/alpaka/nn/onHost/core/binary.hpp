/*
 * SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace alpaka::nn::onHost
{
    /** @brief Description of one tensor inside a flat model artifact.
     *
     * Mirrors ``tensors[]`` of the neural-bunch model format v1. ``offsetBytes`` is the byte
     * offset of the tensor payload inside the ``.bin`` file; ``count`` is the number of
     * elements and ``shape`` is informational (row-major) metadata.
     */
    struct TensorSpec
    {
        std::string name;
        std::size_t offsetBytes;
        std::size_t count;
        std::vector<std::size_t> shape;
    };

    namespace detail
    {
        class Sha256
        {
        public:
            Sha256()
            {
                reset();
            }

            void update(std::uint8_t const* data, std::size_t length)
            {
                totalBytes += static_cast<std::uint64_t>(length);
                while(length != 0u)
                {
                    auto const take = std::min(length, buffer.size() - bufferLength);
                    std::memcpy(buffer.data() + bufferLength, data, take);
                    bufferLength += take;
                    data += take;
                    length -= take;
                    if(bufferLength == buffer.size())
                    {
                        transform(buffer.data());
                        bufferLength = 0u;
                    }
                }
            }

            std::string hexDigest()
            {
                auto const bitLength = totalBytes * 8u;
                std::uint8_t padding[72]{};
                padding[0] = 0x80u;
                auto const padLength = (bufferLength < 56u) ? (56u - bufferLength) : (120u - bufferLength);
                update(padding, padLength);
                std::uint8_t lengthBytes[8]{};
                for(std::size_t i = 0u; i < 8u; ++i)
                    lengthBytes[7u - i] = static_cast<std::uint8_t>(bitLength >> (8u * i));
                update(lengthBytes, 8u);

                constexpr char digits[] = "0123456789abcdef";
                std::string result;
                result.reserve(64u);
                for(auto const word : state)
                {
                    for(int shift = 28; shift >= 0; shift -= 4)
                        result.push_back(digits[(word >> shift) & 0xfu]);
                }
                return result;
            }

        private:
            static constexpr std::array<std::uint32_t, 64> roundConstants{
                0x428a'2f98u, 0x7137'4491u, 0xb5c0'fbcfu, 0xe9b5'dba5u, 0x3956'c25bu, 0x59f1'11f1u, 0x923f'82a4u,
                0xab1c'5ed5u, 0xd807'aa98u, 0x1283'5b01u, 0x2431'85beu, 0x550c'7dc3u, 0x72be'5d74u, 0x80de'b1feu,
                0x9bdc'06a7u, 0xc19b'f174u, 0xe49b'69c1u, 0xefbe'4786u, 0x0fc1'9dc6u, 0x240c'a1ccu, 0x2de9'2c6fu,
                0x4a74'84aau, 0x5cb0'a9dcu, 0x76f9'88dau, 0x983e'5152u, 0xa831'c66du, 0xb003'27c8u, 0xbf59'7fc7u,
                0xc6e0'0bf3u, 0xd5a7'9147u, 0x06ca'6351u, 0x1429'2967u, 0x27b7'0a85u, 0x2e1b'2138u, 0x4d2c'6dfcu,
                0x5338'0d13u, 0x650a'7354u, 0x766a'0abbu, 0x81c2'c92eu, 0x9272'2c85u, 0xa2bf'e8a1u, 0xa81a'664bu,
                0xc24b'8b70u, 0xc76c'51a3u, 0xd192'e819u, 0xd699'0624u, 0xf40e'3585u, 0x106a'a070u, 0x19a4'c116u,
                0x1e37'6c08u, 0x2748'774cu, 0x34b0'bcb5u, 0x391c'0cb3u, 0x4ed8'aa4au, 0x5b9c'ca4fu, 0x682e'6ff3u,
                0x748f'82eeu, 0x78a5'636fu, 0x84c8'7814u, 0x8cc7'0208u, 0x90be'fffau, 0xa450'6cebu, 0xbef9'a3f7u,
                0xc671'78f2u};

            std::array<std::uint32_t, 8> state{};
            std::array<std::uint8_t, 64> buffer{};
            std::uint64_t totalBytes{};
            std::size_t bufferLength{};

            static std::uint32_t rotateRight(std::uint32_t value, unsigned count)
            {
                return (value >> count) | (value << (32u - count));
            }

            void reset()
            {
                state
                    = {0x6a09'e667u,
                       0xbb67'ae85u,
                       0x3c6e'f372u,
                       0xa54f'f53au,
                       0x510e'527fu,
                       0x9b05'688cu,
                       0x1f83'd9abu,
                       0x5be0'cd19u};
                buffer.fill(0u);
                totalBytes = 0u;
                bufferLength = 0u;
            }

            void transform(std::uint8_t const* chunk)
            {
                std::array<std::uint32_t, 64> words{};
                for(std::size_t i = 0u; i < 16u; ++i)
                {
                    words[i] = (static_cast<std::uint32_t>(chunk[4u * i]) << 24)
                               | (static_cast<std::uint32_t>(chunk[4u * i + 1u]) << 16)
                               | (static_cast<std::uint32_t>(chunk[4u * i + 2u]) << 8)
                               | static_cast<std::uint32_t>(chunk[4u * i + 3u]);
                }
                for(std::size_t i = 16u; i < 64u; ++i)
                {
                    auto const s0
                        = rotateRight(words[i - 15u], 7u) ^ rotateRight(words[i - 15u], 18u) ^ (words[i - 15u] >> 3u);
                    auto const s1
                        = rotateRight(words[i - 2u], 17u) ^ rotateRight(words[i - 2u], 19u) ^ (words[i - 2u] >> 10u);
                    words[i] = words[i - 16u] + s0 + words[i - 7u] + s1;
                }

                auto a = state[0];
                auto b = state[1];
                auto c = state[2];
                auto d = state[3];
                auto e = state[4];
                auto f = state[5];
                auto g = state[6];
                auto h = state[7];

                for(std::size_t i = 0u; i < 64u; ++i)
                {
                    auto const s1 = rotateRight(e, 6u) ^ rotateRight(e, 11u) ^ rotateRight(e, 25u);
                    auto const choose = (e & f) ^ (~e & g);
                    auto const temp1 = h + s1 + choose + roundConstants[i] + words[i];
                    auto const s0 = rotateRight(a, 2u) ^ rotateRight(a, 13u) ^ rotateRight(a, 22u);
                    auto const majority = (a & b) ^ (a & c) ^ (b & c);
                    auto const temp2 = s0 + majority;

                    h = g;
                    g = f;
                    f = e;
                    e = d + temp1;
                    d = c;
                    c = b;
                    b = a;
                    a = temp1 + temp2;
                }

                state[0] += a;
                state[1] += b;
                state[2] += c;
                state[3] += d;
                state[4] += e;
                state[5] += f;
                state[6] += g;
                state[7] += h;
            }
        };

        inline std::string toLower(std::string value)
        {
            for(auto& character : value)
            {
                if(character >= 'A' && character <= 'Z')
                    character = static_cast<char>(character - 'A' + 'a');
            }
            return value;
        }
    } // namespace detail

    /** @brief Compute the SHA-256 digest of a file and return it as 64 lowercase hex characters.
     *
     * Throws ``std::runtime_error`` if the file cannot be opened or read.
     */
    inline std::string sha256File(std::string const& path)
    {
        std::ifstream file(path, std::ios::binary);
        if(!file)
            throw std::runtime_error{"sha256File: cannot open '" + path + "'."};

        detail::Sha256 sha;
        std::array<std::uint8_t, 1u << 16u> chunk{};
        while(file)
        {
            file.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(chunk.size()));
            auto const read = static_cast<std::size_t>(file.gcount());
            if(read != 0u)
                sha.update(chunk.data(), read);
        }
        if(file.bad())
            throw std::runtime_error{"sha256File: read error in '" + path + "'."};
        return sha.hexDigest();
    }

    /** @brief Load one tensor from a flat little-endian model artifact into a device buffer.
     *
     * The file is stat'ed and every bound is validated before any device memory is allocated:
     *
     * - ``spec.offsetBytes`` must be 64-byte aligned,
     * - ``count * sizeof(T)`` must not overflow ``std::size_t``,
     * - the byte range ``[offsetBytes, offsetBytes + count * sizeof(T))`` must fit in the file,
     * - when ``spec.shape`` is non-empty its element product must equal ``spec.count``.
     *
     * If @p expectedSha256 is non-empty the whole file digest is compared (case-insensitive)
     * against it before the payload is read and before the device buffer is created. Passing an
     * empty string skips the digest check, which is useful when one loader call verifies a whole
     * artifact first.
     *
     * ``T`` is the tensor dtype: the format only defines ``float`` (float32) and ``double``
     * (float64), so any other type is rejected at compile time. The requested byte range is read
     * into a host buffer and copied to an allocation on ``queue.getDevice()`` with an explicit
     * ``alpaka::onHost::memcpy`` on @p queue. The same code path is used for Host, CUDA, HIP and
     * oneAPI.
     *
     * @return a flat one-dimensional buffer with ``spec.count`` elements. The buffer owns its
     *         device memory; the host staging buffer is kept alive until the enqueued copy has
     *         completed via ``keepAlive(queue)``. The caller must synchronize the queue before
     *         reading the result elsewhere.
     */
    template<typename T_Type>
    auto loadBinaryTensor(
        auto& queue,
        std::string const& path,
        TensorSpec const& spec,
        std::string const& expectedSha256 = std::string{})
    {
        static_assert(
            std::same_as<T_Type, float> || std::same_as<T_Type, double>,
            "alpaka::nn::onHost::loadBinaryTensor supports only float32 and float64 tensors.");

        if(spec.count == 0u)
            throw std::invalid_argument{"loadBinaryTensor: tensor '" + spec.name + "' has zero elements."};

        if(!spec.shape.empty())
        {
            std::size_t product = 1u;
            for(auto const extent : spec.shape)
            {
                if(extent != 0u && product > std::numeric_limits<std::size_t>::max() / extent)
                    throw std::invalid_argument{
                        "loadBinaryTensor: shape of tensor '" + spec.name + "' overflows std::size_t."};
                product *= extent;
            }
            if(product != spec.count)
                throw std::invalid_argument{
                    "loadBinaryTensor: shape product " + std::to_string(product) + " of tensor '" + spec.name
                    + "' does not match count " + std::to_string(spec.count) + "."};
        }

        std::error_code error;
        auto const fileSize = std::filesystem::file_size(path, error);
        if(error)
            throw std::runtime_error{"loadBinaryTensor: cannot stat '" + path + "': " + error.message()};

        if(spec.offsetBytes % 64u != 0u)
            throw std::invalid_argument{
                "loadBinaryTensor: offset " + std::to_string(spec.offsetBytes) + " of tensor '" + spec.name
                + "' is not 64-byte aligned."};

        constexpr auto elementSize = sizeof(T_Type);
        if(spec.count > std::numeric_limits<std::size_t>::max() / elementSize)
            throw std::invalid_argument{
                "loadBinaryTensor: byte size of tensor '" + spec.name + "' overflows std::size_t."};
        auto const byteCount = spec.count * elementSize;

        if(spec.offsetBytes > fileSize || byteCount > fileSize - spec.offsetBytes)
            throw std::invalid_argument{
                "loadBinaryTensor: tensor '" + spec.name + "' (" + std::to_string(byteCount) + " bytes at offset "
                + std::to_string(spec.offsetBytes) + ") exceeds file size " + std::to_string(fileSize) + "."};

        if(!expectedSha256.empty())
        {
            auto const actual = sha256File(path);
            if(actual != detail::toLower(expectedSha256))
                throw std::runtime_error{
                    "loadBinaryTensor: SHA-256 mismatch for '" + path + "': expected " + expectedSha256 + ", got "
                    + actual + "."};
        }

        // Only after every bound has been validated do we touch the filesystem payload and allocate.
        auto host = alpaka::onHost::allocHost<T_Type>(alpaka::Vec{spec.count});
        std::ifstream file(path, std::ios::binary);
        if(!file)
            throw std::runtime_error{"loadBinaryTensor: cannot open '" + path + "'."};
        file.seekg(static_cast<std::streamoff>(spec.offsetBytes));
        file.read(reinterpret_cast<char*>(alpaka::onHost::data(host)), static_cast<std::streamsize>(byteCount));
        if(!file)
            throw std::runtime_error{"loadBinaryTensor: short read for tensor '" + spec.name + "' in '" + path + "'."};

        auto deviceBuffer = alpaka::onHost::alloc<T_Type>(queue.getDevice(), alpaka::Vec{spec.count});
        alpaka::onHost::memcpy(queue, deviceBuffer, host);
        host.keepAlive(queue);
        return deviceBuffer;
    }
} // namespace alpaka::nn::onHost
