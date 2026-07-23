#include "checksum.hpp"

#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace leann::detail {
namespace {

constexpr std::array<std::uint32_t, 64> round_constants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

constexpr std::uint32_t choose(std::uint32_t x, std::uint32_t y,
                               std::uint32_t z) {
    return (x & y) ^ (~x & z);
}

constexpr std::uint32_t majority(std::uint32_t x, std::uint32_t y,
                                 std::uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

constexpr std::uint32_t big_sigma0(std::uint32_t value) {
    return std::rotr(value, 2) ^ std::rotr(value, 13) ^ std::rotr(value, 22);
}

constexpr std::uint32_t big_sigma1(std::uint32_t value) {
    return std::rotr(value, 6) ^ std::rotr(value, 11) ^ std::rotr(value, 25);
}

constexpr std::uint32_t small_sigma0(std::uint32_t value) {
    return std::rotr(value, 7) ^ std::rotr(value, 18) ^ (value >> 3U);
}

constexpr std::uint32_t small_sigma1(std::uint32_t value) {
    return std::rotr(value, 17) ^ std::rotr(value, 19) ^ (value >> 10U);
}

constexpr std::array<std::uint32_t, 256> make_crc32c_table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t entry = 0; entry < table.size(); ++entry) {
        std::uint32_t value = entry;
        for (int bit = 0; bit < 8; ++bit) {
            value = (value >> 1U) ^
                    (0x82f63b78U & (0U - (value & 1U)));
        }
        table[entry] = value;
    }
    return table;
}

constexpr auto crc32c_table = make_crc32c_table();

} // namespace

Sha256::Sha256()
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

void Sha256::transform(const std::uint8_t * block) {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t word = 0; word < 16; ++word) {
        const std::size_t offset = word * 4U;
        schedule[word] =
            (static_cast<std::uint32_t>(block[offset]) << 24U) |
            (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
            (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
            static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t word = 16; word < schedule.size(); ++word) {
        schedule[word] =
            small_sigma1(schedule[word - 2U]) + schedule[word - 7U] +
            small_sigma0(schedule[word - 15U]) + schedule[word - 16U];
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];

    for (std::size_t round = 0; round < schedule.size(); ++round) {
        const std::uint32_t first =
            h + big_sigma1(e) + choose(e, f, g) + round_constants[round] +
            schedule[round];
        const std::uint32_t second =
            big_sigma0(a) + majority(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + first;
        d = c;
        c = b;
        b = a;
        a = first + second;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(std::span<const std::uint8_t> bytes) {
    if (finished_) {
        throw std::logic_error("SHA-256 update after finish");
    }
    if (bytes.size() >
        std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
        throw std::overflow_error("SHA-256 input is too large");
    }
    total_bytes_ += static_cast<std::uint64_t>(bytes.size());
    if (bytes.empty()) {
        return;
    }

    std::size_t consumed = 0;
    if (buffered_ != 0) {
        const std::size_t copied =
            std::min(buffer_.size() - buffered_, bytes.size());
        std::copy_n(bytes.data(), copied, buffer_.data() + buffered_);
        buffered_ += copied;
        consumed += copied;
        if (buffered_ == buffer_.size()) {
            transform(buffer_.data());
            buffered_ = 0;
        }
    }
    while (bytes.size() - consumed >= buffer_.size()) {
        transform(bytes.data() + consumed);
        consumed += buffer_.size();
    }
    const std::size_t remaining = bytes.size() - consumed;
    if (remaining != 0) {
        std::copy_n(bytes.data() + consumed, remaining, buffer_.data());
        buffered_ = remaining;
    }
}

void Sha256::update(std::string_view text) {
    update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(text.data()), text.size()));
}

Sha256Digest Sha256::finish() {
    if (finished_) {
        throw std::logic_error("SHA-256 finish called twice");
    }
    if (total_bytes_ > std::numeric_limits<std::uint64_t>::max() / 8U) {
        throw std::overflow_error("SHA-256 bit length is too large");
    }
    const std::uint64_t total_bits = total_bytes_ * 8U;
    buffer_[buffered_++] = 0x80U;
    if (buffered_ > 56U) {
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
                  buffer_.end(), 0U);
        transform(buffer_.data());
        buffered_ = 0;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
              buffer_.begin() + 56, 0U);
    for (std::size_t byte = 0; byte < 8; ++byte) {
        buffer_[56U + byte] =
            static_cast<std::uint8_t>(total_bits >> (56U - 8U * byte));
    }
    transform(buffer_.data());
    finished_ = true;

    Sha256Digest digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
        digest[word * 4U] =
            static_cast<std::uint8_t>(state_[word] >> 24U);
        digest[word * 4U + 1U] =
            static_cast<std::uint8_t>(state_[word] >> 16U);
        digest[word * 4U + 2U] =
            static_cast<std::uint8_t>(state_[word] >> 8U);
        digest[word * 4U + 3U] =
            static_cast<std::uint8_t>(state_[word]);
    }
    return digest;
}

Sha256Digest sha256(std::string_view text) {
    Sha256 hasher;
    hasher.update(text);
    return hasher.finish();
}

std::uint64_t stream_size(std::istream & input,
                          std::string_view artifact_name) {
    input.clear();
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot determine " +
                                 std::string(artifact_name) + " size");
    }
    input.seekg(0, std::ios::beg);
    if (!input) {
        throw std::runtime_error("cannot rewind " +
                                 std::string(artifact_name));
    }
    return static_cast<std::uint64_t>(end);
}

Sha256Digest sha256_stream_prefix(std::istream & input, std::uint64_t bytes,
                                  std::string_view artifact_name) {
    input.clear();
    input.seekg(0, std::ios::beg);
    if (!input) {
        throw std::runtime_error("cannot rewind " +
                                 std::string(artifact_name) +
                                 " for SHA-256");
    }
    constexpr std::size_t chunk_size = 1024U * 1024U;
    std::vector<std::uint8_t> buffer(chunk_size);
    Sha256 hasher;
    std::uint64_t remaining = bytes;
    while (remaining != 0) {
        const std::size_t requested = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, buffer.size()));
        input.read(reinterpret_cast<char *>(buffer.data()),
                   static_cast<std::streamsize>(requested));
        if (input.gcount() != static_cast<std::streamsize>(requested)) {
            throw std::runtime_error("truncated " +
                                     std::string(artifact_name) +
                                     " while computing SHA-256");
        }
        hasher.update(std::span<const std::uint8_t>(buffer.data(), requested));
        remaining -= requested;
    }
    return hasher.finish();
}

Sha256Digest sha256_file_prefix(const std::filesystem::path & path,
                                std::uint64_t bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file for SHA-256: " +
                                 path.string());
    }
    return sha256_stream_prefix(input, bytes, path.string());
}

void append_sha256_footer(const std::filesystem::path & path) {
    const std::uint64_t size = std::filesystem::file_size(path);
    const Sha256Digest digest = sha256_file_prefix(path, size);
    std::ofstream output(path, std::ios::binary | std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot append SHA-256 footer: " +
                                 path.string());
    }
    output.write(reinterpret_cast<const char *>(digest.data()),
                 static_cast<std::streamsize>(digest.size()));
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize SHA-256 footer: " +
                                 path.string());
    }
}

std::uint64_t verify_sha256_footer(std::istream & input,
                                   std::string_view artifact_name) {
    const std::uint64_t file_size = stream_size(input, artifact_name);
    if (file_size < Sha256Digest{}.size()) {
        throw std::runtime_error(std::string(artifact_name) +
                                 " is missing its SHA-256 footer");
    }
    const std::uint64_t payload_size = file_size - Sha256Digest{}.size();
    if (payload_size >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error(std::string(artifact_name) +
                                 " is too large for stream offsets");
    }
    input.seekg(static_cast<std::streamoff>(payload_size));
    Sha256Digest stored{};
    input.read(reinterpret_cast<char *>(stored.data()),
               static_cast<std::streamsize>(stored.size()));
    if (!input) {
        throw std::runtime_error(std::string(artifact_name) +
                                 " has a truncated SHA-256 footer");
    }
    const Sha256Digest computed =
        sha256_stream_prefix(input, payload_size, artifact_name);
    if (stored != computed) {
        throw std::runtime_error(std::string(artifact_name) +
                                 " SHA-256 checksum mismatch");
    }
    input.clear();
    input.seekg(0, std::ios::beg);
    if (!input) {
        throw std::runtime_error("cannot rewind " +
                                 std::string(artifact_name));
    }
    return payload_size;
}

std::string hex_digest(const Sha256Digest & digest) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result;
    result.resize(digest.size() * 2U);
    for (std::size_t i = 0; i < digest.size(); ++i) {
        result[i * 2U] = digits[digest[i] >> 4U];
        result[i * 2U + 1U] = digits[digest[i] & 0x0fU];
    }
    return result;
}

std::uint32_t crc32c(std::string_view bytes) {
    std::uint32_t checksum = 0xffffffffU;
    for (const unsigned char byte : bytes) {
        checksum =
            crc32c_table[(checksum ^ byte) & 0xffU] ^ (checksum >> 8U);
    }
    return ~checksum;
}

} // namespace leann::detail
