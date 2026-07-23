#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <span>
#include <string>
#include <string_view>

namespace leann::detail {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 {
  public:
    Sha256();

    void update(std::span<const std::uint8_t> bytes);
    void update(std::string_view text);
    [[nodiscard]] Sha256Digest finish();

  private:
    void transform(const std::uint8_t * block);

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffered_ = 0;
    std::uint64_t total_bytes_ = 0;
    bool finished_ = false;
};

[[nodiscard]] Sha256Digest sha256(std::string_view text);
[[nodiscard]] std::uint64_t
stream_size(std::istream & input, std::string_view artifact_name);
[[nodiscard]] Sha256Digest
sha256_stream_prefix(std::istream & input, std::uint64_t bytes,
                     std::string_view artifact_name);
[[nodiscard]] Sha256Digest
sha256_file_prefix(const std::filesystem::path & path, std::uint64_t bytes);
void append_sha256_footer(const std::filesystem::path & path);
[[nodiscard]] std::uint64_t
verify_sha256_footer(std::istream & input, std::string_view artifact_name);
[[nodiscard]] std::string hex_digest(const Sha256Digest & digest);

[[nodiscard]] std::uint32_t crc32c(std::string_view bytes);

} // namespace leann::detail
