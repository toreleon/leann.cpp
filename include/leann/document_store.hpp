#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace leann {

using PairIdentity = std::array<std::uint8_t, 32>;

class Index;

class DocumentStore {
  public:
    static DocumentStore open(const std::filesystem::path & path);

    DocumentStore(DocumentStore && other) noexcept;
    DocumentStore & operator=(DocumentStore && other) noexcept;
    DocumentStore(const DocumentStore &) = delete;
    DocumentStore & operator=(const DocumentStore &) = delete;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::string read(std::uint32_t id) const;
    [[nodiscard]] std::vector<std::string>
    read_many(std::span<const std::uint32_t> ids) const;
    [[nodiscard]] std::uint64_t raw_bytes() const noexcept;
    [[nodiscard]] const PairIdentity & pair_identity() const noexcept;

  private:
    friend class Index;

    DocumentStore() = default;
    static void write(const std::filesystem::path & path,
                      std::span<const std::string> documents);
    static void write(const std::filesystem::path & path,
                      std::span<const std::string> documents,
                      const PairIdentity & pair_identity);

    std::filesystem::path path_;
    mutable std::ifstream stream_;
    mutable std::mutex stream_mutex_;
    PairIdentity pair_identity_{};
    std::vector<std::uint64_t> offsets_;
    std::vector<std::uint32_t> checksums_;
    std::uint64_t data_offset_ = 0;
    std::uint64_t file_size_ = 0;
};

} // namespace leann
