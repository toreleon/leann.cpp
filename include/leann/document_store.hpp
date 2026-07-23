#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace leann {

class DocumentStore {
  public:
    static void write(const std::filesystem::path & path,
                      std::span<const std::string> documents);
    static DocumentStore open(const std::filesystem::path & path);

    DocumentStore(DocumentStore &&) noexcept = default;
    DocumentStore & operator=(DocumentStore &&) noexcept = default;
    DocumentStore(const DocumentStore &) = delete;
    DocumentStore & operator=(const DocumentStore &) = delete;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::string read(std::uint32_t id);
    [[nodiscard]] std::vector<std::string>
    read_many(std::span<const std::uint32_t> ids);
    [[nodiscard]] std::uint64_t raw_bytes() const noexcept;

  private:
    DocumentStore() = default;

    std::filesystem::path path_;
    std::ifstream stream_;
    std::vector<std::uint64_t> offsets_;
    std::uint64_t data_offset_ = 0;
};

} // namespace leann
