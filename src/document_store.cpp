#include "leann/document_store.hpp"

#include "format.hpp"

#include <array>
#include <limits>
#include <stdexcept>

namespace leann {
namespace {

constexpr std::array<char, 8> document_magic{'L', 'E', 'A', 'N', 'D', 'C', '0', '1'};
constexpr std::uint32_t document_version = 1;

} // namespace

void DocumentStore::write(const std::filesystem::path & path,
                          std::span<const std::string> documents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create document store: " + path.string());
    }

    detail::write_bytes(output, document_magic.data(), document_magic.size());
    detail::write_le<std::uint32_t>(output, document_version);
    detail::write_le<std::uint64_t>(output, documents.size());

    std::uint64_t offset = 0;
    detail::write_le<std::uint64_t>(output, offset);
    for (const std::string & document : documents) {
        if (document.size() > std::numeric_limits<std::uint64_t>::max() - offset) {
            throw std::runtime_error("document store exceeds uint64 address space");
        }
        offset += static_cast<std::uint64_t>(document.size());
        detail::write_le<std::uint64_t>(output, offset);
    }
    for (const std::string & document : documents) {
        detail::write_bytes(output, document.data(), document.size());
    }
}

DocumentStore DocumentStore::open(const std::filesystem::path & path) {
    DocumentStore store;
    store.path_ = path;
    const std::uint64_t file_size = std::filesystem::file_size(path);
    store.stream_.open(path, std::ios::binary);
    if (!store.stream_) {
        throw std::runtime_error("cannot open document store: " + path.string());
    }

    std::array<char, document_magic.size()> magic{};
    detail::read_bytes(store.stream_, magic.data(), magic.size());
    if (magic != document_magic) {
        throw std::runtime_error("not a leann.cpp document store: " + path.string());
    }
    const std::uint32_t version = detail::read_le<std::uint32_t>(store.stream_);
    if (version != document_version) {
        throw std::runtime_error("unsupported document store version: " +
                                 std::to_string(version));
    }
    const std::uint64_t count_on_disk =
        detail::read_le<std::uint64_t>(store.stream_);
    const auto table_position =
        static_cast<std::uint64_t>(store.stream_.tellg());
    if (table_position > file_size ||
        count_on_disk == std::numeric_limits<std::uint64_t>::max() ||
        count_on_disk + 1U > (file_size - table_position) / sizeof(std::uint64_t)) {
        throw std::runtime_error("document offset table exceeds file size");
    }
    const std::size_t count =
        detail::checked_size(count_on_disk, "document count");
    store.offsets_.resize(count + 1);
    for (std::uint64_t & offset : store.offsets_) {
        offset = detail::read_le<std::uint64_t>(store.stream_);
    }
    if (store.offsets_.empty() || store.offsets_.front() != 0) {
        throw std::runtime_error("invalid document offsets");
    }
    for (std::size_t i = 1; i < store.offsets_.size(); ++i) {
        if (store.offsets_[i] < store.offsets_[i - 1]) {
            throw std::runtime_error("document offsets are not monotonic");
        }
    }
    store.data_offset_ = static_cast<std::uint64_t>(store.stream_.tellg());

    if (store.data_offset_ + store.offsets_.back() > file_size) {
        throw std::runtime_error("document store points beyond end of file");
    }
    return store;
}

std::size_t DocumentStore::size() const noexcept {
    return offsets_.empty() ? 0 : offsets_.size() - 1;
}

std::string DocumentStore::read(std::uint32_t id) {
    if (id >= size()) {
        throw std::out_of_range("document id is out of range");
    }
    const std::uint64_t begin = offsets_[id];
    const std::uint64_t end = offsets_[id + 1];
    std::string result(detail::checked_size(end - begin, "document size"), '\0');

    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(data_offset_ + begin));
    if (!stream_) {
        throw std::runtime_error("failed to seek document store");
    }
    detail::read_bytes(stream_, result.data(), result.size());
    return result;
}

std::vector<std::string>
DocumentStore::read_many(std::span<const std::uint32_t> ids) {
    std::vector<std::string> result;
    result.reserve(ids.size());
    for (const std::uint32_t id : ids) {
        result.push_back(read(id));
    }
    return result;
}

std::uint64_t DocumentStore::raw_bytes() const noexcept {
    return offsets_.empty() ? 0 : offsets_.back();
}

} // namespace leann
