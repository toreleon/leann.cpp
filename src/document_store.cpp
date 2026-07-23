#include "leann/document_store.hpp"

#include "checksum.hpp"
#include "format.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace leann {
namespace {

constexpr std::array<char, 8> document_magic{'L', 'E', 'A', 'N',
                                              'D', 'C', '0', '2'};
constexpr std::uint32_t document_version = 2;
constexpr std::string_view corpus_identity_domain =
    "leann.cpp corpus identity v1";

void hash_u64_le(detail::Sha256 & hasher, std::uint64_t value) {
    std::array<std::uint8_t, sizeof(value)> bytes{};
    for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
        bytes[byte] = static_cast<std::uint8_t>(value >> (byte * 8U));
    }
    hasher.update(bytes);
}

PairIdentity
make_pair_identity(std::span<const std::string> documents) {
    detail::Sha256 hasher;
    hasher.update(corpus_identity_domain);
    hash_u64_le(hasher, static_cast<std::uint64_t>(documents.size()));
    for (const std::string & document : documents) {
        hash_u64_le(hasher, static_cast<std::uint64_t>(document.size()));
        hasher.update(document);
    }
    return hasher.finish();
}

bool is_zero_identity(const PairIdentity & identity) {
    return std::all_of(identity.begin(), identity.end(),
                       [](std::uint8_t byte) { return byte == 0U; });
}

} // namespace

DocumentStore::DocumentStore(DocumentStore && other) noexcept
    : path_(std::move(other.path_)), stream_(std::move(other.stream_)),
      pair_identity_(other.pair_identity_),
      offsets_(std::move(other.offsets_)),
      checksums_(std::move(other.checksums_)),
      data_offset_(other.data_offset_), file_size_(other.file_size_) {}

DocumentStore &
DocumentStore::operator=(DocumentStore && other) noexcept {
    if (this != &other) {
        path_ = std::move(other.path_);
        stream_ = std::move(other.stream_);
        pair_identity_ = other.pair_identity_;
        offsets_ = std::move(other.offsets_);
        checksums_ = std::move(other.checksums_);
        data_offset_ = other.data_offset_;
        file_size_ = other.file_size_;
    }
    return *this;
}

void DocumentStore::write(const std::filesystem::path & path,
                          std::span<const std::string> documents) {
    write(path, documents, make_pair_identity(documents));
}

void DocumentStore::write(const std::filesystem::path & path,
                          std::span<const std::string> documents,
                          const PairIdentity & pair_identity) {
    if (is_zero_identity(pair_identity)) {
        throw std::invalid_argument("document pair identity must not be zero");
    }

    std::vector<std::uint32_t> checksums;
    checksums.reserve(documents.size());
    for (const std::string & document : documents) {
        checksums.push_back(detail::crc32c(document));
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create document store: " +
                                 path.string());
    }

    detail::write_bytes(output, document_magic.data(), document_magic.size());
    detail::write_le<std::uint32_t>(output, document_version);
    detail::write_bytes(
        output, reinterpret_cast<const char *>(pair_identity.data()),
        pair_identity.size());
    detail::write_le<std::uint64_t>(output, documents.size());
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize document header: " +
                                 path.string());
    }

    const std::uint64_t header_size = std::filesystem::file_size(path);
    const detail::Sha256Digest header_digest =
        detail::sha256_file_prefix(path, header_size);
    output.open(path, std::ios::binary | std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot append document metadata: " +
                                 path.string());
    }
    detail::write_bytes(
        output, reinterpret_cast<const char *>(header_digest.data()),
        header_digest.size());

    std::uint64_t offset = 0;
    detail::write_le<std::uint64_t>(output, offset);
    for (const std::string & document : documents) {
        if (document.size() >
            std::numeric_limits<std::uint64_t>::max() - offset) {
            throw std::runtime_error(
                "document store exceeds uint64 address space");
        }
        offset += static_cast<std::uint64_t>(document.size());
        detail::write_le<std::uint64_t>(output, offset);
    }
    for (const std::uint32_t checksum : checksums) {
        detail::write_le<std::uint32_t>(output, checksum);
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize document metadata: " +
                                 path.string());
    }

    const std::uint64_t metadata_size = std::filesystem::file_size(path);
    const detail::Sha256Digest metadata_digest =
        detail::sha256_file_prefix(path, metadata_size);
    output.open(path, std::ios::binary | std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot append document payload: " +
                                 path.string());
    }
    detail::write_bytes(
        output, reinterpret_cast<const char *>(metadata_digest.data()),
        metadata_digest.size());
    for (const std::string & document : documents) {
        detail::write_bytes(output, document.data(), document.size());
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize document store: " +
                                 path.string());
    }
}

DocumentStore DocumentStore::open(const std::filesystem::path & path) {
    DocumentStore store;
    store.path_ = path;
    store.stream_.open(path, std::ios::binary);
    if (!store.stream_) {
        throw std::runtime_error("cannot open document store: " +
                                 path.string());
    }
    store.file_size_ =
        detail::stream_size(store.stream_, "document store");

    std::array<char, document_magic.size()> magic{};
    detail::read_bytes(store.stream_, magic.data(), magic.size());
    if (magic != document_magic) {
        throw std::runtime_error("not a leann.cpp document store: " +
                                 path.string());
    }
    const std::uint32_t version =
        detail::read_le<std::uint32_t>(store.stream_);
    if (version != document_version) {
        throw std::runtime_error("unsupported document store version: " +
                                 std::to_string(version));
    }
    detail::read_bytes(
        store.stream_, reinterpret_cast<char *>(store.pair_identity_.data()),
        store.pair_identity_.size());
    if (is_zero_identity(store.pair_identity_)) {
        throw std::runtime_error("document store has a zero pair identity");
    }

    const std::uint64_t count_on_disk =
        detail::read_le<std::uint64_t>(store.stream_);
    const auto header_position = store.stream_.tellg();
    if (header_position < 0) {
        throw std::runtime_error("invalid document header position");
    }
    const std::uint64_t header_size =
        static_cast<std::uint64_t>(header_position);
    detail::Sha256Digest stored_header_digest{};
    detail::read_bytes(
        store.stream_, reinterpret_cast<char *>(stored_header_digest.data()),
        stored_header_digest.size());
    const detail::Sha256Digest computed_header_digest =
        detail::sha256_stream_prefix(store.stream_, header_size,
                                     "document header");
    if (stored_header_digest != computed_header_digest) {
        throw std::runtime_error(
            "document header SHA-256 checksum mismatch");
    }

    const std::uint64_t table_position =
        header_size + stored_header_digest.size();
    store.stream_.clear();
    store.stream_.seekg(static_cast<std::streamoff>(table_position));
    if (!store.stream_) {
        throw std::runtime_error("failed to seek document checksum table");
    }
    constexpr std::uint64_t digest_bytes = detail::Sha256Digest{}.size();
    constexpr std::uint64_t first_offset_bytes = sizeof(std::uint64_t);
    constexpr std::uint64_t bytes_per_document =
        sizeof(std::uint64_t) + sizeof(std::uint32_t);
    if (table_position > store.file_size_ ||
        store.file_size_ - table_position < digest_bytes ||
        count_on_disk >
            (std::numeric_limits<std::uint64_t>::max() -
             first_offset_bytes) /
                bytes_per_document) {
        throw std::runtime_error("document metadata exceeds file size");
    }
    const std::uint64_t table_bytes =
        first_offset_bytes + count_on_disk * bytes_per_document;
    if (table_bytes > store.file_size_ - table_position - digest_bytes) {
        throw std::runtime_error("document metadata exceeds file size");
    }

    const std::uint64_t metadata_size = table_position + table_bytes;
    if (metadata_size >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error(
            "document metadata exceeds stream offset limits");
    }
    store.stream_.seekg(static_cast<std::streamoff>(metadata_size));
    detail::Sha256Digest stored_metadata_digest{};
    detail::read_bytes(
        store.stream_,
        reinterpret_cast<char *>(stored_metadata_digest.data()),
        stored_metadata_digest.size());
    const detail::Sha256Digest computed_metadata_digest =
        detail::sha256_stream_prefix(store.stream_, metadata_size,
                                     "document metadata");
    if (stored_metadata_digest != computed_metadata_digest) {
        throw std::runtime_error(
            "document metadata SHA-256 checksum mismatch");
    }

    const std::size_t count =
        detail::checked_size(count_on_disk, "document count");
    store.stream_.clear();
    store.stream_.seekg(static_cast<std::streamoff>(table_position));
    if (!store.stream_) {
        throw std::runtime_error("failed to seek document metadata");
    }
    store.offsets_.resize(count + 1U);
    for (std::uint64_t & offset : store.offsets_) {
        offset = detail::read_le<std::uint64_t>(store.stream_);
    }
    store.checksums_.resize(count);
    for (std::uint32_t & checksum : store.checksums_) {
        checksum = detail::read_le<std::uint32_t>(store.stream_);
    }

    if (store.offsets_.empty() || store.offsets_.front() != 0) {
        throw std::runtime_error("invalid document offsets");
    }
    for (std::size_t i = 1; i < store.offsets_.size(); ++i) {
        if (store.offsets_[i] < store.offsets_[i - 1]) {
            throw std::runtime_error("document offsets are not monotonic");
        }
    }
    store.data_offset_ = metadata_size + digest_bytes;
    if (store.data_offset_ > store.file_size_ ||
        store.offsets_.back() != store.file_size_ - store.data_offset_) {
        throw std::runtime_error(
            "document payload size does not match its offset table");
    }
    return store;
}

std::size_t DocumentStore::size() const noexcept {
    return offsets_.empty() ? 0 : offsets_.size() - 1;
}

std::string DocumentStore::read(std::uint32_t id) const {
    const std::lock_guard lock(stream_mutex_);
    if (id >= size()) {
        throw std::out_of_range("document id is out of range");
    }
    const std::uint64_t begin = offsets_[id];
    const std::uint64_t end = offsets_[id + 1U];
    const std::uint64_t length = end - begin;
    if (data_offset_ > file_size_ || begin > file_size_ - data_offset_ ||
        length > file_size_ - data_offset_ - begin) {
        throw std::runtime_error("document range exceeds file size");
    }
    const std::uint64_t absolute = data_offset_ + begin;
    if (absolute >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamoff>::max()) ||
        length >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("document range exceeds stream limits");
    }
    std::string result(detail::checked_size(length, "document size"), '\0');

    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(absolute));
    if (!stream_) {
        throw std::runtime_error("failed to seek document store");
    }
    detail::read_bytes(stream_, result.data(), result.size());
    if (detail::crc32c(result) != checksums_[id]) {
        throw std::runtime_error("document CRC32C checksum mismatch for id " +
                                 std::to_string(id));
    }
    return result;
}

std::vector<std::string>
DocumentStore::read_many(std::span<const std::uint32_t> ids) const {
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

const PairIdentity & DocumentStore::pair_identity() const noexcept {
    return pair_identity_;
}

} // namespace leann
