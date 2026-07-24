#include "leann/document_store.hpp"
#include "leann/embedder.hpp"
#include "leann/index.hpp"
#include "checksum.hpp"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

class Arguments {
  public:
    Arguments(int argc, char ** argv) {
        for (int i = 2; i < argc; ++i) {
            std::string key = argv[i];
            if (key.starts_with("--")) {
                if (key == "--help") {
                    flags_.insert(key);
                    continue;
                }
                if (i + 1 >= argc) {
                    throw std::invalid_argument("missing value for " + key);
                }
                values_[std::move(key)] = argv[++i];
            } else {
                positional_.push_back(std::move(key));
            }
        }
    }

    [[nodiscard]] bool has(std::string_view key) const {
        return values_.contains(std::string(key)) ||
               flags_.contains(std::string(key));
    }

    [[nodiscard]] std::string get(std::string_view key,
                                  std::string fallback = {}) const {
        const auto found = values_.find(std::string(key));
        return found == values_.end() ? std::move(fallback) : found->second;
    }

    [[nodiscard]] std::string require(std::string_view key) const {
        const std::string value = get(key);
        if (value.empty()) {
            throw std::invalid_argument("required option missing: " +
                                        std::string(key));
        }
        return value;
    }

    [[nodiscard]] std::uint32_t unsigned_value(std::string_view key,
                                               std::uint32_t fallback) const {
        const std::string value = get(key);
        if (value.empty()) {
            return fallback;
        }
        std::uint32_t parsed = 0;
        const auto [end, error] = std::from_chars(
            value.data(), value.data() + value.size(), parsed);
        if (error == std::errc::result_out_of_range) {
            throw std::invalid_argument(std::string(key) + " is too large");
        }
        if (error != std::errc{} ||
            end != value.data() + value.size()) {
            throw std::invalid_argument(
                std::string(key) + " must be an unsigned integer");
        }
        return parsed;
    }

    [[nodiscard]] int int_value(std::string_view key, int fallback) const {
        const std::string value = get(key);
        return value.empty() ? fallback : std::stoi(value);
    }

    [[nodiscard]] double double_value(std::string_view key,
                                      double fallback) const {
        const std::string value = get(key);
        if (value.empty()) {
            return fallback;
        }
        double parsed = 0.0;
        const auto [end, error] = std::from_chars(
            value.data(), value.data() + value.size(), parsed,
            std::chars_format::general);
        if (error != std::errc{} ||
            end != value.data() + value.size() ||
            !std::isfinite(parsed)) {
            throw std::invalid_argument(
                std::string(key) + " must be a finite number");
        }
        return parsed;
    }

    [[nodiscard]] bool bool_value(std::string_view key, bool fallback) const {
        const std::string value = get(key);
        if (value.empty()) {
            return fallback;
        }
        if (value == "1" || value == "true" || value == "yes") {
            return true;
        }
        if (value == "0" || value == "false" || value == "no") {
            return false;
        }
        throw std::invalid_argument(std::string(key) +
                                    " must be true/false or 1/0");
    }

  private:
    std::unordered_map<std::string, std::string> values_;
    std::unordered_set<std::string> flags_;
    std::vector<std::string> positional_;
};

std::vector<std::string> read_lines(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open text file: " + path.string());
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(std::move(line));
        }
    }
    return lines;
}

struct DocumentsSource {
    std::vector<std::string> documents;
    std::uint64_t file_size = 0;
    leann::detail::Sha256Digest sha256{};
};

DocumentsSource
read_documents_source(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open text file: " + path.string());
    }

    DocumentsSource result;
    leann::detail::Sha256 hasher;
    constexpr std::size_t chunk_size = 1024U * 1024U;
    std::array<char, chunk_size> buffer{};
    std::string line;
    auto finish_line = [&] {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            result.documents.push_back(std::move(line));
        }
        line.clear();
    };

    while (input) {
        input.read(buffer.data(),
                   static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count <= 0) {
            continue;
        }
        const auto bytes = static_cast<std::size_t>(count);
        if (bytes > std::numeric_limits<std::uint64_t>::max() -
                        result.file_size) {
            throw std::runtime_error("documents file is too large");
        }
        result.file_size += bytes;
        hasher.update(std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t *>(buffer.data()), bytes));
        for (std::size_t index = 0; index < bytes; ++index) {
            if (buffer[index] == '\n') {
                finish_line();
            } else {
                line.push_back(buffer[index]);
            }
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed to read text file: " + path.string());
    }
    if (!line.empty()) {
        finish_line();
    }
    result.sha256 = hasher.finish();
    return result;
}

template <typename T>
void write_little(std::ostream & output, T value) {
    static_assert(std::is_unsigned_v<T>);
    for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
        output.put(static_cast<char>(value & static_cast<T>(0xffU)));
        value >>= 8U;
    }
    if (!output) {
        throw std::runtime_error("failed to write embedding cache");
    }
}

template <typename T>
T read_little(std::istream & input) {
    static_assert(std::is_unsigned_v<T>);
    T value = 0;
    for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
        const int next = input.get();
        if (next == std::char_traits<char>::eof()) {
            throw std::runtime_error("truncated embedding cache");
        }
        value |= static_cast<T>(static_cast<unsigned char>(next))
                 << (8U * byte);
    }
    return value;
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs,
                          std::string_view description) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(description) + " is too large");
    }
    return lhs + rhs;
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs,
                               std::string_view description) {
    if (lhs != 0 &&
        rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(description) + " is too large");
    }
    return lhs * rhs;
}

struct NamedPath {
    std::string label;
    std::filesystem::path path;
};

struct PathIdentity {
    std::filesystem::path lexical;
    std::filesystem::path weak_canonical;
    bool exists = false;
};

PathIdentity path_identity(const NamedPath & value) {
    std::error_code error;
    auto lexical = std::filesystem::absolute(value.path, error);
    if (error) {
        throw std::runtime_error("cannot resolve " + value.label + " path '" +
                                 value.path.string() + "': " +
                                 error.message());
    }
    lexical = lexical.lexically_normal();
    auto canonical = std::filesystem::weakly_canonical(lexical, error);
    if (error) {
        throw std::runtime_error("cannot canonicalize " + value.label +
                                 " path '" + lexical.string() + "': " +
                                 error.message());
    }
    const bool exists = std::filesystem::exists(lexical, error);
    if (error) {
        throw std::runtime_error("cannot inspect " + value.label + " path '" +
                                 lexical.string() + "': " + error.message());
    }
    return {std::move(lexical), std::move(canonical), exists};
}

bool paths_alias(const NamedPath & lhs, const NamedPath & rhs) {
    const PathIdentity left = path_identity(lhs);
    const PathIdentity right = path_identity(rhs);
    if (left.lexical == right.lexical ||
        left.weak_canonical == right.weak_canonical) {
        return true;
    }
    if (!left.exists || !right.exists) {
        return false;
    }
    std::error_code error;
    const bool equivalent =
        std::filesystem::equivalent(left.lexical, right.lexical, error);
    if (error) {
        throw std::runtime_error(
            "cannot compare " + lhs.label + " path '" +
            left.lexical.string() + "' with " + rhs.label + " path '" +
            right.lexical.string() + "': " + error.message());
    }
    return equivalent;
}

void require_distinct_output_paths(
    std::span<const NamedPath> outputs,
    std::span<const NamedPath> protected_inputs) {
    for (std::size_t left = 0; left < outputs.size(); ++left) {
        for (std::size_t right = left + 1; right < outputs.size(); ++right) {
            if (paths_alias(outputs[left], outputs[right])) {
                throw std::invalid_argument(
                    outputs[left].label + " and " + outputs[right].label +
                    " resolve to the same file");
            }
        }
        for (const NamedPath & input : protected_inputs) {
            if (paths_alias(outputs[left], input)) {
                throw std::invalid_argument(
                    outputs[left].label + " aliases protected " +
                    input.label + " path");
            }
        }
    }
}

void sync_file_contents(const std::filesystem::path & path) {
#ifdef _WIN32
    const HANDLE handle = CreateFileW(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error(
            "cannot open temporary output for durable flush: " +
            std::system_category().message(
                static_cast<int>(GetLastError())));
    }
    if (FlushFileBuffers(handle) == 0) {
        const auto error = static_cast<int>(GetLastError());
        CloseHandle(handle);
        throw std::runtime_error(
            "cannot durably flush temporary output: " +
            std::system_category().message(error));
    }
    if (CloseHandle(handle) == 0) {
        throw std::runtime_error(
            "cannot close temporary output after durable flush: " +
            std::system_category().message(
                static_cast<int>(GetLastError())));
    }
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw std::runtime_error(
            "cannot open temporary output for durable flush: " +
            std::error_code(errno, std::generic_category()).message());
    }
    int result = 0;
    do {
        result = ::fsync(descriptor);
    } while (result != 0 && errno == EINTR);
    if (result != 0) {
        const std::error_code error(errno, std::generic_category());
        ::close(descriptor);
        throw std::runtime_error(
            "cannot durably flush temporary output: " + error.message());
    }
    if (::close(descriptor) != 0) {
        throw std::runtime_error(
            "cannot close temporary output after durable flush: " +
            std::error_code(errno, std::generic_category()).message());
    }
#endif
}

void atomic_replace(const std::filesystem::path & source,
                    const std::filesystem::path & target,
                    std::string_view description) {
#ifdef _WIN32
    if (MoveFileExW(source.c_str(), target.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
        throw std::runtime_error(
            "cannot publish " + std::string(description) + " file: " +
            std::system_category().message(
                static_cast<int>(GetLastError())));
    }
#else
    std::error_code error;
    std::filesystem::rename(source, target, error);
    if (error) {
        throw std::runtime_error(
            "cannot publish " + std::string(description) + " file: " +
            error.message());
    }
#endif
}

std::uint64_t current_process_id() noexcept {
#ifdef _WIN32
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(::getpid());
#endif
}

class AtomicFileOutput {
  public:
    AtomicFileOutput(std::filesystem::path target, std::string description)
        : target_(std::move(target)),
          description_(std::move(description)) {
        if (target_.filename().empty()) {
            throw std::invalid_argument(
                description_ + " output must name a file");
        }
        if (!target_.parent_path().empty()) {
            std::filesystem::create_directories(target_.parent_path());
        }
        static std::atomic<std::uint64_t> sequence{0};
        const auto tick = std::chrono::steady_clock::now()
                              .time_since_epoch()
                              .count();
        const std::uint64_t process_id = current_process_id();
        for (std::uint32_t attempt = 0; attempt < 128; ++attempt) {
            temporary_ = target_.parent_path() /
                         ("." + target_.filename().string() + ".tmp." +
                          std::to_string(process_id) + "." +
                          std::to_string(tick) + "." +
                          std::to_string(sequence.fetch_add(
                              1, std::memory_order_relaxed)) +
                          "." +
                          std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::exists(temporary_, error)) {
                continue;
            }
            if (error) {
                throw std::runtime_error(
                    "cannot inspect temporary " + description_ + " path: " +
                    error.message());
            }
            output_.open(temporary_,
                         std::ios::binary | std::ios::out | std::ios::trunc);
            if (output_) {
                return;
            }
            output_.clear();
        }
        throw std::runtime_error(
            "cannot create temporary " + description_ + " file next to " +
            target_.string());
    }

    AtomicFileOutput(const AtomicFileOutput &) = delete;
    AtomicFileOutput & operator=(const AtomicFileOutput &) = delete;

    ~AtomicFileOutput() {
        if (output_.is_open()) {
            output_.close();
        }
        if (!published_ && !temporary_.empty()) {
            std::error_code ignored;
            std::filesystem::remove(temporary_, ignored);
        }
    }

    std::ostream & stream() noexcept {
        return output_;
    }

    [[nodiscard]] const std::filesystem::path &
    temporary_path() const noexcept {
        return temporary_;
    }

    void publish() {
        output_.flush();
        if (!output_) {
            throw std::runtime_error(
                "failed to flush temporary " + description_ + " file");
        }
        output_.close();
        if (!output_) {
            throw std::runtime_error(
                "failed to close temporary " + description_ + " file");
        }
        sync_file_contents(temporary_);
        atomic_replace(temporary_, target_, description_);
        published_ = true;
    }

  private:
    std::filesystem::path target_;
    std::filesystem::path temporary_;
    std::string description_;
    std::ofstream output_;
    bool published_ = false;
};

std::uint32_t decode_little_u32(std::uint32_t raw) {
    if constexpr (std::endian::native == std::endian::little) {
        return raw;
    } else {
        const auto bytes =
            std::bit_cast<std::array<std::uint8_t, sizeof(raw)>>(raw);
        return static_cast<std::uint32_t>(bytes[0]) |
               (static_cast<std::uint32_t>(bytes[1]) << 8U) |
               (static_cast<std::uint32_t>(bytes[2]) << 16U) |
               (static_cast<std::uint32_t>(bytes[3]) << 24U);
    }
}

class EmbeddingCacheEmbedder final : public leann::Embedder {
  public:
    EmbeddingCacheEmbedder(const std::filesystem::path & cache_path,
                           const DocumentsSource & source,
                           std::string source_option = "--docs")
        : input_(cache_path, std::ios::binary),
          source_option_(std::move(source_option)) {
        if (!input_) {
            throw std::runtime_error("cannot open embedding cache: " +
                                     cache_path.string());
        }

        const std::uint64_t cache_size =
            leann::detail::stream_size(input_, "embedding cache");
        constexpr std::array<char, 8> expected_magic{
            'L', 'E', 'A', 'N', 'N', 'B', 'C', '2'};
        std::array<char, expected_magic.size()> magic{};
        input_.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        if (!input_ || magic != expected_magic) {
            throw std::runtime_error(
                "build embedding cache is not LEANNBC2");
        }

        const std::uint32_t encoded_dimension =
            read_little<std::uint32_t>(input_);
        const std::uint64_t encoded_count =
            read_little<std::uint64_t>(input_);
        const std::uint32_t fingerprint_size =
            read_little<std::uint32_t>(input_);
        const std::uint64_t documents_size =
            read_little<std::uint64_t>(input_);
        leann::detail::Sha256Digest documents_sha256{};
        input_.read(
            reinterpret_cast<char *>(documents_sha256.data()),
            static_cast<std::streamsize>(documents_sha256.size()));
        if (!input_) {
            throw std::runtime_error("truncated build embedding cache header");
        }

        if (encoded_dimension == 0) {
            throw std::runtime_error(
                "build embedding cache has zero dimension");
        }
        if (encoded_count == 0) {
            throw std::runtime_error(
                "build embedding cache has zero documents");
        }
        if (encoded_count != source.documents.size()) {
            throw std::runtime_error(
                "embedding cache row count does not match " +
                source_option_);
        }
        if (documents_size != source.file_size ||
            documents_sha256 != source.sha256) {
            throw std::runtime_error(
                "embedding cache does not match the exact " +
                source_option_ + " file");
        }
        if (fingerprint_size == 0) {
            throw std::runtime_error(
                "build embedding cache has an empty embedder fingerprint");
        }
        constexpr std::uint32_t maximum_fingerprint_size =
            16U * 1024U * 1024U;
        if (fingerprint_size > maximum_fingerprint_size) {
            throw std::runtime_error(
                "build embedding cache fingerprint is unreasonably large");
        }

        constexpr std::uint64_t fixed_header_size =
            expected_magic.size() + sizeof(std::uint32_t) +
            sizeof(std::uint64_t) + sizeof(std::uint32_t) +
            sizeof(std::uint64_t) +
            std::tuple_size_v<leann::detail::Sha256Digest>;
        const std::uint64_t values =
            checked_multiply(encoded_count, encoded_dimension,
                             "build embedding cache payload");
        const std::uint64_t payload_bytes =
            checked_multiply(values, sizeof(float),
                             "build embedding cache payload");
        const std::uint64_t expected_size = checked_add(
            checked_add(fixed_header_size, fingerprint_size,
                        "build embedding cache"),
            payload_bytes, "build embedding cache");
        if (cache_size != expected_size) {
            throw std::runtime_error(
                "build embedding cache has a truncated or trailing payload");
        }

        fingerprint_.resize(fingerprint_size);
        input_.read(fingerprint_.data(),
                    static_cast<std::streamsize>(fingerprint_.size()));
        if (!input_) {
            throw std::runtime_error(
                "truncated build embedding cache fingerprint");
        }
        dimension_ = encoded_dimension;
        remaining_ = encoded_count;
    }

    [[nodiscard]] std::size_t dimension() const noexcept override {
        return dimension_;
    }

    [[nodiscard]] std::string fingerprint() const override {
        return fingerprint_;
    }

    [[nodiscard]] std::vector<leann::Embedding>
    embed(std::span<const std::string> texts) override {
        if (texts.size() > remaining_) {
            throw std::runtime_error(
                "build requested more vectors than the embedding cache "
                "contains");
        }

        const std::uint64_t value_count = checked_multiply(
            static_cast<std::uint64_t>(texts.size()),
            static_cast<std::uint64_t>(dimension_),
            "embedding cache decode batch");
        std::vector<leann::Embedding> result(
            texts.size(), leann::Embedding(dimension_));
        std::vector<double> squared_norms(texts.size(), 0.0);
        constexpr std::size_t maximum_chunk_values = 1024U * 1024U;
        decode_buffer_.resize(static_cast<std::size_t>(
            std::min<std::uint64_t>(value_count, maximum_chunk_values)));
        std::uint64_t decoded = 0;
        std::size_t row = 0;
        std::size_t column = 0;
        while (decoded < value_count) {
            const std::size_t chunk_values = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    value_count - decoded, decode_buffer_.size()));
            const std::size_t chunk_bytes =
                chunk_values * sizeof(std::uint32_t);
            input_.read(
                reinterpret_cast<char *>(decode_buffer_.data()),
                static_cast<std::streamsize>(chunk_bytes));
            if (input_.gcount() !=
                static_cast<std::streamsize>(chunk_bytes)) {
                throw std::runtime_error("truncated embedding cache");
            }
            for (std::size_t index = 0; index < chunk_values; ++index) {
                const float value = std::bit_cast<float>(
                    decode_little_u32(decode_buffer_[index]));
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "build embedding cache contains NaN or infinity");
                }
                result[row][column] = value;
                squared_norms[row] +=
                    static_cast<double>(value) * static_cast<double>(value);
                ++column;
                if (column == dimension_) {
                    column = 0;
                    ++row;
                }
            }
            decoded += chunk_values;
        }
        if (row != texts.size() || column != 0) {
            throw std::logic_error(
                "embedding cache batch decoder lost row alignment");
        }
        for (const double squared_norm : squared_norms) {
            const double norm = std::sqrt(squared_norm);
            if (!std::isfinite(norm) ||
                std::abs(norm - 1.0) > 2.0e-3) {
                throw std::runtime_error(
                    "build embedding cache contains a non-normalized "
                    "vector");
            }
        }
        remaining_ -= texts.size();
        if (remaining_ == 0 &&
            input_.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error(
                "build embedding cache has unexpected trailing bytes");
        }
        return result;
    }

    void require_complete() const {
        if (remaining_ != 0) {
            throw std::runtime_error(
                "build did not consume the complete embedding cache");
        }
    }

  private:
    std::ifstream input_;
    std::size_t dimension_ = 0;
    std::uint64_t remaining_ = 0;
    std::string fingerprint_;
    std::string source_option_;
    std::vector<std::uint32_t> decode_buffer_;
};

void write_embedding_cache(
    const std::filesystem::path & path,
    std::string_view fingerprint,
    std::span<const leann::Embedding> embeddings) {
    if (fingerprint.size() > std::numeric_limits<std::uint32_t>::max() ||
        embeddings.empty() || embeddings.front().empty()) {
        throw std::invalid_argument("invalid embedding cache contents");
    }
    AtomicFileOutput publisher(path, "embedding cache");
    std::ostream & output = publisher.stream();
    constexpr std::array<char, 8> magic{
        'L', 'E', 'A', 'N', 'N', 'B', 'C', '1'};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_little<std::uint32_t>(
        output, static_cast<std::uint32_t>(embeddings.front().size()));
    write_little<std::uint64_t>(output, embeddings.size());
    write_little<std::uint32_t>(
        output, static_cast<std::uint32_t>(fingerprint.size()));
    output.write(fingerprint.data(),
                 static_cast<std::streamsize>(fingerprint.size()));
    for (const leann::Embedding & embedding : embeddings) {
        if (embedding.size() != embeddings.front().size()) {
            throw std::invalid_argument(
                "embedding cache dimensions are inconsistent");
        }
        for (const float value : embedding) {
            write_little<std::uint32_t>(
                output, std::bit_cast<std::uint32_t>(value));
        }
    }
    if (!output) {
        throw std::runtime_error("failed to finalize embedding cache");
    }
    publisher.publish();
}

std::vector<leann::Embedding> load_embedding_cache(
    const std::filesystem::path & path,
    std::string_view expected_fingerprint,
    std::size_t expected_count,
    std::size_t expected_dimension) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open embedding cache: " +
                                 path.string());
    }
    constexpr std::array<char, 8> expected_magic{
        'L', 'E', 'A', 'N', 'N', 'B', 'C', '1'};
    std::array<char, expected_magic.size()> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const auto dimension = read_little<std::uint32_t>(input);
    const auto count = read_little<std::uint64_t>(input);
    const auto fingerprint_size = read_little<std::uint32_t>(input);
    if (fingerprint_size != expected_fingerprint.size()) {
        throw std::runtime_error(
            "embedding cache does not match this corpus/embedder");
    }
    std::string fingerprint(fingerprint_size, '\0');
    input.read(fingerprint.data(),
               static_cast<std::streamsize>(fingerprint.size()));
    if (!input || magic != expected_magic || count != expected_count ||
        dimension != expected_dimension ||
        fingerprint != expected_fingerprint) {
        throw std::runtime_error(
            "embedding cache does not match this corpus/embedder");
    }

    std::vector<leann::Embedding> embeddings(
        expected_count, leann::Embedding(expected_dimension));
    for (leann::Embedding & embedding : embeddings) {
        for (float & value : embedding) {
            value = std::bit_cast<float>(
                read_little<std::uint32_t>(input));
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "embedding cache contains NaN or infinity");
            }
        }
        leann::normalize(embedding);
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(
            "embedding cache has unexpected trailing bytes");
    }
    return embeddings;
}

std::unique_ptr<leann::Embedder> make_embedder(const Arguments & args) {
    const std::string kind = args.get("--embedder", "hash");
    if (kind == "cache") {
        throw std::invalid_argument(
            "--embedder cache is supported only by the build command");
    }
    if (kind == "hash") {
        return std::make_unique<leann::HashEmbedder>(
            args.unsigned_value("--hash-dim", 256));
    }
    if (kind == "llama") {
#ifdef LEANN_WITH_LLAMA
        leann::LlamaEmbedder::Config config;
        config.model_path = args.require("--model");
        config.context_tokens = args.unsigned_value("--ctx", 512);
        config.batch_tokens = args.unsigned_value("--batch-tokens", 2048);
        config.max_sequences = args.unsigned_value("--parallel", 8);
        config.threads = args.int_value("--threads", 0);
        config.gpu_layers = args.int_value("--gpu-layers", 99);
        return std::make_unique<leann::LlamaEmbedder>(std::move(config));
#else
        throw std::runtime_error(
            "this binary was built without llama.cpp; configure with "
            "-DLEANN_ENABLE_LLAMA=ON");
#endif
    }
    throw std::invalid_argument("unknown embedder: " + kind);
}

leann::SearchConfig search_config(const Arguments & args) {
    leann::SearchConfig config;
    config.top_k = args.unsigned_value("--top-k", config.top_k);
    config.ef_search = args.unsigned_value("--ef-search", config.ef_search);
    config.recompute_batch_size =
        args.unsigned_value("--recompute-batch", config.recompute_batch_size);
    config.approximate_scan_limit =
        args.unsigned_value("--scan-limit", config.approximate_scan_limit);
    config.rerank_ratio =
        args.double_value("--rerank-ratio", config.rerank_ratio);
    return config;
}

void print_usage(std::ostream & output) {
    output
        << "leann.cpp — native low-storage vector search\n\n"
        << "Usage:\n"
        << "  leann build  --docs FILE --index PREFIX [embedding/build options]\n"
        << "  leann search --index PREFIX --query TEXT [embedding/search options]\n"
        << "  leann bench  --index PREFIX --queries FILE [embedding/search options]\n"
        << "  leann stats  --index PREFIX\n\n"
        << "Embedding options:\n"
        << "  --embedder hash|llama     query-capable embedding backend\n"
        << "  --model FILE              GGUF embedding model for llama.cpp\n"
        << "  --hash-dim N              hash backend dimension (default 256)\n"
        << "  --ctx N --batch-tokens N --parallel N --threads N --gpu-layers N\n\n"
        << "Build options:\n"
        << "  --embedder cache          stream a verified LEANNBC2 cache (build only)\n"
        << "  --embedding-cache FILE    cache bound to the exact --docs bytes\n"
        << "  --graph-degree N          hnswlib M (default 16)\n"
        << "  --ef-construction N       hnswlib build ef (default 100)\n"
        << "  --low-degree N            outgoing cap for non-hubs (default 3)\n"
        << "  --hub-ratio F             fraction of preserved hubs (default 0.02)\n"
        << "  --approx pq|simhash       approximate distance backend (default pq)\n"
        << "  --pq-subquantizers N      PQ subspaces; must divide dimension (default 64)\n"
        << "  --pq-bits N               bits per PQ code, 1..8 (default 4)\n"
        << "  --pq-iterations N         Lloyd iterations (default 10)\n"
        << "  --pq-training-samples N   maximum training vectors (default 4096)\n"
        << "  --sketch-bits N           SimHash bits for simhash mode (default 128)\n"
        << "  --embedding-batch N       build embedding batch (default 32)\n\n"
        << "Search options:\n"
        << "  --top-k N --ef-search N --recompute-batch N --rerank-ratio F\n"
        << "  --scan-limit N            flat ADC below N nodes; 0 forces graph\n\n"
        << "Benchmark options:\n"
        << "  --dense-baseline 0|1      build and measure dense HNSW (default 0)\n"
        << "  --dense-m N --dense-ef-construction N --dense-ef-search N\n"
        << "  --ground-truth-cache FILE reuse benchmark-only dense embeddings\n"
        << "  --ground-truth FILE       precomputed LEANN_GT1 exact neighbor IDs\n"
        << "  --query-embedding-cache FILE verified LEANNBC2 query vectors\n"
        << "  --warmup-queries N        unmeasured prefix warmup (default 1)\n"
        << "  --report-k N              also report prefix recall; N <= top-k\n"
        << "  --raw-latencies FILE      per-query CSV latency/candidate records\n"
        << "  --max-queries N            deterministic prefix; 0 means all\n";
}

void command_build(const Arguments & args) {
    const auto prefix = std::filesystem::path(args.require("--index"));
    const auto index_path = leann::index_file_from_prefix(prefix);
    const auto documents_path = leann::documents_file_from_prefix(prefix);
    const auto source_path = std::filesystem::path(args.require("--docs"));
    const bool use_embedding_cache = args.get("--embedder", "hash") == "cache";
    const std::string embedding_cache_argument =
        args.get("--embedding-cache");
    if (use_embedding_cache && embedding_cache_argument.empty()) {
        throw std::invalid_argument(
            "--embedder cache requires --embedding-cache");
    }
    if (!use_embedding_cache && !embedding_cache_argument.empty()) {
        throw std::invalid_argument(
            "--embedding-cache requires --embedder cache");
    }

    std::vector<NamedPath> protected_inputs{
        {"--docs", source_path},
    };
    if (!embedding_cache_argument.empty()) {
        protected_inputs.push_back(
            {"--embedding-cache", embedding_cache_argument});
    }
    const std::string model_argument = args.get("--model");
    if (!model_argument.empty()) {
        protected_inputs.push_back({"--model", model_argument});
    }
    const std::array<NamedPath, 2> outputs{{
        {"index output", index_path},
        {"document-store output", documents_path},
    }};
    require_distinct_output_paths(outputs, protected_inputs);

    std::vector<std::string> documents;
    std::unique_ptr<leann::Embedder> embedder;
    EmbeddingCacheEmbedder * cache_embedder = nullptr;
    if (use_embedding_cache) {
        auto source = read_documents_source(source_path);
        auto cache = std::make_unique<EmbeddingCacheEmbedder>(
            embedding_cache_argument, source);
        documents = std::move(source.documents);
        cache_embedder = cache.get();
        embedder = std::move(cache);
    } else {
        documents = read_lines(source_path);
        embedder = make_embedder(args);
    }
    if (documents.empty()) {
        throw std::runtime_error("document input contains no non-empty lines");
    }
    if (!index_path.parent_path().empty()) {
        std::filesystem::create_directories(index_path.parent_path());
    }

    leann::BuildConfig config;
    config.graph_degree =
        args.unsigned_value("--graph-degree", config.graph_degree);
    config.ef_construction =
        args.unsigned_value("--ef-construction", config.ef_construction);
    config.low_degree =
        args.unsigned_value("--low-degree", config.low_degree);
    config.hub_ratio = args.double_value("--hub-ratio", config.hub_ratio);
    const std::string approximation = args.get("--approx", "pq");
    if (approximation == "pq") {
        config.approximation = leann::ApproximationKind::ProductQuantization;
    } else if (approximation == "simhash") {
        config.approximation = leann::ApproximationKind::SimHash;
    } else {
        throw std::invalid_argument("--approx must be pq or simhash");
    }
    config.pq_subquantizers = args.unsigned_value(
        "--pq-subquantizers", config.pq_subquantizers);
    config.pq_bits = args.unsigned_value("--pq-bits", config.pq_bits);
    config.pq_training_iterations = args.unsigned_value(
        "--pq-iterations", config.pq_training_iterations);
    config.pq_training_samples = args.unsigned_value(
        "--pq-training-samples", config.pq_training_samples);
    config.sketch_bits =
        args.unsigned_value("--sketch-bits", config.sketch_bits);
    config.embedding_batch_size =
        args.unsigned_value("--embedding-batch", config.embedding_batch_size);
    config.random_seed = args.unsigned_value("--seed", config.random_seed);

    const auto started = std::chrono::steady_clock::now();
    leann::Index::build(index_path, documents_path, documents, *embedder, config);
    if (cache_embedder != nullptr) {
        cache_embedder->require_complete();
    }
    const double elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started)
            .count();
    const auto index = leann::Index::load(index_path);
    const auto stats = index.stats();
    std::cout << "built " << stats.nodes << " nodes, " << stats.edges
              << " directed edges in " << std::fixed << std::setprecision(3)
              << elapsed << " s\n"
              << "approximation: " << stats.approximation << '\n'
              << "index: " << index_path << " (" << stats.serialized_bytes
              << " bytes)\n"
              << "documents: " << documents_path << " ("
              << std::filesystem::file_size(documents_path) << " bytes)\n"
              << "dense vectors not persisted: "
              << stats.dense_vector_bytes_avoided << " bytes\n";
}

void command_search(const Arguments & args) {
    const auto prefix = std::filesystem::path(args.require("--index"));
    const auto index_path = leann::index_file_from_prefix(prefix);
    auto index = leann::Index::load(index_path);
    auto documents =
        leann::DocumentStore::open(leann::documents_file_from_prefix(prefix));
    index.validate_document_store(documents);
    auto embedder = make_embedder(args);
    const auto response =
        index.search(args.require("--query"), *embedder, documents,
                     search_config(args));

    for (const auto & result : response.results) {
        std::cout << result.id << '\t' << std::fixed << std::setprecision(6)
                  << result.distance << '\t' << documents.read(result.id)
                  << '\n';
    }
    std::cerr << "search_ms=" << std::fixed << std::setprecision(3)
              << response.metrics.elapsed_ms
              << " exact_recomputations="
              << response.metrics.exact_recomputations
              << " approximate_distances="
              << response.metrics.approximate_distances
              << " upper_layer_hops=" << response.metrics.upper_layer_hops
              << " batches=" << response.metrics.embedding_batches << '\n';
}

void command_stats(const Arguments & args) {
    const auto prefix = std::filesystem::path(args.require("--index"));
    const auto index_path = leann::index_file_from_prefix(prefix);
    const auto index = leann::Index::load(index_path);
    auto documents =
        leann::DocumentStore::open(leann::documents_file_from_prefix(prefix));
    index.validate_document_store(documents);
    const auto stats = index.stats();
    const double overhead =
        documents.raw_bytes() == 0
            ? 0.0
            : 100.0 * static_cast<double>(stats.serialized_bytes) /
                  static_cast<double>(documents.raw_bytes());

    std::cout << "nodes=" << stats.nodes << '\n'
              << "edges=" << stats.edges << '\n'
              << "upper_edges=" << stats.upper_edges << '\n'
              << "max_level=" << stats.max_level << '\n'
              << "dimension=" << stats.dimension << '\n'
              << "approximation=" << stats.approximation << '\n'
              << "sketch_bits=" << stats.sketch_bits << '\n'
              << "approximation_code_bytes="
              << stats.approximation_code_bytes << '\n'
              << "approximation_codebook_bytes="
              << stats.approximation_codebook_bytes << '\n'
              << "max_degree=" << stats.max_degree << '\n'
              << "entry_point=" << stats.entry_point << '\n'
              << "index_bytes=" << stats.serialized_bytes << '\n'
              << "raw_document_bytes=" << documents.raw_bytes() << '\n'
              << "index_over_raw_percent=" << std::fixed
              << std::setprecision(3) << overhead << '\n'
              << "dense_vector_bytes_avoided="
              << stats.dense_vector_bytes_avoided << '\n'
              << "pair_identity=" << stats.pair_identity << '\n'
              << "embedder=" << stats.embedder_fingerprint << '\n';
}

std::vector<std::uint32_t>
exact_top_k(std::span<const float> query,
            const std::vector<leann::Embedding> & corpus,
            std::size_t top_k) {
    std::vector<std::uint32_t> ids(corpus.size());
    std::iota(ids.begin(), ids.end(), 0U);
    const std::size_t count = std::min(top_k, ids.size());
    std::partial_sort(
        ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(count), ids.end(),
        [&](std::uint32_t lhs, std::uint32_t rhs) {
            return leann::cosine_distance(query, corpus[lhs]) <
                   leann::cosine_distance(query, corpus[rhs]);
        });
    ids.resize(count);
    return ids;
}

std::uint64_t parse_decimal(std::string_view token,
                            std::string_view description) {
    std::uint64_t value = 0;
    const auto [end, error] =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc{} || end != token.data() + token.size()) {
        throw std::runtime_error("invalid " + std::string(description) +
                                 " in ground-truth file");
    }
    return value;
}

struct PrecomputedGroundTruth {
    std::vector<std::vector<std::uint32_t>> rows;
};

PrecomputedGroundTruth load_ground_truth(
    const std::filesystem::path & path,
    std::size_t expected_queries,
    std::size_t expected_k,
    std::size_t expected_corpus_size) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open ground-truth file: " +
                                 path.string());
    }

    std::string line;
    while (std::getline(input, line) &&
           line.find_first_not_of(" \t\r") == std::string::npos) {
    }
    if (!input && line.find_first_not_of(" \t\r") == std::string::npos) {
        throw std::runtime_error("ground-truth file is empty");
    }

    std::istringstream header(line);
    std::string magic;
    std::string queries_token;
    std::string k_token;
    std::string corpus_token;
    std::string extra;
    if (!(header >> magic >> queries_token >> k_token >> corpus_token) ||
        (header >> extra) || magic != "LEANN_GT1") {
        throw std::runtime_error(
            "ground-truth header must be: "
            "LEANN_GT1 <query-count> <k> <corpus-count>");
    }
    const std::uint64_t encoded_queries =
        parse_decimal(queries_token, "query count");
    const std::uint64_t encoded_k = parse_decimal(k_token, "k");
    const std::uint64_t encoded_corpus =
        parse_decimal(corpus_token, "corpus count");
    if (encoded_queries != expected_queries) {
        throw std::runtime_error(
            "ground-truth query count does not match selected queries");
    }
    if (encoded_k < expected_k) {
        throw std::runtime_error(
            "ground-truth k does not cover --top-k");
    }
    if (encoded_corpus != expected_corpus_size) {
        throw std::runtime_error(
            "ground-truth corpus count does not match the index");
    }
    if (encoded_k > expected_corpus_size) {
        throw std::runtime_error(
            "ground-truth k exceeds the corpus size");
    }

    PrecomputedGroundTruth result;
    result.rows.reserve(expected_queries);
    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        if (result.rows.size() == expected_queries) {
            throw std::runtime_error(
                "ground-truth file has too many rows");
        }

        std::istringstream row_stream(line);
        std::vector<std::uint32_t> row;
        std::unordered_set<std::uint32_t> unique;
        std::string token;
        while (row_stream >> token) {
            const std::uint64_t value =
                parse_decimal(token, "neighbor ID");
            if (value >= expected_corpus_size ||
                value > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "ground-truth neighbor ID is out of range");
            }
            const auto id = static_cast<std::uint32_t>(value);
            if (!unique.insert(id).second) {
                throw std::runtime_error(
                    "ground-truth row contains a duplicate ID");
            }
            row.push_back(id);
        }
        if (row.size() != encoded_k) {
            throw std::runtime_error(
                "ground-truth row does not contain its encoded k IDs");
        }
        row.resize(expected_k);
        result.rows.push_back(std::move(row));
    }
    if (!input.eof()) {
        throw std::runtime_error("failed to read ground-truth file");
    }
    if (result.rows.size() != expected_queries) {
        throw std::runtime_error(
            "ground-truth row count does not match selected queries");
    }
    return result;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t position = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
    return values[std::min(position, values.size() - 1)];
}

struct RawBenchmarkRow {
    std::size_t query_index = 0;
    double recall = 0.0;
    std::optional<double> report_recall;
    double latency_ms = 0.0;
    std::uint64_t exact_recomputations = 0;
    std::uint64_t approximate_distances = 0;
    std::uint64_t upper_layer_hops = 0;
    std::uint64_t embedding_batches = 0;
    std::optional<double> dense_latency_ms;
    std::optional<double> dense_recall;
    std::vector<std::uint32_t> result_ids;
};

void write_raw_benchmark_rows(
    std::ostream & output,
    std::span<const RawBenchmarkRow> rows,
    std::optional<std::uint32_t> report_k) {
    output << "query_index,recall";
    if (report_k) {
        output << ",recall_at_" << *report_k;
    }
    output
        << ",latency_ms,exact_recomputations,"
           "approximate_distances,upper_layer_hops,embedding_batches,"
           "dense_latency_ms,dense_recall,result_ids\n"
        << std::setprecision(17);
    for (const auto & row : rows) {
        output << row.query_index << ',' << row.recall;
        if (report_k) {
            if (!row.report_recall) {
                throw std::logic_error(
                    "raw benchmark row is missing prefix recall");
            }
            output << ',' << *row.report_recall;
        }
        output << ',' << row.latency_ms << ','
               << row.exact_recomputations << ','
               << row.approximate_distances << ',' << row.upper_layer_hops
               << ',' << row.embedding_batches << ',';
        if (row.dense_latency_ms) {
            output << *row.dense_latency_ms;
        }
        output << ',';
        if (row.dense_recall) {
            output << *row.dense_recall;
        }
        output << ',';
        for (std::size_t result = 0; result < row.result_ids.size();
             ++result) {
            if (result != 0) {
                output << ' ';
            }
            output << row.result_ids[result];
        }
        output << '\n';
    }
    if (!output) {
        throw std::runtime_error("failed to write raw latency rows");
    }
}

void command_bench(const Arguments & args) {
    const auto prefix = std::filesystem::path(args.require("--index"));
    const auto index_path = leann::index_file_from_prefix(prefix);
    const auto documents_path =
        leann::documents_file_from_prefix(prefix);
    const auto queries_path =
        std::filesystem::path(args.require("--queries"));
    const std::string ground_truth_argument = args.get("--ground-truth");
    const std::string ground_truth_cache_argument =
        args.get("--ground-truth-cache");
    const std::string query_cache_argument =
        args.get("--query-embedding-cache");
    const std::string raw_latencies_argument =
        args.get("--raw-latencies");

    std::vector<NamedPath> protected_inputs{
        {"index", index_path},
        {"document store", documents_path},
        {"--queries", queries_path},
    };
    if (!ground_truth_argument.empty()) {
        protected_inputs.push_back(
            {"--ground-truth", ground_truth_argument});
    }
    if (!query_cache_argument.empty()) {
        protected_inputs.push_back(
            {"--query-embedding-cache", query_cache_argument});
    }
    const std::string model_argument = args.get("--model");
    if (!model_argument.empty()) {
        protected_inputs.push_back({"--model", model_argument});
    }
    const std::string build_cache_argument =
        args.get("--embedding-cache");
    if (!build_cache_argument.empty()) {
        protected_inputs.push_back(
            {"--embedding-cache", build_cache_argument});
    }
    std::vector<NamedPath> outputs;
    if (!ground_truth_cache_argument.empty()) {
        outputs.push_back(
            {"--ground-truth-cache", ground_truth_cache_argument});
    }
    if (!raw_latencies_argument.empty()) {
        outputs.push_back({"--raw-latencies", raw_latencies_argument});
    }
    require_distinct_output_paths(outputs, protected_inputs);

    std::unique_ptr<AtomicFileOutput> raw_output;
    if (!raw_latencies_argument.empty()) {
        raw_output = std::make_unique<AtomicFileOutput>(
            raw_latencies_argument, "raw latency");
    }

    auto index = leann::Index::load(index_path);
    auto documents =
        leann::DocumentStore::open(documents_path);
    index.validate_document_store(documents);
    auto embedder = make_embedder(args);
    std::vector<std::string> queries;
    std::optional<std::vector<leann::Embedding>>
        cached_query_embeddings;
    if (!query_cache_argument.empty()) {
        auto source = read_documents_source(queries_path);
        EmbeddingCacheEmbedder query_cache(
            query_cache_argument, source, "--queries");
        const auto index_stats = index.stats();
        if (query_cache.fingerprint() != embedder->fingerprint() ||
            query_cache.fingerprint() !=
                index.embedder_fingerprint()) {
            throw std::invalid_argument(
                "query embedding cache fingerprint does not match the "
                "index and query embedder");
        }
        if (query_cache.dimension() != embedder->dimension() ||
            query_cache.dimension() != index_stats.dimension) {
            throw std::invalid_argument(
                "query embedding cache dimension does not match the "
                "index and query embedder");
        }

        cached_query_embeddings.emplace();
        cached_query_embeddings->reserve(source.documents.size());
        constexpr std::size_t query_cache_batch_size = 256;
        for (std::size_t begin = 0; begin < source.documents.size();
             begin += query_cache_batch_size) {
            const std::size_t end = std::min(
                source.documents.size(),
                begin + query_cache_batch_size);
            auto batch = query_cache.embed(std::span<const std::string>(
                source.documents.data() + begin, end - begin));
            for (auto & embedding : batch) {
                cached_query_embeddings->push_back(
                    std::move(embedding));
            }
        }
        query_cache.require_complete();
        queries = std::move(source.documents);
        std::cerr << "loaded_query_embedding_cache="
                  << query_cache_argument << '\n';
    } else {
        queries = read_lines(queries_path);
    }
    if (queries.empty()) {
        throw std::runtime_error("query file contains no non-empty lines");
    }
    const std::size_t full_query_count = queries.size();
    const std::uint32_t max_queries =
        args.unsigned_value("--max-queries", 0);
    const std::size_t selected_query_count =
        max_queries == 0
            ? full_query_count
            : std::min<std::size_t>(full_query_count, max_queries);
    const auto config = search_config(args);
    if (config.top_k == 0 || config.top_k > documents.size()) {
        throw std::invalid_argument(
            "--top-k must be in [1, corpus size]");
    }
    std::optional<std::uint32_t> report_k;
    if (args.has("--report-k")) {
        const std::uint32_t requested =
            args.unsigned_value("--report-k", 0);
        if (requested == 0 || requested > config.top_k) {
            throw std::invalid_argument(
                "--report-k must be in [1, --top-k]");
        }
        if (requested < config.top_k) {
            report_k = requested;
        }
    }
    const bool run_dense_baseline =
        args.bool_value("--dense-baseline", false);
    std::optional<PrecomputedGroundTruth> precomputed_truth;
    if (!ground_truth_argument.empty()) {
        precomputed_truth = load_ground_truth(
            ground_truth_argument, full_query_count, config.top_k,
            documents.size());
        std::cerr << "loaded_ground_truth=" << ground_truth_argument << '\n';
    }
    queries.resize(selected_query_count);
    if (cached_query_embeddings) {
        cached_query_embeddings->resize(selected_query_count);
    }
    if (precomputed_truth) {
        precomputed_truth->rows.resize(selected_query_count);
    }

    std::vector<leann::Embedding> corpus;
    if (!precomputed_truth || run_dense_baseline) {
        const std::uint32_t batch_size =
            args.unsigned_value("--ground-truth-batch", 64);
        if (batch_size == 0) {
            throw std::invalid_argument(
                "--ground-truth-batch must be positive");
        }

        const std::string & cache_argument =
            ground_truth_cache_argument;
        const auto cache_path = std::filesystem::path(cache_argument);
        if (!cache_argument.empty() &&
            std::filesystem::exists(cache_path)) {
            corpus = load_embedding_cache(
                cache_path, embedder->fingerprint(), documents.size(),
                embedder->dimension());
            std::cerr << "loaded_ground_truth_cache=" << cache_path << '\n';
        } else {
            corpus.reserve(documents.size());
            for (std::size_t begin = 0; begin < documents.size();
                 begin += batch_size) {
                const std::size_t end = std::min<std::size_t>(
                    documents.size(), begin + batch_size);
                std::vector<std::uint32_t> ids(end - begin);
                std::iota(ids.begin(), ids.end(),
                          static_cast<std::uint32_t>(begin));
                auto batch_documents = documents.read_many(ids);
                auto batch_embeddings =
                    embedder->embed(batch_documents);
                if (batch_embeddings.size() !=
                    batch_documents.size()) {
                    throw std::runtime_error(
                        "embedder returned the wrong ground-truth batch "
                        "size");
                }
                for (auto & embedding : batch_embeddings) {
                    if (embedding.size() != embedder->dimension()) {
                        throw std::runtime_error(
                            "embedder returned a wrong-dimension "
                            "ground-truth vector");
                    }
                    leann::normalize(embedding);
                    corpus.push_back(std::move(embedding));
                }
            }
            if (!cache_argument.empty()) {
                write_embedding_cache(
                    cache_path, embedder->fingerprint(), corpus);
                std::cerr << "wrote_ground_truth_cache=" << cache_path
                          << '\n';
            }
        }
    }

    std::unique_ptr<hnswlib::InnerProductSpace> dense_space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> dense_index;
    std::vector<float> dense_vectors;
    double dense_build_seconds = 0.0;
    std::uint64_t dense_index_bytes = 0;
    const std::size_t dense_dimension =
        embedder->dimension() + (embedder->dimension() % 2U == 0U ? 1U : 0U);
    if (run_dense_baseline) {
        const std::uint32_t dense_m =
            args.unsigned_value("--dense-m", 16);
        const std::uint32_t dense_ef_construction =
            args.unsigned_value("--dense-ef-construction", 100);
        const std::uint32_t dense_ef_search =
            args.unsigned_value("--dense-ef-search", config.ef_search);
        if (dense_m < 2 || dense_ef_construction < dense_m ||
            dense_ef_search < config.top_k) {
            throw std::invalid_argument(
                "invalid dense HNSW M/ef construction/ef search");
        }

        dense_vectors.assign(corpus.size() * dense_dimension, 0.0F);
        for (std::size_t node = 0; node < corpus.size(); ++node) {
            std::copy(corpus[node].begin(), corpus[node].end(),
                      dense_vectors.begin() +
                          static_cast<std::ptrdiff_t>(node * dense_dimension));
        }
        dense_space =
            std::make_unique<hnswlib::InnerProductSpace>(dense_dimension);
        const auto dense_started = std::chrono::steady_clock::now();
        dense_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            dense_space.get(), corpus.size(), dense_m, dense_ef_construction,
            42);
        for (std::size_t node = 0; node < corpus.size(); ++node) {
            dense_index->addPoint(
                dense_vectors.data() + node * dense_dimension,
                static_cast<hnswlib::labeltype>(node));
        }
        dense_index->setEf(dense_ef_search);
        dense_build_seconds = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() -
                                  dense_started)
                                  .count();
        dense_index_bytes = dense_index->indexFileSize();
    }

    auto embed_query = [&](std::size_t query_index) {
        if (cached_query_embeddings) {
            return (*cached_query_embeddings)[query_index];
        }
        const std::array<std::string, 1> query_batch{
            queries[query_index]};
        auto embedded = embedder->embed(query_batch);
        if (embedded.size() != 1 ||
            embedded.front().size() != embedder->dimension()) {
            throw std::runtime_error(
                "embedder returned an invalid benchmark query batch");
        }
        leann::normalize(embedded.front());
        return std::move(embedded.front());
    };

    std::vector<float> padded_query(dense_dimension, 0.0F);
    const std::size_t warmup_queries = std::min<std::size_t>(
        args.unsigned_value("--warmup-queries", 1), queries.size());
    for (std::size_t query_index = 0; query_index < warmup_queries;
         ++query_index) {
        const auto embedded_query = embed_query(query_index);
        (void)index.search_embedding(
            embedded_query, *embedder, documents, config);
        if (dense_index) {
            std::fill(padded_query.begin(), padded_query.end(), 0.0F);
            std::copy(embedded_query.begin(), embedded_query.end(),
                      padded_query.begin());
            (void)dense_index->searchKnn(
                padded_query.data(), config.top_k);
        }
    }

    double recall_sum = 0.0;
    double report_recall_sum = 0.0;
    std::uint64_t exact_sum = 0;
    std::uint64_t approximate_sum = 0;
    std::uint64_t upper_hops_sum = 0;
    std::vector<double> latencies;
    latencies.reserve(queries.size());
    double dense_recall_sum = 0.0;
    std::vector<double> dense_latencies;
    dense_latencies.reserve(queries.size());
    std::vector<RawBenchmarkRow> raw_rows;
    if (!raw_latencies_argument.empty()) {
        raw_rows.reserve(queries.size());
    }

    for (std::size_t query_index = 0; query_index < queries.size();
         ++query_index) {
        const auto embedded_query = embed_query(query_index);
        std::vector<std::uint32_t> computed_truth;
        std::span<const std::uint32_t> truth;
        if (precomputed_truth) {
            truth = precomputed_truth->rows[query_index];
        } else {
            computed_truth =
                exact_top_k(embedded_query, corpus, config.top_k);
            truth = computed_truth;
        }
        std::unordered_set<std::uint32_t> truth_set(truth.begin(), truth.end());
        std::unordered_set<std::uint32_t> report_truth_set;
        std::size_t report_truth_size = 0;
        if (report_k) {
            report_truth_size =
                std::min<std::size_t>(*report_k, truth.size());
            report_truth_set.insert(
                truth.begin(),
                truth.begin() +
                    static_cast<std::ptrdiff_t>(report_truth_size));
        }

        std::optional<double> query_dense_latency;
        std::optional<double> query_dense_recall;
        if (dense_index) {
            std::fill(padded_query.begin(), padded_query.end(), 0.0F);
            std::copy(embedded_query.begin(), embedded_query.end(),
                      padded_query.begin());
            const auto dense_started = std::chrono::steady_clock::now();
            auto dense_results =
                dense_index->searchKnn(padded_query.data(), config.top_k);
            query_dense_latency =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - dense_started)
                    .count();
            dense_latencies.push_back(*query_dense_latency);
            std::size_t dense_found = 0;
            while (!dense_results.empty()) {
                const auto label =
                    static_cast<std::uint32_t>(dense_results.top().second);
                dense_found += truth_set.contains(label) ? 1U : 0U;
                dense_results.pop();
            }
            query_dense_recall =
                truth.empty()
                    ? 1.0
                    : static_cast<double>(dense_found) / truth.size();
            dense_recall_sum += *query_dense_recall;
        }

        const auto response = index.search_embedding(
            embedded_query, *embedder, documents, config);
        if (response.results.size() != config.top_k) {
            throw std::runtime_error(
                "native search returned a result count other than --top-k");
        }
        std::vector<std::uint32_t> result_ids;
        result_ids.reserve(response.results.size());
        std::unordered_set<std::uint32_t> unique_result_ids;
        unique_result_ids.reserve(response.results.size());
        for (const auto & result : response.results) {
            if (result.id >= documents.size()) {
                throw std::runtime_error(
                    "native search returned an out-of-range result ID");
            }
            if (!unique_result_ids.insert(result.id).second) {
                throw std::runtime_error(
                    "native search returned a duplicate result ID");
            }
            result_ids.push_back(result.id);
        }

        std::size_t found = 0;
        for (const std::uint32_t result_id : result_ids) {
            found += truth_set.contains(result_id) ? 1U : 0U;
        }
        const double query_recall =
            truth.empty() ? 1.0
                          : static_cast<double>(found) / truth.size();
        recall_sum += query_recall;
        std::optional<double> query_report_recall;
        if (report_k) {
            std::size_t report_found = 0;
            const std::size_t response_prefix =
                std::min<std::size_t>(*report_k,
                                      response.results.size());
            for (std::size_t result = 0; result < response_prefix;
                 ++result) {
                report_found += report_truth_set.contains(
                                    result_ids[result])
                                    ? 1U
                                    : 0U;
            }
            query_report_recall =
                report_truth_size == 0
                    ? 1.0
                    : static_cast<double>(report_found) /
                          report_truth_size;
            report_recall_sum += *query_report_recall;
        }
        exact_sum += response.metrics.exact_recomputations;
        approximate_sum += response.metrics.approximate_distances;
        upper_hops_sum += response.metrics.upper_layer_hops;
        latencies.push_back(response.metrics.elapsed_ms);
        if (!raw_latencies_argument.empty()) {
            raw_rows.push_back(
                RawBenchmarkRow{
                    query_index,
                    query_recall,
                    query_report_recall,
                    response.metrics.elapsed_ms,
                    response.metrics.exact_recomputations,
                    response.metrics.approximate_distances,
                    response.metrics.upper_layer_hops,
                    response.metrics.embedding_batches,
                    query_dense_latency,
                    query_dense_recall,
                    std::move(result_ids),
                });
        }
    }
    if (raw_output) {
        write_raw_benchmark_rows(
            raw_output->stream(), raw_rows, report_k);
        raw_output->publish();
        std::cerr << "wrote_raw_latencies=" << raw_latencies_argument
                  << '\n';
    }

    const double count = static_cast<double>(queries.size());
    std::cout << "queries=" << queries.size() << '\n'
              << "warmup_queries=" << warmup_queries << '\n'
              << "recall_at_" << config.top_k << '=' << std::fixed
              << std::setprecision(6) << recall_sum / count << '\n';
    if (report_k) {
        std::cout << "recall_at_" << *report_k << '='
                  << report_recall_sum / count << '\n';
    }
    std::cout << "latency_ms_mean="
              << std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                     count
              << '\n'
              << "latency_ms_p50=" << percentile(latencies, 0.50) << '\n'
              << "latency_ms_p95=" << percentile(latencies, 0.95) << '\n'
              << "exact_recomputations_mean=" << exact_sum / count << '\n'
              << "approximate_distances_mean=" << approximate_sum / count
              << '\n'
              << "upper_layer_hops_mean=" << upper_hops_sum / count << '\n';
    if (dense_index) {
        std::cout << "dense_hnsw_recall_at_" << config.top_k << '='
                  << dense_recall_sum / count << '\n'
                  << "dense_hnsw_latency_ms_mean="
                  << std::accumulate(dense_latencies.begin(),
                                     dense_latencies.end(), 0.0) /
                         count
                  << '\n'
                  << "dense_hnsw_latency_ms_p50="
                  << percentile(dense_latencies, 0.50) << '\n'
                  << "dense_hnsw_latency_ms_p95="
                  << percentile(dense_latencies, 0.95) << '\n'
                  << "dense_hnsw_build_seconds=" << dense_build_seconds << '\n'
                  << "dense_hnsw_index_bytes=" << dense_index_bytes << '\n';
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
            print_usage(std::cerr);
            return 2;
        }
        const std::string command = argv[1];
        const Arguments args(argc, argv);
        if (command == "help" || args.has("--help")) {
            print_usage(std::cout);
            return 0;
        }
        if (command == "build") {
            command_build(args);
        } else if (command == "search") {
            command_search(args);
        } else if (command == "stats") {
            command_stats(args);
        } else if (command == "bench") {
            command_bench(args);
        } else {
            throw std::invalid_argument("unknown command: " + command);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
