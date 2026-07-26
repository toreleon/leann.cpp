#include "leann/document_store.hpp"
#include "leann/embedder.hpp"
#include "leann/index.hpp"
#include "build_lock.hpp"
#include "checksum.hpp"
#include "text.hpp"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <concepts>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
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
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

#ifndef LEANN_VERSION_STRING
// Kept as a fallback definition rather than a generated header so the
// documented CMake-free `make` path still compiles app/main.cpp.
#define LEANN_VERSION_STRING "0.4.0"
#endif

// Options that never consume the following token. Kept global rather than
// per-command because argv is parsed before the command's specification is
// selected; a boolean flag offered to the wrong command is rejected later by
// validate_arguments.
constexpr std::array<std::string_view, 3> boolean_flag_names{
    "--help",
    "--repair",
    "--force-unlock",
};

[[nodiscard]] bool is_boolean_flag(std::string_view key) {
    return std::find(boolean_flag_names.begin(), boolean_flag_names.end(),
                     key) != boolean_flag_names.end();
}

class Arguments {
  public:
    Arguments(int argc, char ** argv) {
        for (int i = 2; i < argc; ++i) {
            std::string key = argv[i];
            if (key.starts_with("--")) {
                if (is_boolean_flag(key)) {
                    flags_.insert(std::move(key));
                    continue;
                }
                // A boolean flag is never a value. Without this,
                // `--index --repair` binds "--repair" as the index prefix and
                // the flag disappears, so the command runs on a nonsense path
                // with the requested behaviour silently switched off.
                if (i + 1 >= argc || is_boolean_flag(argv[i + 1])) {
                    // Recorded rather than rejected here so validation can
                    // tell an unknown option apart from a known one whose
                    // value was omitted.
                    valueless_.insert(std::move(key));
                    continue;
                }
                std::string value = argv[++i];
                // Every occurrence is kept in argv order for the options that
                // are repeatable; values_ keeps last-wins so every existing
                // accessor and the strict validator are unchanged.
                repeated_[key].push_back(value);
                values_[std::move(key)] = std::move(value);
            } else {
                positional_.push_back(std::move(key));
            }
        }
    }

    [[nodiscard]] bool has(std::string_view key) const {
        return values_.contains(std::string(key)) ||
               flags_.contains(std::string(key));
    }

    [[nodiscard]] const std::unordered_map<std::string, std::string> &
    values() const noexcept {
        return values_;
    }

    [[nodiscard]] const std::unordered_set<std::string> &
    flags() const noexcept {
        return flags_;
    }

    [[nodiscard]] const std::unordered_set<std::string> &
    valueless() const noexcept {
        return valueless_;
    }

    [[nodiscard]] const std::vector<std::string> &
    positional() const noexcept {
        return positional_;
    }

    [[nodiscard]] std::string get(std::string_view key,
                                  std::string fallback = {}) const {
        const auto found = values_.find(std::string(key));
        return found == values_.end() ? std::move(fallback) : found->second;
    }

    // Every occurrence of a repeatable option, in the order given on the
    // command line, so `--card a=1 --card b=2` keeps both.
    [[nodiscard]] std::vector<std::string> all(std::string_view key) const {
        const auto found = repeated_.find(std::string(key));
        return found == repeated_.end() ? std::vector<std::string>{}
                                        : found->second;
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
    std::unordered_map<std::string, std::vector<std::string>> repeated_;
    std::unordered_set<std::string> flags_;
    std::unordered_set<std::string> valueless_;
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
        // --model-source is deliberately not read here: make_embedder is
        // shared by search and bench, whose tables do not declare it. The
        // build path passes it through BuildConfig::model_source instead,
        // which overrides the embedder's own answer.
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

// ---------------------------------------------------------------------------
// Structured output
//
// Hand-written because the repository takes no third-party dependency for
// hashing or serialization. Document text is arbitrary corpus bytes, so the
// writer validates UTF-8 and fails closed rather than emitting a document that
// would silently corrupt a consumer's parse.
// ---------------------------------------------------------------------------

// UTF-8 validation lives in src/text.hpp so the library applies exactly the
// same rule when it validates a descriptor. Two copies could disagree, and
// then a build could publish an index whose `stats --format json` never works.
using leann::detail::is_valid_utf8;

class JsonWriter {
  public:
    explicit JsonWriter(std::ostream & output)
        : output_(output),
          saved_flags_(output.flags()),
          saved_precision_(output.precision()),
          saved_fill_(output.fill()) {}

    // Number and fill formatting are stream-sticky, so they are restored
    // rather than left for whatever writes to the stream next.
    ~JsonWriter() {
        output_.flags(saved_flags_);
        output_.precision(saved_precision_);
        output_.fill(saved_fill_);
    }

    JsonWriter(const JsonWriter &) = delete;
    JsonWriter & operator=(const JsonWriter &) = delete;

    void begin_object() {
        separate();
        output_ << "{";
        push();
    }

    void end_object() {
        pop();
        output_ << "}";
        if (depth_ == 0) {
            output_ << '\n';
        }
    }

    void begin_object_field(std::string_view key) {
        write_key(key);
        output_ << "{";
        push();
    }

    void begin_array(std::string_view key) {
        write_key(key);
        output_ << "[";
        push();
    }

    void end_array() {
        pop();
        output_ << "]";
    }

    void begin_element() {
        separate();
        output_ << "{";
        push();
    }

    void field(std::string_view key, std::string_view value) {
        write_key(key);
        write_string(value, key);
    }

    void field(std::string_view key, std::uint64_t value) {
        write_key(key);
        output_ << value;
    }

    // Constrained to exactly bool. An unconstrained overload would win for a
    // string literal, because const char* -> bool is a standard conversion
    // while const char* -> string_view is a user-defined one, so every
    // literal would silently be emitted as `true`.
    template <typename Boolean>
        requires std::same_as<Boolean, bool>
    void field(std::string_view key, Boolean value) {
        write_key(key);
        output_ << (value ? "true" : "false");
    }

    // Fixed notation with an explicit precision so a consumer sees the same
    // digits the text output shows.
    void field(std::string_view key, double value, int precision) {
        write_key(key);
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "refusing to emit a non-finite value for " + std::string(key));
        }
        output_ << std::fixed << std::setprecision(precision) << value
                << std::defaultfloat;
    }

  private:
    void push() {
        ++depth_;
        first_.push_back(true);
    }

    void pop() {
        if (!first_.empty()) {
            const bool empty = first_.back();
            first_.pop_back();
            --depth_;
            if (!empty) {
                newline();
            }
        }
    }

    void newline() {
        output_ << '\n' << std::string(depth_ * 2U, ' ');
    }

    void separate() {
        if (first_.empty()) {
            return;
        }
        if (!first_.back()) {
            output_ << ',';
        }
        first_.back() = false;
        newline();
    }

    void write_key(std::string_view key) {
        separate();
        write_string(key, key);
        output_ << ": ";
    }

    void write_string(std::string_view value, std::string_view context) {
        if (!is_valid_utf8(value)) {
            throw std::runtime_error(
                "cannot emit JSON for " + std::string(context) +
                ": value is not valid UTF-8; use --format text");
        }
        output_ << '"';
        for (const char raw : value) {
            const auto byte = static_cast<unsigned char>(raw);
            switch (byte) {
            case '"':
                output_ << "\\\"";
                break;
            case '\\':
                output_ << "\\\\";
                break;
            case '\n':
                output_ << "\\n";
                break;
            case '\r':
                output_ << "\\r";
                break;
            case '\t':
                output_ << "\\t";
                break;
            case '\b':
                output_ << "\\b";
                break;
            case '\f':
                output_ << "\\f";
                break;
            default:
                if (byte < 0x20U) {
                    output_ << "\\u" << std::hex << std::setw(4)
                            << std::setfill('0')
                            << static_cast<unsigned int>(byte) << std::dec
                            << std::setfill(' ');
                } else {
                    output_ << raw;
                }
                break;
            }
        }
        output_ << '"';
    }

    std::ostream & output_;
    std::ios_base::fmtflags saved_flags_;
    std::streamsize saved_precision_;
    char saved_fill_;
    std::size_t depth_ = 0;
    std::vector<bool> first_;
};

// Buffers a JSON document and forwards it only once it is complete. Without
// this, a document that fails validation partway through — an invalid UTF-8
// chunk, a non-finite metric — would already have written a truncated object
// onto the caller's parsed stream, with the error text interleaved into it.
class JsonDocument {
  public:
    explicit JsonDocument(std::ostream & destination)
        : destination_(destination), writer_(buffer_) {}

    JsonDocument(const JsonDocument &) = delete;
    JsonDocument & operator=(const JsonDocument &) = delete;

    [[nodiscard]] JsonWriter & writer() noexcept {
        return writer_;
    }

    void commit() {
        destination_ << buffer_.str();
    }

  private:
    std::ostream & destination_;
    std::ostringstream buffer_;
    JsonWriter writer_;
};

enum class OutputFormat {
    Text,
    Json,
};

[[nodiscard]] OutputFormat output_format(const Arguments & args) {
    const std::string value = args.get("--format", "text");
    if (value == "text") {
        return OutputFormat::Text;
    }
    if (value == "json") {
        return OutputFormat::Json;
    }
    throw std::invalid_argument("--format must be text or json");
}

// ---------------------------------------------------------------------------
// Cancellation
//
// The flag lives at file scope because only a signal handler can set it, but
// commands read it through a token they are handed. That keeps the signal
// path out of the tests: a test constructs a token over its own flag and
// drives cancellation deterministically without raising anything.
// ---------------------------------------------------------------------------

std::atomic<bool> interrupt_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "the interrupt flag is read and written from a signal handler");

extern "C" void handle_interrupt(int signal_number) {
    interrupt_requested.store(true, std::memory_order_relaxed);
    // Restore the default disposition so a second signal always terminates.
    // Cancellation is cooperative and cannot interrupt a long uncancellable
    // span, so without this a user who keeps pressing Ctrl-C would have no
    // way out. The residue a forced termination leaves is what `leann doctor`
    // is for.
    std::signal(signal_number, SIG_DFL);
}

class CancellationToken {
  public:
    CancellationToken() = default;

    explicit CancellationToken(const std::atomic<bool> * flag) noexcept
        : flag_(flag) {}

    [[nodiscard]] bool requested() const noexcept {
        return flag_ != nullptr && flag_->load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::function<bool()> predicate() const {
        if (flag_ == nullptr) {
            return {};
        }
        const std::atomic<bool> * flag = flag_;
        return [flag] { return flag->load(std::memory_order_relaxed); };
    }

    void throw_if_requested(std::string_view stage) const {
        if (requested()) {
            throw leann::BuildCancelled("cancelled during " +
                                        std::string(stage));
        }
    }

  private:
    const std::atomic<bool> * flag_ = nullptr;
};

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

[[nodiscard]] bool stderr_is_terminal() noexcept {
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return ::isatty(STDERR_FILENO) != 0;
#endif
}

// Writes a single rewritten line to stderr. Never touches stdout, so it cannot
// contaminate a parsed result stream, and stays silent unless asked.
class ProgressReporter {
  public:
    ProgressReporter(const Arguments & args, std::string_view mode_option) {
        const std::string mode = args.get(mode_option, "auto");
        if (mode == "never") {
            enabled_ = false;
        } else if (mode == "always") {
            enabled_ = true;
        } else if (mode == "auto") {
            enabled_ = stderr_is_terminal();
        } else {
            throw std::invalid_argument(std::string(mode_option) +
                                        " must be auto, always, or never");
        }
        started_ = std::chrono::steady_clock::now();
        last_ = started_;
    }

    [[nodiscard]] bool enabled() const noexcept {
        return enabled_;
    }

    void report(std::string_view stage, std::uint64_t completed,
                std::uint64_t total) {
        if (!enabled_) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const bool stage_changed = stage != stage_;
        const double since_last =
            std::chrono::duration<double>(now - last_).count();
        if (!stage_changed && completed != total && since_last < 0.1) {
            return;
        }
        if (stage_changed) {
            stage_ = stage;
            stage_started_ = now;
        }
        last_ = now;

        std::ostringstream line;
        line << stage;
        if (total > 0) {
            const double fraction =
                static_cast<double>(completed) / static_cast<double>(total);
            line << ' ' << completed << '/' << total << " ("
                 << std::fixed << std::setprecision(1) << (fraction * 100.0)
                 << "%)";
            const double elapsed =
                std::chrono::duration<double>(now - stage_started_).count();
            if (elapsed > 0.5 && completed > 0) {
                const double rate = static_cast<double>(completed) / elapsed;
                line << ' ' << std::setprecision(0) << rate << "/s";
                if (completed < total) {
                    line << " eta "
                         << format_duration(
                                static_cast<double>(total - completed) / rate);
                }
            }
        }
        write_line(line.str());
    }

    // Clears the transient line so the command's real output starts clean.
    void finish() {
        if (!enabled_ || width_ == 0) {
            return;
        }
        std::cerr << '\r' << std::string(width_, ' ') << '\r' << std::flush;
        width_ = 0;
    }

    ~ProgressReporter() {
        try {
            finish();
        } catch (...) {
            // A failure to clear the progress line must not mask the error
            // that is already unwinding.
        }
    }

    ProgressReporter(const ProgressReporter &) = delete;
    ProgressReporter & operator=(const ProgressReporter &) = delete;

  private:
    static std::string format_duration(double seconds) {
        if (!std::isfinite(seconds) || seconds < 0.0) {
            return "?";
        }
        const auto total = static_cast<std::uint64_t>(seconds);
        std::ostringstream text;
        if (total >= 3600U) {
            text << (total / 3600U) << 'h' << ((total % 3600U) / 60U) << 'm';
        } else if (total >= 60U) {
            text << (total / 60U) << 'm' << (total % 60U) << 's';
        } else {
            text << total << 's';
        }
        return text.str();
    }

    void write_line(const std::string & line) {
        std::cerr << '\r' << line;
        if (line.size() < width_) {
            std::cerr << std::string(width_ - line.size(), ' ');
        }
        std::cerr << std::flush;
        width_ = std::max(width_, line.size());
    }

    bool enabled_ = false;
    std::size_t width_ = 0;
    std::string stage_;
    std::chrono::steady_clock::time_point started_{};
    std::chrono::steady_clock::time_point stage_started_{};
    std::chrono::steady_clock::time_point last_{};
};

// ---------------------------------------------------------------------------
// Command specifications
//
// Every option each command reads is listed here exactly once. The tables are
// the single source of truth for three things that used to drift apart: the
// help text, the strict unknown-option check, and the documented surface. An
// option missing from a table is rejected at parse time, so adding a new
// args.get() call without a table entry fails loudly instead of silently.
// ---------------------------------------------------------------------------

struct OptionSpec {
    std::string_view flag;
    std::string_view value_name; // empty for a boolean flag
    std::string_view help;
};

struct OptionGroup {
    std::string_view title;
    std::span<const OptionSpec> options;
};

struct CommandSpec {
    std::string_view name;
    std::string_view synopsis;
    std::string_view summary;
    std::span<const OptionGroup> groups;
    // Name of the single positional argument this command accepts, empty when
    // it accepts none. Positionals stay rejected by default: a stray token is
    // otherwise indistinguishable from an option whose value went missing.
    std::string_view positional_name{};
};

constexpr std::array<OptionSpec, 2> output_options{{
    {"--format", "text|json", "output format (default text)"},
    {"--help", "", "print this help and exit"},
}};

constexpr std::array<OptionSpec, 8> embedding_options{{
    {"--embedder", "hash|llama", "embedding backend (default hash)"},
    {"--model", "FILE", "GGUF model, required by --embedder llama"},
    {"--hash-dim", "N", "hash backend dimension (default 256)"},
    {"--ctx", "N", "llama context tokens (default 512)"},
    {"--batch-tokens", "N", "llama batch tokens (default 2048)"},
    {"--parallel", "N", "llama parallel sequences (default 8)"},
    {"--threads", "N", "llama threads, 0 for the default (default 0)"},
    {"--gpu-layers", "N", "llama GPU layers; part of the fingerprint "
                          "(default 99)"},
}};

constexpr std::array<OptionSpec, 18> build_options{{
    {"--docs", "FILE", "one document per line (required)"},
    {"--index", "PREFIX", "writes PREFIX.leann and PREFIX.docs (required)"},
    {"--embedding-cache", "FILE",
     "LEANNBC2 cache bound to the exact --docs bytes; "
     "requires --embedder cache"},
    {"--graph-degree", "N", "hnswlib M (default 16)"},
    {"--ef-construction", "N", "hnswlib build ef (default 100)"},
    {"--low-degree", "N", "outgoing cap for non-hubs (default 3)"},
    {"--hub-ratio", "F", "fraction of preserved hubs (default 0.02)"},
    {"--approx", "pq|simhash", "approximate distance backend (default pq)"},
    {"--pq-subquantizers", "N",
     "PQ subspaces; must divide dimension (default 64)"},
    {"--pq-bits", "N", "bits per PQ code, 1..8 (default 4)"},
    {"--pq-iterations", "N", "Lloyd iterations (default 10)"},
    {"--pq-training-samples", "N", "maximum training vectors (default 4096)"},
    {"--sketch-bits", "N", "SimHash bits for --approx simhash (default 128)"},
    {"--embedding-batch", "N", "build embedding batch (default 32)"},
    {"--seed", "N", "deterministic build seed (default 42)"},
    {"--progress", "auto|always|never",
     "progress on stderr; auto means when stderr is a terminal "
     "(default auto)"},
    {"--format", "text|json", "output format (default text)"},
    {"--help", "", "print this help and exit"},
}};

constexpr std::array<OptionSpec, 2> search_target_options{{
    {"--index", "PREFIX", "artifact prefix to search (required)"},
    {"--query", "TEXT", "query text (required)"},
}};

constexpr std::array<OptionSpec, 5> search_tuning_options{{
    {"--top-k", "N", "results to return (default 3)"},
    {"--ef-search", "N", "candidates recomputed exactly (default 64)"},
    {"--recompute-batch", "N", "embedder batch during rerank (default 16)"},
    {"--scan-limit", "N",
     "flat ADC below N nodes; 0 forces the graph (default 100000)"},
    {"--rerank-ratio", "F", "approximate shortlist ratio (default 0.25)"},
}};

constexpr std::array<OptionSpec, 1> stats_options{{
    {"--index", "PREFIX", "artifact prefix to describe (required)"},
}};

constexpr std::array<OptionSpec, 18> bench_options{{
    {"--index", "PREFIX", "artifact prefix to benchmark (required)"},
    {"--queries", "FILE", "one query per line (required)"},
    {"--ground-truth", "FILE", "precomputed LEANN_GT1 exact neighbour IDs"},
    {"--ground-truth-cache", "FILE",
     "reuse benchmark-only dense corpus embeddings"},
    {"--ground-truth-batch", "N",
     "corpus embedding batch while computing truth (default 64)"},
    {"--query-embedding-cache", "FILE", "verified LEANNBC2 query vectors"},
    {"--embedding-cache", "FILE",
     "protected from being overwritten by an output path"},
    {"--dense-baseline", "0|1", "build and measure dense HNSW (default 0)"},
    {"--dense-m", "N", "dense baseline M (default 16)"},
    {"--dense-ef-construction", "N",
     "dense baseline build ef (default 100)"},
    {"--dense-ef-search", "N",
     "dense baseline query ef (defaults to --ef-search)"},
    {"--warmup-queries", "N", "unmeasured prefix warmup (default 1)"},
    {"--report-k", "N", "also report prefix recall; N <= --top-k"},
    {"--raw-latencies", "FILE", "per-query CSV latency/candidate records"},
    {"--max-queries", "N", "deterministic prefix; 0 means all (default 0)"},
    {"--progress", "auto|always|never",
     "progress on stderr; auto means when stderr is a terminal "
     "(default auto)"},
    {"--format", "text|json", "output format (default text)"},
    {"--help", "", "print this help and exit"},
}};

constexpr std::array<OptionSpec, 5> doctor_options{{
    {"--index", "PREFIX", "artifact prefix to inspect (required)"},
    {"--repair", "",
     "remove leftovers that are provably safe to remove; never a build lock"},
    {"--force-unlock", "",
     "also remove build locks; only when no build is running"},
    {"--format", "text|json", "output format (default text)"},
    {"--help", "", "print this help and exit"},
}};

// What a published index says about itself. These are recorded verbatim in the
// index header and travel with the artifact, so a downloaded pair names the
// model it needs and is queried the way it was built.
constexpr std::array<OptionSpec, 4> descriptor_options{{
    {"--model-source", "URI",
     "model origin recorded in the index, e.g. hf:OWNER/REPO/FILE"},
    {"--document-prefix", "TEXT",
     "prepended to every chunk before embedding, and on rerank"},
    {"--query-prefix", "TEXT", "prepended to the query at search time"},
    {"--card", "KEY=VALUE", "artifact card entry; repeatable"},
}};

constexpr std::array<OptionSpec, 5> pull_options{{
    {"--manifest", "FILE",
     "LEANNMF1 manifest; turns the plan into exact digests"},
    {"--revision", "REV", "repository revision (default main)"},
    {"--dest", "DIR", "directory the plan downloads into (default .)"},
    {"--format", "text|json", "output format (default text)"},
    {"--help", "", "print this help and exit"},
}};

constexpr std::array<OptionSpec, 4> verify_options{{
    {"--index", "PREFIX", "artifact prefix to verify (required)"},
    {"--manifest", "FILE", "LEANNMF1 manifest to check sizes and digests"},
    {"--format", "text|json", "output format (default text)"},
    {"--help", "", "print this help and exit"},
}};

constexpr std::array<OptionGroup, 3> build_groups{{
    {"Build options", build_options},
    {"Embedding options (build also accepts --embedder cache)",
     embedding_options},
    {"Artifact descriptor options", descriptor_options},
}};

constexpr std::array<OptionGroup, 4> search_groups{{
    {"Search options", search_target_options},
    {"Search tuning", search_tuning_options},
    {"Embedding options", embedding_options},
    {"Output options", output_options},
}};

constexpr std::array<OptionGroup, 2> stats_groups{{
    {"Options", stats_options},
    {"Output options", output_options},
}};

constexpr std::array<OptionGroup, 3> bench_groups{{
    {"Benchmark options", bench_options},
    {"Search tuning", search_tuning_options},
    {"Embedding options", embedding_options},
}};

constexpr std::array<OptionGroup, 1> doctor_groups{{
    {"Options", doctor_options},
}};

constexpr std::array<OptionGroup, 1> pull_groups{{
    {"Options", pull_options},
}};

constexpr std::array<OptionGroup, 1> verify_groups{{
    {"Options", verify_options},
}};

constexpr std::array<CommandSpec, 7> command_specs{{
    {"build", "leann build --docs FILE --index PREFIX [options]",
     "Embed a corpus, prune the graph, and publish the artifact pair.",
     build_groups},
    {"search", "leann search --index PREFIX --query TEXT [options]",
     "Retrieve the exact top-k for one query.", search_groups},
    {"stats", "leann stats --index PREFIX [options]",
     "Describe a published artifact pair.", stats_groups},
    {"bench", "leann bench --index PREFIX --queries FILE [options]",
     "Measure recall, latency, and recomputation over a query file.",
     bench_groups},
    {"doctor", "leann doctor --index PREFIX [options]",
     "Report artifact health and leftovers from an interrupted build.",
     doctor_groups},
    // Appended rather than inserted: suggest_option reports the first other
    // command owning a flag, and the pinned "--top-k belongs to leann search"
    // message depends on search still preceding bench.
    {"pull", "leann pull hf:OWNER/NAME [options]",
     "Print the exact commands that fetch a published index. Opens no socket.",
     pull_groups, "hf:OWNER/NAME"},
    {"verify", "leann verify --index PREFIX [options]",
     "Check that an artifact pair loads and matches its manifest.",
     verify_groups},
}};

[[nodiscard]] const CommandSpec * find_command(std::string_view name) {
    const auto found = std::find_if(
        command_specs.begin(), command_specs.end(),
        [&](const CommandSpec & spec) { return spec.name == name; });
    return found == command_specs.end() ? nullptr : &*found;
}

[[nodiscard]] const OptionSpec * find_option(const CommandSpec & command,
                                             std::string_view flag) {
    for (const OptionGroup & group : command.groups) {
        for (const OptionSpec & option : group.options) {
            if (option.flag == flag) {
                return &option;
            }
        }
    }
    return nullptr;
}

// Bounded Levenshtein distance, used only to suggest a correction for a
// mistyped option. Returns limit + 1 once the distance is known to exceed it.
[[nodiscard]] std::size_t edit_distance(std::string_view lhs,
                                        std::string_view rhs,
                                        std::size_t limit) {
    if (lhs.size() > rhs.size()) {
        std::swap(lhs, rhs);
    }
    if (rhs.size() - lhs.size() > limit) {
        return limit + 1;
    }
    std::vector<std::size_t> previous(lhs.size() + 1);
    std::vector<std::size_t> current(lhs.size() + 1);
    std::iota(previous.begin(), previous.end(), std::size_t{0});
    for (std::size_t j = 1; j <= rhs.size(); ++j) {
        current[0] = j;
        std::size_t best = current[0];
        for (std::size_t i = 1; i <= lhs.size(); ++i) {
            const std::size_t substitution =
                previous[i - 1] + (lhs[i - 1] == rhs[j - 1] ? 0U : 1U);
            current[i] = std::min({current[i - 1] + 1, previous[i] + 1,
                                   substitution});
            best = std::min(best, current[i]);
        }
        if (best > limit) {
            return limit + 1;
        }
        previous.swap(current);
    }
    return previous[lhs.size()];
}

// The nearest option of any command, so a flag offered to the wrong command
// still gets a useful hint instead of a bare rejection.
[[nodiscard]] std::string suggest_option(const CommandSpec & command,
                                         std::string_view flag) {
    std::string_view best;
    std::size_t best_distance = 3; // reject suggestions that are not close
    for (const OptionGroup & group : command.groups) {
        for (const OptionSpec & option : group.options) {
            const std::size_t distance =
                edit_distance(flag, option.flag, best_distance);
            if (distance < best_distance) {
                best_distance = distance;
                best = option.flag;
            }
        }
    }
    if (!best.empty()) {
        return "; did you mean " + std::string(best) + "?";
    }
    for (const CommandSpec & other : command_specs) {
        if (other.name == command.name) {
            continue;
        }
        if (find_option(other, flag) != nullptr) {
            return "; " + std::string(flag) + " belongs to leann " +
                   std::string(other.name);
        }
    }
    return {};
}

// Rejects anything the command does not read. Without this a mistyped option
// is silently ignored and the command answers with its defaults, which reads
// as a correct answer to a question that was never asked.
void validate_arguments(const CommandSpec & command,
                        const Arguments & arguments) {
    const auto reject_unknown = [&](const std::string & flag) {
        if (find_option(command, flag) == nullptr) {
            throw std::invalid_argument(
                "unknown option for leann " + std::string(command.name) +
                ": " + flag + suggest_option(command, flag));
        }
    };
    for (const auto & [flag, value] : arguments.values()) {
        reject_unknown(flag);
        (void)value;
    }
    for (const std::string & flag : arguments.flags()) {
        reject_unknown(flag);
    }
    for (const std::string & flag : arguments.valueless()) {
        reject_unknown(flag);
        throw std::invalid_argument("missing value for " + flag);
    }
    if (command.positional_name.empty()) {
        if (!arguments.positional().empty()) {
            throw std::invalid_argument(
                "unexpected argument for leann " + std::string(command.name) +
                ": " + arguments.positional().front());
        }
    } else if (arguments.positional().size() > 1) {
        // A second positional is almost always a quoting mistake, and taking
        // the first silently would run against the wrong repository.
        throw std::invalid_argument(
            "leann " + std::string(command.name) + " takes one " +
            std::string(command.positional_name) + " argument, got " +
            std::to_string(arguments.positional().size()));
    }
}

void print_command_help(std::ostream & output, const CommandSpec & command) {
    output << command.summary << "\n\nUsage:\n  " << command.synopsis << '\n';
    std::size_t width = 0;
    for (const OptionGroup & group : command.groups) {
        for (const OptionSpec & option : group.options) {
            const std::size_t length =
                option.flag.size() +
                (option.value_name.empty() ? 0U : option.value_name.size() + 1U);
            width = std::max(width, length);
        }
    }
    for (const OptionGroup & group : command.groups) {
        output << '\n' << group.title << ":\n";
        for (const OptionSpec & option : group.options) {
            std::string term(option.flag);
            if (!option.value_name.empty()) {
                term += ' ';
                term += option.value_name;
            }
            output << "  " << std::left << std::setw(static_cast<int>(width))
                   << term << std::right << "  " << option.help << '\n';
        }
    }
}

[[nodiscard]] std::string suggest_command(std::string_view name) {
    std::string_view best;
    std::size_t best_distance = 3;
    for (const CommandSpec & command : command_specs) {
        const std::size_t distance =
            edit_distance(name, command.name, best_distance);
        if (distance < best_distance) {
            best_distance = distance;
            best = command.name;
        }
    }
    if (best.empty()) {
        return "; run leann --help for the command list";
    }
    return "; did you mean leann " + std::string(best) + "?";
}

void print_usage(std::ostream & output) {
    output << "leann.cpp " << LEANN_VERSION_STRING
           << " — native low-storage vector search\n\nUsage:\n";
    std::size_t width = 0;
    for (const CommandSpec & command : command_specs) {
        width = std::max(width, command.name.size());
    }
    for (const CommandSpec & command : command_specs) {
        output << "  leann " << std::left << std::setw(static_cast<int>(width))
               << command.name << std::right << "  " << command.summary
               << '\n';
    }
    output << "\n  leann <command> --help   options for one command\n"
           << "  leann --version          print the version and exit\n";
}

// Splits each --card KEY=VALUE on its first '=' and rejects anything that
// would make the card ambiguous downstream. Control characters are refused
// because `stats` writes the card as one line per entry and the manifest is
// tab-separated; invalid UTF-8 is refused here rather than at --format json,
// which would otherwise throw only after the build had already committed.
[[nodiscard]] leann::ArtifactCard parse_card_entries(const Arguments & args) {
    leann::ArtifactCard card;
    for (const std::string & entry : args.all("--card")) {
        const std::size_t separator = entry.find('=');
        if (separator == std::string::npos) {
            throw std::invalid_argument("--card must be KEY=VALUE, got: " +
                                        entry);
        }
        std::string key = entry.substr(0, separator);
        std::string value = entry.substr(separator + 1U);
        if (key.empty()) {
            throw std::invalid_argument("--card key must not be empty: " +
                                        entry);
        }
        // Index::build enforces the same rules; rejecting here means the
        // message names the offending flag instead of the header field.
        for (const std::string_view field : {std::string_view(key),
                                             std::string_view(value)}) {
            if (leann::detail::has_control_characters(field)) {
                throw std::invalid_argument(
                    "--card must not contain control characters: " + key);
            }
            if (!is_valid_utf8(field)) {
                throw std::invalid_argument(
                    "--card must be valid UTF-8: " + key);
            }
        }
        card.emplace_back(std::move(key), std::move(value));
    }
    return card;
}

// Descriptor inputs shared by build. Kept together so the validation order is
// one place rather than scattered through command_build.
struct DescriptorOptions {
    std::string model_source;
    std::string document_prefix;
    std::string query_prefix;
    leann::ArtifactCard card;
};

[[nodiscard]] DescriptorOptions descriptor_options_from(
    const Arguments & args, bool uses_embedding_cache) {
    DescriptorOptions options;
    options.model_source = args.get("--model-source");
    options.document_prefix = args.get("--document-prefix");
    options.query_prefix = args.get("--query-prefix");
    options.card = parse_card_entries(args);
    // A LEANNBC2 cache holds vectors that were already computed over whatever
    // text its producer chose. leann.cpp cannot apply a prefix to a finished
    // vector, and the cache header has no prefix field to check against, so
    // accepting this silently would record a prefix the vectors do not have.
    //
    // The advice matters: baking the prefix into the *cache* alone would still
    // be wrong, because rerank never reads the cache — it re-embeds live from
    // the document store and applies the index's recorded prefix, which would
    // be empty. The prefix has to be in the corpus text itself, so that both
    // the cached build vectors and every later recomputation see it.
    if (uses_embedding_cache && !options.document_prefix.empty()) {
        throw std::invalid_argument(
            "--document-prefix cannot be applied to --embedder cache "
            "vectors, which were already computed; put the prefix in the "
            "--docs text itself and leave --document-prefix unset, so that "
            "rerank recomputes the same text the cache was built from");
    }
    return options;
}

void command_build(const Arguments & args,
                   CancellationToken cancellation = {}) {
    const auto format = output_format(args);
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
    auto descriptor = descriptor_options_from(args, use_embedding_cache);
    config.model_source = std::move(descriptor.model_source);
    config.document_prefix = std::move(descriptor.document_prefix);
    config.query_prefix = std::move(descriptor.query_prefix);
    config.card = std::move(descriptor.card);

    ProgressReporter progress(args, "--progress");
    if (progress.enabled()) {
        config.report_progress = [&progress](const leann::BuildProgress & p) {
            progress.report(p.stage, p.completed, p.total);
        };
    }
    config.should_cancel = cancellation.predicate();

    const auto started = std::chrono::steady_clock::now();
    leann::Index::build(index_path, documents_path, documents, *embedder, config);
    if (cache_embedder != nullptr) {
        cache_embedder->require_complete();
    }
    const double elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started)
            .count();
    progress.finish();
    const auto index = leann::Index::load(index_path);
    const auto stats = index.stats();
    const auto documents_bytes =
        static_cast<std::uint64_t>(std::filesystem::file_size(documents_path));

    if (format == OutputFormat::Json) {
        JsonDocument document(std::cout);
        JsonWriter & json = document.writer();
        json.begin_object();
        json.field("nodes", stats.nodes);
        json.field("edges", stats.edges);
        json.field("build_seconds", elapsed, 3);
        json.field("approximation", stats.approximation);
        json.field("index_path", index_path.string());
        json.field("index_bytes", stats.serialized_bytes);
        json.field("documents_path", documents_path.string());
        json.field("documents_bytes", documents_bytes);
        json.field("dense_vector_bytes_avoided",
                   stats.dense_vector_bytes_avoided);
        json.field("pair_identity", stats.pair_identity);
        json.field("embedder", stats.embedder_fingerprint);
        json.end_object();
        document.commit();
        return;
    }
    std::cout << "built " << stats.nodes << " nodes, " << stats.edges
              << " directed edges in " << std::fixed << std::setprecision(3)
              << elapsed << " s\n"
              << "approximation: " << stats.approximation << '\n'
              << "index: " << index_path.string() << " ("
              << stats.serialized_bytes << " bytes)\n"
              << "documents: " << documents_path.string() << " ("
              << documents_bytes << " bytes)\n"
              << "dense vectors not persisted: "
              << stats.dense_vector_bytes_avoided << " bytes\n";
}

void command_search(const Arguments & args) {
    const auto format = output_format(args);
    const auto prefix = std::filesystem::path(args.require("--index"));
    const auto index_path = leann::index_file_from_prefix(prefix);
    auto index = leann::Index::load(index_path);
    auto documents =
        leann::DocumentStore::open(leann::documents_file_from_prefix(prefix));
    index.validate_document_store(documents);
    auto embedder = make_embedder(args);
    const std::string query = args.require("--query");
    const auto response =
        index.search(query, *embedder, documents, search_config(args));

    if (format == OutputFormat::Json) {
        // Everything on stdout as one object: the point of the JSON mode is a
        // single document a caller can parse without also reading stderr.
        JsonDocument document(std::cout);
        JsonWriter & json = document.writer();
        json.begin_object();
        json.field("query", query);
        json.begin_array("results");
        for (const auto & result : response.results) {
            json.begin_element();
            json.field("id", static_cast<std::uint64_t>(result.id));
            json.field("distance", static_cast<double>(result.distance), 6);
            json.field("document", documents.read(result.id));
            json.end_object();
        }
        json.end_array();
        json.begin_object_field("metrics");
        json.field("search_ms", response.metrics.elapsed_ms, 3);
        json.field("exact_recomputations",
                   response.metrics.exact_recomputations);
        json.field("approximate_distances",
                   response.metrics.approximate_distances);
        json.field("expanded_nodes", response.metrics.expanded_nodes);
        json.field("upper_layer_hops", response.metrics.upper_layer_hops);
        json.field("embedding_batches", response.metrics.embedding_batches);
        json.end_object();
        json.end_object();
        document.commit();
        return;
    }

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
    const auto format = output_format(args);
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

    if (format == OutputFormat::Json) {
        // Same keys as the text form, in the same order.
        JsonDocument document(std::cout);
        JsonWriter & json = document.writer();
        json.begin_object();
        json.field("nodes", stats.nodes);
        json.field("edges", stats.edges);
        json.field("upper_edges", stats.upper_edges);
        json.field("max_level", static_cast<std::uint64_t>(stats.max_level));
        json.field("dimension", static_cast<std::uint64_t>(stats.dimension));
        json.field("approximation", stats.approximation);
        json.field("sketch_bits",
                   static_cast<std::uint64_t>(stats.sketch_bits));
        json.field("approximation_code_bytes", stats.approximation_code_bytes);
        json.field("approximation_codebook_bytes",
                   stats.approximation_codebook_bytes);
        json.field("max_degree", static_cast<std::uint64_t>(stats.max_degree));
        json.field("entry_point",
                   static_cast<std::uint64_t>(stats.entry_point));
        json.field("index_bytes", stats.serialized_bytes);
        json.field("raw_document_bytes", documents.raw_bytes());
        json.field("index_over_raw_percent", overhead, 3);
        json.field("dense_vector_bytes_avoided",
                   stats.dense_vector_bytes_avoided);
        json.field("pair_identity", stats.pair_identity);
        json.field("embedder", stats.embedder_fingerprint);
        json.field("model_source", stats.model_source);
        json.field("model_sha256", stats.model_sha256);
        json.field("model_bytes", stats.model_bytes);
        json.field("pooling_type",
                   static_cast<std::uint64_t>(stats.pooling_type));
        json.field("context_tokens",
                   static_cast<std::uint64_t>(stats.context_tokens));
        json.field("document_prefix", stats.document_prefix);
        json.field("query_prefix", stats.query_prefix);
        // Nested rather than flattened: a publisher-chosen key must not be
        // able to appear at the top level, where a duplicate of an existing
        // field would win last-write in every JSON reader.
        json.begin_object_field("card");
        for (const auto & [key, value] : stats.card) {
            json.field(key, value);
        }
        json.end_object();
        json.end_object();
        document.commit();
        return;
    }

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
              << "embedder=" << stats.embedder_fingerprint << '\n'
              << "model_source=" << stats.model_source << '\n'
              << "model_sha256=" << stats.model_sha256 << '\n'
              << "model_bytes=" << stats.model_bytes << '\n'
              << "pooling_type=" << stats.pooling_type << '\n'
              << "context_tokens=" << stats.context_tokens << '\n'
              << "document_prefix=" << stats.document_prefix << '\n'
              << "query_prefix=" << stats.query_prefix << '\n';
    // Card keys are namespaced and validated to hold no control characters,
    // so an entry cannot forge a line or shadow a key a consumer reads.
    for (const auto & [key, value] : stats.card) {
        std::cout << "card_" << key << '=' << value << '\n';
    }
}

// ---------------------------------------------------------------------------
// doctor
//
// Reports what is actually on disk next to an artifact prefix and, on request,
// removes only what it can prove is safe to remove. Three facts shape it:
//
//   * A build lock carries no proof of liveness. A descriptor is written
//     inside it, but a pid can be recycled and a shared filesystem can be
//     mounted on another host, so liveness is a probe and is reported as such.
//     --repair never removes a lock; that needs the explicit --force-unlock.
//   * A `.bak.*` index can be the only surviving index. When a publication
//     rolls back and the document restore fails, the previous index is
//     deliberately left in its backup rather than reactivated against the
//     wrong chunks. So a backup is removable only once the live pair itself
//     loads and validates.
//   * A `.tmp.*` file is an abandoned build's scratch artifact, but only when
//     no lock is present for the same target; otherwise a build may be
//     writing it right now.
// ---------------------------------------------------------------------------

enum class LeftoverKind {
    Temporary,
    Backup,
};

struct Leftover {
    LeftoverKind kind;
    std::filesystem::path path;
    std::uint64_t bytes = 0;
};

struct LockReport {
    std::filesystem::path path;
    bool present = false;
    std::optional<leann::detail::LockOwner> owner;
    leann::detail::OwnerLiveness liveness =
        leann::detail::OwnerLiveness::Unknown;
};

struct DoctorReport {
    std::filesystem::path index_path;
    std::filesystem::path documents_path;
    bool index_present = false;
    bool documents_present = false;
    bool pair_valid = false;
    std::string pair_error;
    std::string pair_identity;
    std::vector<LockReport> locks;
    std::vector<Leftover> leftovers;
    std::vector<std::string> removed;
    std::vector<std::string> retained;
};

[[nodiscard]] std::string_view
liveness_text(leann::detail::OwnerLiveness liveness) {
    switch (liveness) {
    case leann::detail::OwnerLiveness::Running:
        return "running";
    case leann::detail::OwnerLiveness::Absent:
        return "absent";
    case leann::detail::OwnerLiveness::Unknown:
        break;
    }
    return "unknown";
}

[[nodiscard]] LockReport inspect_lock(const std::filesystem::path & target) {
    LockReport report;
    report.path = leann::detail::lock_path_for(target);
    std::error_code error;
    report.present = std::filesystem::is_directory(report.path, error);
    if (!report.present) {
        return report;
    }
    report.owner = leann::detail::read_lock_owner(report.path);
    report.liveness = leann::detail::owner_liveness(report.owner);
    return report;
}

// Scans the prefix's directory for artifacts adjacent to either member of the
// pair. Matching is on the "<name>.tmp." / "<name>.bak." prefixes the builder
// actually produces, not on a trailing extension.
[[nodiscard]] std::vector<Leftover>
collect_leftovers(const std::filesystem::path & index_path,
                  const std::filesystem::path & documents_path) {
    std::vector<Leftover> leftovers;
    auto directory = index_path.parent_path();
    if (directory.empty()) {
        directory = std::filesystem::path(".");
    }
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) {
        return leftovers;
    }
    const std::array<std::string, 2> stems{
        index_path.filename().string(),
        documents_path.filename().string(),
    };
    // An unreadable directory must not be reported as "no leftovers"; that
    // would be a clean bill of health the tool has no basis for.
    std::filesystem::directory_iterator entries(directory, error);
    if (error) {
        throw std::runtime_error("cannot scan '" + directory.string() +
                                 "' for build leftovers: " + error.message());
    }
    for (const auto & entry : entries) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        for (const std::string & stem : stems) {
            const bool temporary = name.starts_with(stem + ".tmp.");
            const bool backup = name.starts_with(stem + ".bak.");
            if (!temporary && !backup) {
                continue;
            }
            std::error_code size_error;
            const auto size = std::filesystem::file_size(entry.path(),
                                                         size_error);
            leftovers.push_back(
                {temporary ? LeftoverKind::Temporary : LeftoverKind::Backup,
                 entry.path(),
                 size_error ? 0U : static_cast<std::uint64_t>(size)});
            break;
        }
    }
    std::sort(leftovers.begin(), leftovers.end(),
              [](const Leftover & lhs, const Leftover & rhs) {
                  return lhs.path.string() < rhs.path.string();
              });
    return leftovers;
}

void command_doctor(const Arguments & args) {
    const auto format = output_format(args);
    const bool repair = args.has("--repair");
    const bool force_unlock = args.has("--force-unlock");
    const auto prefix = std::filesystem::path(args.require("--index"));

    DoctorReport report;
    report.index_path = leann::index_file_from_prefix(prefix);
    report.documents_path = leann::documents_file_from_prefix(prefix);

    std::error_code error;
    report.index_present =
        std::filesystem::is_regular_file(report.index_path, error);
    report.documents_present =
        std::filesystem::is_regular_file(report.documents_path, error);

    // The pair check is the whole basis for deciding a backup is disposable,
    // so it is the full load path: checksum, open, and cross-validate.
    if (report.index_present && report.documents_present) {
        try {
            const auto index = leann::Index::load(report.index_path);
            auto documents =
                leann::DocumentStore::open(report.documents_path);
            index.validate_document_store(documents);
            report.pair_valid = true;
            report.pair_identity = index.stats().pair_identity;
        } catch (const std::exception & failure) {
            report.pair_error = failure.what();
        }
    } else if (report.index_present != report.documents_present) {
        report.pair_error =
            report.index_present
                ? "document store is missing; the index alone cannot be used"
                : "index is missing; a build may have been interrupted before "
                  "its commit";
    } else {
        report.pair_error = "no artifact pair at this prefix";
    }

    report.locks.push_back(inspect_lock(report.index_path));
    report.locks.push_back(inspect_lock(report.documents_path));
    report.leftovers =
        collect_leftovers(report.index_path, report.documents_path);

    const bool any_lock_present =
        std::ranges::any_of(report.locks,
                            [](const LockReport & lock) {
                                return lock.present;
                            });
    const bool any_owner_running = std::ranges::any_of(
        report.locks, [](const LockReport & lock) {
            return lock.present &&
                   lock.liveness == leann::detail::OwnerLiveness::Running;
        });

    const auto remove_path = [&](const std::filesystem::path & path,
                                 const std::string & reason) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        if (remove_error) {
            report.retained.push_back(path.string() + ": " +
                                      remove_error.message());
        } else {
            report.removed.push_back(path.string() + " (" + reason + ")");
        }
    };

    // Every precondition that can refuse the request is checked before
    // anything is deleted, so no removal can be performed and then discarded
    // by a later throw that prevents the report from being printed.
    if (force_unlock && any_owner_running) {
        throw std::runtime_error(
            "refusing --force-unlock: a lock names a process that is "
            "running on this host");
    }

    if (repair) {
        for (const Leftover & leftover : report.leftovers) {
            // A lock means a build may be mid-transaction. Its temporaries
            // are still being written, and — because publish_artifact_pair
            // parks the previous pair in .bak.* for the whole transaction and
            // restores from exactly those files on rollback — its backups are
            // load-bearing. Neither is ours to remove, whatever the live pair
            // looked like a moment ago.
            if (any_lock_present) {
                report.retained.push_back(
                    leftover.path.string() +
                    ": a build lock is present, so a build may still be "
                    "using this file");
                continue;
            }
            if (leftover.kind == LeftoverKind::Temporary) {
                remove_path(leftover.path, "abandoned build temporary");
            } else {
                if (!report.pair_valid) {
                    report.retained.push_back(
                        leftover.path.string() +
                        ": the live pair does not validate, so this backup "
                        "may be the only usable artifact");
                    continue;
                }
                remove_path(leftover.path,
                            "superseded backup; live pair validates");
            }
        }
    }

    if (force_unlock) {
        for (const LockReport & lock : report.locks) {
            if (!lock.present) {
                continue;
            }
            leann::detail::remove_lock_owner(lock.path);
            remove_path(lock.path, "build lock removed by --force-unlock");
        }
    }

    if (format == OutputFormat::Json) {
        JsonDocument document(std::cout);
        JsonWriter & json = document.writer();
        json.begin_object();
        json.field("index_path", report.index_path.string());
        json.field("documents_path", report.documents_path.string());
        json.field("index_present", report.index_present);
        json.field("documents_present", report.documents_present);
        json.field("pair_valid", report.pair_valid);
        json.field("pair_identity", report.pair_identity);
        json.field("pair_error", report.pair_error);
        json.begin_array("locks");
        for (const LockReport & lock : report.locks) {
            json.begin_element();
            json.field("path", lock.path.string());
            json.field("present", lock.present);
            json.field("owner_pid",
                       lock.owner ? lock.owner->pid : std::uint64_t{0});
            json.field("owner_host", lock.owner ? lock.owner->host : "");
            json.field("owner_started_unix",
                       lock.owner ? lock.owner->started_unix
                                  : std::uint64_t{0});
            json.field("owner_liveness", liveness_text(lock.liveness));
            json.end_object();
        }
        json.end_array();
        json.begin_array("leftovers");
        for (const Leftover & leftover : report.leftovers) {
            json.begin_element();
            json.field("path", leftover.path.string());
            json.field("kind", leftover.kind == LeftoverKind::Temporary
                                   ? "temporary"
                                   : "backup");
            json.field("bytes", leftover.bytes);
            json.end_object();
        }
        json.end_array();
        json.begin_array("removed");
        for (const std::string & entry : report.removed) {
            json.begin_element();
            json.field("detail", entry);
            json.end_object();
        }
        json.end_array();
        json.begin_array("retained");
        for (const std::string & entry : report.retained) {
            json.begin_element();
            json.field("detail", entry);
            json.end_object();
        }
        json.end_array();
        json.end_object();
        document.commit();
        return;
    }

    std::cout << "index: " << report.index_path.string() << " ("
              << (report.index_present ? "present" : "missing") << ")\n"
              << "documents: " << report.documents_path.string() << " ("
              << (report.documents_present ? "present" : "missing") << ")\n"
              << "pair: " << (report.pair_valid ? "valid" : "unusable");
    if (!report.pair_valid) {
        std::cout << " — " << report.pair_error;
    } else {
        std::cout << " (identity " << report.pair_identity << ")";
    }
    std::cout << '\n';

    for (const LockReport & lock : report.locks) {
        if (!lock.present) {
            continue;
        }
        std::cout << "lock: " << lock.path.string() << " held by ";
        if (lock.owner) {
            std::cout << "pid " << lock.owner->pid << " on host "
                      << (lock.owner->host.empty() ? "?" : lock.owner->host)
                      << ", owner " << liveness_text(lock.liveness);
        } else {
            std::cout << "an unrecorded owner, liveness unknown";
        }
        std::cout << '\n';
    }
    if (!any_lock_present) {
        std::cout << "lock: none\n";
    }

    for (const Leftover & leftover : report.leftovers) {
        std::cout << (leftover.kind == LeftoverKind::Temporary ? "temporary: "
                                                               : "backup: ")
                  << leftover.path.string() << " (" << leftover.bytes
                  << " bytes)\n";
    }
    if (report.leftovers.empty()) {
        std::cout << "leftovers: none\n";
    }
    for (const std::string & entry : report.removed) {
        std::cout << "removed: " << entry << '\n';
    }
    for (const std::string & entry : report.retained) {
        std::cout << "retained: " << entry << '\n';
    }
    if (!repair && !report.leftovers.empty()) {
        std::cout << "hint: rerun with --repair to remove what is provably "
                     "safe to remove\n";
    }
}

// ---------------------------------------------------------------------------
// LEANNMF1 manifests, pull, and verify
//
// A published index is two large files plus the model that produced them.
// `pull` turns a repository name into the exact commands that fetch them and
// the digests they must have; it deliberately opens no socket, so the binary
// gains no HTTP client, no TLS surface, and no new dependency. `verify` is the
// other half: it checks what actually landed on disk.
//
// The manifest is a tab-separated text format rather than JSON because the
// repository takes no JSON parser dependency, and because a line-oriented
// grammar is one that can be rejected precisely. Every deviation is an error:
// there is no lenient mode, and an unknown key is a failure rather than
// something to skip, so a newer manifest is never half-understood by an older
// binary.
// ---------------------------------------------------------------------------

constexpr std::string_view manifest_magic = "LEANNMF1";

struct ManifestFile {
    std::string name;
    std::uint64_t bytes = 0;
    leann::detail::Sha256Digest digest{};
};

struct Manifest {
    std::string repo;
    std::string repo_type = "model";
    std::string revision = "main";
    std::string prefix;
    std::vector<ManifestFile> files;
    bool has_model = false;
    ManifestFile model;
    std::string model_source;
};

[[nodiscard]] std::vector<std::string_view> split_tabs(std::string_view line) {
    std::vector<std::string_view> fields;
    std::size_t begin = 0;
    while (true) {
        const std::size_t tab = line.find('\t', begin);
        if (tab == std::string_view::npos) {
            fields.push_back(line.substr(begin));
            return fields;
        }
        fields.push_back(line.substr(begin, tab - begin));
        begin = tab + 1U;
    }
}

// Everything that reaches a printed command line goes through this allowlist.
//
// `pull` writes a URL into a single-quoted shell argument that a person is
// meant to paste into a terminal. An apostrophe in a repository name would
// close that quote, so the rest of the line stops being data and starts being
// shell. Escaping is the wrong answer here: the values are repository IDs,
// git revisions, and file names, none of which legitimately contain anything
// outside this set, so refusing is both safer and simpler to reason about.
constexpr std::string_view manifest_token_characters =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/";

void validate_manifest_token(std::string_view value, const char * field) {
    if (value.empty()) {
        throw std::runtime_error(std::string(field) + " is empty");
    }
    for (const char raw : value) {
        if (manifest_token_characters.find(raw) == std::string_view::npos) {
            throw std::runtime_error(
                std::string("unsafe ") + field + ": " + std::string(value) +
                " (allowed: letters, digits, and . _ - /)");
        }
    }
    // A dot segment is inert in a name but not in a URL path. Every one of
    // these values becomes a path component of the resolve URL, so a revision
    // of "../../other/repo/resolve/main" would silently redirect every
    // printed download to a repository the operator never named.
    if (value.find("..") != std::string_view::npos) {
        throw std::runtime_error(std::string("unsafe ") + field + ": " +
                                 std::string(value) +
                                 " (must not contain '..')");
    }
}

// Rejects anything that could escape the destination directory as well.
void validate_manifest_name(std::string_view name) {
    validate_manifest_token(name, "manifest file name");
    if (name.front() == '/') {
        throw std::runtime_error("unsafe manifest file name: " +
                                 std::string(name));
    }
}

[[nodiscard]] std::uint64_t parse_manifest_bytes(std::string_view field) {
    if (field.empty() ||
        !std::all_of(field.begin(), field.end(),
                     [](char c) { return c >= '0' && c <= '9'; })) {
        throw std::runtime_error("manifest size is not a decimal count: " +
                                 std::string(field));
    }
    std::uint64_t value = 0;
    const auto [end, error] =
        std::from_chars(field.data(), field.data() + field.size(), value);
    if (error != std::errc{} || end != field.data() + field.size()) {
        throw std::runtime_error("manifest size is out of range: " +
                                 std::string(field));
    }
    return value;
}

[[nodiscard]] leann::detail::Sha256Digest
parse_manifest_digest(std::string_view field) {
    constexpr std::string_view prefix = "sha256:";
    if (!field.starts_with(prefix)) {
        throw std::runtime_error("manifest digest must start with sha256:, "
                                 "got: " +
                                 std::string(field));
    }
    const auto digest =
        leann::detail::parse_hex_digest(field.substr(prefix.size()));
    if (!digest) {
        throw std::runtime_error(
            "manifest digest must be 64 lowercase hex characters, got: " +
            std::string(field));
    }
    return *digest;
}

[[nodiscard]] ManifestFile parse_manifest_file_record(
    std::span<const std::string_view> fields, const char * record) {
    if (fields.size() != 4U) {
        throw std::runtime_error(std::string("manifest ") + record +
                                 " record needs 3 fields");
    }
    ManifestFile entry;
    entry.name = std::string(fields[1]);
    entry.bytes = parse_manifest_bytes(fields[2]);
    entry.digest = parse_manifest_digest(fields[3]);
    return entry;
}

[[nodiscard]] Manifest read_manifest(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open manifest: " + path.string());
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        // CR is rejected rather than stripped: a manifest is compared against
        // digests, so a silently rewritten byte stream is the wrong default.
        if (line.find('\r') != std::string::npos) {
            throw std::runtime_error("manifest has CRLF line endings: " +
                                     path.string());
        }
        lines.push_back(line);
    }
    if (lines.empty() || lines.front() != manifest_magic) {
        throw std::runtime_error("not a LEANNMF1 manifest: " + path.string());
    }

    Manifest manifest;
    bool saw_repo = false;
    bool saw_type = false;
    bool saw_revision = false;
    bool saw_prefix = false;
    std::unordered_set<std::string> names;
    for (std::size_t i = 1; i < lines.size(); ++i) {
        const std::string & raw = lines[i];
        if (raw.empty()) {
            throw std::runtime_error("manifest has a blank line: " +
                                     path.string());
        }
        const auto fields = split_tabs(raw);
        const std::string_view key = fields.front();
        const auto once = [&](bool & seen) {
            if (seen) {
                throw std::runtime_error("duplicate manifest key: " +
                                         std::string(key));
            }
            seen = true;
            if (fields.size() != 2U || fields[1].empty()) {
                throw std::runtime_error("manifest " + std::string(key) +
                                         " needs exactly one value");
            }
            return std::string(fields[1]);
        };
        if (key == "repo") {
            manifest.repo = once(saw_repo);
            validate_manifest_token(manifest.repo, "manifest repo");
        } else if (key == "type") {
            manifest.repo_type = once(saw_type);
            if (manifest.repo_type != "model" &&
                manifest.repo_type != "dataset") {
                throw std::runtime_error("manifest type must be model or "
                                         "dataset, got: " +
                                         manifest.repo_type);
            }
        } else if (key == "revision") {
            manifest.revision = once(saw_revision);
            validate_manifest_token(manifest.revision, "manifest revision");
        } else if (key == "prefix") {
            manifest.prefix = once(saw_prefix);
            validate_manifest_name(manifest.prefix);
        } else if (key == "file") {
            auto entry = parse_manifest_file_record(fields, "file");
            validate_manifest_name(entry.name);
            if (!names.insert(entry.name).second) {
                throw std::runtime_error("duplicate manifest file: " +
                                         entry.name);
            }
            manifest.files.push_back(std::move(entry));
        } else if (key == "model") {
            if (manifest.has_model) {
                throw std::runtime_error("duplicate manifest model record");
            }
            manifest.has_model = true;
            manifest.model = parse_manifest_file_record(fields, "model");
            manifest.model_source = manifest.model.name;
            // A model origin legitimately contains characters the token
            // allowlist excludes — "hf:owner/repo/file.gguf" has a colon —
            // and it is only ever printed as prose, never inside a command.
            // Control characters are still refused so the output stays
            // line-oriented.
            if (leann::detail::has_control_characters(manifest.model_source)) {
                throw std::runtime_error("unsafe manifest model source: " +
                                         manifest.model_source);
            }
            manifest.model.name.clear();
        } else {
            throw std::runtime_error("unknown manifest key: " +
                                     std::string(key));
        }
    }

    if (!saw_repo || !saw_prefix) {
        throw std::runtime_error("manifest needs repo and prefix records: " +
                                 path.string());
    }
    if (manifest.repo.find('/') == std::string::npos) {
        throw std::runtime_error("manifest repo must be OWNER/NAME, got: " +
                                 manifest.repo);
    }
    // The pair is the unit of publication, so a manifest that lists only one
    // half of it describes something that cannot be searched.
    for (const std::string & suffix : {".leann", ".docs"}) {
        if (!names.contains(manifest.prefix + suffix)) {
            throw std::runtime_error("manifest does not list " +
                                     manifest.prefix + suffix);
        }
    }
    return manifest;
}

[[nodiscard]] std::string resolve_url(const Manifest & manifest,
                                      std::string_view name) {
    std::string url = "https://huggingface.co/";
    if (manifest.repo_type == "dataset") {
        url += "datasets/";
    }
    url += manifest.repo;
    url += "/resolve/";
    url += manifest.revision;
    url += '/';
    url += name;
    return url;
}

// -L is mandatory: a resolve URL answers with a redirect to the CDN. -f turns
// an HTTP error into a nonzero exit instead of a file full of error markup.
//
// Both arguments are single-quoted. The URL is built only from allowlisted
// tokens, but the destination comes from --dest, and a local directory may
// legitimately contain a space; quoting is what makes that work. Neither can
// contain an apostrophe — the allowlist forbids it for the URL and
// validate_destination forbids it for the path — so the quoting holds.
[[nodiscard]] std::string curl_line(const std::string & url,
                                    const std::filesystem::path & destination) {
    return "curl -fL --retry 3 -o '" + destination.string() + "' '" + url +
           "'";
}

// Every printed path is quoted the same way, not just the ones inside a curl
// line. --dest and --manifest may contain spaces, and a follow-up command that
// only works for paths without them is a command that fails when pasted.
[[nodiscard]] std::string shell_quoted(const std::string & value) {
    return "'" + value + "'";
}

// Looser than the token allowlist because a destination is a local path, not
// a URL component, but still strict enough that the printed command means what
// it reads.
void validate_destination(const std::filesystem::path & destination) {
    const std::string text = destination.string();
    if (text.empty()) {
        throw std::invalid_argument("path argument is empty");
    }
    if (leann::detail::has_control_characters(text) ||
        text.find('\'') != std::string::npos) {
        throw std::invalid_argument(
            "path must not contain quotes or control characters: " + text);
    }
}

// Parses the `hf:OWNER/NAME` positional. Kept strict and tiny: the only
// scheme is hf:, and `datasets/` is the one recognised qualifier.
struct RepoSpec {
    std::string repo;
    std::string repo_type = "model";
};

[[nodiscard]] RepoSpec parse_repo_spec(std::string_view spec) {
    constexpr std::string_view scheme = "hf:";
    if (!spec.starts_with(scheme)) {
        throw std::invalid_argument(
            "repository must start with hf:, got: " + std::string(spec));
    }
    std::string_view rest = spec.substr(scheme.size());
    RepoSpec result;
    if (rest.starts_with("datasets/")) {
        result.repo_type = "dataset";
        rest = rest.substr(std::string_view("datasets/").size());
    }
    const std::size_t slash = rest.find('/');
    if (slash == std::string_view::npos || slash == 0 ||
        slash + 1U >= rest.size() ||
        rest.find('/', slash + 1U) != std::string_view::npos) {
        throw std::invalid_argument(
            "repository must be hf:OWNER/NAME, got: " + std::string(spec));
    }
    // The repository name is interpolated into a printed shell command, so it
    // is held to the same allowlist as everything read from a manifest.
    try {
        validate_manifest_token(rest, "repository");
    } catch (const std::runtime_error & error) {
        throw std::invalid_argument(error.what());
    }
    result.repo = std::string(rest);
    return result;
}

void command_pull(const Arguments & args) {
    const auto format = output_format(args);
    if (args.positional().empty()) {
        throw std::invalid_argument(
            "leann pull needs a repository, for example hf:OWNER/NAME");
    }
    const RepoSpec spec = parse_repo_spec(args.positional().front());
    const std::string revision = args.get("--revision", "main");
    validate_manifest_token(revision, "--revision");
    const auto destination = std::filesystem::path(args.get("--dest", "."));
    validate_destination(destination);
    const std::string manifest_argument = args.get("--manifest");
    if (!manifest_argument.empty()) {
        // Reprinted inside the quoted follow-up command, so it is held to the
        // same rule as --dest.
        validate_destination(manifest_argument);
    }

    Manifest manifest;
    manifest.repo = spec.repo;
    manifest.repo_type = spec.repo_type;
    manifest.revision = revision;
    bool have_manifest = false;
    if (!manifest_argument.empty()) {
        manifest = read_manifest(manifest_argument);
        have_manifest = true;
        if (manifest.repo != spec.repo) {
            throw std::runtime_error("manifest describes " + manifest.repo +
                                     " but the requested repository is " +
                                     spec.repo);
        }
        // A model and a dataset repository of the same name are different
        // namespaces on the Hub. Letting the manifest silently override the
        // qualifier the operator typed would print downloads from a
        // repository they did not ask for.
        if (manifest.repo_type != spec.repo_type) {
            throw std::runtime_error(
                "manifest describes a " + manifest.repo_type +
                " repository but " + std::string(args.positional().front()) +
                " names a " + spec.repo_type + " repository");
        }
        // An explicit --revision is the caller's intent and overrides what the
        // manifest recorded, which may be a branch that has since moved.
        if (args.has("--revision")) {
            manifest.revision = revision;
        }
    }

    const std::string manifest_url = resolve_url(manifest, "leann.manifest");
    std::uint64_t total_bytes = 0;
    for (const ManifestFile & file : manifest.files) {
        total_bytes += file.bytes;
    }

    if (format == OutputFormat::Json) {
        JsonDocument document(std::cout);
        JsonWriter & json = document.writer();
        json.begin_object();
        json.field("repo", manifest.repo);
        json.field("repo_type", manifest.repo_type);
        json.field("revision", manifest.revision);
        json.field("manifest_url", manifest_url);
        json.field("have_manifest", have_manifest);
        json.field("fetches_anything", false);
        if (have_manifest) {
            json.field("prefix", manifest.prefix);
            json.field("total_bytes", total_bytes);
            json.begin_array("files");
            for (const ManifestFile & file : manifest.files) {
                json.begin_element();
                json.field("name", file.name);
                json.field("bytes", file.bytes);
                json.field("sha256", leann::detail::hex_digest(file.digest));
                json.field("url", resolve_url(manifest, file.name));
                json.field("command",
                           curl_line(resolve_url(manifest, file.name),
                                     destination / file.name));
                json.end_object();
            }
            json.end_array();
            if (manifest.has_model) {
                json.begin_object_field("model");
                json.field("source", manifest.model_source);
                json.field("bytes", manifest.model.bytes);
                json.field("sha256",
                           leann::detail::hex_digest(manifest.model.digest));
                json.end_object();
            }
        }
        json.end_object();
        document.commit();
        return;
    }

    std::cout << "repo: " << manifest.repo << " (" << manifest.repo_type
              << " revision " << manifest.revision << ")\n";
    if (!have_manifest) {
        std::cout
            << "step 1 — fetch the manifest:\n  "
            << curl_line(manifest_url, destination / "leann.manifest")
            << "\nstep 2 — print the checked plan:\n  leann pull "
            << args.positional().front() << " --manifest "
            << shell_quoted((destination / "leann.manifest").string()) << "\n";
        return;
    }

    std::cout << "prefix: " << manifest.prefix << '\n'
              << "total_bytes: " << total_bytes << '\n';
    if (manifest.has_model) {
        std::cout << "model: " << manifest.model_source << " ("
                  << manifest.model.bytes << " bytes, sha256:"
                  << leann::detail::hex_digest(manifest.model.digest) << ")\n";
    }
    std::cout << "download:\n";
    for (const ManifestFile & file : manifest.files) {
        std::cout << "  " << curl_line(resolve_url(manifest, file.name),
                                       destination / file.name)
                  << "\n  # " << file.bytes
                  << " bytes, sha256:" << leann::detail::hex_digest(file.digest)
                  << '\n';
    }
    std::cout << "then:\n  leann verify --index "
              << shell_quoted((destination / manifest.prefix).string())
              << " --manifest " << shell_quoted(manifest_argument) << '\n';
}

// The pair half of verify, shared in shape with doctor so the two report the
// same fact in the same words rather than drifting into two vocabularies.
struct VerifiedFile {
    std::string name;
    bool present = false;
    bool size_matches = false;
    bool digest_matches = false;
    std::uint64_t expected_bytes = 0;
    std::uint64_t actual_bytes = 0;
};

void command_verify(const Arguments & args) {
    const auto format = output_format(args);
    const auto prefix = std::filesystem::path(args.require("--index"));
    const std::string manifest_argument = args.get("--manifest");

    bool pair_valid = false;
    std::string pair_error;
    std::string pair_identity;
    std::string model_source;
    std::string model_sha256;
    try {
        const auto index =
            leann::Index::load(leann::index_file_from_prefix(prefix));
        auto documents =
            leann::DocumentStore::open(leann::documents_file_from_prefix(prefix));
        index.validate_document_store(documents);
        const auto stats = index.stats();
        pair_identity = stats.pair_identity;
        model_source = stats.model_source;
        model_sha256 = stats.model_sha256;
        pair_valid = true;
    } catch (const std::exception & error) {
        pair_error = error.what();
    }

    std::vector<VerifiedFile> checked;
    bool manifest_matches = true;
    std::string model_disagreement;
    if (!manifest_argument.empty()) {
        const Manifest manifest = read_manifest(manifest_argument);
        // The index says which model built it and the manifest says which
        // model to fetch. If those disagree, following the manifest gets a
        // model the index will refuse — better to say so here than to let it
        // surface later as a fingerprint mismatch on the first query.
        if (pair_valid && manifest.has_model) {
            const std::string manifest_digest =
                leann::detail::hex_digest(manifest.model.digest);
            if (!model_source.empty() &&
                manifest.model_source != model_source) {
                model_disagreement = "manifest names model " +
                                     manifest.model_source +
                                     " but the index was built from " +
                                     model_source;
            } else if (model_sha256 != std::string(64, '0') &&
                       manifest_digest != model_sha256) {
                model_disagreement =
                    "manifest model digest " + manifest_digest +
                    " does not match the index's " + model_sha256;
            }
            if (!model_disagreement.empty()) {
                manifest_matches = false;
            }
        }
        // Fail closed rather than verifying one pair against another pair's
        // digests, which would report a mismatch that reads like corruption.
        if (manifest.prefix != prefix.filename().string()) {
            throw std::runtime_error(
                "manifest describes prefix '" + manifest.prefix +
                "' but --index names '" + prefix.filename().string() + "'");
        }
        // Named files are resolved beside the artifact prefix, not beside the
        // manifest: the manifest may have been fetched anywhere, but the pair
        // it describes is the one --index points at.
        const auto directory = prefix.parent_path();
        for (const ManifestFile & file : manifest.files) {
            VerifiedFile result;
            result.name = file.name;
            result.expected_bytes = file.bytes;
            const auto path = directory / file.name;
            std::error_code code;
            const auto size = std::filesystem::file_size(path, code);
            if (code) {
                manifest_matches = false;
                checked.push_back(std::move(result));
                continue;
            }
            result.present = true;
            result.actual_bytes = static_cast<std::uint64_t>(size);
            result.size_matches = result.actual_bytes == file.bytes;
            // Only digest a file whose size already agrees: a mismatched size
            // is already a failure and hashing gigabytes proves nothing more.
            result.digest_matches =
                result.size_matches &&
                leann::detail::sha256_file_prefix(path, result.actual_bytes) ==
                    file.digest;
            if (!result.digest_matches) {
                manifest_matches = false;
            }
            checked.push_back(std::move(result));
        }
    }

    const bool ok = pair_valid && manifest_matches;
    if (format == OutputFormat::Json) {
        JsonDocument document(std::cout);
        JsonWriter & json = document.writer();
        json.begin_object();
        json.field("pair_valid", pair_valid);
        if (pair_valid) {
            json.field("pair_identity", pair_identity);
            json.field("model_source", model_source);
        } else {
            json.field("pair_error", pair_error);
        }
        json.field("checked_manifest", !manifest_argument.empty());
        json.field("model_disagreement", model_disagreement);
        json.begin_array("files");
        for (const VerifiedFile & file : checked) {
            json.begin_element();
            json.field("name", file.name);
            json.field("present", file.present);
            json.field("expected_bytes", file.expected_bytes);
            json.field("actual_bytes", file.actual_bytes);
            json.field("digest_matches", file.digest_matches);
            json.end_object();
        }
        json.end_array();
        json.field("ok", ok);
        json.end_object();
        document.commit();
    } else {
        std::cout << "pair: " << (pair_valid ? "valid" : "unusable");
        if (pair_valid) {
            std::cout << " (identity " << pair_identity << ")";
        } else {
            std::cout << " — " << pair_error;
        }
        std::cout << '\n';
        if (pair_valid && !model_source.empty()) {
            std::cout << "model_source: " << model_source << '\n';
        }
        for (const VerifiedFile & file : checked) {
            std::cout << "file: " << file.name << ' ';
            if (!file.present) {
                std::cout << "missing\n";
            } else if (!file.size_matches) {
                std::cout << "size mismatch (expected " << file.expected_bytes
                          << ", got " << file.actual_bytes << ")\n";
            } else if (!file.digest_matches) {
                std::cout << "digest mismatch\n";
            } else {
                std::cout << "ok\n";
            }
        }
        if (!model_disagreement.empty()) {
            std::cout << "model: " << model_disagreement << '\n';
        }
        if (manifest_argument.empty()) {
            std::cout << "manifest: not checked\n";
        }
        std::cout << "verify: " << (ok ? "ok" : "failed") << '\n';
    }
    if (!ok) {
        // A failed verification must not exit 0; the whole point is that a
        // script can branch on it.
        if (!pair_valid) {
            throw std::runtime_error("artifact pair does not load: " +
                                     pair_error);
        }
        throw std::runtime_error(
            model_disagreement.empty()
                ? "manifest does not match the files on disk"
                : model_disagreement);
    }
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

void command_bench(const Arguments & args,
                   CancellationToken cancellation = {}) {
    // Constructed up front, not at the query loop: it validates --progress,
    // and a benchmark can spend an hour computing ground truth before it ever
    // reaches the loop. A typo there should cost a second, not the run.
    ProgressReporter progress(args, "--progress");
    const auto format = output_format(args);
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
    // A precomputed cache holds vectors somebody else embedded, and neither
    // LEANNBC2 nor the ground-truth cache records the prefix its producer
    // used. Against a prefixed index that mismatch is invisible: nothing
    // throws, recall just drops. Refusing the combination is the only way to
    // keep "measured recall" meaning what it says.
    if (!query_cache_argument.empty() && !index.query_prefix().empty()) {
        throw std::invalid_argument(
            "--query-embedding-cache cannot be used with an index that has a "
            "query prefix; the cached vectors do not record one");
    }
    if (!ground_truth_cache_argument.empty() &&
        !index.document_prefix().empty()) {
        throw std::invalid_argument(
            "--ground-truth-cache cannot be used with an index that has a "
            "document prefix; the cached vectors do not record one");
    }
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
                // Ground truth must be computed in the same space the index
                // was built in. bench calls search_embedding directly, so
                // Index cannot apply the prefix on its behalf here.
                if (!index.document_prefix().empty()) {
                    for (std::string & text : batch_documents) {
                        text.insert(0, index.document_prefix());
                    }
                }
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
        // Same reason as the ground-truth pass: search_embedding receives a
        // finished vector, so the index's query prefix has to be applied here.
        // A --query-embedding-cache is exempt only because its vectors already
        // exist; nothing in C++ can check what text produced them.
        const std::array<std::string, 1> query_batch{
            index.query_prefix() + queries[query_index]};
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
        cancellation.throw_if_requested("benchmark warmup");
        progress.report("warmup", query_index, warmup_queries);
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
        // Between queries only: an interrupt must not truncate a measured
        // query and then report the partial result as a benchmark.
        cancellation.throw_if_requested("benchmark");
        progress.report("querying", query_index, queries.size());
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
    progress.finish();
    if (raw_output) {
        write_raw_benchmark_rows(
            raw_output->stream(), raw_rows, report_k);
        raw_output->publish();
        std::cerr << "wrote_raw_latencies=" << raw_latencies_argument
                  << '\n';
    }

    const double count = static_cast<double>(queries.size());
    const double latency_mean =
        std::accumulate(latencies.begin(), latencies.end(), 0.0) / count;
    const std::string recall_key =
        "recall_at_" + std::to_string(config.top_k);
    const double dense_latency_mean =
        dense_latencies.empty()
            ? 0.0
            : std::accumulate(dense_latencies.begin(), dense_latencies.end(),
                              0.0) /
                  count;

    if (format == OutputFormat::Json) {
        // Same keys as the text form, including the top-k-dependent recall
        // key names, so a consumer can switch formats without remapping.
        JsonDocument document(std::cout);
        JsonWriter & json = document.writer();
        json.begin_object();
        json.field("queries", static_cast<std::uint64_t>(queries.size()));
        json.field("warmup_queries",
                   static_cast<std::uint64_t>(warmup_queries));
        json.field(recall_key, recall_sum / count, 6);
        if (report_k) {
            json.field("recall_at_" + std::to_string(*report_k),
                       report_recall_sum / count, 6);
        }
        json.field("latency_ms_mean", latency_mean, 6);
        json.field("latency_ms_p50", percentile(latencies, 0.50), 6);
        json.field("latency_ms_p95", percentile(latencies, 0.95), 6);
        json.field("exact_recomputations_mean",
                   static_cast<double>(exact_sum) / count, 6);
        json.field("approximate_distances_mean",
                   static_cast<double>(approximate_sum) / count, 6);
        json.field("upper_layer_hops_mean",
                   static_cast<double>(upper_hops_sum) / count, 6);
        if (dense_index) {
            json.field("dense_hnsw_" + recall_key, dense_recall_sum / count,
                       6);
            json.field("dense_hnsw_latency_ms_mean", dense_latency_mean, 6);
            json.field("dense_hnsw_latency_ms_p50",
                       percentile(dense_latencies, 0.50), 6);
            json.field("dense_hnsw_latency_ms_p95",
                       percentile(dense_latencies, 0.95), 6);
            json.field("dense_hnsw_build_seconds", dense_build_seconds, 6);
            json.field("dense_hnsw_index_bytes", dense_index_bytes);
        }
        json.end_object();
        document.commit();
        return;
    }

    std::cout << "queries=" << queries.size() << '\n'
              << "warmup_queries=" << warmup_queries << '\n'
              << "recall_at_" << config.top_k << '=' << std::fixed
              << std::setprecision(6) << recall_sum / count << '\n';
    if (report_k) {
        std::cout << "recall_at_" << *report_k << '='
                  << report_recall_sum / count << '\n';
    }
    std::cout << "latency_ms_mean=" << latency_mean << '\n'
              << "latency_ms_p50=" << percentile(latencies, 0.50) << '\n'
              << "latency_ms_p95=" << percentile(latencies, 0.95) << '\n'
              << "exact_recomputations_mean=" << exact_sum / count << '\n'
              << "approximate_distances_mean=" << approximate_sum / count
              << '\n'
              << "upper_layer_hops_mean=" << upper_hops_sum / count << '\n';
    if (dense_index) {
        std::cout << "dense_hnsw_recall_at_" << config.top_k << '='
                  << dense_recall_sum / count << '\n'
                  << "dense_hnsw_latency_ms_mean=" << dense_latency_mean
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
        // --version and --help are handled before Arguments because option
        // parsing starts at argv[2]; in argv[1] they name the request itself.
        const std::string command = argv[1];
        if (command == "--version" || command == "-V" ||
            command == "version") {
            std::cout << "leann.cpp " << LEANN_VERSION_STRING << '\n';
            return 0;
        }
        if (command == "help" || command == "--help" || command == "-h") {
            if (argc >= 3) {
                const CommandSpec * requested = find_command(argv[2]);
                if (requested == nullptr) {
                    throw std::invalid_argument(
                        "unknown command: " + std::string(argv[2]) +
                        suggest_command(argv[2]));
                }
                print_command_help(std::cout, *requested);
                return 0;
            }
            print_usage(std::cout);
            return 0;
        }

        const CommandSpec * spec = find_command(command);
        if (spec == nullptr) {
            throw std::invalid_argument("unknown command: " + command +
                                        suggest_command(command));
        }
        const Arguments args(argc, argv);
        if (args.has("--help")) {
            print_command_help(std::cout, *spec);
            return 0;
        }
        validate_arguments(*spec, args);

        // Handlers are installed only for the commands that actually poll for
        // cancellation, and only inside main. Installing them for every
        // command would leave search, stats, and doctor catching a signal
        // that nothing acts on, making them silently uninterruptible; and
        // installing them at namespace scope would take over the signal
        // handling of any test binary that includes this file.
        const auto install_handlers = [] {
            std::signal(SIGINT, handle_interrupt);
            std::signal(SIGTERM, handle_interrupt);
            return CancellationToken(&interrupt_requested);
        };

        if (command == "build") {
            command_build(args, install_handlers());
        } else if (command == "search") {
            command_search(args);
        } else if (command == "stats") {
            command_stats(args);
        } else if (command == "bench") {
            command_bench(args, install_handlers());
        } else if (command == "doctor") {
            command_doctor(args);
        } else if (command == "pull") {
            // No install_handlers: neither command polls for cancellation, so
            // catching a signal would only make them uninterruptible.
            command_pull(args);
        } else if (command == "verify") {
            command_verify(args);
        } else {
            throw std::invalid_argument("unknown command: " + command);
        }
        return 0;
    } catch (const leann::BuildCancelled & cancelled) {
        // 130 is the conventional shell status for a SIGINT-terminated
        // command, which is what the interrupt handler translates into here.
        std::cerr << "cancelled: " << cancelled.what() << '\n';
        return 130;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
