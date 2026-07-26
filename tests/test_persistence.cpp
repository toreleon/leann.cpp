#include "leann/document_store.hpp"
#include "leann/embedder.hpp"
#include "leann/index.hpp"

#include "artifact_publisher.hpp"
#include "checksum.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error("test failed: " + message);
    }
}

template <typename Exception = std::runtime_error>
void expect_error(const std::function<void()> & action,
                  std::string_view expected_message) {
    try {
        action();
    } catch (const Exception & error) {
        check(std::string_view(error.what()).find(expected_message) !=
                  std::string_view::npos,
              "expected error containing '" + std::string(expected_message) +
                  "', got '" + error.what() + "'");
        return;
    }
    throw std::runtime_error("test failed: expected error containing '" +
                             std::string(expected_message) + "'");
}

std::vector<std::string> make_corpus(std::string_view topic) {
    std::vector<std::string> documents;
    for (int i = 0; i < 24; ++i) {
        const std::string suffix = i < 10 ? "0" + std::to_string(i)
                                          : std::to_string(i);
        documents.push_back(std::string(topic) + " document number " + suffix);
    }
    return documents;
}

leann::BuildConfig build_config() {
    leann::BuildConfig config;
    config.graph_degree = 4;
    config.ef_construction = 20;
    config.low_degree = 2;
    config.hub_ratio = 0.1;
    config.pq_subquantizers = 8;
    config.pq_bits = 3;
    config.pq_training_iterations = 3;
    config.pq_training_samples = 24;
    config.embedding_batch_size = 8;
    return config;
}

void flip_byte(const std::filesystem::path & path, std::uint64_t offset) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
        throw std::runtime_error("cannot open fixture for mutation");
    }
    file.seekg(static_cast<std::streamoff>(offset));
    const int original = file.get();
    if (original == std::char_traits<char>::eof()) {
        throw std::runtime_error("fixture mutation offset is out of range");
    }
    file.seekp(static_cast<std::streamoff>(offset));
    file.put(static_cast<char>(static_cast<unsigned char>(original) ^ 0x01U));
    file.close();
    if (!file) {
        throw std::runtime_error("failed to mutate fixture");
    }
}

void append_byte(const std::filesystem::path & path) {
    std::ofstream output(path, std::ios::binary | std::ios::app);
    output.put('\x5a');
    output.close();
    if (!output) {
        throw std::runtime_error("failed to append fixture byte");
    }
}

void write_text(const std::filesystem::path & path, std::string_view text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.close();
    if (!output) {
        throw std::runtime_error("failed to write publisher fixture");
    }
}

std::vector<char> read_all(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

template <typename T>
void write_le(std::ostream & output, T value) {
    for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
        output.put(static_cast<char>((value >> (byte * 8U)) & T{0xffU}));
    }
}

class CountingEmbedder final : public leann::Embedder {
  public:
    [[nodiscard]] std::size_t dimension() const noexcept override {
        return inner_.dimension();
    }

    [[nodiscard]] std::string fingerprint() const override {
        return inner_.fingerprint();
    }

    [[nodiscard]] std::vector<leann::Embedding>
    embed(std::span<const std::string> texts) override {
        ++calls;
        return inner_.embed(texts);
    }

    std::size_t calls = 0;

  private:
    leann::HashEmbedder inner_{64};
};

// Reports a descriptor whose every field holds a distinct, recognisable value.
// The two u32 fields in particular must differ: a write/read transposition of
// two same-width fields preserves the file size, so the terminal
// "unexpected trailing bytes" check cannot catch it and only a value
// comparison can.
class DescribedEmbedder final : public leann::Embedder {
  public:
    [[nodiscard]] std::size_t dimension() const noexcept override {
        return inner_.dimension();
    }

    [[nodiscard]] std::string fingerprint() const override {
        return inner_.fingerprint();
    }

    [[nodiscard]] leann::EmbedderDescriptor descriptor() const override {
        leann::EmbedderDescriptor described;
        described.source = "hf:owner/repo/model.gguf";
        for (std::size_t i = 0; i < described.sha256.size(); ++i) {
            described.sha256[i] = static_cast<std::uint8_t>(i + 1U);
        }
        described.bytes = 84106624U;
        described.pooling_type = 1U;
        described.context_tokens = 512U;
        return described;
    }

    [[nodiscard]] std::vector<leann::Embedding>
    embed(std::span<const std::string> texts) override {
        return inner_.embed(texts);
    }

  private:
    leann::HashEmbedder inner_{64};
};

class ThrowingEmbedder final : public leann::Embedder {
  public:
    [[nodiscard]] std::size_t dimension() const noexcept override { return 64; }
    [[nodiscard]] std::string fingerprint() const override {
        return "throwing-embedder";
    }
    [[nodiscard]] std::vector<leann::Embedding>
    embed(std::span<const std::string>) override {
        throw std::runtime_error("injected embedding failure");
    }
};

class BlockingEmbedder final : public leann::Embedder {
  public:
    [[nodiscard]] std::size_t dimension() const noexcept override {
        return inner_.dimension();
    }

    [[nodiscard]] std::string fingerprint() const override {
        return inner_.fingerprint();
    }

    [[nodiscard]] std::vector<leann::Embedding>
    embed(std::span<const std::string> texts) override {
        {
            std::unique_lock lock(mutex_);
            started_ = true;
            condition_.notify_all();
            condition_.wait(lock, [&] { return released_; });
        }
        return inner_.embed(texts);
    }

    void wait_until_started() {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return started_; });
    }

    void release() {
        std::lock_guard lock(mutex_);
        released_ = true;
        condition_.notify_all();
    }

  private:
    leann::HashEmbedder inner_{64};
    std::mutex mutex_;
    std::condition_variable condition_;
    bool started_ = false;
    bool released_ = false;
};

class LockPollutingEmbedder final : public leann::Embedder {
  public:
    explicit LockPollutingEmbedder(std::filesystem::path lock_path)
        : lock_path_(std::move(lock_path)) {}

    [[nodiscard]] std::size_t dimension() const noexcept override {
        return inner_.dimension();
    }

    [[nodiscard]] std::string fingerprint() const override {
        return inner_.fingerprint();
    }

    [[nodiscard]] std::vector<leann::Embedding>
    embed(std::span<const std::string> texts) override {
        if (!polluted_) {
            write_text(lock_path_ / "external-entry", "keep");
            polluted_ = true;
        }
        return inner_.embed(texts);
    }

  private:
    leann::HashEmbedder inner_{64};
    std::filesystem::path lock_path_;
    bool polluted_ = false;
};

class FaultingFileOperations final
    : public leann::detail::ArtifactFileOperations {
  public:
    bool exists(const std::filesystem::path & path) override {
        return std::filesystem::exists(path);
    }

    void rename(const std::filesystem::path & from,
                const std::filesystem::path & to,
                std::string_view operation) override {
        ++rename_calls_;
        if (fail_rename_call_ && rename_calls_ == *fail_rename_call_) {
            fail_rename_call_.reset();
            throw std::runtime_error("injected rename failure");
        }
        std::error_code error;
        std::filesystem::rename(from, to, error);
        if (error) {
            throw std::runtime_error(std::string(operation) + ": " +
                                     error.message());
        }
    }

    std::error_code remove(const std::filesystem::path & path) override {
        if (fail_remove_path_ && path == *fail_remove_path_) {
            return std::make_error_code(std::errc::permission_denied);
        }
        std::error_code error;
        std::filesystem::remove(path, error);
        return error;
    }

    void fail_rename_call(std::size_t call) { fail_rename_call_ = call; }
    void fail_remove(const std::filesystem::path & path) {
        fail_remove_path_ = path;
    }

  private:
    std::size_t rename_calls_ = 0;
    std::optional<std::size_t> fail_rename_call_;
    std::optional<std::filesystem::path> fail_remove_path_;
};

struct PublisherPaths {
    std::filesystem::path temporary_index;
    std::filesystem::path temporary_documents;
    std::filesystem::path index;
    std::filesystem::path documents;
    std::filesystem::path index_backup;
    std::filesystem::path documents_backup;
};

PublisherPaths publisher_paths(const std::filesystem::path & directory) {
    return {
        directory / "new-index.tmp",
        directory / "new-documents.tmp",
        directory / "active.leann",
        directory / "active.docs",
        directory / "old-index.bak",
        directory / "old-documents.bak",
    };
}

void seed_new_artifacts(const PublisherPaths & paths) {
    write_text(paths.temporary_index, "new-index");
    write_text(paths.temporary_documents, "new-documents");
}

void check_publish_rollback(const std::filesystem::path & root,
                            std::size_t failed_rename) {
    const auto directory =
        root / ("rollback-" + std::to_string(failed_rename));
    std::filesystem::create_directories(directory);
    const auto paths = publisher_paths(directory);
    seed_new_artifacts(paths);
    write_text(paths.index, "old-index");
    write_text(paths.documents, "old-documents");

    FaultingFileOperations operations;
    operations.fail_rename_call(failed_rename);
    expect_error(
        [&] {
            leann::detail::publish_artifact_pair(
                paths.temporary_index, paths.temporary_documents, paths.index,
                paths.documents, paths.index_backup, paths.documents_backup,
                operations);
        },
        "injected rename failure");
    check(read_all(paths.index) == std::vector<char>({'o', 'l', 'd', '-',
                                                      'i', 'n', 'd', 'e', 'x'}),
          "rollback restores previous index bytes");
    check(read_all(paths.documents) ==
              std::vector<char>({'o', 'l', 'd', '-', 'd', 'o', 'c', 'u', 'm',
                                 'e', 'n', 't', 's'}),
          "rollback restores previous document bytes");
    check(!std::filesystem::exists(paths.index_backup),
          "rollback consumes index backup");
    check(!std::filesystem::exists(paths.documents_backup),
          "rollback consumes document backup");
}

void check_publish_success_state(const std::filesystem::path & root,
                                 std::string_view name, bool old_index,
                                 bool old_documents) {
    const auto directory = root / std::string(name);
    std::filesystem::create_directories(directory);
    const auto paths = publisher_paths(directory);
    seed_new_artifacts(paths);
    if (old_index) {
        write_text(paths.index, "old-index");
    }
    if (old_documents) {
        write_text(paths.documents, "old-documents");
    }
    leann::detail::publish_artifact_pair(
        paths.temporary_index, paths.temporary_documents, paths.index,
        paths.documents, paths.index_backup, paths.documents_backup);
    check(read_all(paths.index) ==
              std::vector<char>({'n', 'e', 'w', '-', 'i', 'n', 'd', 'e', 'x'}),
          "publication installs new index");
    check(read_all(paths.documents) ==
              std::vector<char>({'n', 'e', 'w', '-', 'd', 'o', 'c', 'u', 'm',
                                 'e', 'n', 't', 's'}),
          "publication installs new documents");
    check(!std::filesystem::exists(paths.index_backup),
          "successful publication removes index backup");
    check(!std::filesystem::exists(paths.documents_backup),
          "successful publication removes document backup");
}

void check_no_build_residue(const std::filesystem::path & directory) {
    for (const auto & entry : std::filesystem::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        check(name.find(".tmp.") == std::string::npos,
              "temporary artifact residue: " + name);
        check(name.find(".bak.") == std::string::npos,
              "backup artifact residue: " + name);
        check(!name.ends_with(".lock"), "build lock residue: " + name);
    }
}

} // namespace

int main() {
    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory = std::filesystem::temp_directory_path() /
                           ("leann-cpp-persistence-test-" + unique);
    std::filesystem::create_directories(directory);

    try {
        check(leann::detail::hex_digest(leann::detail::sha256("")) ==
                  "e3b0c44298fc1c149afbf4c8996fb924"
                  "27ae41e4649b934ca495991b7852b855",
              "SHA-256 empty known-answer vector");
        check(leann::detail::hex_digest(leann::detail::sha256("abc")) ==
                  "ba7816bf8f01cfea414140de5dae2223"
                  "b00361a396177a9cb410ff61f20015ad",
              "SHA-256 abc known-answer vector");
        leann::detail::Sha256 split_hasher;
        split_hasher.update("a");
        split_hasher.update("b");
        split_hasher.update("c");
        check(leann::detail::hex_digest(split_hasher.finish()) ==
                  "ba7816bf8f01cfea414140de5dae2223"
                  "b00361a396177a9cb410ff61f20015ad",
              "SHA-256 split-update vector");
        constexpr std::string_view nist_56 =
            "abcdbcdecdefdefgefghfghighijhijk"
            "ijkljklmklmnlmnomnopnopq";
        check(leann::detail::hex_digest(leann::detail::sha256(nist_56)) ==
                  "248d6a61d20638b8e5c026930c3e6039"
                  "a33ce45964ff2167f6ecedd419db06c1",
              "SHA-256 56-byte NIST vector");
        for (const std::size_t size : {55U, 56U, 63U, 64U, 65U}) {
            const std::string boundary(size, 'x');
            const auto whole = leann::detail::sha256(boundary);
            leann::detail::Sha256 streaming;
            for (std::size_t begin = 0; begin < boundary.size(); begin += 7U) {
                streaming.update(std::string_view(boundary).substr(
                    begin, std::min<std::size_t>(7U, boundary.size() - begin)));
            }
            check(streaming.finish() == whole,
                  "SHA-256 boundary split at " + std::to_string(size));
        }
        check(leann::detail::crc32c("123456789") == 0xe3069283U,
              "CRC32C known-answer vector");

        check_publish_rollback(directory, 1);
        check_publish_rollback(directory, 2);
        check_publish_rollback(directory, 3);
        check_publish_rollback(directory, 4);
        check_publish_success_state(directory, "publish-both", true, true);
        check_publish_success_state(directory, "publish-index-only", true,
                                    false);
        check_publish_success_state(directory, "publish-documents-only", false,
                                    true);
        check_publish_success_state(directory, "publish-neither", false,
                                    false);
        {
            const auto cleanup_directory = directory / "cleanup-failure";
            std::filesystem::create_directories(cleanup_directory);
            const auto paths = publisher_paths(cleanup_directory);
            seed_new_artifacts(paths);
            write_text(paths.index, "old-index");
            write_text(paths.documents, "old-documents");
            FaultingFileOperations operations;
            operations.fail_remove(paths.index_backup);
            expect_error(
                [&] {
                    leann::detail::publish_artifact_pair(
                        paths.temporary_index, paths.temporary_documents,
                        paths.index, paths.documents, paths.index_backup,
                        paths.documents_backup, operations);
                },
                "new artifact pair committed; backup cleanup required");
            check(read_all(paths.index) ==
                      std::vector<char>({'n', 'e', 'w', '-', 'i', 'n', 'd',
                                         'e', 'x'}),
                  "cleanup failure still commits new index");
            check(read_all(paths.documents) ==
                      std::vector<char>({'n', 'e', 'w', '-', 'd', 'o', 'c',
                                         'u', 'm', 'e', 'n', 't', 's'}),
                  "cleanup failure still commits new documents");
            check(std::filesystem::exists(paths.index_backup),
                  "failed cleanup retains recoverable index backup");
            check(!std::filesystem::exists(paths.documents_backup),
                  "successful document-backup cleanup is not suppressed");
            std::filesystem::remove(paths.index_backup);
        }

        const auto corpus_a = make_corpus("alpha");
        const auto corpus_b = make_corpus("omega");
        check(corpus_a.size() == corpus_b.size(),
              "cross-pair fixture counts match");
        for (std::size_t i = 0; i < corpus_a.size(); ++i) {
            check(corpus_a[i].size() == corpus_b[i].size(),
                  "cross-pair fixture lengths match");
        }

        const auto prefix_a = directory / "a";
        const auto prefix_b = directory / "b";
        leann::HashEmbedder build_embedder(64);
        leann::Index::build(leann::index_file_from_prefix(prefix_a),
                            leann::documents_file_from_prefix(prefix_a),
                            corpus_a, build_embedder, build_config());
        leann::Index::build(leann::index_file_from_prefix(prefix_b),
                            leann::documents_file_from_prefix(prefix_b),
                            corpus_b, build_embedder, build_config());

        const auto index_a =
            leann::Index::load(leann::index_file_from_prefix(prefix_a));
        const auto index_b =
            leann::Index::load(leann::index_file_from_prefix(prefix_b));
        auto documents_a = leann::DocumentStore::open(
            leann::documents_file_from_prefix(prefix_a));
        auto documents_b = leann::DocumentStore::open(
            leann::documents_file_from_prefix(prefix_b));
        index_a.validate_document_store(documents_a);
        index_b.validate_document_store(documents_b);
        check(index_a.pair_identity() == documents_a.pair_identity(),
              "matching pair identity");
        check(index_a.pair_identity() != index_b.pair_identity(),
              "different corpora have different pair identities");

        leann::SearchConfig search;
        search.top_k = 1;
        search.ef_search = 8;
        search.recompute_batch_size = 4;
        CountingEmbedder counting;
        expect_error<std::invalid_argument>(
            [&] { (void)index_a.search("alpha", counting, documents_b, search); },
            "pair identity mismatch");
        check(counting.calls == 0,
              "pair mismatch is rejected before query embedding");
        expect_error<std::invalid_argument>(
            [&] {
                (void)index_b.search("omega", counting, documents_a, search);
            },
            "pair identity mismatch");
        check(counting.calls == 0,
              "reverse pair mismatch is rejected before query embedding");

        const auto corrupt_index_payload = directory / "index-payload.leann";
        std::filesystem::copy_file(leann::index_file_from_prefix(prefix_a),
                                   corrupt_index_payload);
        const auto index_size =
            std::filesystem::file_size(corrupt_index_payload);
        flip_byte(corrupt_index_payload, index_size - 33U);
        expect_error(
            [&] { (void)leann::Index::load(corrupt_index_payload); },
            "SHA-256 checksum mismatch");

        const auto corrupt_index_footer = directory / "index-footer.leann";
        std::filesystem::copy_file(leann::index_file_from_prefix(prefix_a),
                                   corrupt_index_footer);
        flip_byte(corrupt_index_footer,
                  std::filesystem::file_size(corrupt_index_footer) - 1U);
        expect_error(
            [&] { (void)leann::Index::load(corrupt_index_footer); },
            "SHA-256 checksum mismatch");

        const auto truncated_index = directory / "index-truncated.leann";
        std::filesystem::copy_file(leann::index_file_from_prefix(prefix_a),
                                   truncated_index);
        std::filesystem::resize_file(
            truncated_index, std::filesystem::file_size(truncated_index) - 1U);
        expect_error([&] { (void)leann::Index::load(truncated_index); },
                     "SHA-256 checksum mismatch");

        const auto appended_index = directory / "index-appended.leann";
        std::filesystem::copy_file(leann::index_file_from_prefix(prefix_a),
                                   appended_index);
        append_byte(appended_index);
        expect_error([&] { (void)leann::Index::load(appended_index); },
                     "SHA-256 checksum mismatch");

        const auto corrupt_documents = directory / "documents-data.docs";
        std::filesystem::copy_file(
            leann::documents_file_from_prefix(prefix_a), corrupt_documents);
        flip_byte(corrupt_documents,
                  std::filesystem::file_size(corrupt_documents) - 1U);
        auto lazy_corrupt_store =
            leann::DocumentStore::open(corrupt_documents);
        expect_error(
            [&] {
                (void)lazy_corrupt_store.read(
                    static_cast<std::uint32_t>(corpus_a.size() - 1U));
            },
            "CRC32C checksum mismatch");

        const auto corrupt_metadata = directory / "documents-metadata.docs";
        std::filesystem::copy_file(
            leann::documents_file_from_prefix(prefix_a), corrupt_metadata);
        constexpr std::uint64_t document_header_bytes = 8U + 4U + 32U + 8U;
        const std::uint64_t metadata_digest_offset =
            document_header_bytes + 32U +
            (static_cast<std::uint64_t>(corpus_a.size()) + 1U) * 8U +
            static_cast<std::uint64_t>(corpus_a.size()) * 4U;
        flip_byte(corrupt_metadata, metadata_digest_offset);
        expect_error(
            [&] { (void)leann::DocumentStore::open(corrupt_metadata); },
            "metadata SHA-256 checksum mismatch");

        const auto corrupt_offset_table = directory / "documents-offset.docs";
        std::filesystem::copy_file(
            leann::documents_file_from_prefix(prefix_a),
            corrupt_offset_table);
        constexpr std::uint64_t document_table_offset =
            document_header_bytes + 32U;
        flip_byte(corrupt_offset_table, document_table_offset + 8U);
        expect_error(
            [&] { (void)leann::DocumentStore::open(corrupt_offset_table); },
            "metadata SHA-256 checksum mismatch");

        const auto corrupt_crc_table = directory / "documents-crc.docs";
        std::filesystem::copy_file(
            leann::documents_file_from_prefix(prefix_a), corrupt_crc_table);
        const std::uint64_t crc_table_offset =
            document_table_offset +
            (static_cast<std::uint64_t>(corpus_a.size()) + 1U) * 8U;
        flip_byte(corrupt_crc_table, crc_table_offset);
        expect_error(
            [&] { (void)leann::DocumentStore::open(corrupt_crc_table); },
            "metadata SHA-256 checksum mismatch");

        const auto corrupt_header = directory / "documents-header.docs";
        std::filesystem::copy_file(
            leann::documents_file_from_prefix(prefix_a), corrupt_header);
        flip_byte(corrupt_header, document_header_bytes - 1U);
        expect_error(
            [&] { (void)leann::DocumentStore::open(corrupt_header); },
            "header SHA-256 checksum mismatch");

        const auto truncated_documents = directory / "documents-truncated.docs";
        std::filesystem::copy_file(
            leann::documents_file_from_prefix(prefix_a), truncated_documents);
        std::filesystem::resize_file(
            truncated_documents,
            std::filesystem::file_size(truncated_documents) - 1U);
        expect_error(
            [&] { (void)leann::DocumentStore::open(truncated_documents); },
            "payload size does not match");

        const auto appended_documents = directory / "documents-appended.docs";
        std::filesystem::copy_file(
            leann::documents_file_from_prefix(prefix_a), appended_documents);
        append_byte(appended_documents);
        expect_error(
            [&] { (void)leann::DocumentStore::open(appended_documents); },
            "payload size does not match");

        const auto hostile_documents = directory / "hostile.docs";
        {
            std::ofstream output(hostile_documents,
                                 std::ios::binary | std::ios::trunc);
            output.write("LEANDC02", 8);
            write_le<std::uint32_t>(output, 2U);
            std::array<std::uint8_t, 32> identity{};
            identity.front() = 1U;
            output.write(reinterpret_cast<const char *>(identity.data()),
                         static_cast<std::streamsize>(identity.size()));
            write_le<std::uint64_t>(
                output, std::numeric_limits<std::uint64_t>::max());
            output.close();
        }
        leann::detail::append_sha256_footer(hostile_documents);
        expect_error(
            [&] { (void)leann::DocumentStore::open(hostile_documents); },
            "metadata exceeds file size");

        const auto stable_prefix = directory / "stable";
        leann::Index::build(leann::index_file_from_prefix(stable_prefix),
                            leann::documents_file_from_prefix(stable_prefix),
                            corpus_a, build_embedder, build_config());
        const auto stable_index_before =
            read_all(leann::index_file_from_prefix(stable_prefix));
        const auto stable_documents_before =
            read_all(leann::documents_file_from_prefix(stable_prefix));
        ThrowingEmbedder throwing;
        expect_error(
            [&] {
                leann::Index::build(
                    leann::index_file_from_prefix(stable_prefix),
                    leann::documents_file_from_prefix(stable_prefix), corpus_b,
                    throwing, build_config());
            },
            "injected embedding failure");
        check(stable_index_before ==
                  read_all(leann::index_file_from_prefix(stable_prefix)),
              "failed rebuild preserves previous index bytes");
        check(stable_documents_before ==
                  read_all(leann::documents_file_from_prefix(stable_prefix)),
              "failed rebuild preserves previous document bytes");
        auto stable_index =
            leann::Index::load(leann::index_file_from_prefix(stable_prefix));
        auto stable_documents = leann::DocumentStore::open(
            leann::documents_file_from_prefix(stable_prefix));
        stable_index.validate_document_store(stable_documents);
        leann::Index::build(leann::index_file_from_prefix(stable_prefix),
                            leann::documents_file_from_prefix(stable_prefix),
                            corpus_b, build_embedder, build_config());
        stable_index =
            leann::Index::load(leann::index_file_from_prefix(stable_prefix));
        stable_documents = leann::DocumentStore::open(
            leann::documents_file_from_prefix(stable_prefix));
        stable_index.validate_document_store(stable_documents);
        check(stable_documents.read(0) == corpus_b[0],
              "successful replacement publishes the new matching pair");
        check_no_build_residue(directory);

        const auto locked_prefix = directory / "locked";
        auto lock_path = leann::index_file_from_prefix(locked_prefix);
        lock_path += ".lock";
        std::filesystem::create_directory(lock_path);
        expect_error(
            [&] {
                leann::Index::build(
                    leann::index_file_from_prefix(locked_prefix),
                    leann::documents_file_from_prefix(locked_prefix), corpus_a,
                    build_embedder, build_config());
            },
            "build lock");
        check(std::filesystem::is_directory(lock_path),
              "pre-existing lock remains human-owned");
        std::filesystem::remove(lock_path);
        check_no_build_residue(directory);

        const auto polluted_prefix = directory / "polluted-lock";
        auto polluted_lock =
            std::filesystem::absolute(
                leann::index_file_from_prefix(polluted_prefix))
                .lexically_normal();
        polluted_lock += ".lock";
        LockPollutingEmbedder polluting(polluted_lock);
        expect_error(
            [&] {
                leann::Index::build(
                    leann::index_file_from_prefix(polluted_prefix),
                    leann::documents_file_from_prefix(polluted_prefix),
                    corpus_a, polluting, build_config());
            },
            "artifact pair committed; build lock cleanup required");
        auto polluted_index = leann::Index::load(
            leann::index_file_from_prefix(polluted_prefix));
        auto polluted_documents = leann::DocumentStore::open(
            leann::documents_file_from_prefix(polluted_prefix));
        polluted_index.validate_document_store(polluted_documents);
        check(std::filesystem::exists(polluted_lock / "external-entry"),
              "unowned lock content is retained and reported");
        std::filesystem::remove(polluted_lock / "external-entry");
        std::filesystem::remove(polluted_lock);
        check_no_build_residue(directory);

        const auto concurrent_prefix = directory / "concurrent";
        BlockingEmbedder blocking;
        std::exception_ptr builder_error;
        std::thread first_builder([&] {
            try {
                leann::Index::build(
                    leann::index_file_from_prefix(concurrent_prefix),
                    leann::documents_file_from_prefix(concurrent_prefix),
                    corpus_a, blocking, build_config());
            } catch (...) {
                builder_error = std::current_exception();
            }
        });
        blocking.wait_until_started();
        try {
            expect_error(
                [&] {
                    leann::Index::build(
                        leann::index_file_from_prefix(concurrent_prefix),
                        leann::documents_file_from_prefix(concurrent_prefix),
                        corpus_b, build_embedder, build_config());
                },
                "build lock");
        } catch (...) {
            blocking.release();
            first_builder.join();
            throw;
        }
        blocking.release();
        first_builder.join();
        if (builder_error) {
            std::rethrow_exception(builder_error);
        }
        auto concurrent_index = leann::Index::load(
            leann::index_file_from_prefix(concurrent_prefix));
        auto concurrent_documents = leann::DocumentStore::open(
            leann::documents_file_from_prefix(concurrent_prefix));
        concurrent_index.validate_document_store(concurrent_documents);
        check(concurrent_documents.read(0) == corpus_a[0],
              "winning concurrent builder publishes its own documents");
        check_no_build_residue(directory);

        // -------------------------------------------------------------
        // Embedder descriptor, artifact card, and prefixes (LEANNC04).
        // -------------------------------------------------------------
        {
            auto descriptor_config = build_config();
            descriptor_config.model_source = "hf:owner/repo/model.gguf";
            // Deliberately different lengths AND different content, so a
            // write/read transposition of the two prefix fields cannot round
            // trip cleanly. The same reasoning applies to the two u32s below,
            // which are read back as distinct values.
            descriptor_config.document_prefix = "search_document: ";
            descriptor_config.query_prefix = "q: ";
            descriptor_config.card = {{"license", "apache-2.0"},
                                      {"corpus", "persistence fixture"}};

            const auto descriptor_prefix = directory / "descriptor";
            DescribedEmbedder described_embedder;
            leann::Index::build(
                leann::index_file_from_prefix(descriptor_prefix),
                leann::documents_file_from_prefix(descriptor_prefix), corpus_a,
                described_embedder, descriptor_config);
            const auto descriptor_index = leann::Index::load(
                leann::index_file_from_prefix(descriptor_prefix));
            const auto descriptor_stats = descriptor_index.stats();
            check(descriptor_stats.model_source == "hf:owner/repo/model.gguf",
                  "model source round trips");
            check(descriptor_stats.document_prefix == "search_document: ",
                  "document prefix round trips");
            check(descriptor_stats.query_prefix == "q: ",
                  "query prefix round trips");
            check(descriptor_index.document_prefix() == "search_document: " &&
                      descriptor_index.query_prefix() == "q: ",
                  "prefix accessors agree with stats");
            check(descriptor_stats.card.size() == 2U &&
                      descriptor_stats.card[0].first == "license" &&
                      descriptor_stats.card[0].second == "apache-2.0" &&
                      descriptor_stats.card[1].first == "corpus",
                  "artifact card round trips in insertion order");
            // Distinct values, checked individually: the two u32 fields are
            // the same width, so a transposed write/read pair would preserve
            // the file size and survive every structural check.
            check(descriptor_stats.pooling_type == 1U,
                  "pooling type round trips");
            check(descriptor_stats.context_tokens == 512U,
                  "context tokens round trips, not transposed with pooling");
            check(descriptor_stats.model_bytes == 84106624U,
                  "model size round trips");
            check(descriptor_stats.model_sha256 ==
                      "0102030405060708090a0b0c0d0e0f10"
                      "1112131415161718191a1b1c1d1e1f20",
                  "model digest round trips byte for byte and in order");

            // The prefix must never reach the document store: pair identity is
            // derived from the raw chunk bytes, and a search result must be
            // the chunk, not the chunk with a prompt glued to it.
            auto descriptor_documents = leann::DocumentStore::open(
                leann::documents_file_from_prefix(descriptor_prefix));
            check(descriptor_documents.read(0) == corpus_a[0],
                  "document prefix stays out of the document store");

            // An index built with no descriptor keeps the empty defaults, and
            // an embedder with no model file describes itself with the
            // all-zero "absent" digest rather than a hash of nothing.
            const auto plain_index = leann::Index::load(
                leann::index_file_from_prefix(stable_prefix));
            const auto plain_stats = plain_index.stats();
            check(plain_index.document_prefix().empty() &&
                      plain_index.query_prefix().empty() &&
                      plain_stats.card.empty(),
                  "an index built without a descriptor carries empty defaults");
            check(plain_stats.model_sha256 == std::string(64, '0') &&
                      plain_stats.model_bytes == 0 &&
                      plain_stats.model_source.empty(),
                  "an embedder with no model file reports an absent model");
        }

        {
            // Write-side rejections. Each must fail before the index file is
            // created, so a bad descriptor never leaves a partial artifact.
            const auto rejected_prefix = directory / "rejected";
            const auto attempt = [&](leann::BuildConfig config) {
                return [&, config] {
                    leann::Index::build(
                        leann::index_file_from_prefix(rejected_prefix),
                        leann::documents_file_from_prefix(rejected_prefix),
                        corpus_a, build_embedder, config);
                };
            };
            auto duplicate = build_config();
            duplicate.card = {{"license", "a"}, {"license", "b"}};
            expect_error(attempt(duplicate), "duplicate artifact card key");

            auto control = build_config();
            control.card = {{"note", "one\ntwo"}};
            expect_error(attempt(control), "must not contain control");

            auto newline_prefix = build_config();
            newline_prefix.document_prefix = "bad\nprefix";
            expect_error(attempt(newline_prefix), "must not contain control");

            auto long_source = build_config();
            long_source.model_source =
                std::string(leann::max_model_source_bytes + 1U, 'x');
            expect_error(attempt(long_source), "model source is too long");

            auto long_prefix = build_config();
            long_prefix.query_prefix =
                std::string(leann::max_prefix_bytes + 1U, 'x');
            expect_error(attempt(long_prefix), "query prefix is too long");

            auto many_entries = build_config();
            for (std::size_t i = 0; i <= leann::max_card_entries; ++i) {
                many_entries.card.emplace_back("k" + std::to_string(i), "v");
            }
            expect_error(attempt(many_entries),
                         "artifact card has too many entries");

            auto invalid_utf8 = build_config();
            invalid_utf8.card = {{"note", std::string("\xff\xfe")}};
            expect_error(attempt(invalid_utf8), "must be valid UTF-8");

            check(!std::filesystem::exists(
                      leann::index_file_from_prefix(rejected_prefix)),
                  "a rejected descriptor publishes no index");
            check_no_build_residue(directory);
        }

        {
            // A LEANNC03 artifact must name the migration boundary rather than
            // report that it is not a leann.cpp index at all. Nothing else in
            // the tree covers old-version rejection.
            const auto legacy_index = directory / "legacy.leann";
            {
                std::ofstream output(legacy_index,
                                     std::ios::binary | std::ios::trunc);
                output.write("LEANNC03", 8);
                write_le<std::uint32_t>(output, 3U);
                write_le<std::uint32_t>(output, 1U);
                output.close();
            }
            leann::detail::append_sha256_footer(legacy_index);
            expect_error([&] { (void)leann::Index::load(legacy_index); },
                         "LEANNC03 format");
            expect_error([&] { (void)leann::Index::load(legacy_index); },
                         "rebuild the pair");

            const auto alien_index = directory / "alien.leann";
            {
                std::ofstream output(alien_index,
                                     std::ios::binary | std::ios::trunc);
                output.write("NOTLEANN", 8);
                write_le<std::uint32_t>(output, 4U);
                output.close();
            }
            leann::detail::append_sha256_footer(alien_index);
            expect_error([&] { (void)leann::Index::load(alien_index); },
                         "not a leann.cpp index");
        }

        std::filesystem::remove_all(directory);
        std::cout << "all leann.cpp persistence tests passed\n";
        return 0;
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
}
