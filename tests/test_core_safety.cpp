#include "leann/document_store.hpp"
#include "leann/embedder.hpp"
#include "leann/index.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t embedding_dimension = 64;

void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error("test failed: " + message);
    }
}

template <typename Exception>
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
    } catch (const std::exception & error) {
        throw std::runtime_error(
            "test failed: expected a different exception type; got '" +
            std::string(error.what()) + "'");
    }
    throw std::runtime_error("test failed: expected error containing '" +
                             std::string(expected_message) + "'");
}

std::vector<std::string> make_corpus() {
    std::vector<std::string> documents;
    documents.reserve(32);
    for (std::uint32_t id = 0; id < 32U; ++id) {
        const char fill = static_cast<char>('a' + (id % 26U));
        documents.push_back(
            "document " + std::to_string(id) +
            " local retrieval concurrency checksum payload " +
            std::string(512U + id * 7U, fill));
    }
    return documents;
}

leann::BuildConfig build_config() {
    leann::BuildConfig config;
    config.graph_degree = 4;
    config.ef_construction = 20;
    config.low_degree = 2;
    config.hub_ratio = 0.125;
    config.approximation = leann::ApproximationKind::SimHash;
    config.sketch_bits = 64;
    config.embedding_batch_size = 8;
    config.random_seed = 7;
    return config;
}

leann::SearchConfig search_config() {
    leann::SearchConfig config;
    config.top_k = 2;
    config.ef_search = 6;
    config.recompute_batch_size = 3;
    config.approximate_scan_limit = 1000;
    config.rerank_ratio = 0.5;
    return config;
}

class CountingHashEmbedder final : public leann::Embedder {
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
    leann::HashEmbedder inner_{embedding_dimension};
};

class NonFiniteEmbedder final : public leann::Embedder {
  public:
    explicit NonFiniteEmbedder(float value) : value_(value) {}

    [[nodiscard]] std::size_t dimension() const noexcept override {
        return embedding_dimension;
    }

    [[nodiscard]] std::string fingerprint() const override {
        return "leann-hash-v1:" + std::to_string(embedding_dimension);
    }

    [[nodiscard]] std::vector<leann::Embedding>
    embed(std::span<const std::string> texts) override {
        ++calls;
        std::vector<leann::Embedding> result(
            texts.size(), leann::Embedding(embedding_dimension, 1.0F));
        for (auto & embedding : result) {
            embedding.front() = value_;
        }
        return result;
    }

    std::size_t calls = 0;

  private:
    float value_;
};

struct NonFiniteCase {
    std::string_view name;
    float value;
};

std::array<NonFiniteCase, 3> non_finite_cases() {
    return {{
        {"NaN", std::numeric_limits<float>::quiet_NaN()},
        {"+infinity", std::numeric_limits<float>::infinity()},
        {"-infinity", -std::numeric_limits<float>::infinity()},
    }};
}

class StartGate {
  public:
    explicit StartGate(std::size_t participants)
        : participants_(participants) {}

    void arrive_and_wait() {
        std::unique_lock lock(mutex_);
        ++arrived_;
        condition_.notify_all();
        condition_.wait(lock, [&] { return released_; });
    }

    void release_when_ready() {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return arrived_ == participants_; });
        released_ = true;
        condition_.notify_all();
    }

  private:
    std::size_t participants_;
    std::size_t arrived_ = 0;
    bool released_ = false;
    std::mutex mutex_;
    std::condition_variable condition_;
};

void flip_last_byte(const std::filesystem::path & path) {
    const std::uintmax_t size = std::filesystem::file_size(path);
    check(size > 0U, "document fixture must not be empty");
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
        throw std::runtime_error("cannot open document fixture for mutation");
    }
    const auto offset = static_cast<std::streamoff>(size - 1U);
    file.seekg(offset);
    const int original = file.get();
    if (original == std::char_traits<char>::eof()) {
        throw std::runtime_error("cannot read document fixture byte");
    }
    file.seekp(offset);
    file.put(static_cast<char>(static_cast<unsigned char>(original) ^ 0x01U));
    file.close();
    if (!file) {
        throw std::runtime_error("cannot finalize document fixture mutation");
    }
}

void test_non_finite_build_config(
    const std::filesystem::path & directory,
    std::span<const std::string> corpus) {
    for (const auto & test_case : non_finite_cases()) {
        CountingHashEmbedder embedder;
        auto config = build_config();
        config.hub_ratio = static_cast<double>(test_case.value);
        const auto prefix =
            directory / ("invalid-hub-" + std::string(test_case.name));
        expect_error<std::invalid_argument>(
            [&] {
                leann::Index::build(
                    leann::index_file_from_prefix(prefix),
                    leann::documents_file_from_prefix(prefix), corpus,
                    embedder, config);
            },
            "hub_ratio");
        check(embedder.calls == 0,
              "invalid hub_ratio must be rejected before embedding for " +
                  std::string(test_case.name));
    }
}

void test_non_finite_build_embeddings(
    const std::filesystem::path & directory,
    std::span<const std::string> corpus) {
    for (const auto & test_case : non_finite_cases()) {
        NonFiniteEmbedder embedder(test_case.value);
        const auto prefix =
            directory / ("invalid-build-embedding-" +
                         std::string(test_case.name));
        expect_error<std::runtime_error>(
            [&] {
                leann::Index::build(
                    leann::index_file_from_prefix(prefix),
                    leann::documents_file_from_prefix(prefix), corpus,
                    embedder, build_config());
            },
            "embedder returned NaN or infinity");
        check(embedder.calls == 1,
              "build must reject the first non-finite embedding batch for " +
                  std::string(test_case.name));
    }
}

void test_normalize_rejects_non_finite() {
    for (const auto & test_case : non_finite_cases()) {
        leann::Embedding embedding{1.0F, test_case.value};
        expect_error<std::invalid_argument>(
            [&] { leann::normalize(embedding); },
            "embedding contains NaN or infinity");
    }
}

void test_non_finite_search_inputs(
    const leann::Index & index,
    const leann::DocumentStore & documents) {
    for (const auto & test_case : non_finite_cases()) {
        {
            CountingHashEmbedder embedder;
            auto config = search_config();
            config.rerank_ratio = static_cast<double>(test_case.value);
            expect_error<std::invalid_argument>(
                [&] {
                    (void)index.search("safe local retrieval", embedder,
                                       documents, config);
                },
                "rerank_ratio");
            check(embedder.calls == 0,
                  "invalid rerank_ratio must be rejected before query "
                  "embedding for " +
                      std::string(test_case.name));
        }

        {
            NonFiniteEmbedder embedder(test_case.value);
            expect_error<std::runtime_error>(
                [&] {
                    (void)index.search("unsafe query embedding", embedder,
                                       documents, search_config());
                },
                "embedder returned NaN or infinity");
            check(embedder.calls == 1,
                  "text query must reject its non-finite embedding before "
                  "recomputation for " +
                      std::string(test_case.name));
        }

        {
            CountingHashEmbedder embedder;
            leann::Embedding query(embedding_dimension, 1.0F);
            query.front() = test_case.value;
            expect_error<std::invalid_argument>(
                [&] {
                    (void)index.search_embedding(query, embedder, documents,
                                                 search_config());
                },
                "query embedding contains NaN or infinity");
            check(embedder.calls == 0,
                  "direct non-finite query must be rejected before document "
                  "recomputation for " +
                      std::string(test_case.name));
        }
    }
}

void test_non_finite_recomputed_embeddings(
    const leann::Index & index,
    const leann::DocumentStore & documents) {
    leann::HashEmbedder query_embedder(embedding_dimension);
    const std::array<std::string, 1> queries{"local retrieval checksum"};
    const leann::Embedding query = query_embedder.embed(queries).front();

    for (const auto & test_case : non_finite_cases()) {
        NonFiniteEmbedder embedder(test_case.value);
        expect_error<std::runtime_error>(
            [&] {
                (void)index.search_embedding(query, embedder, documents,
                                             search_config());
            },
            "embedder returned NaN or infinity");
        check(embedder.calls == 1,
              "reranking must reject the first non-finite recomputation "
              "batch for " +
                  std::string(test_case.name));
    }
}

void test_shared_document_reads(const std::filesystem::path & documents_path,
                                std::span<const std::string> corpus) {
    auto opened = leann::DocumentStore::open(documents_path);
    auto moved = std::move(opened);
    const leann::DocumentStore & shared = moved;

    check(shared.read(0) == corpus[0],
          "a moved-to document store remains readable");
    const std::array<std::uint32_t, 4> initial_ids{3U, 1U, 31U, 7U};
    const auto initial = shared.read_many(initial_ids);
    for (std::size_t i = 0; i < initial_ids.size(); ++i) {
        check(initial[i] == corpus[initial_ids[i]],
              "a moved-to document store preserves read_many ordering");
    }

    constexpr std::size_t thread_count = 12;
    constexpr std::size_t rounds = 300;
    StartGate gate(thread_count);
    std::vector<std::exception_ptr> failures(thread_count);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t thread_id = 0; thread_id < thread_count; ++thread_id) {
        workers.emplace_back([&, thread_id] {
            try {
                gate.arrive_and_wait();
                for (std::size_t round = 0; round < rounds; ++round) {
                    const auto id = static_cast<std::uint32_t>(
                        (thread_id * 7U + round * 11U) % corpus.size());
                    check(shared.read(id) == corpus[id],
                          "concurrent read returned the wrong document");

                    const std::array<std::uint32_t, 4> ids{
                        id,
                        static_cast<std::uint32_t>((id + 5U) % corpus.size()),
                        static_cast<std::uint32_t>(
                            (id + 13U) % corpus.size()),
                        static_cast<std::uint32_t>((id + 1U) % corpus.size()),
                    };
                    const auto values = shared.read_many(ids);
                    for (std::size_t i = 0; i < ids.size(); ++i) {
                        check(values[i] == corpus[ids[i]],
                              "concurrent read_many returned the wrong "
                              "document");
                    }
                }
            } catch (...) {
                failures[thread_id] = std::current_exception();
            }
        });
    }
    gate.release_when_ready();
    for (auto & worker : workers) {
        worker.join();
    }
    for (const auto & failure : failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
}

void test_lazy_crc_rejection(const std::filesystem::path & directory,
                             const std::filesystem::path & documents_path,
                             std::span<const std::string> corpus) {
    const auto corrupt_path = directory / "lazy-crc-corrupt.docs";
    std::filesystem::copy_file(documents_path, corrupt_path);
    flip_last_byte(corrupt_path);

    const auto corrupt = leann::DocumentStore::open(corrupt_path);
    check(corrupt.read(0) == corpus[0],
          "uncorrupted payload remains readable before the corrupt document");
    expect_error<std::runtime_error>(
        [&] { (void)corrupt.read(static_cast<std::uint32_t>(corpus.size() - 1U)); },
        "CRC32C checksum mismatch");
}

} // namespace

int main() {
    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory = std::filesystem::temp_directory_path() /
                           ("leann-cpp-core-safety-" + unique);
    std::filesystem::create_directories(directory);

    try {
        const auto corpus = make_corpus();
        test_non_finite_build_config(directory, corpus);
        test_non_finite_build_embeddings(directory, corpus);
        test_normalize_rejects_non_finite();

        const auto prefix = directory / "valid";
        leann::HashEmbedder embedder(embedding_dimension);
        leann::Index::build(leann::index_file_from_prefix(prefix),
                            leann::documents_file_from_prefix(prefix), corpus,
                            embedder, build_config());
        const auto index =
            leann::Index::load(leann::index_file_from_prefix(prefix));
        const auto documents = leann::DocumentStore::open(
            leann::documents_file_from_prefix(prefix));
        index.validate_document_store(documents);

        test_non_finite_search_inputs(index, documents);
        test_non_finite_recomputed_embeddings(index, documents);
        test_shared_document_reads(
            leann::documents_file_from_prefix(prefix), corpus);
        test_lazy_crc_rejection(
            directory, leann::documents_file_from_prefix(prefix), corpus);

        std::filesystem::remove_all(directory);
        std::cout << "all leann.cpp core safety tests passed\n";
        return 0;
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
}
