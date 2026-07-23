#include "leann/document_store.hpp"
#include "leann/embedder.hpp"
#include "leann/index.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error("test failed: " + message);
    }
}

std::vector<std::string> make_corpus() {
    std::vector<std::string> documents;
    for (int i = 0; i < 50; ++i) {
        documents.push_back("apple pear orange fruit orchard harvest recipe " +
                            std::to_string(i));
        documents.push_back("kernel compiler vector database software code " +
                            std::to_string(i));
        documents.push_back("planet galaxy telescope astronomy star orbit " +
                            std::to_string(i));
    }
    return documents;
}

} // namespace

int main() {
    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory =
        std::filesystem::temp_directory_path() / ("leann-cpp-test-" + unique);
    std::filesystem::create_directories(directory);

    try {
        const auto prefix = directory / "sample";
        const auto index_path = leann::index_file_from_prefix(prefix);
        const auto documents_path = leann::documents_file_from_prefix(prefix);
        const auto corpus = make_corpus();
        leann::HashEmbedder embedder(256);

        leann::BuildConfig build;
        build.graph_degree = 12;
        build.ef_construction = 80;
        build.low_degree = 5;
        build.hub_ratio = 0.04;
        build.approximation =
            leann::ApproximationKind::ProductQuantization;
        build.pq_subquantizers = 8;
        build.pq_bits = 5;
        build.pq_training_iterations = 6;
        leann::Index::build(index_path, documents_path, corpus, embedder, build);

        auto index = leann::Index::load(index_path);
        auto documents = leann::DocumentStore::open(documents_path);
        check(index.size() == corpus.size(), "index node count");
        check(documents.size() == corpus.size(), "document count");
        check(documents.read(3) == corpus[3], "document round trip");
        check(index.stats().serialized_bytes <
                  index.stats().dense_vector_bytes_avoided,
              "index should be smaller than omitted dense vectors");
        check(index.stats().approximation == "pq", "PQ stats mode");
        check(index.stats().sketch_bits == 0, "PQ has no SimHash bits");
        check(index.stats().approximation_code_bytes == corpus.size() * 5U,
              "bit-packed PQ code size");
        check(index.stats().approximation_codebook_bytes > 0,
              "PQ codebook is persisted");
        check(index.stats().max_level > 0 &&
                  index.stats().upper_edges > 0,
              "upper HNSW layers are persisted");

        leann::SearchConfig search;
        search.top_k = 3;
        search.ef_search = 80;
        search.recompute_batch_size = 12;
        search.approximate_scan_limit = 0;
        search.rerank_ratio = 0.5;
        const auto response =
            index.search("fresh apple fruit from the orchard", embedder,
                         documents, search);
        check(response.results.size() == 3, "top-k result count");
        check(response.metrics.exact_recomputations <= search.ef_search,
              "exact recomputation budget");
        check(response.metrics.approximate_distances > 0,
              "approximate search was used");
        for (const auto & result : response.results) {
            check(result.id % 3 == 0, "fruit query should retrieve fruit docs");
        }

        bool mismatch_detected = false;
        try {
            leann::HashEmbedder wrong_embedder(128);
            (void)index.search("apple", wrong_embedder, documents, search);
        } catch (const std::invalid_argument &) {
            mismatch_detected = true;
        }
        check(mismatch_detected, "embedder fingerprint mismatch");

        const auto truncated_path = directory / "truncated.leann";
        std::filesystem::copy_file(index_path, truncated_path);
        std::filesystem::resize_file(
            truncated_path, std::filesystem::file_size(truncated_path) - 1);
        bool truncation_detected = false;
        try {
            (void)leann::Index::load(truncated_path);
        } catch (const std::runtime_error &) {
            truncation_detected = true;
        }
        check(truncation_detected, "truncated index validation");

        const auto simhash_prefix = directory / "simhash";
        leann::BuildConfig simhash_build = build;
        simhash_build.approximation = leann::ApproximationKind::SimHash;
        simhash_build.sketch_bits = 192;
        leann::Index::build(
            leann::index_file_from_prefix(simhash_prefix),
            leann::documents_file_from_prefix(simhash_prefix), corpus, embedder,
            simhash_build);
        auto simhash_index =
            leann::Index::load(leann::index_file_from_prefix(simhash_prefix));
        auto simhash_documents = leann::DocumentStore::open(
            leann::documents_file_from_prefix(simhash_prefix));
        check(simhash_index.stats().approximation == "simhash",
              "SimHash compatibility mode");
        check(simhash_index.stats().approximation_code_bytes ==
                  corpus.size() * 24U,
              "SimHash table size");
        check(simhash_index
                      .search("compiler vector database", embedder,
                              simhash_documents, search)
                      .results.size() == search.top_k,
              "SimHash search path");

        std::filesystem::remove_all(directory);
        std::cout << "all leann.cpp tests passed\n";
        return 0;
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
}
