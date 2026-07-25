#pragma once

#include "leann/document_store.hpp"
#include "leann/embedder.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace leann {

enum class ApproximationKind : std::uint32_t {
    SimHash = 1,
    ProductQuantization = 2,
};

// Reported to BuildConfig::report_progress as a build advances. `stage` names
// one of the build's ordered phases and is a literal with static storage
// duration. `total` is 0 when the phase length is not known in advance.
struct BuildProgress {
    std::string_view stage;
    std::uint64_t completed = 0;
    std::uint64_t total = 0;
};

// Thrown by Index::build when BuildConfig::should_cancel returns true. Build
// cancellation is fail-closed: the partially written temporary artifacts and
// the build locks are removed while unwinding, and no existing artifact pair
// is modified, because cancellation is never observed once the publication
// transaction has begun.
class BuildCancelled : public std::runtime_error {
  public:
    explicit BuildCancelled(const std::string & message)
        : std::runtime_error(message) {}
};

struct BuildConfig {
    std::uint32_t graph_degree = 16;
    std::uint32_t ef_construction = 100;
    std::uint32_t low_degree = 3;
    double hub_ratio = 0.02;
    ApproximationKind approximation = ApproximationKind::ProductQuantization;
    std::uint32_t sketch_bits = 128;
    std::uint32_t pq_subquantizers = 64;
    std::uint32_t pq_bits = 4;
    std::uint32_t pq_training_iterations = 10;
    std::uint32_t pq_training_samples = 4096;
    std::uint32_t embedding_batch_size = 32;
    std::uint32_t random_seed = 42;
    // Optional observers. Both are called on the thread that called build, may
    // be empty, and must not throw BuildCancelled themselves.
    std::function<void(const BuildProgress &)> report_progress{};
    std::function<bool()> should_cancel{};
};

struct SearchConfig {
    std::uint32_t top_k = 3;
    std::uint32_t ef_search = 64;
    std::uint32_t recompute_batch_size = 16;
    std::uint32_t approximate_scan_limit = 100000;
    double rerank_ratio = 0.25;
};

struct SearchMetrics {
    std::uint64_t approximate_distances = 0;
    std::uint64_t exact_recomputations = 0;
    std::uint64_t expanded_nodes = 0;
    std::uint64_t embedding_batches = 0;
    std::uint64_t upper_layer_hops = 0;
    double elapsed_ms = 0.0;
};

struct SearchResult {
    std::uint32_t id = 0;
    float distance = 0.0F;
};

struct SearchResponse {
    std::vector<SearchResult> results;
    SearchMetrics metrics;
};

struct IndexStats {
    std::uint64_t nodes = 0;
    std::uint64_t edges = 0;
    std::uint32_t dimension = 0;
    std::uint32_t sketch_bits = 0;
    std::uint32_t max_degree = 0;
    std::uint32_t entry_point = 0;
    std::uint32_t max_level = 0;
    std::uint64_t upper_edges = 0;
    std::uint64_t approximation_code_bytes = 0;
    std::uint64_t approximation_codebook_bytes = 0;
    std::uint64_t serialized_bytes = 0;
    std::uint64_t dense_vector_bytes_avoided = 0;
    std::string approximation;
    std::string pair_identity;
    std::string embedder_fingerprint;
};

class Index {
  public:
    static void build(const std::filesystem::path & index_path,
                      const std::filesystem::path & documents_path,
                      std::span<const std::string> documents,
                      Embedder & embedder,
                      const BuildConfig & config = {});

    static Index load(const std::filesystem::path & index_path);

    [[nodiscard]] SearchResponse search(std::string_view query,
                                        Embedder & embedder,
                                        const DocumentStore & documents,
                                        const SearchConfig & config = {}) const;
    [[nodiscard]] SearchResponse
    search_embedding(std::span<const float> query_embedding,
                     Embedder & embedder,
                     const DocumentStore & documents,
                     const SearchConfig & config = {}) const;

    [[nodiscard]] IndexStats stats() const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const std::string & embedder_fingerprint() const noexcept;
    [[nodiscard]] const PairIdentity & pair_identity() const noexcept;
    void validate_document_store(const DocumentStore & documents) const;

  private:
    struct UpperLayer {
        std::vector<std::uint32_t> nodes;
        std::vector<std::uint64_t> offsets;
        std::vector<std::uint32_t> edges;
    };

    std::filesystem::path path_;
    std::uint32_t dimension_ = 0;
    ApproximationKind approximation_ = ApproximationKind::SimHash;
    std::uint32_t sketch_bits_ = 0;
    std::uint32_t max_degree_ = 0;
    std::uint32_t entry_point_ = 0;
    std::uint32_t max_level_ = 0;
    std::uint64_t sketch_seed_ = 0;
    std::uint32_t pq_subquantizers_ = 0;
    std::uint32_t pq_bits_ = 0;
    std::uint32_t pq_centroids_ = 0;
    std::uint32_t pq_subdimension_ = 0;
    PairIdentity pair_identity_{};
    std::string embedder_fingerprint_;
    std::vector<std::uint64_t> offsets_;
    std::vector<std::uint32_t> edges_;
    std::vector<UpperLayer> upper_layers_;
    std::vector<std::uint64_t> sketches_;
    std::vector<float> pq_codebook_;
    std::vector<std::uint8_t> pq_codes_;
};

[[nodiscard]] std::filesystem::path
index_file_from_prefix(const std::filesystem::path & prefix);
[[nodiscard]] std::filesystem::path
documents_file_from_prefix(const std::filesystem::path & prefix);

} // namespace leann
