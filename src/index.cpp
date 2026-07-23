#include "leann/index.hpp"

#include "format.hpp"
#include "product_quantizer.hpp"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace leann {
namespace {

constexpr std::array<char, 8> index_magic{'L', 'E', 'A', 'N', 'N', 'C', '0', '2'};
constexpr std::uint32_t index_version = 2;
constexpr std::uint32_t cosine_metric = 1;

using Clock = std::chrono::steady_clock;

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::vector<std::uint64_t> make_sketch(std::span<const float> embedding,
                                       std::uint32_t bits,
                                       std::uint64_t seed) {
    if (embedding.empty() || bits == 0 || bits % 64U != 0U) {
        throw std::invalid_argument("invalid SimHash configuration");
    }
    std::vector<std::uint64_t> result(bits / 64U, 0);
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        double projection = 0.0;
        for (std::uint64_t lane = 0; lane < 16; ++lane) {
            const std::uint64_t random = splitmix64(
                seed ^ (static_cast<std::uint64_t>(bit) << 32U) ^ lane);
            const std::size_t coordinate =
                static_cast<std::size_t>(random % embedding.size());
            const double sign = (random >> 63U) == 0U ? 1.0 : -1.0;
            projection += sign * embedding[coordinate];
        }
        if (projection >= 0.0) {
            result[bit / 64U] |= std::uint64_t{1} << (bit % 64U);
        }
    }
    return result;
}

float sketch_distance(std::span<const std::uint64_t> query,
                      std::span<const std::uint64_t> sketches,
                      std::size_t node) {
    const std::size_t words = query.size();
    std::uint64_t different = 0;
    for (std::size_t word = 0; word < words; ++word) {
        different += std::popcount(query[word] ^ sketches[node * words + word]);
    }
    return static_cast<float>(different) /
           static_cast<float>(words * std::numeric_limits<std::uint64_t>::digits);
}

void validate_embedding(const Embedding & embedding, std::size_t expected) {
    if (embedding.size() != expected) {
        throw std::runtime_error("embedder returned dimension " +
                                 std::to_string(embedding.size()) + ", expected " +
                                 std::to_string(expected));
    }
    if (!std::all_of(embedding.begin(), embedding.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("embedder returned NaN or infinity");
    }
}

void validate_build_config(const BuildConfig & config) {
    if (config.graph_degree < 2 ||
        config.graph_degree >
            std::numeric_limits<std::uint32_t>::max() / 2U) {
        throw std::invalid_argument(
            "graph_degree must be in [2, UINT32_MAX / 2]");
    }
    if (config.ef_construction < config.graph_degree) {
        throw std::invalid_argument(
            "ef_construction must be at least graph_degree");
    }
    if (config.low_degree == 0 ||
        config.low_degree > config.graph_degree * 2U) {
        throw std::invalid_argument(
            "low_degree must be in [1, 2 * graph_degree]");
    }
    if (config.hub_ratio < 0.0 || config.hub_ratio > 1.0) {
        throw std::invalid_argument("hub_ratio must be in [0, 1]");
    }
    if (config.sketch_bits < 64 || config.sketch_bits % 64U != 0U) {
        if (config.approximation == ApproximationKind::SimHash) {
            throw std::invalid_argument(
                "sketch_bits must be a multiple of 64 and at least 64");
        }
    }
    if (config.approximation == ApproximationKind::SimHash &&
        config.sketch_bits > 4096U) {
        throw std::invalid_argument("sketch_bits must not exceed 4096");
    }
    if (config.approximation == ApproximationKind::ProductQuantization) {
        if (config.pq_subquantizers == 0 || config.pq_bits == 0 ||
            config.pq_bits > 8 || config.pq_training_iterations == 0 ||
            config.pq_training_samples == 0) {
            throw std::invalid_argument("invalid product quantization settings");
        }
    } else if (config.approximation != ApproximationKind::SimHash) {
        throw std::invalid_argument("unsupported approximation kind");
    }
    if (config.embedding_batch_size == 0) {
        throw std::invalid_argument("embedding_batch_size must be positive");
    }
}

void validate_search_config(const SearchConfig & config) {
    if (config.top_k == 0 || config.ef_search < config.top_k) {
        throw std::invalid_argument("ef_search must be at least top_k > 0");
    }
    if (config.recompute_batch_size == 0) {
        throw std::invalid_argument("recompute_batch_size must be positive");
    }
    if (config.rerank_ratio <= 0.0 || config.rerank_ratio > 1.0) {
        throw std::invalid_argument("rerank_ratio must be in (0, 1]");
    }
}

template <typename T>
void write_vector(std::ostream & output, std::span<const T> values) {
    for (const T value : values) {
        detail::write_le<T>(output, value);
    }
}

template <typename T>
std::vector<T> read_vector(std::istream & input, std::size_t count) {
    std::vector<T> values(count);
    for (T & value : values) {
        value = detail::read_le<T>(input);
    }
    return values;
}

void write_float_vector(std::ostream & output, std::span<const float> values) {
    for (const float value : values) {
        detail::write_le<std::uint32_t>(output, std::bit_cast<std::uint32_t>(value));
    }
}

std::vector<float> read_float_vector(std::istream & input, std::size_t count) {
    std::vector<float> values(count);
    for (float & value : values) {
        value = std::bit_cast<float>(detail::read_le<std::uint32_t>(input));
        if (!std::isfinite(value)) {
            throw std::runtime_error("PQ codebook contains NaN or infinity");
        }
    }
    return values;
}

struct SerializedUpperLayer {
    std::vector<std::uint32_t> nodes;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint32_t> edges;
};

} // namespace

void Index::build(const std::filesystem::path & index_path,
                  const std::filesystem::path & documents_path,
                  std::span<const std::string> documents,
                  Embedder & embedder,
                  const BuildConfig & config) {
    validate_build_config(config);
    if (documents.empty()) {
        throw std::invalid_argument("cannot build an empty index");
    }
    if (documents.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("this index format supports at most 2^32-1 nodes");
    }
    if (embedder.dimension() == 0 ||
        embedder.dimension() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid embedder dimension");
    }

    auto temporary_index_path = index_path;
    temporary_index_path += ".tmp";
    auto temporary_documents_path = documents_path;
    temporary_documents_path += ".tmp";
    std::error_code cleanup_error;
    std::filesystem::remove(temporary_index_path, cleanup_error);
    cleanup_error.clear();
    std::filesystem::remove(temporary_documents_path, cleanup_error);
    struct TemporaryCleanup {
        std::filesystem::path index;
        std::filesystem::path documents;
        bool committed = false;
        ~TemporaryCleanup() {
            if (!committed) {
                std::error_code ignored;
                std::filesystem::remove(index, ignored);
                std::filesystem::remove(documents, ignored);
            }
        }
    } temporary_cleanup{temporary_index_path, temporary_documents_path};

    DocumentStore::write(temporary_documents_path, documents);

    std::vector<Embedding> embeddings;
    embeddings.reserve(documents.size());
    for (std::size_t begin = 0; begin < documents.size();
         begin += config.embedding_batch_size) {
        const std::size_t end =
            std::min(documents.size(), begin + config.embedding_batch_size);
        auto batch = embedder.embed(documents.subspan(begin, end - begin));
        if (batch.size() != end - begin) {
            throw std::runtime_error("embedder returned the wrong batch size");
        }
        for (Embedding & embedding : batch) {
            validate_embedding(embedding, embedder.dimension());
            normalize(embedding);
            embeddings.push_back(std::move(embedding));
        }
    }

    const std::size_t count = embeddings.size();
    // hnswlib places its size_t label immediately after a 4-byte-aligned
    // vector payload. With an even FP32 dimension that address is misaligned
    // for size_t on 64-bit platforms. A temporary zero coordinate preserves
    // every inner product while making hnswlib's label slot naturally aligned.
    const std::size_t hnsw_dimension =
        embedder.dimension() + (embedder.dimension() % 2U == 0U ? 1U : 0U);
    if (count > std::numeric_limits<std::size_t>::max() / hnsw_dimension) {
        throw std::runtime_error("temporary HNSW vectors are too large");
    }
    std::vector<float> hnsw_vectors(count * hnsw_dimension, 0.0F);
    for (std::size_t i = 0; i < count; ++i) {
        std::copy(embeddings[i].begin(), embeddings[i].end(),
                  hnsw_vectors.begin() +
                      static_cast<std::ptrdiff_t>(i * hnsw_dimension));
    }

    hnswlib::InnerProductSpace space(hnsw_dimension);
    hnswlib::HierarchicalNSW<float> hnsw(
        &space, count, config.graph_degree, config.ef_construction,
        config.random_seed);
    for (std::size_t i = 0; i < count; ++i) {
        hnsw.addPoint(hnsw_vectors.data() + i * hnsw_dimension,
                      static_cast<hnswlib::labeltype>(i));
    }

    std::vector<std::vector<std::uint32_t>> original(count);
    std::vector<std::uint32_t> internal_to_external(count);
    for (std::size_t internal = 0; internal < count; ++internal) {
        const auto label = hnsw.getExternalLabel(
            static_cast<hnswlib::tableint>(internal));
        if (label >= count) {
            throw std::runtime_error("hnswlib returned an invalid label");
        }
        internal_to_external[internal] = static_cast<std::uint32_t>(label);
    }
    for (std::size_t internal = 0; internal < count; ++internal) {
        const std::uint32_t source = internal_to_external[internal];
        for (const hnswlib::tableint neighbor :
             hnsw.getConnectionsWithLock(
                 static_cast<hnswlib::tableint>(internal), 0)) {
            if (neighbor >= count) {
                throw std::runtime_error("hnswlib returned an invalid edge");
            }
            original[source].push_back(internal_to_external[neighbor]);
        }
    }

    const std::uint32_t maximum_level =
        hnsw.maxlevel_ < 0 ? 0U : static_cast<std::uint32_t>(hnsw.maxlevel_);
    std::vector<SerializedUpperLayer> upper_layers(maximum_level);
    for (std::uint32_t level = 1; level <= maximum_level; ++level) {
        std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>>
            adjacency;
        for (std::size_t internal = 0; internal < count; ++internal) {
            if (hnsw.element_levels_[internal] < static_cast<int>(level)) {
                continue;
            }
            std::vector<std::uint32_t> neighbors;
            for (const hnswlib::tableint neighbor :
                 hnsw.getConnectionsWithLock(
                     static_cast<hnswlib::tableint>(internal),
                     static_cast<int>(level))) {
                if (neighbor >= count) {
                    throw std::runtime_error(
                        "hnswlib returned an invalid upper-layer edge");
                }
                neighbors.push_back(internal_to_external[neighbor]);
            }
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                            neighbors.end());
            adjacency.emplace_back(internal_to_external[internal],
                                   std::move(neighbors));
        }
        std::sort(adjacency.begin(), adjacency.end(),
                  [](const auto & lhs, const auto & rhs) {
                      return lhs.first < rhs.first;
                  });

        SerializedUpperLayer & destination = upper_layers[level - 1U];
        destination.offsets.push_back(0);
        for (auto & [node, neighbors] : adjacency) {
            destination.nodes.push_back(node);
            destination.edges.insert(destination.edges.end(), neighbors.begin(),
                                     neighbors.end());
            destination.offsets.push_back(destination.edges.size());
        }
    }

    std::vector<std::uint32_t> degree_order(count);
    std::iota(degree_order.begin(), degree_order.end(), 0U);
    std::stable_sort(degree_order.begin(), degree_order.end(),
                     [&](std::uint32_t lhs, std::uint32_t rhs) {
                         return original[lhs].size() > original[rhs].size();
                     });
    std::vector<bool> is_hub(count, false);
    const std::size_t hub_count =
        config.hub_ratio == 0.0
            ? 0
            : std::max<std::size_t>(
                  1, static_cast<std::size_t>(
                         std::ceil(config.hub_ratio * static_cast<double>(count))));
    for (std::size_t i = 0; i < std::min(count, hub_count); ++i) {
        is_hub[degree_order[i]] = true;
    }

    const std::uint32_t maximum_degree = config.graph_degree * 2U;
    std::vector<std::vector<std::uint32_t>> graph(count);
    auto closer_to = [&](std::uint32_t source, std::uint32_t lhs,
                         std::uint32_t rhs) {
        return cosine_distance(embeddings[source], embeddings[lhs]) <
               cosine_distance(embeddings[source], embeddings[rhs]);
    };

    for (std::uint32_t source = 0; source < count; ++source) {
        auto neighbors = original[source];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());
        std::stable_sort(neighbors.begin(), neighbors.end(),
                         [&](std::uint32_t lhs, std::uint32_t rhs) {
                             return closer_to(source, lhs, rhs);
                         });
        const std::size_t cap =
            is_hub[source] ? maximum_degree : config.low_degree;
        neighbors.resize(std::min(neighbors.size(), cap));
        for (const std::uint32_t neighbor : neighbors) {
            if (neighbor == source) {
                continue;
            }
            graph[source].push_back(neighbor);
            graph[neighbor].push_back(source);
        }
    }

    std::vector<std::uint64_t> offsets(count + 1, 0);
    std::vector<std::uint32_t> edges;
    for (std::uint32_t source = 0; source < count; ++source) {
        auto & neighbors = graph[source];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());
        std::stable_sort(neighbors.begin(), neighbors.end(),
                         [&](std::uint32_t lhs, std::uint32_t rhs) {
                             return closer_to(source, lhs, rhs);
                         });
        neighbors.resize(std::min<std::size_t>(neighbors.size(), maximum_degree));
        edges.insert(edges.end(), neighbors.begin(), neighbors.end());
        offsets[source + 1] = edges.size();
    }

    std::uint64_t sketch_seed = 0;
    std::vector<std::uint64_t> sketches;
    detail::ProductQuantizerModel pq;
    if (config.approximation == ApproximationKind::SimHash) {
        sketch_seed =
            splitmix64(static_cast<std::uint64_t>(config.random_seed) ^
                       0x4c45414e4e435050ULL);
        sketches.reserve(count * (config.sketch_bits / 64U));
        for (const Embedding & embedding : embeddings) {
            auto sketch =
                make_sketch(embedding, config.sketch_bits, sketch_seed);
            sketches.insert(sketches.end(), sketch.begin(), sketch.end());
        }
    } else {
        pq = detail::train_product_quantizer(
            embeddings, config.pq_subquantizers, config.pq_bits,
            config.pq_training_iterations, config.pq_training_samples,
            config.random_seed);
    }

    const std::uint32_t entry_point = static_cast<std::uint32_t>(
        hnsw.getExternalLabel(hnsw.enterpoint_node_));
    const std::string fingerprint = embedder.fingerprint();
    if (fingerprint.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("embedder fingerprint is too long");
    }

    std::ofstream output(temporary_index_path,
                         std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create index: " +
                                 temporary_index_path.string());
    }
    detail::write_bytes(output, index_magic.data(), index_magic.size());
    detail::write_le<std::uint32_t>(output, index_version);
    detail::write_le<std::uint32_t>(output, cosine_metric);
    detail::write_le<std::uint32_t>(
        output, static_cast<std::uint32_t>(embedder.dimension()));
    detail::write_le<std::uint32_t>(
        output, static_cast<std::uint32_t>(config.approximation));
    detail::write_le<std::uint32_t>(output, maximum_degree);
    detail::write_le<std::uint32_t>(output, entry_point);
    detail::write_le<std::uint32_t>(output, maximum_level);
    detail::write_le<std::uint32_t>(
        output, config.approximation == ApproximationKind::SimHash
                    ? config.sketch_bits
                    : 0U);
    detail::write_le<std::uint32_t>(output, pq.subquantizers);
    detail::write_le<std::uint32_t>(output, pq.bits);
    detail::write_le<std::uint32_t>(output, pq.centroids);
    detail::write_le<std::uint32_t>(output, pq.subdimension);
    detail::write_le<std::uint64_t>(output, sketch_seed);
    detail::write_le<std::uint64_t>(output, count);
    detail::write_le<std::uint64_t>(output, edges.size());
    detail::write_le<std::uint64_t>(output, pq.codebook.size());
    detail::write_le<std::uint64_t>(
        output, config.approximation == ApproximationKind::SimHash
                    ? sketches.size() * sizeof(std::uint64_t)
                    : pq.codes.size());
    detail::write_le<std::uint32_t>(
        output, static_cast<std::uint32_t>(fingerprint.size()));
    detail::write_bytes(output, fingerprint.data(), fingerprint.size());
    write_vector<std::uint64_t>(output, offsets);
    write_vector<std::uint32_t>(output, edges);
    for (const SerializedUpperLayer & layer : upper_layers) {
        detail::write_le<std::uint64_t>(output, layer.nodes.size());
        detail::write_le<std::uint64_t>(output, layer.edges.size());
        write_vector<std::uint32_t>(output, layer.nodes);
        write_vector<std::uint64_t>(output, layer.offsets);
        write_vector<std::uint32_t>(output, layer.edges);
    }
    if (config.approximation == ApproximationKind::SimHash) {
        write_vector<std::uint64_t>(output, sketches);
    } else {
        write_float_vector(output, pq.codebook);
        detail::write_bytes(
            output, reinterpret_cast<const char *>(pq.codes.data()),
            pq.codes.size());
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize index: " +
                                 temporary_index_path.string());
    }

    // All expensive work has succeeded. Replace the two artifacts only now,
    // so embedding/build errors leave the previous index untouched.
    std::error_code replace_error;
    std::filesystem::remove(documents_path, replace_error);
    replace_error.clear();
    std::filesystem::rename(temporary_documents_path, documents_path,
                            replace_error);
    if (replace_error) {
        throw std::runtime_error("failed to publish document store: " +
                                 replace_error.message());
    }
    std::filesystem::remove(index_path, replace_error);
    replace_error.clear();
    std::filesystem::rename(temporary_index_path, index_path, replace_error);
    if (replace_error) {
        throw std::runtime_error("failed to publish index: " +
                                 replace_error.message());
    }
    temporary_cleanup.committed = true;
}

Index Index::load(const std::filesystem::path & index_path) {
    std::ifstream input(index_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open index: " + index_path.string());
    }

    std::array<char, index_magic.size()> magic{};
    detail::read_bytes(input, magic.data(), magic.size());
    if (magic != index_magic) {
        throw std::runtime_error("not a leann.cpp index: " + index_path.string());
    }
    const std::uint32_t version = detail::read_le<std::uint32_t>(input);
    if (version != index_version) {
        throw std::runtime_error("unsupported index version: " +
                                 std::to_string(version));
    }
    const std::uint32_t metric = detail::read_le<std::uint32_t>(input);
    if (metric != cosine_metric) {
        throw std::runtime_error("unsupported index metric");
    }

    Index index;
    index.path_ = index_path;
    index.dimension_ = detail::read_le<std::uint32_t>(input);
    const std::uint32_t approximation =
        detail::read_le<std::uint32_t>(input);
    if (approximation !=
            static_cast<std::uint32_t>(ApproximationKind::SimHash) &&
        approximation != static_cast<std::uint32_t>(
                             ApproximationKind::ProductQuantization)) {
        throw std::runtime_error("unsupported approximation kind");
    }
    index.approximation_ = static_cast<ApproximationKind>(approximation);
    index.max_degree_ = detail::read_le<std::uint32_t>(input);
    index.entry_point_ = detail::read_le<std::uint32_t>(input);
    index.max_level_ = detail::read_le<std::uint32_t>(input);
    index.sketch_bits_ = detail::read_le<std::uint32_t>(input);
    index.pq_subquantizers_ = detail::read_le<std::uint32_t>(input);
    index.pq_bits_ = detail::read_le<std::uint32_t>(input);
    index.pq_centroids_ = detail::read_le<std::uint32_t>(input);
    index.pq_subdimension_ = detail::read_le<std::uint32_t>(input);
    index.sketch_seed_ = detail::read_le<std::uint64_t>(input);
    const std::uint64_t count_on_disk =
        detail::read_le<std::uint64_t>(input);
    const std::uint64_t edges_on_disk =
        detail::read_le<std::uint64_t>(input);
    const std::uint64_t codebook_values_on_disk =
        detail::read_le<std::uint64_t>(input);
    const std::uint64_t code_bytes_on_disk =
        detail::read_le<std::uint64_t>(input);
    const std::uint64_t fingerprint_size =
        detail::read_le<std::uint32_t>(input);

    if (count_on_disk == 0 || index.dimension_ == 0 ||
        index.entry_point_ >= count_on_disk || index.max_level_ > 64U) {
        throw std::runtime_error("invalid index header");
    }

    const std::uint64_t file_size = std::filesystem::file_size(index_path);
    const auto payload_position = static_cast<std::uint64_t>(input.tellg());
    if (payload_position > file_size) {
        throw std::runtime_error("invalid index file size");
    }
    std::uint64_t remaining = file_size - payload_position;
    auto consume = [&](std::uint64_t elements, std::uint64_t element_size,
                       const char * field) {
        if (element_size == 0 || elements > remaining / element_size) {
            throw std::runtime_error(std::string(field) +
                                     " exceeds index file size");
        }
        remaining -= elements * element_size;
    };
    if (count_on_disk == std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("node count is too large");
    }

    const std::size_t count =
        detail::checked_size(count_on_disk, "node count");
    const std::size_t edge_count =
        detail::checked_size(edges_on_disk, "edge count");
    const std::size_t fingerprint_size_in_memory =
        detail::checked_size(fingerprint_size, "embedder fingerprint");
    const std::size_t codebook_values = detail::checked_size(
        codebook_values_on_disk, "PQ codebook values");
    const std::size_t code_bytes =
        detail::checked_size(code_bytes_on_disk, "approximation codes");

    consume(fingerprint_size, 1, "embedder fingerprint");
    index.embedder_fingerprint_.resize(fingerprint_size_in_memory);
    detail::read_bytes(input, index.embedder_fingerprint_.data(),
                       fingerprint_size_in_memory);

    consume(count_on_disk + 1U, sizeof(std::uint64_t), "CSR offsets");
    index.offsets_ = read_vector<std::uint64_t>(input, count + 1);
    consume(edges_on_disk, sizeof(std::uint32_t), "CSR edges");
    index.edges_ = read_vector<std::uint32_t>(input, edge_count);

    if (index.offsets_.front() != 0 ||
        index.offsets_.back() != index.edges_.size()) {
        throw std::runtime_error("invalid CSR offsets");
    }
    for (std::size_t i = 1; i < index.offsets_.size(); ++i) {
        if (index.offsets_[i] < index.offsets_[i - 1] ||
            index.offsets_[i] > index.edges_.size()) {
            throw std::runtime_error("invalid CSR offsets");
        }
    }
    if (!std::all_of(index.edges_.begin(), index.edges_.end(),
                     [&](std::uint32_t id) { return id < count; })) {
        throw std::runtime_error("CSR contains an invalid node id");
    }

    index.upper_layers_.resize(index.max_level_);
    for (std::uint32_t level = 0; level < index.max_level_; ++level) {
        consume(2, sizeof(std::uint64_t), "upper-layer header");
        const std::uint64_t layer_nodes_on_disk =
            detail::read_le<std::uint64_t>(input);
        const std::uint64_t layer_edges_on_disk =
            detail::read_le<std::uint64_t>(input);
        if (layer_nodes_on_disk > count_on_disk ||
            layer_nodes_on_disk ==
                std::numeric_limits<std::uint64_t>::max()) {
            throw std::runtime_error("invalid upper-layer node count");
        }
        const std::size_t layer_nodes =
            detail::checked_size(layer_nodes_on_disk, "upper-layer nodes");
        const std::size_t layer_edges =
            detail::checked_size(layer_edges_on_disk, "upper-layer edges");
        consume(layer_nodes_on_disk, sizeof(std::uint32_t),
                "upper-layer nodes");
        consume(layer_nodes_on_disk + 1U, sizeof(std::uint64_t),
                "upper-layer offsets");
        consume(layer_edges_on_disk, sizeof(std::uint32_t),
                "upper-layer edges");

        UpperLayer & layer = index.upper_layers_[level];
        layer.nodes = read_vector<std::uint32_t>(input, layer_nodes);
        layer.offsets =
            read_vector<std::uint64_t>(input, layer_nodes + 1U);
        layer.edges = read_vector<std::uint32_t>(input, layer_edges);
        if (!std::is_sorted(layer.nodes.begin(), layer.nodes.end()) ||
            std::adjacent_find(layer.nodes.begin(), layer.nodes.end()) !=
                layer.nodes.end() ||
            !std::all_of(layer.nodes.begin(), layer.nodes.end(),
                         [&](std::uint32_t id) { return id < count; }) ||
            layer.offsets.empty() || layer.offsets.front() != 0 ||
            layer.offsets.back() != layer.edges.size()) {
            throw std::runtime_error("invalid upper-layer CSR data");
        }
        for (std::size_t i = 1; i < layer.offsets.size(); ++i) {
            if (layer.offsets[i] < layer.offsets[i - 1] ||
                layer.offsets[i] > layer.edges.size()) {
                throw std::runtime_error("invalid upper-layer offsets");
            }
        }
        if (!std::all_of(layer.edges.begin(), layer.edges.end(),
                         [&](std::uint32_t id) {
                             return id < count &&
                                    std::binary_search(layer.nodes.begin(),
                                                       layer.nodes.end(), id);
                         })) {
            throw std::runtime_error("invalid upper-layer edge");
        }
        if (!std::binary_search(layer.nodes.begin(), layer.nodes.end(),
                                index.entry_point_)) {
            throw std::runtime_error(
                "upper layer does not contain the HNSW entry point");
        }
    }

    if (index.approximation_ == ApproximationKind::SimHash) {
        if (index.sketch_bits_ < 64U || index.sketch_bits_ % 64U != 0U ||
            index.pq_subquantizers_ != 0U || index.pq_bits_ != 0U ||
            index.pq_centroids_ != 0U || index.pq_subdimension_ != 0U ||
            codebook_values != 0U) {
            throw std::runtime_error("invalid SimHash metadata");
        }
        if (count_on_disk >
            std::numeric_limits<std::uint64_t>::max() /
                (index.sketch_bits_ / 8U)) {
            throw std::runtime_error("SimHash table is too large");
        }
        const std::uint64_t expected_bytes =
            count_on_disk * (index.sketch_bits_ / 8U);
        if (code_bytes_on_disk != expected_bytes) {
            throw std::runtime_error("invalid SimHash table size");
        }
        consume(code_bytes_on_disk, 1, "SimHash table");
        index.sketches_ =
            read_vector<std::uint64_t>(input, code_bytes / sizeof(std::uint64_t));
    } else {
        if (index.sketch_bits_ != 0U || index.pq_subquantizers_ == 0U ||
            index.pq_bits_ == 0U || index.pq_bits_ > 8U ||
            index.pq_centroids_ == 0U ||
            index.pq_centroids_ > (1U << index.pq_bits_) ||
            index.pq_subdimension_ == 0U ||
            static_cast<std::uint64_t>(index.pq_subquantizers_) *
                    index.pq_subdimension_ !=
                index.dimension_) {
            throw std::runtime_error("invalid PQ metadata");
        }
        const std::uint64_t codebook_rows =
            static_cast<std::uint64_t>(index.pq_subquantizers_) *
            index.pq_centroids_;
        if (codebook_rows >
            std::numeric_limits<std::uint64_t>::max() /
                index.pq_subdimension_) {
            throw std::runtime_error("PQ codebook is too large");
        }
        const std::uint64_t expected_codebook =
            codebook_rows * index.pq_subdimension_;
        const std::uint64_t bytes_per_vector =
            (static_cast<std::uint64_t>(index.pq_subquantizers_) *
                 index.pq_bits_ +
             7U) /
            8U;
        if (count_on_disk >
            std::numeric_limits<std::uint64_t>::max() / bytes_per_vector) {
            throw std::runtime_error("PQ code table is too large");
        }
        if (codebook_values_on_disk != expected_codebook ||
            code_bytes_on_disk != count_on_disk * bytes_per_vector) {
            throw std::runtime_error("invalid PQ table size");
        }
        consume(codebook_values_on_disk, sizeof(std::uint32_t), "PQ codebook");
        index.pq_codebook_ = read_float_vector(input, codebook_values);
        consume(code_bytes_on_disk, 1, "PQ codes");
        index.pq_codes_.resize(code_bytes);
        detail::read_bytes(
            input, reinterpret_cast<char *>(index.pq_codes_.data()), code_bytes);
    }
    if (remaining != 0) {
        throw std::runtime_error("index has unexpected trailing bytes");
    }
    return index;
}

SearchResponse Index::search(std::string_view query,
                             Embedder & embedder,
                             DocumentStore & documents,
                             const SearchConfig & config) const {
    std::string owned_query(query);
    const std::array<std::string, 1> queries{std::move(owned_query)};
    auto query_embeddings = embedder.embed(queries);
    if (query_embeddings.size() != 1) {
        throw std::runtime_error("embedder returned the wrong query batch size");
    }
    normalize(query_embeddings.front());
    return search_embedding(query_embeddings.front(), embedder, documents, config);
}

SearchResponse
Index::search_embedding(std::span<const float> query_embedding,
                        Embedder & embedder,
                        DocumentStore & documents,
                        const SearchConfig & config) const {
    validate_search_config(config);
    if (embedder.fingerprint() != embedder_fingerprint_) {
        throw std::invalid_argument(
            "embedder fingerprint mismatch: index uses '" +
            embedder_fingerprint_ + "', query uses '" + embedder.fingerprint() +
            "'");
    }
    if (query_embedding.size() != dimension_ ||
        embedder.dimension() != dimension_) {
        throw std::invalid_argument("query embedding dimension mismatch");
    }
    if (documents.size() != size()) {
        throw std::invalid_argument("document store and index node counts differ");
    }

    Embedding normalized_query(query_embedding.begin(), query_embedding.end());
    normalize(normalized_query);
    const auto start = Clock::now();
    std::vector<std::uint64_t> query_sketch;
    detail::ProductQuantizerView pq_view{
        dimension_,
        pq_subquantizers_,
        pq_bits_,
        pq_centroids_,
        pq_subdimension_,
        pq_codebook_,
        pq_codes_,
    };
    std::vector<float> pq_distance_table;
    if (approximation_ == ApproximationKind::SimHash) {
        query_sketch =
            make_sketch(normalized_query, sketch_bits_, sketch_seed_);
    } else {
        pq_distance_table = pq_view.distance_table(normalized_query);
    }

    struct Candidate {
        float distance;
        std::uint32_t id;
    };
    const auto nearer_first = [](const Candidate & lhs, const Candidate & rhs) {
        return lhs.distance != rhs.distance ? lhs.distance > rhs.distance
                                            : lhs.id > rhs.id;
    };
    const auto farther_first = [](const Candidate & lhs, const Candidate & rhs) {
        return lhs.distance != rhs.distance ? lhs.distance < rhs.distance
                                            : lhs.id < rhs.id;
    };
    std::priority_queue<Candidate, std::vector<Candidate>,
                        decltype(nearer_first)>
        approximate_frontier(nearer_first);
    std::priority_queue<Candidate, std::vector<Candidate>,
                        decltype(farther_first)>
        approximate_shortlist(farther_first);

    SearchMetrics metrics;

    auto approximate_distance = [&](std::uint32_t node) {
        ++metrics.approximate_distances;
        if (approximation_ == ApproximationKind::SimHash) {
            return sketch_distance(query_sketch, sketches_, node);
        }
        return pq_view.approximate_distance(pq_distance_table, node);
    };

    // Greedily descend the compact upper HNSW layers with only approximate
    // distances. Expensive document embedding starts after this routing step.
    std::uint32_t routed_entry = entry_point_;
    float routed_distance = approximate_distance(routed_entry);
    for (std::size_t layer_index = upper_layers_.size(); layer_index-- > 0;) {
        const UpperLayer & layer = upper_layers_[layer_index];
        while (true) {
            const auto current =
                std::lower_bound(layer.nodes.begin(), layer.nodes.end(),
                                 routed_entry);
            if (current == layer.nodes.end() || *current != routed_entry) {
                throw std::runtime_error(
                    "upper-layer route does not contain its entry node");
            }
            const std::size_t position =
                static_cast<std::size_t>(current - layer.nodes.begin());
            const std::size_t begin =
                static_cast<std::size_t>(layer.offsets[position]);
            const std::size_t end =
                static_cast<std::size_t>(layer.offsets[position + 1U]);
            std::uint32_t best_node = routed_entry;
            float best_distance = routed_distance;
            for (std::size_t edge = begin; edge < end; ++edge) {
                const std::uint32_t neighbor = layer.edges[edge];
                const float distance = approximate_distance(neighbor);
                if (distance < best_distance ||
                    (distance == best_distance && neighbor < best_node)) {
                    best_node = neighbor;
                    best_distance = distance;
                }
            }
            if (best_node == routed_entry) {
                break;
            }
            routed_entry = best_node;
            routed_distance = best_distance;
            ++metrics.upper_layer_hops;
        }
    }

    const std::size_t exact_budget =
        std::min<std::size_t>(config.ef_search, size());
    const std::size_t approximate_beam = std::min<std::size_t>(
        size(), std::max<std::size_t>(
                    exact_budget,
                    static_cast<std::size_t>(std::ceil(
                        static_cast<double>(exact_budget) /
                        config.rerank_ratio))));
    std::vector<std::uint8_t> visited(size(), 0);
    visited[routed_entry] = 1;
    approximate_frontier.push({routed_distance, routed_entry});
    approximate_shortlist.push({routed_distance, routed_entry});

    if (config.approximate_scan_limit > 0 &&
        size() <= config.approximate_scan_limit) {
        approximate_shortlist.pop();
        for (std::uint32_t node = 0; node < size(); ++node) {
            const float distance =
                node == routed_entry ? routed_distance
                                     : approximate_distance(node);
            approximate_shortlist.push({distance, node});
            if (approximate_shortlist.size() > approximate_beam) {
                approximate_shortlist.pop();
            }
        }
        metrics.expanded_nodes = size();
    } else {
        while (!approximate_frontier.empty()) {
            const Candidate current = approximate_frontier.top();
            if (approximate_shortlist.size() >= approximate_beam) {
                const Candidate worst = approximate_shortlist.top();
                if (current.distance > worst.distance ||
                    (current.distance == worst.distance &&
                     current.id > worst.id)) {
                    break;
                }
            }
            approximate_frontier.pop();
            ++metrics.expanded_nodes;

            const std::size_t begin =
                static_cast<std::size_t>(offsets_[current.id]);
            const std::size_t end =
                static_cast<std::size_t>(offsets_[current.id + 1U]);
            for (std::size_t edge = begin; edge < end; ++edge) {
                const std::uint32_t neighbor = edges_[edge];
                if (visited[neighbor] != 0) {
                    continue;
                }
                visited[neighbor] = 1;
                const float distance = approximate_distance(neighbor);
                const bool competitive =
                    approximate_shortlist.size() < approximate_beam ||
                    distance < approximate_shortlist.top().distance ||
                    (distance == approximate_shortlist.top().distance &&
                     neighbor < approximate_shortlist.top().id);
                if (!competitive) {
                    continue;
                }
                const Candidate candidate{distance, neighbor};
                approximate_frontier.push(candidate);
                approximate_shortlist.push(candidate);
                if (approximate_shortlist.size() > approximate_beam) {
                    approximate_shortlist.pop();
                }
            }
        }
    }

    std::vector<Candidate> candidates;
    candidates.reserve(approximate_shortlist.size());
    while (!approximate_shortlist.empty()) {
        candidates.push_back(approximate_shortlist.top());
        approximate_shortlist.pop();
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate & lhs, const Candidate & rhs) {
                  return lhs.distance != rhs.distance
                             ? lhs.distance < rhs.distance
                             : lhs.id < rhs.id;
              });
    candidates.resize(std::min(candidates.size(), exact_budget));

    std::vector<SearchResult> ranked;
    ranked.reserve(candidates.size());
    auto evaluate = [&](std::span<const std::uint32_t> ids) {
        if (ids.empty()) {
            return;
        }
        auto texts = documents.read_many(ids);
        auto embeddings = embedder.embed(texts);
        if (embeddings.size() != ids.size()) {
            throw std::runtime_error("embedder returned the wrong batch size");
        }
        ++metrics.embedding_batches;
        metrics.exact_recomputations += ids.size();
        for (std::size_t i = 0; i < ids.size(); ++i) {
            validate_embedding(embeddings[i], dimension_);
            normalize(embeddings[i]);
            const float distance =
                cosine_distance(normalized_query, embeddings[i]);
            ranked.push_back({ids[i], distance});
        }
    };

    for (std::size_t begin = 0; begin < candidates.size();
         begin += config.recompute_batch_size) {
        const std::size_t end = std::min(
            candidates.size(), begin + config.recompute_batch_size);
        std::vector<std::uint32_t> batch;
        batch.reserve(end - begin);
        for (std::size_t candidate = begin; candidate < end; ++candidate) {
            batch.push_back(candidates[candidate].id);
        }
        evaluate(batch);
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const SearchResult & lhs, const SearchResult & rhs) {
                  if (lhs.distance != rhs.distance) {
                      return lhs.distance < rhs.distance;
                  }
                  return lhs.id < rhs.id;
              });
    ranked.resize(std::min<std::size_t>(ranked.size(), config.top_k));

    metrics.elapsed_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return {std::move(ranked), metrics};
}

IndexStats Index::stats() const {
    IndexStats result;
    result.nodes = size();
    result.edges = edges_.size();
    result.dimension = dimension_;
    result.sketch_bits = sketch_bits_;
    result.max_degree = max_degree_;
    result.entry_point = entry_point_;
    result.max_level = max_level_;
    for (const UpperLayer & layer : upper_layers_) {
        result.upper_edges += layer.edges.size();
    }
    if (approximation_ == ApproximationKind::SimHash) {
        result.approximation = "simhash";
        result.approximation_code_bytes =
            sketches_.size() * sizeof(std::uint64_t);
    } else {
        result.approximation = "pq";
        result.approximation_code_bytes = pq_codes_.size();
        result.approximation_codebook_bytes =
            pq_codebook_.size() * sizeof(float);
    }
    result.serialized_bytes = std::filesystem::file_size(path_);
    result.dense_vector_bytes_avoided =
        static_cast<std::uint64_t>(size()) * dimension_ * sizeof(float);
    result.embedder_fingerprint = embedder_fingerprint_;
    return result;
}

std::size_t Index::size() const noexcept {
    return offsets_.empty() ? 0 : offsets_.size() - 1;
}

const std::string & Index::embedder_fingerprint() const noexcept {
    return embedder_fingerprint_;
}

std::filesystem::path
index_file_from_prefix(const std::filesystem::path & prefix) {
    if (prefix.extension() == ".leann") {
        return prefix;
    }
    auto result = prefix;
    result += ".leann";
    return result;
}

std::filesystem::path
documents_file_from_prefix(const std::filesystem::path & prefix) {
    auto result = prefix;
    if (result.extension() == ".leann") {
        result.replace_extension();
    }
    result += ".docs";
    return result;
}

} // namespace leann
