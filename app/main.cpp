#include "leann/document_store.hpp"
#include "leann/embedder.hpp"
#include "leann/index.hpp"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <array>
#include <bit>
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
        const unsigned long parsed = std::stoul(value);
        if (parsed > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(std::string(key) + " is too large");
        }
        return static_cast<std::uint32_t>(parsed);
    }

    [[nodiscard]] int int_value(std::string_view key, int fallback) const {
        const std::string value = get(key);
        return value.empty() ? fallback : std::stoi(value);
    }

    [[nodiscard]] double double_value(std::string_view key,
                                      double fallback) const {
        const std::string value = get(key);
        return value.empty() ? fallback : std::stod(value);
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

void write_embedding_cache(
    const std::filesystem::path & path,
    std::string_view fingerprint,
    std::span<const leann::Embedding> embeddings) {
    if (fingerprint.size() > std::numeric_limits<std::uint32_t>::max() ||
        embeddings.empty() || embeddings.front().empty()) {
        throw std::invalid_argument("invalid embedding cache contents");
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create embedding cache: " +
                                 temporary.string());
    }
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
    output.close();
    if (!output) {
        throw std::runtime_error("failed to finalize embedding cache");
    }
    std::error_code error;
    std::filesystem::remove(path, error);
    error.clear();
    std::filesystem::rename(temporary, path, error);
    if (error) {
        throw std::runtime_error("failed to publish embedding cache: " +
                                 error.message());
    }
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
        << "  --embedder hash|llama     hash is a deterministic test backend\n"
        << "  --model FILE              GGUF embedding model for llama.cpp\n"
        << "  --hash-dim N              hash backend dimension (default 256)\n"
        << "  --ctx N --batch-tokens N --parallel N --threads N --gpu-layers N\n\n"
        << "Build options:\n"
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
        << "  --max-queries N            deterministic prefix; 0 means all\n";
}

void command_build(const Arguments & args) {
    const auto prefix = std::filesystem::path(args.require("--index"));
    const auto index_path = leann::index_file_from_prefix(prefix);
    const auto documents_path = leann::documents_file_from_prefix(prefix);
    const auto documents = read_lines(args.require("--docs"));
    if (documents.empty()) {
        throw std::runtime_error("document input contains no non-empty lines");
    }
    if (!index_path.parent_path().empty()) {
        std::filesystem::create_directories(index_path.parent_path());
    }

    auto embedder = make_embedder(args);
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

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t position = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
    return values[std::min(position, values.size() - 1)];
}

void command_bench(const Arguments & args) {
    const auto prefix = std::filesystem::path(args.require("--index"));
    auto index = leann::Index::load(leann::index_file_from_prefix(prefix));
    auto documents =
        leann::DocumentStore::open(leann::documents_file_from_prefix(prefix));
    index.validate_document_store(documents);
    auto embedder = make_embedder(args);
    auto queries = read_lines(args.require("--queries"));
    if (queries.empty()) {
        throw std::runtime_error("query file contains no non-empty lines");
    }
    const std::uint32_t max_queries =
        args.unsigned_value("--max-queries", 0);
    if (max_queries > 0 && queries.size() > max_queries) {
        queries.resize(max_queries);
    }
    const auto config = search_config(args);
    const std::uint32_t batch_size =
        args.unsigned_value("--ground-truth-batch", 64);
    if (batch_size == 0) {
        throw std::invalid_argument("--ground-truth-batch must be positive");
    }

    const std::string cache_argument = args.get("--ground-truth-cache");
    const auto cache_path = std::filesystem::path(cache_argument);
    std::vector<leann::Embedding> corpus;
    if (!cache_argument.empty() && std::filesystem::exists(cache_path)) {
        corpus = load_embedding_cache(
            cache_path, embedder->fingerprint(), documents.size(),
            embedder->dimension());
        std::cerr << "loaded_ground_truth_cache=" << cache_path << '\n';
    } else {
        corpus.reserve(documents.size());
        for (std::size_t begin = 0; begin < documents.size();
             begin += batch_size) {
            const std::size_t end =
                std::min<std::size_t>(documents.size(), begin + batch_size);
            std::vector<std::uint32_t> ids(end - begin);
            std::iota(ids.begin(), ids.end(),
                      static_cast<std::uint32_t>(begin));
            auto batch_documents = documents.read_many(ids);
            auto batch_embeddings = embedder->embed(batch_documents);
            if (batch_embeddings.size() != batch_documents.size()) {
                throw std::runtime_error(
                    "embedder returned the wrong ground-truth batch size");
            }
            for (auto & embedding : batch_embeddings) {
                if (embedding.size() != embedder->dimension()) {
                    throw std::runtime_error(
                        "embedder returned a wrong-dimension ground-truth "
                        "vector");
                }
                leann::normalize(embedding);
                corpus.push_back(std::move(embedding));
            }
        }
        if (!cache_argument.empty()) {
            write_embedding_cache(cache_path, embedder->fingerprint(), corpus);
            std::cerr << "wrote_ground_truth_cache=" << cache_path << '\n';
        }
    }

    const bool run_dense_baseline =
        args.bool_value("--dense-baseline", false);
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

    double recall_sum = 0.0;
    std::uint64_t exact_sum = 0;
    std::uint64_t approximate_sum = 0;
    std::uint64_t upper_hops_sum = 0;
    std::vector<double> latencies;
    latencies.reserve(queries.size());
    double dense_recall_sum = 0.0;
    std::vector<double> dense_latencies;
    dense_latencies.reserve(queries.size());
    std::vector<float> padded_query(dense_dimension, 0.0F);

    for (const std::string & query : queries) {
        const std::array<std::string, 1> query_batch{query};
        auto embedded_query = embedder->embed(query_batch);
        if (embedded_query.size() != 1 ||
            embedded_query.front().size() != embedder->dimension()) {
            throw std::runtime_error(
                "embedder returned an invalid benchmark query batch");
        }
        leann::normalize(embedded_query.front());
        const auto truth =
            exact_top_k(embedded_query.front(), corpus, config.top_k);
        std::unordered_set<std::uint32_t> truth_set(truth.begin(), truth.end());

        if (dense_index) {
            std::fill(padded_query.begin(), padded_query.end(), 0.0F);
            std::copy(embedded_query.front().begin(),
                      embedded_query.front().end(), padded_query.begin());
            const auto dense_started = std::chrono::steady_clock::now();
            auto dense_results =
                dense_index->searchKnn(padded_query.data(), config.top_k);
            dense_latencies.push_back(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - dense_started)
                    .count());
            std::size_t dense_found = 0;
            while (!dense_results.empty()) {
                const auto label =
                    static_cast<std::uint32_t>(dense_results.top().second);
                dense_found += truth_set.contains(label) ? 1U : 0U;
                dense_results.pop();
            }
            dense_recall_sum +=
                truth.empty()
                    ? 1.0
                    : static_cast<double>(dense_found) / truth.size();
        }

        const auto response =
            index.search_embedding(embedded_query.front(), *embedder, documents,
                                   config);

        std::size_t found = 0;
        for (const auto & result : response.results) {
            found += truth_set.contains(result.id) ? 1U : 0U;
        }
        recall_sum +=
            truth.empty() ? 1.0
                          : static_cast<double>(found) / truth.size();
        exact_sum += response.metrics.exact_recomputations;
        approximate_sum += response.metrics.approximate_distances;
        upper_hops_sum += response.metrics.upper_layer_hops;
        latencies.push_back(response.metrics.elapsed_ms);
    }

    const double count = static_cast<double>(queries.size());
    std::cout << "queries=" << queries.size() << '\n'
              << "recall_at_" << config.top_k << '=' << std::fixed
              << std::setprecision(6) << recall_sum / count << '\n'
              << "latency_ms_mean="
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
