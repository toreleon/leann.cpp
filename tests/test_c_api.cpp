#include "leann/leann.h"

#include "leann/document_store.hpp"
#include "leann/embedder.hpp"
#include "leann/index.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t embedding_dimension = 32;

void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error("test failed: " + message);
    }
}

void check_status(leann_status actual,
                  leann_status expected,
                  std::string_view message_fragment,
                  const std::string & context) {
    check(actual == expected,
          context + ": expected status " +
              std::to_string(static_cast<int>(expected)) + ", got " +
              std::to_string(static_cast<int>(actual)));
    const std::string detail = leann_last_error();
    check(detail.find(message_fragment) != std::string::npos,
          context + ": expected error containing '" +
              std::string(message_fragment) + "', got '" + detail + "'");
}

std::vector<std::string> make_corpus(std::string_view topic) {
    std::vector<std::string> documents;
    documents.reserve(24);
    documents.push_back(
        std::string("binary ") + std::string("alpha\0omega", 11) +
        " length delimited payload");
    documents.push_back("UTF-8 retrieval: cà phê và dữ liệu cục bộ");
    for (std::uint32_t id = 2; id < 24U; ++id) {
        documents.push_back(
            std::string(topic) + " document " + std::to_string(id) +
            " compact local retrieval vector graph selective recomputation");
    }
    return documents;
}

leann::BuildConfig
build_config(leann::ApproximationKind approximation) {
    leann::BuildConfig config;
    config.graph_degree = 4;
    config.ef_construction = 20;
    config.low_degree = 2;
    config.hub_ratio = 0.125;
    config.approximation = approximation;
    config.sketch_bits = 64;
    config.pq_subquantizers = 8;
    config.pq_bits = 2;
    config.pq_training_iterations = 3;
    config.pq_training_samples = 24;
    config.embedding_batch_size = 6;
    config.random_seed = 19;
    return config;
}

leann::SearchConfig non_default_search_config() {
    leann::SearchConfig config;
    config.top_k = 5;
    config.ef_search = 11;
    config.recompute_batch_size = 4;
    config.approximate_scan_limit = 0;
    config.rerank_ratio = 0.5;
    return config;
}

leann_search_options_v1
c_options(const leann::SearchConfig & config) {
    leann_search_options_v1 options{};
    check(leann_search_options_init(&options) == LEANN_STATUS_OK,
          "search options initialization");
    options.top_k = config.top_k;
    options.ef_search = config.ef_search;
    options.recompute_batch_size = config.recompute_batch_size;
    options.approximate_scan_limit = config.approximate_scan_limit;
    options.rerank_ratio = config.rerank_ratio;
    return options;
}

struct Fixture {
    std::filesystem::path prefix;
    std::vector<std::string> corpus;
};

std::string path_utf8(const std::filesystem::path & path) {
    const std::u8string encoded = path.u8string();
    std::string bytes(encoded.size(), '\0');
    std::memcpy(bytes.data(), encoded.data(), encoded.size());
    return bytes;
}

Fixture build_fixture(const std::filesystem::path & root,
                      const std::filesystem::path & name,
                      std::string_view topic,
                      leann::ApproximationKind approximation) {
    Fixture fixture{root / name, make_corpus(topic)};
    leann::HashEmbedder embedder(embedding_dimension);
    leann::Index::build(
        leann::index_file_from_prefix(fixture.prefix),
        leann::documents_file_from_prefix(fixture.prefix), fixture.corpus,
        embedder, build_config(approximation));
    return fixture;
}

enum class CallbackMode {
    Normal,
    ReturnCode,
    Nan,
    Infinity,
    Zero,
    ThrowStandard,
    ThrowUnknown,
    ThrowBadAlloc,
    Slow,
};

struct CallbackState {
    explicit CallbackState(std::size_t dimension = embedding_dimension)
        : hash(dimension), dimension(dimension) {}

    leann::HashEmbedder hash;
    std::size_t dimension;
    CallbackMode mode = CallbackMode::Normal;
    int return_code = 0;

    std::mutex mutex;
    std::size_t calls = 0;
    std::size_t active = 0;
    std::size_t maximum_active = 0;
    std::vector<std::size_t> strides;
    std::vector<std::size_t> batch_sizes;
    std::vector<std::string> observed_texts;
};

class ActiveCallbackGuard {
  public:
    explicit ActiveCallbackGuard(CallbackState & state) : state_(state) {
        const std::lock_guard lock(state_.mutex);
        ++state_.active;
        state_.maximum_active =
            std::max(state_.maximum_active, state_.active);
        ++state_.calls;
    }

    ~ActiveCallbackGuard() {
        const std::lock_guard lock(state_.mutex);
        --state_.active;
    }

  private:
    CallbackState & state_;
};

int hash_callback(void * user_data,
                  const leann_string_view * texts,
                  std::size_t text_count,
                  float * output_embeddings,
                  std::size_t embedding_stride) {
    auto & state = *static_cast<CallbackState *>(user_data);
    ActiveCallbackGuard active(state);
    if (texts == nullptr || output_embeddings == nullptr ||
        embedding_stride < state.dimension) {
        return -900;
    }

    std::vector<std::string> owned_texts;
    owned_texts.reserve(text_count);
    for (std::size_t i = 0; i < text_count; ++i) {
        if (texts[i].size != 0U && texts[i].data == nullptr) {
            return -901;
        }
        owned_texts.emplace_back(
            texts[i].size == 0U ? std::string{}
                                : std::string(texts[i].data, texts[i].size));
    }
    {
        const std::lock_guard lock(state.mutex);
        state.strides.push_back(embedding_stride);
        state.batch_sizes.push_back(text_count);
        state.observed_texts.insert(state.observed_texts.end(),
                                    owned_texts.begin(), owned_texts.end());
    }

    switch (state.mode) {
    case CallbackMode::ReturnCode:
        return state.return_code;
    case CallbackMode::ThrowStandard:
        throw std::runtime_error(
            "secret model contents and document text must not escape");
    case CallbackMode::ThrowUnknown:
        throw 17;
    case CallbackMode::ThrowBadAlloc:
        throw std::bad_alloc();
    case CallbackMode::Slow:
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        break;
    default:
        break;
    }

    const auto embeddings = state.hash.embed(owned_texts);
    for (std::size_t row = 0; row < embeddings.size(); ++row) {
        std::copy(embeddings[row].begin(), embeddings[row].end(),
                  output_embeddings + row * embedding_stride);
    }
    if (!owned_texts.empty()) {
        switch (state.mode) {
        case CallbackMode::Nan:
            output_embeddings[0] =
                std::numeric_limits<float>::quiet_NaN();
            break;
        case CallbackMode::Infinity:
            output_embeddings[0] =
                std::numeric_limits<float>::infinity();
            break;
        case CallbackMode::Zero:
            std::fill(output_embeddings,
                      output_embeddings + text_count * embedding_stride,
                      0.0F);
            break;
        default:
            break;
        }
    }
    return 0;
}

leann_searcher * open_searcher(const std::filesystem::path & prefix,
                               CallbackState & state) {
    const std::string fingerprint = state.hash.fingerprint();
    leann_embedder_v1 embedder{
        static_cast<std::uint32_t>(sizeof(leann_embedder_v1)),
        LEANN_C_API_VERSION,
        &state,
        state.dimension,
        {fingerprint.data(), fingerprint.size()},
        hash_callback,
    };
    leann_searcher * searcher = nullptr;
    const std::string path = path_utf8(prefix);
    const leann_status status =
        leann_searcher_open(path.c_str(), &embedder, &searcher);
    if (status != LEANN_STATUS_OK) {
        throw std::runtime_error(
            "C API open failed: " + std::string(leann_last_error()));
    }
    check(searcher != nullptr, "successful open returns a searcher");
    // descriptor and fingerprint intentionally expire on return; callback
    // code and state remain valid for the searcher's lifetime.
    return searcher;
}

std::vector<leann_result_view_v1>
read_result_views(const leann_results * results) {
    std::vector<leann_result_view_v1> views;
    const std::size_t count = leann_results_count(results);
    check(std::string(leann_last_error()).empty(),
          "successful result count clears last error");
    views.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        check(leann_results_get(results, i, &views[i]) == LEANN_STATUS_OK,
              "read C result view");
    }
    return views;
}

void compare_metrics(const leann::SearchMetrics & expected,
                     const leann_search_metrics_v1 & actual) {
    check(actual.approximate_distances == expected.approximate_distances,
          "approximate distance metric parity");
    check(actual.exact_recomputations == expected.exact_recomputations,
          "exact recomputation metric parity");
    check(actual.expanded_nodes == expected.expanded_nodes,
          "expanded-node metric parity");
    check(actual.embedding_batches == expected.embedding_batches,
          "embedding-batch metric parity");
    check(actual.upper_layer_hops == expected.upper_layer_hops,
          "upper-layer-hop metric parity");
    check(std::isfinite(actual.elapsed_ms) && actual.elapsed_ms >= 0.0,
          "elapsed time is finite and nonnegative");
}

void test_parity(const Fixture & fixture) {
    auto index =
        leann::Index::load(leann::index_file_from_prefix(fixture.prefix));
    auto documents = leann::DocumentStore::open(
        leann::documents_file_from_prefix(fixture.prefix));
    leann::HashEmbedder direct_embedder(embedding_dimension);
    const std::string query =
        std::string("compact local") + '\0' +
        " vector graph selective recomputation";
    const leann::SearchConfig config = non_default_search_config();
    const leann::SearchResponse direct =
        index.search(query, direct_embedder, documents, config);

    CallbackState state;
    leann_searcher * searcher = open_searcher(fixture.prefix, state);
    leann_search_options_v1 options = c_options(config);
    leann_results * results = nullptr;
    check(leann_searcher_search(
              searcher, {query.data(), query.size()}, &options, &results) ==
              LEANN_STATUS_OK,
          "non-default C search succeeds");
    check(results != nullptr, "successful C search returns results");

    const auto views = read_result_views(results);
    check(views.size() == direct.results.size(), "result count parity");
    for (std::size_t i = 0; i < views.size(); ++i) {
        check(views[i].id == direct.results[i].id, "result id/order parity");
        check(views[i].distance == direct.results[i].distance,
              "exact distance parity");
        const std::string text(views[i].text.data, views[i].text.size);
        check(text == documents.read(views[i].id), "result text parity");
    }

    leann_search_metrics_v1 metrics{};
    check(leann_results_metrics(results, &metrics) == LEANN_STATUS_OK,
          "read C search metrics");
    compare_metrics(direct.metrics, metrics);
    {
        const std::lock_guard lock(state.mutex);
        check(state.calls == 1U + direct.metrics.embedding_batches,
              "callback receives one query batch plus recomputation batches");
        check(!state.batch_sizes.empty() && state.batch_sizes.front() == 1U,
              "query callback batch has one text");
        std::size_t recomputed = 0;
        for (std::size_t batch = 1; batch < state.batch_sizes.size();
             ++batch) {
            check(state.batch_sizes[batch] <= config.recompute_batch_size,
                  "document callback respects recomputation batch size");
            recomputed += state.batch_sizes[batch];
        }
        check(recomputed == direct.metrics.exact_recomputations,
              "document callback batch counts match search metrics");
        check(!state.observed_texts.empty() &&
                  state.observed_texts.front() == query,
              "callback receives the length-delimited query");
        check(std::all_of(state.strides.begin(), state.strides.end(),
                          [](std::size_t stride) {
                              return stride == embedding_dimension;
                          }),
              "callback stride equals the embedding dimension");
    }

    leann_results_destroy(results);
    leann_searcher_destroy(searcher);

    CallbackState suffix_state;
    const std::filesystem::path index_path =
        leann::index_file_from_prefix(fixture.prefix);
    searcher = open_searcher(index_path, suffix_state);
    leann_searcher_destroy(searcher);
}

void test_default_options(const Fixture & fixture) {
    auto index =
        leann::Index::load(leann::index_file_from_prefix(fixture.prefix));
    auto documents = leann::DocumentStore::open(
        leann::documents_file_from_prefix(fixture.prefix));
    leann::HashEmbedder direct_embedder(embedding_dimension);
    const std::string query = "default local retrieval";
    const auto direct =
        index.search(query, direct_embedder, documents, leann::SearchConfig{});

    CallbackState state;
    leann_searcher * searcher = open_searcher(fixture.prefix, state);
    leann_results * results = nullptr;
    check(leann_searcher_search(
              searcher, {query.data(), query.size()}, nullptr, &results) ==
              LEANN_STATUS_OK,
          "null options select C++ defaults");
    const auto views = read_result_views(results);
    check(views.size() == direct.results.size(),
          "default result count parity");
    for (std::size_t i = 0; i < views.size(); ++i) {
        check(views[i].id == direct.results[i].id &&
                  views[i].distance == direct.results[i].distance,
              "default result parity");
    }
    leann_results_destroy(results);
    leann_searcher_destroy(searcher);
}

void test_result_ownership(const Fixture & fixture) {
    CallbackState state;
    leann_searcher * searcher = open_searcher(fixture.prefix, state);
    leann::SearchConfig config;
    config.top_k = static_cast<std::uint32_t>(fixture.corpus.size());
    config.ef_search = static_cast<std::uint32_t>(fixture.corpus.size());
    config.recompute_batch_size = 5;
    config.approximate_scan_limit = 1000;
    config.rerank_ratio = 1.0;
    leann_search_options_v1 options = c_options(config);
    const std::string query = "binary alpha omega payload";
    leann_results * results = nullptr;
    check(leann_searcher_search(
              searcher, {query.data(), query.size()}, &options, &results) ==
              LEANN_STATUS_OK,
          "ownership search succeeds");
    leann_searcher_destroy(searcher);

    const auto views = read_result_views(results);
    check(views.size() == fixture.corpus.size(),
          "ownership search returns the complete corpus");
    bool saw_nul_document = false;
    bool saw_utf8_document = false;
    for (const auto & view : views) {
        const std::string text(view.text.data, view.text.size);
        check(text == fixture.corpus.at(view.id),
              "result bytes survive searcher destruction");
        saw_nul_document =
            saw_nul_document ||
            text.find('\0') != std::string::npos;
        saw_utf8_document = saw_utf8_document || view.id == 1U;
    }
    check(saw_nul_document, "embedded NUL document is preserved");
    check(saw_utf8_document, "UTF-8 document is preserved");
    {
        const std::lock_guard lock(state.mutex);
        check(std::find(state.observed_texts.begin(),
                        state.observed_texts.end(), fixture.corpus[0]) !=
                  state.observed_texts.end(),
              "callback receives embedded NUL document bytes with length");
        check(std::find(state.observed_texts.begin(),
                        state.observed_texts.end(), fixture.corpus[1]) !=
                  state.observed_texts.end(),
              "callback receives UTF-8 document bytes");
    }
    leann_results_destroy(results);
}

void copy_pair(const Fixture & fixture,
               const std::filesystem::path & target_prefix) {
    std::filesystem::copy_file(
        leann::index_file_from_prefix(fixture.prefix),
        leann::index_file_from_prefix(target_prefix));
    std::filesystem::copy_file(
        leann::documents_file_from_prefix(fixture.prefix),
        leann::documents_file_from_prefix(target_prefix));
}

void flip_byte(const std::filesystem::path & path, std::uintmax_t offset) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    check(static_cast<bool>(file), "open artifact for mutation");
    file.seekg(static_cast<std::streamoff>(offset));
    const int value = file.get();
    check(value != std::char_traits<char>::eof(),
          "read artifact mutation byte");
    file.seekp(static_cast<std::streamoff>(offset));
    file.put(static_cast<char>(static_cast<unsigned char>(value) ^ 0x01U));
    file.close();
    check(static_cast<bool>(file), "finalize artifact mutation");
}

void test_open_errors(const std::filesystem::path & root,
                      const Fixture & fixture,
                      const Fixture & other) {
    CallbackState state;
    const std::string fingerprint = state.hash.fingerprint();
    leann_embedder_v1 embedder{
        static_cast<std::uint32_t>(sizeof(leann_embedder_v1)),
        LEANN_C_API_VERSION,
        &state,
        state.dimension,
        {fingerprint.data(), fingerprint.size()},
        hash_callback,
    };
    const std::string prefix = path_utf8(fixture.prefix);
    leann_searcher * output =
        reinterpret_cast<leann_searcher *>(static_cast<std::uintptr_t>(1));

    check_status(leann_searcher_open(prefix.c_str(), nullptr, &output),
                 LEANN_STATUS_INVALID_ARGUMENT, "embedder must not be null",
                 "null embedder");
    check(output == nullptr, "failed open clears its output handle");
    check_status(leann_searcher_open(prefix.c_str(), &embedder, nullptr),
                 LEANN_STATUS_INVALID_ARGUMENT,
                 "searcher output must not be null", "null open output");
    check_status(leann_searcher_open(nullptr, &embedder, &output),
                 LEANN_STATUS_INVALID_ARGUMENT, "index prefix",
                 "null prefix");

    leann_embedder_v1 invalid = embedder;
    invalid.struct_size =
        static_cast<std::uint32_t>(sizeof(leann_embedder_v1) - 1U);
    check_status(leann_searcher_open(prefix.c_str(), &invalid, &output),
                 LEANN_STATUS_INVALID_ARGUMENT, "struct_size",
                 "small embedder struct");
    invalid = embedder;
    invalid.api_version = LEANN_C_API_VERSION + 1U;
    check_status(leann_searcher_open(prefix.c_str(), &invalid, &output),
                 LEANN_STATUS_INVALID_ARGUMENT, "api_version",
                 "unknown embedder version");
    invalid = embedder;
    invalid.embed_batch = nullptr;
    check_status(leann_searcher_open(prefix.c_str(), &invalid, &output),
                 LEANN_STATUS_INVALID_ARGUMENT, "callback",
                 "null embedder callback");
    invalid = embedder;
    invalid.fingerprint = {nullptr, 1U};
    check_status(leann_searcher_open(prefix.c_str(), &invalid, &output),
                 LEANN_STATUS_INVALID_ARGUMENT, "fingerprint data",
                 "null fingerprint bytes");

    invalid = embedder;
    invalid.struct_size =
        static_cast<std::uint32_t>(sizeof(leann_embedder_v1) + 32U);
    output = nullptr;
    check(leann_searcher_open(prefix.c_str(), &invalid, &output) ==
              LEANN_STATUS_OK,
          "larger embedder struct is accepted");
    leann_searcher_destroy(output);

    CallbackState wrong_dimension(embedding_dimension * 2U);
    const std::string wrong_dimension_fingerprint =
        wrong_dimension.hash.fingerprint();
    invalid = {
        static_cast<std::uint32_t>(sizeof(leann_embedder_v1)),
        LEANN_C_API_VERSION,
        &wrong_dimension,
        wrong_dimension.dimension,
        {wrong_dimension_fingerprint.data(),
         wrong_dimension_fingerprint.size()},
        hash_callback,
    };
    check_status(leann_searcher_open(prefix.c_str(), &invalid, &output),
                 LEANN_STATUS_ARTIFACT_ERROR, "dimension",
                 "dimension mismatch");
    check(wrong_dimension.calls == 0U,
          "dimension mismatch is rejected before callback");

    const std::string wrong_fingerprint = "different-fingerprint";
    invalid = embedder;
    invalid.fingerprint =
        {wrong_fingerprint.data(), wrong_fingerprint.size()};
    check_status(leann_searcher_open(prefix.c_str(), &invalid, &output),
                 LEANN_STATUS_ARTIFACT_ERROR, "fingerprint",
                 "fingerprint mismatch");
    check(state.calls == 0U,
          "fingerprint mismatch is rejected before callback");

    const std::string missing = path_utf8(root / "missing");
    check_status(leann_searcher_open(missing.c_str(), &embedder, &output),
                 LEANN_STATUS_IO_ERROR, "cannot open index",
                 "missing artifacts");

    const auto truncated = root / "truncated";
    copy_pair(fixture, truncated);
    const auto truncated_index = leann::index_file_from_prefix(truncated);
    std::filesystem::resize_file(
        truncated_index, std::filesystem::file_size(truncated_index) - 1U);
    const std::string truncated_string = path_utf8(truncated);
    check_status(
        leann_searcher_open(truncated_string.c_str(), &embedder, &output),
        LEANN_STATUS_ARTIFACT_ERROR, "index", "truncated index");

    const auto truncated_documents = root / "truncated-documents";
    copy_pair(fixture, truncated_documents);
    const auto truncated_document_path =
        leann::documents_file_from_prefix(truncated_documents);
    std::filesystem::resize_file(
        truncated_document_path,
        std::filesystem::file_size(truncated_document_path) - 1U);
    const std::string truncated_documents_string =
        path_utf8(truncated_documents);
    check_status(
        leann_searcher_open(truncated_documents_string.c_str(), &embedder,
                            &output),
        LEANN_STATUS_ARTIFACT_ERROR, "payload size",
        "truncated document artifact");

    const auto checksum = root / "checksum";
    copy_pair(fixture, checksum);
    flip_byte(leann::index_file_from_prefix(checksum), 16U);
    const std::string checksum_string = path_utf8(checksum);
    check_status(
        leann_searcher_open(checksum_string.c_str(), &embedder, &output),
        LEANN_STATUS_ARTIFACT_ERROR, "checksum", "index checksum mismatch");

    const auto cross_pair = root / "cross-pair";
    std::filesystem::copy_file(
        leann::index_file_from_prefix(fixture.prefix),
        leann::index_file_from_prefix(cross_pair));
    std::filesystem::copy_file(
        leann::documents_file_from_prefix(other.prefix),
        leann::documents_file_from_prefix(cross_pair));
    const std::string cross_pair_string = path_utf8(cross_pair);
    check_status(
        leann_searcher_open(cross_pair_string.c_str(), &embedder, &output),
        LEANN_STATUS_ARTIFACT_ERROR, "valid pair", "cross-paired artifacts");
}

void test_search_argument_errors(const Fixture & fixture) {
    CallbackState state;
    leann_searcher * searcher = open_searcher(fixture.prefix, state);
    const std::string query = "safe local query";
    leann_results * output =
        reinterpret_cast<leann_results *>(static_cast<std::uintptr_t>(1));

    check_status(
        leann_searcher_search(nullptr, {query.data(), query.size()}, nullptr,
                              &output),
        LEANN_STATUS_INVALID_ARGUMENT, "searcher must not be null",
        "null searcher");
    check(output == nullptr, "failed search clears its output handle");
    check_status(
        leann_searcher_search(searcher, {query.data(), query.size()}, nullptr,
                              nullptr),
        LEANN_STATUS_INVALID_ARGUMENT, "results output", "null result output");
    check_status(leann_searcher_search(searcher, {query.data(), 0U}, nullptr,
                                       &output),
                 LEANN_STATUS_INVALID_ARGUMENT, "query must not be empty",
                 "empty query");
    check_status(
        leann_searcher_search(searcher, {nullptr, 1U}, nullptr, &output),
        LEANN_STATUS_INVALID_ARGUMENT, "query data", "null query bytes");

    leann_search_options_v1 options = c_options(non_default_search_config());
    options.struct_size =
        static_cast<std::uint32_t>(sizeof(options) - 1U);
    check_status(
        leann_searcher_search(searcher, {query.data(), query.size()}, &options,
                              &output),
        LEANN_STATUS_INVALID_ARGUMENT, "struct_size",
        "small search options struct");
    options = c_options(non_default_search_config());
    options.top_k = 0U;
    check_status(
        leann_searcher_search(searcher, {query.data(), query.size()}, &options,
                              &output),
        LEANN_STATUS_INVALID_ARGUMENT, "top_k", "invalid search values");

    check(leann_results_count(nullptr) == 0U,
          "null result count returns zero");
    check(std::string(leann_last_error()).find("results must not be null") !=
              std::string::npos,
          "null result count records an error");
    leann_result_view_v1 view{9U, 9.0F, {query.data(), query.size()}};
    check_status(leann_results_get(nullptr, 0U, &view),
                 LEANN_STATUS_INVALID_ARGUMENT, "results must not be null",
                 "null results view");
    check(view.text.data == nullptr && view.text.size == 0U,
          "failed result view is cleared");
    check_status(leann_results_metrics(nullptr, nullptr),
                 LEANN_STATUS_INVALID_ARGUMENT, "metrics output",
                 "null metrics output");

    leann_searcher_destroy(searcher);
    leann_searcher_destroy(nullptr);
    leann_results_destroy(nullptr);
    check(std::string(leann_last_error()).empty(),
          "null-safe destroy clears prior error");
}

void test_callback_errors(const Fixture & fixture) {
    const std::string query = "callback failure query";
    const std::array<std::pair<CallbackMode, leann_status>, 7> cases{{
        {CallbackMode::ReturnCode, LEANN_STATUS_EMBEDDER_ERROR},
        {CallbackMode::Nan, LEANN_STATUS_EMBEDDER_ERROR},
        {CallbackMode::Infinity, LEANN_STATUS_EMBEDDER_ERROR},
        {CallbackMode::Zero, LEANN_STATUS_EMBEDDER_ERROR},
        {CallbackMode::ThrowStandard, LEANN_STATUS_EMBEDDER_ERROR},
        {CallbackMode::ThrowUnknown, LEANN_STATUS_EMBEDDER_ERROR},
        {CallbackMode::ThrowBadAlloc, LEANN_STATUS_OUT_OF_MEMORY},
    }};

    for (const auto & [mode, expected] : cases) {
        CallbackState state;
        state.mode = mode;
        state.return_code = 73;
        leann_searcher * searcher = open_searcher(fixture.prefix, state);
        leann_results * output =
            reinterpret_cast<leann_results *>(static_cast<std::uintptr_t>(1));
        const leann_status status = leann_searcher_search(
            searcher, {query.data(), query.size()}, nullptr, &output);
        check(status == expected,
              "callback failure maps to the expected status");
        check(output == nullptr, "callback failure clears result output");
        const std::string detail = leann_last_error();
        if (mode == CallbackMode::ReturnCode) {
            check(detail.find("73") != std::string::npos,
                  "callback error includes caller code");
        }
        check(detail.find("secret model contents") == std::string::npos,
              "callback exception contents do not cross the ABI");
        check(detail.find("document text") == std::string::npos,
              "callback exception does not leak document text");
        leann_searcher_destroy(searcher);
    }
}

void test_lazy_document_corruption(const Fixture & fixture,
                                   const std::filesystem::path & root) {
    const auto corrupt = root / "lazy-corrupt";
    copy_pair(fixture, corrupt);
    CallbackState state;
    leann_searcher * searcher = open_searcher(corrupt, state);
    const auto document_path = leann::documents_file_from_prefix(corrupt);
    flip_byte(document_path, std::filesystem::file_size(document_path) - 1U);

    leann::SearchConfig config;
    config.top_k = static_cast<std::uint32_t>(fixture.corpus.size());
    config.ef_search = static_cast<std::uint32_t>(fixture.corpus.size());
    config.recompute_batch_size = 6;
    config.approximate_scan_limit = 1000;
    config.rerank_ratio = 1.0;
    auto options = c_options(config);
    const std::string query = "read every document";
    leann_results * results = nullptr;
    check_status(
        leann_searcher_search(searcher, {query.data(), query.size()}, &options,
                              &results),
        LEANN_STATUS_ARTIFACT_ERROR, "CRC32C", "lazy document corruption");
    check(results == nullptr,
          "document corruption does not return partial results");
    leann_searcher_destroy(searcher);
}

void test_concurrent_searches(const Fixture & fixture) {
    CallbackState state;
    state.mode = CallbackMode::Slow;
    leann_searcher * searcher = open_searcher(fixture.prefix, state);
    const auto options = c_options(non_default_search_config());
    constexpr std::size_t worker_count = 6;
    std::barrier gate(static_cast<std::ptrdiff_t>(worker_count));
    std::array<leann_status, worker_count> statuses{};
    std::array<std::size_t, worker_count> counts{};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            const std::string query =
                "concurrent local retrieval " + std::to_string(worker);
            gate.arrive_and_wait();
            leann_results * results = nullptr;
            statuses[worker] = leann_searcher_search(
                searcher, {query.data(), query.size()}, &options, &results);
            if (results != nullptr) {
                counts[worker] = leann_results_count(results);
            }
            leann_results_destroy(results);
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        check(statuses[worker] == LEANN_STATUS_OK,
              "concurrent search status");
        check(counts[worker] == options.top_k,
              "concurrent result handle is independent");
    }
    {
        const std::lock_guard lock(state.mutex);
        check(state.maximum_active == 1U,
              "calls into one embedding callback are serialized");
    }
    leann_searcher_destroy(searcher);
}

void test_thread_local_errors() {
    std::barrier gate(2);
    std::array<std::string, 2> details;
    std::thread first([&] {
        leann_result_view_v1 view{};
        (void)leann_results_get(nullptr, 0U, &view);
        gate.arrive_and_wait();
        details[0] = leann_last_error();
    });
    std::thread second([&] {
        const std::string query = "thread local";
        leann_results * results = nullptr;
        (void)leann_searcher_search(
            nullptr, {query.data(), query.size()}, nullptr, &results);
        gate.arrive_and_wait();
        details[1] = leann_last_error();
    });
    first.join();
    second.join();
    check(details[0].find("results must not be null") != std::string::npos,
          "first thread retains its error");
    check(details[1].find("searcher must not be null") != std::string::npos,
          "second thread retains its error");
}

} // namespace

int main() {
    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto root = std::filesystem::temp_directory_path() /
                      ("leann-cpp-c-api-" + unique);
    std::filesystem::create_directories(root);

    try {
        check(leann_api_version() == LEANN_C_API_VERSION,
              "runtime API version");
        check(std::string(leann_last_error()).empty(),
              "version call clears the thread error");
        check_status(leann_search_options_init(nullptr),
                     LEANN_STATUS_INVALID_ARGUMENT,
                     "options must not be null", "null options init");

        const Fixture pq = build_fixture(
            root, "pq", "pq", leann::ApproximationKind::ProductQuantization);
        const std::filesystem::path unicode_name(
            std::u8string(u8"simhash-cà-phê"));
        const Fixture simhash = build_fixture(
            root, unicode_name, "simhash",
            leann::ApproximationKind::SimHash);
        const Fixture other = build_fixture(
            root, "other", "other corpus",
            leann::ApproximationKind::SimHash);

        test_parity(pq);
        test_parity(simhash);
        test_default_options(pq);
        test_result_ownership(pq);
        test_open_errors(root, pq, other);
        test_search_argument_errors(pq);
        test_callback_errors(pq);
        test_lazy_document_corruption(pq, root);
        test_concurrent_searches(pq);
        test_thread_local_errors();

        std::filesystem::remove_all(root);
        std::cout << "all leann.cpp C API tests passed\n";
        return 0;
    } catch (...) {
        std::filesystem::remove_all(root);
        throw;
    }
}
