#include "leann/leann.h"

#include "leann/document_store.hpp"
#include "leann/embedder.hpp"
#include "leann/index.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t error_capacity = 1024;
thread_local std::array<char, error_capacity> last_error_buffer{};

void clear_error() noexcept {
    last_error_buffer.front() = '\0';
}

void set_error(std::string_view message) noexcept {
    const std::size_t count =
        std::min(message.size(), last_error_buffer.size() - 1U);
    if (count != 0U) {
        std::memcpy(last_error_buffer.data(), message.data(), count);
    }
    last_error_buffer[count] = '\0';
}

class StatusError final : public std::exception {
  public:
    StatusError(leann_status status, std::string message)
        : status_(status), message_(std::move(message)) {}

    [[nodiscard]] leann_status status() const noexcept { return status_; }
    [[nodiscard]] const char * what() const noexcept override {
        return message_.c_str();
    }

  private:
    leann_status status_;
    std::string message_;
};

class EmbedderError final : public std::exception {
  public:
    explicit EmbedderError(int callback_code) noexcept {
        (void)std::snprintf(message_.data(), message_.size(),
                            "embedding callback returned error code %d",
                            callback_code);
    }

    explicit EmbedderError(const char * message) noexcept {
        const std::size_t length =
            std::min(std::strlen(message), message_.size() - 1U);
        std::memcpy(message_.data(), message, length);
        message_[length] = '\0';
    }

    [[nodiscard]] const char * what() const noexcept override {
        return message_.data();
    }

  private:
    std::array<char, 160> message_{};
};

[[nodiscard]] leann_status
translate_exception(leann_status fallback) noexcept {
    try {
        throw;
    } catch (const StatusError & error) {
        set_error(error.what());
        return error.status();
    } catch (const EmbedderError & error) {
        set_error(error.what());
        return LEANN_STATUS_EMBEDDER_ERROR;
    } catch (const std::bad_alloc &) {
        set_error("out of memory");
        return LEANN_STATUS_OUT_OF_MEMORY;
    } catch (const std::filesystem::filesystem_error &) {
        set_error("filesystem operation failed");
        return LEANN_STATUS_IO_ERROR;
    } catch (const std::invalid_argument & error) {
        set_error(error.what());
        return LEANN_STATUS_INVALID_ARGUMENT;
    } catch (const std::out_of_range & error) {
        set_error(error.what());
        return LEANN_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception & error) {
        if (fallback == LEANN_STATUS_ARTIFACT_ERROR ||
            fallback == LEANN_STATUS_IO_ERROR) {
            set_error(error.what());
        } else {
            set_error("internal C++ exception");
        }
        return fallback;
    } catch (...) {
        set_error("unknown internal C++ exception");
        return LEANN_STATUS_INTERNAL_ERROR;
    }
}

[[noreturn]] void invalid_argument(const char * message) {
    throw StatusError(LEANN_STATUS_INVALID_ARGUMENT, message);
}

[[nodiscard]] std::string copy_string_view(leann_string_view view,
                                           const char * field) {
    if (view.size != 0U && view.data == nullptr) {
        invalid_argument(field);
    }
    return view.size == 0U ? std::string{}
                           : std::string(view.data, view.size);
}

class CallbackEmbedder final : public leann::Embedder {
  public:
    CallbackEmbedder(void * user_data,
                     std::size_t dimension,
                     std::string fingerprint,
                     leann_embed_batch_fn callback)
        : user_data_(user_data), dimension_(dimension),
          fingerprint_(std::move(fingerprint)), callback_(callback) {}

    [[nodiscard]] std::size_t dimension() const noexcept override {
        return dimension_;
    }

    [[nodiscard]] std::string fingerprint() const override {
        return fingerprint_;
    }

    [[nodiscard]] std::vector<leann::Embedding>
    embed(std::span<const std::string> texts) override {
        if (texts.empty()) {
            return {};
        }
        if (dimension_ >
            std::numeric_limits<std::size_t>::max() / texts.size()) {
            throw std::bad_alloc();
        }

        std::vector<leann_string_view> views;
        views.reserve(texts.size());
        for (const std::string & text : texts) {
            views.push_back({text.data(), text.size()});
        }

        std::vector<float> flattened(texts.size() * dimension_, 0.0F);
        int callback_code = 0;
        {
            const std::lock_guard lock(callback_mutex_);
            try {
                callback_code =
                    callback_(user_data_, views.data(), views.size(),
                              flattened.data(), dimension_);
            } catch (const std::bad_alloc &) {
                throw;
            } catch (const std::exception &) {
                throw EmbedderError(
                    "embedding callback threw a standard exception");
            } catch (...) {
                throw EmbedderError(
                    "embedding callback threw an unknown exception");
            }
        }
        if (callback_code != 0) {
            throw EmbedderError(callback_code);
        }

        for (std::size_t row = 0; row < texts.size(); ++row) {
            double squared_norm = 0.0;
            for (std::size_t column = 0; column < dimension_; ++column) {
                const float value =
                    flattened[row * dimension_ + column];
                if (!std::isfinite(value)) {
                    throw EmbedderError(
                        "embedding callback returned NaN or infinity");
                }
                squared_norm += static_cast<double>(value) * value;
            }
            if (!std::isfinite(squared_norm) || squared_norm <= 0.0) {
                throw EmbedderError(
                    "embedding callback returned an invalid vector norm");
            }
        }

        std::vector<leann::Embedding> result;
        result.reserve(texts.size());
        for (std::size_t row = 0; row < texts.size(); ++row) {
            const auto begin =
                flattened.begin() +
                static_cast<std::ptrdiff_t>(row * dimension_);
            result.emplace_back(
                begin, begin + static_cast<std::ptrdiff_t>(dimension_));
        }
        return result;
    }

  private:
    void * user_data_;
    std::size_t dimension_;
    std::string fingerprint_;
    leann_embed_batch_fn callback_;
    std::mutex callback_mutex_;
};

leann::Index load_index(const std::filesystem::path & path) {
    try {
        return leann::Index::load(path);
    } catch (const std::bad_alloc &) {
        throw;
    } catch (const std::filesystem::filesystem_error &) {
        throw StatusError(LEANN_STATUS_IO_ERROR,
                          "cannot access index artifact");
    } catch (const std::runtime_error & error) {
        if (std::string_view(error.what()).starts_with("cannot open index:")) {
            throw StatusError(LEANN_STATUS_IO_ERROR, error.what());
        }
        throw StatusError(LEANN_STATUS_ARTIFACT_ERROR, error.what());
    }
}

leann::DocumentStore
load_documents(const std::filesystem::path & path) {
    try {
        return leann::DocumentStore::open(path);
    } catch (const std::bad_alloc &) {
        throw;
    } catch (const std::filesystem::filesystem_error &) {
        throw StatusError(LEANN_STATUS_IO_ERROR,
                          "cannot access document artifact");
    } catch (const std::runtime_error & error) {
        if (std::string_view(error.what())
                .starts_with("cannot open document store:")) {
            throw StatusError(LEANN_STATUS_IO_ERROR, error.what());
        }
        throw StatusError(LEANN_STATUS_ARTIFACT_ERROR, error.what());
    }
}

struct OwnedResult {
    std::uint32_t id;
    float distance;
    std::string text;
};

} // namespace

struct leann_searcher {
    leann_searcher(leann::Index loaded_index,
                   leann::DocumentStore loaded_documents,
                   void * user_data,
                   std::size_t dimension,
                   std::string fingerprint,
                   leann_embed_batch_fn callback)
        : embedder(user_data, dimension, std::move(fingerprint), callback),
          index(std::move(loaded_index)),
          documents(std::move(loaded_documents)) {}

    CallbackEmbedder embedder;
    leann::Index index;
    leann::DocumentStore documents;
};

struct leann_results {
    std::vector<OwnedResult> values;
    leann::SearchMetrics metrics;
};

extern "C" LEANN_C_API std::uint32_t
leann_api_version(void) noexcept {
    clear_error();
    return LEANN_C_API_VERSION;
}

extern "C" LEANN_C_API const char *
leann_last_error(void) noexcept {
    return last_error_buffer.data();
}

extern "C" LEANN_C_API leann_status
leann_search_options_init(leann_search_options_v1 * options) noexcept {
    clear_error();
    try {
        if (options == nullptr) {
            invalid_argument("options must not be null");
        }
        const leann::SearchConfig defaults;
        *options = {
            static_cast<std::uint32_t>(sizeof(*options)),
            defaults.top_k,
            defaults.ef_search,
            defaults.recompute_batch_size,
            defaults.approximate_scan_limit,
            defaults.rerank_ratio,
        };
        return LEANN_STATUS_OK;
    } catch (...) {
        return translate_exception(LEANN_STATUS_INTERNAL_ERROR);
    }
}

extern "C" LEANN_C_API leann_status
leann_searcher_open(const char * index_prefix_utf8,
                    const leann_embedder_v1 * embedder,
                    leann_searcher ** output) noexcept {
    clear_error();
    if (output != nullptr) {
        *output = nullptr;
    }
    try {
        if (output == nullptr) {
            invalid_argument("searcher output must not be null");
        }
        if (index_prefix_utf8 == nullptr || index_prefix_utf8[0] == '\0') {
            invalid_argument("index prefix must not be null or empty");
        }
        if (embedder == nullptr) {
            invalid_argument("embedder must not be null");
        }
        if (embedder->struct_size < sizeof(leann_embedder_v1)) {
            invalid_argument("embedder struct_size is too small");
        }
        if (embedder->api_version != LEANN_C_API_VERSION) {
            invalid_argument("unsupported embedder api_version");
        }
        if (embedder->dimension == 0U) {
            invalid_argument("embedder dimension must be positive");
        }
        if (embedder->embed_batch == nullptr) {
            invalid_argument("embedder callback must not be null");
        }
        const std::string fingerprint = copy_string_view(
            embedder->fingerprint,
            "embedder fingerprint data must not be null when size is nonzero");

        const std::string_view prefix_bytes(index_prefix_utf8);
        std::u8string utf8_prefix(prefix_bytes.size(), u8'\0');
        std::memcpy(utf8_prefix.data(), prefix_bytes.data(),
                    prefix_bytes.size());
        const std::filesystem::path prefix(utf8_prefix);
        auto index =
            load_index(leann::index_file_from_prefix(prefix));
        auto documents =
            load_documents(leann::documents_file_from_prefix(prefix));
        try {
            index.validate_document_store(documents);
        } catch (const std::bad_alloc &) {
            throw;
        } catch (const std::exception &) {
            throw StatusError(
                LEANN_STATUS_ARTIFACT_ERROR,
                "index and document artifacts do not form a valid pair");
        }

        const leann::IndexStats stats = index.stats();
        if (embedder->dimension != stats.dimension) {
            throw StatusError(
                LEANN_STATUS_ARTIFACT_ERROR,
                "embedder dimension does not match the index");
        }
        if (fingerprint != index.embedder_fingerprint()) {
            throw StatusError(
                LEANN_STATUS_ARTIFACT_ERROR,
                "embedder fingerprint does not match the index");
        }

        auto searcher = std::make_unique<leann_searcher>(
            std::move(index), std::move(documents), embedder->user_data,
            embedder->dimension, fingerprint, embedder->embed_batch);
        *output = searcher.release();
        return LEANN_STATUS_OK;
    } catch (...) {
        return translate_exception(LEANN_STATUS_ARTIFACT_ERROR);
    }
}

extern "C" LEANN_C_API void
leann_searcher_destroy(leann_searcher * searcher) noexcept {
    clear_error();
    try {
        delete searcher;
    } catch (...) {
        (void)translate_exception(LEANN_STATUS_INTERNAL_ERROR);
    }
}

extern "C" LEANN_C_API leann_status
leann_searcher_search(leann_searcher * searcher,
                      leann_string_view query,
                      const leann_search_options_v1 * options,
                      leann_results ** output) noexcept {
    clear_error();
    if (output != nullptr) {
        *output = nullptr;
    }
    try {
        if (output == nullptr) {
            invalid_argument("results output must not be null");
        }
        if (searcher == nullptr) {
            invalid_argument("searcher must not be null");
        }
        if (query.size == 0U) {
            invalid_argument("query must not be empty");
        }
        if (query.data == nullptr) {
            invalid_argument("query data must not be null");
        }

        leann::SearchConfig config;
        if (options != nullptr) {
            if (options->struct_size < sizeof(leann_search_options_v1)) {
                invalid_argument("search options struct_size is too small");
            }
            config.top_k = options->top_k;
            config.ef_search = options->ef_search;
            config.recompute_batch_size = options->recompute_batch_size;
            config.approximate_scan_limit =
                options->approximate_scan_limit;
            config.rerank_ratio = options->rerank_ratio;
        }

        const leann::SearchResponse response = searcher->index.search(
            std::string_view(query.data, query.size), searcher->embedder,
            searcher->documents, config);

        auto results = std::make_unique<leann_results>();
        results->metrics = response.metrics;
        results->values.reserve(response.results.size());
        for (const leann::SearchResult & result : response.results) {
            results->values.push_back(
                {result.id, result.distance,
                 searcher->documents.read(result.id)});
        }
        *output = results.release();
        return LEANN_STATUS_OK;
    } catch (...) {
        return translate_exception(LEANN_STATUS_ARTIFACT_ERROR);
    }
}

extern "C" LEANN_C_API std::size_t
leann_results_count(const leann_results * results) noexcept {
    clear_error();
    try {
        if (results == nullptr) {
            invalid_argument("results must not be null");
        }
        return results->values.size();
    } catch (...) {
        (void)translate_exception(LEANN_STATUS_INTERNAL_ERROR);
        return 0U;
    }
}

extern "C" LEANN_C_API leann_status
leann_results_get(const leann_results * results,
                  std::size_t position,
                  leann_result_view_v1 * output) noexcept {
    clear_error();
    if (output != nullptr) {
        *output = {};
    }
    try {
        if (output == nullptr) {
            invalid_argument("result view output must not be null");
        }
        if (results == nullptr) {
            invalid_argument("results must not be null");
        }
        if (position >= results->values.size()) {
            invalid_argument("result position is out of range");
        }
        const OwnedResult & result = results->values[position];
        *output = {
            result.id,
            result.distance,
            {result.text.data(), result.text.size()},
        };
        return LEANN_STATUS_OK;
    } catch (...) {
        return translate_exception(LEANN_STATUS_INTERNAL_ERROR);
    }
}

extern "C" LEANN_C_API leann_status
leann_results_metrics(const leann_results * results,
                      leann_search_metrics_v1 * output) noexcept {
    clear_error();
    if (output != nullptr) {
        *output = {};
    }
    try {
        if (output == nullptr) {
            invalid_argument("metrics output must not be null");
        }
        if (results == nullptr) {
            invalid_argument("results must not be null");
        }
        *output = {
            results->metrics.approximate_distances,
            results->metrics.exact_recomputations,
            results->metrics.expanded_nodes,
            results->metrics.embedding_batches,
            results->metrics.upper_layer_hops,
            results->metrics.elapsed_ms,
        };
        return LEANN_STATUS_OK;
    } catch (...) {
        return translate_exception(LEANN_STATUS_INTERNAL_ERROR);
    }
}

extern "C" LEANN_C_API void
leann_results_destroy(leann_results * results) noexcept {
    clear_error();
    try {
        delete results;
    } catch (...) {
        (void)translate_exception(LEANN_STATUS_INTERNAL_ERROR);
    }
}
