#ifndef LEANN_LEANN_H
#define LEANN_LEANN_H

#include <stddef.h>
#include <stdint.h>

#define LEANN_C_API_VERSION 1u

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(LEANN_C_API_SHARED)
#if defined(LEANN_C_API_EXPORTS)
#define LEANN_C_API __declspec(dllexport)
#else
#define LEANN_C_API __declspec(dllimport)
#endif
#else
#define LEANN_C_API
#endif
#elif defined(LEANN_C_API_EXPORTS) && defined(__GNUC__)
#define LEANN_C_API __attribute__((visibility("default")))
#else
#define LEANN_C_API
#endif

#ifdef __cplusplus
#define LEANN_C_API_NOEXCEPT noexcept
extern "C" {
#else
#define LEANN_C_API_NOEXCEPT
#endif

typedef enum leann_status {
    LEANN_STATUS_OK = 0,
    LEANN_STATUS_INVALID_ARGUMENT = 1,
    LEANN_STATUS_IO_ERROR = 2,
    LEANN_STATUS_ARTIFACT_ERROR = 3,
    LEANN_STATUS_EMBEDDER_ERROR = 4,
    LEANN_STATUS_OUT_OF_MEMORY = 5,
    LEANN_STATUS_INTERNAL_ERROR = 6
} leann_status;

typedef struct leann_string_view {
    const char * data;
    size_t size;
} leann_string_view;

/*
 * The callback writes text_count vectors. Vector i begins at
 * output_embeddings + i * embedding_stride. Return zero on success or a
 * caller-defined nonzero error code.
 */
typedef int (*leann_embed_batch_fn)(
    void * user_data,
    const leann_string_view * texts,
    size_t text_count,
    float * output_embeddings,
    size_t embedding_stride);

/*
 * The caller owns every referenced object. The descriptor and fingerprint
 * bytes need only remain valid until leann_searcher_open() returns; the
 * searcher copies their values. The callback code and the object referenced by
 * user_data must remain valid until leann_searcher_destroy(). leann.cpp never
 * frees user_data. Calls into one callback are serialized.
 *
 * This v1 layout is frozen. A larger struct_size is accepted but trailing
 * bytes are ignored; future layouts use new versioned types and entry points.
 */
typedef struct leann_embedder_v1 {
    uint32_t struct_size;
    uint32_t api_version;
    void * user_data;
    size_t dimension;
    leann_string_view fingerprint;
    leann_embed_batch_fn embed_batch;
} leann_embedder_v1;

typedef struct leann_search_options_v1 {
    uint32_t struct_size;
    uint32_t top_k;
    uint32_t ef_search;
    uint32_t recompute_batch_size;
    uint32_t approximate_scan_limit;
    double rerank_ratio;
} leann_search_options_v1;

/*
 * This v1 layout is frozen. Future options layouts use a new versioned type
 * and initializer rather than extending this object in place.
 */

/*
 * text is length-delimited and may contain embedded NUL bytes. Its storage is
 * owned by the leann_results handle and remains valid until that handle is
 * destroyed.
 */
typedef struct leann_result_view_v1 {
    uint32_t id;
    float distance;
    leann_string_view text;
} leann_result_view_v1;

typedef struct leann_search_metrics_v1 {
    uint64_t approximate_distances;
    uint64_t exact_recomputations;
    uint64_t expanded_nodes;
    uint64_t embedding_batches;
    uint64_t upper_layer_hops;
    double elapsed_ms;
} leann_search_metrics_v1;

typedef struct leann_searcher leann_searcher;
typedef struct leann_results leann_results;

LEANN_C_API uint32_t leann_api_version(void) LEANN_C_API_NOEXCEPT;

/*
 * Returns thread-local detail for the most recent failed C API call on this
 * thread. The pointer remains valid until the next C API call on this thread.
 * Calling leann_last_error() itself does not clear the message.
 */
LEANN_C_API const char * leann_last_error(void) LEANN_C_API_NOEXCEPT;

/*
 * Writes exactly sizeof(leann_search_options_v1) bytes. The caller must pass a
 * complete v1 object; future initializers have different versioned names.
 */
LEANN_C_API leann_status
leann_search_options_init(
    leann_search_options_v1 * options) LEANN_C_API_NOEXCEPT;

/*
 * Opens <index_prefix_utf8>.leann and <index_prefix_utf8>.docs. Passing a
 * prefix that already ends in .leann has the same meaning as the C++ API and
 * CLI. On failure, *output is set to NULL.
 */
LEANN_C_API leann_status
leann_searcher_open(const char * index_prefix_utf8,
                    const leann_embedder_v1 * embedder,
                    leann_searcher ** output) LEANN_C_API_NOEXCEPT;

/*
 * Destroying NULL is safe. Destroying a searcher while another thread is
 * searching through it is invalid; finish all in-flight searches first.
 */
LEANN_C_API void
leann_searcher_destroy(leann_searcher * searcher) LEANN_C_API_NOEXCEPT;

/*
 * Searches one length-delimited query. A NULL options pointer selects the C++
 * defaults. On failure, *output is set to NULL. Recursive search on the same
 * searcher from its embedding callback is invalid.
 */
LEANN_C_API leann_status
leann_searcher_search(leann_searcher * searcher,
                      leann_string_view query,
                      const leann_search_options_v1 * options,
                      leann_results ** output) LEANN_C_API_NOEXCEPT;

LEANN_C_API size_t
leann_results_count(const leann_results * results) LEANN_C_API_NOEXCEPT;

LEANN_C_API leann_status
leann_results_get(const leann_results * results,
                  size_t position,
                  leann_result_view_v1 * output) LEANN_C_API_NOEXCEPT;

LEANN_C_API leann_status
leann_results_metrics(const leann_results * results,
                      leann_search_metrics_v1 * output) LEANN_C_API_NOEXCEPT;

LEANN_C_API void
leann_results_destroy(leann_results * results) LEANN_C_API_NOEXCEPT;

#ifdef __cplusplus
} /* extern "C" */
#endif

#undef LEANN_C_API_NOEXCEPT

#endif /* LEANN_LEANN_H */
