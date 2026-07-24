# leann.cpp C API

The version 1 C API opens an existing leann.cpp artifact pair and searches it
through an embedding callback owned by the application. The public header,
`leann/leann.h`, is valid C11, has no C++ or llama.cpp types, and exposes only
opaque handles.

This API does not build or mutate indexes, load or own an embedding model, run
a server, or change the `.leann`/`.docs` formats. Build an index with the C++
API or CLI first. A pinned llama.cpp callback example is a follow-up; the ABI
is intentionally independent of any llama.cpp revision.

## Lifecycle

The example below shows the complete open, search, read, and destroy
lifecycle. `app_embed_many()` represents the application's already-loaded
embedding engine. It must write one finite, nonzero vector for every input
string.

```c
#include <leann/leann.h>

#include <stdio.h>
#include <string.h>

struct app_embedding_session {
    /* Application-owned model/session state. */
    void * model;
};

static int
app_embed_many(struct app_embedding_session * session,
               const leann_string_view * texts,
               size_t text_count,
               float * output,
               size_t stride);

static int
embed_batch(void * user_data,
            const leann_string_view * texts,
            size_t text_count,
            float * output_embeddings,
            size_t embedding_stride) {
    struct app_embedding_session * session =
        (struct app_embedding_session *)user_data;
    return app_embed_many(session, texts, text_count, output_embeddings,
                          embedding_stride);
}

int
main(void) {
    struct app_embedding_session session = {0};
    static const char fingerprint[] = "my-embedder-v1:768";
    leann_embedder_v1 embedder = {
        (uint32_t)sizeof(leann_embedder_v1),
        LEANN_C_API_VERSION,
        &session,
        768U,
        {fingerprint, sizeof(fingerprint) - 1U},
        embed_batch,
    };
    leann_searcher * searcher = NULL;
    leann_results * results = NULL;
    leann_search_options_v1 options;
    const char query_bytes[] = "compact local vector search";
    const leann_string_view query = {
        query_bytes,
        sizeof(query_bytes) - 1U,
    };

    leann_status status =
        leann_searcher_open("out/demo", &embedder, &searcher);
    if (status != LEANN_STATUS_OK) {
        fprintf(stderr, "open failed: %s\n", leann_last_error());
        return 1;
    }

    status = leann_search_options_init(&options);
    if (status == LEANN_STATUS_OK) {
        options.top_k = 3U;
        options.ef_search = 64U;
        options.recompute_batch_size = 16U;
        options.approximate_scan_limit = 100000U;
        options.rerank_ratio = 0.25;
        status =
            leann_searcher_search(searcher, query, &options, &results);
    }
    if (status != LEANN_STATUS_OK) {
        fprintf(stderr, "search failed: %s\n", leann_last_error());
        leann_searcher_destroy(searcher);
        return 1;
    }

    for (size_t position = 0; position < leann_results_count(results);
         ++position) {
        leann_result_view_v1 result;
        status = leann_results_get(results, position, &result);
        if (status != LEANN_STATUS_OK) {
            fprintf(stderr, "result failed: %s\n", leann_last_error());
            leann_results_destroy(results);
            leann_searcher_destroy(searcher);
            return 1;
        }
        printf("%u\t%.8f\t", result.id, result.distance);
        fwrite(result.text.data, 1U, result.text.size, stdout);
        fputc('\n', stdout);
    }

    {
        leann_search_metrics_v1 metrics;
        status = leann_results_metrics(results, &metrics);
        if (status == LEANN_STATUS_OK) {
            fprintf(stderr, "exact recomputations: %llu\n",
                    (unsigned long long)metrics.exact_recomputations);
        }
    }

    leann_results_destroy(results);
    leann_searcher_destroy(searcher);
    return 0;
}
```

Passing `NULL` instead of search options selects the current C++ defaults.
Calling `leann_search_options_init()` is preferred when overriding fields.

## Callback contract

- `leann_searcher_open()` validates the index/document pair, dimension, and
  fingerprint before the first callback.
- The application owns all callback state. The embedder descriptor and
  fingerprint bytes need only remain valid until `leann_searcher_open()`
  returns because leann.cpp copies their values. The callback code and the
  object referenced by `user_data` must remain valid until
  `leann_searcher_destroy()`. leann.cpp never frees `user_data`.
- For text `i`, write its vector at
  `output_embeddings + i * embedding_stride`. The stride is at least
  `dimension`.
- Return zero on success. A nonzero caller-defined value becomes
  `LEANN_STATUS_EMBEDDER_ERROR`, and the numeric code appears in
  `leann_last_error()`.
- Every requested vector must be finite and have a nonzero, finite norm.
  leann.cpp retains its existing normalization and dimension checks.
- Calls into a searcher's callback are serialized. This makes one
  non-reentrant model session safe across concurrent searches. Traversal and
  immutable document access can still overlap.
- A callback must not recursively search through the same searcher.

## Result ownership

`leann_results` owns copies of the returned document bytes. A
`leann_result_view_v1` points into that owned storage and remains valid until
`leann_results_destroy()`, even after the searcher is destroyed. Text is
length-delimited, can contain embedded NUL bytes, and is not guaranteed to be
NUL-terminated.

Each successful search returns an independent result handle. Destroy functions
accept `NULL`. Destroying a searcher concurrently with an in-flight search is
invalid; join all search workers first.

## Errors and ABI compatibility

Every exported function prevents C++ exceptions from crossing the ABI
boundary.
Functions that return a handle set the output to `NULL` on failure.
`leann_last_error()` returns thread-local detail that remains valid until the
next C API call on that thread. It does not include model contents or document
text.

| Status | Meaning |
|---|---|
| `LEANN_STATUS_OK` | The operation completed. |
| `LEANN_STATUS_INVALID_ARGUMENT` | A pointer, string view, struct, version, or search option is invalid. |
| `LEANN_STATUS_IO_ERROR` | A required artifact cannot be accessed. |
| `LEANN_STATUS_ARTIFACT_ERROR` | An artifact is malformed, corrupt, mismatched, or fails integrity validation. |
| `LEANN_STATUS_EMBEDDER_ERROR` | The callback failed or returned an invalid vector. |
| `LEANN_STATUS_OUT_OF_MEMORY` | Allocation failed. |
| `LEANN_STATUS_INTERNAL_ERROR` | An unexpected implementation failure was contained. |

The versioned input/configuration structs `leann_embedder_v1` and
`leann_search_options_v1` carry `struct_size`; the embedder descriptor also
carries `api_version`. Initialize those fields to `sizeof(the_struct)` and
`LEANN_C_API_VERSION`, respectively. Version 1 rejects too-small input structs
and unknown API versions. `leann_result_view_v1` and
`leann_search_metrics_v1` are output-only frozen layouts and therefore do not
carry `struct_size`. Every `_v1` layout is frozen:
`leann_search_options_init()` writes exactly `sizeof(leann_search_options_v1)`
bytes, and trailing bytes in a larger caller object are ignored. Future
layouts use new versioned types and entry points rather than extending a v1
object in place.

The prefix passed to `leann_searcher_open()` is UTF-8 and follows CLI
semantics: `out/demo` resolves to `out/demo.leann` and `out/demo.docs`; passing
`out/demo.leann` resolves the same pair.
