#include "leann/leann.h"

#include <stddef.h>
#include <stdint.h>

static int
unused_embed(void * user_data,
             const leann_string_view * texts,
             size_t text_count,
             float * output_embeddings,
             size_t embedding_stride) {
    (void)user_data;
    (void)texts;
    (void)text_count;
    (void)output_embeddings;
    (void)embedding_stride;
    return 0;
}

int
main(void) {
    struct guarded_options {
        uint64_t before;
        leann_search_options_v1 options;
        uint64_t after;
    } guarded = {
        UINT64_C(0x0123456789abcdef),
        {0U, 0U, 0U, 0U, 0U, 0.0},
        UINT64_C(0xfedcba9876543210),
    };
    leann_embedder_v1 embedder = {
        (uint32_t)sizeof(leann_embedder_v1),
        LEANN_C_API_VERSION,
        NULL,
        8U,
        {NULL, 0U},
        unused_embed,
    };
    leann_searcher * searcher = NULL;
    leann_results * results = NULL;
    leann_result_view_v1 view = {0U, 0.0F, {NULL, 0U}};
    leann_search_metrics_v1 metrics = {0U, 0U, 0U, 0U, 0U, 0.0};

    (void)embedder;
    (void)searcher;
    (void)results;
    (void)view;
    (void)metrics;
    if (leann_api_version() != LEANN_C_API_VERSION) {
        return 1;
    }
    if (leann_search_options_init(&guarded.options) != LEANN_STATUS_OK) {
        return 2;
    }
    if (guarded.options.struct_size !=
        (uint32_t)sizeof(guarded.options)) {
        return 3;
    }
    if (guarded.before != UINT64_C(0x0123456789abcdef) ||
        guarded.after != UINT64_C(0xfedcba9876543210)) {
        return 4;
    }
    leann_searcher_destroy(NULL);
    leann_results_destroy(NULL);
    return 0;
}
