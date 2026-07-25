#define main leann_cli_entry_for_test
#include "../app/main.cpp"
#undef main

#include <chrono>
#include <functional>
#include <iterator>
#include <set>
#include <thread>

namespace {

void check(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error("check failed: " + std::string(message));
    }
}

void expect_failure(const std::function<void()> & action,
                    std::string_view expected_message) {
    try {
        action();
    } catch (const std::exception & error) {
        check(std::string_view(error.what()).find(expected_message) !=
                  std::string_view::npos,
              "failure message");
        return;
    }
    throw std::runtime_error("expected operation to fail");
}

void write_text(const std::filesystem::path & path, std::string_view text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create test file");
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.close();
    if (!output) {
        throw std::runtime_error("cannot finalize test file");
    }
}

std::vector<char> read_binary(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open test file");
    }
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

Arguments make_arguments(std::vector<std::string> storage) {
    std::vector<char *> argv;
    argv.reserve(storage.size());
    for (auto & argument : storage) {
        argv.push_back(argument.data());
    }
    return Arguments(static_cast<int>(argv.size()), argv.data());
}

std::pair<std::string, std::string>
capture_streams(const std::function<void()> & action) {
    std::ostringstream captured_output;
    std::ostringstream captured_error;
    auto * old_output = std::cout.rdbuf(captured_output.rdbuf());
    auto * old_error = std::cerr.rdbuf(captured_error.rdbuf());
    try {
        action();
    } catch (...) {
        std::cout.rdbuf(old_output);
        std::cerr.rdbuf(old_error);
        throw;
    }
    std::cout.rdbuf(old_output);
    std::cerr.rdbuf(old_error);
    return {captured_output.str(), captured_error.str()};
}

std::pair<std::string, std::string>
run_benchmark(const Arguments & arguments) {
    return capture_streams([&] { command_bench(arguments); });
}

void check_no_atomic_temps(
    const std::filesystem::path & directory,
    const std::filesystem::path & target,
    std::string_view message) {
    const std::string temporary_prefix =
        "." + target.filename().string() + ".tmp.";
    for (const auto & entry :
         std::filesystem::directory_iterator(directory)) {
        check(!entry.path().filename().string().starts_with(
                  temporary_prefix),
              message);
    }
}

void write_v2_cache(
    const std::filesystem::path & path,
    const DocumentsSource & source,
    std::string_view fingerprint,
    std::span<const leann::Embedding> embeddings) {
    check(!embeddings.empty() && !embeddings.front().empty(),
          "test cache shape");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create test cache");
    }
    constexpr std::array<char, 8> magic{
        'L', 'E', 'A', 'N', 'N', 'B', 'C', '2'};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_little<std::uint32_t>(
        output, static_cast<std::uint32_t>(embeddings.front().size()));
    write_little<std::uint64_t>(output, embeddings.size());
    write_little<std::uint32_t>(
        output, static_cast<std::uint32_t>(fingerprint.size()));
    write_little<std::uint64_t>(output, source.file_size);
    output.write(
        reinterpret_cast<const char *>(source.sha256.data()),
        static_cast<std::streamsize>(source.sha256.size()));
    output.write(fingerprint.data(),
                 static_cast<std::streamsize>(fingerprint.size()));
    for (const auto & embedding : embeddings) {
        check(embedding.size() == embeddings.front().size(),
              "test cache dimensions");
        for (const float value : embedding) {
            write_little<std::uint32_t>(
                output, std::bit_cast<std::uint32_t>(value));
        }
    }
    output.close();
    if (!output) {
        throw std::runtime_error("cannot finalize test cache");
    }
}

void test_v2_stream_and_binding(const std::filesystem::path & directory) {
    const auto documents_path = directory / "documents.txt";
    const auto cache_path = directory / "vectors.leannbc";
    write_text(documents_path, "alpha\nbeta\n");
    const auto source = read_documents_source(documents_path);
    const std::string fingerprint =
        "llama.cpp-v1:test-model:4:4:1:gpu-layers=0";
    const std::vector<leann::Embedding> vectors{
        {1.0F, 0.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
    };
    write_v2_cache(cache_path, source, fingerprint, vectors);

    EmbeddingCacheEmbedder cache(cache_path, source);
    check(cache.dimension() == 4, "v2 cache dimension");
    check(cache.fingerprint() == fingerprint, "v2 cache fingerprint");
    const std::array<std::string, 1> first{"alpha"};
    const auto first_vector = cache.embed(first);
    check(first_vector == std::vector<leann::Embedding>{vectors.front()},
          "first streamed vector");
    expect_failure([&] { cache.require_complete(); },
                   "did not consume the complete");
    const std::array<std::string, 1> second{"beta"};
    const auto second_vector = cache.embed(second);
    check(second_vector == std::vector<leann::Embedding>{vectors.back()},
          "second streamed vector");
    cache.require_complete();
    expect_failure([&] { (void)cache.embed(first); },
                   "more vectors than");

    write_text(documents_path, "Alpha\nbeta\n");
    const auto changed_source = read_documents_source(documents_path);
    expect_failure(
        [&] {
            EmbeddingCacheEmbedder changed(cache_path, changed_source);
            (void)changed;
        },
        "exact --docs file");

    write_text(documents_path, "alpha\nbeta\n");
    const auto restored_source = read_documents_source(documents_path);
    {
        std::ofstream output(cache_path, std::ios::binary | std::ios::app);
        output.put('\0');
    }
    expect_failure(
        [&] {
            EmbeddingCacheEmbedder trailing(cache_path, restored_source);
            (void)trailing;
        },
        "truncated or trailing");

    const std::vector<leann::Embedding> non_normalized{
        {2.0F, 0.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
    };
    write_v2_cache(cache_path, restored_source, fingerprint, non_normalized);
    EmbeddingCacheEmbedder invalid(cache_path, restored_source);
    expect_failure([&] { (void)invalid.embed(first); },
                   "non-normalized");

    const std::vector<leann::Embedding> nan_vectors{
        {std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
    };
    write_v2_cache(cache_path, restored_source, fingerprint, nan_vectors);
    EmbeddingCacheEmbedder nan_cache(cache_path, restored_source);
    expect_failure([&] { (void)nan_cache.embed(first); },
                   "NaN or infinity");

    const std::vector<leann::Embedding> infinite_vectors{
        {std::numeric_limits<float>::infinity(), 0.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
    };
    write_v2_cache(
        cache_path, restored_source, fingerprint, infinite_vectors);
    EmbeddingCacheEmbedder infinite_cache(cache_path, restored_source);
    expect_failure([&] { (void)infinite_cache.embed(first); },
                   "NaN or infinity");

    write_v2_cache(cache_path, restored_source, fingerprint, vectors);
    std::filesystem::resize_file(
        cache_path, std::filesystem::file_size(cache_path) - 1U);
    expect_failure(
        [&] {
            EmbeddingCacheEmbedder truncated(cache_path, restored_source);
            (void)truncated;
        },
        "truncated or trailing");

    write_v2_cache(cache_path, restored_source, "", vectors);
    expect_failure(
        [&] {
            EmbeddingCacheEmbedder empty_fingerprint(
                cache_path, restored_source);
            (void)empty_fingerprint;
        },
        "empty embedder fingerprint");

    expect_failure(
        [] {
            (void)checked_multiply(
                std::numeric_limits<std::uint64_t>::max(), 2,
                "test multiplication");
        },
        "too large");
    expect_failure(
        [] {
            (void)checked_add(
                std::numeric_limits<std::uint64_t>::max(), 1,
                "test addition");
        },
        "too large");
}

void test_output_alias_rejection(const std::filesystem::path & directory) {
    const auto input_path = directory / "alias-input.bin";
    write_text(input_path, "protected");
    const std::array<NamedPath, 1> exact_output{{
        {"output", input_path},
    }};
    const std::array<NamedPath, 1> protected_input{{
        {"input", input_path},
    }};
    expect_failure(
        [&] {
            require_distinct_output_paths(exact_output, protected_input);
        },
        "aliases protected");

    std::error_code error;
    const auto symlink_path = directory / "alias-symlink.bin";
    std::filesystem::create_symlink(input_path, symlink_path, error);
    if (!error) {
        const std::array<NamedPath, 1> symlink_output{{
            {"symlink output", symlink_path},
        }};
        expect_failure(
            [&] {
                require_distinct_output_paths(
                    symlink_output, protected_input);
            },
            "aliases protected");
    }

    error.clear();
    const auto hardlink_path = directory / "alias-hardlink.bin";
    std::filesystem::create_hard_link(input_path, hardlink_path, error);
    if (!error) {
        const std::array<NamedPath, 1> hardlink_output{{
            {"hardlink output", hardlink_path},
        }};
        expect_failure(
            [&] {
                require_distinct_output_paths(
                    hardlink_output, protected_input);
            },
            "aliases protected");
    }

    const auto documents_path = directory / "collision-documents.txt";
    const auto prefix = directory / "collision";
    const auto cache_path = leann::index_file_from_prefix(prefix);
    write_text(documents_path, "zero\none\n");
    const auto source = read_documents_source(documents_path);
    const std::vector<leann::Embedding> vectors{
        {1.0F, 0.0F},
        {0.0F, 1.0F},
    };
    write_v2_cache(cache_path, source, "cache-fingerprint", vectors);
    const auto original_cache = read_binary(cache_path);
    const auto arguments = make_arguments({
        "leann",
        "build",
        "--index",
        prefix.string(),
        "--docs",
        documents_path.string(),
        "--embedder",
        "cache",
        "--embedding-cache",
        cache_path.string(),
    });
    expect_failure([&] { command_build(arguments); },
                   "aliases protected --embedding-cache");
    check(read_binary(cache_path) == original_cache,
          "build alias rejection preserves embedding cache");
}

void test_v1_benchmark_cache(const std::filesystem::path & directory) {
    const auto path = directory / "legacy.bench.f32";
    const std::string fingerprint = "legacy-test";
    const std::vector<leann::Embedding> vectors{
        {1.0F, 0.0F},
        {0.0F, 1.0F},
    };
    write_embedding_cache(path, fingerprint, vectors);
    const auto loaded =
        load_embedding_cache(path, fingerprint, vectors.size(), 2);
    check(loaded == vectors, "legacy v1 benchmark cache round trip");
}

void test_atomic_v1_cache_publication(
    const std::filesystem::path & directory) {
    const auto preserved_path = directory / "preserved.bench.f32";
    const std::vector<leann::Embedding> original_vectors{
        {1.0F, 0.0F},
        {0.0F, 1.0F},
    };
    write_embedding_cache(
        preserved_path, "preserved", original_vectors);
    const auto original_bytes = read_binary(preserved_path);

    const std::vector<leann::Embedding> inconsistent_vectors{
        {1.0F, 0.0F},
        {1.0F},
    };
    expect_failure(
        [&] {
            write_embedding_cache(
                preserved_path, "replacement", inconsistent_vectors);
        },
        "dimensions are inconsistent");
    check(read_binary(preserved_path) == original_bytes,
          "late cache write failure preserves existing target");
    check_no_atomic_temps(
        directory, preserved_path,
        "late cache write failure cleans its temporary file");

    const auto missing_source = directory / "missing-publish-source";
    expect_failure(
        [&] {
            atomic_replace(
                missing_source, preserved_path, "embedding cache");
        },
        "cannot publish embedding cache");
    check(read_binary(preserved_path) == original_bytes,
          "atomic rename failure preserves existing target");

    std::filesystem::path abandoned_temporary;
    {
        AtomicFileOutput abandoned(
            preserved_path, "embedding cache");
        abandoned.stream() << "incomplete";
        abandoned_temporary = abandoned.temporary_path();
        check(std::filesystem::exists(abandoned_temporary),
              "abandoned cache temporary exists during write");
    }
    check(!std::filesystem::exists(abandoned_temporary),
          "abandoned cache temporary is removed");
    check(read_binary(preserved_path) == original_bytes,
          "abandoned cache write preserves existing target");

    const auto concurrent_path = directory / "concurrent.bench.f32";
    constexpr std::size_t row_count = 4096;
    constexpr std::size_t dimension = 16;
    std::vector<leann::Embedding> vectors_a(
        row_count, leann::Embedding(dimension, 0.0F));
    std::vector<leann::Embedding> vectors_b(
        row_count, leann::Embedding(dimension, 0.0F));
    for (std::size_t row = 0; row < row_count; ++row) {
        vectors_a[row][row % dimension] = 1.0F;
        vectors_b[row][(row + 1U) % dimension] = 1.0F;
    }

    std::atomic<bool> start{false};
    std::exception_ptr failure_a;
    std::exception_ptr failure_b;
    auto writer = [&](std::string_view fingerprint,
                      const std::vector<leann::Embedding> & vectors,
                      std::exception_ptr & failure) {
        try {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (std::size_t iteration = 0; iteration < 12;
                 ++iteration) {
                write_embedding_cache(
                    concurrent_path, fingerprint, vectors);
            }
        } catch (...) {
            failure = std::current_exception();
        }
    };
    std::thread thread_a(
        writer, "concurrent-a", std::cref(vectors_a),
        std::ref(failure_a));
    std::thread thread_b(
        writer, "concurrent-b", std::cref(vectors_b),
        std::ref(failure_b));
    start.store(true, std::memory_order_release);
    thread_a.join();
    thread_b.join();
    if (failure_a) {
        std::rethrow_exception(failure_a);
    }
    if (failure_b) {
        std::rethrow_exception(failure_b);
    }

    bool matches_a = false;
    bool matches_b = false;
    try {
        matches_a =
            load_embedding_cache(
                concurrent_path, "concurrent-a", row_count, dimension) ==
            vectors_a;
    } catch (const std::exception &) {
    }
    try {
        matches_b =
            load_embedding_cache(
                concurrent_path, "concurrent-b", row_count, dimension) ==
            vectors_b;
    } catch (const std::exception &) {
    }
    check(matches_a || matches_b,
          "concurrent cache writers publish one complete artifact");
    check_no_atomic_temps(
        directory, concurrent_path,
        "concurrent cache writers leave no temporary files");
}

void test_ground_truth_parser(const std::filesystem::path & directory) {
    const auto path = directory / "truth.txt";
    write_text(path, "LEANN_GT1 2 2 4\n0 2\n3 1\n");
    const auto truth = load_ground_truth(path, 2, 2, 4);
    check(truth.rows ==
              std::vector<std::vector<std::uint32_t>>{{0, 2}, {3, 1}},
          "valid precomputed ground truth");

    write_text(path, "LEANN_GT1 2 2 4\n0 0\n3 1\n");
    expect_failure([&] { (void)load_ground_truth(path, 2, 2, 4); },
                   "duplicate ID");

    write_text(path, "LEANN_GT1 2 2 4\n0 4\n3 1\n");
    expect_failure([&] { (void)load_ground_truth(path, 2, 2, 4); },
                   "out of range");

    write_text(path, "LEANN_GT1 2 3 4\n0 1 2\n3 1 0\n");
    const auto wider_truth = load_ground_truth(path, 2, 2, 4);
    check(wider_truth.rows ==
              std::vector<std::vector<std::uint32_t>>{{0, 1}, {3, 1}},
          "wider ground truth is truncated to requested top-k");

    write_text(path, "LEANN_GT1 2 1 4\n0\n3\n");
    expect_failure([&] { (void)load_ground_truth(path, 2, 2, 4); },
                   "does not cover --top-k");

    write_text(path, "LEANN_GT1 2 2 4\n0 2\n");
    expect_failure([&] { (void)load_ground_truth(path, 2, 2, 4); },
                   "row count");
}

void test_cached_fingerprint_is_persisted(
    const std::filesystem::path & directory) {
    const auto source_path = directory / "build-documents.txt";
    const auto cache_path = directory / "build-vectors.leannbc";
    write_text(source_path, "zero\none\ntwo\nthree\n");
    const auto source = read_documents_source(source_path);
    const std::string fingerprint =
        "llama.cpp-v1:real-model:4:4:1:gpu-layers=99";
    const std::vector<leann::Embedding> vectors{
        {1.0F, 0.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 1.0F, 0.0F},
        {0.0F, 0.0F, 0.0F, 1.0F},
    };
    write_v2_cache(cache_path, source, fingerprint, vectors);
    EmbeddingCacheEmbedder cache(cache_path, source);

    leann::BuildConfig config;
    config.graph_degree = 2;
    config.ef_construction = 4;
    config.low_degree = 1;
    config.hub_ratio = 0.25;
    config.approximation = leann::ApproximationKind::SimHash;
    config.sketch_bits = 64;
    config.embedding_batch_size = 2;
    const auto index_path = directory / "cached.leann";
    const auto documents_path = directory / "cached.docs";
    leann::Index::build(index_path, documents_path, source.documents, cache,
                        config);
    cache.require_complete();
    const auto index = leann::Index::load(index_path);
    check(index.embedder_fingerprint() == fingerprint,
          "cached llama fingerprint persisted in index");
}

void test_bench_skips_corpus_with_precomputed_truth(
    const std::filesystem::path & directory) {
    const std::vector<std::string> documents{
        "zero", "one", "two", "three"};
    leann::HashEmbedder embedder(64);
    leann::BuildConfig config;
    config.graph_degree = 2;
    config.ef_construction = 4;
    config.low_degree = 1;
    config.approximation = leann::ApproximationKind::SimHash;
    config.sketch_bits = 64;
    const auto prefix = directory / "bench";
    leann::Index::build(
        leann::index_file_from_prefix(prefix),
        leann::documents_file_from_prefix(prefix), documents, embedder,
        config);

    const auto queries_path = directory / "queries.txt";
    const auto truth_path = directory / "bench-truth.txt";
    const auto query_cache_path = directory / "query-vectors.leannbc2";
    const auto raw_path = directory / "raw-latencies.csv";
    write_text(queries_path, "zero\none\n");
    write_text(truth_path, "LEANN_GT1 2 2 4\n0 1\n1 0\n");
    const auto query_source = read_documents_source(queries_path);
    auto query_vectors = embedder.embed(query_source.documents);
    for (auto & vector : query_vectors) {
        leann::normalize(vector);
    }
    write_v2_cache(query_cache_path, query_source,
                   embedder.fingerprint(), query_vectors);
    std::vector<std::string> storage{
        "leann",
        "bench",
        "--index",
        prefix.string(),
        "--queries",
        queries_path.string(),
        "--embedder",
        "hash",
        "--hash-dim",
        "64",
        "--top-k",
        "2",
        "--report-k",
        "1",
        "--ef-search",
        "4",
        "--ground-truth",
        truth_path.string(),
        "--query-embedding-cache",
        query_cache_path.string(),
        "--warmup-queries",
        "1",
        "--raw-latencies",
        raw_path.string(),
        "--ground-truth-batch",
        "0",
    };
    const auto arguments = make_arguments(storage);
    const auto [captured_output, captured_error] =
        run_benchmark(arguments);
    check(captured_output.find("queries=2") != std::string::npos,
          "benchmark used precomputed ground truth");
    check(captured_output.find("warmup_queries=1") !=
              std::string::npos,
          "benchmark reports unmeasured warmup");
    check(captured_output.find("recall_at_2=") !=
              std::string::npos &&
              captured_output.find("recall_at_1=") !=
                  std::string::npos,
          "one search reports full and prefix recall");
    check(captured_error.find("loaded_ground_truth=") !=
              std::string::npos,
          "benchmark reports precomputed ground truth");
    check(captured_error.find(
              "loaded_query_embedding_cache=") != std::string::npos,
          "benchmark reports query embedding cache");
    const auto raw_lines = read_lines(raw_path);
    check(raw_lines.size() == 3,
          "raw output excludes warmup and includes two measured rows");
    check(raw_lines.front().starts_with(
              "query_index,recall,recall_at_1,latency_ms"),
          "raw latency CSV includes requested prefix recall");
    check(raw_lines.front().ends_with(",result_ids"),
          "raw latency CSV exposes ranked result IDs");
    for (std::size_t row = 1; row < raw_lines.size(); ++row) {
        const auto separator = raw_lines[row].find_last_of(',');
        check(separator != std::string::npos,
              "raw result IDs have a CSV separator");
        std::istringstream ids_stream(
            raw_lines[row].substr(separator + 1));
        std::vector<std::uint32_t> ids{
            std::istream_iterator<std::uint32_t>(ids_stream),
            std::istream_iterator<std::uint32_t>()};
        check(ids.size() == 2, "raw result IDs contain exactly top-k");
        check(std::set<std::uint32_t>(ids.begin(), ids.end()).size() ==
                  ids.size(),
              "raw result IDs are unique");
        check(std::ranges::all_of(
                  ids, [](std::uint32_t id) { return id < 4; }),
              "raw result IDs are in range");
    }
    check_no_atomic_temps(
        directory, raw_path,
        "atomic raw output leaves no temporary file");

    const auto prefix_raw_path = directory / "raw-prefix.csv";
    auto prefix_storage = storage;
    prefix_storage.insert(
        prefix_storage.end(),
        {"--max-queries", "1", "--raw-latencies",
         prefix_raw_path.string()});
    const auto prefix_arguments = make_arguments(prefix_storage);
    const auto [prefix_output, prefix_error] =
        run_benchmark(prefix_arguments);
    (void)prefix_error;
    check(prefix_output.find("queries=1") != std::string::npos,
          "--max-queries slices after full truth validation");
    check(read_lines(prefix_raw_path).size() == 2,
          "--max-queries slices raw rows consistently");

    auto alias_storage = storage;
    alias_storage.insert(
        alias_storage.end(),
        {"--raw-latencies", truth_path.string()});
    const auto alias_arguments = make_arguments(alias_storage);
    const auto original_truth = read_binary(truth_path);
    expect_failure([&] { command_bench(alias_arguments); },
                   "aliases protected --ground-truth");
    check(read_binary(truth_path) == original_truth,
          "raw output alias rejection preserves ground truth");

    write_v2_cache(query_cache_path, query_source,
                   "wrong-fingerprint", query_vectors);
    expect_failure([&] { command_bench(arguments); },
                   "fingerprint does not match");
}

void test_cache_embedder_is_build_only() {
    const auto arguments = make_arguments(
        {"leann", "search", "--embedder", "cache"});
    expect_failure([&] { (void)make_embedder(arguments); },
                   "supported only by the build command");
}

void test_strict_cli_numeric_parsing() {
    const auto valid = make_arguments({
        "leann",
        "bench",
        "--count",
        "4294967295",
        "--ratio",
        "0.125",
    });
    check(valid.unsigned_value("--count", 0) ==
              std::numeric_limits<std::uint32_t>::max(),
          "maximum uint32 CLI value is accepted");
    check(valid.double_value("--ratio", 0.0) == 0.125,
          "finite double CLI value is accepted");

    for (const std::string & invalid :
         {"12junk", "-1", "+1", " 1", "1 "}) {
        const auto arguments = make_arguments(
            {"leann", "bench", "--count", invalid});
        expect_failure(
            [&] { (void)arguments.unsigned_value("--count", 0); },
            "must be an unsigned integer");
    }
    const auto overflow = make_arguments({
        "leann", "bench", "--count", "4294967296"});
    expect_failure(
        [&] { (void)overflow.unsigned_value("--count", 0); },
        "is too large");

    for (const std::string & invalid :
         {"0.5junk", "nan", "NaN", "inf", "-inf", "1e9999",
          " 0.5", "0.5 "}) {
        const auto arguments = make_arguments(
            {"leann", "bench", "--ratio", invalid});
        expect_failure(
            [&] { (void)arguments.double_value("--ratio", 0.0); },
            "must be a finite number");
    }
}

// Builds a small artifact pair with the hash embedder so the CLI commands can
// be driven against a real index. `hash_dimension` must match the --hash-dim
// the command under test passes, because it is part of the fingerprint.
void build_test_pair(const std::filesystem::path & prefix,
                     const std::vector<std::string> & documents,
                     std::uint32_t hash_dimension) {
    leann::HashEmbedder embedder(hash_dimension);
    leann::BuildConfig config;
    config.graph_degree = 2;
    config.ef_construction = 4;
    config.low_degree = 1;
    config.approximation = leann::ApproximationKind::SimHash;
    config.sketch_bits = 64;
    leann::Index::build(leann::index_file_from_prefix(prefix),
                        leann::documents_file_from_prefix(prefix), documents,
                        embedder, config);
}

// A mistyped option used to be ignored, so the command answered with its
// default and the wrong answer looked like a right one.
void test_strict_option_validation() {
    const CommandSpec * const search = find_command("search");
    const CommandSpec * const stats = find_command("stats");
    const CommandSpec * const doctor = find_command("doctor");
    check(search != nullptr && stats != nullptr && doctor != nullptr,
          "the command table exposes search, stats, and doctor");

    const auto typo = make_arguments({"leann", "search", "--index", "p",
                                      "--query", "q", "--topk", "2"});
    expect_failure([&] { validate_arguments(*search, typo); },
                   "unknown option for leann search: --topk");
    expect_failure([&] { validate_arguments(*search, typo); },
                   "did you mean --top-k?");

    const auto misplaced =
        make_arguments({"leann", "stats", "--index", "p", "--top-k", "2"});
    expect_failure([&] { validate_arguments(*stats, misplaced); },
                   "--top-k belongs to leann search");

    // Everything a command documents must also validate.
    const auto accepted = make_arguments(
        {"leann", "search", "--index", "p", "--query", "q", "--top-k", "2",
         "--ef-search", "8", "--rerank-ratio", "0.5", "--format", "json"});
    validate_arguments(*search, accepted);

    // A known option missing its value keeps the older, more specific
    // message; an unknown one is still reported as unknown.
    const auto truncated = make_arguments({"leann", "stats", "--index"});
    expect_failure([&] { validate_arguments(*stats, truncated); },
                   "missing value for --index");
    const auto truncated_unknown =
        make_arguments({"leann", "stats", "--nonsense"});
    expect_failure([&] { validate_arguments(*stats, truncated_unknown); },
                   "unknown option for leann stats: --nonsense");

    const auto positional =
        make_arguments({"leann", "stats", "--index", "p", "junk"});
    expect_failure([&] { validate_arguments(*stats, positional); },
                   "unexpected argument for leann stats: junk");

    // A boolean flag must not swallow the token after it.
    const auto flags =
        make_arguments({"leann", "doctor", "--repair", "--index", "p"});
    validate_arguments(*doctor, flags);
    check(flags.get("--index") == "p",
          "a boolean flag leaves the following option intact");
    check(flags.has("--repair"), "a boolean flag is recorded");

    // ...and must not be consumed AS a value either. Binding "--repair" to
    // --index would run doctor on a nonsense prefix with the requested
    // repair silently switched off, and exit 0.
    const auto swallowed =
        make_arguments({"leann", "doctor", "--index", "--repair"});
    check(swallowed.has("--repair"),
          "a boolean flag is not consumed as another option's value");
    check(swallowed.get("--index").empty(),
          "the option that lost its value keeps no bogus value");
    expect_failure([&] { validate_arguments(*doctor, swallowed); },
                   "missing value for --index");
    const auto wrong_flag = make_arguments(
        {"leann", "search", "--index", "p", "--query", "q", "--repair"});
    expect_failure([&] { validate_arguments(*search, wrong_flag); },
                   "unknown option for leann search: --repair");

    // The tables are the documented surface, so they must stay well formed.
    for (const CommandSpec & command : command_specs) {
        std::set<std::string_view> seen;
        for (const OptionGroup & group : command.groups) {
            for (const OptionSpec & option : group.options) {
                check(option.flag.starts_with("--"),
                      "every option flag starts with --");
                check(!option.help.empty(), "every option carries help text");
                check(seen.insert(option.flag).second,
                      "no option is listed twice for one command");
            }
        }
        check(seen.contains("--help"),
              "every command offers --help");
    }
}

// Document text is arbitrary corpus bytes, so the writer has to escape what
// JSON requires and refuse what it cannot represent.
void test_json_writer_escaping() {
    std::ostringstream output;
    {
        JsonWriter json(output);
        json.begin_object();
        json.field("quoted", "say \"hi\"");
        json.field("slashed", "back\\slash");
        json.field("spaced", std::string("tab\there\nline"));
        json.field("control", std::string("bell\x07 here"));
        json.field("unicode", "na\xC3\xAFve \xE6\x97\xA5\xE6\x9C\xAC");
        json.field("count", std::uint64_t{42});
        json.field("ratio", 0.125, 3);
        json.end_object();
    }
    const std::string text = output.str();
    check(text.find("\\\"hi\\\"") != std::string::npos,
          "quotes are escaped");
    check(text.find("back\\\\slash") != std::string::npos,
          "backslashes are escaped");
    check(text.find("tab\\there") != std::string::npos, "tabs are escaped");
    check(text.find("\\n") != std::string::npos, "newlines are escaped");
    check(text.find("\\u0007") != std::string::npos,
          "other control bytes become \\u escapes");
    check(text.find("na\xC3\xAFve") != std::string::npos,
          "valid multi-byte UTF-8 passes through unchanged");
    check(text.find("\"ratio\": 0.125") != std::string::npos,
          "doubles keep the requested precision");

    check(is_valid_utf8("plain ascii"), "ascii is valid UTF-8");
    check(is_valid_utf8(std::string_view("\xF0\x9F\x98\x80", 4)),
          "a four-byte sequence is valid UTF-8");
    check(!is_valid_utf8(std::string_view("\xFF\xFE", 2)),
          "an invalid lead byte is rejected");
    check(!is_valid_utf8(std::string_view("\xC0\xAF", 2)),
          "an overlong encoding is rejected");
    check(!is_valid_utf8(std::string_view("\xED\xA0\x80", 3)),
          "a surrogate half is rejected");
    check(!is_valid_utf8(std::string_view("\xF5\x80\x80\x80", 4)),
          "a code point above U+10FFFF is rejected");
    check(!is_valid_utf8(std::string_view("\xE6\x97", 2)),
          "a truncated sequence is rejected");

    std::ostringstream rejected;
    JsonWriter failing(rejected);
    failing.begin_object();
    expect_failure(
        [&] { failing.field("document", std::string_view("\xFF\xFE", 2)); },
        "not valid UTF-8");
}

void test_search_json_output(const std::filesystem::path & directory) {
    const auto prefix = directory / "json-search";
    const std::vector<std::string> documents{
        "plain document",
        "quotes \" and \\ backslash",
        "tab\tseparated",
        "third entry",
    };
    build_test_pair(prefix, documents, 64);

    const auto arguments = make_arguments(
        {"leann", "search", "--index", prefix.string(), "--query",
         "quotes", "--hash-dim", "64", "--top-k", "2", "--format", "json"});
    const auto [output, error] =
        capture_streams([&] { command_search(arguments); });

    check(error.empty(),
          "JSON search writes nothing to stderr, so stdout is the whole "
          "answer");
    check(output.starts_with("{") && output.ends_with("}\n"),
          "JSON search emits one object");
    check(output.find("\"query\": \"quotes\"") != std::string::npos,
          "the query is echoed");
    check(output.find("\"results\"") != std::string::npos,
          "results are present");
    check(output.find("\"metrics\"") != std::string::npos,
          "metrics move into the document instead of stderr");
    check(output.find("quotes \\\" and \\\\ backslash") != std::string::npos,
          "document text is escaped rather than emitted raw");
    check(output.find("tab\\tseparated") != std::string::npos ||
              output.find("\"id\"") != std::string::npos,
          "documents are readable through the JSON path");

    // Exactly top-k results, each with the three documented fields.
    std::size_t id_fields = 0;
    for (std::size_t at = output.find("\"id\""); at != std::string::npos;
         at = output.find("\"id\"", at + 1)) {
        ++id_fields;
    }
    check(id_fields == 2, "JSON search returns exactly --top-k results");

    // The text format must stay byte-identical, because the benchmark
    // harness parses it.
    const auto text_arguments = make_arguments(
        {"leann", "search", "--index", prefix.string(), "--query", "quotes",
         "--hash-dim", "64", "--top-k", "2"});
    const auto [text_output, text_error] =
        capture_streams([&] { command_search(text_arguments); });
    check(!text_output.empty() && text_output.find('\t') != std::string::npos,
          "the default format is still tab separated");
    check(text_error.find("search_ms=") != std::string::npos,
          "the default format still reports metrics on stderr");

    expect_failure(
        [&] {
            const auto bad = make_arguments(
                {"leann", "search", "--index", prefix.string(), "--query",
                 "q", "--format", "yaml"});
            command_search(bad);
        },
        "--format must be text or json");
}

// A document the JSON writer cannot represent must produce no output at all,
// not a truncated object with the error interleaved into it.
void test_json_failure_is_atomic(const std::filesystem::path & directory) {
    const auto prefix = directory / "json-invalid";
    const std::vector<std::string> documents{
        "valid document",
        std::string("\xFF\xFE invalid bytes", 16),
        "another valid document",
        "a fourth document",
    };
    build_test_pair(prefix, documents, 64);

    const auto arguments = make_arguments(
        {"leann", "search", "--index", prefix.string(), "--query", "invalid",
         "--hash-dim", "64", "--top-k", "4", "--format", "json"});
    std::string emitted = "not empty";
    bool failed = false;
    try {
        const auto [output, error] =
            capture_streams([&] { command_search(arguments); });
        emitted = output;
        (void)error;
    } catch (const std::exception & failure) {
        failed = true;
        check(std::string_view(failure.what()).find("not valid UTF-8") !=
                  std::string_view::npos,
              "the failure names the reason");
    }
    check(failed, "a document that is not valid UTF-8 fails the JSON path");

    // capture_streams rethrows after restoring the buffers, so anything the
    // writer had already emitted would have reached the captured stdout.
    const auto [partial, partial_error] = capture_streams([&] {
        try {
            command_search(arguments);
        } catch (const std::exception &) {
            // deliberately swallowed; the point is what reached stdout
        }
    });
    check(partial.empty(),
          "a failed JSON document writes nothing to stdout");
    (void)partial_error;

    // The same corpus is still fully serviceable through the text format.
    const auto text_arguments = make_arguments(
        {"leann", "search", "--index", prefix.string(), "--query", "invalid",
         "--hash-dim", "64", "--top-k", "4"});
    const auto [text_output, text_error] =
        capture_streams([&] { command_search(text_arguments); });
    check(!text_output.empty(),
          "the text format still serves a corpus JSON cannot represent");
    (void)text_error;
}

void test_doctor_inspection_and_repair(
    const std::filesystem::path & directory) {
    const auto working = directory / "doctor";
    std::filesystem::create_directories(working);
    const auto prefix = working / "pair";
    build_test_pair(prefix, {"alpha", "beta", "gamma", "delta"}, 64);
    const auto index_path = leann::index_file_from_prefix(prefix);
    const auto documents_path = leann::documents_file_from_prefix(prefix);

    auto temporary_path = index_path;
    temporary_path += ".tmp.abcdef";
    auto backup_path = index_path;
    backup_path += ".bak.abcdef";
    write_text(temporary_path, "abandoned");
    write_text(backup_path, "superseded");

    const auto report_arguments =
        make_arguments({"leann", "doctor", "--index", prefix.string()});
    const auto [report, ignored_error] =
        capture_streams([&] { command_doctor(report_arguments); });
    check(report.find("pair: valid") != std::string::npos,
          "doctor validates the live pair");
    check(report.find(".tmp.abcdef") != std::string::npos &&
              report.find(".bak.abcdef") != std::string::npos,
          "doctor finds leftovers named by prefix, not by extension");
    check(std::filesystem::exists(temporary_path),
          "reporting alone removes nothing");

    // A lock means a build may still own the temporary file.
    const auto lock_path = leann::detail::lock_path_for(index_path);
    std::filesystem::create_directory(lock_path);
    const auto locked_arguments = make_arguments(
        {"leann", "doctor", "--index", prefix.string(), "--repair"});
    const auto [locked_report, locked_error] =
        capture_streams([&] { command_doctor(locked_arguments); });
    check(locked_report.find("retained:") != std::string::npos,
          "a temporary is retained while a lock is present");
    check(std::filesystem::exists(temporary_path),
          "repair does not remove a temporary under an active lock");
    check(std::filesystem::exists(lock_path),
          "repair never removes a build lock");
    std::filesystem::remove(lock_path);

    const auto repair_arguments = make_arguments(
        {"leann", "doctor", "--index", prefix.string(), "--repair"});
    const auto [repaired, repair_error] =
        capture_streams([&] { command_doctor(repair_arguments); });
    check(!std::filesystem::exists(temporary_path),
          "repair removes an abandoned temporary once no lock is present");
    check(!std::filesystem::exists(backup_path),
          "repair removes a backup once the live pair validates");
    check(std::filesystem::exists(index_path) &&
              std::filesystem::exists(documents_path),
          "repair never touches the live pair");
    (void)repaired;

    // A backup is load-bearing while a build holds a lock: publication parks
    // the previous pair in .bak.* for the whole transaction and rolls back
    // from exactly those files. A valid-looking live pair is not licence to
    // remove it.
    write_text(backup_path, "mid-transaction backup");
    std::filesystem::create_directory(lock_path);
    const auto locked_backup_arguments = make_arguments(
        {"leann", "doctor", "--index", prefix.string(), "--repair"});
    const auto [locked_backup, locked_backup_error] =
        capture_streams([&] { command_doctor(locked_backup_arguments); });
    check(std::filesystem::exists(backup_path),
          "repair does not remove a backup while a build lock is present");
    std::filesystem::remove(lock_path);
    std::filesystem::remove(backup_path);

    // When the live pair is unusable the backup may be the only index left,
    // so it must survive --repair.
    auto surviving_backup = index_path;
    surviving_backup += ".bak.feed01";
    std::filesystem::rename(index_path, surviving_backup);
    const auto broken_arguments = make_arguments(
        {"leann", "doctor", "--index", prefix.string(), "--repair"});
    const auto [broken, broken_error] =
        capture_streams([&] { command_doctor(broken_arguments); });
    check(broken.find("pair: unusable") != std::string::npos,
          "doctor reports an unusable pair");
    check(std::filesystem::exists(surviving_backup),
          "a backup survives repair while the live pair does not validate");
    std::filesystem::rename(surviving_backup, index_path);
}

void test_doctor_reports_lock_ownership(
    const std::filesystem::path & directory) {
    const auto working = directory / "doctor-lock";
    std::filesystem::create_directories(working);
    const auto prefix = working / "pair";
    build_test_pair(prefix, {"alpha", "beta", "gamma", "delta"}, 64);
    const auto index_path = leann::index_file_from_prefix(prefix);
    const auto lock_path = leann::detail::lock_path_for(index_path);

    // A lock created by this process is honestly reported as running, and
    // --force-unlock must refuse it.
    std::filesystem::create_directory(lock_path);
    leann::detail::write_lock_owner(lock_path);
    const auto owner = leann::detail::read_lock_owner(lock_path);
    check(owner.has_value(), "a lock descriptor round trips");
    check(owner->pid == leann::detail::current_process_id(),
          "the descriptor records the owning process");
    check(leann::detail::owner_liveness(owner) ==
              leann::detail::OwnerLiveness::Running,
          "this process is observed as running");

    // The refusal must also come before any deletion, so a combined request
    // cannot delete leftovers and then throw away the report of having done
    // so.
    auto doomed_temporary = index_path;
    doomed_temporary += ".tmp.777777";
    write_text(doomed_temporary, "scratch");
    expect_failure(
        [&] {
            const auto arguments = make_arguments(
                {"leann", "doctor", "--index", prefix.string(), "--repair",
                 "--force-unlock"});
            capture_streams([&] { command_doctor(arguments); });
        },
        "refusing --force-unlock");
    check(std::filesystem::exists(lock_path),
          "a lock whose owner is running is never removed");
    check(std::filesystem::exists(doomed_temporary),
          "a refused request deletes nothing at all");
    std::filesystem::remove(doomed_temporary);

    // An unowned lock cannot be called stale, only unknown.
    leann::detail::remove_lock_owner(lock_path);
    check(!leann::detail::read_lock_owner(lock_path).has_value(),
          "a lock without a descriptor reports no owner");
    check(leann::detail::owner_liveness(std::nullopt) ==
              leann::detail::OwnerLiveness::Unknown,
          "liveness without a descriptor is unknown, never stale");
    const auto arguments = make_arguments(
        {"leann", "doctor", "--index", prefix.string(), "--force-unlock"});
    const auto [output, error] =
        capture_streams([&] { command_doctor(arguments); });
    check(output.find("unrecorded owner") != std::string::npos,
          "an unowned lock is described honestly");
    check(!std::filesystem::exists(lock_path),
          "--force-unlock removes a lock that names no running owner");
}

// Cancellation is driven through a token over a local flag rather than a real
// signal, so the test is deterministic and never touches process state.
void test_build_cancellation(const std::filesystem::path & directory) {
    const auto working = directory / "cancel";
    std::filesystem::create_directories(working);
    const auto source_path = working / "documents.txt";
    write_text(source_path, "alpha\nbeta\ngamma\ndelta\n");
    const auto prefix = working / "cancelled";

    std::atomic<bool> flag{true};
    const CancellationToken token(&flag);
    const auto arguments = make_arguments(
        {"leann", "build", "--docs", source_path.string(), "--index",
         prefix.string(), "--hash-dim", "64", "--progress", "never"});
    bool cancelled = false;
    try {
        capture_streams([&] { command_build(arguments, token); });
    } catch (const leann::BuildCancelled & stop) {
        cancelled = true;
        check(std::string_view(stop.what()).find("cancelled") !=
                  std::string_view::npos,
              "cancellation names itself");
    }
    check(cancelled, "an already-requested cancellation stops the build");

    // Fail-closed: nothing published, nothing left behind.
    check(!std::filesystem::exists(leann::index_file_from_prefix(prefix)),
          "a cancelled build publishes no index");
    check(!std::filesystem::exists(leann::documents_file_from_prefix(prefix)),
          "a cancelled build publishes no document store");
    for (const auto & entry :
         std::filesystem::directory_iterator(working)) {
        const std::string name = entry.path().filename().string();
        check(name.find(".lock") == std::string::npos,
              "a cancelled build leaves no build lock");
        check(name.find(".tmp.") == std::string::npos,
              "a cancelled build leaves no temporary artifact");
        check(name.find(".bak.") == std::string::npos,
              "a cancelled build leaves no backup artifact");
    }

    // A token that is never triggered leaves the build untouched.
    std::atomic<bool> quiet{false};
    const CancellationToken open(&quiet);
    capture_streams([&] { command_build(arguments, open); });
    check(std::filesystem::exists(leann::index_file_from_prefix(prefix)),
          "an untriggered token does not disturb the build");
}

// The progress callback must fire in order and must never be required.
void test_build_progress_reporting(const std::filesystem::path & directory) {
    const auto working = directory / "progress";
    std::filesystem::create_directories(working);
    const auto prefix = working / "observed";
    std::vector<std::string> documents;
    documents.reserve(64);
    for (int index = 0; index < 64; ++index) {
        documents.push_back("document " + std::to_string(index));
    }

    std::vector<std::string> stages;
    std::uint64_t last_completed = 0;
    bool monotonic = true;
    leann::HashEmbedder embedder(64);
    leann::BuildConfig config;
    config.graph_degree = 2;
    config.ef_construction = 4;
    config.low_degree = 1;
    config.approximation = leann::ApproximationKind::SimHash;
    config.sketch_bits = 64;
    config.embedding_batch_size = 8;
    config.report_progress = [&](const leann::BuildProgress & progress) {
        if (stages.empty() || stages.back() != progress.stage) {
            stages.emplace_back(progress.stage);
            last_completed = 0;
        }
        if (progress.completed < last_completed) {
            monotonic = false;
        }
        last_completed = progress.completed;
        if (progress.total != 0 && progress.completed > progress.total) {
            monotonic = false;
        }
    };
    leann::Index::build(leann::index_file_from_prefix(prefix),
                        leann::documents_file_from_prefix(prefix), documents,
                        embedder, config);

    check(!stages.empty(), "a build reports progress when asked");
    check(monotonic,
          "progress never moves backwards and never exceeds its total");
    check(std::find(stages.begin(), stages.end(), "embedding") != stages.end(),
          "the embedding phase is reported");
    check(std::find(stages.begin(), stages.end(), "publishing") !=
              stages.end(),
          "the publication phase is reported last");
    check(stages.back() == "publishing",
          "publication is the final reported phase");
}

} // namespace

int main() {
    const std::string unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory = std::filesystem::temp_directory_path() /
                           ("leann-cli-cache-test-" + unique);
    std::filesystem::create_directories(directory);
    try {
        test_v2_stream_and_binding(directory);
        test_output_alias_rejection(directory);
        test_v1_benchmark_cache(directory);
        test_atomic_v1_cache_publication(directory);
        test_ground_truth_parser(directory);
        test_cached_fingerprint_is_persisted(directory);
        test_bench_skips_corpus_with_precomputed_truth(directory);
        test_cache_embedder_is_build_only();
        test_strict_cli_numeric_parsing();
        test_strict_option_validation();
        test_json_writer_escaping();
        test_search_json_output(directory);
        test_json_failure_is_atomic(directory);
        test_doctor_inspection_and_repair(directory);
        test_doctor_reports_lock_ownership(directory);
        test_build_cancellation(directory);
        test_build_progress_reporting(directory);
        std::filesystem::remove_all(directory);
        std::cout << "all CLI cache tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
