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
run_benchmark(const Arguments & arguments) {
    std::ostringstream captured_output;
    std::ostringstream captured_error;
    auto * old_output = std::cout.rdbuf(captured_output.rdbuf());
    auto * old_error = std::cerr.rdbuf(captured_error.rdbuf());
    try {
        command_bench(arguments);
    } catch (...) {
        std::cout.rdbuf(old_output);
        std::cerr.rdbuf(old_error);
        throw;
    }
    std::cout.rdbuf(old_output);
    std::cerr.rdbuf(old_error);
    return {captured_output.str(), captured_error.str()};
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
        std::filesystem::remove_all(directory);
        std::cout << "all CLI cache tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
