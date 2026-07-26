// Mutation fuzzing of the compact index parser.
//
// `Index::load` verifies a SHA-256 footer over the whole file before it parses
// any field (src/index.cpp), so a naively corrupted index is rejected at the
// digest and the parser is never reached. That gate is already covered by the
// truncation tests in test_index.cpp and test_c_api.cpp.
//
// The interesting adversary is the one SECURITY.md describes: a checksum is
// integrity, not authenticity, so anyone who can rewrite an artifact can
// rewrite its digest. This suite therefore mutates a field and then *re-seals*
// the file with a valid footer, which is the only way to exercise the header
// parser and the arithmetic derived from declared counts, offsets, and sizes.
//
// The contract under test is deliberately weak and therefore hard to argue
// with: for every mutant, `Index::load` must either succeed or throw something
// derived from std::exception. It must not crash, read out of bounds, overflow
// an integer conversion, or terminate. Out-of-bounds reads and undefined
// behaviour are caught by running this suite under ASan/UBSan, which is where
// its real value lies -- a pass without sanitizers proves much less.
//
// The header is swept positionally rather than through a hardcoded field-offset
// map. A map would silently rot the first time the format changes, and would
// only ever probe the fields whoever wrote it thought of.

#include "leann/document_store.hpp"
#include "leann/embedder.hpp"
#include "leann/index.hpp"

#include "checksum.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error("test failed: " + message);
    }
}

std::vector<std::string> make_corpus() {
    std::vector<std::string> documents;
    for (int i = 0; i < 20; ++i) {
        documents.push_back("apple pear orange orchard harvest " +
                            std::to_string(i));
        documents.push_back("kernel compiler vector database code " +
                            std::to_string(i));
        documents.push_back("planet galaxy telescope orbit star " +
                            std::to_string(i));
    }
    return documents;
}

[[nodiscard]] std::vector<std::uint8_t>
read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    check(input.good(), "fixture is readable");
    return std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

void write_file(const std::filesystem::path & path,
                const std::vector<std::uint8_t> & bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    check(output.good(), "mutant is writable");
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
    output.close();
    check(output.good(), "mutant is flushed");
}

// Writes `body` and re-seals it with a valid SHA-256 footer, so the mutation
// survives the digest gate and actually reaches the header parser.
void reseal(const std::filesystem::path & path,
            const std::vector<std::uint8_t> & body) {
    write_file(path, body);
    leann::detail::append_sha256_footer(path);
}

enum class Outcome { Survived, Rejected };

// The whole contract: every mutant either survives a full load-and-search
// cleanly, or is refused with a std::exception. Anything else -- a segfault, a
// sanitizer report, an abort -- fails the suite by killing the process, which
// is the intended detection mechanism.
//
// Searching matters as much as loading. A mutated node count or CSR offset can
// pass every load-time check and only dereference out of bounds once the graph
// is actually walked, so a load-only campaign would report a clean run while
// leaving the interesting population -- the mutants that *do* load -- entirely
// unexercised.
[[nodiscard]] Outcome attempt(const std::filesystem::path & path,
                              const leann::DocumentStore & documents,
                              leann::Embedder & embedder) {
    try {
        auto index = leann::Index::load(path);
        // Touch the loaded state so a mutant cannot pass by producing an
        // index that is structurally invalid but never inspected.
        volatile auto size = index.size();
        (void)size;
        volatile auto serialized = index.stats().serialized_bytes;
        (void)serialized;

        leann::SearchConfig search;
        search.top_k = 3;
        search.ef_search = 32;
        search.recompute_batch_size = 8;
        search.rerank_ratio = 0.5;
        // Both sides of the size threshold: below it the flat ADC scan runs,
        // above it the bounded base-graph beam does, and they dereference the
        // declared counts and offsets differently.
        for (const std::uint32_t limit : {0U, 1000000U}) {
            search.approximate_scan_limit = limit;
            const auto response = index.search("orchard harvest vector",
                                               embedder, documents, search);
            volatile auto found = response.results.size();
            (void)found;
        }
        return Outcome::Survived;
    } catch (const std::exception &) {
        return Outcome::Rejected;
    }
}

struct Tally {
    std::size_t attempts = 0;
    std::size_t survived = 0;
    std::size_t rejected = 0;

    void record(Outcome outcome) {
        ++attempts;
        if (outcome == Outcome::Survived) {
            ++survived;
        } else {
            ++rejected;
        }
    }
};

void store_u32(std::vector<std::uint8_t> & bytes, std::size_t offset,
               std::uint32_t value) {
    for (std::size_t i = 0; i < 4U; ++i) {
        bytes[offset + i] = static_cast<std::uint8_t>((value >> (8U * i)) &
                                                      0xFFU);
    }
}

} // namespace

int main() {
    const auto directory =
        std::filesystem::temp_directory_path() / "leann-cpp-artifact-fuzz";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    int status = 0;
    try {
        const auto prefix = directory / "fixture";
        const auto index_path = leann::index_file_from_prefix(prefix);
        const auto documents_path = leann::documents_file_from_prefix(prefix);
        const auto corpus = make_corpus();
        leann::HashEmbedder embedder(64);

        leann::BuildConfig build;
        build.graph_degree = 8;
        build.ef_construction = 40;
        build.low_degree = 3;
        build.hub_ratio = 0.05;
        build.approximation = leann::ApproximationKind::ProductQuantization;
        build.pq_subquantizers = 8;
        build.pq_bits = 4;
        build.pq_training_iterations = 4;
        leann::Index::build(index_path, documents_path, corpus, embedder,
                            build);

        const auto documents = leann::DocumentStore::open(documents_path);

        check(attempt(index_path, documents, embedder) == Outcome::Survived,
              "the unmutated fixture loads and searches");

        const auto original = read_file(index_path);
        constexpr std::size_t footer_bytes = 32U;
        check(original.size() > footer_bytes,
              "fixture is larger than its footer");
        const std::vector<std::uint8_t> body(
            original.begin(),
            original.end() - static_cast<std::ptrdiff_t>(footer_bytes));

        const auto mutant_path = directory / "mutant.leann";

        // A re-sealed but otherwise untouched body must still load. Without
        // this the whole suite could pass vacuously by re-sealing incorrectly
        // and having every mutant rejected at the digest after all.
        reseal(mutant_path, body);
        check(attempt(mutant_path, documents, embedder) == Outcome::Survived,
              "re-sealing preserves a valid index, so mutants reach the "
              "parser rather than bouncing off the digest");

        Tally header;
        // Every 32-bit field in the header region, set to values that provoke
        // overflow, sign confusion, and zero-size arithmetic.
        constexpr std::array<std::uint32_t, 8> probes{
            0U,          1U,          2U,
            0x0000FFFFU, 0x7FFFFFFFU, 0x80000000U,
            0xFFFFFFFEU, 0xFFFFFFFFU};
        const std::size_t sweep =
            std::min<std::size_t>(body.size(), 512U);
        for (std::size_t offset = 0; offset + 4U <= sweep; offset += 4U) {
            for (const std::uint32_t probe : probes) {
                auto mutant = body;
                store_u32(mutant, offset, probe);
                reseal(mutant_path, mutant);
                header.record(attempt(mutant_path, documents, embedder));
            }
        }

        Tally flips;
        // Single-bit flips spread across the whole body, including the payload
        // regions the header sweep never reaches.
        const std::size_t stride =
            body.size() > 256U ? body.size() / 256U : 1U;
        for (std::size_t offset = 0; offset < body.size(); offset += stride) {
            for (const int bit : {0, 7}) {
                auto mutant = body;
                mutant[offset] = static_cast<std::uint8_t>(
                    mutant[offset] ^ (1U << bit));
                reseal(mutant_path, mutant);
                flips.record(attempt(mutant_path, documents, embedder));
            }
        }

        Tally truncations;
        // Re-sealed truncations: the file is internally consistent as far as
        // the digest is concerned, but every declared extent now overruns it.
        const std::size_t truncation_step =
            body.size() > 128U ? body.size() / 128U : 1U;
        for (std::size_t length = 0; length < body.size();
             length += truncation_step) {
            const std::vector<std::uint8_t> shortened(
                body.begin(),
                body.begin() + static_cast<std::ptrdiff_t>(length));
            reseal(mutant_path, shortened);
            truncations.record(attempt(mutant_path, documents, embedder));
        }

        const std::size_t total =
            header.attempts + flips.attempts + truncations.attempts;
        std::cout << "  header field sweep: " << header.attempts
                  << " mutants (" << header.survived << " survived, "
                  << header.rejected << " rejected)\n"
                  << "  bit flips:          " << flips.attempts
                  << " mutants (" << flips.survived << " survived, "
                  << flips.rejected << " rejected)\n"
                  << "  truncations:        " << truncations.attempts
                  << " mutants (" << truncations.survived << " survived, "
                  << truncations.rejected << " rejected)\n";

        check(total > 500U, "the campaign actually ran a meaningful number "
                            "of mutants");
        // A campaign in which nothing is rejected would mean the mutations are
        // landing somewhere inert, not that the parser is robust.
        check(header.rejected > 0U, "header mutations are reaching the parser");
        check(truncations.rejected > 0U, "truncations are detected");

        std::cout << "all leann.cpp artifact fuzz tests passed (" << total
                  << " mutants, no crash, no sanitizer report)\n";
    } catch (const std::exception & error) {
        std::cerr << "artifact fuzz failure: " << error.what() << "\n";
        status = 1;
    }

    std::error_code cleanup;
    std::filesystem::remove_all(directory, cleanup);
    return status;
}
