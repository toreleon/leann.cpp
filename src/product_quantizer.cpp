#include "product_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace leann::detail {
namespace {

std::uint32_t unpack_code(std::span<const std::uint8_t> packed,
                          std::size_t vector_offset_bits,
                          std::uint32_t bits) {
    const std::size_t byte = vector_offset_bits / 8U;
    const std::uint32_t shift =
        static_cast<std::uint32_t>(vector_offset_bits % 8U);
    std::uint32_t value = packed[byte] >> shift;
    if (shift + bits > 8U) {
        value |= static_cast<std::uint32_t>(packed[byte + 1]) << (8U - shift);
    }
    return value & ((1U << bits) - 1U);
}

void pack_code(std::span<std::uint8_t> packed,
               std::size_t vector_offset_bits,
               std::uint32_t bits,
               std::uint32_t value) {
    const std::size_t byte = vector_offset_bits / 8U;
    const std::uint32_t shift =
        static_cast<std::uint32_t>(vector_offset_bits % 8U);
    packed[byte] |= static_cast<std::uint8_t>(value << shift);
    if (shift + bits > 8U) {
        packed[byte + 1] |=
            static_cast<std::uint8_t>(value >> (8U - shift));
    }
}

float squared_distance(std::span<const float> point,
                       std::span<const float> centroid) {
    double result = 0.0;
    for (std::size_t i = 0; i < point.size(); ++i) {
        const double delta =
            static_cast<double>(point[i]) - centroid[i];
        result += delta * delta;
    }
    if (!std::isfinite(result) ||
        result > std::numeric_limits<float>::max()) {
        throw std::runtime_error("PQ squared distance is not finite");
    }
    return static_cast<float>(result);
}

} // namespace

std::size_t ProductQuantizerView::bytes_per_vector() const noexcept {
    return (static_cast<std::size_t>(subquantizers) * bits + 7U) / 8U;
}

std::vector<float>
ProductQuantizerView::distance_table(std::span<const float> query) const {
    if (query.size() != dimension) {
        throw std::invalid_argument("PQ query dimension mismatch");
    }
    if (!std::all_of(query.begin(), query.end(), [](float value) {
            return std::isfinite(value) && std::abs(value) <= 1.0F;
        })) {
        throw std::invalid_argument(
            "PQ query must contain finite normalized coordinates");
    }
    std::vector<float> table(
        static_cast<std::size_t>(subquantizers) * centroids, 0.0F);
    for (std::uint32_t sub = 0; sub < subquantizers; ++sub) {
        const std::size_t query_begin =
            static_cast<std::size_t>(sub) * subdimension;
        for (std::uint32_t centroid = 0; centroid < centroids; ++centroid) {
            const std::size_t codebook_begin =
                (static_cast<std::size_t>(sub) * centroids + centroid) *
                subdimension;
            double negative_dot = 0.0;
            for (std::uint32_t coordinate = 0; coordinate < subdimension;
                 ++coordinate) {
                const float centroid_value =
                    codebook[codebook_begin + coordinate];
                if (!std::isfinite(centroid_value) ||
                    std::abs(centroid_value) > 1.0F) {
                    throw std::runtime_error(
                        "PQ codebook contains a non-finite or "
                        "out-of-range value");
                }
                negative_dot -=
                    static_cast<double>(query[query_begin + coordinate]) *
                    centroid_value;
            }
            if (!std::isfinite(negative_dot) ||
                negative_dot > std::numeric_limits<float>::max() ||
                negative_dot < -std::numeric_limits<float>::max()) {
                throw std::runtime_error("PQ distance table is not finite");
            }
            table[static_cast<std::size_t>(sub) * centroids + centroid] =
                static_cast<float>(negative_dot);
        }
    }
    return table;
}

float ProductQuantizerView::approximate_distance(
    std::span<const float> table, std::size_t node) const {
    if (table.size() !=
        static_cast<std::size_t>(subquantizers) * centroids) {
        throw std::invalid_argument("PQ distance table size mismatch");
    }
    const std::size_t bytes = bytes_per_vector();
    if (node >= codes.size() / bytes) {
        throw std::out_of_range("PQ node id is out of range");
    }
    const auto packed =
        std::span<const std::uint8_t>(codes).subspan(node * bytes, bytes);
    double result = 1.0;
    for (std::uint32_t sub = 0; sub < subquantizers; ++sub) {
        const std::uint32_t code =
            unpack_code(packed, static_cast<std::size_t>(sub) * bits, bits);
        if (code >= centroids) {
            throw std::runtime_error("PQ code references an invalid centroid");
        }
        const float contribution =
            table[static_cast<std::size_t>(sub) * centroids + code];
        if (!std::isfinite(contribution)) {
            throw std::invalid_argument(
                "PQ distance table contains NaN or infinity");
        }
        result += contribution;
    }
    if (!std::isfinite(result) ||
        result > std::numeric_limits<float>::max() ||
        result < -std::numeric_limits<float>::max()) {
        throw std::runtime_error("PQ approximate distance is not finite");
    }
    return static_cast<float>(result);
}

std::size_t ProductQuantizerModel::bytes_per_vector() const noexcept {
    return view().bytes_per_vector();
}

std::vector<float>
ProductQuantizerModel::distance_table(std::span<const float> query) const {
    return view().distance_table(query);
}

float ProductQuantizerModel::approximate_distance(
    std::span<const float> table, std::size_t node) const {
    return view().approximate_distance(table, node);
}

ProductQuantizerView ProductQuantizerModel::view() const noexcept {
    return {
        dimension,
        subquantizers,
        bits,
        centroids,
        subdimension,
        codebook,
        codes,
    };
}

ProductQuantizerModel train_product_quantizer(
    std::span<const Embedding> embeddings,
    std::uint32_t subquantizers,
    std::uint32_t bits,
    std::uint32_t iterations,
    std::uint32_t max_training_samples,
    std::uint32_t seed) {
    if (embeddings.empty()) {
        throw std::invalid_argument("cannot train PQ on no embeddings");
    }
    const std::size_t dimension = embeddings.front().size();
    if (dimension == 0 || subquantizers == 0 ||
        dimension % subquantizers != 0) {
        throw std::invalid_argument(
            "PQ subquantizers must evenly divide the embedding dimension");
    }
    if (bits == 0 || bits > 8) {
        throw std::invalid_argument("PQ bits must be in [1, 8]");
    }
    if (iterations == 0 || max_training_samples == 0) {
        throw std::invalid_argument(
            "PQ iterations and training sample count must be positive");
    }
    for (const Embedding & embedding : embeddings) {
        if (embedding.size() != dimension) {
            throw std::invalid_argument("PQ training dimensions differ");
        }
        if (!std::all_of(embedding.begin(), embedding.end(), [](float value) {
                return std::isfinite(value) && std::abs(value) <= 1.0F;
            })) {
            throw std::invalid_argument(
                "PQ training embeddings must contain finite normalized "
                "coordinates");
        }
    }

    std::vector<std::size_t> sample_ids(embeddings.size());
    std::iota(sample_ids.begin(), sample_ids.end(), 0U);
    std::mt19937 generator(seed);
    if (sample_ids.size() > max_training_samples) {
        std::shuffle(sample_ids.begin(), sample_ids.end(), generator);
        sample_ids.resize(max_training_samples);
    }
    const std::uint32_t requested_centroids = 1U << bits;
    const std::uint32_t centroids = static_cast<std::uint32_t>(
        std::min<std::size_t>(requested_centroids, sample_ids.size()));
    const std::uint32_t subdimension =
        static_cast<std::uint32_t>(dimension / subquantizers);

    ProductQuantizerModel model;
    model.dimension = static_cast<std::uint32_t>(dimension);
    model.subquantizers = subquantizers;
    model.bits = bits;
    model.centroids = centroids;
    model.subdimension = subdimension;
    model.codebook.resize(static_cast<std::size_t>(subquantizers) * centroids *
                          subdimension);

    std::vector<std::uint32_t> assignments(sample_ids.size(), 0);
    std::vector<float> minimum_distances(sample_ids.size());
    for (std::uint32_t sub = 0; sub < subquantizers; ++sub) {
        const std::size_t coordinate_begin =
            static_cast<std::size_t>(sub) * subdimension;
        float * sub_codebook =
            model.codebook.data() +
            static_cast<std::size_t>(sub) * centroids * subdimension;

        std::uniform_int_distribution<std::size_t> first_distribution(
            0, sample_ids.size() - 1);
        const Embedding & first =
            embeddings[sample_ids[first_distribution(generator)]];
        std::copy_n(first.data() + coordinate_begin, subdimension,
                    sub_codebook);

        std::fill(minimum_distances.begin(), minimum_distances.end(),
                  std::numeric_limits<float>::infinity());
        for (std::uint32_t centroid = 1; centroid < centroids; ++centroid) {
            const float * previous =
                sub_codebook +
                static_cast<std::size_t>(centroid - 1U) * subdimension;
            double total = 0.0;
            for (std::size_t sample = 0; sample < sample_ids.size(); ++sample) {
                const Embedding & point = embeddings[sample_ids[sample]];
                const float distance = squared_distance(
                    std::span<const float>(point).subspan(coordinate_begin,
                                                         subdimension),
                    std::span<const float>(previous, subdimension));
                minimum_distances[sample] =
                    std::min(minimum_distances[sample], distance);
                total += minimum_distances[sample];
            }

            std::size_t selected = centroid % sample_ids.size();
            if (total > 0.0) {
                std::uniform_real_distribution<double> distribution(0.0, total);
                double target = distribution(generator);
                for (std::size_t sample = 0; sample < sample_ids.size();
                     ++sample) {
                    target -= minimum_distances[sample];
                    if (target <= 0.0) {
                        selected = sample;
                        break;
                    }
                }
            }
            const Embedding & point = embeddings[sample_ids[selected]];
            std::copy_n(point.data() + coordinate_begin, subdimension,
                        sub_codebook +
                            static_cast<std::size_t>(centroid) * subdimension);
        }

        std::vector<double> sums(
            static_cast<std::size_t>(centroids) * subdimension);
        std::vector<std::uint32_t> counts(centroids);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0U);
            for (std::size_t sample = 0; sample < sample_ids.size(); ++sample) {
                const Embedding & point = embeddings[sample_ids[sample]];
                const auto subvector =
                    std::span<const float>(point).subspan(coordinate_begin,
                                                         subdimension);
                float best_distance = std::numeric_limits<float>::infinity();
                std::uint32_t best = 0;
                for (std::uint32_t centroid = 0; centroid < centroids;
                     ++centroid) {
                    const float distance = squared_distance(
                        subvector,
                        std::span<const float>(
                            sub_codebook +
                                static_cast<std::size_t>(centroid) *
                                    subdimension,
                            subdimension));
                    if (distance < best_distance) {
                        best_distance = distance;
                        best = centroid;
                    }
                }
                assignments[sample] = best;
                ++counts[best];
                for (std::uint32_t coordinate = 0;
                     coordinate < subdimension; ++coordinate) {
                    sums[static_cast<std::size_t>(best) * subdimension +
                         coordinate] += subvector[coordinate];
                }
                minimum_distances[sample] = best_distance;
            }

            for (std::uint32_t centroid = 0; centroid < centroids; ++centroid) {
                float * destination =
                    sub_codebook +
                    static_cast<std::size_t>(centroid) * subdimension;
                if (counts[centroid] == 0) {
                    const auto farthest = std::max_element(
                        minimum_distances.begin(), minimum_distances.end());
                    const std::size_t sample = static_cast<std::size_t>(
                        std::distance(minimum_distances.begin(), farthest));
                    const Embedding & point = embeddings[sample_ids[sample]];
                    std::copy_n(point.data() + coordinate_begin, subdimension,
                                destination);
                    *farthest = 0.0F;
                    continue;
                }
                for (std::uint32_t coordinate = 0;
                     coordinate < subdimension; ++coordinate) {
                    destination[coordinate] = static_cast<float>(
                        sums[static_cast<std::size_t>(centroid) * subdimension +
                             coordinate] /
                        counts[centroid]);
                }
            }
        }
    }

    const std::size_t bytes_per_vector = model.bytes_per_vector();
    model.codes.assign(embeddings.size() * bytes_per_vector, 0);
    for (std::size_t node = 0; node < embeddings.size(); ++node) {
        auto packed = std::span<std::uint8_t>(model.codes)
                          .subspan(node * bytes_per_vector, bytes_per_vector);
        for (std::uint32_t sub = 0; sub < subquantizers; ++sub) {
            const std::size_t coordinate_begin =
                static_cast<std::size_t>(sub) * subdimension;
            const auto subvector =
                std::span<const float>(embeddings[node])
                    .subspan(coordinate_begin, subdimension);
            float best_distance = std::numeric_limits<float>::infinity();
            std::uint32_t best = 0;
            for (std::uint32_t centroid = 0; centroid < centroids; ++centroid) {
                const std::size_t codebook_begin =
                    (static_cast<std::size_t>(sub) * centroids + centroid) *
                    subdimension;
                const float distance = squared_distance(
                    subvector,
                    std::span<const float>(
                        model.codebook.data() + codebook_begin, subdimension));
                if (distance < best_distance) {
                    best_distance = distance;
                    best = centroid;
                }
            }
            pack_code(packed, static_cast<std::size_t>(sub) * bits, bits, best);
        }
    }
    return model;
}

} // namespace leann::detail
