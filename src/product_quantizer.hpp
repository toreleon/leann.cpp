#pragma once

#include "leann/embedder.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace leann::detail {

struct ProductQuantizerView {
    std::uint32_t dimension = 0;
    std::uint32_t subquantizers = 0;
    std::uint32_t bits = 0;
    std::uint32_t centroids = 0;
    std::uint32_t subdimension = 0;
    std::span<const float> codebook;
    std::span<const std::uint8_t> codes;

    [[nodiscard]] std::size_t bytes_per_vector() const noexcept;
    [[nodiscard]] std::vector<float>
    distance_table(std::span<const float> query) const;
    [[nodiscard]] float approximate_distance(
        std::span<const float> table, std::size_t node) const;
};

struct ProductQuantizerModel {
    std::uint32_t dimension = 0;
    std::uint32_t subquantizers = 0;
    std::uint32_t bits = 0;
    std::uint32_t centroids = 0;
    std::uint32_t subdimension = 0;
    std::vector<float> codebook;
    std::vector<std::uint8_t> codes;

    [[nodiscard]] std::size_t bytes_per_vector() const noexcept;
    [[nodiscard]] std::vector<float>
    distance_table(std::span<const float> query) const;
    [[nodiscard]] float approximate_distance(
        std::span<const float> table, std::size_t node) const;
    [[nodiscard]] ProductQuantizerView view() const noexcept;
};

[[nodiscard]] ProductQuantizerModel train_product_quantizer(
    std::span<const Embedding> embeddings,
    std::uint32_t subquantizers,
    std::uint32_t bits,
    std::uint32_t iterations,
    std::uint32_t max_training_samples,
    std::uint32_t seed);

} // namespace leann::detail
