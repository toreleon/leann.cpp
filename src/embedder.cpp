#include "leann/embedder.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace leann {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

std::uint64_t fnv1a(std::string_view value, std::uint64_t seed = fnv_offset) {
    std::uint64_t hash = seed;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= fnv_prime;
    }
    return hash;
}

void add_feature(Embedding & output, std::string_view feature, float weight) {
    const std::uint64_t hash = fnv1a(feature);
    const std::size_t position = static_cast<std::size_t>(hash % output.size());
    const float sign = ((hash >> 32U) & 1U) == 0U ? 1.0F : -1.0F;
    output[position] += weight * sign;
}

Embedding hash_embed(std::string_view text, std::size_t dimension) {
    Embedding output(dimension, 0.0F);
    std::string normalized;
    normalized.reserve(text.size());
    for (const unsigned char value : text) {
        if (std::isalnum(value) != 0 || value >= 0x80U) {
            normalized.push_back(static_cast<char>(std::tolower(value)));
        } else {
            normalized.push_back(' ');
        }
    }

    std::string previous;
    std::size_t cursor = 0;
    while (cursor < normalized.size()) {
        while (cursor < normalized.size() && normalized[cursor] == ' ') {
            ++cursor;
        }
        const std::size_t begin = cursor;
        while (cursor < normalized.size() && normalized[cursor] != ' ') {
            ++cursor;
        }
        if (begin == cursor) {
            continue;
        }

        const std::string token = normalized.substr(begin, cursor - begin);
        add_feature(output, token, 1.0F);
        if (!previous.empty()) {
            std::string bigram = previous;
            bigram.push_back('\x1f');
            bigram.append(token);
            add_feature(output, bigram, 0.6F);
        }
        previous = token;
    }

    if (std::all_of(output.begin(), output.end(),
                    [](float value) { return value == 0.0F; })) {
        add_feature(output, "<empty>", 1.0F);
    }
    normalize(output);
    return output;
}

} // namespace

HashEmbedder::HashEmbedder(std::size_t dimension) : dimension_(dimension) {
    if (dimension_ < 8) {
        throw std::invalid_argument("hash embedding dimension must be at least 8");
    }
}

std::size_t HashEmbedder::dimension() const noexcept {
    return dimension_;
}

std::string HashEmbedder::fingerprint() const {
    return "leann-hash-v1:" + std::to_string(dimension_);
}

std::vector<Embedding>
HashEmbedder::embed(std::span<const std::string> texts) {
    std::vector<Embedding> result;
    result.reserve(texts.size());
    for (const std::string & text : texts) {
        result.push_back(hash_embed(text, dimension_));
    }
    return result;
}

void normalize(Embedding & embedding) {
    double squared_norm = 0.0;
    for (const float value : embedding) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "embedding contains NaN or infinity");
        }
        squared_norm += static_cast<double>(value) * value;
    }
    if (!std::isfinite(squared_norm)) {
        throw std::runtime_error("embedding norm is not finite");
    }
    if (squared_norm <= 0.0) {
        throw std::runtime_error("embedding has zero norm");
    }
    const double scale = 1.0 / std::sqrt(squared_norm);
    if (!std::isfinite(scale)) {
        throw std::runtime_error("embedding normalization scale is not finite");
    }
    for (float & value : embedding) {
        const double normalized = static_cast<double>(value) * scale;
        if (!std::isfinite(normalized) ||
            normalized > std::numeric_limits<float>::max() ||
            normalized < -std::numeric_limits<float>::max()) {
            throw std::runtime_error(
                "normalized embedding contains NaN or infinity");
        }
        value = static_cast<float>(normalized);
    }
}

float cosine_distance(std::span<const float> lhs, std::span<const float> rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("embedding dimensions do not match");
    }
    double dot = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i])) {
            throw std::invalid_argument(
                "cosine distance input contains NaN or infinity");
        }
        dot += static_cast<double>(lhs[i]) * rhs[i];
    }
    const double distance = 1.0 - dot;
    if (!std::isfinite(distance) ||
        distance > std::numeric_limits<float>::max() ||
        distance < -std::numeric_limits<float>::max()) {
        throw std::runtime_error("cosine distance is not finite");
    }
    return static_cast<float>(distance);
}

} // namespace leann
