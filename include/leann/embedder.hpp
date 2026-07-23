#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace leann {

using Embedding = std::vector<float>;

class Embedder {
  public:
    virtual ~Embedder() = default;

    [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
    [[nodiscard]] virtual std::string fingerprint() const = 0;
    [[nodiscard]] virtual std::vector<Embedding>
    embed(std::span<const std::string> texts) = 0;
};

class HashEmbedder final : public Embedder {
  public:
    explicit HashEmbedder(std::size_t dimension = 256);

    [[nodiscard]] std::size_t dimension() const noexcept override;
    [[nodiscard]] std::string fingerprint() const override;
    [[nodiscard]] std::vector<Embedding>
    embed(std::span<const std::string> texts) override;

  private:
    std::size_t dimension_;
};

#ifdef LEANN_WITH_LLAMA

class LlamaEmbedder final : public Embedder {
  public:
    struct Config {
        std::string model_path;
        // Per-sequence token capacity. llama.cpp receives this multiplied by
        // max_sequences as its shared context size.
        std::uint32_t context_tokens = 512;
        std::uint32_t batch_tokens = 2048;
        std::uint32_t max_sequences = 8;
        int threads = 0;
        int gpu_layers = 99;
    };

    explicit LlamaEmbedder(Config config);
    ~LlamaEmbedder() override;
    LlamaEmbedder(LlamaEmbedder &&) noexcept;
    LlamaEmbedder & operator=(LlamaEmbedder &&) noexcept;
    LlamaEmbedder(const LlamaEmbedder &) = delete;
    LlamaEmbedder & operator=(const LlamaEmbedder &) = delete;

    [[nodiscard]] std::size_t dimension() const noexcept override;
    [[nodiscard]] std::string fingerprint() const override;
    [[nodiscard]] std::vector<Embedding>
    embed(std::span<const std::string> texts) override;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif

void normalize(Embedding & embedding);
[[nodiscard]] float cosine_distance(std::span<const float> lhs,
                                    std::span<const float> rhs);

} // namespace leann
