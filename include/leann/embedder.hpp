#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace leann {

using Embedding = std::vector<float>;

// Resolvable identity of the weights behind an embedder, recorded in the index
// so a downloaded artifact names the model it needs instead of only failing a
// fingerprint comparison after the fact.
//
// This is a pointer plus a content digest, not a trust root. `source` is an
// unverified publisher claim; `sha256` identifies the exact file that was read
// at build time and lets a fetched copy be checked against it. Neither
// establishes that the publisher is trustworthy.
struct EmbedderDescriptor {
    // Where the weights came from, in the publisher's own words. The CLI
    // records "hf:OWNER/REPO/FILE" when told to; otherwise the local path.
    std::string source;
    // SHA-256 of the model file as read at build time. All zero when the
    // embedder has no model file (the hash backend, a caller callback).
    std::array<std::uint8_t, 32> sha256{};
    // Size of that same file in bytes. Deliberately the file size, not
    // llama.cpp's tensor-byte total, which is a different and smaller number.
    std::uint64_t bytes = 0;
    // Pooling mode and per-sequence token budget, both of which change the
    // embedding a given text produces.
    std::uint32_t pooling_type = 0;
    std::uint32_t context_tokens = 0;
};

class Embedder {
  public:
    virtual ~Embedder() = default;

    [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
    [[nodiscard]] virtual std::string fingerprint() const = 0;
    [[nodiscard]] virtual std::vector<Embedding>
    embed(std::span<const std::string> texts) = 0;

    // Non-pure so the eight existing implementations keep compiling and the
    // interface stays cheap to implement. An embedder with no model file
    // correctly describes itself with the default: empty source, zero digest.
    // Called once per build, never on the search path, so an implementation
    // may do real work here (hashing a multi-gigabyte GGUF, say).
    [[nodiscard]] virtual EmbedderDescriptor descriptor() const {
        return {};
    }
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
        // Recorded in the index as the model's origin. Empty means the index
        // records model_path instead, which is honest but not fetchable.
        std::string model_source;
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
    [[nodiscard]] EmbedderDescriptor descriptor() const override;
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
