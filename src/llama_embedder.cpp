#include "leann/embedder.hpp"

#include "checksum.hpp"

#include <llama.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace leann {
namespace {

struct BackendLifetime {
    BackendLifetime() {
        llama_log_set(log_callback, this);
        llama_backend_init();
    }
    ~BackendLifetime() {
        llama_backend_free();
    }

    static void log_callback(ggml_log_level level, const char * text,
                             void * user_data) {
        auto * lifetime = static_cast<BackendLifetime *>(user_data);
        std::lock_guard lock(lifetime->log_mutex);
        if (level != GGML_LOG_LEVEL_CONT) {
            lifetime->emit_continuation =
                level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR;
        }
        if (lifetime->emit_continuation) {
            std::fputs(text, stderr);
        }
    }

    std::mutex log_mutex;
    bool emit_continuation = false;
};

BackendLifetime & backend_lifetime() {
    static BackendLifetime lifetime;
    return lifetime;
}

struct ModelDeleter {
    void operator()(llama_model * model) const {
        llama_model_free(model);
    }
};

struct ContextDeleter {
    void operator()(llama_context * context) const {
        llama_free(context);
    }
};

struct BatchGuard {
    explicit BatchGuard(std::uint32_t tokens)
        : batch(llama_batch_init(static_cast<std::int32_t>(tokens), 0, 1)) {}
    ~BatchGuard() {
        llama_batch_free(batch);
    }
    BatchGuard(const BatchGuard &) = delete;
    BatchGuard & operator=(const BatchGuard &) = delete;

    llama_batch batch;
};

std::vector<llama_token> tokenize(const llama_vocab * vocab,
                                  const std::string & text) {
    const auto text_size = static_cast<std::int32_t>(
        std::min<std::size_t>(text.size(), std::numeric_limits<std::int32_t>::max()));
    std::int32_t count =
        llama_tokenize(vocab, text.data(), text_size, nullptr, 0, true, false);
    if (count == std::numeric_limits<std::int32_t>::min()) {
        throw std::runtime_error("llama.cpp token count overflow");
    }
    if (count < 0) {
        count = -count;
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(count));
    count = llama_tokenize(vocab, text.data(), text_size, tokens.data(), count,
                           true, false);
    if (count < 0) {
        throw std::runtime_error("llama.cpp tokenization failed");
    }
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

} // namespace

class LlamaEmbedder::Impl {
  public:
    explicit Impl(Config input_config) : config(std::move(input_config)) {
        (void)backend_lifetime();
        if (config.model_path.empty()) {
            throw std::invalid_argument("llama.cpp model path is required");
        }
        if (config.context_tokens == 0 || config.batch_tokens == 0 ||
            config.max_sequences == 0) {
            throw std::invalid_argument(
                "llama.cpp context, batch, and sequence sizes must be positive");
        }

        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.gpu_layers;
        model.reset(
            llama_model_load_from_file(config.model_path.c_str(), model_params));
        if (!model) {
            throw std::runtime_error("failed to load GGUF embedding model: " +
                                     config.model_path);
        }
        if (llama_model_has_encoder(model.get()) &&
            llama_model_has_decoder(model.get())) {
            throw std::runtime_error(
                "encoder-decoder models are not supported for embeddings");
        }

        auto context_params = llama_context_default_params();
        const std::uint64_t shared_context =
            static_cast<std::uint64_t>(config.context_tokens) *
            config.max_sequences;
        if (shared_context > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "llama.cpp context_tokens * max_sequences is too large");
        }
        context_params.n_ctx = std::max(
            static_cast<std::uint32_t>(shared_context), config.batch_tokens);
        context_params.n_batch = config.batch_tokens;
        context_params.n_ubatch = config.batch_tokens;
        context_params.n_seq_max = config.max_sequences;
        context_params.embeddings = true;
        const int threads =
            config.threads > 0
                ? config.threads
                : std::max(1U, std::thread::hardware_concurrency());
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context.reset(llama_init_from_model(model.get(), context_params));
        if (!context) {
            throw std::runtime_error(
                "failed to create llama.cpp embedding context");
        }
        // UNSPECIFIED is negative and would become a nonsense value in the
        // index's unsigned pooling field, so it is rejected here alongside
        // NONE rather than being recorded as 4294967295.
        if (llama_pooling_type(context.get()) == LLAMA_POOLING_TYPE_NONE ||
            llama_pooling_type(context.get()) < 0) {
            throw std::runtime_error(
                "GGUF model has no sequence pooling; use an embedding model "
                "with mean, CLS, or last-token pooling metadata");
        }

        const int output_dimension = llama_model_n_embd_out(model.get());
        if (output_dimension <= 0) {
            throw std::runtime_error("GGUF model reports an invalid embedding size");
        }
        dimension = static_cast<std::size_t>(output_dimension);

        std::array<char, 512> description{};
        llama_model_desc(model.get(), description.data(), description.size());
        fingerprint_value =
            "llama.cpp-v1:" + std::string(description.data()) + ":" +
            std::to_string(llama_model_size(model.get())) + ":" +
            std::to_string(dimension) + ":" +
            std::to_string(static_cast<int>(llama_pooling_type(context.get()))) +
            ":gpu-layers=" + std::to_string(config.gpu_layers);
    }

    std::vector<Embedding> embed(std::span<const std::string> texts) {
        std::lock_guard lock(mutex);
        if (texts.empty()) {
            return {};
        }

        std::vector<std::vector<llama_token>> inputs;
        inputs.reserve(texts.size());
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        for (const std::string & text : texts) {
            auto tokens = tokenize(vocab, text);
            if (tokens.empty()) {
                throw std::runtime_error("llama.cpp produced an empty token sequence");
            }
            const std::size_t sequence_capacity =
                llama_n_ctx_seq(context.get());
            if (tokens.size() > sequence_capacity ||
                tokens.size() > config.batch_tokens) {
                throw std::runtime_error(
                    "document has " + std::to_string(tokens.size()) +
                    " tokens but the llama.cpp per-sequence/batch limit is " +
                    std::to_string(std::min<std::size_t>(
                        sequence_capacity, config.batch_tokens)));
            }
            inputs.push_back(std::move(tokens));
        }

        std::vector<Embedding> output(texts.size());
        std::size_t begin = 0;
        while (begin < inputs.size()) {
            std::size_t end = begin;
            std::size_t total_tokens = 0;
            while (end < inputs.size() &&
                   end - begin < config.max_sequences &&
                   total_tokens + inputs[end].size() <= config.batch_tokens) {
                total_tokens += inputs[end].size();
                ++end;
            }
            if (end == begin) {
                throw std::runtime_error(
                    "a tokenized document does not fit in the llama.cpp batch");
            }

            BatchGuard guard(config.batch_tokens);
            llama_batch & batch = guard.batch;
            batch.n_tokens = 0;
            for (std::size_t sequence = begin; sequence < end; ++sequence) {
                const llama_seq_id sequence_id =
                    static_cast<llama_seq_id>(sequence - begin);
                const auto & tokens = inputs[sequence];
                for (std::size_t position = 0; position < tokens.size();
                     ++position) {
                    const int slot = batch.n_tokens++;
                    batch.token[slot] = tokens[position];
                    batch.pos[slot] = static_cast<llama_pos>(position);
                    batch.n_seq_id[slot] = 1;
                    batch.seq_id[slot][0] = sequence_id;
                    batch.logits[slot] = 1;
                }
            }

            const bool encoder = llama_model_has_encoder(model.get());
            if (!encoder) {
                llama_memory_clear(llama_get_memory(context.get()), true);
            }
            const int result =
                encoder ? llama_encode(context.get(), batch)
                        : llama_decode(context.get(), batch);
            if (result != 0) {
                throw std::runtime_error(
                    "llama.cpp failed to compute embeddings (code " +
                    std::to_string(result) + ")");
            }

            for (std::size_t sequence = begin; sequence < end; ++sequence) {
                const llama_seq_id sequence_id =
                    static_cast<llama_seq_id>(sequence - begin);
                const float * values =
                    llama_get_embeddings_seq(context.get(), sequence_id);
                if (values == nullptr) {
                    throw std::runtime_error(
                        "llama.cpp did not return a pooled sequence embedding");
                }
                Embedding embedding(values, values + dimension);
                normalize(embedding);
                output[sequence] = std::move(embedding);
            }
            begin = end;
        }
        return output;
    }

    // Digesting the GGUF is deferred to the first call because only a build
    // needs it. A search loads the same model and must not pay to hash it.
    EmbedderDescriptor describe() {
        std::lock_guard lock(mutex);
        if (!descriptor_value.source.empty()) {
            return descriptor_value;
        }
        EmbedderDescriptor result;
        result.source = config.model_source.empty() ? config.model_path
                                                    : config.model_source;
        result.pooling_type = static_cast<std::uint32_t>(
            llama_pooling_type(context.get()));
        result.context_tokens = config.context_tokens;
        // The file size, deliberately: llama_model_size() reports the total
        // tensor bytes, which is a smaller and different number, and it is a
        // fetched file that a consumer has to check a digest against.
        const std::filesystem::path model_file(config.model_path);
        result.bytes =
            static_cast<std::uint64_t>(std::filesystem::file_size(model_file));
        result.sha256 = detail::sha256_file_prefix(model_file, result.bytes);
        descriptor_value = result;
        return result;
    }

    Config config;
    std::unique_ptr<llama_model, ModelDeleter> model;
    std::unique_ptr<llama_context, ContextDeleter> context;
    std::size_t dimension = 0;
    std::string fingerprint_value;
    EmbedderDescriptor descriptor_value;
    std::mutex mutex;
};

LlamaEmbedder::LlamaEmbedder(Config config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

LlamaEmbedder::~LlamaEmbedder() = default;
LlamaEmbedder::LlamaEmbedder(LlamaEmbedder &&) noexcept = default;
LlamaEmbedder & LlamaEmbedder::operator=(LlamaEmbedder &&) noexcept = default;

std::size_t LlamaEmbedder::dimension() const noexcept {
    return impl_->dimension;
}

std::string LlamaEmbedder::fingerprint() const {
    return impl_->fingerprint_value;
}

EmbedderDescriptor LlamaEmbedder::descriptor() const {
    return impl_->describe();
}

std::vector<Embedding>
LlamaEmbedder::embed(std::span<const std::string> texts) {
    return impl_->embed(texts);
}

} // namespace leann
