#include "llm_inference.h"
#include <algorithm>
#include <android/log.h>
#include <chrono>
#include <sstream>
#include <sys/stat.h>

#define LOG_TAG "LLMInference"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

LLMInference::LLMInference() : model(nullptr), ctx(nullptr) {
    LOGI("LLMInference instance created.");
}

LLMInference::~LLMInference() {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    llama_backend_free();
}

bool LLMInference::loadModel(const std::string &modelPath, int nThreads,
                             int nCtx) {
    struct stat st;
    if (stat(modelPath.c_str(), &st) != 0) {
        LOGE("Model file does not exist at %s", modelPath.c_str());
        return false;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();

#if defined(GGML_USE_OPENCL) || defined(GGML_USE_VULKAN)
    model_params.n_gpu_layers = -1;
#else
    model_params.n_gpu_layers = 0;
#endif

    model_params.use_mmap = true;
    model_params.use_mlock = false;

    model = llama_model_load_from_file(modelPath.c_str(), model_params);
    if (!model) {
        LOGE("Failed to load model file.");
        return false;
    }

    saved_ctx_params = llama_context_default_params();
    saved_ctx_params.n_ctx = nCtx;

    int threads_to_use = (nThreads > 0) ? nThreads : 4;
    saved_ctx_params.n_threads = threads_to_use;
    saved_ctx_params.n_threads_batch = threads_to_use;

    // Optimized batch size
    saved_ctx_params.n_batch = 512;
    saved_ctx_params.n_ubatch = 512;

    // Flash Attention
//    saved_ctx_params.flash_attn = true;

    // Optimized KV cache
    saved_ctx_params.type_k = GGML_TYPE_Q8_0;
    saved_ctx_params.type_v = GGML_TYPE_Q8_0;

    saved_ctx_params.defrag_thold = 0.1f;

    ctx = llama_init_from_model(model, saved_ctx_params);
    if (!ctx) {
        LOGE("Context initialization failed.");
        llama_model_free(model);
        model = nullptr;
        return false;
    }

    return true;
}

std::string LLMInference::generate(const std::string &prompt, int maxTokens,
                                   float temperature, float topP, int topK,
                                   StreamCallback callback) {
    if (!model || !ctx)
        return "";

    // Tokenize prompt
    const auto *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 32);

    int n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_prompt < 0) {
        tokens.resize(-n_prompt);
        n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                  tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_prompt);

    // Smart cache reuse
    int n_common = 0;
    size_t n_cached = kv_cache_tokens.size();

    for (; n_common < n_cached && n_common < n_prompt; n_common++) {
        if (kv_cache_tokens[n_common] != tokens[n_common])
            break;
    }

    // Remove divergent tokens from cache
    if (n_common < n_cached) {
        llama_memory_t lmem = llama_get_memory(ctx);
        llama_memory_seq_rm(lmem, -1, n_common, -1);
        kv_cache_tokens.resize(n_common);
    }

    // Update cache
    kv_cache_tokens.insert(kv_cache_tokens.end(),
                           tokens.begin() + n_common, tokens.end());

    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(topK));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Prefill only new tokens
    llama_batch batch = llama_batch_init(saved_ctx_params.n_batch, 0, 1);
    int n_processed = n_common;

    while (n_processed < n_prompt) {
        int n_to_process = std::min(static_cast<int>(saved_ctx_params.n_batch),
                                    n_prompt - n_processed);
        batch.n_tokens = 0;

        for (int i = 0; i < n_to_process; i++) {
            int idx = n_processed + i;
            batch.token[batch.n_tokens] = tokens[idx];
            batch.pos[batch.n_tokens] = idx;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = (idx == n_prompt - 1);
            batch.n_tokens++;
        }

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            llama_sampler_free(sampler);
            return "Error: Prefill failed";
        }
        n_processed += n_to_process;
    }

    // Generation loop
    std::string response;
    int n_cur = n_prompt;

    for (int i = 0; i < maxTokens; i++) {
        const llama_token id = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, id))
            break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        if (n > 0) {
            std::string piece(buf, n);
            response += piece;
            if (callback)
                callback(piece);
        }

        batch.n_tokens = 1;
        batch.token[0] = id;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (llama_decode(ctx, batch) != 0)
            break;

        kv_cache_tokens.push_back(id);
        n_cur++;
    }

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    return response;
}

void LLMInference::clearCache() {
    if (ctx) {
        llama_memory_t lmem = llama_get_memory(ctx);
        llama_memory_seq_rm(lmem, -1, -1, -1);
        kv_cache_tokens.clear();
    }
}