//
// Created by Wizcoder on 02/01/26.
//

#ifndef SMSCLASSIFIERNEW_LLM_INFERENCE_H
#define SMSCLASSIFIERNEW_LLM_INFERENCE_H

#include "llama.h"
#include <functional>
#include <string>
#include <vector>

using StreamCallback = std::function<void(const std::string &)>;

class LLMInference {
public:
  LLMInference();
  ~LLMInference();

  bool loadModel(const std::string &modelPath, int nThreads = 4,
                 int nCtx = 2048);

  std::string generate(const std::string &prompt, int maxTokens = 256,
                       float temperature = 0.7f, float topP = 0.9f,
                       int topK = 40, StreamCallback callback = nullptr);

    void clearCache();

private:
    llama_model *model;
    llama_context *ctx;
    llama_context_params saved_ctx_params{};
    std::vector<llama_token> kv_cache_tokens;
};

#endif // SMSCLASSIFIERNEW_LLM_INFERENCE_H
