#include "inf.h"
#include "../../third_party/llama.cpp/ggml/include/ggml.h"
#include "../../third_party/llama.cpp/include/llama.h"
#include <string>
#include <vector>

// Global state
static llama_model   * model   = nullptr;
static llama_context * ctx     = nullptr;
static llama_sampler * sampler = nullptr;
static const llama_vocab * vocab = nullptr;

bool inference_init(const char* model_path, int n_ctx) {
    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only for now
    model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        printf("Failed to load model from %s\n", model_path);
        return false;
    }

    // Get vocab
    vocab = llama_model_get_vocab(model);

    // Create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 512;
    ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("Failed to create context\n");
        return false;
    }

    // Build sampler chain
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    return true;
}

std::string infer(const std::string& input) {
    // Tokenize input
    std::vector<llama_token> tokens(input.size() + 32);
    int n_tokens = llama_tokenize(vocab, input.c_str(), input.size(),
                                  tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        return "";
    }
    tokens.resize(n_tokens);

    // Decode input tokens
    for (int i = 0; i < n_tokens; i++) {
        llama_batch batch = llama_batch_get_one(&tokens[i], 1);
        if (llama_decode(ctx, batch) != 0) {
            return "";
        }
    }

    // Generate output
    std::string output;
    while (true) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n > 0) output.append(buf, n);

        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch) != 0) break;
    }

    return output;
}

void inference_free() {
    if (sampler) llama_sampler_free(sampler);
    if (ctx)     llama_free(ctx);
    if (model)   llama_model_free(model);
    sampler = nullptr;
    ctx     = nullptr;
    model   = nullptr;
}