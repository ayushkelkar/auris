#include "inf.h"
#include "../../third_party/llama.cpp/include/llama.h"
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>

// Initializing all the states
static llama_model   * model   = nullptr;
static llama_context * ctx     = nullptr;
static llama_sampler * sampler = nullptr;
static const llama_vocab * vocab = nullptr;

bool inference_init(const char* model_path, int n_ctx) {
    // Load the model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.use_mmap = false;
    model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        printf("Failed to load model from %s\n", model_path);
        return false;
    }

    // Get vocabulary
    vocab = llama_model_get_vocab(model);

    // Create Context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 512;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("Failed to create context\n");
        return false;
    }

    // Build Sampler Chain
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    return true;
}

// Track how many tokens are in the KV cache
static int n_past = 0;
// Track system prompt tokens so we only process it once
static int n_system_tokens = 0;

std::string infer(const std::string& input) {
    // Build system prompt once
    static const std::string system_prompt =
        "<|im_start|>system\nYou are Cela, a helpful AI companion.<|im_end|>\n";

    // On first call, process system prompt and cache it
    if (n_past == 0) {
        std::vector<llama_token> sys_tokens(system_prompt.size() + 32);
        int n = llama_tokenize(vocab, system_prompt.c_str(), system_prompt.size(),
                               sys_tokens.data(), sys_tokens.size(), true, true);
        if (n < 0) return "";
        sys_tokens.resize(n);

        llama_batch batch = llama_batch_get_one(sys_tokens.data(), n);
        if (llama_decode(ctx, batch) != 0) return "";
        n_past = n;
        n_system_tokens = n;
    }

    // Build only the new user turn
    std::string turn =
        "<|im_start|>user\n" + input + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n";

    std::vector<llama_token> tokens(turn.size() + 32);
    int n_tokens = llama_tokenize(vocab, turn.c_str(), turn.size(),
                                  tokens.data(), tokens.size(), false, true);
    if (n_tokens < 0) return "";
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) return "";
    n_past += n_tokens;

    // Generate output
    std::string output;
    auto t_start = std::chrono::high_resolution_clock::now();
    int n_generated = 0;

    while (true) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            output.append(buf, n);
            fwrite(buf, 1, n, stdout);
            fflush(stdout);
            n_generated++;
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, next) != 0) break;
        n_past++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "\n\n%.2f tokens/sec (%d tokens in %.2fs)\n",
            n_generated / elapsed, n_generated, elapsed);

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