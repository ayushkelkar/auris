//#include "mmu/vec.h"
//#include "mmu/db.h"
//#include "mmu/search.h"
#include "inference/inf.h"
#include <cstdio>
#include <iostream>

int main() {
    /*
    // Defining all the variables
    int num_vectors = 0, dims = 0, index = 0;
    int8_t** vectors = nullptr;

    // This is the hardcoded query vector
    float query[32] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    if (!load_db("mmu/mockdb.bin", vectors, num_vectors, dims)) {
        return 1;
    }
    // Now the database of mock vectors is fully loaded

    int8_t query_quantized[32];
    normalize(query, dims);
    quantize(query, query_quantized, dims);
    // Now we have a normalized and quantized query vector

    // Performing brute force search on vectors array
    index = brute_force_search(query_quantized, vectors, num_vectors, dims);
    printf("Best match index: %d\n", index);

    // save_db("mmu/mockdb.bin", vectors, num_vectors, dims);

    free_db(vectors, num_vectors);
    */

    // Inference test
    if (!inference_init("/home/ayush/.models/qwen3.6/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")) {
    printf("Failed to init inference\n");
    return 1;
    }

    std::string input;
    while (true) {
        printf("You: ");
        std::getline(std::cin, input);
        if (input == "exit") break;
        std::string response = infer(input);
        printf("LLM: %s\n", response.c_str());
    }
    inference_free();
    return 0;
}