#include "mmu/vec.h"
#include "mmu/db.h"
#include "mmu/search.h"

int main() {
    // This is the hardcoded query
    float query[32] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int8_t** vectors = nullptr;
    int num_vectors = 0;
    int dims = 32;
    load_db("mmu/mockdb.bin", vectors, num_vectors, dims);
    // Now the database of mock vectors is fully loaded

    int8_t query_quantized[32];
    normalize(query, dims);
    quantize(query, query_quantized, dims);
    // Now we have a quantized query vector

    int index = 0;
    index = brute_force_search(query_quantized, vectors, num_vectors, dims);
    printf("Best match index: %d\n", index);
    return 0;
}