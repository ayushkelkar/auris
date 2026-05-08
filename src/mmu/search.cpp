#include "vec.h"

int brute_force_search(int8_t* query, int8_t** vectors, int num_vectors, int dims) {
    float best_score = -2.0f, score = 0.0f;
    int best_index = -1;
    for (int i = 0; i < num_vectors; i++) {
        score = vnni_dot_product(query, vectors[i], dims);
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }
    return best_index;
}