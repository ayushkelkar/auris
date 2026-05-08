#include "vec.h"

// This is the search function to search through the database one by one
int brute_force_search(int8_t* query, int8_t** vectors, int num_vectors, int dims) {
    // Initializing best_score to -2.0 because it will needs to always pass as true first time to the if(score > best score)
    // Its a simple algorithm: score = dot product pairwise at i. If the dot is > best_score, the best score = dot.
    // More score == More similarity.
    // cosine function (basically the dot product in a nutshell) increases as the angle decreases (so more similarity).
    // so more dot product == more score == more similarity
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