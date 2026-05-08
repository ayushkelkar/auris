#include "db.h"
#include <cstdio>
#include <cstdint>

// Loads the db but also changed the function. load_db now no longer saves or deletes stuff from the db.
// It exits returning false if the file fails to open.
// Loads the vectors to memory if file opens.
bool load_db(const char* path, int8_t** &vectors, int &num_vectors, int &dims) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open file: %s\n", path);
        return false;
    }
    fread(&num_vectors, sizeof(int), 1, f);
    fread(&dims, sizeof(int), 1, f);
    vectors = new int8_t*[num_vectors];
    for(int i = 0; i < num_vectors; i++) {
        vectors[i] = new int8_t[dims];
        fread(vectors[i], sizeof(int8_t), dims, f);
    }
    fclose(f);
    return true;
}

// This save_db is for future use. Not implementing right now.
/*
void save_db(const char* path, int8_t** vectors, int num_vectors, int dims) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("Failed to open file for writing: %s\n", path);
        return;
    }
    fwrite(&num_vectors, sizeof(int), 1, f);
    fwrite(&dims, sizeof(int), 1, f);
    for(int i = 0; i < num_vectors; i++) {
        fwrite(vectors[i], sizeof(int8_t), dims, f);
    }
    fclose(f);
}
*/

// Since I'm using new everywhere, every load eats up RAM, its never deleted. This handles that.
void free_db(int8_t** vectors, int num_vectors) {
    for(int i = 0; i < num_vectors; i++)
        delete[] vectors[i];
    delete[] vectors;
}