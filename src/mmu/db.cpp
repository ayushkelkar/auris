#include <cstdio>
#include <cstdint>

void load_db(const char* path, int8_t** &vectors, int &num_vectors, int &dims) {
    FILE* f = fopen(path, "rb");
    if (!f) printf("Failed to open File.\n");
    else {
        fread(&num_vectors, sizeof(int), 1, f);
        fread(&dims, sizeof(int), 1, f);
        vectors = new int8_t*[num_vectors];
        for(int i = 0; i < num_vectors; i++) {
            vectors[i] = new int8_t[dims];
            fread(vectors[i], sizeof(int8_t), dims, f);
        }
        fclose(f);
    }
}