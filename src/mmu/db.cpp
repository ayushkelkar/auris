#include <cstdio>
#include <cstdint>

// This function loads the database binary into memory
// Also just note: vectors is a pointer to a pointer (indicated by **). In db.cpp its a pointer pointing to the pointer vectors which points to an array location.
void load_db(const char* path, int8_t** &vectors, int &num_vectors, int &dims) {
    // Loading the file from path
    FILE* f = fopen(path, "rb");
    if (!f) printf("Failed to open File.\n");
    else {
        fread(&num_vectors, sizeof(int), 1, f); // Reads the pointer num_vectors
        fread(&dims, sizeof(int), 1, f); // Reads the pointer dims
        vectors = new int8_t*[num_vectors]; // Initializes a new int8_t pointer to vectors array the size of num_vectors
        // Assigns all values from database to vectors[i]
        for(int i = 0; i < num_vectors; i++) {
            vectors[i] = new int8_t[dims];
            fread(vectors[i], sizeof(int8_t), dims, f);
        }
        fclose(f); // Closes the file stream
    }
}