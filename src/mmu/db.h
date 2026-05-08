#pragma once
#include <cstdint>

bool load_db(const char* path, int8_t** &vectors, int &num_vectors, int &dims);
//void save_db(const char* path, int8_t** vectors, int num_vectors, int dims);
void free_db(int8_t** vectors, int num_vectors);