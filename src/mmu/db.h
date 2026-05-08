#pragma once
#include <cstdint>

void load_db(const char* path, int8_t** &vectors, int &num_vectors, int &dims);