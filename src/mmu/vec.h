#pragma once
#include <cmath>
#include <cstdint>
#include <immintrin.h>

float dot_product(float* a, float* b, int dims);
float magnitude(float* a, int dims);
void normalize(float* a, int dims);
void quantize(float* input, int8_t* output, int dims);
float vnni_dot_product(int8_t* a, int8_t* b, int dims);