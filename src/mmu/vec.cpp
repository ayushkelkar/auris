#include "vec.h"

// Returns the dot product of 2 vectors a and b with dimension dims
float dot_product(float* a, float* b, int dims) {
    float result = 0.0f;
    for (int i = 0; i < dims; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Returns the magnitude of the vector
float magnitude(float* a, int dims) {
    return sqrt(dot_product(a, a, dims));
}

// This normalizes the vector(s) to reduce the cosine similarity function (no longer present here as it is no longer needed) to just (a dot b). Eliminates finding the magnitude and dividing by it.
void normalize(float* a, int dims) {
    float mag = magnitude(a, dims);
    for (int i = 0; i < dims; i++) {
        a[i] = a[i] / mag;
    }
}

// So this takes the input as the normalized vector, then quantizes it in a linear mapping from [-1, 1] -> [-127, 127]
void quantize(float* input, int8_t* output, int dims) {
    for(int i = 0; i < dims ; i++) {
        output[i] = (int8_t)std::round(input[i]*127);
    }
}

float vnni_dot_product(int8_t* a, int8_t* b, int dims) {
    // This function has 3 distinct parts. 
    // Part A1: Offset Encoding a by 128 so it is unsigned:
    uint8_t a_unsigned[dims];
    int32_t sumofb = 0;
    for(int i = 0; i < dims ; i++) {
        a_unsigned[i] =a[i]+128;
        // Part A2: Summing all elements of b into an int32 array:
        sumofb += b[i];
    }
    // Now we have an unsigned int8 a, and sum of all elements of b.  

    // Part B: SIMD Loop
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < dims; i += 32) {
        __m256i va = _mm256_loadu_si256((__m256i*)(a_unsigned + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b + i));
        acc = _mm256_dpbusd_avx_epi32(acc, va, vb);
    }
    // Part C: Storing the result as: res = sum of all elements in accumulator (an int8 value) and then subtracting (128 * sum of all terms of b) then dividing by 127^2 because without that, the res is in a quantized int8 space, not [-1,1]
    int32_t res = 0;
    int result[8];
    _mm256_storeu_si256((__m256i*)result, acc);
    for (int i = 0; i < 8; i++) res += result[i];
    res -= (128*sumofb);
    return (float)res/16129.0f;
}