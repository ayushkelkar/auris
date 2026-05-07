#include <iostream>
#include <cmath>
#include <cstdint>

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

// Performs cosine similarity on the vectors a and b
// Specifically: cos(angle) = (a dot b)/(mag(a).mag(b))
float cosine_similarity(float* a, float* b, int dims) {
    float dot   = dot_product(a, b, dims);
    float mag_a = magnitude(a, dims);
    float mag_b = magnitude(b, dims);
    return dot / (mag_a * mag_b);
}

// This normalizes the vector(s) to reduce the cosine similarity function to just (a dot b). Eliminates finding the magnitude and dividing by it.
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

// Now this is the opposite of quantize()
void dequantize(int8_t* input, float* output, int dims) {
    for(int i = 0; i < dims ; i++) {
        output[i] = input[i]/127.0f;
    }
}

int main() {
    float a[3] = {5.0f, 0.0f, 0.0f};
    float b[3] = {-1.0f, 0.0f, 0.0f};
    int dims = 3; // Passing dims manually for now
    float dot = dot_product(a, b, dims);
    float mag_a = magnitude(a, dims);
    float cos   = cosine_similarity(a, b, dims);

    std::cout << "Magnitude: " << mag_a << std::endl;
    std::cout << "Cosine similarity: " << cos << std::endl;
    std::cout << "Dot product: " << dot << std::endl;
    normalize(a, dims);
    std::cout << "Normalized vector A: (";
    for (int i = 0; i < dims; i++) {
        std::cout << a[i];
        if (i < dims-1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
    
    int8_t output[3];
    quantize(a, output, dims);
    
    std::cout << "After Quantization: (";
    for(int j = 0; j < dims ; j++) {
        std::cout << (int)output[j];
        if(j < dims-1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;

    float output2[3];
    dequantize(output, output2, dims);

    std::cout << "After De-Quantization: (";
    for(int j = 0; j < dims ; j++) {
        std::cout << output2[j];
        if(j < dims-1) std::cout << ", ";
    }

    std::cout << ")" << std::endl;
    return 0;
}