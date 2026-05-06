#include <iostream>

float dot_product(float a[3], float b[3]) {
    float result = 0.0f;
    result += a[0] * b[0];
    result += a[1] * b[1];
    result += a[2] * b[2];
    return result;
}

int main() {
    float a[3] = {1.0f, 2.0f, 3.0f};
    float b[3] = {4.0f, 5.0f, 6.0f};

    float dot = dot_product(a, b);
    std::cout << "Dot product: " << dot << std::endl;
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

    return 0;
}