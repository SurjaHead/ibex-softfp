#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "activation_functions.h"

// Generate random float between min and max
float random_float(float min, float max) {
    float scale = rand() / (float) RAND_MAX;
    return min + scale * (max - min);
}

void test_activation(const char* name, float (*func)(float), float x) {
    float result = func(x);
    printf("%s(%f) = %f\n", name, x, result);
}

void test_activation_with_param(const char* name, float (*func)(float, float), float x, float param) {
    float result = func(x, param);
    printf("%s(%f, %f) = %f\n", name, x, param, result);
}

int main() {
    srand(time(NULL));
    
    // Test with random inputs
    for (int i = 0; i < 5; i++) {
        float x = random_float(-10.0f, 10.0f);
        printf("\nTest %d with x = %f:\n", i + 1, x);
        
        test_activation("relu", relu, x);
        test_activation_with_param("leaky_relu", leaky_relu, x, 0.01f);
        test_activation("sigmoid", sigmoid, x);
        test_activation("tanh", tanh_activation, x);
        test_activation("gelu", gelu, x);
        test_activation("mish", mish, x);
        test_activation("silu", silu, x);
    }
    
    return 0;
} 