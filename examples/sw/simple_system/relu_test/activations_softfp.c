#include <math.h>
#include "activations_softfp.h"

/*
 * All activations below rely only on expf/tanhf/logf and primitive
 * arithmetic.  Because the build links in softfp_wrappers.c, every floating
 *-point operation ends up going through the SoftFP library, so they run on
 * integer-only Ibex just like the previous sigmoid test.
 */

static inline float soft_exp(float x)   { return expf(x); }
static inline float soft_tanh(float x)  { return tanhf(x); }
static inline float soft_log(float x)   { return logf(x); }

/* ---------------- standard activations ---------------- */

/* 1. ReLU */
float relu(float x) { return (x > 0.0f) ? x : 0.0f; }

/* 2. Leaky ReLU */
float leaky_relu(float x, float negative_slope) {
    return (x >= 0.0f) ? x : negative_slope * x;
}

/* 3. ELU */
float elu(float x, float alpha) {
    return (x >= 0.0f) ? x : alpha * (soft_exp(x) - 1.0f);
}

/* 4. SiLU / Swish */
float silu(float x) {
    float sig = 1.0f / (1.0f + soft_exp(-x));
    return x * sig;
}

/* 5. Sigmoid */
float sigmoid(float x) {
    return 1.0f / (1.0f + soft_exp(-x));
}

/* 6. tanh */
float tanh_act(float x) { return soft_tanh(x); }

/* 7. GELU (Gaussian Error Linear Unit) */
float gelu(float x) {
    const float k = 0.044715f;          /* coefficient in original paper */
    const float c = 0.7978845608f;      /* sqrt(2 / pi) */
    float x3 = x * x * x;
    float t  = x + k * x3;
    return 0.5f * x * (1.0f + soft_tanh(c * t));
}

/* 8. Mish */
float mish(float x) {
    float sp = soft_log(1.0f + soft_exp(x));   /* softplus(x) */
    return x * soft_tanh(sp);
}

/* 9. Softmax */
void softmax(const float* input, float* output, int size) {
    /* Find the maximum value for numerical stability */
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    /* Compute sum of exponentials (shifted by max_val for stability) */
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        float exp_val = soft_exp(input[i] - max_val);
        output[i] = exp_val;
        sum_exp += exp_val;
    }
    
    /* Normalize by the sum */
    for (int i = 0; i < size; i++) {
        output[i] = output[i] / sum_exp;
    }
}

/* ---------------- primitive operations for benchmarking ---------------- */

float op_exp(float x)           { return soft_exp(x); }
float op_log(float x)           { return soft_log(x); }
float op_pow(float x, float y)  { return powf(x, y); }
float op_div(float x, float y)  { return x / y; }
float op_mul(float x, float y)  { return x * y; }
float op_add(float x, float y)  { return x + y; } 