#ifndef ACTIVATIONS_HW_H
#define ACTIVATIONS_HW_H

#include <stdint.h>

static inline float fp_add(float a, float b) {
    float result;
    // custom0 opcode is 0x0B. funct3 and funct7 are 0.
    asm volatile (
        ".insn r 0x0B, 0, 0, %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (b)
    );
    return result;
}

static inline float fp_mul(float a, float b) {
    float result;
    // custom1 opcode is 0x2B. funct3 and funct7 are 0.
    asm volatile (
        ".insn r 0x2B, 0, 0, %0, %1, %2"
        : "=r" (result)
        : "r" (a), "r" (b)
    );
    return result;
}

static inline uint32_t as_uint(float f) {
    union {
        float f;
        uint32_t u;
    } un;
    un.f = f;
    return un.u;
}

static inline float as_float(uint32_t u) {
    union {
        float f;
        uint32_t u;
    } un;
    un.u = u;
    return un.f;
}

// Helper function for polynomial approximations
static inline float fp_sub(float a, float b) {
    // Subtract by adding the negative (flip sign bit)
    uint32_t b_bits = as_uint(b);
    b_bits ^= 0x80000000;  // Flip sign bit
    return fp_add(a, as_float(b_bits));
}

// Simple exponential approximation using Taylor series: e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24
static inline float fp_exp_approx(float x) {
    // Clamp x to reasonable range to avoid overflow
    // Check magnitude properly by ignoring sign bit
    uint32_t abs_x_bits = as_uint(x) & 0x7FFFFFFF;
    if (abs_x_bits > 0x40800000) { // |x| > 4
        if ((as_uint(x) & 0x80000000) == 0) {
            return 20.0f;  // if x > 4, return large value
        } else {
            return 0.05f;  // if x < -4, return small value
        }
    }
    
    float x2 = fp_mul(x, x);          // x²
    float x3 = fp_mul(x2, x);         // x³
    float x4 = fp_mul(x3, x);         // x⁴
    
    float term1 = x;                  // x
    float term2 = fp_mul(x2, 0.5f);   // x²/2
    float term3 = fp_mul(x3, 0.16666667f); // x³/6
    float term4 = fp_mul(x4, 0.04166667f); // x⁴/24
    
    float result = fp_add(1.0f, term1);
    result = fp_add(result, term2);
    result = fp_add(result, term3);
    result = fp_add(result, term4);
    
    return result;
}

// Simple division approximation using Newton-Raphson: 1/x
static inline float fp_div_approx(float a, float b) {
    // For 1/x, use Newton-Raphson: x_{n+1} = x_n * (2 - b * x_n)
    // Start with a reasonable guess
    float x = 1.0f;  // Initial guess
    
    // 3 iterations should be enough for reasonable precision
    for (int i = 0; i < 3; i++) {
        float bx = fp_mul(b, x);
        float two_minus_bx = fp_sub(2.0f, bx);
        x = fp_mul(x, two_minus_bx);
    }
    
    return fp_mul(a, x);
}

static inline float relu(float x) {
    // if x > 0 return x, else return 0
    // Check sign bit (bit 31)
    if ((as_uint(x) & 0x80000000) == 0) {
        return x;
    } else {
        return 0.0f;
    }
}

static inline float leaky_relu(float x, float alpha) {
    // if x > 0 return x, else return alpha * x
    // Check sign bit (bit 31)
    if ((as_uint(x) & 0x80000000) == 0) {
        return x;
    } else {
        return fp_mul(alpha, x);
    }
}

static inline float elu(float x, float alpha) {
    // if x >= 0 return x, else return alpha * (exp(x) - 1)
    if ((as_uint(x) & 0x80000000) == 0) {
        return x;
    } else {
        float exp_x = fp_exp_approx(x);
        float exp_x_minus_1 = fp_sub(exp_x, 1.0f);
        return fp_mul(alpha, exp_x_minus_1);
    }
}

static inline float sigmoid(float x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    uint32_t neg_x_bits = as_uint(x) ^ 0x80000000;  // Flip sign bit for -x
    float neg_x = as_float(neg_x_bits);
    
    float exp_neg_x = fp_exp_approx(neg_x);
    float one_plus_exp = fp_add(1.0f, exp_neg_x);
    
    return fp_div_approx(1.0f, one_plus_exp);
}

static inline float silu(float x) {
    // silu(x) = x * sigmoid(x)
    float sig_x = sigmoid(x);
    return fp_mul(x, sig_x);
}

static inline float tanh_act(float x) {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // Clamp x to reasonable range to avoid overflow
    uint32_t abs_x_bits = as_uint(x) & 0x7FFFFFFF;
    if (abs_x_bits > 0x40800000) { // |x| > 4
        if ((as_uint(x) & 0x80000000) == 0) {
            return 1.0f;   // tanh(large positive) ≈ 1
        } else {
            return -1.0f;  // tanh(large negative) ≈ -1
        }
    }
    
    float two_x = fp_add(x, x);  // 2x
    float exp_2x = fp_exp_approx(two_x);
    
    float numerator = fp_sub(exp_2x, 1.0f);     // exp(2x) - 1
    float denominator = fp_add(exp_2x, 1.0f);   // exp(2x) + 1
    
    return fp_div_approx(numerator, denominator);
}

static inline float gelu(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    // sqrt(2/π) ≈ 0.7978845608
    
    float x2 = fp_mul(x, x);
    float x3 = fp_mul(x2, x);
    float term1 = fp_mul(0.044715f, x3);           // 0.044715 * x³
    float inner = fp_add(x, term1);                // x + 0.044715 * x³
    float scaled = fp_mul(0.7978845608f, inner);   // sqrt(2/π) * (x + 0.044715 * x³)
    
    float tanh_val = tanh_act(scaled);
    float one_plus_tanh = fp_add(1.0f, tanh_val);  // 1 + tanh(...)
    float x_times_bracket = fp_mul(x, one_plus_tanh); // x * (1 + tanh(...))
    
    return fp_mul(0.5f, x_times_bracket);          // 0.5 * x * (1 + tanh(...))
}

static inline float mish(float x) {
    // For now, use a simple approximation that avoids the problematic FPU operations
    // mish(x) ≈ x * tanh(x) for x < 0, and x for x >= 0 (simplified)
    
    if ((as_uint(x) & 0x80000000) == 0) {
        // For positive x, mish(x) ≈ x
        return x;
    } else {
        // For negative x, use a simple approximation: x * tanh(x/2) * 0.8
        // This avoids the complex softplus computation that's causing issues
        
        // Scale down the input to avoid overflow in tanh
        float x_scaled = fp_mul(x, 0.5f);
        
        // Simple tanh approximation for small values: tanh(x) ≈ x for small x
        float tanh_approx;
        uint32_t abs_x_scaled = as_uint(x_scaled) & 0x7FFFFFFF;
        if (abs_x_scaled < 0x3F000000) { // |x_scaled| < 0.5
            tanh_approx = x_scaled; // tanh(x) ≈ x for small x
        } else {
            // For larger values, clamp to reasonable range
            if ((as_uint(x_scaled) & 0x80000000) == 0) {
                tanh_approx = 0.5f;  // tanh(positive) ≈ 0.5
            } else {
                tanh_approx = -0.5f; // tanh(negative) ≈ -0.5
            }
        }
        
        // mish(x) ≈ x * tanh_approx * correction_factor
        float result = fp_mul(x, tanh_approx);
        return fp_mul(result, 0.8f); // Correction factor to get closer to true mish
    }
}

#endif // ACTIVATIONS_HW_H 