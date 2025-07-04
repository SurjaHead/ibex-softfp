// Copyright lowRISC contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "simple_system_common.h"
#include "pcount.h"
#include <math.h>
#include "activations_hw.h"

/* Provide minimal errno support for newlib's libm (needed by expf). */
int *__errno(void) {
  static int _errno = 0;
  return &_errno;
}

/* Sigmoid implemented with single-precision floating point using expf from
 * newlib.  Compiled in soft-float mode so all FP operations are emulated in
 * software (either by libgcc helpers or the optional softfp library).
 */
// static float sigmoid_float(float x) {
//   return 1.0f / (1.0f + expf(-x));
// }

// /* Utility: enable the mcycle counter (clear CY bit of mcountinhibit) */
static inline void enable_cycle_counter(void) {
  asm volatile ("csrci mcountinhibit, 0x1");   /* bit 0 = CY */
}

int main(void) {
  // We need these to be volatile so the compiler doesn't optimize them away.
  volatile unsigned int cycles_before, cycles_after, cycles_taken, average_cycles;
  volatile float input_val = -1.0f;
  volatile float result_float;
  const int num_iterations = 1000;

  /* --------------------------------------------------
   * First, test basic FPU operations to isolate issues
   * -------------------------------------------------- */
  
  puts("Testing basic FPU operations first...\n");
  
  // Test 1: Simple addition
  volatile float a = 1.0f;
  volatile float b = 2.0f;
  volatile float add_result = fp_add(a, b);
  puts("fp_add(1.0, 2.0) = 0x");
  union { float f; uint32_t u32; } conv1; conv1.f = add_result;
  puthex(conv1.u32);
  puts(" (should be 0x40400000 = 3.0)\n");
  
  // Test 2: Simple multiplication
  volatile float c = 2.0f;
  volatile float d = 3.0f;
  volatile float mul_result = fp_mul(c, d);
  puts("fp_mul(2.0, 3.0) = 0x");
  union { float f; uint32_t u32; } conv2; conv2.f = mul_result;
  puthex(conv2.u32);
  puts(" (should be 0x40C00000 = 6.0)\n");
  
  // Test 3: Negative multiplication
  volatile float e = -1.0f;
  volatile float f_val = 0.5f;
  volatile float neg_mul_result = fp_mul(e, f_val);
  puts("fp_mul(-1.0, 0.5) = 0x");
  union { float f; uint32_t u32; } conv3; conv3.f = neg_mul_result;
  puthex(conv3.u32);
  puts(" (should be 0xBF000000 = -0.5)\n");
  
  // Test 4: Test subtraction directly
  puts("Testing fp_sub function...\n");
  volatile float sub_a = 3.0f;
  volatile float sub_b = 1.0f;
  volatile float sub_result = fp_sub(sub_a, sub_b);
  puts("fp_sub(3.0, 1.0) = 0x");
  union { float f; uint32_t u32; } conv_sub; conv_sub.f = sub_result;
  puthex(conv_sub.u32);
  puts(" (should be 0x40000000 = 2.0)\n");
  
  // Test 5: Test what happens when we manually implement subtraction
  volatile float manual_sub_b_neg = as_float(as_uint(sub_b) ^ 0x80000000);  // Flip sign bit
  volatile float manual_sub_result = fp_add(sub_a, manual_sub_b_neg);
  puts("manual sub: fp_add(3.0, -1.0) = 0x");
  union { float f; uint32_t u32; } conv_manual; conv_manual.f = manual_sub_result;
  puthex(conv_manual.u32);
  puts(" (should be 0x40000000 = 2.0)\n");
  
  /* --------------------------------------------------
   * Now test mish step by step to debug the issue
   * -------------------------------------------------- */
  
  puts("Testing mish(-1.0) step by step...\n");
  
  // Step 1: Test exp approximation
  volatile float test_exp = fp_exp_approx(-1.0f);
  puts("fp_exp_approx(-1.0) = 0x");
  union { float f; uint32_t u32; } conv4; conv4.f = test_exp;
  puthex(conv4.u32);
  puts(" (should be ~0x3EBC5AB2 = ~0.368)\n");
  
  // Step 2: Test tanh approximation  
  puts("Testing tanh_act(0.5) step by step...\n");
  volatile float tanh_input = 0.5f;
  
  // Manual tanh computation
  volatile float two_tanh_x = fp_add(tanh_input, tanh_input);  // 2x
  puts("two_x = 0x");
  union { float f; uint32_t u32; } conv_two_x; conv_two_x.f = two_tanh_x;
  puthex(conv_two_x.u32);
  puts(" (should be 0x3F800000 = 1.0)\n");
  
  volatile float exp_2x = fp_exp_approx(two_tanh_x);
  puts("exp(2x) = 0x");
  union { float f; uint32_t u32; } conv_exp_2x; conv_exp_2x.f = exp_2x;
  puthex(conv_exp_2x.u32);
  puts(" (should be ~0x402DF854 = ~2.718)\n");
  
  // Create separate copies to avoid aliasing
  volatile float exp_2x_copy1 = exp_2x;
  volatile float exp_2x_copy2 = exp_2x;
  volatile float one_val = 1.0f;
  
  puts("exp_2x_copy1 bits = 0x");
  union { float f; uint32_t u32; } conv_copy1; conv_copy1.f = exp_2x_copy1;
  puthex(conv_copy1.u32);
  puts(" exp_2x_copy2 bits = 0x");
  union { float f; uint32_t u32; } conv_copy2; conv_copy2.f = exp_2x_copy2;
  puthex(conv_copy2.u32);
  puts(" one_val bits = 0x");
  union { float f; uint32_t u32; } conv_one; conv_one.f = one_val;
  puthex(conv_one.u32);
  putchar('\n');
  
  volatile float numerator = fp_sub(exp_2x_copy1, one_val);     // exp(2x) - 1
  puts("numerator = exp(2x) - 1 = 0x");
  union { float f; uint32_t u32; } conv_num; conv_num.f = numerator;
  puthex(conv_num.u32);
  putchar('\n');
  
  volatile float denominator = fp_add(exp_2x_copy2, one_val);   // exp(2x) + 1
  puts("denominator = exp(2x) + 1 = 0x");
  union { float f; uint32_t u32; } conv_den; conv_den.f = denominator;
  puthex(conv_den.u32);
  putchar('\n');
  
  volatile float tanh_result = fp_div_approx(numerator, denominator);
  puts("tanh result = num/den = 0x");
  union { float f; uint32_t u32; } conv_tanh; conv_tanh.f = tanh_result;
  puthex(conv_tanh.u32);
  puts(" (should be ~0x3F0B5B80 = ~0.462)\n");
  
  // Also test the actual tanh_act function
  volatile float test_tanh = tanh_act(0.5f);
  puts("tanh_act(0.5) = 0x");
  union { float f; uint32_t u32; } conv5; conv5.f = test_tanh;
  puthex(conv5.u32);
  puts(" (should be ~0x3F0B5B80 = ~0.462)\n");
  
  // Step 3: Test a simple mish computation manually
  // For mish(-1.0), let's manually trace through the logic
  puts("Manual mish(-1.0) computation...\n");
  volatile float x = -1.0f;
  
  // Check if |x| > 4
  volatile uint32_t x_bits = as_uint(x);
  volatile uint32_t abs_x_bits = x_bits & 0x7FFFFFFF;
  puts("abs(x) bits = 0x");
  puthex(abs_x_bits);
  puts(" (checking if > 0x40800000)\n");
  
  if (abs_x_bits > 0x40800000) {
    puts("Taking large |x| path\n");
  } else {
    puts("Taking moderate |x| path\n");
    
    volatile float exp_x = fp_exp_approx(x);
    puts("exp_x = 0x");
    union { float f; uint32_t u32; } conv6; conv6.f = exp_x;
    puthex(conv6.u32);
    putchar('\n');
    
    // Since x < -2, we should take the softplus_x = exp_x path
    if (x_bits > 0xC0000000) { // x < -2
      puts("Taking x < -2 path: softplus_x = exp_x\n");
      volatile float softplus_x = exp_x;
      puts("softplus_x = 0x");
      union { float f; uint32_t u32; } conv7; conv7.f = softplus_x;
      puthex(conv7.u32);
      putchar('\n');
      
      volatile float tanh_softplus = tanh_act(softplus_x);
      puts("tanh(softplus_x) = 0x");
      union { float f; uint32_t u32; } conv8; conv8.f = tanh_softplus;
      puthex(conv8.u32);
      putchar('\n');
      
      volatile float final_result = fp_mul(x, tanh_softplus);
      puts("final mish result = 0x");
      union { float f; uint32_t u32; } conv9; conv9.f = final_result;
      puthex(conv9.u32);
      putchar('\n');
    }
  }

  /* --------------------------------------------------
   * Choose ONE activation to benchmark by uncommenting
   * exactly one of the lines below.
   * -------------------------------------------------- */

  // result_float = relu(input_val);
  // result_float = leaky_relu(input_val, 0.01f);
  // result_float = elu(input_val, 1.0f);
  // result_float = silu(input_val);
  // result_float = sigmoid(input_val);
  // result_float = tanh_act(input_val);
  // result_float = gelu(input_val);
  result_float = mish(input_val);

  // --- Measure 1000 iterations of mish activation ---
  enable_cycle_counter();
  pcount_reset();

  cycles_before = pcount_get();
  
  // Run the activation function 1000 times
  for (int i = 0; i < num_iterations; i++) {
    result_float = mish(input_val);
  }
  
  cycles_after = pcount_get();

  cycles_taken = cycles_after - cycles_before;
  average_cycles = cycles_taken / num_iterations;

  puts("Total cycles for 1000 iterations (hex): 0x");
  puthex(cycles_taken);
  putchar('\n');

  puts("Average cycles per activation (hex): 0x");
  puthex(average_cycles);
  putchar('\n');

  puts("Average cycles per activation (decimal): ");
  // Simple decimal printing (since we don't have printf)
  unsigned int temp = average_cycles;
  char digits[10];
  int digit_count = 0;
  
  if (temp == 0) {
    putchar('0');
  } else {
    while (temp > 0) {
      digits[digit_count++] = '0' + (temp % 10);
      temp /= 10;
    }
    // Print digits in reverse order
    for (int i = digit_count - 1; i >= 0; i--) {
      putchar(digits[i]);
    }
  }
  putchar('\n');

  puts("Final result (float bits hex): 0x");
  union { float f; uint32_t u32; } conv; conv.f = result_float;
  puthex(conv.u32);
  putchar('\n');
  
  /* Move the values into registers a0/a1 *after* printing so that
   * subsequent library calls cannot overwrite them. */
  asm volatile ("mv a0, %0" : : "r" (average_cycles));
  asm volatile ("mv a1, %0" : : "r" (conv.u32));

  /* End simulation cleanly */
  sim_halt();
}
