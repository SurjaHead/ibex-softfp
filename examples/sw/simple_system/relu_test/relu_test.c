// Copyright lowRISC contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "simple_system_common.h"
#include "pcount.h"
#include <math.h>
#include "activations_softfp.h"

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
  volatile unsigned int cycles_before, cycles_after, cycles_taken;
  volatile float input_val = -4.0f;
  volatile float result_float;

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
  // result_float = mish(input_val);

  // --- Measure a single sigmoid call ---
  enable_cycle_counter();
  pcount_reset();

  cycles_before = pcount_get();
  result_float = mish(input_val);
  cycles_after = pcount_get();

  cycles_taken = cycles_after - cycles_before;

  puts("Cycles taken (hex): 0x");
  puthex(cycles_taken);
  putchar('\n');

  puts("mish result (float bits hex): 0x");
  union { float f; uint32_t u32; } conv; conv.f = result_float;
  puthex(conv.u32);
  putchar('\n');
  /* Move the values into registers a0/a1 *after* printing so that
   * subsequent library calls cannot overwrite them. */
  asm volatile ("mv a0, %0" : : "r" (cycles_taken));
  asm volatile ("mv a1, %0" : : "r" (conv.u32));

  /* End simulation cleanly */
  sim_halt();
}
