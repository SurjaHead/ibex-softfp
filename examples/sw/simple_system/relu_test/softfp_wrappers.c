#include <stdint.h>
#include "softfp.h"

static inline uint32_t float_to_bits(float f) {
  union { float f; uint32_t u; } tmp = { f };
  return tmp.u;
}

static inline float bits_to_float(uint32_t u) {
  union { uint32_t u; float f; } tmp = { u };
  return tmp.f;
}

/* RISC-V GCC soft-float helper wrappers using SoftFP implementation (32-bit). */

float __addsf3(float a, float b) {
  uint32_t flags = 0;
  uint32_t res = add_sf32(float_to_bits(a), float_to_bits(b), RM_RNE, &flags);
  return bits_to_float(res);
}

float __subsf3(float a, float b) {
  uint32_t flags = 0;
  uint32_t res = sub_sf32(float_to_bits(a), float_to_bits(b), RM_RNE, &flags);
  return bits_to_float(res);
}

float __mulsf3(float a, float b) {
  uint32_t flags = 0;
  uint32_t res = mul_sf32(float_to_bits(a), float_to_bits(b), RM_RNE, &flags);
  return bits_to_float(res);
}

float __divsf3(float a, float b) {
  uint32_t flags = 0;
  uint32_t res = div_sf32(float_to_bits(a), float_to_bits(b), RM_RNE, &flags);
  return bits_to_float(res);
}

int __fixsfsi(float a) {
  uint32_t flags = 0;
  return cvt_sf32_i32(float_to_bits(a), RM_RTZ, &flags);
}

float __floatsisf(int a) {
  uint32_t flags = 0;
  uint32_t res = cvt_i32_sf32(a, RM_RNE, &flags);
  return bits_to_float(res);
} 