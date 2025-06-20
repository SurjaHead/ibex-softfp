#include <stddef.h>
#include "simple_system_common.h"

void *memcpy(void *dest, const void *src, size_t n) {
  char *d = (char *)dest;
  const char *s = (const char *)src;
  for (size_t i = 0; i < n; i++) {
    d[i] = s[i];
  }
  return dest;
}

/* Very small abort implementation: stop simulation. */
void abort(void) {
  puts("abort called\n");
  sim_halt();
  while (1) {}
}

void __assert_func(const char *file, int line, const char *func, const char *expr) {
  puts("assert failed\n");
  sim_halt();
  while (1) {}
} 