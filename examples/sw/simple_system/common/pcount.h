#ifndef PCOUNT_H_
#define PCOUNT_H_

#include "simple_system_common.h"

// This is based on the PCOUNT_READ macro in simple_system_common.h
static inline unsigned int pcount_get(void) {
  unsigned int count;
  asm volatile("csrr %0, mcycle" : "=r"(count));
  return count;
}

// This is based on the logic in pcount_enable in simple_system_common.h
// pcount_inhibit(1) should inhibit (disable) counters.
static inline void pcount_inhibit(int inhibit) {
    pcount_enable(!inhibit);
}

#endif // PCOUNT_H_ 