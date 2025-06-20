#ifndef BYTESWAP_RISCV_H
#define BYTESWAP_RISCV_H

// Use GCC/Clang built-in functions for byte swapping for portability.
#define bswap_16 __builtin_bswap16
#define bswap_32 __builtin_bswap32
#define bswap_64 __builtin_bswap64

#endif // BYTESWAP_RISCV_H 