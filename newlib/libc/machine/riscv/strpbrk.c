#if defined(__OPTIMIZE_SIZE__) || defined(PREFER_SIZE_OVER_SPEED) || !defined(__riscv_vector) || __riscv_xlen != 64
# include "../../string/strpbrk.c"
#else
/* strpbrk defined in strpbrk-asm.S */
#endif
