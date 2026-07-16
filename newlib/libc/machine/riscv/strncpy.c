#if defined(__OPTIMIZE_SIZE__) || defined(PREFER_SIZE_OVER_SPEED) || !defined(__riscv_vector) || __riscv_xlen != 64
# include "../../string/strncpy.c"
#else
/* strncpy defined in strncpy-asm.S */
#endif
