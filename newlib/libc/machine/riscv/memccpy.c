#if defined(__OPTIMIZE_SIZE__) || defined(PREFER_SIZE_OVER_SPEED) || !defined(__riscv_vector) || defined(__riscv_e)
# include "../../string/memccpy.c"
#else
/* memccpy defined in memccpy-asm.S */
#endif
