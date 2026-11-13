#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)
ENTRY(strchrnul)
.Lloop:
  vsetvli t0, zero, e8, m1, ta, ma
  vle8ff.v v0, (a0)
  vmseq.vx v8, v0, a1
  vmseq.vi v9, v0, 0
  vfirst.m a2, v8
  vfirst.m a3, v9
  bltz a2, .Lcheck_nul
  bltz a3, .Lfound_char
  bltu a2, a3, .Lfound_char
.Lfound_nul:
  add a0, a0, a3
  ret
.Lfound_char:
  add a0, a0, a2
  ret
.Lcheck_nul:
  bgez a3, .Lfound_nul
  csrr a4, vl
  add a0, a0, a4
  j .Lloop
END(strchrnul)
#endif
