#include <sys/asm.h>

#if defined(__riscv_vector) && __riscv_xlen == 64 && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)
/* int strncasecmp(const char *a0, const char *a1, size_t a2)
   Case-insensitive compare of at most a2 bytes.  vle8ff keeps loads from
   crossing a page boundary; vl is also capped by the remaining count so we
   never read past the n-th byte.  The case folding covers the ASCII letters
   A-Z only (the C/POSIX locale).  */
ENTRY(strncasecmp)
  beqz a2, .Lzero
  li t0, 65            /* 'A' */
  li t1, 26
  li t2, 32
.Lloop:
  vsetvli zero, a2, e8, m1, ta, mu
  vle8ff.v v8, (a0)
  vle8ff.v v16, (a1)
  csrr a3, vl
  /* fold left */
  vsub.vx v24, v8, t0
  vmsltu.vx v0, v24, t1
  vadd.vx v8, v8, t2, v0.t
  /* fold right */
  vsub.vx v24, v16, t0
  vmsltu.vx v0, v24, t1
  vadd.vx v16, v16, t2, v0.t
  /* NUL in left or mismatch */
  vmseq.vi v1, v8, 0
  vmsne.vv v2, v8, v16
  vmor.mm v1, v1, v2
  sub a2, a2, a3
  vfirst.m a4, v1
  bgez a4, .Lend
  add a0, a0, a3
  add a1, a1, a3
  bnez a2, .Lloop
  /* exhausted n with no NUL/diff: all compared bytes were equal -> 0 */
  li a0, 0
  ret
.Lend:
  add a0, a0, a4
  add a1, a1, a4
  lbu a4, 0(a0)
  lbu a5, 0(a1)
  addi a6, a4, -65
  sltiu a6, a6, 26
  slli a6, a6, 5
  add a4, a4, a6
  addi a6, a5, -65
  sltiu a6, a6, 26
  slli a6, a6, 5
  add a5, a5, a6
  sub a0, a4, a5
  ret
.Lzero:
  li a0, 0
  ret
END(strncasecmp)
#endif
