#include <sys/asm.h>

#if defined(__riscv_vector) && __riscv_xlen == 64 && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)
/* int strcasecmp(const char *a0, const char *a1)
   Case-insensitive compare.  Scans with vle8ff so a vector load is never
   forced across a page boundary into an unmapped page: element 0 is always
   loaded, and a fault on any later element merely truncates vl.  The case
   folding covers the ASCII letters A-Z only (the C/POSIX locale).  */
ENTRY(strcasecmp)
  li t0, 65            /* 'A' */
  li t1, 26            /* range of upper-case letters */
  li t2, 32            /* 'a' - 'A' */
.Lloop:
  vsetvli a2, zero, e8, m1, ta, mu
  vle8ff.v v8, (a0)
  vle8ff.v v16, (a1)
  csrr a2, vl
  /* fold left: if (b-'A') <u 26 then b += 32 */
  vsub.vx v24, v8, t0
  vmsltu.vx v0, v24, t1
  vadd.vx v8, v8, t2, v0.t
  /* fold right */
  vsub.vx v24, v16, t0
  vmsltu.vx v0, v24, t1
  vadd.vx v16, v16, t2, v0.t
  /* compare folded bytes */
  vmseq.vi v1, v8, 0       /* NUL in left */
  vmsne.vv v2, v8, v16     /* mismatch */
  vfirst.m a3, v1
  vfirst.m a4, v2
  bgez a3, .Lnul
  bgez a4, .Ldiff
  add a0, a0, a2
  add a1, a1, a2
  j .Lloop
.Lnul:                   /* a3 >= 0 (NUL position) */
  bltz a4, .Lend
  bge a4, a3, .Lend    /* NUL at or before mismatch */
  mv a3, a4
  j .Lend
.Ldiff:                  /* a3 < 0, a4 >= 0 */
  mv a3, a4
.Lend:
  add a0, a0, a3
  add a1, a1, a3
  lbu a4, 0(a0)
  lbu a5, 0(a1)
  /* tolower(a4) */
  addi a6, a4, -65
  sltiu a6, a6, 26
  slli a6, a6, 5
  add a4, a4, a6
  /* tolower(a5) */
  addi a6, a5, -65
  sltiu a6, a6, 26
  slli a6, a6, 5
  add a5, a5, a6
  sub a0, a4, a5
  ret
END(strcasecmp)
#endif
