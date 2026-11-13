#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__riscv_e) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)
/* char *strstr(const char *a0, const char *a1)
   Locate substring a1 in a0.  A vectorized scan (vle8ff, page-boundary safe)
   finds candidate positions of needle[0]; each candidate is verified with a
   byte loop that stops at the needle's NUL (match) or at a mismatch/the
   haystack NUL.  Both phases stop at a NUL, so no load ever crosses past the
   end of either string into an unmapped page.  */
ENTRY(strstr)
  lbu t0, 0(a1)            /* c0 = needle[0] */
  beqz t0, .Lret_h         /* empty needle -> return haystack */
.Lloop:
  vsetvli t1, zero, e8, m1, ta, ma
  vle8ff.v v8, (a0)
  csrr t1, vl
  vmseq.vx v1, v8, t0      /* lanes == c0 */
  vmseq.vi v2, v8, 0       /* NUL lanes */
  vfirst.m t2, v1
  vfirst.m t3, v2
  bltz t2, .Lno_c0         /* no c0 in this chunk */
  bltz t3, .Lcandidate     /* c0 present, no NUL -> candidate */
  bltu t2, t3, .Lcandidate /* c0 occurs before NUL */
  /* NUL at or before c0 -> end of haystack reached */
.Lnotfound:
  li a0, 0
  ret
.Lno_c0:
  bgez t3, .Lnotfound      /* hit NUL with no c0 -> not found */
  add a0, a0, t1
  j .Lloop
.Lcandidate:
  add a0, a0, t2           /* a0 -> candidate (h[*]==c0) */
  mv t4, a0                /* haystack cursor */
  mv t5, a1                /* needle cursor */
.Lverify:
  lbu t6, 0(t5)
  beqz t6, .Lmatch         /* needle exhausted -> match at a0 */
  lbu a2, 0(t4)
  bne a2, t6, .Ladvance    /* mismatch (a2 may be NUL) */
  addi t4, t4, 1
  addi t5, t5, 1
  j .Lverify
.Ladvance:
  addi a0, a0, 1           /* slide one byte and rescan */
  j .Lloop
.Lmatch:
.Lret_h:
  ret
END(strstr)
#endif
