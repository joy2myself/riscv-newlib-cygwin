/* strnlen - RVV 1.0 candidate v2: vector-only, no vle8ff.
 *
 *   size_t strnlen (const char *s, size_t n)
 *
 * Algorithm
 *   One e8/m8 loop whose vl is min(n_left, bytes-to-page).  vsetvli with
 *   rd != x0 clamps to VLMAX and returns the real vl in t2, so the pointers
 *   advance by exactly the number of bytes actually touched -- the routine
 *   provably never reads byte n or a byte past the page, not even in the
 *   tail of a vector load.  When the loop runs out of n the answer is n.
 *
 *   The cap-by-n also means a call with n == strlen(s) never reads the
 *   terminator: it returns n because n bytes were all non-zero.
 *
 * Page safety: vl <= distance-to-page for every plain vle8.v; no vle8ff,
 * no backwards read.  RV32/RV64: XLEN independent.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strnlen)
  beqz     a1, .Lzero
  mv       a2, a0                 /* start                              */
  mv       a3, a1                 /* bytes left                         */
  mv       a4, a1                 /* requested n                        */
  li       t6, 4095
  li       t5, 4096

.Lloop:
  and      t0, a0, t6
  sub      t0, t5, t0             /* bytes to page end                  */
  bgeu     a3, t0, 1f
  mv       t0, a3
1:
  vsetvli  t2, t0, e8, m8, ta, ma /* t2 = min(t0, VLMAX)                */
  vle8.v   v8, (a0)
  vmseq.vi v0, v8, 0
  vfirst.m t0, v0
  add      a0, a0, t2
  sub      a3, a3, t2
  bltz     t0, .Lnext

  sub      a0, a0, a2
  sub      a0, a0, t2
  add      a0, a0, t0
  ret

.Lnext:
  bnez     a3, .Lloop
  mv       a0, a4
  ret
.Lzero:
  li       a0, 0
  ret
END(strnlen)

#endif
