/* strncasecmp - RVV 1.0 candidate v3: one e8/m1 VLMAX probe chunk then the
 * e8/m8 loop.  No vle8ff.
 *
 *   int strncasecmp (const char *a, const char *b, size_t n)
 *
 * Same ASCII-only fold and chunk body as strcasecmp-v2, with a smaller
 * (LMUL=m1) first chunk and vl additionally capped by the remaining count
 * so no byte at or after a+n / b+n is touched.
 * Exhausting n without a NUL or difference returns 0.
 *
 * Page safety: vl = min(n_left, distance-to-a's-page, distance-to-b's-page);
 * vsetvli returns the clamped vl in t2 and both pointers advance by it.
 * No vle8ff, no backwards read, no read past n.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

.macro FOLD_SCALAR reg, tmp
  addi  \tmp, \reg, -65
  sltiu \tmp, \tmp, 26
  slli  \tmp, \tmp, 5
  add   \reg, \reg, \tmp
.endm

ENTRY(strncasecmp)
  li       a5, 65                 /* 'A' */
  li       a6, 26                 /* alphabet size */
  li       a7, 32                 /* case bit */
  beqz     a2, .Lzero
  mv       a3, a2                 /* bytes left                         */
  li       t6, 4095
  li       t5, 4096

  and      t0, a0, t6
  sub      t0, t5, t0
  and      t1, a1, t6
  sub      t1, t5, t1
  bgeu     t1, t0, 1f
  mv       t0, t1
1:
  bgeu     a3, t0, 2f
  mv       t0, a3
2:
  vsetvli  t2, t0, e8, m1, ta, mu
  vle8.v   v8, (a0)
  vle8.v   v16, (a1)
  vsub.vx  v24, v8, a5
  vmsltu.vx v0, v24, a6
  vadd.vx  v8, v8, a7, v0.t
  vsub.vx  v24, v16, a5
  vmsltu.vx v0, v24, a6
  vadd.vx  v16, v16, a7, v0.t
  vmsne.vv v0, v8, v16
  vmseq.vi v1, v8, 0
  vmor.mm  v0, v0, v1
  vfirst.m t0, v0
  add      a0, a0, t2
  add      a1, a1, t2
  sub      a3, a3, t2
  bgez     t0, .Lhit
  beqz     a3, .Lzero

.Lbig_loop:
  and      t0, a0, t6
  sub      t0, t5, t0
  and      t1, a1, t6
  sub      t1, t5, t1
  bgeu     t1, t0, 4f
  mv       t0, t1
4:
  bgeu     a3, t0, 5f
  mv       t0, a3
5:
  vsetvli  t2, t0, e8, m8, ta, mu
  vle8.v   v8, (a0)
  vle8.v   v16, (a1)
  vsub.vx  v24, v8, a5
  vmsltu.vx v0, v24, a6
  vadd.vx  v8, v8, a7, v0.t
  vsub.vx  v24, v16, a5
  vmsltu.vx v0, v24, a6
  vadd.vx  v16, v16, a7, v0.t
  vmsne.vv v0, v8, v16
  vmseq.vi v1, v8, 0
  vmor.mm  v0, v0, v1
  vfirst.m t0, v0
  add      a0, a0, t2
  add      a1, a1, t2
  sub      a3, a3, t2
  bltz     t0, 6f
.Lhit:
  sub      a0, a0, t2
  sub      a1, a1, t2
  add      a0, a0, t0
  add      a1, a1, t0
  lbu      t0, 0(a0)
  lbu      t1, 0(a1)
  FOLD_SCALAR t0, t2
  FOLD_SCALAR t1, t2
  sub      a0, t0, t1
  ret
6:
  bnez     a3, .Lbig_loop
.Lzero:
  li       a0, 0
  ret
END(strncasecmp)

#endif
