/* strncmp - RVV 1.0 candidate v2: vector-first, no vle8ff.
 *
 *   int strncmp (const char *a, const char *b, size_t n)
 *
 * Same chunk body as strcmp-v2 (two plain loads, vmsne.vv, vmseq.vi, vmor.mm,
 * vfirst.m), with vl additionally capped by the remaining count so not one
 * byte past the n-th is ever read.  Exhausting n without a NUL or a
 * difference returns 0.
 *
 * Page safety: vl = min(n_left, bytes-to-a's-page, bytes-to-b's-page);
 * vsetvli returns the clamped vl in t2, both pointers advance by it.
 * No vle8ff and no read outside [a,a+n) / [b,b+n).
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strncmp)
  beqz     a2, .Lzero
  mv       a3, a2                 /* bytes left                        */
  li       t6, 4095
  li       t5, 4096
  li       t3, 64

  and      t0, a0, t6
  sub      t0, t5, t0
  and      t1, a1, t6
  sub      t1, t5, t1
  bgeu     t1, t0, 1f
  mv       t0, t1
1:
  bgeu     t0, t3, 2f
  mv       t3, t0
2:
  bgeu     a3, t3, 3f
  mv       t3, a3
3:
  vsetvli  t2, t3, e8, m4, ta, ma
  vle8.v   v8, (a0)
  vle8.v   v16, (a1)
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
  vsetvli  t2, t0, e8, m8, ta, ma
  vle8.v   v8, (a0)
  vle8.v   v16, (a1)
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
  sub      a0, t0, t1
  ret
6:
  bnez     a3, .Lbig_loop
.Lzero:
  li       a0, 0
  ret
END(strncmp)

#endif
