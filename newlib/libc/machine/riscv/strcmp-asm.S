/* strcmp - RVV 1.0 candidate v2: vector-first, no vle8ff.
 *
 *   int strcmp (const char *a, const char *b)
 *
 * Algorithm
 *   1. One 64-byte probe chunk (e8/m4): two plain loads, vmsne.vv, vmseq.vi
 *      against zero, vmor.mm, one vfirst.m.
 *   2. e8/m8 loop, VLMAX bytes per iteration, same body.
 *   The stop mask is "a[i] != b[i] OR a[i] == 0".  A NUL in b alone is
 *   covered by the first term because a[i] != 0 there.  On the first hit the
 *   two original bytes are reloaded and subtracted, so the returned value is
 *   the unsigned char difference and has the sign the standard requires.
 *
 * Page safety
 *   vl = min(bytes left in a's page, bytes left in b's page), and vsetvli
 *   with rd != x0 both clamps to VLMAX and returns the real vl in t2; the
 *   pointers advance by exactly that.  A plain vle8.v therefore never reads
 *   past either page.  No vle8ff, no backwards reads.
 *
 * RV32/RV64: XLEN independent, no stack, no callee-saved registers.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strcmp)
  li       t6, 4095
  li       t5, 4096
  li       t3, 65

  /* 64-byte probe, capped by both page distances */
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
  vsetvli  t2, t3, e8, m8, ta, ma
  vle8.v   v8, (a0)
  vle8.v   v16, (a1)
  vmsne.vv v0, v8, v16
  vmseq.vi v1, v8, 0
  vmor.mm  v0, v0, v1
  vfirst.m t0, v0
  add      a0, a0, t2
  add      a1, a1, t2
  bgez     t0, .Lhit

  /* ---- streaming e8/m8 loop ---------------------------------------- */
.Lbig_loop:
  and      t0, a0, t6
  sub      t0, t5, t0
  and      t1, a1, t6
  sub      t1, t5, t1
  bgeu     t1, t0, 3f
  mv       t0, t1
3:
  vsetvli  t2, t0, e8, m8, ta, ma
  vle8.v   v8, (a0)
  vle8.v   v16, (a1)
  vmsne.vv v0, v8, v16
  vmseq.vi v1, v8, 0
  vmor.mm  v0, v0, v1
  vfirst.m t0, v0
  add      a0, a0, t2
  add      a1, a1, t2
  bltz     t0, .Lbig_loop

.Lhit:
  sub      a0, a0, t2
  sub      a1, a1, t2
  add      a0, a0, t0
  add      a1, a1, t0
  lbu      t0, 0(a0)
  lbu      t1, 0(a1)
  sub      a0, t0, t1
  ret
END(strcmp)

#endif
