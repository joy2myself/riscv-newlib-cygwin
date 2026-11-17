/* strcasecmp - RVV 1.0 candidate v3: one e8/m1 VLMAX probe chunk then the
 * e8/m8 loop.  No vle8ff.
 *
 *   int strcasecmp (const char *a, const char *b)
 *
 * Folding (ASCII A-Z only, as the C locale requires):
 *     if ((unsigned)(x - 'A') < 26) x += 32;
 * done in-vector on both operands with one vmsltu.vi + masked vadd.vx each;
 * masked lanes are preserved because the vtype is ta,mu.  None of '@', '[',
 * '`', '{', '~' or any 0x80..0xff byte folds.
 *
 * The first chunk uses LMUL=m1 at VLMAX (16/32/64 bytes for VLEN 128/256/
 * 512) instead of a fixed 64-byte m4 chunk: less vector register pressure
 * and a shorter latency chain for the 8/16-byte rows, at the cost of one
 * more iteration for 32-63-byte strings.  Body: 2 plain loads + 6 fold ops
 * + vmsne.vv + vmseq.vi + vmor.mm + vfirst.m; then an e8/m8 VLMAX loop.
 * vl = min(the two page distances), clamped by VLMAX; the two loads can never
 * cross a page.  On the first hit the original bytes are reloaded and folded
 * scalar-wise, so the returned difference has the standard's sign.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

.macro FOLD_SCALAR reg, tmp
  addi  \tmp, \reg, -65
  sltiu \tmp, \tmp, 26
  slli  \tmp, \tmp, 5
  add   \reg, \reg, \tmp
.endm

ENTRY(strcasecmp)
  li       a5, 65                 /* 'A' */
  li       a6, 26                 /* alphabet size */
  li       a7, 32                 /* case bit */
  li       t6, 4095
  li       t5, 4096

  and      t0, a0, t6
  sub      t0, t5, t0
  and      t1, a1, t6
  sub      t1, t5, t1
  bgeu     t1, t0, 1f
  mv       t0, t1
1:
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
  bgez     t0, .Lhit

.Lbig_loop:
  and      t0, a0, t6
  sub      t0, t5, t0
  and      t1, a1, t6
  sub      t1, t5, t1
  bgeu     t1, t0, 3f
  mv       t0, t1
3:
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
  bltz     t0, .Lbig_loop

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
END(strcasecmp)

#endif
