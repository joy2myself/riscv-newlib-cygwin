/* strrchr - RVV 1.0 candidate v3: two-pass (find the terminator with the
 * vle8ff-free vector scan, then scan backwards for c with 8-byte scalar
 * words).  No vle8ff.
 *
 *   char *strrchr (const char *s, int c)
 *
 * This is the simple alternative to strrchr-v2: no per-chunk c bookkeeping,
 * so the forward pass is as cheap as strlen, and the backward pass is a
 * scalar word loop that stops at the first match.  Cost is best when the
 * last c is close to the end (one or two words) and worst when c is absent
 * or far away (a second pass at ~1.1 instructions/byte).  v2 is O(1) in
 * that respect; measure both on the boards.
 *
 * Page safety: the forward pass caps vl by the page distance; the backward
 * pass only reads inside [s, terminator], which the forward pass has already
 * proved mapped, and never goes below s.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__riscv_e) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

#if __riscv_xlen == 64
.macro RZIDX dst, msk, tmp
  bltz \msk, .Lr7\@
  slli \tmp, \msk, 8
  bltz \tmp, .Lr6\@
  slli \tmp, \msk, 16
  bltz \tmp, .Lr5\@
  slli \tmp, \msk, 24
  bltz \tmp, .Lr4\@
  slli \tmp, \msk, 32
  bltz \tmp, .Lr3\@
  slli \tmp, \msk, 40
  bltz \tmp, .Lr2\@
  slli \tmp, \msk, 48
  bltz \tmp, .Lr1\@
  li \dst, 0
  j .Lre\@
.Lr7\@:
  li \dst, 7
  j .Lre\@
.Lr6\@:
  li \dst, 6
  j .Lre\@
.Lr5\@:
  li \dst, 5
  j .Lre\@
.Lr4\@:
  li \dst, 4
  j .Lre\@
.Lr3\@:
  li \dst, 3
  j .Lre\@
.Lr2\@:
  li \dst, 2
  j .Lre\@
.Lr1\@:
  li \dst, 1
.Lre\@:
.endm

ENTRY(strrchr)
  andi     a1, a1, 255
  mv       a2, a0
  li       t6, 4095
  li       t4, 64
  li       t5, 4096 - 64
  vsetvli  zero, t4, e8, m4, ta, ma
  and      t0, a0, t6
  bgeu     t0, t5, .Lprobe_part

  vle8.v   v8, (a0)
  vmseq.vi v0, v8, 0
  vfirst.m t0, v0
  bgez     t0, .Lprobe_hit
  addi     a0, a0, 64
  j        .Lbig_setup
.Lprobe_hit:
  add      a3, a0, t0
  j        .Lscan_back

.Lprobe_part:
  li       t1, 4096
  sub      t1, t1, t0
  vsetvli  zero, t1, e8, m4, ta, ma
  vle8.v   v8, (a0)
  vmseq.vi v0, v8, 0
  vfirst.m t0, v0
  add      a0, a0, t1
  bltz     t0, .Lbig_setup
  sub      a0, a0, t1
  add      a3, a0, t0
  j        .Lscan_back

.Lbig_setup:
  vsetvli  t3, zero, e8, m8, ta, ma
  li       t5, 4096
  sub      t5, t5, t3
.Lbig_loop:
  and      t0, a0, t6
  bgeu     t0, t5, .Lbig_part
  vle8.v   v8, (a0)
  vmseq.vi v0, v8, 0
  vfirst.m t0, v0
  bgez     t0, .Lbig_hit
  add      a0, a0, t3
  j        .Lbig_loop
.Lbig_hit:
  add      a3, a0, t0
  j        .Lscan_back
.Lbig_part:
  li       t1, 4096
  sub      t1, t1, t0
  vsetvli  zero, t1, e8, m8, ta, ma
  vle8.v   v8, (a0)
  vmseq.vi v0, v8, 0
  vfirst.m t0, v0
  add      a0, a0, t1
  bltz     t0, .Lbig_restore
  sub      a0, a0, t1
  add      a3, a0, t0
  j        .Lscan_back
.Lbig_restore:
  vsetvli  t3, zero, e8, m8, ta, ma
  j        .Lbig_loop

  /* ---- a3 = terminator address ------------------------------------- */
.Lscan_back:
  beqz     a1, .Lret_nul
  li       t0, 0x01010101
  slli     t1, t0, 32
  or       t0, t0, t1
  mul      a4, a1, t0             /* splat(c)                           */
  li       t5, 0x7f7f7f7f
  slli     t1, t5, 32
  or       t5, t5, t1             /* 0x7f7f..7f                         */
  mv       a0, a3

.Lbbyte:
  beq      a0, a2, .Lnull
  andi     t0, a0, 7
  beqz     t0, .Lbword
  addi     a0, a0, -1
  lbu      t0, 0(a0)
  bne      t0, a1, .Lbbyte
  ret

.Lbword:
  sub      t0, a0, a2
  li       t1, 8
  bltu     t0, t1, .Lbtail
  addi     a0, a0, -8
  ld       t0, 0(a0)
  xor      t0, t0, a4
  and      t1, t0, t5
  add      t1, t1, t5
  or       t1, t1, t0
  or       t1, t1, t5
  not      t1, t1
  beqz     t1, .Lbword
  RZIDX    t1, t1, t2
  add      a0, a0, t1
  ret

.Lbtail:
  beq      a0, a2, .Lnull
  addi     a0, a0, -1
  lbu      t0, 0(a0)
  bne      t0, a1, .Lbtail
  ret

.Lret_nul:
  mv       a0, a3
  ret
.Lnull:
  li       a0, 0
  ret
END(strrchr)

#else /* RV32: keep the previous vector-only implementation */
ENTRY(strrchr)
  li a6, 0
.Lloop32:
  vsetvli a2, zero, e8, m1, ta, ma
  vle8ff.v v8, (a0)
  csrr a2, vl
  vmseq.vi v1, v8, 0
  vfirst.m a3, v1
  vmseq.vx v2, v8, a1
  vmsif.m v3, v1
  vmand.mm v0, v2, v3
  vfirst.m a4, v0
  bltz a4, .Lskip32
  vsetvli zero, a2, e16, m2, ta, ma
  vid.v v16
  vmv.s.x v24, x0
  vredmaxu.vs v24, v16, v24, v0.t
  vmv.x.s a5, v24
  add a6, a0, a5
.Lskip32:
  bgez a3, .Ldone32
  add a0, a0, a2
  j .Lloop32
.Ldone32:
  mv a0, a6
  ret
END(strrchr)
#endif

#endif
