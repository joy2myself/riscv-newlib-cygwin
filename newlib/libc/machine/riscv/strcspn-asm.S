/* strcspn, RVV 1.0, v2.
 *
 * size_t strcspn(const char *s, const char *set)
 *
 * Same two shapes as strspn-v2:
 *   k <= 8 : vmseq.vx against each set byte, OR the masks together with the
 *            NUL mask (NUL always stops);
 *   k >  8 : 256-bit membership bitmap + vrgather lookup (m2 chunks).
 * VLMAX-aligned full vle8.v loads keep every load inside one page.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__riscv_e) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

#define SET_MAX_DIRECT 8

/* Direct chunk: v1 = mask of lanes that stop the scan (in set, or NUL).  */
.macro CSPN_DCHUNK
  vle8.v   v8, (a0)
  vmseq.vi v1, v8, 0
  mv       a3, a1
  mv       t3, t5
  beqz     t3, 9f
8:
  lbu      t4, 0(a3)
  vmseq.vx v2, v8, t4
  vmor.mm  v1, v1, v2
  addi     a3, a3, 1
  addi     t3, t3, -1
  bnez     t3, 8b
9:
.endm

/* Bitmap chunk: v1 = mask of lanes that stop the scan (in set, or NUL).  */
.macro CSPN_BCHUNK
  vle8.v      v8, (a0)
  vsrl.vi     v10, v8, 3
  vand.vi     v12, v8, 7
  vsll.vv     v14, v18, v12
  vrgather.vv v20, v16, v10
  vand.vv     v22, v20, v14
  vmsne.vi    v1, v22, 0             /* in set */
  vmseq.vi    v2, v8, 0              /* NUL */
  vmor.mm     v1, v1, v2
.endm

ENTRY(strcspn)
  mv      a2, a0
  li      t5, 0
  addi    t3, zero, SET_MAX_DIRECT+1
.Lkloop:
  add     t4, a1, t5
  lbu     t4, 0(t4)
  beqz    t4, .Lkdone
  addi    t5, t5, 1
  bltu    t5, t3, .Lkloop
.Lkdone:
  addi    t3, zero, SET_MAX_DIRECT
  bltu    t3, t5, .Lbitmap

  /* ---------------- direct path (0 <= k <= 8) ---------------- */
  vsetvli t0, zero, e8, m8, ta, ma
  neg     t1, t0
  and     t1, a0, t1
  add     t1, t1, t0
  sub     t1, t1, a0
  beqz    t1, .Ldmain
  vsetvli zero, t1, e8, m8, ta, ma
  CSPN_DCHUNK
  vfirst.m t6, v1
  bgez    t6, .Lfound
  add     a0, a0, t1
.Ldmain:
  vsetvli t0, zero, e8, m8, ta, ma
.Ldloop:
  CSPN_DCHUNK
  vfirst.m t6, v1
  bgez    t6, .Lfound
  add     a0, a0, t0
  j       .Ldloop

  /* ---------------- bitmap path (k >= 9) ---------------- */
.Lbitmap:
  addi    sp, sp, -48
  sw      zero, 0(sp)                    /* 32-byte bitmap, XLEN independent */
  sw      zero, 4(sp)
  sw      zero, 8(sp)
  sw      zero, 12(sp)
  sw      zero, 16(sp)
  sw      zero, 20(sp)
  sw      zero, 24(sp)
  sw      zero, 28(sp)
  li      t4, 1
.Lbuild:
  lbu     t0, 0(a1)
  beqz    t0, .Lbuilt
  srli    t1, t0, 3
  andi    t2, t0, 7
  sll     t2, t4, t2
  add     t1, sp, t1
  lbu     t3, 0(t1)
  or      t3, t3, t2
  sb      t3, 0(t1)
  addi    a1, a1, 1
  j       .Lbuild
.Lbuilt:
  vsetvli t0, zero, e8, m2, ta, ma
  vmv.v.i v18, 1
  li      t3, 32
  vsetvli zero, t3, e8, m2, ta, ma
  vle8.v  v16, (sp)
  neg     t1, t0
  and     t1, a0, t1
  add     t1, t1, t0
  sub     t1, t1, a0
  beqz    t1, .Lbmain
  vsetvli zero, t1, e8, m2, ta, ma
  CSPN_BCHUNK
  vfirst.m t6, v1
  bgez    t6, .Lbfound
  add     a0, a0, t1
.Lbmain:
  vsetvli t0, zero, e8, m2, ta, ma
.Lbloop:
  CSPN_BCHUNK
  vfirst.m t6, v1
  bgez    t6, .Lbfound
  add     a0, a0, t0
  j       .Lbloop
.Lbfound:
  addi    sp, sp, 48
.Lfound:
  add     a0, a0, t6
  sub     a0, a0, a2
  ret
END(strcspn)
#endif
