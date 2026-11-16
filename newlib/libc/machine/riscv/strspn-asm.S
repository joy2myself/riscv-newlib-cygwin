/* strspn, RVV 1.0, v2.
 *
 * size_t strspn(const char *s, const char *set)
 *
 * Two shapes, chosen once per call from the set length k:
 *   k <= 8 : compare the input chunk against each set byte with vmseq.vx and
 *            AND the "not equal" masks (no table, no gather, no memory
 *            traffic for the set beyond k scalar loads per chunk);
 *   k >  8 : build a 256-bit membership bitmap (32 stack bytes) and test each
 *            input byte with two vector shifts + one vrgather (m2 chunks).
 * Both scan with VLMAX-aligned full vle8.v loads (page-safe: VLMAX divides
 * the page size, so an aligned block never crosses a page), with a
 * sub-VLMAX leading chunk to reach the alignment.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__riscv_e) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

#define SET_MAX_DIRECT 8

/* Direct chunk: v1 = mask of lanes NOT in the set (and not NUL: NUL is never
   in a C-string set, so it is naturally "not in set").  */
.macro SPN_DCHUNK
  vle8.v   v8, (a0)
  vmset.m  v1
  mv       a3, a1
  mv       t3, t5
  beqz     t3, 9f
8:
  lbu      t4, 0(a3)
  vmsne.vx v2, v8, t4
  vmand.mm v1, v1, v2
  addi     a3, a3, 1
  addi     t3, t3, -1
  bnez     t3, 8b
9:
.endm

/* Bitmap chunk: v1 = mask of lanes whose byte is not in the bitmap.  */
.macro SPN_BCHUNK
  vle8.v      v8, (a0)
  vsrl.vi     v10, v8, 3
  vand.vi     v12, v8, 7
  vsll.vv     v14, v18, v12
  vrgather.vv v20, v16, v10
  vand.vv     v22, v20, v14
  vmseq.vi    v1, v22, 0
.endm

ENTRY(strspn)
  mv      a2, a0                     /* start */
  /* k = min(strlen(set), SET_MAX_DIRECT+1) */
  li      t5, 0
  addi    t3, zero, SET_MAX_DIRECT+1
.Lkloop:
  add     t4, a1, t5
  lbu     t4, 0(t4)
  beqz    t4, .Lkdone
  addi    t5, t5, 1
  bltu    t5, t3, .Lkloop
.Lkdone:
  beqz    t5, .Lzero_len             /* empty set: span is 0 */
  addi    t3, zero, SET_MAX_DIRECT
  bltu    t3, t5, .Lbitmap

  /* ---------------- direct path (1 <= k <= 8) ---------------- */
  vsetvli t0, zero, e8, m8, ta, ma
  neg     t1, t0
  and     t1, a0, t1
  add     t1, t1, t0
  sub     t1, t1, a0                 /* lead */
  beqz    t1, .Ldmain
  vsetvli zero, t1, e8, m8, ta, ma
  SPN_DCHUNK
  vfirst.m t6, v1
  bgez    t6, .Lfound
  add     a0, a0, t1
.Ldmain:
  vsetvli t0, zero, e8, m8, ta, ma
.Ldloop:
  SPN_DCHUNK
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
  vle8.v  v16, (sp)                  /* 256-bit membership bitmap */
  neg     t1, t0
  and     t1, a0, t1
  add     t1, t1, t0
  sub     t1, t1, a0                 /* lead */
  beqz    t1, .Lbmain
  vsetvli zero, t1, e8, m2, ta, ma
  SPN_BCHUNK
  vfirst.m t6, v1
  bgez    t6, .Lbfound
  add     a0, a0, t1
.Lbmain:
  vsetvli t0, zero, e8, m2, ta, ma
.Lbloop:
  SPN_BCHUNK
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
.Lzero_len:
  li      a0, 0
  ret
END(strspn)
#endif
