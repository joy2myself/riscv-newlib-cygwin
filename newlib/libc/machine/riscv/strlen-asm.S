/* strlen - RVV 1.0 candidate v3: scalar 16-byte word pair up to 32 bytes,
 * then the e8/m8 streaming loop.  No vle8ff.
 *
 *   size_t strlen (const char *s)
 *
 * Why a scalar head
 *   glibc is a 8-byte scalar word loop; on the C920v2 scalar loads are its
 *   strongest suit and vector-scalar co-issue is weak.  The first 32 bytes
 *   are therefore handled with two independent 8-byte `ld`s per iteration
 *   (twice glibc's bytes per branch), and only longer strings pay the
 *   one-off vector configuration.
 *
 * Page safety
 *   Two 8-byte loads at s..s+15 are only issued when the very same page
 *   holds all 16 bytes (`andi`-free `and`+`bgeu` on the page offset).  If
 *   not, a byte loop walks to the page end and then the aligned-by-page
 *   path resumes.  No backwards read, no vle8ff, no load ever crosses a
 *   page boundary.
 *
 * Short-string result
 *   The first zero byte in a 16-byte pair is located with the classic
 *   borrow/high-bit mask (no Zbb on either board) and a small branch ladder
 *   (base RV64 has no ctz).  In the common cases the ladder costs one or
 *   two tests.
 *
 * RV32: the scalar head is RV64-only; RV32 goes straight to the vector
 * loop, which is XLEN independent.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

#if __riscv_xlen == 64
/* byte index of the lowest set 0x80-group in \msk (mask != 0) */
.macro ZIDX dst, msk, tmp
  slli \tmp, \msk, 56
  bltz \tmp, .Lz0\@
  slli \tmp, \msk, 48
  bltz \tmp, .Lz1\@
  slli \tmp, \msk, 40
  bltz \tmp, .Lz2\@
  slli \tmp, \msk, 32
  bltz \tmp, .Lz3\@
  slli \tmp, \msk, 24
  bltz \tmp, .Lz4\@
  slli \tmp, \msk, 16
  bltz \tmp, .Lz5\@
  slli \tmp, \msk, 8
  bltz \tmp, .Lz6\@
  li \dst, 7
  j .Lze\@
.Lz0\@:
  li \dst, 0
  j .Lze\@
.Lz1\@:
  li \dst, 1
  j .Lze\@
.Lz2\@:
  li \dst, 2
  j .Lze\@
.Lz3\@:
  li \dst, 3
  j .Lze\@
.Lz4\@:
  li \dst, 4
  j .Lze\@
.Lz5\@:
  li \dst, 5
  j .Lze\@
.Lz6\@:
  li \dst, 6
.Lze\@:
.endm
#endif

ENTRY(strlen)
  mv       a2, a0

#if __riscv_xlen == 64
  li       t3, 0x7f7f7f7f
  slli     t5, t3, 32
  or       t3, t3, t5             /* SEV = 0x7f7f..7f                  */
  li       a3, 4095
  li       a6, 4096 - 16
  addi     a5, a0, 32             /* scalar head limit                 */

.Lhead:
  and      t0, a0, a3
  bgeu     t0, a6, .Lhead_byte
  ld       t0, 0(a0)
  ld       t1, 8(a0)
  /* exact zero-byte mask: ~(((w & 0x7f..) + 0x7f..) | w | 0x7f..)      */
  and      t2, t0, t3
  add      t2, t2, t3
  or       t2, t2, t0
  or       t2, t2, t3
  not      t2, t2                 /* z0                                */
  and      t6, t1, t3
  add      t6, t6, t3
  or       t6, t6, t1
  or       t6, t6, t3
  not      t6, t6                 /* z1                                */
  or       a4, t2, t6
  bnez     a4, .Lhead_hit
  addi     a0, a0, 16
  bltu     a0, a5, .Lhead
  j        .Lbig_setup

.Lhead_hit:
  beqz     t2, .Lhead_hit1
  ZIDX     t0, t2, t5
  sub      a0, a0, a2
  add      a0, a0, t0
  ret
.Lhead_hit1:
  ZIDX     t0, t6, t5
  sub      a0, a0, a2
  add      a0, a0, t0
  addi     a0, a0, 8              /* second word starts at byte 8       */
  ret

.Lhead_byte:                       /* < 16 bytes left in this page      */
  lbu      t0, 0(a0)
  beqz     t0, .Lbyte_zero
  addi     a0, a0, 1
  and      t0, a0, a3
  bnez     t0, .Lhead_byte
  j        .Lhead                  /* page aligned again                */
.Lbyte_zero:
  sub      a0, a0, a2
  ret
#else
  j        .Lbig_setup
#endif

  /* ---- streaming loop: e8/m8, VLMAX bytes per iteration ------------- */
.Lbig_setup:
  /* vsetvli x0,x0 preserves vl; write the result to get vl=VLMAX.      */
  vsetvli  t3, zero, e8, m8, ta, ma   /* t3 = VLMAX bytes (8*vlenb)     */
  li       t5, 4096
  sub      t5, t5, t3             /* safe while off <= 4096 - VLMAX     */

.Lbig_loop:
  and      t0, a0, a3
  bgeu     t0, t5, .Lbig_part
  vle8.v   v8, (a0)
  vmseq.vi v0, v8, 0
  vfirst.m t0, v0
  add      a0, a0, t3
  bltz     t0, .Lbig_loop
  sub      a0, a0, a2
  sub      a0, a0, t3
  add      a0, a0, t0
  ret

.Lbig_part:
  li       t1, 4096
  sub      t1, t1, t0             /* vl = bytes left in page, <= VLMAX  */
  vsetvli  zero, t1, e8, m8, ta, ma
  vle8.v   v8, (a0)
  vmseq.vi v0, v8, 0
  vfirst.m t0, v0
  add      a0, a0, t1
  bltz     t0, .Lbig_restore
  sub      a0, a0, a2
  sub      a0, a0, t1
  add      a0, a0, t0
  ret

.Lbig_restore:
  vsetvli  t3, zero, e8, m8, ta, ma   /* vl=VLMAX again                  */
  j        .Lbig_loop
END(strlen)

#endif /* __riscv_vector && !size-optimized */
