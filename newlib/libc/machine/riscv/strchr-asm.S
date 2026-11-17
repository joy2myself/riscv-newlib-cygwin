/* strchr - RVV 1.0 candidate v3: scalar 16-byte word pair head, then the
 * e8/m8 streaming loop.  No vle8ff.
 *
 *   char *strchr (const char *s, int c)
 *
 * The head builds c splatted to all eight byte lanes with a multiply and
 * uses the exact per-byte zero mask
 *      ~(((w & 0x7f..) + 0x7f..) | w | 0x7f..)
 * on both w and w^splat.  The OR of the two masks has its lowest set bit at
 * the first byte that is either c or the terminator (the exact formula has
 * no borrow-chain false positives), so a short high-bit ladder plus one byte
 * check resolves the common short string with no vector configuration.
 * Strings longer than 32 bytes fall through to the same m8 loop as v2.
 *
 * Page safety: 16-byte pair only when all 16 bytes are in the page; the
 * vector loop caps vl by the page distance.  No backwards read.
 * RV32 falls straight through to the vector loop (the ladder is RV64-only).
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__riscv_e) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

#if __riscv_xlen == 64
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

ENTRY(strchr)
  andi     a1, a1, 255
  li       a6, 4095
#if __riscv_xlen == 64
  li       t0, 0x01010101
  slli     t1, t0, 32
  or       t0, t0, t1
  mul      a3, a1, t0             /* c in all 8 byte lanes              */
  li       t3, 0x7f7f7f7f
  slli     t1, t3, 32
  or       t3, t3, t1             /* 0x7f7f..7f                         */
  li       a6, 4095
  li       a7, 4096 - 16
  addi     a5, a0, 32

.Lhead:
  and      t0, a0, a6
  bgeu     t0, a7, .Lbig_setup
  ld       t0, 0(a0)
  ld       t1, 8(a0)
  xor      t2, t0, a3
  xor      t5, t1, a3
  /* exact zero-byte masks: word 0 and its c-match */
  and      t6, t0, t3
  add      t6, t6, t3
  or       t6, t6, t0
  or       t6, t6, t3
  not      t6, t6                 /* NUL mask, word 0                   */
  and      a2, t2, t3
  add      a2, a2, t3
  or       a2, a2, t2
  or       a2, a2, t3
  not      a2, a2                 /* c mask, word 0                     */
  or       t6, t6, a2
  /* word 1 */
  and      a2, t1, t3
  add      a2, a2, t3
  or       a2, a2, t1
  or       a2, a2, t3
  not      a2, a2
  and      t2, t5, t3
  add      t2, t2, t3
  or       t2, t2, t5
  or       t2, t2, t3
  not      t2, t2
  or       a2, a2, t2
  /* first event in the 16-byte pair */
  bnez     t6, .Lhit0
  bnez     a2, .Lhit1
  addi     a0, a0, 16
  bltu     a0, a5, .Lhead
  j        .Lbig_setup

.Lhit0:
  ZIDX     t0, t6, t2
  add      a0, a0, t0
  j        .Lfound
.Lhit1:
  ZIDX     t0, a2, t2
  add      a0, a0, t0
  addi     a0, a0, 8
  j        .Lfound
#else
  j        .Lbig_setup
#endif

.Lbig_setup:
  vsetvli  t3, zero, e8, m8, ta, ma
  li       t5, 4096
  sub      t5, t5, t3
.Lbig_loop:
  and      t0, a0, a6
  bgeu     t0, t5, .Lbig_part
  vle8.v   v8, (a0)
  vmseq.vi v1, v8, 0
  vmseq.vx v2, v8, a1
  vmor.mm  v0, v1, v2
  vfirst.m t0, v0
  add      a0, a0, t3
  bltz     t0, .Lbig_loop
  sub      a0, a0, t3
  add      a0, a0, t0
  j        .Lfound
.Lbig_part:
  li       t1, 4096
  sub      t1, t1, t0
  vsetvli  zero, t1, e8, m8, ta, ma
  vle8.v   v8, (a0)
  vmseq.vi v1, v8, 0
  vmseq.vx v2, v8, a1
  vmor.mm  v0, v1, v2
  vfirst.m t0, v0
  add      a0, a0, t1
  bltz     t0, .Lbig_restore
  sub      a0, a0, t1
  add      a0, a0, t0
  j        .Lfound
.Lbig_restore:
  vsetvli  t3, zero, e8, m8, ta, ma
  j        .Lbig_loop
.Lfound:
  lbu      t0, 0(a0)
  beq      t0, a1, .Lret
  li       a0, 0
.Lret:
  ret
END(strchr)

#endif
