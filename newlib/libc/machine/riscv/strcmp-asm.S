/* strcmp - RVV 1.0 candidate v3: scalar 16-byte word pair head, then the
 * e8/m8 streaming loop.  No vle8ff.
 *
 *   int strcmp (const char *a, const char *b)
 *
 * Why
 *   The C920v2 is a scalar-first core and the 8/16-byte rows dominate real
 *   calls.  Two independent 8-byte loads per string compare 16 bytes per
 *   iteration with four loads and two xors; the equal-and-no-NUL fast path
 *   has no vector setup at all.  A difference (the rare case, and the case
 *   the benchmark does not use) is resolved by a short byte loop that stops
 *   at the earlier of the difference and the terminator.
 *
 * Page safety: the 16-byte pair is only issued when all 16 bytes are in the
 * same page (single `and`+`bgeu`); otherwise the vector loop takes over,
 * which caps vl by both page distances.  No load crosses a page and nothing
 * before either string is read.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strcmp)
#if __riscv_xlen == 64
  li       a3, 0x7f7f7f7f
  slli     t0, a3, 32
  or       a3, a3, t0             /* SEV = 0x7f7f..7f                   */
  li       a6, 4095
  li       a7, 4096 - 16
  addi     a5, a0, 32

.Lhead:
  and      t0, a0, a6
  bgeu     t0, a7, .Lvec
  ld       t0, 0(a0)
  ld       t1, 0(a1)
  ld       t2, 8(a0)
  ld       t3, 8(a1)
  xor      t4, t0, t1
  xor      t5, t2, t3
  or       t4, t4, t5
  bnez     t4, .Ldiff16
  and      t4, t0, a3
  add      t4, t4, a3
  or       t4, t4, t0
  or       t4, t4, a3
  not      t4, t4
  and      t5, t2, a3
  add      t5, t5, a3
  or       t5, t5, t2
  or       t5, t5, a3
  not      t5, t5
  or       t4, t4, t5
  bnez     t4, .Lequal
  addi     a0, a0, 16
  addi     a1, a1, 16
  bltu     a0, a5, .Lhead
#endif

.Lvec:
  li       t6, 4095
  li       t5, 4096
.Lbig_loop:
  and      t0, a0, t6
  sub      t0, t5, t0
  and      t1, a1, t6
  sub      t1, t5, t1
  bgeu     t1, t0, 1f
  mv       t0, t1
1:
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
  sub      a0, a0, t2
  sub      a1, a1, t2
  add      a0, a0, t0
  add      a1, a1, t0
  lbu      t0, 0(a0)
  lbu      t1, 0(a1)
  sub      a0, t0, t1
  ret

.Lequal:
  li       a0, 0
  ret

.Ldiff16:
  li       t0, 16
.Ldloop:
  lbu      t1, 0(a0)
  lbu      t2, 0(a1)
  bne      t1, t2, .Ldret
  beqz     t1, .Ldzero
  addi     a0, a0, 1
  addi     a1, a1, 1
  addi     t0, t0, -1
  bnez     t0, .Ldloop
  li       a0, 0
  ret
.Ldret:
  sub      a0, t1, t2
  ret
.Ldzero:
  li       a0, 0
  ret
END(strcmp)

#endif
