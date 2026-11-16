/* strchr - RVV 1.0 candidate v2: vector-first, no vle8ff.
 *
 *   char *strchr (const char *s, int c)
 *
 * Algorithm
 *   1. 65-byte probe (e8/m8, vl=65) with a page-distance cap.  The 65th
 *      byte matters: it covers a target/terminator at index 64.
 *   2. e8/m8 streaming loop, VLMAX bytes per iteration.
 *   Per chunk: one load, two byte compares (== 0, == (char)c), OR of the
 *   masks, one vfirst.m.  The first index where either happens is the only
 *   candidate; a single byte load at the hit decides whether it was c (return
 *   the pointer) or the terminator (return NULL).  c is truncated to char
 *   first, so strchr(s, 0x141) == strchr(s, 0x41) and c == 0 returns the
 *   pointer to the terminator.
 *
 * Page safety
 *   The chunk length is min(remaining-to-page, VLMAX): a plain vle8.v is
 *   used and can never cross into the PROT_NONE page.  When the page ends
 *   first, vl is set to exactly the bytes left, so the next iteration is
 *   page aligned.  No backwards reads.
 *
 * RV32/RV64: XLEN independent (vector path only), no callee-saved registers.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__riscv_e) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strchr)
  andi     a1, a1, 255            /* (char) c                           */
  li       t6, 4095
  li       t4, 65
  li       t5, 4096 - 65
  vsetvli  zero, t4, e8, m8, ta, ma
  and      t0, a0, t6
  bgeu     t0, t5, .Lprobe_part

  vle8.v   v8, (a0)
  vmseq.vi v1, v8, 0
  vmseq.vx v2, v8, a1
  vmor.mm   v0, v1, v2
  vfirst.m t0, v0
  addi     a0, a0, 65
  bgez     t0, .Lprobe_hit
  j        .Lbig_setup

.Lprobe_hit:
  addi     a0, a0, -65
  add      a0, a0, t0
  j        .Lfound

.Lprobe_part:
  li       t1, 4096
  sub      t1, t1, t0
  vsetvli  zero, t1, e8, m8, ta, ma
  vle8.v   v8, (a0)
  vmseq.vi v1, v8, 0
  vmseq.vx v2, v8, a1
  vmor.mm   v0, v1, v2
  vfirst.m t0, v0
  add      a0, a0, t1
  bltz     t0, .Lbig_setup
  sub      a0, a0, t1
  add      a0, a0, t0
  j        .Lfound

  /* ---- streaming e8/m8 loop ---------------------------------------- */
.Lbig_setup:
  vsetvli  t3, zero, e8, m8, ta, ma   /* t3 = VLMAX bytes               */
  li       t5, 4096
  sub      t5, t5, t3
.Lbig_loop:
  and      t0, a0, t6
  bgeu     t0, t5, .Lbig_part
  vle8.v   v8, (a0)
  vmseq.vi v1, v8, 0
  vmseq.vx v2, v8, a1
  vmor.mm   v0, v1, v2
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
  vmor.mm   v0, v1, v2
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
