/* strlen - RVV 1.0 candidate v2: vector-first, page-safe, no vle8ff.
 *
 *   size_t strlen (const char *s)
 *
 * Algorithm
 *   1. One 65-byte probe load (LMUL=m8, vl=65) finds the terminator of the
 *      common short string.  vl=65 is legal for every VLEN >= 128 (m8
 *      VLMAX = VLEN bytes >= 128) and, unlike a 64-byte probe, it covers a
 *      terminator at index 64 -- one of the benchmarked sizes.
 *   2. If the probe misses, switch once to e8/m8 with vl=VLMAX and stream
 *      128/256/512/1024 bytes per iteration (VLEN 128..1024).
 *
 * Page safety (no vle8ff anywhere)
 *   Only the distance to the next 4 KiB page boundary is examined, with a
 *   single `andi` + `bgeu` pair.  If the whole next chunk fits before the
 *   boundary a plain vle8.v is used (it cannot cross into an unmapped page);
 *   otherwise vl is narrowed to exactly the bytes left in the page and the
 *   next iteration starts page-aligned.  The narrowing vsetvli also works
 *   when the distance is larger than VLMAX: the hardware clamps vl to VLMAX,
 *   and the chunk still fits in the page (distance >= VLMAX by construction).
 *   Every load is at or after s, so bytes before the object are never read.
 *   `strlen` may only touch bytes through the NUL; every vector load either
 *   lies entirely inside the mapped page or is narrowed to the page end.
 *
 * RV32/RV64: no pointer-width assumptions, VLEN is read from vlenb, one
 * global symbol, no stack traffic, no callee-saved register.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

#define PROBE_BYTES 65
#define PGSZ        4096

ENTRY(strlen)
  mv       a2, a0                 /* start                              */
  li       t6, 4095               /* page offset mask (andi imm is s12) */
  li       t4, PROBE_BYTES
  li       t5, PGSZ - PROBE_BYTES
  vsetvli  zero, t4, e8, m8, ta, ma
  and      t0, a0, t6
  bgeu     t0, t5, .Lprobe_part

  /* ---- 64-byte probe, all of it inside the page --------------------- */
  vle8.v   v8, (a0)
  vmseq.vi v0, v8, 0
  vfirst.m t0, v0
  addi     a0, a0, PROBE_BYTES
  bgez     t0, .Lprobe_hit
  j        .Lbig_setup

.Lprobe_hit:
  sub      a0, a0, a2
  addi     a0, a0, -PROBE_BYTES
  add      a0, a0, t0
  ret

.Lprobe_part:                      /* fewer than 64 bytes left in page   */
  li       t1, PGSZ
  sub      t1, t1, t0             /* vl = distance to page end (1..63)  */
  vsetvli  zero, t1, e8, m8, ta, ma
  vle8.v   v8, (a0)
  vmseq.vi v0, v8, 0
  vfirst.m t0, v0
  add      a0, a0, t1
  bltz     t0, .Lbig_setup
  sub      a0, a0, a2
  sub      a0, a0, t1
  add      a0, a0, t0
  ret

  /* ---- streaming loop: e8/m8, VLMAX bytes per iteration ------------- */
.Lbig_setup:
  /* vsetvli x0,x0 keeps the old vl -- it does NOT select VLMAX.  Writing
     the result into a register both sets vl=VLMAX and hands it to us.   */
  vsetvli  t3, zero, e8, m8, ta, ma   /* t3 = VLMAX bytes (8*vlenb)     */
  li       t5, PGSZ
  sub      t5, t5, t3             /* safe while off <= 4096 - VLMAX     */

.Lbig_loop:
  and      t0, a0, t6
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

.Lbig_part:                        /* chunk would cross the page         */
  li       t1, PGSZ
  sub      t1, t1, t0             /* vl = distance, <= VLMAX by test    */
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
  vsetvli  t3, zero, e8, m8, ta, ma   /* back to vl=VLMAX               */
  j        .Lbig_loop
END(strlen)

#endif /* __riscv_vector && !size-optimized */
