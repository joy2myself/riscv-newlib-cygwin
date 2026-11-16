/* strncpy, RVV 1.0, v2: aligned vle8.v loads, masked exact stores, m8.
 *
 * char *strncpy(char *dst, const char *src, size_t n)
 *
 * Bulk: VLMAX bytes per iteration, src aligned to VLMAX so every full load is
 * inside one page.  vl is capped by the remaining n so the source is never
 * read further than n bytes.  The store is masked with vmsif (0..nul
 * inclusive, all ones when the chunk has no NUL) so exactly the bytes the
 * standard requires are written; the zero fill up to n is written with
 * unmasked e8/m8 stores whose vl is the exact remainder.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strncpy)
  mv      a3, a0                     /* running destination */
  beqz    a2, .Lret
  vsetvli t0, zero, e8, m8, ta, ma   /* t0 = VLMAX */
  neg     t1, t0
  and     t1, a1, t1
  add     t1, t1, t0
  sub     t1, t1, a1                 /* lead */
  beqz    t1, .Lbulk
  bltu    a2, t1, .Ltail             /* n < lead: one short chunk */

  /* leading partial chunk, vl = lead (inside one aligned block) */
  vsetvli zero, t1, e8, m8, ta, ma
  vle8.v  v8, (a1)
  vmseq.vi v1, v8, 0
  vmsif.m v0, v1
  vse8.v  v8, (a3), v0.t
  vfirst.m t2, v1
  bgez    t2, .Lzero
  add     a1, a1, t1
  add     a3, a3, t1
  sub     a2, a2, t1
  beqz    a2, .Lret

.Lbulk:
  vsetvli t0, zero, e8, m8, ta, ma
.Lloop:
  bltu    a2, t0, .Ltail
  vle8.v  v8, (a1)
  vmseq.vi v1, v8, 0
  vmsif.m v0, v1
  vse8.v  v8, (a3), v0.t
  vfirst.m t2, v1
  bgez    t2, .Lzero
  add     a1, a1, t0
  add     a3, a3, t0
  sub     a2, a2, t0
  bnez    a2, .Lloop
  ret

.Ltail:
  vsetvli zero, a2, e8, m8, ta, ma
  vle8.v  v8, (a1)
  vmseq.vi v1, v8, 0
  vmsif.m v0, v1
  vfirst.m t2, v1
  vse8.v  v8, (a3), v0.t
  bltz    t2, .Lret                  /* no NUL: exactly n bytes copied */
  /* fall through */

  /* NUL at index t2 of a chunk whose vl was a2: zero fill the rest of n */
.Lzero:
  addi    t3, t2, 1
  add     a3, a3, t3
  sub     a2, a2, t3
  beqz    a2, .Lret
.Lfill:
  vsetvli t1, a2, e8, m8, ta, ma
  vmv.v.i v8, 0                     /* zero at the fill vl, not the chunk vl */
  vse8.v  v8, (a3)
  add     a3, a3, t1
  sub     a2, a2, t1
  bnez    a2, .Lfill
.Lret:
  ret
END(strncpy)
#endif
