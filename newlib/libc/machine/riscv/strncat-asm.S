/* strncat, RVV 1.0, v2.
 *
 * char *strncat(char *dst, const char *src, size_t n)
 *
 * strlen(dst) with VLMAX-aligned vle8.v loads; then at most n bytes of src
 * are appended: a chunk store is masked with vmsif (payload + NUL when the
 * source ends inside the chunk, all ones otherwise) and the terminating NUL
 * is written by the store itself or by a single sb when n ran out first.
 * The source is never read past src+n and never past its NUL.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strncat)
  mv      a3, a0                     /* running dst */
  vsetvli t0, zero, e8, m8, ta, ma
  neg     t1, t0
  and     t1, a3, t1
  add     t1, t1, t0
  sub     t1, t1, a3                 /* lead (strlen phase) */
  beqz    t1, .Lscan
  vsetvli zero, t1, e8, m8, ta, ma
  vle8.v  v8, (a3)
  vmseq.vi v1, v8, 0
  vfirst.m t2, v1
  bgez    t2, .Lfound
  add     a3, a3, t1
.Lscan:
  vsetvli t0, zero, e8, m8, ta, ma
.Lsloop:
  vle8.v  v8, (a3)
  vmseq.vi v1, v8, 0
  vfirst.m t2, v1
  add     a3, a3, t0
  bltz    t2, .Lsloop
  sub     a3, a3, t0
.Lfound:
  add     a3, a3, t2                 /* a3 -> terminating NUL of dst */

  /* append phase */
  beqz    a2, .Lterm
  neg     t1, t0
  and     t1, a1, t1
  add     t1, t1, t0
  sub     t1, t1, a1                 /* lead (source) */
  beqz    t1, .Lbulk
  bltu    a2, t1, .Ltail
  vsetvli zero, t1, e8, m8, ta, ma
  vle8.v  v8, (a1)
  vmseq.vi v1, v8, 0
  vmsif.m v0, v1
  vse8.v  v8, (a3), v0.t
  vfirst.m t2, v1
  bgez    t2, .Lret
  add     a1, a1, t1
  add     a3, a3, t1
  sub     a2, a2, t1
  beqz    a2, .Lterm
.Lbulk:
  vsetvli t0, zero, e8, m8, ta, ma
.Lloop:
  bltu    a2, t0, .Ltail
  vle8.v  v8, (a1)
  vmseq.vi v1, v8, 0
  vmsif.m v0, v1
  vse8.v  v8, (a3), v0.t
  vfirst.m t2, v1
  bgez    t2, .Lret
  add     a1, a1, t0
  add     a3, a3, t0
  sub     a2, a2, t0
  bnez    a2, .Lloop
.Lterm:
  sb      zero, (a3)
  ret
.Ltail:
  vsetvli zero, a2, e8, m8, ta, ma
  vle8.v  v8, (a1)
  vmseq.vi v1, v8, 0
  vmsif.m v0, v1
  vse8.v  v8, (a3), v0.t
  vfirst.m t2, v1
  bgez    t2, .Lret
  add     a3, a3, a2
  sb      zero, (a3)
.Lret:
  ret
END(strncat)
#endif
