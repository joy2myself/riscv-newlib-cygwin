/* strcpy, RVV 1.0, v2: aligned full-vector loads, masked exact stores.
 *
 * Differences from the shipped version:
 *   - the source pointer is aligned to VLMAX so that every vle8.v in the
 *     hot loop is entirely inside one page (VLMAX divides the page size),
 *     which removes the per-element vle8ff.v slow path entirely;
 *   - e8/m8 gives VLMAX = vlenb*8 bytes per iteration (32..1024 B) instead
 *     of e8/m1 (16..128 B);
 *   - vsetvli lives outside the loop; vl is constant, so no csrr vl.
 *
 * Destination safety: the store is masked with vmsif (bits 0..nul inclusive,
 * or all ones when the chunk has no NUL), so exactly the payload and its
 * terminating NUL are written; masked-off lanes do not access memory, so the
 * store can never touch a byte past the destination object or cross a page.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strcpy)
  mv      a2, a0                     /* return value */
  vsetvli t0, zero, e8, m8, ta, ma   /* t0 = VLMAX */
  neg     t1, t0
  and     t1, a1, t1
  add     t1, t1, t0
  sub     t1, t1, a1                 /* lead = bytes to a VLMAX boundary */
  beqz    t1, .Lcopy

  /* leading partial chunk: [a1, a1+lead) is inside one aligned block */
  vsetvli zero, t1, e8, m8, ta, ma
  vle8.v  v8, (a1)
  vmseq.vi v1, v8, 0
  vmsif.m v0, v1
  vfirst.m t2, v1
  vse8.v  v8, (a2), v0.t
  bgez    t2, .Lret
  add     a1, a1, t1
  add     a2, a2, t1

.Lcopy:
  vsetvli t0, zero, e8, m8, ta, ma
.Lloop:
  vle8.v  v8, (a1)                   /* full VLMAX, page-safe by alignment */
  vmseq.vi v1, v8, 0
  vmsif.m v0, v1                     /* 1 up to and including the NUL */
  vse8.v  v8, (a2), v0.t
  vfirst.m t2, v1
  add     a1, a1, t0
  add     a2, a2, t0
  bltz    t2, .Lloop
.Lret:
  ret
END(strcpy)
#endif
