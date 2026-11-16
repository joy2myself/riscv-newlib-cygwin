/* strcat, RVV 1.0, v2.
 *
 * char *strcat(char *dst, const char *src)
 *
 * Phase 1 is strlen(dst) with VLMAX-aligned vle8.v loads (no vle8ff, no
 * per-chunk vsetvli/csrr); phase 2 is the strcpy-v2 loop at the NUL.
 * Both phases only ever read inside one page per load.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strcat)
  mv      a2, a0                     /* running dst */
  vsetvli t0, zero, e8, m8, ta, ma
  neg     t1, t0
  and     t1, a2, t1
  add     t1, t1, t0
  sub     t1, t1, a2                 /* lead */
  beqz    t1, .Lscan
  vsetvli zero, t1, e8, m8, ta, ma
  vle8.v  v8, (a2)
  vmseq.vi v1, v8, 0
  vfirst.m t2, v1
  bgez    t2, .Lfound
  add     a2, a2, t1
.Lscan:
  vsetvli t0, zero, e8, m8, ta, ma
.Lsloop:
  vle8.v  v8, (a2)
  vmseq.vi v1, v8, 0
  vfirst.m t2, v1
  add     a2, a2, t0
  bltz    t2, .Lsloop
  sub     a2, a2, t0
.Lfound:
  add     a2, a2, t2                 /* a2 -> terminating NUL of dst */

  /* phase 2: copy src (a1) to a2 */
  neg     t1, t0
  and     t1, a1, t1
  add     t1, t1, t0
  sub     t1, t1, a1                 /* lead */
  beqz    t1, .Lcopy
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
  vle8.v  v8, (a1)
  vmseq.vi v1, v8, 0
  vmsif.m v0, v1
  vse8.v  v8, (a2), v0.t
  vfirst.m t2, v1
  add     a1, a1, t0
  add     a2, a2, t0
  bltz    t2, .Lloop
.Lret:
  ret
END(strcat)
#endif
