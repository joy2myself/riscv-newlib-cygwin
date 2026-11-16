/* strchrnul - RVV 1.0 candidate v2: vector-first, no vle8ff.
 *
 *   char *strchrnul (const char *s, int c)
 *
 * Same scan as strchr-v2: 64-byte e8/m4 probe then an e8/m8 VLMAX loop, one
 * load + two compares + mask OR + vfirst.m per chunk.  The difference is
 * that the first occurrence of either c or the terminator is itself the
 * answer, so the hit path is just "return chunk + index" -- no byte reload.
 *
 * Page safety identical to strchr-v2: vl = min(bytes-to-page, VLMAX) for a
 * plain vle8.v, never vle8ff, never a load past the page or before s.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strchrnul)
  andi     a1, a1, 255
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
  ret

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
  ret

.Lbig_setup:
  vsetvli  t3, zero, e8, m8, ta, ma
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
  ret

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
  ret

.Lbig_restore:
  vsetvli  t3, zero, e8, m8, ta, ma
  j        .Lbig_loop
END(strchrnul)

#endif
