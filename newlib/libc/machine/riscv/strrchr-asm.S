/* strrchr - RVV 1.0 candidate v2: forward vector scan, one bounded scalar
 * rescan of the last chunk that contained c.  No vle8ff.
 *
 *   char *strrchr (const char *s, int c)
 *
 * Algorithm
 *   1. Forward e8/m8 probe (65 B) then e8/m8 loop, exactly like strlen, but
 *      each chunk also computes the mask of bytes equal to (char)c and, if
 *      it has a hit at or before the terminator, remembers the chunk start
 *      and its length (a4/a5).
 *   2. When the terminator is found, only the *last remembered chunk* can
 *      hold the answer.  It is rescanned backwards with aligned/unaligned
 *      8-byte scalar words (xor + zero-byte trick + a high-to-low bit
 *      ladder).  That rescan is at most one vector chunk of work, so the
 *      call stays O(n) with a tiny constant and no second full pass.
 *
 *   c == 0 is handled naturally: the terminator itself is the last
 *   occurrence, so the rescan window includes it.
 *
 * Page safety: every forward load uses vl = min(bytes-to-page, VLMAX); the
 * backward rescan only touches [chunk_start, chunk_end) which the forward
 * load already proved to be inside one page.  Nothing before s is read.
 *
 * RV32: the scalar rescan is RV64-specific, so RV32 keeps the previous
 * vector-only implementation (in the #else branch) -- unchanged behaviour.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__riscv_e) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

#if __riscv_xlen == 64
/* highest byte index of a 0x80-group mask (mask != 0) */
.macro RZIDX dst, msk, tmp
  bltz \msk, .Lr7\@
  slli \tmp, \msk, 8
  bltz \tmp, .Lr6\@
  slli \tmp, \msk, 16
  bltz \tmp, .Lr5\@
  slli \tmp, \msk, 24
  bltz \tmp, .Lr4\@
  slli \tmp, \msk, 32
  bltz \tmp, .Lr3\@
  slli \tmp, \msk, 40
  bltz \tmp, .Lr2\@
  slli \tmp, \msk, 48
  bltz \tmp, .Lr1\@
  li \dst, 0
  j .Lre\@
.Lr7\@:
  li \dst, 7
  j .Lre\@
.Lr6\@:
  li \dst, 6
  j .Lre\@
.Lr5\@:
  li \dst, 5
  j .Lre\@
.Lr4\@:
  li \dst, 4
  j .Lre\@
.Lr3\@:
  li \dst, 3
  j .Lre\@
.Lr2\@:
  li \dst, 2
  j .Lre\@
.Lr1\@:
  li \dst, 1
.Lre\@:
.endm

/* load one chunk, update (a4,a5) if it holds a usable c.  Leaves the
   terminator index in t2 (-1 when none).  \lastlen is the chunk's vl. */
.macro SCAN_CHUNK lastlen
  vle8.v   v8, (a0)
  vmseq.vi v2, v8, 0
  vfirst.m t2, v2
  vmseq.vx v1, v8, a1
  vfirst.m t1, v1
  bltz     t1, .Lsc_no_c\@
  bltz     t2, .Lsc_c\@
  bgt      t1, t2, .Lsc_no_c\@
.Lsc_c\@:
  mv       a4, a0
  mv       a5, \lastlen
.Lsc_no_c\@:
.endm

ENTRY(strrchr)
  andi     a1, a1, 255
  li       a6, 4095
  li       a3, 4096
  li       a4, 0
  li       a5, 0
  li       t3, 65
  li       a2, 4096 - 65
  vsetvli  zero, t3, e8, m8, ta, ma
  and      t0, a0, a6
  bgeu     t0, a2, .Lprobe_part

  SCAN_CHUNK t3
  bgez     t2, .Lfound_nul
  addi     a0, a0, 65
  j        .Lbig_setup

.Lprobe_part:
  sub      t4, a3, t0
  vsetvli  zero, t4, e8, m8, ta, ma
  SCAN_CHUNK t4
  bgez     t2, .Lfound_nul
  add      a0, a0, t4
  j        .Lbig_setup

.Lbig_setup:
  vsetvli  t3, zero, e8, m8, ta, ma   /* t3 = VLMAX bytes               */
  li       a2, 4096
  sub      a2, a2, t3
.Lbig_loop:
  and      t0, a0, a6
  bgeu     t0, a2, .Lbig_part
  SCAN_CHUNK t3
  bgez     t2, .Lfound_nul
  add      a0, a0, t3
  j        .Lbig_loop

.Lbig_part:
  sub      t4, a3, t0
  vsetvli  zero, t4, e8, m8, ta, ma
  SCAN_CHUNK t4
  bgez     t2, .Lfound_nul
  add      a0, a0, t4
  vsetvli  t3, zero, e8, m8, ta, ma
  j        .Lbig_loop

  /* ---- terminator found: rescan the last c-chunk backwards ---------- */
.Lfound_nul:
  beqz     a4, .Lnull
  add      t3, a0, t2             /* terminator address                 */
  add      t4, a4, a5             /* end of the last c-chunk            */
  beq      a4, a0, 1f
  mv       t3, t4
  j        2f
1:
  bnez     a1, 2f                 /* c != 0: window ends at the NUL     */
  addi     t3, t3, 1              /* c == 0: include the NUL            */
2:
  /* exact per-byte zero mask (no borrow chain, the high-to-low search
     must not see the false positives the classic ONES/HIGHS formula
     leaves above a zero byte):  ~((x&0x7f) + 0x7f | x | 0x7f)          */
  li       t5, 0x7f7f7f7f
  slli     t6, t5, 32
  or       t5, t5, t6             /* 0x7f7f..7f                         */
  li       t0, 0x01010101
  slli     t1, t0, 32
  or       t0, t0, t1
  mul      a3, a1, t0             /* c splatted to all 8 bytes          */
  mv       a0, t3                 /* q = window end                     */

.Lbbyte:
  beq      a0, a4, .Lnull
  andi     t0, a0, 7
  beqz     t0, .Lbword
  addi     a0, a0, -1
  lbu      t0, 0(a0)
  bne      t0, a1, .Lbbyte
  ret

.Lbword:
  sub      t0, a0, a4
  li       t1, 8
  bltu     t0, t1, .Lbtail
  addi     a0, a0, -8
  ld       t0, 0(a0)
  xor      t0, t0, a3
  and      t1, t0, t5
  add      t1, t1, t5
  or       t1, t1, t0
  or       t1, t1, t5
  not      t1, t1
  beqz     t1, .Lbword
  RZIDX    t1, t1, t2
  add      a0, a0, t1
  ret

.Lbtail:
  beq      a0, a4, .Lnull
  addi     a0, a0, -1
  lbu      t0, 0(a0)
  bne      t0, a1, .Lbtail
  ret

.Lnull:
  li       a0, 0
  ret
END(strrchr)

#else /* RV32: keep the previous vector-only implementation */
ENTRY(strrchr)
  li a6, 0
.Lloop32:
  vsetvli a2, zero, e8, m1, ta, ma
  vle8ff.v v8, (a0)
  csrr a2, vl
  vmseq.vi v1, v8, 0
  vfirst.m a3, v1
  vmseq.vx v2, v8, a1
  vmsif.m v3, v1
  vmand.mm v0, v2, v3
  vfirst.m a4, v0
  bltz a4, .Lskip32
  vsetvli zero, a2, e16, m2, ta, ma
  vid.v v16
  vmv.s.x v24, x0
  vredmaxu.vs v24, v16, v24, v0.t
  vmv.x.s a5, v24
  add a6, a0, a5
.Lskip32:
  bgez a3, .Ldone32
  add a0, a0, a2
  j .Lloop32
.Ldone32:
  mv a0, a6
  ret
END(strrchr)
#endif

#endif /* __riscv_vector && !__riscv_e && !size-optimized */
