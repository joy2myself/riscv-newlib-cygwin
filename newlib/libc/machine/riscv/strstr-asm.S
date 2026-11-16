/* strstr, RVV 1.0, v2: one-pass block filter, no candidate restart.
 *
 * char *strstr(const char *hay, const char *needle)
 *
 * The shipped version finds needle[0] with a vector scan and, after a failed
 * verification, restarts the whole vector scan at candidate+1: on a haystack
 * that is dense in needle[0] ("aaaa...a" with "aab") every byte costs a fresh
 * VLMAX-byte load + compare, i.e. O(n * VLMAX) instead of O(n).
 *
 * This version evaluates every byte of a VLMAX-aligned block exactly once:
 *
 *   mask = AND(j = 0..FB-1) (hay[i+j] == needle[j]),  FB = min(m, 8)
 *
 * A set bit means "the first FB needle bytes match here".  For FB == m that is
 * an exact match, so no verification and no advance-by-one happens at all:
 * the whole block is consumed in one pass.  For longer needles only the filter
 * hits are verified (scalar, stopping at a NUL, so the verifier never reads
 * past the haystack terminator).  Candidates at or after the haystack NUL are
 * removed with a vmsbf mask before they are looked at, so stray bytes after
 * the terminator can never produce a match.
 *
 * Page safety: blocks are VLMAX-aligned and VLMAX divides the page size, so
 * every load [B+j, B+j+W), W = VLMAX-FB+1, lies inside one page.  The last
 * FB-1 bytes of a block (not reachable by the shifted loads) are checked by a
 * scalar loop.  Both the scalar loop and the verifier stop at the NUL, and the
 * bytes that precede the haystack inside the first (aligned-down) block are
 * skipped, so no access can touch an unmapped page.
 *
 * Register discipline: mask destinations live in v0..v5, data groups in
 * v8/v16, so a mask result never overlaps a source vector group.
 */
#include <sys/asm.h>

#if defined(__riscv_vector) && !defined(__riscv_e) && !defined(__OPTIMIZE_SIZE__) && !defined(PREFER_SIZE_OVER_SPEED)

ENTRY(strstr)
  lbu     t0, 0(a1)                  /* needle[0] */
  beqz    t0, .Lreth                 /* empty needle -> haystack */
  li      t5, 1
  li      t6, 9
.Lmlen:
  add     t3, a1, t5
  lbu     t3, 0(t3)
  beqz    t3, .Lmshort
  addi    t5, t5, 1
  bltu    t5, t6, .Lmlen
  li      t5, 9                      /* needle longer than 8 */
  li      t6, 8
  j       .Lmknown
.Lmshort:
  mv      t6, t5                     /* FB = m, 1..8 */
.Lmknown:
  vsetvli a4, zero, e8, m8, ta, ma   /* a4 = VLMAX */
  sub     t3, a4, t6
  addi    t3, t3, 1                  /* W = VLMAX - FB + 1 */
  mv      a3, a0                     /* haystack base */
  neg     t1, a4
  and     a2, a0, t1                 /* B = align_down(hay, VLMAX) */
  beq     a2, a0, .Lsetmain          /* already aligned: plain main loop */
  /* Partial first block: rebase it on the haystack itself so that no lane
     precedes hay (no NUL-drop loop, no skip mask).  lead = B+VLMAX - hay and
     the filter window is lead-(FB-1), so hay+j+W' = B+VLMAX stays inside one
     page.  The FB-1 bytes before B+VLMAX are handled by the edge loop.  */
  add     a2, a2, a4                 /* a2 = next VLMAX boundary */
  sub     a4, a2, a0                 /* a4 = lead, 1..VLMAX-1 */
  sub     t3, a4, t6
  addi    t3, t3, 1                  /* W' = lead - FB + 1 */
  mv      a2, a0                     /* base = hay */
  bgtz    t3, .Lsetmain
  /* lead < FB: fewer than FB bytes before the boundary, scan them scalar */
  mv      t1, a4
.Lprescan:
  lbu     t2, 0(a2)
  beqz    t2, .Lnull
  beq     t2, t0, .Lprecand
.Lprenext:
  addi    a2, a2, 1
  addi    t1, t1, -1
  bnez    t1, .Lprescan
  j       .Lrestore
.Lprecand:
  mv      a0, a2
  addi    a5, a2, 1
  addi    a7, a1, 1
.Lprever:
  lbu     a6, 0(a7)
  beqz    a6, .Lret
  lbu     t2, 0(a5)
  bne     t2, a6, .Lprenext
  addi    a5, a5, 1
  addi    a7, a7, 1
  j       .Lprever
.Lsetmain:
  vsetvli zero, t3, e8, m8, ta, ma
  li      t4, 0                      /* no NUL seen yet in this block */

.Lblk:
  vle8.v   v8, (a2)
  vmseq.vx v0, v8, t0
  li       t1, 1
.Lfilter:
  bgeu     t1, t6, .Lfiltered
  add      t2, a2, t1
  vle8.v   v16, (t2)                 /* lane i = hay[B+i+j] */
  add      t2, a1, t1
  lbu      t2, 0(t2)                 /* needle[j] */
  vmseq.vx v4, v16, t2
  vmand.mm v0, v0, v4
  addi     t1, t1, 1
  j        .Lfilter
.Lfiltered:
  vmseq.vi v1, v8, 0
  vfirst.m a6, v1
  /* The block is aligned down, so the first block may contain bytes (and
     even a NUL) before the haystack.  Such a NUL is not the terminator.  */
.Lnul_loop:
  bltz     a6, .Lno_nul
  add      t2, a2, a6
  bgeu     t2, a3, .Lhave_nul
  vmsof.m  v4, v1
  vmxor.mm v1, v1, v4
  vfirst.m a6, v1
  j        .Lnul_loop
.Lhave_nul:
  li       t4, 1
  vmsof.m  v4, v1
  vmsbf.m  v5, v4                    /* lanes before the terminator */
  vmand.mm v0, v0, v5
  vfirst.m a5, v0
  bgez     a5, .Lcandloop
  li       a0, 0
  ret
.Lno_nul:
  vfirst.m a5, v0
  bltz     a5, .Ledge
  /* fall through */

.Lcandloop:
  add      a5, a2, a5                /* candidate address */
  bltu     a5, a3, .Ldrop            /* before the haystack: ignore */
  beq      t6, t5, .Lhit             /* FB == m: exact match */
  mv       a0, a5
  add      t1, a5, t6
  add      t2, a1, t6
.Lver:
  lbu      a7, 0(t2)
  beqz     a7, .Lret
  lbu      a5, 0(t1)
  bne      a5, a7, .Ldrop
  addi     t1, t1, 1
  addi     t2, t2, 1
  j        .Lver
.Ldrop:
  vmsof.m  v4, v0
  vmxor.mm v0, v0, v4
  vfirst.m a5, v0
  bgez     a5, .Lcandloop
  bnez     t4, .Lnull                /* terminator was ahead: not found */
  j        .Ledge

.Ledge:
  /* last FB-1 bytes of the block: scalar, in order */
  sub      t4, a4, t3                /* FB-1 */
  beqz     t4, .Lnext
  li       t1, 0
.Ledge_loop:
  add      t2, a2, t3
  add      t2, t2, t1
  bltu     t2, a3, .Ledge_next
  lbu      t2, 0(t2)
  beqz     t2, .Lnull
  beq      t2, t0, .Ledgever
.Ledge_next:
  addi     t1, t1, 1
  bltu     t1, t4, .Ledge_loop
  j        .Lnext
.Ledgever:
  add      a0, a2, t3
  add      a0, a0, t1                /* candidate start */
  addi     t2, a0, 1
  addi     a7, a1, 1
.Leve:
  lbu      a5, 0(a7)
  beqz     a5, .Lret
  lbu      t1, 0(t2)
  bne      t1, a5, .Ledgefail
  addi     t2, t2, 1
  addi     a7, a7, 1
  j        .Leve
.Ledgefail:
  sub      t1, a0, a2
  sub      t1, t1, t3
  addi     t1, t1, 1
  bltu     t1, t4, .Ledge_loop
  j        .Lnext
.Lnext:
  add      a2, a2, a4
.Lrestore:
  vsetvli a4, zero, e8, m8, ta, ma   /* after the partial block: VLMAX, W */
  sub      t3, a4, t6
  addi     t3, t3, 1
  vsetvli zero, t3, e8, m8, ta, ma
  j        .Lblk

.Lhit:
  mv       a0, a5
.Lret:
  ret
.Lnull:
  li       a0, 0
  ret
.Lreth:
  ret
END(strstr)
#endif
