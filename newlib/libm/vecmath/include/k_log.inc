/* k_log.inc -- log / log2 / log10 / log1p / log2p1 kernels for libvecmath.

   LMUL-generic template: include after vecmath_rvvcompat.h,
   vecmath_dd.h and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.

   log vectorizes the ARM optimized-routines implementation that is live
   in newlib for rv64 hard-float (libm/common/log.c): table-gather
   reduction + polynomial, near-1 special polynomial, and a masked scalar
   fixup for x < 2^-1022 / negative / inf / NaN (covers divzero, invalid
   and subnormal normalization with exact scalar semantics and flags).
   log1p is the fdlibm s_log1p.c port.

   log2/log2p1 use a table core of their own (below): a 512-entry table
   of {10-bit reciprocal, log2(1/r) split as a double-double}, one FMA
   whose residual is exact, and a degree-7 polynomial in that exact
   residual -- plain double, no division and no double-double combine.
   log10/log2p1's 1/ln2 and log10's log10(e) scalings are folded into the
   core rather than applied as a second rounding afterwards.

   Data: own table copies (vm_data.h / tables/ / vm_log2_table.h), never
   libm symbols -- glibc and newlib order the struct fields differently.  */

#include <vm_data.h>
#include <vm_log2_table.h>

/* log near-1 region: bits(1-2^-4) .. bits(1+0x1.09p-4).  */
#define VM_LOG_LO	0x3fee000000000000ULL
#define VM_LOG_HILO	0x0003090000000000ULL
/* shared reduction offset (log.c/log2.c OFF).  */
#define VM_LOG_OFF	0x3fe6000000000000ULL

/* log10(e) and 1/ln2 as dd pairs (>= 106 bits).  */
#define VM_LOG10E_HI	4.34294481903251816668e-01
#define VM_LOG10E_LO	1.09831965021676507274e-17
#define VM_INVLN2_HI	1.44269504088896338700e+00
#define VM_INVLN2_LO	2.03552737409310331110e-17

/* ------------------------------------------------------------------ */
/* log(x) -- vectorization of libm/common/log.c (N=128, poly orders
   6 / 12).  fixup must be scalar log().

   VM_LOG_DD returns the result as a double-double pair (yh, yl): the
   main path's (hi, lo) decomposition already carries ~106 bits, so the
   dd tap is exact to ~2^-105 relative, which is what log10 scales by
   log10(e) before the final rounding.  */
#define VM_LOG_DD(yh, yl, x, vl, fixup) do {				\
  const struct vm_log_data *vmlgd__D = &__vecmath_log_data;		\
  vm_u64_t vmlgd__ix = VM_BITS (x);					\
  vm_u64_t vmlgd__top = VM_VSRL_VX (vmlgd__ix, 48, (vl));		\
  vm_b_t vmlgd__mspec = VM_VMSGEU_VX (				\
    VM_VSUB_VX (vmlgd__top, 0x10, (vl)), 0x7fe0, (vl));			\
  vm_b_t vmlgd__mnear = VM_VMSLTU_VX (					\
    VM_VSUB_VX (vmlgd__ix, VM_LOG_LO, (vl)), VM_LOG_HILO, (vl));	\
  vm_f_t vmlgd__xs = VM_VFMERGE_VFM ((x), 1.0, vmlgd__mspec, (vl));	\
  vm_f_t vmlgd__xn = VM_VFMERGE_VFM ((x), 1.0,				\
				     VM_VMNOT_M (vmlgd__mnear, (vl)), (vl)); \
  /* ---- near-1 polynomial (order 12) ---- */				\
  vm_f_t vmlgd__r = VM_VFSUB_VF (vmlgd__xn, 1.0, (vl));			\
  vm_f_t vmlgd__r2 = VM_VFMUL_VV (vmlgd__r, vmlgd__r, (vl));		\
  vm_f_t vmlgd__r3 = VM_VFMUL_VV (vmlgd__r, vmlgd__r2, (vl));		\
  vm_f_t vmlgd__q = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly1[7], (vl)), \
				  vmlgd__D->poly1[8], vmlgd__r, (vl));	\
  vmlgd__q = VM_VFMACC_VF (vmlgd__q, vmlgd__D->poly1[9], vmlgd__r2, (vl)); \
  vmlgd__q = VM_VFMACC_VF (vmlgd__q, vmlgd__D->poly1[10], vmlgd__r3, (vl)); \
  vm_f_t vmlgd__p = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly1[4], (vl)), \
				  vmlgd__D->poly1[5], vmlgd__r, (vl));	\
  vmlgd__p = VM_VFMACC_VF (vmlgd__p, vmlgd__D->poly1[6], vmlgd__r2, (vl)); \
  vmlgd__p = VM_VFMACC_VV (vmlgd__p, vmlgd__r3, vmlgd__q, (vl));	\
  vm_f_t vmlgd__P = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly1[1], (vl)), \
				  vmlgd__D->poly1[2], vmlgd__r, (vl));	\
  vmlgd__P = VM_VFMACC_VF (vmlgd__P, vmlgd__D->poly1[3], vmlgd__r2, (vl)); \
  vmlgd__P = VM_VFMACC_VV (vmlgd__P, vmlgd__r3, vmlgd__p, (vl));	\
  vm_f_t vmlgd__npoly = VM_VFMUL_VV (vmlgd__r3, vmlgd__P, (vl));	\
  vm_f_t vmlgd__w = VM_VFMUL_VF (vmlgd__r, 0x1p27, (vl));		\
  vm_f_t vmlgd__rhi = VM_VFSUB_VV (VM_VFADD_VV (vmlgd__r, vmlgd__w, (vl)), \
				   vmlgd__w, (vl));			\
  vm_f_t vmlgd__rlo = VM_VFSUB_VV (vmlgd__r, vmlgd__rhi, (vl));		\
  vmlgd__w = VM_VFMUL_VF (VM_VFMUL_VV (vmlgd__rhi, vmlgd__rhi, (vl)),	\
			  -0.5, (vl));					\
  vm_f_t vmlgd__nhi = VM_VFADD_VV (vmlgd__r, vmlgd__w, (vl));		\
  vm_f_t vmlgd__nlo = VM_VFADD_VV (VM_VFSUB_VV (vmlgd__r, vmlgd__nhi, (vl)), \
				   vmlgd__w, (vl));			\
  vmlgd__nlo = VM_VFADD_VV (vmlgd__nlo,				\
    VM_VFMUL_VV (VM_VFMUL_VF (vmlgd__rlo, -0.5, (vl)),			\
		 VM_VFADD_VV (vmlgd__rhi, vmlgd__r, (vl)), (vl)), (vl));	\
  /* near-1 dd = (nhi, nlo) + (npoly, 0).  */				\
  vm_f_t vmlgd__zh = VM_VFMV_V_F (0.0, (vl));				\
  vm_f_t vmlgd__dh, vmlgd__dl;						\
  VM_DD_ADD (vmlgd__dh, vmlgd__dl, vmlgd__nhi, vmlgd__nlo,		\
	     vmlgd__npoly, vmlgd__zh, (vl));				\
  /* ---- main path ---- */						\
  vm_u64_t vmlgd__ixb = VM_BITS (vmlgd__xs);				\
  vm_u64_t vmlgd__tmp = VM_VSUB_VX (vmlgd__ixb, VM_LOG_OFF, (vl));	\
  vm_u64_t vmlgd__off = VM_VSLL_VX (VM_VAND_VX (			\
    VM_VSRL_VX (vmlgd__tmp, 45, (vl)), 127, (vl)), 4, (vl));		\
  vm_f_t vmlgd__invc = VM_VLUXEI64 (&vmlgd__D->tab[0].invc, vmlgd__off, (vl)); \
  vm_f_t vmlgd__logc = VM_VLUXEI64 (&vmlgd__D->tab[0].logc, vmlgd__off, (vl)); \
  vm_i64_t vmlgd__k = VM_VSRA_VX (VM_REINT_U_I (vmlgd__tmp), 52, (vl));	\
  vm_u64_t vmlgd__iz = VM_VSUB_VV (vmlgd__ixb,				\
    VM_VAND_VX (vmlgd__tmp, 0xfff0000000000000ULL, (vl)), (vl));	\
  vm_f_t vmlgd__z = VM_FROM_BITS (vmlgd__iz);				\
  vm_f_t vmlgd__kd = VM_VFCVT_F_X_V (vmlgd__k, (vl));			\
  vmlgd__r = VM_VFMACC_VV (VM_VFMV_V_F (-1.0, (vl)),			\
			   vmlgd__invc, vmlgd__z, (vl));		\
  vmlgd__w = VM_VFMACC_VF (vmlgd__logc, vmlgd__D->ln2hi, vmlgd__kd, (vl)); \
  vm_f_t vmlgd__mhi = VM_VFADD_VV (vmlgd__w, vmlgd__r, (vl));		\
  vm_f_t vmlgd__mlo = VM_VFADD_VV (					\
    VM_VFADD_VV (VM_VFSUB_VV (vmlgd__w, vmlgd__mhi, (vl)), vmlgd__r, (vl)), \
    VM_VFMUL_VF (vmlgd__kd, vmlgd__D->ln2lo, (vl)), (vl));		\
  vmlgd__r2 = VM_VFMUL_VV (vmlgd__r, vmlgd__r, (vl));			\
  vmlgd__q = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly[3], (vl)),	\
			   vmlgd__D->poly[4], vmlgd__r, (vl));		\
  vmlgd__q = VM_VFMUL_VV (vmlgd__r2, vmlgd__q, (vl));			\
  vmlgd__p = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly[1], (vl)),	\
			   vmlgd__D->poly[2], vmlgd__r, (vl));		\
  vmlgd__p = VM_VFADD_VV (vmlgd__p, vmlgd__q, (vl));			\
  vmlgd__p = VM_VFMUL_VV (VM_VFMUL_VV (vmlgd__r, vmlgd__r2, (vl)),	\
			  vmlgd__p, (vl));				\
  vm_f_t vmlgd__pr = VM_VFADD_VV (					\
    VM_VFMUL_VF (vmlgd__r2, vmlgd__D->poly[0], (vl)), vmlgd__p, (vl));	\
  /* main dd = (mhi, mlo) + (pr, 0).  */				\
  vm_f_t vmlgd__eh, vmlgd__el;						\
  VM_DD_ADD (vmlgd__eh, vmlgd__el, vmlgd__mhi, vmlgd__mlo,		\
	     vmlgd__pr, vmlgd__zh, (vl));				\
  /* merge dd pairs, collapse, fixup special lanes.  */			\
  vm_f_t vmlgd__h = VM_VFMERGE_VVM (vmlgd__eh, vmlgd__dh, vmlgd__mnear, (vl)); \
  vm_f_t vmlgd__l = VM_VFMERGE_VVM (vmlgd__el, vmlgd__dl, vmlgd__mnear, (vl)); \
  vm_f_t vmlgd__yc;							\
  VM_DD_COLLAPSE (vmlgd__yc, vmlgd__h, vmlgd__l, (vl));			\
  (yh) = VM_FIXUP1 ((x), vmlgd__yc, vmlgd__mspec, (vl), (fixup)); \
  (yl) = VM_VFMERGE_VFM (vmlgd__l, 0.0, vmlgd__mspec, (vl));		\
} while (0)

/* ------------------------------------------------------------------ */
/* Lean plain-double log (Arm log.c main path + near-1 path, no dd).
   The dd form (VM_LOG_DD) is kept above for log10 and for
   k_gamma/k_hyperinv, which need the extra bits.  This one is the
   'false position' the -O2 scalar log is: the near-1 polynomial and the
   table path share the same 0.5-0.6 ulp budget, so the final rounding is
   all that is left to do.  */
/* Plain-double log core (Arm log.c main + near-1 path), no fixup: the
   caller gets the value and the special-lane mask, and applies the
   scalar fallback itself.  Shared by log, log2 and log2p1.  */
#define VM_LOG_PLAIN(yoh, yolo, mo, x, vl) do {				\
  const struct vm_log_data *vlD__ = &__vecmath_log_data;		\
  vm_f_t vlx__ = (x);							\
  vm_u64_t vlix__ = VM_BITS (vlx__);					\
  vm_u64_t vltop__ = VM_VSRL_VX (vlix__, 48, (vl));			\
  vm_b_t vlmsp__ = VM_VMSGEU_VX (VM_VSUB_VX (vltop__, 0x10, (vl)),	\
				 0x7fe0, (vl));				\
  vm_b_t vlmne__ = VM_VMSLTU_VX (VM_VSUB_VX (vlix__, VM_LOG_LO, (vl)),	\
				 VM_LOG_HILO, (vl));			\
  vm_f_t vlxs__ = VM_VFMERGE_VFM (vlx__, 1.0, vlmsp__, (vl));		\
  vm_f_t vlxn__ = VM_VFMERGE_VFM (vlx__, 1.0,				\
				  VM_VMNOT_M (vlmne__, (vl)), (vl));	\
  vm_f_t vlyn__, vlylo__;						\
  {									\
    /* ---- near-1 path ---- */						\
    vm_f_t r = VM_VFSUB_VF (vlxn__, 1.0, (vl));				\
    vm_f_t r2 = VM_VFMUL_VV (r, r, (vl));				\
    vm_f_t r3 = VM_VFMUL_VV (r, r2, (vl));				\
    vm_f_t b = VM_VFMACC_VF (VM_VFMV_V_F (vlD__->poly1[7], (vl)),	\
			     vlD__->poly1[8], r, (vl));			\
    b = VM_VFMACC_VF (b, vlD__->poly1[9], r2, (vl));			\
    b = VM_VFMACC_VF (b, vlD__->poly1[10], r3, (vl));			\
    vm_f_t a = VM_VFMACC_VF (VM_VFMV_V_F (vlD__->poly1[4], (vl)),	\
			     vlD__->poly1[5], r, (vl));			\
    a = VM_VFMACC_VF (a, vlD__->poly1[6], r2, (vl));			\
    a = VM_VFMACC_VV (a, r3, b, (vl));					\
    vm_f_t pp = VM_VFMACC_VF (VM_VFMV_V_F (vlD__->poly1[1], (vl)),	\
			      vlD__->poly1[2], r, (vl));		\
    pp = VM_VFMACC_VF (pp, vlD__->poly1[3], r2, (vl));			\
    pp = VM_VFMACC_VV (pp, r3, a, (vl));				\
    vlyn__ = VM_VFMUL_VV (r3, pp, (vl));				\
    vm_f_t w = VM_VFMUL_VF (r, 0x1p27, (vl));				\
    vm_f_t rhi = VM_VFSUB_VV (VM_VFADD_VV (r, w, (vl)), w, (vl));	\
    vm_f_t rlo = VM_VFSUB_VV (r, rhi, (vl));				\
    w = VM_VFMUL_VF (VM_VFMUL_VV (rhi, rhi, (vl)),			\
		     vlD__->poly1[0], (vl));				\
    vm_f_t hi = VM_VFADD_VV (r, w, (vl));				\
    vm_f_t lo = VM_VFADD_VV (VM_VFSUB_VV (r, hi, (vl)), w, (vl));	\
    lo = VM_VFADD_VV (lo, VM_VFMUL_VF (VM_VFMUL_VV (			\
      rlo, VM_VFADD_VV (rhi, r, (vl)), (vl)), vlD__->poly1[0], (vl)),	\
      (vl));								\
    {									\
      /* track the last two roundings: y = (npoly + lo) + hi */		\
      vm_f_t vls1__ = VM_VFADD_VV (vlyn__, lo, (vl));			\
      vm_f_t vle1__ = VM_VFADD_VV (					\
	VM_VFSUB_VV (vlyn__, vls1__, (vl)), lo, (vl));			\
      vlyn__ = VM_VFADD_VV (vls1__, hi, (vl));				\
      vlylo__ = VM_VFADD_VV (						\
	VM_VFADD_VV (VM_VFSUB_VV (vls1__, vlyn__, (vl)), hi, (vl)),	\
	vle1__, (vl));							\
    }									\
    /* ---- main path ---- */						\
    vm_u64_t ixb = VM_BITS (vlxs__);					\
    vm_u64_t tmp = VM_VSUB_VX (ixb, VM_LOG_OFF, (vl));			\
    vm_u64_t off = VM_VSLL_VX (VM_VAND_VX (				\
      VM_VSRL_VX (tmp, 45, (vl)), 127, (vl)), 4, (vl));			\
    vm_f_t invc = VM_VLUXEI64 (&vlD__->tab[0].invc, off, (vl));		\
    vm_f_t logc = VM_VLUXEI64 (&vlD__->tab[0].logc, off, (vl));		\
    vm_i64_t k = VM_VSRA_VX (VM_REINT_U_I (tmp), 52, (vl));		\
    vm_u64_t iz = VM_VSUB_VV (ixb,						\
      VM_VAND_VX (tmp, 0xfff0000000000000ULL, (vl)), (vl));		\
    vm_f_t z = VM_FROM_BITS (iz);					\
    vm_f_t kd = VM_VFCVT_F_X_V (k, (vl));				\
    r = VM_VFMACC_VV (VM_VFMV_V_F (-1.0, (vl)), invc, z, (vl));		\
    w = VM_VFMACC_VF (logc, vlD__->ln2hi, kd, (vl));			\
    hi = VM_VFADD_VV (w, r, (vl));					\
    lo = VM_VFADD_VV (VM_VFADD_VV (VM_VFSUB_VV (w, hi, (vl)), r, (vl)),	\
		      VM_VFMUL_VF (kd, vlD__->ln2lo, (vl)), (vl));	\
    r2 = VM_VFMUL_VV (r, r, (vl));					\
    {									\
      vm_f_t u = VM_VFMACC_VF (VM_VFMV_V_F (vlD__->poly[3], (vl)),	\
			       vlD__->poly[4], r, (vl));		\
      vm_f_t q = VM_VFMACC_VF (VM_VFMV_V_F (vlD__->poly[1], (vl)),	\
			       vlD__->poly[2], r, (vl));		\
      q = VM_VFADD_VV (q, VM_VFMUL_VV (r2, u, (vl)), (vl));		\
      vm_f_t pr = VM_VFADD_VV (VM_VFMUL_VF (r2, vlD__->poly[0], (vl)),	\
	VM_VFMUL_VV (VM_VFMUL_VV (r, r2, (vl)), q, (vl)), (vl));	\
      /* y = (lo + pr) + hi, tracked */				\
      vm_f_t vlt__, vlte__;						\
      VM_TWO_SUM (vlt__, vlte__, lo, pr, (vl));				\
      vm_f_t ym = VM_VFADD_VV (vlt__, hi, (vl));			\
      {									\
	vm_f_t vlye__ = VM_VFADD_VV (					\
	  VM_VFADD_VV (VM_VFSUB_VV (vlt__, ym, (vl)), hi, (vl)),	\
	  vlte__, (vl));						\
	vm_f_t vlmh__ = vlyn__;						\
	vm_f_t vlml__ = vlylo__;					\
	vlyn__ = VM_VFMERGE_VVM (ym, vlmh__, vlmne__, (vl));		\
	vlylo__ = VM_VFMERGE_VVM (vlye__, vlml__, vlmne__, (vl));	\
      }									\
    }									\
  }									\
  (yoh) = vlyn__;							\
  (yolo) = vlylo__;							\
  (mo) = vlmsp__;							\
} while (0)

static inline vm_f_t
VM_KFN (log_k) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_f_t yh, yl, y;
  vm_b_t mspec;
  VM_LOG_PLAIN (yh, yl, mspec, x, vl);
  y = VM_VFADD_VV (yh, yl, (vl));
  return VM_FIXUP1 (x, y, mspec, vl, fixup);
}

/* ------------------------------------------------------------------ */
/* Table-driven log2 core: plain double, one exact FMA residual, no
   division, no double-double combine.

   x = 2^k * m with m in [1,2).  The top 9 mantissa bits select a table
   entry {r, l1, l2}: r is a <= 10-bit approximation of 1/m over that
   slice and l1 + l2 = log2(1/r) split so that l1 is a multiple of
   2^-40 (the pair carries ~93 bits).  Then

     z = fma (r, m, -1)          EXACT: r has <= 10 significant bits,
                                 so r*m - 1 is a multiple of 2^-62 of
                                 magnitude <= 2^-8, i.e. <= 53 bits
     log2(x) = (k + l1) + p      p = log2(1+z) to ~0.55 ulp

   so one fma plus a degree-7 polynomial in z does the whole reduction:

     p = w*C1 + w^2*Q(w)         (C1 = 1/ln2)

   with Q of degree 5 in w (Estrin), and l2 folded into the w^2 term.
   The coarse part k + l1 is EXACT for every exponent (l1 is a multiple
   of 2^-40, |k| <= 1024), so only the final rounding is left.  Table
   entries 0/1 use r = 1 and 509..511 use r = 1/2, both exact powers of
   two: z = m-1 / z = m/2-1 is then exact by Sterbenz and l1 = 0 / 1
   exactly, so x near 1 (and just below a power of two) carries the
   whole answer in p with full relative precision instead of cancelling
   against a coarse table value.

   Yields the pair hi = k + l1 (exact coarse part), lo = p; the caller
   rounds hi + lo once (log2) or folds one more term into l2 first
   (log2p1, next).  */

/* log2p1's c/u tail, in log2 units, folded into the polynomial's
   constant term (l2) so that the answer still gets exactly one rounding
   -- adding it after the core would re-round a value of the answer's own
   size whenever x is tiny.  cu = c/(u*ln2) with 1/u = 2^-j * r * (1-z),
   where u = 2^j*m, r is the core's table reciprocal and z = r*m-1 its
   exact residual.  r*(1-z) is 1/m to z^2 ~ 2^-16 relative, and r itself
   is exact for the r = 1 / r = 1/2 table entries and within 2^-10
   elsewhere -- far inside the ~|x| relative accuracy the tail needs: the
   tail is at most 2^-52.5 absolute, so it only matters when |x| is
   small, and there u ~ 1 and r*(1-z) = 1/(1+z) to z^2.  */
static inline vm_f_t
VM_KFN (log2p1_tail) (vm_f_t c, vm_f_t r, vm_f_t z, vm_u64_t ub, size_t vl)
{
  vm_u64_t vlt_i2e = VM_VSUB_VV (VM_VMV_V_X (0x7fe0000000000000ULL, (vl)),	\
    VM_VAND_VX (ub, 0x7ff0000000000000ULL, (vl)), (vl));
  vm_f_t vlt_t = VM_VFMUL_VV (c, VM_FROM_BITS (vlt_i2e), (vl));
  vlt_t = VM_VFMUL_VV (vlt_t, r, (vl));
  vlt_t = VM_VFMUL_VV (vlt_t, VM_VFRSUB_VF (z, 1.0, (vl)), (vl));
  /* 1/ln2 to a single double: its 0.064 ulp bias only shows up on
     near-tie lanes, and correcting it with the lo part would underflow
     (and raise a spurious flag) whenever the tail itself is subnormal.  */
  return VM_VFMUL_VF (vlt_t, VM_L2T_C1, (vl));
}

#define VM_L2T_L2_PLAIN							\
  VM_VLUXEI64 (&__vm_log2_tab[0].l2, vl2t_off, (vl))
#define VM_L2T_L2_TAIL(c, r, z, ub)					\
  VM_VFADD_VV (VM_L2T_L2_PLAIN,						\
	       VM_KFN (log2p1_tail) ((c), (r), (z), (ub), (vl)), (vl))

#define VM_L2T_CORE(yh, yl, x, L2EXPR, vl) do {				\
  vm_u64_t vl2t_b = VM_BITS (x);					\
  vm_u64_t vl2t_off = VM_VAND_VX (VM_VSRL_VX (vl2t_b, 38, (vl)),	\
				  0x3fe0, (vl));			\
  vm_f_t vl2t_m = VM_FROM_BITS (VM_VOR_VX (				\
    VM_VAND_VX (vl2t_b, 0x000fffffffffffffULL, (vl)),			\
    0x3ff0000000000000ULL, (vl)));					\
  vm_i64_t vl2t_k = VM_REINT_U_I (VM_VSUB_VX (				\
    VM_VSRL_VX (vl2t_b, 52, (vl)), 1023, (vl)));			\
  vm_f_t vl2t_r = VM_VLUXEI64 (&__vm_log2_tab[0].r, vl2t_off, (vl));	\
  vm_f_t vl2t_l1 = VM_VLUXEI64 (&__vm_log2_tab[0].l1, vl2t_off, (vl));	\
  vm_f_t vl2t_w = VM_VFMACC_VV (VM_VFMV_V_F (-1.0, (vl)),		\
				vl2t_r, vl2t_m, (vl));			\
  vm_f_t vl2t_l2 = (L2EXPR);						\
  vm_f_t vl2t_w2 = VM_VFMUL_VV (vl2t_w, vl2t_w, (vl));			\
  vm_f_t vl2t_a0 = VM_VFMACC_VF (					\
    VM_VFMV_V_F (__vm_log2_tab_poly[0], (vl)),				\
    __vm_log2_tab_poly[1], vl2t_w, (vl));				\
  vm_f_t vl2t_a1 = VM_VFMACC_VF (					\
    VM_VFMV_V_F (__vm_log2_tab_poly[2], (vl)),				\
    __vm_log2_tab_poly[3], vl2t_w, (vl));				\
  vm_f_t vl2t_a2 = VM_VFMACC_VF (					\
    VM_VFMV_V_F (__vm_log2_tab_poly[4], (vl)),				\
    __vm_log2_tab_poly[5], vl2t_w, (vl));				\
  vl2t_a0 = VM_VFMACC_VV (vl2t_a0, vl2t_w2, vl2t_a1, (vl));		\
  vl2t_a2 = VM_VFMACC_VV (vl2t_a0, VM_VFMUL_VV (vl2t_w2, vl2t_w2, (vl)), \
			  vl2t_a2, (vl));				\
  vl2t_a2 = VM_VFMACC_VV (vl2t_l2, vl2t_w2, vl2t_a2, (vl));		\
  (yl) = VM_VFMACC_VF (vl2t_a2, VM_L2T_C1, vl2t_w, (vl));		\
  (yh) = VM_VFADD_VV (VM_VFCVT_F_X_V (vl2t_k, (vl)), vl2t_l1, (vl));	\
} while (0)

#define VM_LOG2_TAB_CORE(yh, yl, x, vl)					\
  VM_L2T_CORE (yh, yl, x, VM_L2T_L2_PLAIN, vl)

static inline vm_f_t
VM_KFN (log2_k) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_f_t yh, yl;
  vm_u64_t top16 = VM_VSRL_VX (VM_BITS (x), 48, (vl));
  vm_b_t mspec = VM_VMSGEU_VX (VM_VSUB_VX (top16, 0x10, (vl)), 0x7fe0, (vl));
  VM_LOG2_TAB_CORE (yh, yl, x, vl);
  return VM_FIXUP1 (x, VM_VFADD_VV (yh, yl, (vl)), mspec, vl, fixup);
}

/* ------------------------------------------------------------------ */
/* log1p(x) -- vectorization of fdlibm common/s_log1p.c.  All branches
   are closed-form; x <= -1 goes through scalar fixup (divzero at -1,
   invalid below), inf/nan computed as x+x inline.  */
#define VM_LN2_HI_L1P 6.93147180369123816490e-01
#define VM_LN2_LO_L1P 1.90821492927058770002e-10
#define VM_LP1 6.666666666666735130e-01
#define VM_LP2 3.999999999940941908e-01
#define VM_LP3 2.857142874366239149e-01
#define VM_LP4 2.222219843214978396e-01
#define VM_LP5 1.818357216161805012e-01
#define VM_LP6 1.531383769920937332e-01
#define VM_LP7 1.479819860511658591e-01

/* Horner: Lp1 + z*(Lp2 + z*(... z*Lp7)); each step is p <- Lp_k + z*p
   (vfmacc with the coefficient as accumulator, not p + z*Lp_k).  */
static inline vm_f_t
VM_KFN (log1p_poly) (vm_f_t z, size_t vl)
{
  vm_f_t p = VM_VFMV_V_F (VM_LP7, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP6, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP5, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP4, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP3, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP2, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP1, vl), z, p, vl);
  return p;
}

/* log1p with double-double output.  The fdlibm chain
   k*ln2_hi - ((hfsq - (core + t1)) - f) is not accurate enough as a
   plain double for log2p1, which scales by 1/ln2 and would inflate
   past 1 ulp in the same-binade case; as dd it stays ~2^-105
   relative.  */
#define VM_LOG1P_DD(yh, yl, x, vl, fixup) do {				\
  vm_u64_t vmlp__ix = VM_BITS (x);					\
  vm_i64_t vmlp__hx = VM_VSRA_VX (VM_REINT_U_I (vmlp__ix), 32, (vl));	\
  vm_u64_t vmlp__ab = VM_VAND_VX (vmlp__ix, VM_ABSMASK_U64, (vl));	\
  vm_b_t vmlp__neg = VM_VMSNE_VX (					\
    VM_VAND_VX (vmlp__ix, VM_SIGNBIT_U64, (vl)), 0, (vl));		\
  vm_b_t vmlp__dom = VM_VMAND_MM (vmlp__neg,				\
    VM_VMSGEU_VX (vmlp__ab, 0x3ff0000000000000ULL, (vl)), (vl));	\
  vm_b_t vmlp__infnan = VM_VMSGE_VX (vmlp__hx, (int64_t) 0x7ff00000, (vl)); \
  vm_b_t vmlp__lt54 = VM_VMSLTU_VX (vmlp__ab,				\
    0x3c90000000000000ULL, (vl));					\
  vm_b_t vmlp__lt29 = VM_VMSLTU_VX (vmlp__ab,				\
    0x3e20000000000000ULL, (vl));					\
  vmlp__lt29 = VM_VMAND_MM (vmlp__lt29,					\
    VM_VMNOT_M (vmlp__lt54, (vl)), (vl));				\
  vm_b_t vmlp__reg1 = VM_VMSLT_VX (vmlp__hx, (int64_t) 0x3fda827a, (vl)); \
  vm_b_t vmlp__mid = VM_VMAND_MM (vmlp__reg1,				\
    VM_VMOR_MM (VM_VMSGT_VX (vmlp__hx, 0, (vl)),			\
      VM_VMSLE_VX (vmlp__hx, (int64_t) (int32_t) 0xbfd2bec3, (vl)), (vl)), \
    (vl));								\
  vmlp__mid = VM_VMAND_MM (vmlp__mid, VM_VMNOT_M (			\
    VM_VMOR_MM (vmlp__lt29, vmlp__lt54, (vl)), (vl)), (vl));		\
  vm_b_t vmlp__k = VM_VMNOT_M (VM_VMOR_MM (				\
    VM_VMOR_MM (vmlp__dom, vmlp__infnan, (vl)),				\
    VM_VMOR_MM (VM_VMOR_MM (vmlp__lt54, vmlp__lt29, (vl)),		\
		vmlp__mid, (vl)), (vl)), (vl));				\
  vm_f_t vmlp__zh = VM_VFMV_V_F (0.0, (vl));				\
  /* dd accumulators, default = x (tiny region, exact).  */		\
  vm_f_t vmlp__rh = (x), vmlp__rl = vmlp__zh;				\
  /* ---- tiny2: x - x^2/2 via two_prod ---- */				\
  vm_f_t vmlp__x29 = VM_VFMERGE_VFM ((x), 0.0,				\
    VM_VMNOT_M (vmlp__lt29, (vl)), (vl));				\
  vm_f_t vmlp__ph, vmlp__pl;						\
  VM_TWO_PROD (vmlp__ph, vmlp__pl, vmlp__x29,				\
	       VM_VFMUL_VF (vmlp__x29, 0.5, (vl)), (vl));		\
  vm_f_t vmlp__th, vmlp__tl;						\
  VM_DD_SUB (vmlp__th, vmlp__tl, vmlp__x29, vmlp__zh,			\
	     vmlp__ph, vmlp__pl, (vl));					\
  vmlp__rh = VM_VFMERGE_VVM (vmlp__rh, vmlp__th, vmlp__lt29, (vl));	\
  vmlp__rl = VM_VFMERGE_VVM (vmlp__rl, vmlp__tl, vmlp__lt29, (vl));	\
  /* ---- shared core for mid (k=0) and k paths ---- */			\
  /* mid: f = x sanitized to 0.1 off-region.  */			\
  vm_f_t vmlp__xm = VM_VFMERGE_VFM ((x), 0.1,				\
    VM_VMNOT_M (vmlp__mid, (vl)), (vl));				\
  vm_f_t vmlp__hf_m = VM_VFMUL_VF (VM_VFMUL_VV (vmlp__xm, vmlp__xm, (vl)), \
				   0.5, (vl));				\
  vm_f_t vmlp__sm = VM_VFDIV_VV (vmlp__xm,				\
    VM_VFADD_VF (vmlp__xm, 2.0, (vl)), (vl));				\
  vm_f_t vmlp__zm = VM_VFMUL_VV (vmlp__sm, vmlp__sm, (vl));		\
  vm_f_t vmlp__Rm = VM_VFMUL_VV (vmlp__zm,				\
    VM_KFN (log1p_poly) (vmlp__zm, (vl)), (vl));			\
  vm_f_t vmlp__cm = VM_VFMUL_VV (vmlp__sm,				\
    VM_VFADD_VV (vmlp__hf_m, vmlp__Rm, (vl)), (vl));			\
  /* dd: f - (hfsq - core).  */						\
  VM_DD_SUB (vmlp__th, vmlp__tl, vmlp__hf_m, vmlp__zh,			\
	     vmlp__cm, vmlp__zh, (vl));					\
  vm_f_t vmlp__mh, vmlp__ml;						\
  VM_DD_SUB (vmlp__mh, vmlp__ml, vmlp__xm, vmlp__zh,			\
	     vmlp__th, vmlp__tl, (vl));					\
  vmlp__rh = VM_VFMERGE_VVM (vmlp__rh, vmlp__mh, vmlp__mid, (vl));	\
  vmlp__rl = VM_VFMERGE_VVM (vmlp__rl, vmlp__ml, vmlp__mid, (vl));	\
  /* ---- k path ---- */						\
  vm_f_t vmlp__xk = VM_VFMERGE_VFM ((x), 1.0,				\
    VM_VMNOT_M (vmlp__k, (vl)), (vl));					\
  vm_b_t vmlp__big = VM_VMSGE_VX (VM_VSRA_VX (				\
    VM_REINT_U_I (VM_BITS (vmlp__xk)), 32, (vl)),			\
    (int64_t) 0x43400000, (vl));					\
  vm_f_t vmlp__u = VM_VFMERGE_VVM (					\
    VM_VFADD_VF (vmlp__xk, 1.0, (vl)), vmlp__xk, vmlp__big, (vl));	\
  vm_u64_t vmlp__ub = VM_BITS (vmlp__u);				\
  vm_i64_t vmlp__ki = VM_REINT_U_I (VM_VSUB_VX (			\
    VM_VSRL_VX (vmlp__ub, 52, (vl)), 1023, (vl)));			\
  vm_f_t vmlp__c1 = VM_VFRSUB_VF (					\
    VM_VFSUB_VV (vmlp__u, vmlp__xk, (vl)), 1.0, (vl));			\
  vm_f_t vmlp__c2 = VM_VFSUB_VV (vmlp__xk,				\
    VM_VFSUB_VF (vmlp__u, 1.0, (vl)), (vl));				\
  vm_f_t vmlp__c = VM_VFMERGE_VVM (vmlp__c2, vmlp__c1,			\
    VM_VMSGT_VX (vmlp__ki, 0, (vl)), (vl));				\
  vmlp__c = VM_VFMERGE_VFM (vmlp__c, 0.0, vmlp__big, (vl));		\
  vmlp__c = VM_VFDIV_VV (vmlp__c, vmlp__u, (vl));			\
  vm_u64_t vmlp__huf = VM_VAND_VX (VM_VSRL_VX (vmlp__ub, 32, (vl)),	\
				   0xfffff, (vl));			\
  vm_b_t vmlp__hus = VM_VMSLTU_VX (vmlp__huf, 0x6a09e, (vl));		\
  vm_u64_t vmlp__low = VM_VAND_VX (vmlp__ub, 0xffffffffULL, (vl));	\
  vm_u64_t vmlp__ubn = VM_VMERGE_VVM (					\
    VM_VOR_VV (vmlp__low, VM_VSLL_VX (				\
      VM_VOR_VX (vmlp__huf, 0x3fe00000, (vl)), 32, (vl)), (vl)),	\
    VM_VOR_VV (vmlp__low, VM_VSLL_VX (				\
      VM_VOR_VX (vmlp__huf, 0x3ff00000, (vl)), 32, (vl)), (vl)),	\
    vmlp__hus, (vl));							\
  vmlp__u = VM_FROM_BITS (vmlp__ubn);					\
  vmlp__ki = VM_REINT_U_I (VM_VMERGE_VVM (				\
    VM_VADD_VX (VM_REINT_I_U (vmlp__ki), 1, (vl)),			\
    VM_REINT_I_U (vmlp__ki), vmlp__hus, (vl)));				\
  vm_f_t vmlp__f = VM_VFSUB_VF (vmlp__u, 1.0, (vl));			\
  vm_f_t vmlp__hf = VM_VFMUL_VF (VM_VFMUL_VV (vmlp__f, vmlp__f, (vl)),	\
				 0.5, (vl));				\
  vm_u64_t vmlp__huf2 = VM_VMERGE_VVM (VM_VSRL_VX (			\
    VM_VSUB_VV (VM_VMV_V_X (0x100000, (vl)), vmlp__huf, (vl)), 2, (vl)), \
    vmlp__huf, vmlp__hus, (vl));					\
  vm_f_t vmlp__kd = VM_VFCVT_F_X_V (vmlp__ki, (vl));			\
  /* t1dd = c + kd*ln2_lo (dd).  */					\
  vm_f_t vmlp__t1h, vmlp__t1l;						\
  VM_TWO_PROD (vmlp__t1h, vmlp__t1l, vmlp__kd,				\
	       VM_VFMV_V_F (VM_LN2_LO_L1P, (vl)), (vl));		\
  vm_f_t vmlp__t1dh, vmlp__t1dl;					\
  VM_DD_ADD (vmlp__t1dh, vmlp__t1dl, vmlp__c, vmlp__zh,			\
	     vmlp__t1h, vmlp__t1l, (vl));				\
  vm_f_t vmlp__khi = VM_VFMUL_VF (vmlp__kd, VM_LN2_HI_L1P, (vl));	\
  vm_b_t vmlp__hu0 = VM_VMSEQ_VX (vmlp__huf2, 0, (vl));			\
  vm_b_t vmlp__f0 = VM_VMFEQ_VF (vmlp__f, 0.0, (vl));			\
  vm_b_t vmlp__k0 = VM_VMSEQ_VX (VM_REINT_I_U (vmlp__ki), 0, (vl));	\
  /* ---- branch A (hu == 0) ---- */					\
  vm_f_t vmlp__Ra = VM_VFMUL_VV (vmlp__hf, VM_VFNMSAC_VF (		\
    VM_VFMV_V_F (1.0, (vl)), 0.66666666666666663, vmlp__f, (vl)), (vl)); \
  /* yA_k = khi - ((Ra - t1) - f)  [all dd] */				\
  vm_f_t vmlp__ah, vmlp__al;						\
  VM_DD_SUB (vmlp__ah, vmlp__al, vmlp__Ra, vmlp__zh,			\
	     vmlp__t1dh, vmlp__t1dl, (vl));				\
  VM_DD_SUB (vmlp__ah, vmlp__al, vmlp__ah, vmlp__al,			\
	     vmlp__f, vmlp__zh, (vl));					\
  vm_f_t vmlp__yakh, vmlp__yakl;					\
  VM_DD_SUB (vmlp__yakh, vmlp__yakl, vmlp__khi, vmlp__zh,		\
	     vmlp__ah, vmlp__al, (vl));					\
  /* yA_k0 = f - Ra */							\
  vm_f_t vmlp__ya0h, vmlp__ya0l;					\
  VM_DD_SUB (vmlp__ya0h, vmlp__ya0l, vmlp__f, vmlp__zh,			\
	     vmlp__Ra, vmlp__zh, (vl));					\
  /* yA_f0 = khi + t1 (k != 0), else 0 */				\
  vm_f_t vmlp__yafh, vmlp__yafl;					\
  VM_DD_ADD (vmlp__yafh, vmlp__yafl, vmlp__khi, vmlp__zh,		\
	     vmlp__t1dh, vmlp__t1dl, (vl));				\
  vmlp__yafh = VM_VFMERGE_VFM (vmlp__yafh, 0.0, vmlp__k0, (vl));	\
  vmlp__yafl = VM_VFMERGE_VFM (vmlp__yafl, 0.0, vmlp__k0, (vl));	\
  vm_f_t vmlp__yah = VM_VFMERGE_VVM (vmlp__yakh, vmlp__ya0h,		\
				     vmlp__k0, (vl));			\
  vm_f_t vmlp__yal = VM_VFMERGE_VVM (vmlp__yakl, vmlp__ya0l,		\
				     vmlp__k0, (vl));			\
  vmlp__yah = VM_VFMERGE_VVM (vmlp__yah, vmlp__yafh, vmlp__f0, (vl));	\
  vmlp__yal = VM_VFMERGE_VVM (vmlp__yal, vmlp__yafl, vmlp__f0, (vl));	\
  /* ---- branch B (hu != 0) ---- */					\
  vm_f_t vmlp__s = VM_VFDIV_VV (vmlp__f,				\
    VM_VFADD_VF (vmlp__f, 2.0, (vl)), (vl));				\
  vm_f_t vmlp__z = VM_VFMUL_VV (vmlp__s, vmlp__s, (vl));		\
  vm_f_t vmlp__R = VM_VFMUL_VV (vmlp__z,				\
    VM_KFN (log1p_poly) (vmlp__z, (vl)), (vl));				\
  vm_f_t vmlp__core = VM_VFMUL_VV (vmlp__s,				\
    VM_VFADD_VV (vmlp__hf, vmlp__R, (vl)), (vl));			\
  /* inner = (hfsq - (core + t1)) - f ; y = khi - inner */		\
  vm_f_t vmlp__bh, vmlp__bl;						\
  VM_DD_ADD (vmlp__bh, vmlp__bl, vmlp__core, vmlp__zh,			\
	     vmlp__t1dh, vmlp__t1dl, (vl));				\
  VM_DD_SUB (vmlp__bh, vmlp__bl, vmlp__hf, vmlp__zh,			\
	     vmlp__bh, vmlp__bl, (vl));					\
  VM_DD_SUB (vmlp__bh, vmlp__bl, vmlp__bh, vmlp__bl,			\
	     vmlp__f, vmlp__zh, (vl));					\
  vm_f_t vmlp__ybh, vmlp__ybl;						\
  VM_DD_SUB (vmlp__ybh, vmlp__ybl, vmlp__khi, vmlp__zh,			\
	     vmlp__bh, vmlp__bl, (vl));					\
  /* yB_k0 = f - (hfsq - core) */					\
  vm_f_t vmlp__yb0h, vmlp__yb0l;					\
  VM_DD_SUB (vmlp__yb0h, vmlp__yb0l, vmlp__hf, vmlp__zh,		\
	     vmlp__core, vmlp__zh, (vl));				\
  VM_DD_SUB (vmlp__yb0h, vmlp__yb0l, vmlp__f, vmlp__zh,			\
	     vmlp__yb0h, vmlp__yb0l, (vl));				\
  vm_f_t vmlp__kbh = VM_VFMERGE_VVM (vmlp__ybh, vmlp__yb0h,		\
				     vmlp__k0, (vl));			\
  vm_f_t vmlp__kbl = VM_VFMERGE_VVM (vmlp__ybl, vmlp__yb0l,		\
				     vmlp__k0, (vl));			\
  /* select A/B by hu0 */						\
  vm_f_t vmlp__kh = VM_VFMERGE_VVM (vmlp__kbh, vmlp__yah,		\
				    vmlp__hu0, (vl));			\
  vm_f_t vmlp__kl = VM_VFMERGE_VVM (vmlp__kbl, vmlp__yal,		\
				    vmlp__hu0, (vl));			\
  vmlp__rh = VM_VFMERGE_VVM (vmlp__rh, vmlp__kh, vmlp__k, (vl));	\
  vmlp__rl = VM_VFMERGE_VVM (vmlp__rl, vmlp__kl, vmlp__k, (vl));	\
  /* ---- inf/nan lanes: x + x (masked; quiets sNaN, raises invalid \
     exactly like scalar); fixup for x <= -1.  ---- */			\
  vm_f_t vmlp__xpx = VM_VFADD_VV_M (vmlp__infnan, (x), (x), (vl));	\
  vmlp__rh = VM_VFMERGE_VVM (vmlp__rh, vmlp__xpx, vmlp__infnan, (vl));	\
  vmlp__rl = VM_VFMERGE_VFM (vmlp__rl, 0.0, vmlp__infnan, (vl));	\
  vm_f_t vmlp__yc = VM_VFADD_VV (vmlp__rh, vmlp__rl, (vl));		\
  /* +-0 dd pairs: -0 + 0 collapses to +0; restore the hi sign when the	\
     dd is exactly zero (log1p(+-0) = +-0).  */				\
  vm_b_t vmlp__z0 = VM_VMAND_MM (VM_VMFEQ_VF (vmlp__rh, 0.0, (vl)),	\
				 VM_VMFEQ_VF (vmlp__rl, 0.0, (vl)), (vl)); \
  vmlp__yc = VM_VFMERGE_VVM (vmlp__yc, vmlp__rh, vmlp__z0, (vl));	\
  vmlp__yc = VM_FIXUP1 ((x), vmlp__yc, vmlp__dom, (vl), (fixup)); \
  (yh) = vmlp__yc;							\
  (yl) = VM_VFMERGE_VFM (vmlp__rl, 0.0, vmlp__dom, (vl));		\
} while (0)

/* ------------------------------------------------------------------ */
/* Lean plain-double log1p (fdlibm s_log1p.c).  The final composition is
   carried as a dd pair (a handful of error-free transforms) so log2p1
   can scale it by 1/ln2 without paying an extra half-ulp rounding; the
   log1p entry itself just collapses the pair.
   Callers must hand in a finite x > -1 (special lanes go to scalar).  */
#define VM_LOG1P_FAST(yh, yl, x, vl) do {				\
  vm_u64_t vlp__bx = VM_BITS (x);					\
  vm_i64_t vlp__hx = VM_VSRA_VX (VM_REINT_U_I (vlp__bx), 32, (vl));	\
  vm_u64_t vlp__ab = VM_VAND_VX (vlp__bx, VM_ABSMASK_U64, (vl));	\
  vm_b_t vlp__lt029 = VM_VMSLTU_VX (					\
    VM_VAND_VX (VM_REINT_I_U (vlp__hx), 0x7fffffff, (vl)),		\
    0x3e200000, (vl));							\
  vm_b_t vlp__lt054 = VM_VMSLTU_VX (					\
    VM_VAND_VX (VM_REINT_I_U (vlp__hx), 0x7fffffff, (vl)),		\
    0x3c900000, (vl));							\
  /* fdlibm's k = 0 region: -0.2929 < x < 0.41422 (and |x| < 2^-29)  */	\
  vm_b_t vlp__midc = VM_VMSLT_VX (vlp__hx, 0x3fda827a, (vl));		\
  vm_b_t vlp__c2 = VM_VMOR_MM (VM_VMSGT_VX (vlp__hx, 0, (vl)),		\
    VM_VMSLE_VX (vlp__hx, (int64_t) (int32_t) 0xbfd2bec3, (vl)), (vl));	\
  vm_b_t vlp__mid = VM_VMAND_MM (					\
    VM_VMAND_MM (vlp__midc, vlp__c2, (vl)),				\
    VM_VMNOT_M (vlp__lt029, (vl)), (vl));				\
  vm_b_t vlp__mfx = VM_VMOR_MM (vlp__mid, vlp__lt029, (vl));		\
  vm_b_t vlp__ge53 = VM_VMSGE_VX (vlp__hx, 0x43400000, (vl));		\
  /* u = 1+x (x itself when x >= 2^53); c = exact residual of u.  */	\
  vm_f_t vlp__u = VM_VFMERGE_VVM (VM_VFADD_VF (x, 1.0, (vl)), x,	\
				  vlp__ge53, (vl));			\
  vm_u64_t vlp__ub = VM_BITS (vlp__u);					\
  vm_u64_t vlp__uh = VM_VSRL_VX (vlp__ub, 32, (vl));			\
  vm_i64_t vlp__kraw = VM_REINT_U_I (					\
    VM_VSUB_VX (VM_VSRL_VX (vlp__ub, 52, (vl)), 1023, (vl)));		\
  vm_f_t vlp__cA = VM_VFRSUB_VF (					\
    VM_VFSUB_VV (vlp__u, x, (vl)), 1.0, (vl));				\
  vm_f_t vlp__cB = VM_VFSUB_VV (x, VM_VFSUB_VF (vlp__u, 1.0, (vl)), (vl)); \
  vm_b_t vlp__kpos = VM_VMSGT_VX (vlp__kraw, 0, (vl));			\
  vm_f_t vlp__c = VM_VFMERGE_VVM (vlp__cB, vlp__cA, vlp__kpos, (vl));	\
  vlp__c = VM_VFMERGE_VFM (vlp__c, 0.0, vlp__ge53, (vl));		\
  /* true division: u = 1+x can be subnormal, where a reciprocal \
     estimate would overflow to inf and raise a spurious flag.  */ \
  vlp__c = VM_VFDIV_VV (vlp__c, vlp__u, (vl));				\
  /* normalize u to [1,2) or [0.5,1); hu is the high 20 mantissa bits */ \
  vm_u64_t vlp__hu = VM_VAND_VX (vlp__uh, 0x000fffff, (vl));		\
  vm_b_t vlp__hus = VM_VMSLTU_VX (vlp__hu, 0x6a09e, (vl));		\
  vm_u64_t vlp__unb = VM_VMERGE_VVM (					\
    VM_VOR_VV (VM_VAND_VX (vlp__ub, 0xffffffffULL, (vl)),		\
	       VM_VSLL_VX (VM_VOR_VX (vlp__hu, 0x3fe00000, (vl)), 32, (vl)), (vl)), \
    VM_VOR_VV (VM_VAND_VX (vlp__ub, 0xffffffffULL, (vl)),		\
	       VM_VSLL_VX (VM_VOR_VX (vlp__hu, 0x3ff00000, (vl)), 32, (vl)), (vl)), \
    vlp__hus, (vl));							\
  vm_i64_t vlp__ki = VM_REINT_U_I (VM_VMERGE_VVM (			\
    VM_VADD_VX (VM_REINT_I_U (vlp__kraw), 1, (vl)),			\
    VM_REINT_I_U (vlp__kraw), vlp__hus, (vl)));				\
  vm_f_t vlp__f = VM_VFSUB_VF (VM_FROM_BITS (vlp__unb), 1.0, (vl));	\
  vlp__f = VM_VFMERGE_VVM (vlp__f, x, vlp__mfx, (vl));			\
  /* hfsq = f*f/2 with its fma residual; |x| < 2^-54 lanes are parked
     on a benign f so the square cannot underflow (they return x).  */	\
  vm_f_t vlp__fh = VM_VFMERGE_VFM (vlp__f, 0.5, vlp__lt054, (vl));	\
  vm_f_t vlp__ff = VM_VFMUL_VV (vlp__fh, vlp__fh, (vl));		\
  vm_f_t vlp__hfsq = VM_VFMUL_VF (vlp__ff, 0.5, (vl));			\
  vm_f_t vlp__hfsqe = VM_VFMUL_VF (					\
    VM_VFNEG_V (VM_VFNMSAC_VV (vlp__ff, vlp__fh, vlp__fh, (vl)), (vl)),	\
    0.5, (vl));								\
  vm_f_t vlp__s; VM_QDIV (vlp__s, vlp__fh, VM_VFADD_VF (vlp__fh, 2.0, (vl)), (vl)); \
  vm_f_t vlp__z = VM_VFMUL_VV (vlp__s, vlp__s, (vl));			\
  vm_f_t vlp__X = VM_VFMUL_VV (vlp__s,					\
    VM_VFADD_VV (vlp__hfsq,						\
		 VM_VFMUL_VV (vlp__z,					\
			      VM_KFN (log1p_poly) (vlp__z, (vl)), (vl)), (vl)), (vl)); \
  /* k-path lanes with hu == 0 (|f| < 2^-20): fdlibm swaps the role of
     hfsq for R = hfsq*(1-(2/3)f) and drops the s*(...) term.  */	\
  vm_f_t vlp__hfsqr = vlp__hfsq;					\
  {									\
    vm_b_t vlp__hu0 = VM_VMAND_MM (					\
      VM_VMSEQ_VX (vlp__hu, 0, (vl)),					\
      VM_VMNOT_M (vlp__mfx, (vl)), (vl));				\
    vm_f_t vlp__R2 = VM_VFMUL_VV (vlp__hfsq,				\
      VM_VFMACC_VF (VM_VFMV_V_F (1.0, (vl)), -0.66666666666666666,	\
		    vlp__fh, (vl)), (vl));				\
    vlp__hfsqr = VM_VFMERGE_VVM (vlp__hfsq, vlp__R2, vlp__hu0, (vl));	\
    vlp__X = VM_VFMERGE_VFM (vlp__X, 0.0, vlp__hu0, (vl));		\
  }									\
  /* fdlibm's tiny ladder returns x - x*x/2, i.e. X = 0.  */		\
  vlp__X = VM_VFMERGE_VFM (vlp__X, 0.0, vlp__lt029, (vl));		\
  vm_f_t vlp__kd = VM_VFCVT_F_X_V (vlp__ki, (vl));			\
  vm_f_t vlp__khi = VM_VFMUL_VF (vlp__kd, VM_LN2_HI_L1P, (vl));		\
  vm_f_t vlp__kc = VM_VFMACC_VF (vlp__c, VM_LN2_LO_L1P, vlp__kd, (vl));	\
  /* k == 0 shape: y = f - (hfsq - X), tracked, hfsq's residual folded. */ \
  vm_f_t vlp__i, vlp__ie, vlp__yA, vlp__yAe;				\
  VM_TWO_SUM (vlp__i, vlp__ie, vlp__hfsqr, VM_VFNEG_V (vlp__X, (vl)), (vl)); \
  VM_TWO_SUM (vlp__yA, vlp__yAe, vlp__f, VM_VFNEG_V (vlp__i, (vl)), (vl)); \
  vlp__yAe = VM_VFSUB_VV (VM_VFSUB_VV (vlp__yAe, vlp__ie, (vl)),		\
			  vlp__hfsqe, (vl));				\
  /* k != 0 shape: y = k*ln2_hi - ((hfsq - (X + klo_c)) - f), tracked.  */ \
  vm_f_t vlp__s1, vlp__s1e, vlp__s2, vlp__s2e, vlp__t2, vlp__t2e;	\
  VM_TWO_SUM (vlp__s1, vlp__s1e, vlp__X, vlp__kc, (vl));		\
  VM_TWO_SUM (vlp__s2, vlp__s2e, vlp__hfsqr, VM_VFNEG_V (vlp__s1, (vl)), (vl)); \
  VM_TWO_SUM (vlp__t2, vlp__t2e, vlp__s2, VM_VFNEG_V (vlp__f, (vl)), (vl)); \
  {									\
    vm_f_t vlp__yB, vlp__yBe;						\
    VM_TWO_SUM (vlp__yB, vlp__yBe, vlp__khi, VM_VFNEG_V (vlp__t2, (vl)), (vl)); \
    vlp__yBe = VM_VFSUB_VV (						\
      VM_VFSUB_VV (VM_VFADD_VV (vlp__yBe, vlp__s1e, (vl)),		\
		   vlp__t2e, (vl)), vlp__s2e, (vl));			\
    vlp__yBe = VM_VFSUB_VV (vlp__yBe, vlp__hfsqe, (vl));		\
    (yh) = VM_VFMERGE_VVM (vlp__yB, vlp__yA, vlp__mfx, (vl));		\
    (yl) = VM_VFMERGE_VVM (vlp__yBe, vlp__yAe, vlp__mfx, (vl));		\
  }									\
  /* |x| < 2^-54: exactly x.  */						\
  (yh) = VM_VFMERGE_VVM ((yh), x, vlp__lt054, (vl));			\
  (yl) = VM_VFMERGE_VFM ((yl), 0.0, vlp__lt054, (vl));			\
} while (0)

/* Special-lane mask for log1p/log2p1: x <= -1 (invalid / divbyzero) or
   non-finite (inf, NaN).  */
#define VM_LOG1P_SPEC_M(x, vl, mo) do {					\
  vm_u64_t vls__b = VM_BITS (x);					\
  vm_i64_t vls__h = VM_VSRA_VX (VM_REINT_U_I (vls__b), 32, (vl));	\
  vm_u64_t vls__a = VM_VAND_VX (vls__b, VM_ABSMASK_U64, (vl));		\
  (mo) = VM_VMOR_MM (							\
    VM_VMAND_MM (VM_VMSLT_VX (vls__h, 0, (vl)),				\
		 VM_VMSGEU_VX (vls__a, 0x3ff0000000000000ULL, (vl)), (vl)), \
    VM_VMSGEU_VX (vls__a, 0x7ff0000000000000ULL, (vl)), (vl));		\
} while (0)

static inline vm_f_t
VM_KFN (log1p_k) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_b_t m_spec;
  VM_LOG1P_SPEC_M (x, vl, m_spec);
  vm_f_t xs = VM_VFMERGE_VFM (x, 0.5, m_spec, vl);
  vm_f_t yh, yl, y;
  VM_LOG1P_FAST (yh, yl, xs, vl);
  VM_DD_COLLAPSE (y, yh, yl, vl);
  /* log1p(+-0) = +-0.  */
  y = VM_VFMERGE_VVM (y, x, VM_VMFEQ_VF (x, 0.0, vl), vl);
  return VM_FIXUP1 (x, y, m_spec, vl, fixup);
}

/* Multiply the dd (ah, al) by the dd constant (ch, cl) -- 1/ln2 here --
   and collapse to one correctly rounded double.  Both parts are scaled
   by 2^900 first: the dd corrections are ~2^-53 of the product and
   unscaled would underflow on lanes whose answer is normal.  The final
   rounding happens once, on the scaled pair; a subnormal or zero result
   is always inexact (log2(1+x) is irrational for every rational
   x != 0), so UNDERFLOW is raised explicitly there -- the scaled
   product sometimes lands exactly on the subnormal grid and stays
   silent.  */
#define VM_LOG_MUL_DD_CONST(y, ah, al, ch, cl, vl) do {			\
  vm_f_t vlq__hs = VM_VFMUL_VF ((ah), 0x1p900, (vl));			\
  vm_f_t vlq__ls = VM_VFMUL_VF ((al), 0x1p900, (vl));			\
  vm_f_t vlq__p = VM_VFMUL_VF (vlq__hs, (ch), (vl));			\
  vm_f_t vlq__e = VM_VFMACC_VF (VM_VFNEG_V (vlq__p, (vl)), (ch),	\
				 vlq__hs, (vl));			\
  vlq__e = VM_VFMACC_VF (vlq__e, (cl), vlq__hs, (vl));			\
  vlq__e = VM_VFMACC_VF (vlq__e, (ch), vlq__ls, (vl));			\
  vm_f_t vlq__s, vlq__se;						\
  VM_TWO_SUM (vlq__s, vlq__se, vlq__p, vlq__e, (vl));			\
  VM_DD_COLLAPSE ((y), vlq__s, vlq__se, (vl));			\
  (y) = VM_VFMUL_VF ((y), 0x1p-900, (vl));				\
  {									\
    vm_u64_t vlq__yb = VM_VAND_VX (VM_BITS (y), VM_ABSMASK_U64, (vl));	\
    vm_u64_t vlq__hb = VM_VAND_VX (VM_BITS (ah), VM_ABSMASK_U64, (vl));	\
    vm_b_t vlq__mu = VM_VMAND_MM (					\
      VM_VMSLTU_VX (vlq__yb, 0x0010000000000000ULL, (vl)),		\
      VM_VMSNE_VX (vlq__hb, 0, (vl)), (vl));				\
    vm_f_t vlq__uf = VM_VFMUL_VF_MU (vlq__mu, VM_VFMV_V_F (0.0, (vl)),	\
				     VM_VFMV_V_F (0x1p-1022, (vl)),	\
				     0x1p-1022, (vl));			\
    (y) = VM_VFADD_VV_MU (vlq__mu, (y), (y), vlq__uf, (vl));		\
  }									\
  /* exact zeros keep the sign of the dd */				\
  (y) = VM_VFMERGE_VVM ((y), VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, (vl)), (ah), \
					    (vl)),			\
			VM_VMFEQ_VF ((y), 0.0, (vl)), (vl));		\
} while (0)

/* log10(x) = log(x) * log10(e), scaled as dd before the final rounding.
   fixup must be scalar log(); the dd scale preserves the fixup's
   +-inf/NaN results (log10e > 0).  */
static inline vm_f_t
VM_KFN (log10) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_f_t lgh, lgl;
  VM_LOG_DD (lgh, lgl, x, vl, fixup);	/* fixup = scalar log */
  vm_u64_t cls = VM_VFCLASS_V (lgh, vl);
  vm_b_t m_fin = VM_VMSNE_VX (VM_VAND_VX (cls, VM_CLS_FINITE, vl), 0, vl);
  vm_f_t lgs = VM_VFMERGE_VFM (lgh, 1.0, VM_VMNOT_M (m_fin, vl), vl);
  vm_f_t lglz = VM_VFMERGE_VFM (lgl, 0.0, VM_VMNOT_M (m_fin, vl), vl);
  vm_f_t rh, rl;
  VM_DD_MUL (rh, rl, lgs, lglz,
	     VM_VFMV_V_F (VM_LOG10E_HI, vl), VM_VFMV_V_F (VM_LOG10E_LO, vl),
	     vl);
  vm_f_t y;
  VM_DD_COLLAPSE (y, rh, rl, vl);
  return VM_VFMERGE_VVM (lgh, y, m_fin, vl);
}

/* ------------------------------------------------------------------ */
/* log2p1(x) = log2(1+x) = log2(u) + log2(1+c/u), with u = RN(1+x) and
   c = x-(u-1) its exact rounding residual: the table core runs on u and
   the c/(u*ln2) tail is folded into the core's l2 term (see
   log2p1_tail), so the answer is one rounded sum -- important when x is
   tiny, where the tail *is* the answer.
   fixup must be scalar log1p() -- it agrees with log2p1 on every special
   lane (-inf at -1, NaN below it, +inf at +inf).  */
/* Rare-lane rescue for log2p1, deliberately out of line: the generic
   scalar_fixup carries 300-double buffers whose frame would otherwise be
   charged to every strip of the hot loop.  */
__attribute__ ((noinline)) static vm_f_t
VM_KFN (log2p1_fixup) (vm_f_t x, vm_f_t y, vm_b_t m, size_t vl,
		       double (*fn) (double))
{
  return VM_FIXUP1 (x, y, m, vl, fn);
}

/* Scalar log2p1 for the fixup lanes: log2(1+x) agrees with log2p1 on
   every special lane (-inf at -1, NaN below it, +inf at inf) and is the
   plain scalar answer for the rare huge lanes the kernel hands off.  */
#ifndef VM_LOG2P1_SCALAR_DEFINED
#define VM_LOG2P1_SCALAR_DEFINED
static inline double
vm_log2p1_scalar (double x)
{
  return (log2) (1.0 + x);
}
#endif
#define log2p1_scalar vm_log2p1_scalar

static inline vm_f_t
VM_KFN (log2p1) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  /* log2(1+x) = log2(u) + c/(u*ln2), u = RN(1+x), c = x-(u-1) exact.
     The core works in log2 units, so the only correction is the c/u tail
     in log2 units; it is folded into the core's l2 term and the sum
     rounds once.  The tail is at most 2^-24 of the answer for
     |x| >= 2^-29 and needs only ~|x| relative accuracy, which the
     table's r*(1-z) gives (see log2p1_tail).  */
  vm_b_t m_spec;
  VM_LOG1P_SPEC_M (x, vl, m_spec);
  vm_f_t xs = VM_VFMERGE_VFM (x, 0.5, m_spec, vl);

  /* u = RN(1+xs), c = xs-(u-1) its exact residual (exact for u < 2;
     above that it is below 2^-53/|answer|).  */
  vm_f_t u = VM_VFADD_VF (xs, 1.0, (vl));
  vm_f_t c = VM_VFSUB_VV (xs, VM_VFSUB_VF (u, 1.0, (vl)), (vl));

  vm_f_t yh, yl;
  VM_L2T_CORE (yh, yl, u, VM_L2T_L2_TAIL (c, vl2t_r, vl2t_w, vl2t_b), vl);
  {
    vm_f_t y = VM_VFADD_VV (yh, yl, (vl));
    /* A nonzero subnormal answer is always inexact (log2(1+x) is
       irrational for rational x != 0) and must raise UNDERFLOW; the
       product above can land exactly on the subnormal grid, so raise it
       with a masked flush-to-zero multiply.  */
    {
      vm_u64_t yb = VM_VAND_VX (VM_BITS (y), VM_ABSMASK_U64, (vl));
      vm_b_t m_uf = VM_VMAND_MM (
	VM_VMAND_MM (VM_VMSLTU_VX (yb, 0x0010000000000000ULL, (vl)),
		     VM_VMSNE_VX (yb, 0, (vl)), (vl)),
	VM_VMSNE_VX (VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, (vl)),
		     0, (vl)), (vl));
      vm_f_t uf = VM_VFMUL_VF_MU (m_uf, VM_VFMV_V_F (0.0, (vl)),
				  VM_VFMV_V_F (0x1p-1022, (vl)),
				  0x1p-1022, (vl));
      y = VM_VFADD_VV_MU (m_uf, y, y, uf, (vl));
    }
    /* log2p1(+-0) = +-0.  */
    y = VM_VFMERGE_VVM (y, x, VM_VMFEQ_VF (x, 0.0, (vl)), vl);
    if (VM_VFIRST_M (m_spec, vl) < 0)
      return y;
    return VM_KFN (log2p1_fixup) (x, y, m_spec, vl, fixup);
  }
}
