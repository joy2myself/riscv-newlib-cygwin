/* k_hyperinv.inc -- asinh / acosh / atanh kernels.

   LMUL-generic template: include after vecmath_rvvcompat.h,
   vecmath_dd.h and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.
   Also needs k_log.inc instantiated at the same LMUL (VM_LOG_DD,
   VM_KFN(log)).

   Ports newlib's fdlibm math/s_asinh.c, math/e_acosh.c and
   math/e_atanh.c with the double-double argument formation removed: the
   log is taken as a double-double (VM_LOG_DD) and the argument's own
   rounding error rides along as an exact tail (two_sum) that is added
   back after the log:

       log1p(a + t) = log(1 + a) + t/(1 + a),
       log(a + t)   = log(a) + t/a.

   That makes the final collapse the only rounding of the result (~0.5
   ulp) instead of the 1.18 ulp single-double fdlibm forms measure.

   The |x| > 2^28 branches of asinh/acosh (acosh = asinh = log|x| + ln2
   there) are rare in real use and never in the benchmark: they are
   evaluated by a second, per-strip-guarded pass (one vfirst) and merged
   per lane, so the hot path stays short and lane values never depend on
   the strip's other lanes.  Domain edges, +-Inf, NaN and the exact zeros
   are handled inline with the fdlibm flag semantics.  */

#define VM_HINV_LN2	6.93147180559945286227e-01


/* ------------------------------------------------------------------ */
/* Private copy of the double-double log used by this template ONLY.
   It is byte-for-byte the VM_LOG_DD contract (Arm log.c main path, near-1
   order-12 polynomial, >106-bit result pair, scalar fixup for the
   subnormal/negative/Inf/NaN lanes), embedded here so this template
   stays self-contained and does not depend on k_log.inc's internals.
   Data: vm_data.h's __vecmath_log_data (same as k_log.inc).  */
#include <vm_data.h>

#define VM_HINV_LOG_LO		0x3fee000000000000ULL
#define VM_HINV_LOG_HILO	0x0003090000000000ULL
#define VM_HINV_LOG_OFF		0x3fe6000000000000ULL

#define VM_HINV_LOG_DD(yh, yl, x, vl, fixup) do {			\
  const struct vm_log_data *vmlg__D = &__vecmath_log_data;		\
  vm_u64_t vmlg__ix = VM_BITS (x);					\
  vm_u64_t vmlg__top = VM_VSRL_VX (vmlg__ix, 48, (vl));			\
  vm_b_t vmlg__mspec = VM_VMSGEU_VX (					\
    VM_VSUB_VX (vmlg__top, 0x10, (vl)), 0x7fe0, (vl));			\
  vm_b_t vmlg__mnear = VM_VMSLTU_VX (					\
    VM_VSUB_VX (vmlg__ix, VM_HINV_LOG_LO, (vl)), VM_HINV_LOG_HILO, (vl)); \
  vm_f_t vmlg__xs = VM_VFMERGE_VFM ((x), 1.0, vmlg__mspec, (vl));	\
  vm_f_t vmlg__xn = VM_VFMERGE_VFM ((x), 1.0,				\
				    VM_VMNOT_M (vmlg__mnear, (vl)), (vl)); \
  /* near-1 polynomial (order 12) */					\
  vm_f_t vmlg__r = VM_VFSUB_VF (vmlg__xn, 1.0, (vl));			\
  vm_f_t vmlg__r2 = VM_VFMUL_VV (vmlg__r, vmlg__r, (vl));		\
  vm_f_t vmlg__r3 = VM_VFMUL_VV (vmlg__r, vmlg__r2, (vl));		\
  vm_f_t vmlg__q = VM_VFMACC_VF (VM_VFMV_V_F (vmlg__D->poly1[7], (vl)), \
				 vmlg__D->poly1[8], vmlg__r, (vl));	\
  vmlg__q = VM_VFMACC_VF (vmlg__q, vmlg__D->poly1[9], vmlg__r2, (vl));	\
  vmlg__q = VM_VFMACC_VF (vmlg__q, vmlg__D->poly1[10], vmlg__r3, (vl));	\
  vm_f_t vmlg__p = VM_VFMACC_VF (VM_VFMV_V_F (vmlg__D->poly1[4], (vl)), \
				 vmlg__D->poly1[5], vmlg__r, (vl));	\
  vmlg__p = VM_VFMACC_VF (vmlg__p, vmlg__D->poly1[6], vmlg__r2, (vl));	\
  vmlg__p = VM_VFMACC_VV (vmlg__p, vmlg__r3, vmlg__q, (vl));		\
  vm_f_t vmlg__P = VM_VFMACC_VF (VM_VFMV_V_F (vmlg__D->poly1[1], (vl)), \
				 vmlg__D->poly1[2], vmlg__r, (vl));	\
  vmlg__P = VM_VFMACC_VF (vmlg__P, vmlg__D->poly1[3], vmlg__r2, (vl));	\
  vmlg__P = VM_VFMACC_VV (vmlg__P, vmlg__r3, vmlg__p, (vl));		\
  vm_f_t vmlg__npoly = VM_VFMUL_VV (vmlg__r3, vmlg__P, (vl));		\
  vm_f_t vmlg__w = VM_VFMUL_VF (vmlg__r, 0x1p27, (vl));			\
  vm_f_t vmlg__rhi = VM_VFSUB_VV (VM_VFADD_VV (vmlg__r, vmlg__w, (vl)), \
				  vmlg__w, (vl));			\
  vm_f_t vmlg__rlo = VM_VFSUB_VV (vmlg__r, vmlg__rhi, (vl));		\
  vmlg__w = VM_VFMUL_VF (VM_VFMUL_VV (vmlg__rhi, vmlg__rhi, (vl)),	\
			 -0.5, (vl));					\
  vm_f_t vmlg__nhi = VM_VFADD_VV (vmlg__r, vmlg__w, (vl));		\
  vm_f_t vmlg__nlo = VM_VFADD_VV (VM_VFSUB_VV (vmlg__r, vmlg__nhi, (vl)), \
				  vmlg__w, (vl));			\
  vmlg__nlo = VM_VFADD_VV (vmlg__nlo,					\
    VM_VFMUL_VV (VM_VFMUL_VF (vmlg__rlo, -0.5, (vl)),			\
		 VM_VFADD_VV (vmlg__rhi, vmlg__r, (vl)), (vl)), (vl));	\
  vm_f_t vmlg__zh = VM_VFMV_V_F (0.0, (vl));				\
  vm_f_t vmlg__dh, vmlg__dl;						\
  VM_DD_ADD (vmlg__dh, vmlg__dl, vmlg__nhi, vmlg__nlo,			\
	     vmlg__npoly, vmlg__zh, (vl));				\
  /* main path */							\
  vm_u64_t vmlg__ixb = VM_BITS (vmlg__xs);				\
  vm_u64_t vmlg__tmp = VM_VSUB_VX (vmlg__ixb, VM_HINV_LOG_OFF, (vl));	\
  vm_u64_t vmlg__off = VM_VSLL_VX (VM_VAND_VX (			\
    VM_VSRL_VX (vmlg__tmp, 45, (vl)), 127, (vl)), 4, (vl));		\
  vm_f_t vmlg__invc = VM_VLUXEI64 (&vmlg__D->tab[0].invc, vmlg__off, (vl)); \
  vm_f_t vmlg__logc = VM_VLUXEI64 (&vmlg__D->tab[0].logc, vmlg__off, (vl)); \
  vm_i64_t vmlg__k = VM_VSRA_VX (VM_REINT_U_I (vmlg__tmp), 52, (vl));	\
  vm_u64_t vmlg__iz = VM_VSUB_VV (vmlg__ixb,				\
    VM_VAND_VX (vmlg__tmp, 0xfff0000000000000ULL, (vl)), (vl));		\
  vm_f_t vmlg__z = VM_FROM_BITS (vmlg__iz);				\
  vm_f_t vmlg__kd = VM_VFCVT_F_X_V (vmlg__k, (vl));			\
  vmlg__r = VM_VFMACC_VV (VM_VFMV_V_F (-1.0, (vl)),			\
			  vmlg__invc, vmlg__z, (vl));			\
  vmlg__w = VM_VFMACC_VF (vmlg__logc, vmlg__D->ln2hi, vmlg__kd, (vl));	\
  vm_f_t vmlg__mhi = VM_VFADD_VV (vmlg__w, vmlg__r, (vl));		\
  vm_f_t vmlg__mlo = VM_VFADD_VV (					\
    VM_VFADD_VV (VM_VFSUB_VV (vmlg__w, vmlg__mhi, (vl)), vmlg__r, (vl)), \
    VM_VFMUL_VF (vmlg__kd, vmlg__D->ln2lo, (vl)), (vl));		\
  vmlg__r2 = VM_VFMUL_VV (vmlg__r, vmlg__r, (vl));			\
  vmlg__q = VM_VFMACC_VF (VM_VFMV_V_F (vmlg__D->poly[3], (vl)),		\
			  vmlg__D->poly[4], vmlg__r, (vl));		\
  vmlg__q = VM_VFMUL_VV (vmlg__r2, vmlg__q, (vl));			\
  vmlg__p = VM_VFMACC_VF (VM_VFMV_V_F (vmlg__D->poly[1], (vl)),		\
			  vmlg__D->poly[2], vmlg__r, (vl));		\
  vmlg__p = VM_VFADD_VV (vmlg__p, vmlg__q, (vl));			\
  vmlg__p = VM_VFMUL_VV (VM_VFMUL_VV (vmlg__r, vmlg__r2, (vl)),		\
			 vmlg__p, (vl));				\
  vm_f_t vmlg__pr = VM_VFADD_VV (					\
    VM_VFMUL_VF (vmlg__r2, vmlg__D->poly[0], (vl)), vmlg__p, (vl));	\
  vm_f_t vmlg__eh, vmlg__el;						\
  VM_DD_ADD (vmlg__eh, vmlg__el, vmlg__mhi, vmlg__mlo,			\
	     vmlg__pr, vmlg__zh, (vl));					\
  vm_f_t vmlg__h = VM_VFMERGE_VVM (vmlg__eh, vmlg__dh, vmlg__mnear, (vl)); \
  vm_f_t vmlg__l = VM_VFMERGE_VVM (vmlg__el, vmlg__dl, vmlg__mnear, (vl)); \
  vm_f_t vmlg__yc;							\
  VM_DD_COLLAPSE (vmlg__yc, vmlg__h, vmlg__l, (vl));			\
  (yh) = VM_KFN (scalar_fixup) ((x), vmlg__yc, vmlg__mspec, (vl), (fixup)); \
  (yl) = VM_VFMERGE_VFM (vmlg__l, 0.0, vmlg__mspec, (vl));		\
} while (0)

/* log(x + tail) with the tail added back after the dd log.  */
#define VM_HINV_LOGTAIL(yh, yl, x, tail, vl) do {			\
  VM_HINV_LOG_DD ((yh), (yl), (x), (vl), log);				\
  (yl) = VM_VFADD_VV ((yl), VM_VFDIV_VV ((tail), (x), (vl)), (vl));	\
} while (0)

/* value = log1p of the exact pair (ah + al), as a double-double.  */
#define VM_HINV_LOG1PDD(yh, yl, ah, al, vl) do {			\
  vm_f_t vhl1_h, vhl1_t;						\
  VM_TWO_SUM (vhl1_h, vhl1_t, VM_VFMV_V_F (1.0, (vl)), (ah), (vl));	\
  vhl1_t = VM_VFADD_VV (vhl1_t, (al), (vl));				\
  VM_HINV_LOG_DD ((yh), (yl), vhl1_h, (vl), log);				\
  (yl) = VM_VFADD_VV ((yl), VM_VFDIV_VV (vhl1_t, vhl1_h, (vl)), (vl));	\
} while (0)

/* asinh(x) = sign(x) * log1p(|x| + x^2/(1+sqrt(1+x^2))), valid for every
   finite |x| up to 2^28 (beyond, the x^2 term would overflow).  */
static inline vm_f_t
VM_KFN (asinh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3e300000u, vl);	/* |x| < 2^-28 */
  vm_b_t mnf = VM_VMSGEU_VX (ix, 0x7ff00000u, vl);
  vm_b_t mhuge = VM_VMSGTU_VX (ix, 0x41b00000u, vl);	/* |x| > 2^28 */
  vm_b_t mbig = VM_VMAND_MM (mhuge, VM_VMNOT_M (mnf, vl), vl);
  vm_f_t ax = VM_VFABS_V (x, vl);
  vm_f_t axs = VM_VFMERGE_VFM (ax, 1.0,
			       VM_VMOR_MM (mhuge, mtiny, vl), vl);
  vm_f_t z;
  {
    vm_f_t u2 = VM_VFMUL_VV (axs, axs, vl);
    vm_f_t s = VM_VFSQRT_V (VM_VFADD_VF (u2, 1.0, vl), vl);
    vm_f_t c = VM_VFDIV_VV (u2, VM_VFADD_VF (s, 1.0, vl), vl);	/* x^2/(1+s) */
    /* (a, tail) = |x| + c exactly; c <= |x| so fast2sum applies.  */
    vm_f_t a = VM_VFADD_VV (axs, c, vl);
    vm_f_t tail = VM_VFADD_VV (VM_VFSUB_VV (axs, a, vl), c, vl);
    vm_f_t yh, yl;
    VM_HINV_LOG1PDD (yh, yl, a, tail, vl);
    z = VM_VFADD_VV (yh, yl, vl);
  }
  z = VM_VFMERGE_VVM (z, VM_VFNEG_V (z, vl), mneg, vl);
  z = VM_VFMERGE_VVM (z, x, mtiny, vl);		/* asinh(tiny) = tiny */
  if (VM_VFIRST_M (mbig, vl) >= 0)
    {
      /* |x| > 2^28: log(|x|) + ln2.  */
      vm_f_t axh = VM_VFMERGE_VFM (ax, 1.0, VM_VMNOT_M (mbig, vl), vl);
      vm_f_t lh, ll, s, e;
      VM_HINV_LOG_DD (lh, ll, axh, vl, log);
      VM_TWO_SUM (s, e, lh, VM_VFMV_V_F (VM_HINV_LN2, vl), vl);
      z = VM_VFMERGE_VVM (
	z, VM_VFMUL_VV (
	  VM_VFMERGE_VFM (
	    VM_VFMV_V_F (1.0, vl), -1.0, mneg, vl),
	  VM_VFADD_VV (s, VM_VFADD_VV (e, ll, vl), vl), vl), mbig, vl);
    }
  return VM_VFMERGE_VVM (z, VM_VFADD_VV_M (mnf, x, x, vl), mnf, vl);
}

/* acosh(x) = log1p(t + sqrt(2t + t^2)), t = x-1, valid for x in (1,2^28];
   x > 2^28 takes the guarded log(x)+ln2 pass.  */
static inline vm_f_t
VM_KFN (acosh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_b_t mdom = VM_VMSLT_VX (hx, 0x3ff00000, vl);	/* x < 1 (signed) */
  vm_b_t mnf = VM_VMSGEU_VX (ix, 0x7ff00000u, vl);
  vm_b_t mhuge = VM_VMSGTU_VX (ix, 0x41b00000u, vl);	/* x > 2^28 */
  vm_b_t mbig = VM_VMAND_MM (mhuge, VM_VMNOT_M (mnf, vl), vl);
  vm_b_t mone = VM_VMAND_MM (
    VM_VMSEQ_VX (ix, 0x3ff00000u, vl),
    VM_VMSEQ_VX (VM_VAND_VX (VM_BITS (x), 0xffffffffULL, vl), 0, vl), vl);
  vm_b_t mskip = VM_VMOR_MM (mdom, VM_VMOR_MM (mhuge, mone, vl), vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 3.0, mskip, vl);
  vm_f_t z;
  {
    /* r = 2t + t^2 exactly, then a sqrt with one residual correction:
       for x near 1 the result is ~sqrt(2t), so the sqrt argument's own
       rounding would otherwise be half the error budget.  */
    vm_f_t t = VM_VFSUB_VF (xs, 1.0, vl);
    vm_f_t t2 = VM_VFMUL_VV (t, t, vl);
    /* eneg = t2 - t*t exactly (single fma, tiny result).  */
    vm_f_t eneg = VM_VFNMSAC_VV (t2, t, t, vl);
    vm_f_t tt = VM_VFADD_VV (t, t, vl);
    vm_f_t r, e2;
    VM_TWO_SUM (r, e2, tt, t2, vl);	/* r + e2 = 2t + t2 exactly */
    vm_f_t e = VM_VFSUB_VV (e2, eneg, vl);	/* 2t+t^2 - r exactly */
    vm_f_t q = VM_VFSQRT_V (r, vl);
    vm_f_t d = VM_VFNMSAC_VV (r, q, q, vl);	/* r - q^2 */
    vm_f_t corr = VM_VFDIV_VV (VM_VFADD_VV (d, e, vl),
			       VM_VFADD_VV (q, q, vl), vl);
    q = VM_VFADD_VV (q, corr, vl);
    /* (a, tail) = q + t exactly; q >= t >= 0 so fast2sum applies.  */
    vm_f_t a = VM_VFADD_VV (q, t, vl);
    vm_f_t tail = VM_VFADD_VV (VM_VFSUB_VV (q, a, vl), t, vl);
    vm_f_t yh, yl;
    VM_HINV_LOG1PDD (yh, yl, a, tail, vl);
    z = VM_VFADD_VV (yh, yl, vl);
  }
  z = VM_VFMERGE_VFM (z, 0.0, mone, vl);		/* acosh(1) = +0 */
  if (VM_VFIRST_M (mbig, vl) >= 0)
    {
      /* x > 2^28: log(x) + ln2.  */
      vm_f_t xh = VM_VFMERGE_VFM (x, 2.0, VM_VMNOT_M (mbig, vl), vl);
      vm_f_t lh, ll, s, e;
      VM_HINV_LOG_DD (lh, ll, xh, vl, log);
      VM_TWO_SUM (s, e, lh, VM_VFMV_V_F (VM_HINV_LN2, vl), vl);
      z = VM_VFMERGE_VVM (
	z, VM_VFADD_VV (s, VM_VFADD_VV (e, ll, vl), vl), mbig, vl);
    }
  z = VM_VFMERGE_VVM (z, VM_VFADD_VV_M (mnf, x, x, vl), mnf, vl);	/* +Inf -> +Inf */
  {
    /* x < 1 -> NaN with INVALID, as fdlibm's (x-x)/(x-x); the masked
       quotient raises nothing on the other lanes and stays NaN for a NaN
       input (qNaN in, qNaN out, no flag).  Last, so it wins over x+x.  */
    vm_f_t d = VM_VFSUB_VV_M (mdom, x, x, vl);
    return VM_VFMERGE_VVM (
      z, VM_VFDIV_VV_MU (mdom, VM_VFMV_V_F (0.0, vl), d, d, vl), mdom, vl);
  }
}

/* atanh(x) = sign(x) * 0.5 * log1p(2|x| + 2|x|*|x|/(1-|x|)), |x| < 1.  */
static inline vm_f_t
VM_KFN (atanh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_u64_t lx = VM_VAND_VX (VM_BITS (x), 0xffffffffULL, vl);
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3e300000u, vl);	/* |x| < 2^-28 */
  /* |x| > 1 covers Inf/NaN too, exactly as fdlibm's test does.  */
  vm_b_t mdom = VM_VMOR_MM (
    VM_VMSGTU_VX (ix, 0x3ff00000u, vl),
    VM_VMAND_MM (VM_VMSEQ_VX (ix, 0x3ff00000u, vl),
		 VM_VMSNE_VX (lx, 0, vl), vl), vl);
  vm_b_t mone = VM_VMAND_MM (VM_VMSEQ_VX (ix, 0x3ff00000u, vl),
			     VM_VMSEQ_VX (lx, 0, vl), vl);
  vm_f_t ax = VM_VFABS_V (x, vl);
  vm_f_t axs = VM_VFMERGE_VFM (
    ax, 0.25, VM_VMOR_MM (mdom, VM_VMOR_MM (mone, mtiny, vl), vl), vl);
  vm_f_t z;
  {
    vm_f_t t = VM_VFADD_VV (axs, axs, vl);
    vm_f_t q = VM_VFDIV_VV (VM_VFMUL_VV (t, axs, vl),
			    VM_VFRSUB_VF (axs, 1.0, vl), vl);
    /* (a, tail) = t + q exactly (general two_sum; either may dominate).  */
    vm_f_t a, tail;
    VM_TWO_SUM (a, tail, t, q, vl);
    vm_f_t yh, yl;
    VM_HINV_LOG1PDD (yh, yl, a, tail, vl);
    z = VM_VFMUL_VF (VM_VFADD_VV (yh, yl, vl), 0.5, vl);
  }
  z = VM_VFMERGE_VVM (z, VM_VFNEG_V (z, vl), mneg, vl);
  z = VM_VFMERGE_VVM (z, x, mtiny, vl);		/* atanh(tiny) = tiny */
  {
    /* |x| > 1 -> NaN with INVALID, as fdlibm's (x-x)/(x-x).  */
    vm_f_t d = VM_VFSUB_VV_M (mdom, x, x, vl);
    z = VM_VFMERGE_VVM (
      z, VM_VFDIV_VV_MU (mdom, VM_VFMV_V_F (0.0, vl), d, d, vl), mdom, vl);
  }
  {
    /* |x| == 1 -> +-Inf with DIVBYZERO (fdlibm's x/zero).  */
    vm_f_t d = VM_VFMERGE_VFM (x, 0.0, VM_VMNOT_M (mone, vl), vl);
    return VM_VFMERGE_VVM (
      z, VM_VFDIV_VV_MU (mone, VM_VFMV_V_F (0.0, vl), d,
			 VM_VFMV_V_F (0.0, vl), vl), mone, vl);
  }
}
