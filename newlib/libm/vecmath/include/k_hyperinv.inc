/* k_hyperinv.inc -- asinh / acosh / atanh kernels.

   LMUL-generic template: include after vecmath_rvvcompat.h,
   vecmath_dd.h and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.
   Pulls in k_log.inc at the same LMUL for the log cores.

   Ports newlib's fdlibm math/s_asinh.c, math/e_acosh.c and
   math/e_atanh.c with the double-double argument formation removed:
   the hot paths take log1p of an exact (a, tail) pair built from the
   function's own radical, so the final collapse is the only rounding
   (~0.5 ulp).  Every hot path now rides on the 256-entry exact-residual
   ln table core (VM_LNT_CORE in k_log.inc): the tail is folded into the
   core's constant term through the core's own table reciprocal, so the
   log needs no division and no double-double combine.  The |x| > 2^28
   passes keep the older VM_LOG_DD form; they are off the hot path.

   The |x| > 2^28 branches of asinh/acosh (acosh = asinh = log|x| + ln2
   there) are rare in real use and never in the benchmark: they are
   evaluated by a second, per-strip-guarded pass (one vfirst) and merged
   per lane, so the hot path stays short and lane values never depend on
   the strip's other lanes.  Domain edges, +-Inf, NaN and the exact zeros
   are handled inline with the fdlibm flag semantics.  */

#include <k_log.inc>

#define VM_HINV_LN2	6.93147180559945286227e-01

/* The dd log is only needed by the rare |x| > 2^28 passes now.  */
#define VM_HINV_LOG_DD(yh, yl, x, vl, fixup) \
  VM_LOG_DD ((yh), (yl), (x), (vl), (fixup))

/* log1p of the exact pair (ah + al) as one correctly rounded double,
   through the ln table core: u = RN(1+ah), c = ah-(u-1) exact (Sterbenz
   for u < 2; above that |c| <= 1 and c/u is below the answer's ulp), and
   the tail c+al is folded into the core's constant term as (c+al)/u with
   the core's own table reciprocal -- no division.  */
#define VM_HINV_LOG1P_FAST(y, ah, al, vl) do {				\
  vm_f_t vhl1__u = VM_VFADD_VF ((ah), 1.0, (vl));			\
  vm_f_t vhl1__c = VM_VFSUB_VV ((ah),					\
				VM_VFSUB_VF (vhl1__u, 1.0, (vl)), (vl)); \
  vhl1__c = VM_VFADD_VV (vhl1__c, (al), (vl));				\
  VM_LNT_CORE ((y), vhl1__u, VM_LNT_C2_TAIL (vhl1__c), (vl));		\
} while (0)

/* asinh(x) = sign(x) * log1p(|x| + x^2/(1+sqrt(1+x^2))), valid for every
   finite |x| up to 2^28 (beyond, log1p is approximated by log|x| + ln2).
   The lane masks come from the integer bits, not from FP compares: the
   RVV ordering compares are *signaling*, so they would raise INVALID on
   the NaN lanes that fdlibm's asinh returns quietly.  */
static inline vm_f_t
VM_KFN (asinh) (vm_f_t x, size_t vl)
{
  vm_u64_t ix = VM_VAND_VX (VM_BITS (x), 0x7fffffffffffffffULL, vl);
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3e30000000000000ULL, vl);	/* < 2^-28 */
  vm_b_t mbig = VM_VMSGTU_VX (ix, 0x41b0000000000000ULL, vl);	/* > 2^28 */
  vm_b_t mnf = VM_VMSGEU_VX (ix, 0x7ff0000000000000ULL, vl);
  vm_f_t ax = VM_FROM_BITS (ix);
  vm_f_t z;
  {
    /* x*x is the one thing that cannot be evaluated on the tiny
       (underflow) and huge (overflow) lanes, so sanitize those.  */
    vm_f_t axs = VM_VFMERGE_VFM (ax, 1.0,
				 VM_VMOR_MM (mtiny, mbig, vl), vl);
    vm_f_t u2 = VM_VFMUL_VV (axs, axs, vl);
    vm_f_t s = VM_VFSQRT_V (VM_VFADD_VF (u2, 1.0, vl), vl);
    vm_f_t c = VM_VFDIV_VV (u2, VM_VFADD_VF (s, 1.0, vl), vl);
    /* log1p(|x| + c) through the ln table core: u = RN(1+|x|+c), and
       c' = (|x|+c)-(u-1) is exact -- |x|-(u-1) is exact because the two
       are within a factor of two, so only the last add rounds.  */
    vm_f_t u = VM_VFADD_VV (VM_VFADD_VF (axs, 1.0, vl), c, vl);
    vm_f_t cu = VM_VFADD_VV (
      VM_VFSUB_VV (axs, VM_VFSUB_VF (u, 1.0, vl), vl), c, vl);
    VM_LNT_CORE (z, u, VM_LNT_C2_TAIL (cu), vl);
  }
  z = VM_VFSGNJ_VV (z, x, vl);			/* odd function */
  z = VM_VFMERGE_VVM (z, x, mtiny, vl);		/* asinh(tiny) = tiny */
  if (VM_VFIRST_M (mbig, vl) >= 0)
    {
      /* |x| > 2^28 (and Inf/NaN, whose results the mnf merge below
	 overwrites): log|x| + ln2; the dd log keeps this off the hot path
	 while staying well inside half an ulp on every finite lane.  */
      vm_b_t mfin = VM_VMOR_MM (VM_VMNOT_M (mbig, vl), mnf, vl);
      vm_f_t axh = VM_VFMERGE_VFM (ax, 1.0, mfin, vl);
      vm_f_t lh, ll, s, e;
      VM_HINV_LOG_DD (lh, ll, axh, vl, log);
      VM_TWO_SUM (s, e, lh, VM_VFMV_V_F (VM_HINV_LN2, vl), vl);
      z = VM_VFMERGE_VVM (
	z, VM_VFSGNJ_VV (
	  VM_VFADD_VV (s, VM_VFADD_VV (e, ll, vl), vl), x, vl), mbig, vl);
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
    /* r = t*(2+t) = 2t+t^2 with its exact product residual, then a sqrt
       with one residual correction: for x near 1 the answer is
       ~sqrt(2t), so the sqrt argument's own rounding would otherwise be
       half the error budget.  The correction rides in the tail, so the
       (a, tail) pair below is the exact log1p argument.  */
    vm_f_t t = VM_VFSUB_VF (xs, 1.0, vl);
    vm_f_t r, e;
    VM_TWO_PROD (r, e, t, VM_VFADD_VF (t, 2.0, vl), vl);
    vm_f_t q = VM_VFSQRT_V (r, vl);
    vm_f_t d = VM_VFNMSAC_VV (r, q, q, vl);	/* r - q^2 */
    vm_f_t corr = VM_VFDIV_VV (VM_VFADD_VV (d, e, vl),
			       VM_VFADD_VV (q, q, vl), vl);
    /* (a, tail) = q + t exactly; q >= t >= 0 so fast2sum applies.  */
    vm_f_t a = VM_VFADD_VV (q, t, vl);
    vm_f_t tail = VM_VFADD_VV (VM_VFSUB_VV (q, a, vl), t, vl);
    tail = VM_VFADD_VV (tail, corr, vl);
    VM_HINV_LOG1P_FAST (z, a, tail, vl);
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
    /* a = 2|x| + 2|x|^2/(1-|x|) = expm1(2 atanh x): the product and the
       quotient each keep their exact fma residual, and the quotient's
       residual (times a cheap 1/d estimate) rides in the tail, so
       (a, tail) is the exact log1p argument and the table log below is
       the only rounding.  */
    vm_f_t t = VM_VFADD_VV (axs, axs, vl);
    vm_f_t n, en;
    VM_TWO_PROD (n, en, t, axs, vl);		/* n + en = t*axs exactly */
    vm_f_t d = VM_VFRSUB_VF (axs, 1.0, vl);	/* 1-|x|, exact */
    vm_f_t r, q, e;
    q = VM_VFDIV_VV (n, d, vl);
    e = VM_VFNMSAC_VV (n, q, d, vl);		/* n - q*d, exact */
    e = VM_VFADD_VV (e, en, vl);
    r = VM_VFREC7_V (d, vl);
    /* (a, tail) = t + q exactly (general two_sum; either may dominate),
       plus the quotient's residual e/d as a tail correction.  */
    vm_f_t a, tail;
    VM_TWO_SUM (a, tail, t, q, vl);
    tail = VM_VFADD_VV (tail, VM_VFMUL_VV (e, r, vl), vl);
    VM_HINV_LOG1P_FAST (z, a, tail, vl);
    z = VM_VFMUL_VF (z, 0.5, vl);
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
