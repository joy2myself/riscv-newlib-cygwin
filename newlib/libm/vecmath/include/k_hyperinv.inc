/* k_hyperinv.inc -- asinh / acosh / atanh kernels.

   LMUL-generic template: include after vecmath_rvvcompat.h,
   vecmath_dd.h and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.
   Pulls k_log.inc in at the same LMUL for the natural-log table core
   (VM_LNT_CORE) and for VM_LOG_DD, which only the rare |x| > 2^28 passes
   still use.

   Ports newlib's fdlibm math/s_asinh.c, math/e_acosh.c and
   math/e_atanh.c.  Each hot path forms the argument of log1p as an exact
   (a, tail) pair and hands it to the table core:

       log1p(a + tail) = ln(u) + (a - (u-1) + tail)/u,  u = RN(1+a),

   with the tail folded into the core's constant term.  The core takes
   1/u (and 2^-j) from its own table reciprocal -- no division -- and the
   answer is a single rounded sum, so the final collapse is the only
   rounding (~0.5 ulp) instead of the 1.18 ulp the single-double fdlibm
   forms measure.  The old private double-double log copy is gone.

   The argument pairs are formed with the error-free transformations of
   vecmath_dd.h: asinh needs no explicit pair (u = (1+|x|) + c0 is formed
   directly and c' = (|x| - (u-1)) + c0 is the exact excess over u-1),
   acosh carries the two_prod residual of r = t*(2+t) plus the sqrt's own
   one-step correction in the tail, and atanh carries the fma residuals
   of both the product 2|x|*|x| and the quotient by 1-|x|, scaled by a
   7-bit vfrec7 reciprocal (the residual only needs a few bits).

   The |x| > 2^28 branches of asinh/acosh (acosh = asinh = log|x| + ln2
   there) are rare in real use and never in the benchmark: they are
   evaluated by a second, per-strip-guarded pass (one vfirst) and merged
   per lane, so the hot path stays short and lane values never depend on
   the strip's other lanes.  That pass also carries the +-Inf/NaN merge,
   since every non-finite lane is above 2^28.  Domain edges and the exact
   zeros are handled inline with the fdlibm flag semantics.

   Masks are integer bit-pattern compares only: RVV's signaling float
   compares (vmflt/vmfle) raise INVALID on NaN lanes, i.e. on lanes whose
   NaN the caller only wants propagated.  */

#include <k_log.inc>

#define VM_HINV_LN2	6.93147180559945286227e-01

/* y = log1p(a + tail): the exact (a, tail) argument pair through the
   natural-log table core.  u = RN(1+a), and c = a-(u-1) (exact) + tail is
   the excess of the true argument over u-1, at most one ulp of u, which
   the core folds into its constant term via its table reciprocal.  */
#define VM_HINV_LOG1P(y, a, tail, vl) do {				\
  vm_f_t vhl__u = VM_VFADD_VF ((a), 1.0, (vl));				\
  vm_f_t vhl__c = VM_VFADD_VV (						\
    VM_VFSUB_VV ((a), VM_VFSUB_VF (vhl__u, 1.0, (vl)), (vl)),		\
    (tail), (vl));							\
  VM_LNT_CORE ((y), vhl__u, VM_LNT_C2_TAIL (vhl__c), (vl));		\
} while (0)

/* asinh(x) = sign(x) * log1p(|x| + x^2/(1+sqrt(1+x^2))), valid for every
   finite |x| up to 2^28 (beyond, the x^2 term would overflow).  */
static inline vm_f_t
VM_KFN (asinh) (vm_f_t x, size_t vl)
{
  vm_u64_t vhi_b = VM_BITS (x);
  vm_u64_t vhi_ax = VM_VAND_VX (vhi_b, 0x7fffffffffffffffULL, vl);
  vm_b_t vhi_mtiny = VM_VMSLTU_VX (vhi_ax, 0x3e30000000000000ULL, vl);
  vm_b_t vhi_mhuge = VM_VMSGTU_VX (vhi_ax, 0x41b0000000000000ULL, vl);
  vm_f_t vhi_a = VM_VFABS_V (x, vl);
  /* |x| >= 2^28 and the subnormal/tiny lanes are parked on 1.0: the
     former would overflow x^2, the latter would underflow it (both
     spurious).  Their values are restored below.  */
  vm_f_t vhi_as = VM_VFMERGE_VFM (
    vhi_a, 1.0, VM_VMOR_MM (vhi_mhuge, vhi_mtiny, vl), vl);
  vm_f_t vhi_x2 = VM_VFMUL_VV (vhi_as, vhi_as, vl);
  vm_f_t vhi_s = VM_VFSQRT_V (VM_VFADD_VF (vhi_x2, 1.0, vl), vl);
  vm_f_t vhi_c = VM_VFDIV_VV (vhi_x2, VM_VFADD_VF (vhi_s, 1.0, vl), vl);
  /* u = RN(1 + |x| + c) and c' = (|x| - (u-1)) + c: exactly the excess of
     the true log1p argument 1+|x|+c over u, and (u-1) is exact here
     (|x| <= 2^28).  Two ops fewer than going through a = |x|+c.  */
  vm_f_t vhi_u = VM_VFADD_VV (VM_VFADD_VF (vhi_as, 1.0, vl), vhi_c, vl);
  vm_f_t vhi_t = VM_VFADD_VV (
    VM_VFSUB_VV (vhi_as, VM_VFSUB_VF (vhi_u, 1.0, vl), vl), vhi_c, vl);
  vm_f_t vhi_z;
  VM_LNT_CORE (vhi_z, vhi_u, VM_LNT_C2_TAIL (vhi_t), vl);
  vhi_z = VM_VFSGNJ_VV (vhi_z, x, vl);			/* odd function */
  vhi_z = VM_VFMERGE_VVM (vhi_z, x, vhi_mtiny, vl);	/* asinh(tiny) = tiny */
  if (VM_VFIRST_M (vhi_mhuge, vl) >= 0)
    {
      /* |x| > 2^28: sign(x) * (log|x| + ln2); non-finite lanes are above
	 2^28 too, and asinh(+-Inf) = +-Inf, asinh(NaN) = NaN (= x).  */
      vm_b_t vhi_mnf = VM_VMSGEU_VX (vhi_ax, 0x7ff0000000000000ULL, vl);
      vm_b_t vhi_mbig = VM_VMAND_MM (vhi_mhuge, VM_VMNOT_M (vhi_mnf, vl), vl);
      vm_f_t vhi_ah = VM_VFMERGE_VFM (vhi_a, 1.0, VM_VMNOT_M (vhi_mbig, vl), vl);
      vm_f_t vhi_lh, vhi_ll, vhi_sh, vhi_se;
      VM_LOG_DD (vhi_lh, vhi_ll, vhi_ah, vl, log);
      VM_TWO_SUM (vhi_sh, vhi_se, vhi_lh, VM_VFMV_V_F (VM_HINV_LN2, vl), vl);
      vhi_z = VM_VFMERGE_VVM (
	vhi_z,
	VM_VFSGNJ_VV (VM_VFADD_VV (
	  vhi_sh, VM_VFADD_VV (vhi_se, vhi_ll, vl), vl), x, vl),
	vhi_mbig, vl);
      vhi_z = VM_VFMERGE_VVM (
	vhi_z, VM_VFADD_VV_M (vhi_mnf, x, x, vl), vhi_mnf, vl);
    }
  return vhi_z;
}

/* acosh(x) = log1p(t + sqrt(2t + t^2)), t = x-1, for x in (1, 2^28];
   x > 2^28 takes the guarded log(x)+ln2 pass.  */
static inline vm_f_t
VM_KFN (acosh) (vm_f_t x, size_t vl)
{
  vm_u64_t vhi_b = VM_BITS (x);
  vm_u64_t vhi_ax = VM_VAND_VX (vhi_b, 0x7fffffffffffffffULL, vl);
  vm_b_t vhi_mdom = VM_VMSLT_VX (VM_REINT_U_I (vhi_b), 0x3ff0000000000000ULL,
				 vl);		/* x < 1 (signed, so <0 too) */
  vm_b_t vhi_mone = VM_VMSEQ_VX (vhi_b, 0x3ff0000000000000ULL, vl);
  vm_b_t vhi_mhuge = VM_VMSGTU_VX (vhi_ax, 0x41b0000000000000ULL, vl);
  vm_f_t vhi_xs = VM_VFMERGE_VFM (
    x, 2.0, VM_VMOR_MM (vhi_mdom, VM_VMOR_MM (vhi_mhuge, vhi_mone, vl), vl),
    vl);
  /* r = t*(2+t) exactly: two_prod of the product, and ts = RN(2+t) has
     its own rounding e2 = t-(ts-2) (exact) carried as t*e2.  */
  vm_f_t vhi_t = VM_VFSUB_VF (vhi_xs, 1.0, vl);		/* exact */
  vm_f_t vhi_ts = VM_VFADD_VF (vhi_t, 2.0, vl);
  vm_f_t vhi_rh, vhi_rl;
  VM_TWO_PROD (vhi_rh, vhi_rl, vhi_t, vhi_ts, vl);
  vm_f_t vhi_e2 = VM_VFSUB_VV (vhi_t, VM_VFSUB_VF (vhi_ts, 2.0, vl), vl);
  vhi_rl = VM_VFMACC_VV (vhi_rl, vhi_t, vhi_e2, vl);
  /* q = sqrt(r), one Newton correction for the sqrt rounding and rl,
     carried in the tail: sqrt(rh+rl) = q + (rh+rl-q^2)/(2q).  */
  vm_f_t vhi_q = VM_VFSQRT_V (vhi_rh, vl);
  vm_f_t vhi_d = VM_VFNMSAC_VV (vhi_rh, vhi_q, vhi_q, vl);
  vm_f_t vhi_e = VM_VFADD_VV (vhi_d, vhi_rl, vl);
  vm_f_t vhi_rc = VM_VFREC7_V (VM_VFADD_VV (vhi_q, vhi_q, vl), vl);
  vm_f_t vhi_corr = VM_VFMUL_VV (vhi_e, vhi_rc, vl);
  /* (A, tail) = q + t exactly (q >= t >= 0), plus the sqrt correction.  */
  vm_f_t vhi_A = VM_VFADD_VV (vhi_q, vhi_t, vl);
  vm_f_t vhi_tail = VM_VFADD_VV (
    VM_VFADD_VV (VM_VFSUB_VV (vhi_q, vhi_A, vl), vhi_t, vl), vhi_corr, vl);
  vm_f_t vhi_z;
  VM_HINV_LOG1P (vhi_z, vhi_A, vhi_tail, vl);
  vhi_z = VM_VFMERGE_VFM (vhi_z, 0.0, vhi_mone, vl);	/* acosh(1) = +0 */
  if (VM_VFIRST_M (vhi_mhuge, vl) >= 0)
    {
      /* x > 2^28: log(x) + ln2; acosh(+Inf) = +Inf, acosh(NaN) = NaN.  */
      vm_b_t vhi_mnf = VM_VMSGEU_VX (vhi_ax, 0x7ff0000000000000ULL, vl);
      vm_b_t vhi_mbig = VM_VMAND_MM (vhi_mhuge, VM_VMNOT_M (vhi_mnf, vl), vl);
      vm_f_t vhi_xh = VM_VFMERGE_VFM (x, 2.0, VM_VMNOT_M (vhi_mbig, vl), vl);
      vm_f_t vhi_lh, vhi_ll, vhi_sh, vhi_se;
      VM_LOG_DD (vhi_lh, vhi_ll, vhi_xh, vl, log);
      VM_TWO_SUM (vhi_sh, vhi_se, vhi_lh, VM_VFMV_V_F (VM_HINV_LN2, vl), vl);
      vhi_z = VM_VFMERGE_VVM (
	vhi_z, VM_VFADD_VV (vhi_sh, VM_VFADD_VV (vhi_se, vhi_ll, vl), vl),
	vhi_mbig, vl);
      vhi_z = VM_VFMERGE_VVM (
	vhi_z, VM_VFADD_VV_M (vhi_mnf, x, x, vl), vhi_mnf, vl);
    }
  {
    /* x < 1 -> NaN with INVALID, as fdlibm's (x-x)/(x-x); the masked
       quotient raises nothing on the other lanes.  Last, so it wins over
       the non-finite merge (acosh(-Inf) = NaN, not -Inf).  */
    vm_f_t vhi_dd = VM_VFSUB_VV_M (vhi_mdom, x, x, vl);
    return VM_VFMERGE_VVM (
      vhi_z, VM_VFDIV_VV_MU (vhi_mdom, VM_VFMV_V_F (0.0, vl),
			     vhi_dd, vhi_dd, vl), vhi_mdom, vl);
  }
}

/* atanh(x) = sign(x) * 0.5 * log1p(2|x| + 2|x|*|x|/(1-|x|)), |x| < 1.  */
static inline vm_f_t
VM_KFN (atanh) (vm_f_t x, size_t vl)
{
  vm_u64_t vhi_b = VM_BITS (x);
  vm_u64_t vhi_ax = VM_VAND_VX (vhi_b, 0x7fffffffffffffffULL, vl);
  vm_b_t vhi_mtiny = VM_VMSLTU_VX (vhi_ax, 0x3e30000000000000ULL, vl);
  /* |x| >= 1 covers Inf/NaN too, exactly as fdlibm's test does.  */
  vm_b_t vhi_mdom = VM_VMSGTU_VX (vhi_ax, 0x3ff0000000000000ULL, vl);
  vm_b_t vhi_mone = VM_VMSEQ_VX (vhi_ax, 0x3ff0000000000000ULL, vl);
  /* out-of-domain, |x| == 1 and the tiny lanes are parked on 0.25: their
     true values would divide by zero or underflow x^2.  */
  vm_f_t vhi_a = VM_VFABS_V (x, vl);
  vm_f_t vhi_as = VM_VFMERGE_VFM (
    vhi_a, 0.25, VM_VMOR_MM (vhi_mdom, VM_VMOR_MM (vhi_mone, vhi_mtiny, vl), vl),
    vl);
  vm_f_t vhi_t = VM_VFADD_VV (vhi_as, vhi_as, vl);	/* t = 2|x| */
  vm_f_t vhi_p = VM_VFMUL_VV (vhi_t, vhi_as, vl);	/* p = 2|x|^2 */
  vm_f_t vhi_pe = VM_VFNMSAC_VV (vhi_p, vhi_t, vhi_as, vl);	/* p - t|x| */
  vm_f_t vhi_d = VM_VFRSUB_VF (vhi_as, 1.0, vl);	/* 1-|x| */
  vm_f_t vhi_q = VM_VFDIV_VV (vhi_p, vhi_d, vl);	/* q = p/d */
  vm_f_t vhi_qe = VM_VFNMSAC_VV (vhi_p, vhi_q, vhi_d, vl);	/* p - q*d */
  /* true (p - pe)/d = q + (qe - pe)/d; the residual needs only the
     ~7 bits vfrec7 gives, it is a 2^-53 correction of the answer.  */
  vm_f_t vhi_rc = VM_VFREC7_V (vhi_d, vl);
  vm_f_t vhi_e = VM_VFMUL_VV (VM_VFSUB_VV (vhi_qe, vhi_pe, vl), vhi_rc, vl);
  /* (A, tail) = t + q exactly (either may dominate), plus the residual.  */
  vm_f_t vhi_A, vhi_e0;
  VM_TWO_SUM (vhi_A, vhi_e0, vhi_t, vhi_q, vl);
  vm_f_t vhi_z;
  VM_HINV_LOG1P (vhi_z, vhi_A, VM_VFADD_VV (vhi_e0, vhi_e, vl), vl);
  vhi_z = VM_VFMUL_VF (vhi_z, 0.5, vl);
  vhi_z = VM_VFSGNJ_VV (vhi_z, x, vl);			/* odd function */
  vhi_z = VM_VFMERGE_VVM (vhi_z, x, vhi_mtiny, vl);	/* atanh(tiny) = tiny */
  {
    /* |x| > 1 (Inf/NaN included) -> NaN with INVALID, as fdlibm's
       (x-x)/(x-x); the masked quotient raises nothing on other lanes.  */
    vm_f_t vhi_dd = VM_VFSUB_VV_M (vhi_mdom, x, x, vl);
    vhi_z = VM_VFMERGE_VVM (
      vhi_z, VM_VFDIV_VV_MU (vhi_mdom, VM_VFMV_V_F (0.0, vl),
			     vhi_dd, vhi_dd, vl), vhi_mdom, vl);
  }
  {
    /* |x| == 1 -> +-Inf with DIVBYZERO (fdlibm's x/zero).  */
    vm_f_t vhi_dd = VM_VFMERGE_VFM (x, 0.0, VM_VMNOT_M (vhi_mone, vl), vl);
    return VM_VFMERGE_VVM (
      vhi_z, VM_VFDIV_VV_MU (vhi_mone, VM_VFMV_V_F (0.0, vl),
			     vhi_dd, VM_VFMV_V_F (0.0, vl), vl), vhi_mone, vl);
  }
}
