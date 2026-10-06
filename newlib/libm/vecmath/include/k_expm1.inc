/* k_expm1.inc -- expm1 kernel for libvecmath.

   LMUL-generic template: include after vecmath_rvvcompat.h and
   vecmath_local.h (k_fixup.inc) with VM_LMUL defined.

   Vectorized fdlibm common/s_expm1.c: reduction x = k*ln2 + (xr + c)
   with the exact k*ln2_hi split, the scaled-Q polynomial for expm1(xr),
   and the source's (i)..(vii) branch matrix as masked selects.  Exponent
   scaling is SET_HIGH_WORD(y, high + (k << 20)), i.e. bits(y)+(k << 52).
   Special lanes (|x| >= 56 ln2, inf/NaN) are computed inline with their
   exact scalar flag behavior.  */

/* VM_EXPM1_DD expands VM_EXP_DD, so it also needs k_exp.inc at the same
   VM_LMUL (as k_hyper.inc does).  vecmath_dd.h is included for
   VM_TWO_SUM; the plain expm1 entry point does not expand VM_EXPM1_DD,
   so it pulls in neither the exp kernel nor the dd layer.  */
#include <vecmath_dd.h>

#define VM_LN2_HI_E1	6.93147180369123816490e-01
#define VM_LN2_LO_E1	1.90821492927058770002e-10
#define VM_INVLN2_E1	1.44269504088896338700e+00
#define VM_O_THRESH_E1	7.09782712893383973096e+02
#define VM_Q1_E1	-3.33333333333331316428e-02
#define VM_Q2_E1	 1.58730158725481460165e-03
#define VM_Q3_E1	-7.93650757867487942473e-05
#define VM_Q4_E1	 4.00821782732936239552e-06
#define VM_Q5_E1	-2.01099218183624371326e-07

static inline vm_f_t
VM_KFN (expm1) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ahx = VM_REINT_I_U (hx);
  ahx = VM_VAND_VX (ahx, 0x7fffffff, vl);
  vm_b_t m_neg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t m_tiny = VM_VMSLTU_VX (ahx, 0x3c900000, vl);
  /* fdlibm: `ix < 0x3fd62e43` for the |x| < ln2/2 branch and
     `ix < 0x3ff0a2b2` for the next.  */
  vm_b_t m_small = VM_VMSLTU_VX (VM_VSUB_VX (ahx, 0x3c900000, vl),
				 0x3fd62e43 - 0x3c900000, vl);
  vm_b_t m_mid = VM_VMSLTU_VX (VM_VSUB_VX (ahx, 0x3fd62e43, vl),
			       0x3ff0a2b2 - 0x3fd62e43, vl);
  vm_b_t m_bigred = VM_VMSLTU_VX (VM_VSUB_VX (ahx, 0x3ff0a2b2, vl),
				  0x4043687a - 0x3ff0a2b2, vl);
  vm_b_t m_vbig = VM_VMSGEU_VX (ahx, 0x4043687a, vl);
  vm_b_t m_infnan = VM_VMSGEU_VX (ahx, 0x7ff00000, vl);
  /* |x| < 2^-54: x - (t - (huge + x)) == x, with scalar's inexact.  */
  vm_f_t huge = VM_VFMV_V_F (1.0e+300, vl);
  /* masked: only tiny lanes may raise the inexact of scalar's huge+x.  */
  vm_f_t u = VM_VFADD_VV_M (m_tiny, x, huge, vl);
  vm_f_t y_tiny = VM_VFSUB_VV_M (m_tiny, x, VM_VFSUB_VV (u, u, vl), vl);

  /* |x| >= 56*ln2 block (scalar order): -inf -> -1; NaN -> x+x; finite
     positive above o_threshold -> overflow; other big lanes fall through
     to the reduction.  */
  vm_b_t m_bigfin = VM_VMAND_MM (m_vbig, VM_VMNOT_M (m_infnan, vl), vl);
  vm_b_t m_negbig = VM_VMAND_MM (m_bigfin, m_neg, vl);
  vm_b_t m_oflow = VM_VMAND_MM (
    m_bigfin, VM_VMNOT_M (m_neg, vl), vl);
  m_oflow = VM_VMAND_MM (m_oflow,
			 VM_VMFGT_VF (VM_VFMERGE_VFM (x, 0.0, m_infnan, vl),
				      VM_O_THRESH_E1, vl), vl);

  /* ---- reduction for the k != 0 red lanes ----
     The reduction set is the mid and general-k regions plus the finite
     non-negative |x| >= 56*ln2 fall-through.  Everything else (inf/NaN,
     overflow, negative big lanes -> -1) must stay out: their kf would
     overflow the int64 conversion and raise INVALID.  */
  vm_b_t m_red = VM_VMOR_MM (m_mid, m_bigred, vl);
  m_red = VM_VMOR_MM (m_red,
		      VM_VMAND_MM (VM_VMAND_MM (m_bigfin,
						VM_VMNOT_M (m_oflow, vl), vl),
				   VM_VMNOT_M (m_neg, vl), vl), vl);
  /* sanitized reduction input (0.1 = harmless filler).  */
  vm_f_t xred = VM_VFMERGE_VFM (x, 0.1, VM_VMNOT_M (m_red, vl), vl);
  vm_f_t hi_m = VM_VFMERGE_VVM (VM_VFSUB_VF (xred, VM_LN2_HI_E1, vl),
				VM_VFADD_VF (xred, VM_LN2_HI_E1, vl),
				m_neg, vl);
  vm_f_t lo_m = VM_VFMERGE_VFM (VM_VFMV_V_F (VM_LN2_LO_E1, vl),
				-VM_LN2_LO_E1, m_neg, vl);
  vm_i64_t k_m = VM_REINT_U_I (
    VM_VMERGE_VXM (VM_VMV_V_X (1, vl), -1, m_neg, vl));
  vm_f_t half = VM_VFMERGE_VFM (VM_VFMV_V_F (0.5, vl), -0.5, m_neg, vl);
  vm_f_t kf = VM_VFMACC_VF (half, VM_INVLN2_E1, xred, vl); /* ±0.5+x*invln2 */
  vm_i64_t k_g = VM_VFCVT_RTZ_X_F_V (kf, vl);		/* C truncation */
  vm_f_t kd_g = VM_VFCVT_F_X_V (k_g, vl);
  vm_f_t hi_g = VM_VFNMSAC_VF (xred, VM_LN2_HI_E1, kd_g, vl); /* x-kd*ln2hi */
  vm_f_t lo_g = VM_VFMUL_VF (kd_g, VM_LN2_LO_E1, vl);
  vm_f_t hi = VM_VFMERGE_VVM (hi_g, hi_m, m_mid, vl);
  vm_f_t lo = VM_VFMERGE_VVM (lo_g, lo_m, m_mid, vl);
  vm_i64_t k = VM_REINT_U_I (VM_VMERGE_VVM (
    VM_REINT_I_U (k_g), VM_REINT_I_U (k_m), m_mid, vl));
  /* k = 0 lanes (m_small): xr = x, c = 0.  Others: 0.1 (results masked). */
  vm_f_t xr = VM_VFMERGE_VFM (x, 0.1, VM_VMNOT_M (VM_VMOR_MM (m_red, m_small, vl), vl), vl);
  vm_f_t xr_red = VM_VFSUB_VV (hi, lo, vl);
  vm_f_t c_red = VM_VFSUB_VV (VM_VFSUB_VV (hi, xr_red, vl), lo, vl);
  xr = VM_VFMERGE_VVM (xr, xr_red, m_red, vl);
  vm_f_t c = VM_VFMERGE_VFM (c_red, 0.0, VM_VMNOT_M (m_red, vl), vl);

  /* ---- polynomial ---- */
  vm_f_t hfx = VM_VFMUL_VF (xr, 0.5, vl);
  vm_f_t hxs = VM_VFMUL_VV (xr, hfx, vl);
  vm_f_t p = VM_VFMV_V_F (VM_Q5_E1, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q4_E1, vl), hxs, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q3_E1, vl), hxs, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q2_E1, vl), hxs, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q1_E1, vl), hxs, p, vl);
  vm_f_t r1 = VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), hxs, p, vl);
  vm_f_t t3 = VM_VFNMSAC_VV (VM_VFMV_V_F (3.0, vl), r1, hfx, vl);
  vm_f_t e = VM_VFMUL_VV (hxs, VM_VFDIV_VV (VM_VFSUB_VV (r1, t3, vl),
					    VM_VFNMSAC_VV (VM_VFMV_V_F (6.0, vl),
							   xr, t3, vl), vl), vl);
  /* k == 0: xr - (xr*e - hxs) */
  vm_f_t y_k0 = VM_VFSUB_VV (xr, VM_VFMSAC_VV (hxs, xr, e, vl), vl);

  /* ---- k != 0 assembly ---- */
  vm_f_t e1 = VM_VFMACC_VV (VM_VFNEG_V (c, vl), xr, VM_VFSUB_VV (e, c, vl), vl);
  vm_f_t e2 = VM_VFSUB_VV (e1, hxs, vl);
  vm_f_t y_m1 = VM_VFMACC_VF (VM_VFMV_V_F (-0.5, vl), 0.5,
			      VM_VFSUB_VV (xr, e2, vl), vl);	/* 0.5(xr-e2)-0.5 */
  vm_b_t m_lt = VM_VMFLT_VF (xr, -0.25, vl);
  vm_f_t y1a = VM_VFMUL_VF (VM_VFSUB_VV (e2, VM_VFADD_VF (xr, 0.5, vl), vl),
			    -2.0, vl);
  vm_f_t y1b = VM_VFMACC_VF (VM_VFMV_V_F (1.0, vl), 2.0,
			     VM_VFSUB_VV (xr, e2, vl), vl);
  vm_f_t y_p1 = VM_VFMERGE_VVM (y1b, y1a, m_lt, vl);
  /* general: (1 - (e2 - xr)) scaled by 2^k, then -1. */
  vm_f_t yg = VM_VFSUB_VV (VM_VFMV_V_F (1.0, vl), VM_VFSUB_VV (e2, xr, vl), vl);
  vm_u64_t k52 = VM_VSLL_VX (VM_REINT_I_U (k), 52, vl);
  yg = VM_VFSUB_VF (VM_FROM_BITS (VM_VADD_VV (VM_BITS (yg), k52, vl)), 1.0, vl);
  /* k in [2,20): t = 1 - 2^-k (upper-word trick).  */
  vm_u64_t s2 = VM_VSRL_VV (VM_VMV_V_X (0x200000, vl), VM_REINT_I_U (k), vl);
  vm_f_t t20 = VM_FROM_BITS (VM_VSLL_VX (
    VM_VSUB_VV (VM_VMV_V_X (0x3ff00000, vl), s2, vl), 32, vl));
  vm_f_t y20 = VM_VFSUB_VV (t20, VM_VFSUB_VV (e2, xr, vl), vl);
  y20 = VM_FROM_BITS (VM_VADD_VV (VM_BITS (y20), k52, vl));
  /* k in [20,57]: t = 2^-k; y = xr - (e2 + t) + 1, scaled.  */
  vm_f_t t57 = VM_FROM_BITS (VM_VSLL_VX (
    VM_VSUB_VV (VM_VMV_V_X (0x3ff, vl), VM_REINT_I_U (k), vl), 52, vl));
  vm_f_t y57 = VM_VFADD_VF (VM_VFSUB_VV (xr, VM_VFADD_VV (e2, t57, vl), vl),
			    1.0, vl);
  y57 = VM_FROM_BITS (VM_VADD_VV (VM_BITS (y57), k52, vl));
  /* select by k: -1, +1, <=-2 || >56, [2,20), [20,57] */
  vm_b_t m_km1 = VM_VMSEQ_VX (VM_REINT_I_U (k), -1, vl);
  vm_b_t m_kp1 = VM_VMSEQ_VX (VM_REINT_I_U (k), 1, vl);
  vm_b_t m_kgen = VM_VMOR_MM (VM_VMSLE_VX (k, -2, vl),
			      VM_VMSGT_VX (k, 56, vl), vl);
  vm_b_t m_k20 = VM_VMAND_MM (VM_VMSGE_VX (k, 2, vl),
			      VM_VMSLT_VX (k, 20, vl), vl);
  vm_b_t m_k57 = VM_VMAND_MM (VM_VMSGE_VX (k, 20, vl),
			      VM_VMSLE_VX (k, 56, vl), vl);
  vm_f_t yk = VM_VFMERGE_VVM (yg, y_m1, m_km1, vl);
  yk = VM_VFMERGE_VVM (yk, y_p1, m_kp1, vl);
  yk = VM_VFMERGE_VVM (yk, y20, m_k20, vl);
  yk = VM_VFMERGE_VVM (yk, y57, m_k57, vl);
  yk = VM_VFMERGE_VVM (yk, yg, m_kgen, vl);

  /* ---- assemble regions ---- */
  vm_f_t y = VM_VFMERGE_VVM (yk, y_k0, m_small, vl);
  y = VM_VFMERGE_VVM (y, y_tiny, m_tiny, vl);
  {
    /* negative finite big: -1 with scalar's inexact (tiny - 1).  */
    vm_f_t yneg1 = VM_VFADD_VF (VM_VFMV_V_F (1.0e-300, vl), -1.0, vl);
    y = VM_VFMERGE_VVM (y, yneg1, m_negbig, vl);
    /* overflow: masked big*big raises OVERFLOW+INEXACT on those lanes
       only, and feeds the result so it stays live.  */
    vm_f_t yov = VM_VFMUL_VF_M (m_oflow, VM_VFMV_V_F (0x1p1000, vl),
				0x1p1000, vl);
    y = VM_VFMERGE_VVM (y, yov, m_oflow, vl);
    /* inf/NaN: +inf -> x; -inf -> -1 exactly; NaN -> x + x.  */
    vm_f_t xpx = VM_VFADD_VV_M (m_infnan, x, x, vl);
    vm_b_t m_neginf = VM_VMAND_MM (
      m_neg, VM_VMSEQ_VX (VM_BITS (x), 0xfff0000000000000ULL, vl), vl);
    vm_f_t yinf = VM_VFMERGE_VVM (xpx, VM_VFMV_V_F (-1.0, vl),
				  m_neginf, vl);
    y = VM_VFMERGE_VVM (y, yinf, m_infnan, vl);
  }
  return y;
}

/* Double-double expm1, for the |x| < 56*ln2 range the hyperbolic kernels
   feed it: sinh/cosh/tanh amplify the single-double kernel's error ~1:1,
   so carry (hi, lo).  Two regimes, chosen to avoid any cancellation:
     * 2^-54 <= |x| < ln2/2 -- the k = 0 polynomial assembled in dd:
       expm1(x) = x - (x*e - hxs), with hxs = x^2/2.
     * ln2/2 <= |x| < 56*ln2 -- exp(x) - 1; exp(x) >= sqrt(2) there, so
       the dd exp error grows by at most 3.41x and one exact two_sum
       suffices.
   Other lanes keep the plain kernel's value with lo = 0: |x| < 2^-54 is
   exactly x, and |x| >= 56*ln2 is never reached by sinh/cosh/tanh.  The
   plain kernel runs on every lane, so the gated flags (INVALID,
   DIVBYZERO, OVERFLOW, UNDERFLOW) stay bit-identical to scalar expm1.  */
#define VM_EXPM1_DD(hi, lo, x, vl) do {					\
  vm_i64_t vme1_hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, (vl));	\
  vm_u64_t vme1_ahx = VM_VAND_VX (VM_REINT_I_U (vme1_hx), 0x7fffffff, (vl)); \
  (hi) = VM_KFN (expm1) ((x), (vl));					\
  (lo) = VM_VFMV_V_F (0.0, (vl));					\
  /* Region masks over the dd range; both bounds match the plain		\
     kernel's.  Lanes outside it keep lo = 0.  */			\
  vm_b_t vme1_mk0 = VM_VMSLTU_VX (					\
    VM_VSUB_VX (vme1_ahx, 0x3c900000, (vl)),				\
    0x3fd62e43 - 0x3c900000, (vl));					\
  vm_b_t vme1_mex = VM_VMSLTU_VX (					\
    VM_VSUB_VX (vme1_ahx, 0x3fd62e43, (vl)),				\
    0x4043687a - 0x3fd62e43, (vl));					\
  /* ---- 2^-54 <= |x| < ln2/2: k = 0 polynomial, in dd.  */		\
  do {									\
    vm_f_t vme1_xr = VM_VFMERGE_VFM ((x), 0.25,				\
				     VM_VMNOT_M (vme1_mk0, (vl)), (vl)); \
    vm_f_t vme1_hfx = VM_VFMUL_VF (vme1_xr, 0.5, (vl));			\
    vm_f_t vme1_hxs = VM_VFMUL_VV (vme1_xr, vme1_hfx, (vl));		\
    /* hfx = 0.5*xr is exact, so this is x^2/2's only rounding.  */	\
    vm_f_t vme1_rhxs = VM_VFNEG_V (					\
      VM_VFNMSAC_VV (vme1_hxs, vme1_xr, vme1_hfx, (vl)), (vl));		\
    vm_f_t vme1_p = VM_VFMV_V_F (VM_Q5_E1, (vl));			\
    vme1_p = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q4_E1, (vl)), vme1_hxs, vme1_p, (vl)); \
    vme1_p = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q3_E1, (vl)), vme1_hxs, vme1_p, (vl)); \
    vme1_p = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q2_E1, (vl)), vme1_hxs, vme1_p, (vl)); \
    vme1_p = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q1_E1, (vl)), vme1_hxs, vme1_p, (vl)); \
    vm_f_t vme1_r1 = VM_VFMACC_VV (VM_VFMV_V_F (1.0, (vl)), vme1_hxs, vme1_p, (vl)); \
    vm_f_t vme1_t3 = VM_VFNMSAC_VV (VM_VFMV_V_F (3.0, (vl)), vme1_r1, vme1_hfx, (vl)); \
    vm_f_t vme1_e = VM_VFMUL_VV (					\
      vme1_hxs,								\
      VM_VFDIV_VV (VM_VFSUB_VV (vme1_r1, vme1_t3, (vl)),		\
		   VM_VFNMSAC_VV (VM_VFMV_V_F (6.0, (vl)), vme1_xr,	\
				  vme1_t3, (vl)), (vl)), (vl));		\
    /* u = xr*e - hxs exactly, as ph + (pl - rhxs); y = xr - u.  */	\
    vm_f_t vme1_ph = VM_VFMUL_VV (vme1_xr, vme1_e, (vl));		\
    vm_f_t vme1_pl = VM_VFNEG_V (					\
      VM_VFNMSAC_VV (vme1_ph, vme1_xr, vme1_e, (vl)), (vl));		\
    vm_f_t vme1_uh, vme1_ul;						\
    VM_TWO_SUM (vme1_uh, vme1_ul, vme1_ph,				\
		VM_VFNEG_V (vme1_hxs, (vl)), (vl));			\
    vme1_ul = VM_VFADD_VV (vme1_ul,					\
			   VM_VFSUB_VV (vme1_pl, vme1_rhxs, (vl)), (vl)); \
    vm_f_t vme1_yh, vme1_yl;						\
    VM_TWO_SUM (vme1_yh, vme1_yl, vme1_xr,				\
		VM_VFNEG_V (vme1_uh, (vl)), (vl));			\
    vme1_yl = VM_VFSUB_VV (vme1_yl, vme1_ul, (vl));			\
    (hi) = VM_VFMERGE_VVM ((hi), vme1_yh, vme1_mk0, (vl));		\
    (lo) = VM_VFMERGE_VVM ((lo), vme1_yl, vme1_mk0, (vl));		\
  } while (0);								\
  /* ---- ln2/2 <= |x| < 56*ln2: exp(x) - 1, exact two_sum.  */		\
  do {									\
    vm_f_t vme1_xe = VM_VFMERGE_VFM ((x), 1.0,				\
				     VM_VMNOT_M (vme1_mex, (vl)), (vl)); \
    vm_f_t vme1_eh, vme1_el, vme1_sh, vme1_sl;				\
    /* Both regions fed here are < 56*ln2 << 512, so exp's guard	\
       epilogue is unreachable and the scalar fixup never runs.  */	\
    VM_EXP_DD (vme1_eh, vme1_el, vme1_xe, (vl), exp);			\
    VM_TWO_SUM (vme1_sh, vme1_sl, vme1_eh, VM_VFMV_V_F (-1.0, (vl)), (vl)); \
    vme1_sl = VM_VFADD_VV (vme1_sl, vme1_el, (vl));			\
    (hi) = VM_VFMERGE_VVM ((hi), vme1_sh, vme1_mex, (vl));		\
    (lo) = VM_VFMERGE_VVM ((lo), vme1_sl, vme1_mex, (vl));		\
  } while (0);								\
} while (0)
