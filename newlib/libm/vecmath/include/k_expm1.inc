/* k_expm1.inc -- expm1 kernel for libvecmath.

   LMUL-generic template: include after vecmath_rvvcompat.h and
   vecmath_local.h (k_fixup.inc) with VM_LMUL defined.

   One reduction, one polynomial, one reconstruction.  Every lane with
   |x| >= 2^-54 is reduced with k = rint(x/ln2) and the exact two-word
   split hi = x - k*ln2_hi, lo = k*ln2_lo, xr = hi - lo, c = (hi-xr)-lo;
   fdlibm's rational gives e ~= (expm1(xr) - xr - xr^2/2)/xr, and the
   result is rebuilt with a single fused multiply-add

       y = fma (2^k, u, 2^k - 1),   u = xr - (xr*(e-c) - c - hxs).

   For k == 0 that is fdlibm's k == 0 formula exactly (2^0 = 1 and
   2^0 - 1 = 0 leave the FMA returning u = xr - (xr*e - hxs)), so no
   per-k case split survives: 2^k is one integer shift and 2^k - 1 one
   subtraction, and the single FMA rounds the whole reconstruction once,
   where fdlibm's 1-2^-k / 2^-k / 1-(e-x) sub-cases each needed their own
   trick to keep the trailing "-1" exact.

   Special lanes are one clamp plus two merges.  x <= -56 ln2 (and -inf)
   clamps to -56 ln2: k = -56 and |2^k*u| <= 2^-56*1.5 < ulp(1)/2, so the
   FMA returns exactly -1 as the scalar does.  x >= 709.78 (and +inf)
   clamps to 710: k = 1024, so the FMA overflows to +inf with
   OVERFLOW|INEXACT, which is the scalar's huge*huge.  k == 1024 is the
   one k whose 2^k is not representable, so the reconstruction clamps k
   to 1023 there and feeds 2u+1 instead of u (2^1024*(1+u) =
   2^1023*(2+2u)); +inf and NaN stay out of the reduction entirely and
   are merged back as x+x, so a NaN keeps its payload and -inf still
   takes the -1 path above.  */

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

/* 0x4043687A00000000: fdlibm's |x| >= 56*ln2 boundary, i.e. the first
   input for which expm1(x) rounds to exactly -1.  */
#define VM_N56LN2_E1	(-3.88162231445312500000e+01)
/* Clamp target for the overflow side: any x >= o_threshold maps here and
   the reconstruction's FMA then overflows to +inf.  */
#define VM_PBIG_E1	7.10e+02

/* One-off wrappers not carried by the shared compat layer.  */
#define VM_E1_VMIN_VX(a, s, vl) VM_I64FN (vmin_vx_) (a, s, vl)

static inline vm_f_t
VM_KFN (expm1) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ahx = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  /* Reduction set: 2^-54 <= |x| < +inf, i.e. the finite lanes that are
     not tiny, plus -inf (which clamps to -56 ln2 and lands on -1 like
     every other x <= -56 ln2).  +inf, NaN and the tiny lanes stay out and
     are merged back at the end.  Both tests are integer: an FP compare
     raises INVALID on a quiet NaN on some implementations, and the high
     word alone cannot tell -inf from a NaN whose payload lives in the low
     word.  */
  vm_b_t m_range = VM_VMSLTU_VX (VM_VSUB_VX (ahx, 0x3c900000, vl),
				 0x7ff00000 - 0x3c900000, vl);
  vm_b_t m_neginf = VM_VMSEQ_VX (VM_BITS (x), 0xfff0000000000000ULL, vl);
  vm_b_t m_red = VM_VMOR_MM (m_range, m_neginf, vl);
  vm_b_t m_excl = VM_VMNOT_M (m_red, vl);
  vm_b_t m_infnan = VM_VMSGEU_VX (ahx, 0x7ff00000, vl);
  /* Clamped, sanitized reduction input: |k| <= 56 for the -1 region and
     k = 1024 for the overflow region, so the exponent construction below
     can never leave [-1023, 1024].  */
  vm_f_t xc = VM_VFMAX_VF (x, VM_N56LN2_E1, vl);
  xc = VM_VFMIN_VF (xc, VM_PBIG_E1, vl);
  vm_f_t xred = VM_VFMERGE_VFM (xc, 1.0, m_excl, vl);

  /* ---- argument reduction ---- */
  vm_f_t kf = VM_VFMUL_VF (xred, VM_INVLN2_E1, vl);
  vm_i64_t k = VM_VFCVT_X_F_V (kf, vl);		/* rint (round to nearest) */
  vm_f_t kd = VM_VFCVT_F_X_V (k, vl);
  vm_f_t hi = VM_VFNMSAC_VF (xred, VM_LN2_HI_E1, kd, vl); /* x-kd*ln2hi */
  vm_f_t lo = VM_VFMUL_VF (kd, VM_LN2_LO_E1, vl);
  vm_f_t xr = VM_VFSUB_VV (hi, lo, vl);
  vm_f_t c = VM_VFSUB_VV (VM_VFSUB_VV (hi, xr, vl), lo, vl);

  /* ---- polynomial: e ~ (expm1(xr) - xr - xr^2/2)/xr ---- */
  vm_f_t hfx = VM_VFMUL_VF (xr, 0.5, vl);
  vm_f_t hxs = VM_VFMUL_VV (xr, hfx, vl);
  /* Estrin over hxs: one extra vfmul for z2 buys a 3-deep instead of
     5-deep chain, which measured 5% faster on muse-pi (29.3 vs 30.9
     ns/element) for 1.5% more instructions.  */
  vm_f_t z2 = VM_VFMUL_VV (hxs, hxs, vl);
  vm_f_t pa = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q1_E1, vl), hxs,
			    VM_VFMV_V_F (VM_Q2_E1, vl), vl);
  vm_f_t pb = VM_VFMACC_VV (VM_VFMV_V_F (VM_Q3_E1, vl), hxs,
			    VM_VFMV_V_F (VM_Q4_E1, vl), vl);
  pb = VM_VFMACC_VV (pb, z2, VM_VFMV_V_F (VM_Q5_E1, vl), vl);
  vm_f_t p = VM_VFMACC_VV (pa, z2, pb, vl);
  vm_f_t r1 = VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), hxs, p, vl);
  vm_f_t t3 = VM_VFNMSAC_VV (VM_VFMV_V_F (3.0, vl), r1, hfx, vl);
  vm_f_t e = VM_VFMUL_VV (hxs, VM_VFDIV_VV (VM_VFSUB_VV (r1, t3, vl),
					    VM_VFNMSAC_VV (VM_VFMV_V_F (6.0, vl),
							   xr, t3, vl), vl), vl);

  /* ---- reconstruction ---- */
  vm_f_t ec = VM_VFSUB_VV (e, c, vl);
  vm_f_t e2 = VM_VFMSAC_VV (c, xr, ec, vl);	/* xr*(e-c) - c */
  e2 = VM_VFSUB_VV (e2, hxs, vl);
  vm_f_t u = VM_VFSUB_VV (xr, e2, vl);
  /* k == 1024 has no representable 2^k: clamp the exponent to 1023 and
     feed 2u+1 (2^1024*(1+u) == 2^1023*(2+2u)); for k <= 1023 the merge
     keeps u untouched.  */
  vm_i64_t kk = VM_E1_VMIN_VX (k, 1023, vl);
  vm_u64_t abits = VM_VADD_VX (VM_VSLL_VX (VM_REINT_I_U (kk), 52, vl),
			       0x3ffULL << 52, vl);
  vm_f_t a = VM_FROM_BITS (abits);
  vm_f_t b = VM_VFSUB_VF (a, 1.0, vl);
  vm_b_t m_k24 = VM_VMSGT_VX (k, 1023, vl);
  vm_f_t u2 = VM_VFMACC_VF (VM_VFMV_V_F (1.0, vl), 2.0, u, vl);
  vm_f_t uu = VM_VFMERGE_VVM (u, u2, m_k24, vl);
  vm_f_t y = VM_VFMACC_VV (b, a, uu, vl);
  /* k == 1 with xr < -0.25 is the one sub-region where the plain
     reconstruction is a full ulp loose: there |u| crosses 0.25, so u's
     own rounding (ulp 2^-54) is amplified by 2^1 into a whole ulp of the
     result.  fdlibm's 0.5-shifted form computes xr + 0.5 - e2 at half
     the final scale, which buys the bit back.  */
  {
    vm_b_t m_ltq = VM_VMFLT_VF (xr, -0.25, vl);
    vm_b_t m_k1 = VM_VMSEQ_VX (VM_REINT_I_U (k), 1, vl);
    vm_b_t m_y1a = VM_VMAND_MM (m_ltq, m_k1, vl);
    vm_f_t y1a = VM_VFMUL_VF (
      VM_VFSUB_VV (e2, VM_VFADD_VF (xr, 0.5, vl), vl), -2.0, vl);
    y = VM_VFMERGE_VVM (y, y1a, m_y1a, vl);
  }

  /* ---- special lanes ---- */
  vm_f_t xpx = VM_VFADD_VV_M (m_infnan, x, x, vl);
  vm_f_t vy = VM_VFMERGE_VVM (x, xpx, m_infnan, vl);
  y = VM_VFMERGE_VVM (y, vy, m_excl, vl);
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
