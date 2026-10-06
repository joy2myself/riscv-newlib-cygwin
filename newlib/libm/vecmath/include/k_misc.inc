/* k_misc.inc -- sqrt / rsqrt / cbrt kernels for libvecmath.

   LMUL-generic template: include after vecmath_rvvcompat.h (and
   vecmath_local.h for the bit constants) with VM_LMUL defined.  May be
   re-included at several VM_LMUL values (all names are VM_KFN-scoped).

     sqrt  -- vfsqrt, correctly rounded (<= 0.5 ulp).
     rsqrt -- correctly-rounded sqrt then correctly-rounded reciprocal:
              <= 1 ulp.
     cbrt  -- exact vectorization of fdlibm s_cbrt.c (same operation
              order and constants): <= 0.667 ulp, matching scalar.  */

static inline vm_f_t VM_KFN(sqrt)(vm_f_t x, size_t vl)
{
  return VM_VFSQRT_V(x, vl);
}

static inline vm_f_t VM_KFN(rsqrt)(vm_f_t x, size_t vl)
{
  /* s = sqrt(x) (NaN+invalid for x<0, exact for 0/inf/NaN); y = 1/s
     (divbyzero raised naturally for s == +-0).  One masked Newton step
     refines it:
       e  = 1 - x*y^2      (via r = x*y, e = 1 - r*y, both fused)
       y += y * (e/2)
     Masked to finite x > 0, so special lanes keep the raw exact result.
     Subnormal x legitimately raises UNDERFLOW from the sqrt.  */
  vm_f_t s = VM_VFSQRT_V(x, vl);
  vm_f_t y = VM_VFRDIV_VF(s, 1.0, vl);

  /* Refinement mask via vfclass (positive normal/subnormal lanes only):
     ordered compares (vmfgt/vmflt) are signaling in RVV -- they raise
     invalid even for qNaN operands -- so classification must go through
     vfclass, which never raises.  */
  vm_b_t mref = VM_VMSNE_VX(
    VM_VAND_VX(VM_VFCLASS_V(x, vl),
	       VM_CLS_POSNORM | VM_CLS_POSSUB, vl), 0, vl);
  vm_f_t r = VM_VFMUL_VV_M(mref, x, y, vl);		/* r = x*y ~ sqrt(x) */
  vm_f_t one = VM_VFMV_V_F(1.0, vl);
  /* e = 1 - r*y  (vfnmsac: vd - vs1*vs2, fused => exact residual).  */
  vm_f_t e = VM_VFNMSAC_VV_M(mref, one, r, y, vl);
  vm_f_t eh = VM_VFMUL_VF_M(mref, e, 0.5, vl);
  y = VM_VFMACC_VV_M(mref, y, y, eh, vl);		/* y += y*(e/2) */

  /* SYCL semantics: rsqrt(+-0) = +Inf (the -0 lane would give -Inf).  */
  vm_b_t mz = VM_VMFEQ_VF(x, 0.0, vl);	/* true for +0 and -0.  */
  vm_f_t pinf = VM_VINF(vl);
  y = VM_VFMERGE_VVM(y, pinf, mz, vl);
  return y;
}

/* Vectorized fdlibm cbrt (common/s_cbrt.c).  */
static inline vm_f_t VM_KFN(cbrt)(vm_f_t x, size_t vl)
{
  /* fdlibm constants.  */
  const uint64_t B1 = 715094163ULL;	/* (682-0.03306235651)*2^20 */
  const uint64_t B2 = 696219795ULL;	/* (664-0.03306235651)*2^20 */
  const double C = 5.42857142857142815906e-01;	/* 19/35     */
  const double D = -7.05306122448979611050e-01;	/* -864/1225 */
  const double E = 1.41428571428571436819e+00;	/* 99/70     */
  const double F = 1.60714285714285720630e+00;	/* 45/28     */
  const double G = 3.57142857142857150787e-01;	/* 5/14      */

  vm_f_t ax = VM_FROM_BITS(VM_VAND_VX(VM_BITS(x), VM_ABSMASK_U64, vl));
  vm_u64_t cls = VM_VFCLASS_V(ax, vl);
  vm_b_t m_nan = VM_VMSNE_VX(VM_VAND_VX(cls, VM_CLS_NAN, vl), 0, vl);
  vm_b_t m_inf = VM_VMSNE_VX(VM_VAND_VX(cls, VM_CLS_INF, vl), 0, vl);
  vm_b_t m_zero = VM_VMSNE_VX(VM_VAND_VX(cls, VM_CLS_ZERO, vl), 0, vl);
  vm_b_t m_sub = VM_VMSNE_VX(VM_VAND_VX(cls, VM_CLS_SUB, vl), 0, vl);
  vm_b_t m_special = VM_VMOR_MM(m_nan, m_inf, vl);
  vm_b_t m_safe = VM_VMOR_MM(m_special, m_zero, vl);

  /* |x| with subnormals scaled by 2^54 -- used for the exponent seed
     only.  _MU form: inactive (normal) lanes keep ax exactly; the masked
     multiply also avoids spurious overflow flags on huge ax.  */
  vm_f_t axn = VM_VFMUL_VF_MU(m_sub, ax, ax, 0x1p54, vl);
  /* Keep zero/NaN/Inf lanes off the arithmetic paths (their results are
     merged back at the end): substitute 1.0 in both the seed source and
     the Newton denominator, or garbage seeds from Inf/NaN bit patterns
     overflow intermediates and raise spurious flags.  */
  axn = VM_VFMERGE_VFM(axn, 1.0, m_safe, vl);
  /* Newton denominator: the original |x| (fdlibm divides by x, not by
     the scaled value -- B2 already compensates the 2^18 in the seed).  */
  vm_f_t axd = VM_VFMERGE_VFM(ax, 1.0, m_safe, vl);

  /* Rough seed to ~5 bits: t = bits((highword(axn)/3 + B) << 32).
     highword/3 via the exact 32-bit magic multiply (0xAAAAAAAB, >>33).
     For subnormals B2 = B1 - 18*2^20 folds out the 2^54 scaling
     (cbrt(2^54) = 2^18), so t seeds cbrt(|x|) directly.  */
  vm_u64_t highw = VM_VSRL_VX(VM_BITS(axn), 32, vl);
  vm_u64_t third = VM_VSRL_VX(VM_VMUL_VX(highw, 0xaaaaaaabULL, vl), 33, vl);
  vm_u64_t bvec = VM_VMERGE_VXM(VM_VMV_V_X(B1, vl), B2, m_sub, vl);
  vm_f_t t = VM_FROM_BITS(VM_VSLL_VX(VM_VADD_VV(third, bvec, vl), 32, vl));

  /* Newton step to ~23 bits (identical sequence to fdlibm).  */
  vm_f_t r = VM_VFDIV_VV(VM_VFMUL_VV(t, t, vl), axd, vl);
  vm_f_t s = VM_VFMACC_VV(VM_VFMV_V_F(C, vl), r, t, vl);	/* C + r*t */
  vm_f_t v = VM_VFADD_VV(VM_VFADD_VF(s, E, vl),
			 VM_VFRDIV_VF(s, D, vl), vl);		/* s+E+D/s */
  vm_f_t wgt = VM_VFMUL_VV(t, VM_VFADD_VF(VM_VFRDIV_VF(v, F, vl),
					   G, vl), vl);		/* t*(G+F/v) */
  t = wgt;

  /* Chop to 20 bits and round UP: t = bits((bits(t) + 2^32) & highmask),
     identical to fdlibm's INSERT_WORDS(high+1, 0).  */
  t = VM_FROM_BITS(VM_VAND_VX(VM_VADD_VX(VM_BITS(t), 0x100000000ULL, vl),
			      VM_HIGHWORD_U64, vl));

  /* Final Newton step to ~53 bits, error < 0.667 ulp (fdlibm).  */
  s = VM_VFMUL_VV(t, t, vl);		/* exact */
  r = VM_VFDIV_VV(axd, s, vl);
  wgt = VM_VFADD_VV(t, t, vl);
  r = VM_VFDIV_VV(VM_VFSUB_VV(r, t, vl), VM_VFADD_VV(wgt, r, vl), vl);
  t = VM_VFMACC_VV(t, t, r, vl);	/* t + t*r */

  /* Restore the sign of x.  */
  vm_f_t y = VM_VFSGNJX_VV(t, x, vl);

  /* Specials: NaN/Inf -> x+x (same as fdlibm, quiets sNaN and raises
     invalid for sNaN exactly like the scalar path); +-0 -> x.  */
  vm_f_t xpx = VM_VFADD_VV_M(m_special, x, x, vl);
  y = VM_VFMERGE_VVM(y, xpx, m_special, vl);
  y = VM_VFMERGE_VVM(y, x, m_zero, vl);
  return y;
}
