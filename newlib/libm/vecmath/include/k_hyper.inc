/* k_hyper.inc -- sinh / cosh / tanh kernels.

   LMUL-generic template: include after vecmath_rvvcompat.h,
   vecmath_dd.h and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.
   Also needs k_expm1.inc and k_exp.inc instantiated at the same LMUL.

   Ports fdlibm math/e_sinh.c, math/e_cosh.c and math/s_tanh.c.  All three
   are even or odd, so everything runs on |x| and the sign is re-applied at
   the end.  Range tests become masks, and the exp/expm1 arguments on lanes
   that fdlibm shortcuts (|x| < 2^-28 / 2^-55, +-Inf, NaN) are replaced by
   a benign 1.0, so a lane cannot raise a gated flag belonging to a branch
   it is not in.

   fdlibm's single-double recipe reaches 1.45 ulp here, so the formula is
   carried as a dd pair and collapsed once at the end: the |x| < 22 region
   feeds expm1's own error into the result, and the top region's
   exp(0.5|x|)^2 doubles whatever exp rounds off.  The regions between them
   scale exp by +-0.5, an exact scale, so a single-double exp suffices.  */

/* |x| >= 22: sinh/cosh use exp(|x|) up to log(DBL_MAX) and the two-step
   exp(0.5|x|)^2 above it (fdlibm's overflow-avoiding form).  */
#define VM_HYP_22	0x40360000u
#define VM_HYP_LOGE	0x40862e42u	/* log(DBL_MAX) */
#define VM_HYP_OFL	0x408633ceu	/* 710.475860073944 */
#define VM_HYP_OFL_LOW	0x8fb9f87du

/* Collapse the dd pair at the end of each kernel.  A plain add is wrong
   for the lanes that carry an exact signed zero -- tanh(+-0) = +-0 and
   sinh(+-0) = +-0 -- because -0 + +0 is +0.  Wherever lo is exactly zero
   the pair's value *is* hi, so take hi back on those lanes.  vmfeq is a
   quiet compare, so a lane holding NaN does not raise INVALID here.  */
#define VM_HYP_COLLAPSE(r, yh, yl, vl) do {				\
  (r) = VM_VFADD_VV ((yh), (yl), (vl));					\
  (r) = VM_VFMERGE_VVM ((r), (yh), VM_VMFEQ_VF ((yl), 0.0, (vl)), (vl)); \
} while (0)

static inline vm_f_t
VM_KFN (sinh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_u64_t lx = VM_VAND_VX (VM_BITS (x), 0xffffffffULL, vl);
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3e300000u, vl);	/* |x| < 2^-28 */
  vm_b_t mnonfin = VM_VMSGEU_VX (ix, 0x7ff00000u, vl);
  vm_b_t mbig = VM_VMSGEU_VX (ix, VM_HYP_22, vl);	/* |x| >= 22 */
  vm_b_t mtop = VM_VMSGEU_VX (ix, VM_HYP_LOGE, vl);
  vm_b_t muted = VM_VMOR_MM (
    VM_VMSLTU_VX (ix, VM_HYP_OFL, vl),
    VM_VMAND_MM (VM_VMSEQ_VX (ix, VM_HYP_OFL, vl),
		 VM_VMSLEU_VX (lx, VM_HYP_OFL_LOW, vl), vl), vl);
  /* fdlibm overflows for |x| > 710.4758, above the top region's start:
     these lanes must not fall through into the w^2 branch, where inf*inf
     inside a dd two_prod is NaN + INVALID.  */
  vm_b_t mof = VM_VMAND_MM (mtop, VM_VMNOT_M (muted, vl), vl);
  vm_f_t h = VM_VFMERGE_VFM (VM_VFMV_V_F (0.5, vl), -0.5, mneg, vl);
  /* lanes that take a shortcut must not feed the transcendentals */
  vm_b_t mskip = VM_VMOR_MM (mtiny, mnonfin, vl);
  vm_f_t ag = VM_VFMERGE_VFM (VM_VFABS_V (x, vl), 1.0, mskip, vl);
  vm_f_t yh, yl;
  {
    /* |x| < 22: t = expm1(|x|), sinh = h*(t + t/(t+1)), in dd.  fdlibm
       splits this into 2t - t*t/(t+1) below 1 and t + t/(t+1) above; the
       two are the same value, and dd needs just the one form.  */
    vm_f_t th, tl, dwh, dwl, qh, ql;
    /* expm1 is only read back for |x| < 22.  Above that it is masked to
       a benign 1.0 so its own overflow region (|x| > 709.78) cannot
       raise OVERFLOW on a lane whose answer comes from elsewhere.  */
    VM_EXPM1_DD (th, tl, VM_VFMERGE_VFM (ag, 1.0, mbig, vl), vl);
    VM_DD_ADD_D (dwh, dwl, th, tl, 1.0, vl);
    VM_DD_DIV (qh, ql, th, tl, dwh, dwl, vl);
    VM_DD_ADD (yh, yl, th, tl, qh, ql, vl);
    /* h is +-0.5: an exact scale, so it cannot disturb the pair.  */
    yh = VM_VFMUL_VV (h, yh, vl);
    yl = VM_VFMUL_VV (h, yl, vl);
  }
  {
    /* 22 <= |x| < log(DBL_MAX): h*exp(|x|).  The exp argument is masked
       to these lanes only: above log(DBL_MAX) it would overflow and
       raise a spurious flag on lanes whose result comes from the square
       below (cosh/sinh do not overflow until 710.4758).  */
    vm_b_t mb = VM_VMAND_MM (mbig, VM_VMNOT_M (mtop, vl), vl);
    vm_f_t ae = VM_VFMERGE_VFM (ag, 1.0, VM_VMNOT_M (mb, vl), vl);
    vm_f_t e = VM_KFN (exp) (ae, vl, exp);
    yh = VM_VFMERGE_VVM (yh, VM_VFMUL_VV (h, e, vl), mb, vl);
    yl = VM_VFMERGE_VFM (yl, 0.0, mb, vl);
  }
  {
    /* top range: h * exp(0.5|x|)^2, which also covers the overflow lanes
       (the square goes to +Inf and raises OVERFLOW there).  exp must be dd
       here: its 0.5 ulp is doubled by the square.  w itself is far from
       overflow (0.5*710.5 = 355.2), so the overflow lives entirely in the
       square; the mof lanes are sanitized out of it because inf*inf would
       inject a NaN into the dd two_prod and raise INVALID.  Their OVERFLOW
       comes from the masked product further down.  */
    vm_f_t wh, wl, uh, ul, vh, v2l;
    VM_EXP_DD (wh, wl, VM_VFMUL_VF (ag, 0.5, vl), vl, exp);
    wh = VM_VFMERGE_VFM (wh, 1.0, mof, vl);
    wl = VM_VFMERGE_VFM (wl, 0.0, mof, vl);
    /* (0.5*w)*w, never w*w first: the square overflows marginally
       earlier than the true product does.  */
    VM_DD_MUL_D (uh, ul, wh, wl, 0.5, vl);
    VM_DD_MUL (vh, v2l, uh, ul, wh, wl, vl);
    {
      vm_f_t sg = VM_VFMERGE_VFM (VM_VFMV_V_F (1.0, vl), -1.0, mneg, vl);
      vm_b_t mt = VM_VMAND_MM (mtop, VM_VMNOT_M (mof, vl), vl);
      yh = VM_VFMERGE_VVM (yh, VM_VFMUL_VV (sg, vh, vl), mt, vl);
      yl = VM_VFMERGE_VVM (yl, VM_VFMUL_VV (sg, v2l, vl), mt, vl);
    }
  }
  /* overflow: x*shuge (masked -- the product overflows for plenty of
     lanes that are not in this branch).  */
  yh = VM_VFMERGE_VVM (yh, VM_VFMUL_VV_M (mof, x, VM_VFMV_V_F (1.0e307, vl),
					  vl), mof, vl);
  yl = VM_VFMERGE_VFM (yl, 0.0, mof, vl);
  yh = VM_VFMERGE_VVM (yh, x, mtiny, vl);		/* sinh(tiny) = tiny */
  yl = VM_VFMERGE_VFM (yl, 0.0, mtiny, vl);
  yh = VM_VFMERGE_VVM (yh, VM_VFADD_VV_M (mnonfin, x, x, vl), mnonfin, vl);
  yl = VM_VFMERGE_VFM (yl, 0.0, mnonfin, vl);
  {
    vm_f_t vr;
    VM_HYP_COLLAPSE (vr, yh, yl, vl);
    return vr;
  }
}

static inline vm_f_t
VM_KFN (cosh) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_u64_t ix = VM_VAND_VX (
    VM_VSRL_VX (VM_BITS (x), 32, vl), 0x7fffffffULL, vl);
  vm_u64_t lx = VM_VAND_VX (VM_BITS (x), 0xffffffffULL, vl);
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3c800000u, vl);	/* |x| < 2^-55 */
  vm_b_t mhalf = VM_VMSLTU_VX (ix, 0x3fd62e43u, vl);	/* |x| < ln2/2 */
  vm_b_t mnonfin = VM_VMSGEU_VX (ix, 0x7ff00000u, vl);
  vm_b_t mbig = VM_VMSGEU_VX (ix, VM_HYP_22, vl);
  vm_b_t mtop = VM_VMSGEU_VX (ix, VM_HYP_LOGE, vl);
  vm_b_t muted = VM_VMOR_MM (
    VM_VMSLTU_VX (ix, VM_HYP_OFL, vl),
    VM_VMAND_MM (VM_VMSEQ_VX (ix, VM_HYP_OFL, vl),
		 VM_VMSLEU_VX (lx, VM_HYP_OFL_LOW, vl), vl), vl);
  /* fdlibm overflows for |x| > 710.4758, above the top region's start:
     these lanes must not fall through into the w^2 branch, where inf*inf
     inside a dd two_prod is NaN + INVALID.  */
  vm_b_t mof = VM_VMAND_MM (mtop, VM_VMNOT_M (muted, vl), vl);
  vm_b_t mskip = VM_VMOR_MM (mtiny, mnonfin, vl);
  vm_f_t ax = VM_VFABS_V (x, vl);
  vm_f_t ag = VM_VFMERGE_VFM (ax, 1.0, mskip, vl);
  vm_f_t yh, yl;
  {
    /* |x| < ln2/2: 1 + t^2/(2*(1+t)) with t = expm1(|x|), in dd.  */
    vm_f_t th, tl, dwh, dwl, t2h, t2l, d2h, d2l, qh, ql;
    /* Only |x| < ln2/2 reads expm1 back; mask the rest to a benign 1.0
       (see sinh) so their expm1 flags stay out of the result.  */
    VM_EXPM1_DD (th, tl, VM_VFMERGE_VFM (ag, 1.0, VM_VMNOT_M (mhalf, vl), vl),
		 vl);
    VM_DD_ADD_D (dwh, dwl, th, tl, 1.0, vl);
    VM_DD_MUL (t2h, t2l, th, tl, th, tl, vl);
    VM_DD_MUL_D (d2h, d2l, dwh, dwl, 2.0, vl);
    VM_DD_DIV (qh, ql, t2h, t2l, d2h, d2l, vl);
    VM_DD_ADD_D (yh, yl, qh, ql, 1.0, vl);
    yh = VM_VFMERGE_VFM (yh, 1.0, mtiny, vl);	/* cosh(tiny) = 1 */
    yl = VM_VFMERGE_VFM (yl, 0.0, mtiny, vl);
  }
  {
    /* ln2/2 <= |x| < 22: (exp(|x|) + 1/exp(|x|))/2, in dd: the 1/e term
       spends half an ulp of its own on top of what the /2 damping leaves
       of the exp error.  */
    vm_b_t mmid = VM_VMAND_MM (VM_VMNOT_M (mhalf, vl),
			       VM_VMNOT_M (mbig, vl), vl);
    vm_f_t vhalf = VM_VFMV_V_F (0.5, vl);
    vm_f_t vshalf = VM_VFMV_V_F (0.0, vl);
    vm_f_t ae = VM_VFMERGE_VFM (ag, 1.0, VM_VMNOT_M (mmid, vl), vl);
    vm_f_t eh, el, qh, ql, ph, pl, rh, rl;
    VM_EXP_DD (eh, el, ae, vl, exp);
    VM_DD_DIV (qh, ql, vhalf, vshalf, eh, el, vl);	/* (1/2)/e */
    VM_DD_MUL_D (ph, pl, eh, el, 0.5, vl);		/* e/2 */
    VM_DD_ADD (rh, rl, ph, pl, qh, ql, vl);
    yh = VM_VFMERGE_VVM (yh, rh, mmid, vl);
    yl = VM_VFMERGE_VVM (yl, rl, mmid, vl);
  }
  {
    /* 22 <= |x| < log(DBL_MAX): exp(|x|)/2, an exact +-0.5 scale.  */
    vm_b_t mb = VM_VMAND_MM (mbig, VM_VMNOT_M (mtop, vl), vl);
    vm_f_t ae = VM_VFMERGE_VFM (ag, 1.0, VM_VMNOT_M (mb, vl), vl);
    yh = VM_VFMERGE_VVM (yh, VM_VFMUL_VF (VM_KFN (exp) (ae, vl, exp), 0.5, vl),
			 mb, vl);
    yl = VM_VFMERGE_VFM (yl, 0.0, mb, vl);
  }
  {
    /* top range: exp(0.5|x|)^2/2, dd for the same reason as sinh.  */
    vm_f_t wh, wl, uh, ul, vh, v2l;
    VM_EXP_DD (wh, wl, VM_VFMUL_VF (ag, 0.5, vl), vl, exp);
    wh = VM_VFMERGE_VFM (wh, 1.0, mof, vl);
    wl = VM_VFMERGE_VFM (wl, 0.0, mof, vl);
    VM_DD_MUL_D (uh, ul, wh, wl, 0.5, vl);
    VM_DD_MUL (vh, v2l, uh, ul, wh, wl, vl);
    {
      vm_b_t mt = VM_VMAND_MM (mtop, VM_VMNOT_M (mof, vl), vl);
      yh = VM_VFMERGE_VVM (yh, vh, mt, vl);
      yl = VM_VFMERGE_VVM (yl, v2l, mt, vl);
    }
  }
  /* overflow lanes: fdlibm's __math_oflow(0) -> +Inf with overflow (the
     scalar fixup reproduces the exact flag behaviour).  */
  yh = VM_KFN (scalar_fixup) (x, yh, mof, vl, fixup);
  yl = VM_VFMERGE_VFM (yl, 0.0, mof, vl);
  yh = VM_VFMERGE_VVM (yh, VM_VFMUL_VV_M (mnonfin, x, x, vl), mnonfin, vl);
  yl = VM_VFMERGE_VFM (yl, 0.0, mnonfin, vl);
  {
    vm_f_t vr;
    VM_HYP_COLLAPSE (vr, yh, yl, vl);
    return vr;
  }
}

static inline vm_f_t
VM_KFN (tanh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t msmall = VM_VMSLTU_VX (ix, 0x3c800000u, vl);	/* |x| < 2^-55 */
  vm_b_t mnonfin = VM_VMSGEU_VX (ix, 0x7ff00000u, vl);
  vm_b_t mbig = VM_VMSGEU_VX (ix, VM_HYP_22, vl);	/* |x| >= 22 */
  vm_b_t mge1 = VM_VMSGEU_VX (ix, 0x3ff00000u, vl);	/* |x| >= 1 */
  vm_b_t mskip = VM_VMOR_MM (msmall, mnonfin, vl);
  vm_f_t ax = VM_VFMERGE_VFM (VM_VFABS_V (x, vl), 0.5, mskip, vl);
  vm_f_t yh, yl;
  {
    /* |x| < 1: z = -t/(t+2) with t = expm1(-2|x|)
       |x| >= 1: z = 1 - 2/(t+2) with t = expm1(2|x|)
       The division residual is what fdlibm loses: z is read off a
       correction term of size ~1 whose own 0.5 ulp, doubled and compared
       against a result of ~1.5, is the whole 0.73 ulp.  dd carries the
       quotient to ~2^-106, so only the final collapse rounds.  */
    vm_f_t vone = VM_VFMV_V_F (1.0, vl);
    vm_f_t vtwo = VM_VFMV_V_F (2.0, vl);
    vm_f_t vzero = VM_VFMV_V_F (0.0, vl);
    /* Each expm1 is masked to the lanes that read it back, at 0.5 (so
       the -2x / 2x multiplier keeps it in expm1's benign range): |x| >= 1
       would otherwise push 2|x| to infinity and put inf-inf through the
       dd two_sum -> spurious INVALID, and |x| >= 22 asks for expm1(44+)
       that nothing consumes.  */
    vm_b_t mz2 = VM_VMAND_MM (mge1, VM_VMNOT_M (mbig, vl), vl);
    vm_f_t a1 = VM_VFMERGE_VFM (ax, 0.5, mge1, vl);
    vm_f_t a2 = VM_VFMERGE_VFM (ax, 0.5, VM_VMNOT_M (mz2, vl), vl);
    vm_f_t t1h, t1l, dh, dl;
    VM_EXPM1_DD (t1h, t1l, VM_VFMUL_VF (a1, -2.0, vl), vl);
    VM_DD_ADD_D (dh, dl, t1h, t1l, 2.0, vl);
    VM_DD_DIV (yh, yl, t1h, t1l, dh, dl, vl);
    yh = VM_VFNEG_V (yh, vl);
    yl = VM_VFNEG_V (yl, vl);
    {
      vm_f_t t2h, t2l, dh2, dl2, qh, ql, rh, rl;
      VM_EXPM1_DD (t2h, t2l, VM_VFMUL_VF (a2, 2.0, vl), vl);
      VM_DD_ADD_D (dh2, dl2, t2h, t2l, 2.0, vl);
      VM_DD_DIV (qh, ql, vtwo, vzero, dh2, dl2, vl);
      VM_DD_SUB (rh, rl, vone, vzero, qh, ql, vl);	/* 1 - 2/(t+2) */
      yh = VM_VFMERGE_VVM (yh, rh, mge1, vl);
      yl = VM_VFMERGE_VVM (yl, rl, mge1, vl);
    }
    yh = VM_VFMERGE_VFM (yh, 1.0 - 1.0e-300, mbig, vl);	/* |x| > 22 */
    yl = VM_VFMERGE_VFM (yl, 0.0, mbig, vl);
    yh = VM_VFMERGE_VVM (yh, VM_VFNEG_V (yh, vl), mneg, vl);
    yl = VM_VFMERGE_VVM (yl, VM_VFNEG_V (yl, vl), mneg, vl);
  }
  /* tanh(tiny) = tiny: masked, because x*(x+1) overflows on the |x| >
     1e154 lanes that are nowhere near this branch.  */
  yh = VM_VFMERGE_VVM (yh, VM_VFMUL_VV_M (msmall, x, VM_VFADD_VF (x, 1.0, vl),
					  vl), msmall, vl);
  yl = VM_VFMERGE_VFM (yl, 0.0, msmall, vl);
  {
    /* +-Inf -> +-1 as 1/x +- 1 (1/x is 0 there and raises nothing); the
       reciprocal input is masked to the non-finite lanes, so neither the
       +-0 lanes (division by zero) nor the huge finite ones (1/1.8e308
       underflows to subnormal) contribute a flag.  NaN stays NaN.  */
    vm_f_t iv = VM_VFRDIV_VF (
      VM_VFMERGE_VFM (x, 1.0, VM_VMNOT_M (mnonfin, vl), vl), 1.0, vl);
    vm_f_t s = VM_VFMERGE_VVM (VM_VFADD_VF (iv, 1.0, vl),
			       VM_VFSUB_VF (iv, 1.0, vl), mneg, vl);
    yh = VM_VFMERGE_VVM (yh, s, mnonfin, vl);
    yl = VM_VFMERGE_VFM (yl, 0.0, mnonfin, vl);
  }
  {
    vm_f_t vr;
    VM_HYP_COLLAPSE (vr, yh, yl, vl);
    return vr;
  }
}
