/* k_hyperinv.inc -- asinh / acosh / atanh kernels.

   LMUL-generic template: include after vecmath_rvvcompat.h, vecmath_dd.h
   and vecmath_local.h (k_fixup.inc) with VM_LMUL defined; also needs
   k_log.inc instantiated at the same LMUL for VM_LOG_DD / VM_LOG1P_DD.

   Ports fdlibm math/s_asinh.c, math/e_acosh.c and math/e_atanh.c.  All
   three are one log or log1p of a carefully formed argument, and the dd
   log's own error is only its final rounding, so the ARGUMENT decides the
   total (1.18 ulp with single-double argument formation).  Each argument
   is therefore built as an exact pair, with the piece the rounding threw
   away riding in afterwards:

       log1p(x + e) == log1p(x) + e/(1+x),   log(x + e) == log(x) + e/x.

   Range tests become masks, and each transcendental's argument is masked
   to the lanes that read it back, so a branch cannot raise a flag on a
   lane whose answer comes from elsewhere.  */

#define VM_HINV_LN2	6.93147180559945286227e-01

/* log1p of an exact pair: (arg, tail) came out of a two_sum.  */
#define VM_HINV_LOG1P(yh, yl, arg, tail, vl) do {			\
  VM_LOG1P_DD ((yh), (yl), (arg), (vl), log1p);				\
  (yl) = VM_VFADD_VV ((yl),						\
		      VM_VFDIV_VV ((tail),				\
				   VM_VFADD_VF ((arg), 1.0, (vl)), (vl)), \
		      (vl));						\
} while (0)

#define VM_HINV_LOG(yh, yl, arg, tail, vl) do {				\
  VM_LOG_DD ((yh), (yl), (arg), (vl), log);				\
  (yl) = VM_VFADD_VV ((yl), VM_VFDIV_VV ((tail), (arg), (vl)), (vl));	\
} while (0)

/* Add a double to a dd pair exactly (used for the "+ ln2" tails).  */
#define VM_HINV_ADD(yh, yl, c, vl) do {					\
  vm_f_t vh_a, vh_b;							\
  VM_TWO_SUM (vh_a, vh_b, (yh), VM_VFMV_V_F ((c), (vl)), (vl));		\
  (yh) = vh_a;								\
  (yl) = VM_VFADD_VV (vh_b, (yl), (vl));				\
} while (0)

/* asinh(x) = sign(x) * log(|x| + sqrt(x*x+1)).  */
static inline vm_f_t
VM_KFN (asinh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3e300000u, vl);	/* |x| < 2^-28 */
  vm_b_t mnonfin = VM_VMSGEU_VX (ix, 0x7ff00000u, vl);
  vm_b_t mhuge = VM_VMSGTU_VX (ix, 0x41b00000u, vl);	/* |x| > 2^28 */
  vm_b_t mbig = VM_VMSGTU_VX (ix, 0x40000000u, vl);	/* |x| > 2 */
  vm_b_t mskip = VM_VMOR_MM (mtiny, mnonfin, vl);
  vm_f_t ax = VM_VFABS_V (x, vl);
  vm_f_t w;
  {
    /* 2^-28 <= |x| <= 2: log1p(|x| + x^2/(1 + sqrt(1 + x^2))).  */
    vm_b_t ml = VM_VMAND_MM (VM_VMNOT_M (mbig, vl), VM_VMNOT_M (mskip, vl),
			     vl);
    vm_f_t a = VM_VFMERGE_VFM (ax, 1.0, VM_VMNOT_M (ml, vl), vl);
    vm_f_t t = VM_VFMUL_VV (a, a, vl);
    vm_f_t c = VM_VFDIV_VV (t, VM_VFADD_VF (
      VM_VFSQRT_V (VM_VFADD_VF (t, 1.0, vl), vl), 1.0, vl), vl);
    vm_f_t arg, tail, yh, yl;
    VM_TWO_SUM (arg, tail, a, c, vl);
    VM_HINV_LOG1P (yh, yl, arg, tail, vl);
    VM_DD_COLLAPSE (w, yh, yl, vl);
  }
  {
    /* 2 < |x| <= 2^28: log(2|x| + 1/(sqrt(x^2+1) + |x|)).  */
    vm_b_t mb = VM_VMAND_MM (mbig, VM_VMNOT_M (mhuge, vl), vl);
    vm_f_t a = VM_VFMERGE_VFM (ax, 1.5, VM_VMNOT_M (mb, vl), vl);
    vm_f_t s = VM_VFSQRT_V (VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), a, a, vl),
			    vl);
    vm_f_t inv = VM_VFRDIV_VF (VM_VFADD_VV (s, a, vl), 1.0, vl);
    vm_f_t arg, tail, yh, yl, r;
    VM_TWO_SUM (arg, tail, VM_VFADD_VV (a, a, vl), inv, vl);
    VM_HINV_LOG (yh, yl, arg, tail, vl);
    VM_DD_COLLAPSE (r, yh, yl, vl);
    w = VM_VFMERGE_VVM (w, r, mb, vl);
  }
  {
    /* |x| > 2^28: log(|x|) + ln2.  */
    /* non-finite lanes answer below, and must not reach the two_sum
       (inf - inf is NaN + INVALID) */
    vm_b_t mhf = VM_VMAND_MM (mhuge, VM_VMNOT_M (mnonfin, vl), vl);
    vm_f_t a = VM_VFMERGE_VFM (ax, 1.5, VM_VMNOT_M (mhf, vl), vl);
    vm_f_t yh, yl, r;
    VM_LOG_DD (yh, yl, a, vl, log);
    VM_HINV_ADD (yh, yl, VM_HINV_LN2, vl);
    VM_DD_COLLAPSE (r, yh, yl, vl);
    w = VM_VFMERGE_VVM (w, r, mhf, vl);
  }
  w = VM_VFMERGE_VVM (w, VM_VFNEG_V (w, vl), mneg, vl);
  w = VM_VFMERGE_VVM (w, x, mtiny, vl);		/* asinh(tiny) = tiny */
  return VM_VFMERGE_VVM (w, VM_VFADD_VV_M (mnonfin, x, x, vl), mnonfin, vl);
}

/* acosh(x) = log(x + sqrt(x*x-1)), defined for x >= 1.  */
static inline vm_f_t
VM_KFN (acosh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_u64_t lx = VM_VAND_VX (VM_BITS (x), 0xffffffffULL, vl);
  /* signed compare: fdlibm tests hx, so x < 1 catches every negative --
     and -Inf and -NaN with it, which is why this overrides the non-finite
     answer below rather than the other way round.  */
  vm_b_t mdom = VM_VMSLT_VX (hx, 0x3ff00000, vl);	/* x < 1 */
  vm_b_t mnonfin = VM_VMSGEU_VX (ix, 0x7ff00000u, vl);
  vm_b_t mhuge = VM_VMSGTU_VX (ix, 0x41b00000u, vl);	/* x > 2^28 */
  vm_b_t mone = VM_VMAND_MM (VM_VMSEQ_VX (ix, 0x3ff00000u, vl),
			     VM_VMSEQ_VX (lx, 0, vl), vl);	/* x == 1 */
  vm_b_t mbig = VM_VMSGTU_VX (ix, 0x40000000u, vl);	/* x > 2 */
  vm_f_t w;
  {
    /* 1 < x <= 2: log1p(t + sqrt(2t + t^2)) with t = x - 1 (exact).  */
    vm_b_t ml = VM_VMNOT_M (
      VM_VMOR_MM (VM_VMOR_MM (mbig, mone, vl),
		  VM_VMOR_MM (mdom, mnonfin, vl), vl), vl);
    vm_f_t t = VM_VFMERGE_VFM (VM_VFSUB_VF (x, 1.0, vl), 1.0,
			       VM_VMNOT_M (ml, vl), vl);
    vm_f_t q = VM_VFMACC_VV (VM_VFMUL_VF (t, 2.0, vl), t, t, vl);
    /* sqrt rounds too, and here it is not a correction but most of the
       argument (acosh(1+e) ~ sqrt(2e)), so it has to be dd.  */
    vm_f_t sh, sl, arg, tail, yh, yl;
    /* For x within an ulp of 1, 2t can be subnormal; the dd sqrt's
       Newton step then divides by a zero root and raises DIVBYZERO,
       which fdlibm's plain sqrt does not.  Those lanes keep the
       plain sqrt (its half ulp is all that grid holds anyway).  */
    vm_b_t mqn = VM_VMSGEU_VX (
      VM_VAND_VX (VM_BITS (q), 0x7fffffffffffffffULL, vl),
      (uint64_t) (1ULL << 52), vl);
    VM_DD_SQRT (sh, sl,
		VM_VFMERGE_VFM (q, 1.0, VM_VMNOT_M (mqn, vl), vl),
		VM_VFMV_V_F (0.0, vl), vl);
    sh = VM_VFMERGE_VVM (sh, VM_VFSQRT_V (q, vl), VM_VMNOT_M (mqn, vl),
			 vl);
    sl = VM_VFMERGE_VFM (sl, 0.0, VM_VMNOT_M (mqn, vl), vl);
    VM_TWO_SUM (arg, tail, t, sh, vl);
    tail = VM_VFADD_VV (tail, sl, vl);
    VM_HINV_LOG1P (yh, yl, arg, tail, vl);
    VM_DD_COLLAPSE (w, yh, yl, vl);
  }
  {
    /* 2 < x <= 2^28: log(2x - 1/(x + sqrt(x^2-1))).  */
    /* negative x with |x| > 2 lands here too, and a + sqrt(a^2-1) is ~0
       for it: fdlibm answers x < 1 first, so keep those lanes out.  */
    vm_b_t mb = VM_VMAND_MM (
      VM_VMAND_MM (mbig, VM_VMNOT_M (mhuge, vl), vl),
      VM_VMNOT_M (mdom, vl), vl);
    vm_f_t a = VM_VFMERGE_VFM (x, 3.0, VM_VMNOT_M (mb, vl), vl);
    vm_f_t s = VM_VFSQRT_V (VM_VFSUB_VF (VM_VFMUL_VV (a, a, vl), 1.0, vl), vl);
    vm_f_t inv = VM_VFRDIV_VF (VM_VFADD_VV (a, s, vl), 1.0, vl);
    vm_f_t arg, tail, yh, yl, r;
    VM_TWO_SUM (arg, tail, VM_VFADD_VV (a, a, vl),
		VM_VFNEG_V (inv, vl), vl);
    VM_HINV_LOG (yh, yl, arg, tail, vl);
    VM_DD_COLLAPSE (r, yh, yl, vl);
    w = VM_VFMERGE_VVM (w, r, mb, vl);
  }
  {
    /* x > 2^28: log(x) + ln2.  */
    vm_b_t mhf = VM_VMAND_MM (
      VM_VMAND_MM (mhuge, VM_VMNOT_M (mnonfin, vl), vl),
      VM_VMNOT_M (mdom, vl), vl);
    vm_f_t a = VM_VFMERGE_VFM (x, 3.0, VM_VMNOT_M (mhf, vl), vl);
    vm_f_t yh, yl, r;
    VM_LOG_DD (yh, yl, a, vl, log);
    VM_HINV_ADD (yh, yl, VM_HINV_LN2, vl);
    VM_DD_COLLAPSE (r, yh, yl, vl);
    w = VM_VFMERGE_VVM (w, r, mhf, vl);
  }
  w = VM_VFMERGE_VVM (w, VM_VFADD_VV_M (mnonfin, x, x, vl), mnonfin, vl);
  w = VM_VFMERGE_VFM (w, 0.0, mone, vl);	/* acosh(1) = 0 */
  {
    /* x < 1 -> NaN with INVALID, as fdlibm's (x-x)/(x-x); masked, since
       that quotient is 0/0 on every lane and only these may raise.  */
    vm_f_t z = VM_VFSUB_VV_M (mdom, x, x, vl);
    w = VM_VFMERGE_VVM (
      w, VM_VFDIV_VV_MU (mdom, VM_VFMV_V_F (0.0, vl), z, z, vl), mdom, vl);
  }
  return w;
}

/* atanh(x) = 0.5 * log1p(2x/(1-x)), defined on |x| <= 1.  */
static inline vm_f_t
VM_KFN (atanh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_u64_t lx = VM_VAND_VX (VM_BITS (x), 0xffffffffULL, vl);
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3e300000u, vl);	/* |x| < 2^-28 */
  /* ix > 1 covers Inf and NaN as well, exactly as fdlibm's test does.  */
  vm_b_t mdom = VM_VMSGTU_VX (ix, 0x3ff00000u, vl);	/* |x| > 1 */
  vm_b_t mone = VM_VMAND_MM (VM_VMSEQ_VX (ix, 0x3ff00000u, vl),
			     VM_VMSEQ_VX (lx, 0, vl), vl);	/* |x| == 1 */
  vm_b_t mhalf = VM_VMSLTU_VX (ix, 0x3fe00000u, vl);	/* |x| < 0.5 */
  vm_b_t mskip = VM_VMOR_MM (VM_VMOR_MM (mtiny, mdom, vl), mone, vl);
  vm_f_t ax = VM_VFABS_V (x, vl);
  vm_f_t w;
  {
    /* |x| < 0.5: t = 2|x|, 0.5 * log1p(t + t*|x|/(1 - |x|)).  */
    vm_b_t m1 = VM_VMAND_MM (mhalf, VM_VMNOT_M (mskip, vl), vl);
    vm_f_t a = VM_VFMERGE_VFM (ax, 0.25, VM_VMNOT_M (m1, vl), vl);
    vm_f_t t = VM_VFMUL_VF (a, 2.0, vl);
    /* t*x/(1-x), not t/(1-x): the factor |x| is what makes this a
       correction rather than a second copy of the leading term.  */
    vm_f_t c = VM_VFMUL_VV (
      VM_VFMUL_VV (t, a, vl),
      VM_VFRDIV_VF (VM_VFRSUB_VF (a, 1.0, vl), 1.0, vl), vl);
    vm_f_t arg, tail, yh, yl, r;
    VM_TWO_SUM (arg, tail, t, c, vl);
    VM_HINV_LOG1P (yh, yl, arg, tail, vl);
    VM_DD_COLLAPSE (r, yh, yl, vl);
    w = VM_VFMUL_VF (r, 0.5, vl);
  }
  {
    /* 0.5 <= |x| < 1: 0.5 * log1p(2|x|/(1 - |x|)).  The quotient's own
       rounding rides along the same way the sums above do.  */
    vm_b_t m2 = VM_VMAND_MM (VM_VMNOT_M (mhalf, vl),
			     VM_VMNOT_M (mskip, vl), vl);
    vm_f_t a = VM_VFMERGE_VFM (ax, 0.75, VM_VMNOT_M (m2, vl), vl);
    vm_f_t num = VM_VFADD_VV (a, a, vl);
    vm_f_t den = VM_VFRSUB_VF (a, 1.0, vl);
    vm_f_t q = VM_VFDIV_VV (num, den, vl);
    vm_f_t qr = VM_VFDIV_VV (VM_VFNMSAC_VV (num, q, den, vl), den, vl);
    vm_f_t yh, yl, r;
    VM_LOG1P_DD (yh, yl, q, vl, log1p);
    yl = VM_VFADD_VV (yl, VM_VFDIV_VV (qr, VM_VFADD_VF (q, 1.0, vl), vl), vl);
    VM_DD_COLLAPSE (r, yh, yl, vl);
    w = VM_VFMERGE_VVM (w, VM_VFMUL_VF (r, 0.5, vl), m2, vl);
  }
  w = VM_VFMERGE_VVM (w, VM_VFNEG_V (w, vl), mneg, vl);
  w = VM_VFMERGE_VVM (w, x, mtiny, vl);		/* atanh(tiny) = tiny */
  {
    /* |x| > 1 -> NaN with INVALID, as fdlibm's (x-x)/(x-x).  */
    vm_f_t z = VM_VFSUB_VV_M (mdom, x, x, vl);
    w = VM_VFMERGE_VVM (
      w, VM_VFDIV_VV_MU (mdom, VM_VFMV_V_F (0.0, vl), z, z, vl), mdom, vl);
  }
  {
    /* |x| == 1 -> +-Inf with DIVBYZERO (fdlibm's x/zero); the division
       carries the sign itself.  */
    vm_f_t z = VM_VFMERGE_VFM (x, 0.0, VM_VMNOT_M (mone, vl), vl);
    w = VM_VFMERGE_VVM (
      w, VM_VFDIV_VV_MU (mone, VM_VFMV_V_F (0.0, vl), z,
			 VM_VFMV_V_F (0.0, vl), vl), mone, vl);
  }
  return w;
}
