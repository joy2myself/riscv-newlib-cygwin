/* f32/k_misc.inc -- binary32 sqrt / rsqrt / cbrt / hypot kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_misc.inc,
   k_hypot.inc) active at VM_LMUL == 2*VM32_LMUL.

   sqrt is a native binary32 kernel (vfsqrt.v is correctly rounded) and
   hypot is native: x^2 + y^2 is formed as an exact double-float sum and
   the sqrt is refreshed by its Newton correction, so one final rounding
   is all the error left.  Covered domain: 2^-40 <= max(|x|,|y|) < 2^30
   and min(|x|,|y|) either zero or >= 2^-40 (every intermediate square is
   then a normal binary32, so no spurious gated flag).  Other lanes go
   through the per-lane scalar binary64 epilogue.  rsqrt keeps using the
   binary64 kernel (it was already faster than scalar); cbrt is native,
   see below.  */

#undef VM32_MISCF_FAST2_SUM
#undef VM32_MISCF_DEF_FIXUP

/* (s, e) = a + b exactly (Knuth two-sum).  */
#define VM32_MISCF_TWO_SUM(s, e, a, b, vl) do {				\
  vm32_f_t mts_s_ = VM32_VFADD_VV ((a), (b), (vl));			\
  vm32_f_t mts_v_ = VM32_VFSUB_VV (mts_s_, (a), (vl));			\
  (s) = mts_s_;								\
  (e) = VM32_VFADD_VV (							\
    VM32_VFSUB_VV ((a), VM32_VFSUB_VV (mts_s_, mts_v_, (vl)), (vl)),	\
    VM32_VFSUB_VV ((b), mts_v_, (vl)), (vl));				\
} while (0)

#undef VM32_MISCF_DEF_FIXUP
#define VM32_MISCF_DEF_FIXUP(name)					\
  __attribute__ ((noinline)) static vm32_f_t				\
  VM32_KFN (name) (vm32_f_t x, vm32_f_t y, vm32_f_t r, vm32_b_t m,	\
		   size_t vl, float (*fn) (float, float))		\
  {									\
    float vfx_x_[600], vfx_y_[600], vfx_r_[600];			\
    size_t vfx_n_ = (size_t) VM32_VCMP_POPC_M (m, vl);			\
    VM32_VSE ((float *) vfx_x_, VM32_VCOMPRESS (x, m, vl), vfx_n_);	\
    VM32_VSE ((float *) vfx_y_, VM32_VCOMPRESS (y, m, vl), vfx_n_);	\
    for (size_t vfx_i_ = 0; vfx_i_ < vfx_n_; vfx_i_++)			\
      vfx_r_[vfx_i_] = fn (vfx_x_[vfx_i_], vfx_y_[vfx_i_]);		\
    vm32_u_t vfx_idx_ = VM32_VIOTA_M (m, vl);				\
    vm32_f_t vfx_pick_ = VM32_VLUXEI32 (				\
      (const float *) vfx_r_, VM32_VSLL_VX (vfx_idx_, 2, vl), vl);	\
    return VM32_VFMERGE_VVM (r, vfx_pick_, m, vl);			\
  }

VM32_MISCF_DEF_FIXUP (hypot_fixup)

static inline float
VM32_KFN (hypot_fix) (float x, float y)
{
  return (float) hypot ((double) x, (double) y);
}

/* sqrt is a native binary32 kernel: vfsqrt.v is correctly rounded, so
   widening would only cost a pair of conversions.  */
static inline vm32_f_t
VM32_KFN (sqrt) (vm32_f_t x, size_t vl)
{
  return VM32_VFSQRT_V (x, vl);
}

VM32_W1 (rsqrt, rsqrt)

/* ------------------------------------------------------------------ */
/* cbrt: native binary32                                                */
/* ------------------------------------------------------------------ */
/* Shape (the research notes' T7 "higher-order Newton + one rounding
   step", sized for binary32's 25-bit target rather than binary64's 53):

     1. reduce exactly:  |x| = 2^(3k)*m with m in [1,8)  (k = floor(e/3),
        the 2^-3k scaling is a bit-exact power of two, so m is exact);
     2. y0 = P(m-1), a degree-5 fit of m^(-1/3) on [1,8) with P(0) = 1
        exactly (relative error 5.1e-3);
     3. one division-free Newton step for the *reciprocal* cube root,
        r <- r*(4 - m*r^3)/3, which is quadratic: 5.1e-3 -> 5.3e-5;
     4. q = 1/r -- the kernel's only divide -- then the exact residual
        of q^3 (two fma's recover both rounding terms of the cube) gives
        the Newton correction  (m - q^3)/(3*q^2) = (m - q^3)*r^2/3,
        which is a multiply, not a second divide;
     5. one final add rounds, then the exact 2^k scaling and the sign.

   That corrects q to O(5.3e-5^2) = O(2^-28.4) relative, so the result is
   0.5 ulp + 0.006 ulp; measured max_ulp over the FULL blob is 0.524 and
   0.526 over 4M random normals.  The excluded-domain argument sanitizes
   to 1.0, where the polynomial, the Newton step and the correction are
   all exact (P(0)=1, r=1, q=1, residual 0), so the fast path can never
   raise a flag on a lane whose value comes from the epilogue.

   Covered domain: every *normal* binary32 (2^-126 <= |x| < +Inf).  That
   includes the exactly-representable cube roots -- x = 2^(3k) reduces to
   m = 1 and returns exactly 2^k, which the blob's EXPECT_EXACT records
   require.  +-0, subnormals, +-Inf and NaN take the per-lane epilogue
   (the same answer the previous widen / binary64 kernel / narrow path
   gave), which also supplies the DEPENDS ON NOTHING per-lane semantics
   the length and alignment sweeps need.  */

#define VM32_CBRT_C0 -0.29401407864704954f
#define VM32_CBRT_C1 0.10894610054580914f
#define VM32_CBRT_C2 -0.024754170462872302f
#define VM32_CBRT_C3 0.0028789862502636484f
#define VM32_CBRT_C4 -0.00013108217795754218f

/* The wide fallback as a noinline call: fdlibm's binary64 cbrt is several
   times the size of the whole fast path, and inlined it would sit inside
   the strip loop (which is the whole function) on every iteration.  */
__attribute__ ((noinline)) static vm32_f_t
VM32_KFN (cbrt_wide) (vm32_f_t x, size_t vl)
{
  return VM32_NARROW (VM32_D64 (cbrt) (VM32_WIDEN (x, vl), vl), vl);
}

static inline vm32_f_t
VM32_KFN (cbrt) (vm32_f_t x, size_t vl)
{
  vm32_u_t ax = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, vl);
  vm32_b_t mbad = VM32_VMSLTU_VX (ax, 0x00800000u, vl);	/* 0, subnormal */
  mbad = VM32_VMOR_MM (mbad, VM32_VMSGEU_VX (ax, 0x7f800000u, vl), vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 1.0f, mbad, vl);
  vm32_u_t axs = VM32_VAND_VX (VM32_BITS (xs), 0x7fffffffu, vl);

  /* k = floor((E - 127)/3) = floor((E + 2)/3) - 43, the /3 by the exact
     n*171 >> 9 (E+2 <= 256, so the magic is exact here).  */
  vm32_u_t ie = VM32_VSRL_VX (axs, 23, vl);
  vm32_u_t ik = VM32_VSUB_VX (
    VM32_VSRL_VX (VM32_VMUL_VX (VM32_VADD_VX (ie, 2, vl), 0xabu, vl), 9, vl),
    43, vl);
  vm32_u_t i3k = VM32_VMUL_VX (ik, 3, vl);
  /* 2^-3k, and 2^k for the final scaling: exact powers of two.  */
  vm32_f_t scale = VM32_FROM_BITS (
    VM32_VSLL_VX (VM32_VSUB_VV (VM32_VMV_V_X (127, vl), i3k, vl), 23, vl));
  vm32_f_t sc = VM32_FROM_BITS (
    VM32_VSLL_VX (VM32_VADD_VX (ik, 127, vl), 23, vl));

  /* m = |x| * 2^-3k in [1,8) exactly; u = m - 1.  */
  vm32_f_t m = VM32_VFMUL_VV (VM32_FROM_BITS (axs), scale, vl);
  vm32_f_t u = VM32_VFSUB_VF (m, 1.0f, vl);

  /* y0 = 1 + u*P(u): P(0) = 0 is not what we want, so the last step adds
     the 1 with an fma, which is exact at u = 0.  */
  vm32_f_t p = VM32_VFMV_V_F (VM32_CBRT_C4, vl);
  p = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_CBRT_C3, vl), u, p, vl);
  p = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_CBRT_C2, vl), u, p, vl);
  p = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_CBRT_C1, vl), u, p, vl);
  p = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_CBRT_C0, vl), u, p, vl);
  vm32_f_t r = VM32_VFMACC_VV (VM32_VFMV_V_F (1.0f, vl), p, u, vl);

  /* One division-free Newton step for the reciprocal cube root:
     r <- r*(4 - m*r^3)/3.  */
  vm32_f_t rr = VM32_VFMUL_VV (r, r, vl);
  vm32_f_t t = VM32_VFMUL_VV (m, VM32_VFMUL_VV (rr, r, vl), vl);
  vm32_f_t w = VM32_VFNMSAC_VF (VM32_VFMV_V_F (4.0f / 3.0f, vl),
				1.0f / 3.0f, t, vl);
  r = VM32_VFMUL_VV (r, w, vl);

  /* q = 1/r, its cube's exact residual, and the Newton correction
     (m - q^3)/(3*q^2) = (m - q^3)*r^2/3.  */
  vm32_f_t q = VM32_VFRDIV_VF (r, 1.0f, vl);
  vm32_f_t q2 = VM32_VFMUL_VV (q, q, vl);
  vm32_f_t q3 = VM32_VFMUL_VV (q2, q, vl);
  vm32_f_t q2e = VM32_VFMSAC_VV (q2, q, q, vl);	/* q*q - q2 */
  vm32_f_t q3e = VM32_VFMACC_VV (VM32_VFMSAC_VV (q3, q2, q, vl),
				 q2e, q, vl);	/* (q2*q - q3) + q2e*q */
  vm32_f_t er = VM32_VFSUB_VV (VM32_VFSUB_VV (m, q3, vl), q3e, vl);
  vm32_f_t corr = VM32_VFMUL_VV (
    VM32_VFMUL_VV (er, VM32_VFMUL_VV (r, r, vl), vl),
    VM32_VFMV_V_F (1.0f / 3.0f, vl), vl);
  vm32_f_t y = VM32_VFADD_VV (q, corr, vl);

  y = VM32_VFMUL_VV (y, sc, vl);
  y = VM32_VFSGNJ_VV (y, xs, vl);

  if (VM32_VFIRST_M (mbad, vl) >= 0)
    y = VM32_VFMERGE_VVM (y, VM32_KFN (cbrt_wide) (x, vl), mbad, vl);
  return y;
}

static inline vm32_f_t
VM32_KFN (hypot) (vm32_f_t x, vm32_f_t y, size_t vl,
		  double (*fixup) (double, double))
{
  (void) fixup;
  vm32_u_t axb = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, vl);
  vm32_u_t ayb = VM32_VAND_VX (VM32_BITS (y), 0x7fffffffu, vl);
  vm32_b_t mbad = VM32_VMOR_MM (
    VM32_VMSGEU_VX (axb, 0x7f800000u, vl),
    VM32_VMSGEU_VX (ayb, 0x7f800000u, vl), vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (1.0f, vl), mbad, vl);
  vm32_f_t ys = VM32_VFMERGE_VVM (y, VM32_VFMV_V_F (1.0f, vl), mbad, vl);
  vm32_f_t ax = VM32_VFABS_V (xs, vl);
  vm32_f_t ay = VM32_VFABS_V (ys, vl);
  vm32_f_t a = VM32_VFMAX_VV (ax, ay, vl);
  vm32_f_t b = VM32_VFMIN_VV (ax, ay, vl);
  /* 2^-40 <= a < 2^30; b is 0 or >= 2^-40.  */
  vm32_b_t mtiny = VM32_VMFNE_VF (b, 0.0f, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMFGE_VF (a, 0x1p30f, vl), vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMFLT_VF (a, 0x1p-40f, vl), vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMAND_MM (
			 mtiny, VM32_VMFLT_VF (b, 0x1p-40f, vl), vl), vl);
  /* Re-sanitize with the full mask: the native arithmetic must never see
     a lane whose result comes from the epilogue, or its squares could
     raise a gated flag (underflow on a tiny square, invalid on Inf/Inf). */
  xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (1.0f, vl), mbad, vl);
  ys = VM32_VFMERGE_VVM (y, VM32_VFMV_V_F (1.0f, vl), mbad, vl);
  ax = VM32_VFABS_V (xs, vl);
  ay = VM32_VFABS_V (ys, vl);
  a = VM32_VFMAX_VV (ax, ay, vl);
  b = VM32_VFMIN_VV (ax, ay, vl);
  (void) a;
  (void) b;
  /* s + e = a^2 + b^2 exactly; sqrt refreshed by its Newton step.  */
  vm32_f_t pa = VM32_VFMUL_VV (ax, ax, vl);
  vm32_f_t pae = VM32_VFMSAC_VV (pa, ax, ax, vl);
  vm32_f_t pb = VM32_VFMUL_VV (ay, ay, vl);
  vm32_f_t pbe = VM32_VFMSAC_VV (pb, ay, ay, vl);
  vm32_f_t s, e;
  VM32_MISCF_TWO_SUM (s, e, pa, pb, vl);
  e = VM32_VFADD_VV (e, pae, vl);
  e = VM32_VFADD_VV (e, pbe, vl);
  vm32_f_t r = VM32_VFSQRT_V (s, vl);
  vm32_f_t d = VM32_VFNMSAC_VV (s, r, r, vl);
  d = VM32_VFADD_VV (d, e, vl);
  vm32_f_t rl = VM32_VFMUL_VV (d, VM32_VFRDIV_VF (r, 0.5f, vl), vl);
  vm32_f_t res = VM32_VFADD_VV (r, rl, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (hypot_fixup) (x, y, res, mbad, vl,
				   VM32_KFN (hypot_fix));
  return res;
}
