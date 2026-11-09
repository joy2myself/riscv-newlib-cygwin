/* f32/k_hyperinv.inc -- binary32 inverse hyperbolic kernels
   (asinh / acosh / atanh).

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h, f32/wrap.h and f32/k_log.inc (for the native log
   kernels and the double-float log reduction), with the binary64 kernels
   (k_log.inc, k_hyperinv.inc) active at VM_LMUL == 2*VM32_LMUL.

   Native binary32 fast path over 2^-20 <= |x| <= 20 (asinh), 1+2^-23 <=
   x <= 20 (acosh) and 2^-30 <= |x| <= 0.9 (atanh); every other lane goes
   through the per-lane scalar binary64 epilogue, which reproduces the
   value and gated flag behaviour of the previous widen / binary64 kernel
   / narrow path.

   Accuracy: each function forms its log argument as a double-float (the
   sqrt step in asinh/acosh is refreshed by a Newton correction, and the
   log's own tail is carried as a second word), so the single final
   rounding is all the error left.  */

/* ------------------------------------------------------------------ */
/* f32 double-float helpers (file-local names).                       */
/* ------------------------------------------------------------------ */

#undef VM32_HINV_FAST2_SUM
#undef VM32_HINV_DD_SQRT
#undef VM32_HINV_LOG_SUM_DD
#undef VM32_HINV_DEF_FIXUP

/* (s, e) = a + b exactly, requires |a| >= |b|.  */
#define VM32_HINV_FAST2_SUM(s, e, a, b, vl) do {			\
  (s) = VM32_VFADD_VV ((a), (b), (vl));					\
  (e) = VM32_VFSUB_VV ((b), VM32_VFSUB_VV ((s), (a), (vl)), (vl));	\
} while (0)

/* (rh, rl) = sqrt(sh + sl), sh a normal positive binary32, sl its tail.
   rh is the correctly rounded sqrt of sh and rl the Newton correction
   covering both the sqrt rounding and sl.  */
#define VM32_HINV_DD_SQRT(rh, rl, sh, sl, vl) do {			\
  vm32_f_t hsq_r_ = VM32_VFSQRT_V ((sh), (vl));				\
  vm32_f_t hsq_d_ = VM32_VFNMSAC_VV ((sh), hsq_r_, hsq_r_, (vl));	\
  hsq_d_ = VM32_VFADD_VV (hsq_d_, (sl), (vl));				\
  vm32_f_t hsq_i_ = VM32_VFRDIV_VF (hsq_r_, 0.5f, (vl));		\
  (rh) = hsq_r_;							\
  (rl) = VM32_VFMUL_VV (hsq_d_, hsq_i_, (vl));				\
} while (0)

/* (h, l) = log(x) for a double-float x = xh + xl (normal positive xh);
   log(xh + xl) = log(xh) + u - u^2/2 with u = xl/xh, |u| <= 2^-24.  */
#define VM32_HINV_LOG_SUM_DD(h, l, xh, xl, vl) do {			\
  vm32_f_t hld_h_, hld_l_;						\
  VM32_LOGF_LOG_DD (hld_h_, hld_l_, (xh), (vl));			\
  vm32_f_t hld_c_ = VM32_VFDIV_VV ((xl), (xh), (vl));			\
  vm32_f_t hld_t_ = VM32_VFADD_VV (hld_l_, hld_c_, (vl));		\
  hld_t_ = VM32_VFNMSAC_VF (hld_t_, 0.5f,				\
			    VM32_VFMUL_VV (hld_c_, hld_c_, (vl)), (vl));\
  (h) = hld_h_;								\
  (l) = hld_t_;								\
} while (0)

/* ---- per-lane scalar epilogue -------------------------------------- */

#undef VM32_HINV_DEF_FIXUP
#define VM32_HINV_DEF_FIXUP(name)					\
  __attribute__ ((noinline)) static vm32_f_t				\
  VM32_KFN (name) (vm32_f_t x, vm32_f_t r, vm32_b_t m, size_t vl,	\
		   float (*fn) (float))					\
  {									\
    float vfx_x_[600], vfx_r_[600];					\
    size_t vfx_n_ = (size_t) VM32_VCMP_POPC_M (m, vl);			\
    VM32_VSE ((float *) vfx_x_, VM32_VCOMPRESS (x, m, vl), vfx_n_);	\
    for (size_t vfx_i_ = 0; vfx_i_ < vfx_n_; vfx_i_++)			\
      vfx_r_[vfx_i_] = fn (vfx_x_[vfx_i_]);				\
    vm32_u_t vfx_idx_ = VM32_VIOTA_M (m, vl);				\
    vm32_f_t vfx_pick_ = VM32_VLUXEI32 (				\
      (const float *) vfx_r_, VM32_VSLL_VX (vfx_idx_, 2, vl), vl);	\
    return VM32_VFMERGE_VVM (r, vfx_pick_, m, vl);			\
  }

VM32_HINV_DEF_FIXUP (asinh_fixup)
VM32_HINV_DEF_FIXUP (acosh_fixup)
VM32_HINV_DEF_FIXUP (atanh_fixup)

static inline float
VM32_KFN (asinh_fix) (float x)
{
  return (float) asinh ((double) x);
}

static inline float
VM32_KFN (acosh_fix) (float x)
{
  return (float) acosh ((double) x);
}

static inline float
VM32_KFN (atanh_fix) (float x)
{
  return (float) atanh ((double) x);
}

/* ---- asinh --------------------------------------------------------- */

/* |x| < 0.1 uses an odd polynomial; above that log(|x| + sqrt(x^2+1)).
   NaN/Inf/0 and |x| >= 20 go to the epilogue.  */
#define VM32_HINV_BAD_ASINH(x, vl)					\
  VM32_VMOR_MM (							\
    VM32_VMSGEU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x41a00000u, (vl)),					\
    VM32_VMSLTU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x35800000u, (vl)), (vl))

static inline vm32_f_t
VM32_KFN (asinh) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32_HINV_BAD_ASINH (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (1.0f, (vl)), mbad, vl);
  vm32_f_t ax = VM32_VFABS_V (xs, vl);
  /* |x| <= 0.1: x + x*t*G(t), t = x^2, G deg-3 minimax.  */
  vm32_f_t t = VM32_VFMUL_VV (xs, xs, vl);
  vm32_f_t G = VM32_VFMUL_VF (t, 0.029939228294132231f, vl);
  G = VM32_VFADD_VF (G, -0.044640099259263884f, vl);
  G = VM32_VFMACC_VV (VM32_VFMV_V_F (0.074999994491439154f, vl), t, G, vl);
  G = VM32_VFMACC_VV (VM32_VFMV_V_F (-0.16666666666494565f, vl), t, G, vl);
  vm32_f_t rest = VM32_VFMUL_VV (VM32_VFMUL_VV (xs, t, vl), G, vl);
  vm32_f_t small = VM32_VFADD_VV (xs, rest, vl);
  /* |x| > 0.1: log(ax + sqrt(ax^2+1)), argument as a double-float.  */
  vm32_f_t p = VM32_VFMUL_VV (ax, ax, vl);
  vm32_f_t pe = VM32_VFMSAC_VV (p, ax, ax, vl);
  vm32_f_t sh, se;
  VM32_LOGF_TWO_SUM (sh, se, VM32_VFMV_V_F (1.0f, (vl)), p, vl);
  se = VM32_VFADD_VV (se, pe, vl);
  vm32_f_t rh, rl;
  VM32_HINV_DD_SQRT (rh, rl, sh, se, vl);
  vm32_f_t ah, ae;
  VM32_LOGF_TWO_SUM (ah, ae, ax, rh, vl);
  ae = VM32_VFADD_VV (ae, rl, vl);
  vm32_f_t h, l;
  VM32_HINV_LOG_SUM_DD (h, l, ah, ae, vl);
  vm32_f_t big = VM32_VFADD_VV (h, l, vl);
  vm32_b_t msmall = VM32_VMFLE_VF (ax, 0.1f, vl);
  vm32_f_t r = VM32_VFMERGE_VVM (big, small, msmall, vl);
  r = VM32_VFSGNJ_VV (r, xs, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (asinh_fixup) (x, r, mbad, vl, VM32_KFN (asinh_fix));
  return r;
}

/* ---- acosh --------------------------------------------------------- */

/* x < 1+2^-23, x >= 20.5, Inf/NaN to the epilogue.  */
#define VM32_HINV_BAD_ACOSH(x, vl)					\
  VM32_VMOR_MM (							\
    VM32_VMSLTU_VX (VM32_BITS (x), 0x3f800001u, (vl)),			\
    VM32_VMSGTU_VX (VM32_BITS (x), 0x41a40000u, (vl)), (vl))

static inline vm32_f_t
VM32_KFN (acosh) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32_HINV_BAD_ACOSH (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (2.0f, (vl)), mbad, vl);
  /* log(x + sqrt(x^2-1)).  x^2-1 is exact through the two-sum of x^2 and
     -1 plus the product tail; the sqrt is refreshed by its Newton step.  */
  vm32_f_t p = VM32_VFMUL_VV (xs, xs, vl);
  vm32_f_t pe = VM32_VFMSAC_VV (p, xs, xs, vl);
  vm32_f_t dh, de;
  VM32_LOGF_TWO_SUM (dh, de, p, VM32_VFMV_V_F (-1.0f, (vl)), vl);
  de = VM32_VFADD_VV (de, pe, vl);
  vm32_f_t rh, rl;
  VM32_HINV_DD_SQRT (rh, rl, dh, de, vl);
  vm32_f_t ah, ae;
  VM32_LOGF_TWO_SUM (ah, ae, xs, rh, vl);
  ae = VM32_VFADD_VV (ae, rl, vl);
  vm32_f_t h, l;
  VM32_HINV_LOG_SUM_DD (h, l, ah, ae, vl);
  vm32_f_t r = VM32_VFADD_VV (h, l, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (acosh_fixup) (x, r, mbad, vl, VM32_KFN (acosh_fix));
  return r;
}

/* ---- atanh --------------------------------------------------------- */

/* 2^-12 <= |x| <= 0.9; everything else to the epilogue.  Below 2^-12
   atanh(x) rounds to x (the x^3/3 term is under half an ulp), which the
   scalar epilogue returns exactly; keeping those lanes out of the fast
   path also keeps its log argument off the rounding-tie boundary.  */
#define VM32_HINV_BAD_ATANH(x, vl)					\
  VM32_VMOR_MM (							\
    VM32_VMSGTU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x3f666666u, (vl)),					\
    VM32_VMSLTU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x39800000u, (vl)), (vl))

static inline vm32_f_t
VM32_KFN (atanh) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32_HINV_BAD_ATANH (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (0.0f, (vl)), mbad, vl);
  vm32_f_t ax = VM32_VFABS_V (xs, vl);
  /* atanh|x| = 0.5*log((1+|x|)/(1-|x|)); both sums are exact pairs.  */
  vm32_f_t nh, nl, dh, dl;
  VM32_LOGF_TWO_SUM (nh, nl, VM32_VFMV_V_F (1.0f, (vl)), ax, vl);
  VM32_LOGF_TWO_SUM (dh, dl, VM32_VFMV_V_F (1.0f, (vl)),
		     VM32_VFNEG_V (ax, vl), vl);
  vm32_f_t qh = VM32_VFDIV_VV (nh, dh, vl);
  vm32_f_t e = VM32_VFNMSAC_VV (nh, qh, dh, vl);
  e = VM32_VFADD_VV (e, nl, vl);
  e = VM32_VFNMSAC_VV (e, qh, dl, vl);
  vm32_f_t ql = VM32_VFDIV_VV (e, dh, vl);
  vm32_f_t ph, pl;
  VM32_HINV_LOG_SUM_DD (ph, pl, qh, ql, vl);
  vm32_f_t r = VM32_VFADD_VV (VM32_VFMUL_VF (ph, 0.5f, vl),
			      VM32_VFMUL_VF (pl, 0.5f, vl), vl);
  r = VM32_VFSGNJ_VV (r, xs, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (atanh_fixup) (x, r, mbad, vl, VM32_KFN (atanh_fix));
  return r;
}
