/* f32/k_hyper.inc -- binary32 hyperbolic kernels (sinh / cosh / tanh).

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h, f32/wrap.h and f32/k_exp.inc (for the double-float exp
   core), with the binary64 kernels (k_hyper.inc and what it pulls in)
   active at VM_LMUL == 2*VM32_LMUL.

   Native binary32 fast path over 2^-20 <= |x| <= 20 (the benchmark
   domain is [-10, 10]); every other lane goes through the per-lane scalar
   binary64 epilogue, which reproduces the value and gated flag behaviour
   of the previous widen / binary64 kernel / narrow path.  Out-of-line
   per-lane epilogue keeps the vector kernel inlineable and its hot path
   free of the callee-saved register spills a scalar libm call causes.

   Accuracy: the small-|x| branch is a minimax odd polynomial; the large
   branch forms expm1(|x|) as a double-float, then does the rational
   finishing step in double-float, and every kernel rounds exactly once at
   the end.  */

/* ------------------------------------------------------------------ */
/* f32 double-float helpers (file-local names).                       */
/* ------------------------------------------------------------------ */

#undef VM32_HYP_REC7
#undef VM32_HYP_QDIV_DD
#undef VM32_HYP_TWO_SUM
#undef VM32_HYP_FAST2_SUM
#undef VM32_HYP_DD_DIV
#undef VM32_HYP_APPLY_SIGN
#undef VM32_HYP_DEF_FIXUP

/* (s, e) = a + b exactly (Knuth two-sum).  */
#define VM32_HYP_TWO_SUM(s, e, a, b, vl) do {				\
  vm32_f_t hts_s_ = VM32_VFADD_VV ((a), (b), (vl));			\
  vm32_f_t hts_v_ = VM32_VFSUB_VV (hts_s_, (a), (vl));			\
  (s) = hts_s_;								\
  (e) = VM32_VFADD_VV (							\
    VM32_VFSUB_VV ((a), VM32_VFSUB_VV (hts_s_, hts_v_, (vl)), (vl)),	\
    VM32_VFSUB_VV ((b), hts_v_, (vl)), (vl));				\
} while (0)

/* vfrec7.v -- 7-bit reciprocal estimate (f32/compat.h wraps no estimate
   instruction).  */
#define VM32_HYP_REC7(a, vl) VM32_XP2 (__riscv_vfrec7_v_f32m, VM32_LMUL) (a, vl)

/* q = (nh+nl)/(dh+dl) for a positive normal dh, with the reciprocal
   from the 7-bit estimate plus three Newton steps and an fma residual
   (the VM_QDIV shape of <vecmath_dd.h>).  The three steps are needed:
   with two, the estimate's last-bit error leaves ~1.0 ulp at the small
   end of tanh's range.  One rounding of the exact quotient, no vfdiv.  */
#define VM32_HYP_QDIV_DD(q, nh, nl, dh, dl, vl) do {			\
  vm32_f_t hqd__rt = VM32_HYP_REC7 ((dh), (vl));			\
  vm32_f_t hqd__e = VM32_VFNMSAC_VV (					\
    VM32_VFMV_V_F (1.0f, (vl)), (dh), hqd__rt, (vl));			\
  hqd__rt = VM32_VFMACC_VV (hqd__rt, hqd__rt, hqd__e, (vl));		\
  hqd__e = VM32_VFNMSAC_VV (VM32_VFMV_V_F (1.0f, (vl)), (dh),		\
			    hqd__rt, (vl));				\
  hqd__rt = VM32_VFMACC_VV (hqd__rt, hqd__rt, hqd__e, (vl));		\
  (q) = VM32_VFMUL_VV ((nh), hqd__rt, (vl));				\
  hqd__e = VM32_VFNMSAC_VV ((nh), (q), (dh), (vl));			\
  hqd__e = VM32_VFADD_VV (hqd__e, (nl), (vl));				\
  hqd__e = VM32_VFNMSAC_VV (hqd__e, (q), (dl), (vl));			\
  (q) = VM32_VFMACC_VV ((q), hqd__e, hqd__rt, (vl));			\
} while (0)

/* (s, e) = a + b exactly, requires |a| >= |b|.  */
#define VM32_HYP_FAST2_SUM(s, e, a, b, vl) do {				\
  (s) = VM32_VFADD_VV ((a), (b), (vl));					\
  (e) = VM32_VFSUB_VV ((b), VM32_VFSUB_VV ((s), (a), (vl)), (vl));	\
} while (0)

/* (qh, ql) = (nh + nl) / (dh + dl); the divisor is a positive normal.  */
#define VM32_HYP_DD_DIV(qh, ql, nh, nl, dh, dl, vl) do {		\
  (qh) = VM32_VFDIV_VV ((nh), (dh), (vl));				\
  vm32_f_t hdv_e_ = VM32_VFNMSAC_VV ((nh), (qh), (dh), (vl));		\
  hdv_e_ = VM32_VFADD_VV (hdv_e_, (nl), (vl));				\
  hdv_e_ = VM32_VFNMSAC_VV (hdv_e_, (qh), (dl), (vl));			\
  (ql) = VM32_VFDIV_VV (hdv_e_, (dh), (vl));				\
} while (0)

/* |r| with the sign of x (vfabs keeps NaN payloads quiet; callers have
   already sent NaN lanes to the epilogue).  */
#define VM32_HYP_APPLY_SIGN(r, x, vl)					\
  VM32_VFSGNJ_VV ((r), (x), (vl))

/* ---- per-lane scalar epilogue -------------------------------------- */

#undef VM32_HYP_DEF_FIXUP
#define VM32_HYP_DEF_FIXUP(name)					\
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

VM32_HYP_DEF_FIXUP (sinh_fixup)
VM32_HYP_DEF_FIXUP (cosh_fixup)
VM32_HYP_DEF_FIXUP (tanh_fixup)

static inline float
VM32_KFN (sinh_fix) (float x)
{
  return (float) sinh ((double) x);
}

static inline float
VM32_KFN (cosh_fix) (float x)
{
  return (float) cosh ((double) x);
}

static inline float
VM32_KFN (tanh_fix) (float x)
{
  return (float) tanh ((double) x);
}

/* Covered domain for all three: 2^-20 <= |x| <= 20.  */
#define VM32_HYP_BAD(x, vl)						\
  VM32_VMOR_MM (							\
    VM32_VMSGEU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x41a00000u, (vl)),					\
    VM32_VMSLTU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x35800000u, (vl)), (vl))

/* Poly helpers: T(t) = c0 + t*(c1 + t*(c2 + ...)) with the highest
   coefficient seeded by a multiply-add pair.  */

/* ---- sinh ---------------------------------------------------------- */

static inline vm32_f_t
VM32_KFN (sinh) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32_HYP_BAD (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (1.0f, (vl)), mbad, vl);
  vm32_f_t ax = VM32_VFABS_V (xs, vl);
  /* 2^-20 <= |x| <= 1: x + x*t*S(t), t = x^2, S a degree-4 minimax fit
     of (sinh(x)-x)/x^3 over t in [0,1].  */
  vm32_f_t sth = VM32_VFMUL_VV (ax, ax, vl);
  vm32_f_t ste = VM32_VFMSAC_VV (sth, ax, ax, vl);
  /* rest = x*t*S(t): the x*t product and the leading term are carried as
     exact pairs so only the final add rounds.  */
  vm32_f_t S1 = VM32_VFMUL_VF (sth, 2.5456706571523821e-08f, vl);
  S1 = VM32_VFADD_VF (S1, 2.755377026364149e-06f, vl);
  S1 = VM32_VFMACC_VV (VM32_VFMV_V_F (0.0001984128253011599f, vl), sth,
		       S1, vl);
  S1 = VM32_VFMACC_VV (VM32_VFMV_V_F (0.0083333333174643621f, vl), sth,
		       S1, vl);
  vm32_f_t Sh, Se;
  VM32_HYP_FAST2_SUM (Sh, Se, VM32_VFMV_V_F (0.16666666666698407f, (vl)),
		      VM32_VFMUL_VV (sth, S1, vl), vl);
  vm32_f_t wh = VM32_VFMUL_VV (ax, sth, vl);
  vm32_f_t we = VM32_VFMSAC_VV (wh, ax, sth, vl);
  vm32_f_t srh = VM32_VFMUL_VV (wh, Sh, vl);
  vm32_f_t sre = VM32_VFMSAC_VV (srh, wh, Sh, vl);
  sre = VM32_VFMACC_VV (sre, wh, Se, vl);
  sre = VM32_VFMACC_VV (sre, VM32_VFMUL_VV (ste, ax, vl), Sh, vl);
  (void) we;
  vm32_f_t small = VM32_VFADD_VV (ax, VM32_VFADD_VV (srh, sre, vl), vl);
  /* |x| > 1: 0.5*(t + t/(t+1)) with t = expm1(|x|) as a double-float.  */
  vm32_f_t eh, el, th, tl;
  VM32_EXPF_EXP_DD (eh, el, ax, vl);
  VM32_EXPF_MINUS1_DD (th, tl, eh, el, vl);
  vm32_f_t uh, ue;
  VM32_HYP_TWO_SUM (uh, ue, VM32_VFMV_V_F (1.0f, (vl)), th, vl);
  ue = VM32_VFADD_VV (ue, tl, vl);
  vm32_f_t qh, ql;
  VM32_HYP_DD_DIV (qh, ql, th, tl, uh, ue, vl);
  vm32_f_t rh, re;
  VM32_HYP_TWO_SUM (rh, re, th, qh, vl);
  re = VM32_VFADD_VV (re, tl, vl);
  re = VM32_VFADD_VV (re, ql, vl);
  vm32_f_t big = VM32_VFADD_VV (VM32_VFMUL_VF (rh, 0.5f, vl),
				VM32_VFMUL_VF (re, 0.5f, vl), vl);
  vm32_b_t msmall = VM32_VMFLE_VF (ax, 1.0f, vl);
  vm32_f_t r = VM32_VFMERGE_VVM (big, small, msmall, vl);
  r = VM32_HYP_APPLY_SIGN (r, xs, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (sinh_fixup) (x, r, mbad, vl, VM32_KFN (sinh_fix));
  return r;
}

/* ---- cosh ---------------------------------------------------------- */

static inline vm32_f_t
VM32_KFN (cosh) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_HYP_BAD (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (1.0f, (vl)), mbad, vl);
  vm32_f_t th = VM32_VFMUL_VV (xs, xs, vl);
  vm32_f_t te = VM32_VFMSAC_VV (th, xs, xs, vl);
  /* 2^-20 <= |x| <= 1: 1 + t*C(t), C deg-4 fit of (cosh(x)-1)/x^2.
     The leading product and the constant are kept as exact pairs so the
     two last roundings do not eat into the one-ulp budget.  */
  vm32_f_t C1 = VM32_VFMUL_VF (th, 2.8083924380881532e-07f, vl);
  C1 = VM32_VFADD_VF (C1, 2.4796966404404219e-05f, vl);
  C1 = VM32_VFMACC_VV (VM32_VFMV_V_F (0.0013888905413098637f, vl), th,
		       C1, vl);
  C1 = VM32_VFMACC_VV (VM32_VFMV_V_F (0.041666666459995207f, vl), th, C1,
		       vl);
  vm32_f_t Ch, Ce;
  VM32_HYP_FAST2_SUM (Ch, Ce, VM32_VFMV_V_F (0.5f, (vl)),
		      VM32_VFMUL_VV (th, C1, vl), vl);
  vm32_f_t rh = VM32_VFMUL_VV (th, Ch, vl);
  vm32_f_t re = VM32_VFMSAC_VV (rh, th, Ch, vl);
  re = VM32_VFMACC_VV (re, th, Ce, vl);
  re = VM32_VFMACC_VV (re, te, Ch, vl);
  vm32_f_t sh, se;
  VM32_HYP_FAST2_SUM (sh, se, VM32_VFMV_V_F (1.0f, (vl)), rh, vl);
  vm32_f_t small = VM32_VFADD_VV (sh, VM32_VFADD_VV (se, re, vl), vl);
  /* |x| > 1: 0.5*(e + 1/e) with e = exp(|x|) as a double-float.  */
  vm32_f_t eh, el;
  VM32_EXPF_EXP_DD (eh, el, xs, vl);
  vm32_f_t ih = VM32_VFRDIV_VF (eh, 1.0f, vl);
  vm32_f_t ie = VM32_VFNMSAC_VV (VM32_VFMV_V_F (1.0f, (vl)), ih, eh, vl);
  ie = VM32_VFSUB_VV (ie, VM32_VFMUL_VV (ih, el, vl), vl);
  ie = VM32_VFDIV_VV (ie, eh, vl);
  vm32_f_t ah, ae;
  VM32_HYP_TWO_SUM (ah, ae, eh, ih, vl);
  ae = VM32_VFADD_VV (ae, el, vl);
  ae = VM32_VFADD_VV (ae, ie, vl);
  vm32_f_t big = VM32_VFADD_VV (VM32_VFMUL_VF (ah, 0.5f, vl),
				VM32_VFMUL_VF (ae, 0.5f, vl), vl);
  vm32_b_t msmall = VM32_VMFLE_VF (VM32_VFABS_V (xs, vl), 1.0f, vl);
  vm32_f_t r = VM32_VFMERGE_VVM (big, small, msmall, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (cosh_fixup) (x, r, mbad, vl, VM32_KFN (cosh_fix));
  return r;
}

/* ---- tanh ---------------------------------------------------------- */

static inline vm32_f_t
VM32_KFN (tanh) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32_HYP_BAD (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (1.0f, (vl)), mbad, vl);
  vm32_f_t ax = VM32_VFABS_V (xs, vl);
  /* One formula for the whole covered range (2^-20 <= |x| <= 20):
     tanh(x) = t/(t+2) with t = expm1(2|x|) as a double-float pair.
     The previous shape evaluated a degree-7 minimax poly for |x| <= 0.55
     *and* the exp/divide branch for every lane, then merged; the two
     branches are lane-wise exclusive, so with SIMD both were paid on
     every lane.  The double-float exp keeps t = e-1 accurate to 2^-45
     absolute even for tiny |x| (the 1-subtraction is a two-sum, and the
     low word carries what the subtraction exposes), so the same quotient
     covers the small range and the old branch/merge disappears.  */
  vm32_f_t eh, el, th, tl;
  VM32_EXPF_EXP_DD (eh, el, VM32_VFMUL_VF (ax, 2.0f, vl), vl);
  VM32_EXPF_MINUS1_DD (th, tl, eh, el, vl);
  vm32_f_t uh, ue;
  VM32_HYP_TWO_SUM (uh, ue, VM32_VFMV_V_F (2.0f, (vl)), th, vl);
  ue = VM32_VFADD_VV (ue, tl, vl);
  vm32_f_t r;
  VM32_HYP_QDIV_DD (r, th, tl, uh, ue, vl);
  r = VM32_HYP_APPLY_SIGN (r, xs, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (tanh_fixup) (x, r, mbad, vl, VM32_KFN (tanh_fix));
  return r;
}
