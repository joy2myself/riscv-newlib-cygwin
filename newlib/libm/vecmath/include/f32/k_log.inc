/* f32/k_log.inc -- binary32 log / log2 / log10 / log1p / log2p1 kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_log.inc)
   active at VM_LMUL == 2*VM32_LMUL.

   Shape: a native binary32 fast path over a covered domain (all normal
   positive x for log/log2/log10; x > -0.94, |x| >= 2^-30, x < 2^60 for
   log1p/log2p1), plus a *per-lane* scalar epilogue for the remaining
   lanes.  The epilogue evaluates the lane in binary64 through the scalar
   libm and narrows, which is what the previous widen / binary64 kernel /
   narrow path did for those lanes, so values and gated FP flags are
   inherited from the known-good path.  Per-lane (not per-strip) fixup is
   what keeps the result of a lane independent of the strip it happens to
   sit in: the framework's length and alignment sweeps require exactly
   that.

   Fast path: one shared 128-entry mantissa-slice table (k_log_tab.inc)
   in log2 units.  For u = 2^k * m with m in [1,2) the top 7 mantissa bits
   select the slice, whose entry {r, l1, l2} satisfies

     z = fma (r, m, -1)     EXACT in binary32 (r has 7 significant bits,
                            so r*m is a multiple of 2^-30, and |z| < 2^-6.8
                            leaves at most 22 significant bits -- verified
                            per slice against both ends of the slice)
     log2(1/r) = l1 + 127 + l2,   l1 = round_{2^-16}(log2(1/r)) - 127

   l1 is a multiple of 2^-16, so float (biased exponent) + l1 is exact for
   every binary32 exponent; l2 is packed as a 17-bit signed integer on the
   2^-33 grid into the 17 free low mantissa bits of r, so a single
   gathered 32-bit word yields r and l2.  Two gathers at 8-byte stride,
   both inside one cache line, replace the three gathers (and the
   double-float chain with its 2-3 divisions) of the previous core.

   log2(1+z) = z*C1 + z^2*Q(z), C1 = log2(e) as an f32 pair, Q the
   z-series of -C1/2 + C1 z/3 - C1 z^2/4 (|z| < 2^-6.8 makes the
   truncation 2^-36, far below the final rounding).  The low part is
   accumulated in 2^33-scaled form so that the packed l2 field and the
   un-scaling both come for free; the result is one final rounding of
   s + (se + lo), where (s, se) = hi + w exactly.

   log and log10 scale the log2-units hi by ln2 / log10(2) as an exact
   pair (ph, pl) and use their own polynomial, so the table is shared by
   all five entries.  Slices 0, 126 and 127 use r = 1 and r = 1/2
   exactly: l1 is then an exact integer and l2 is exactly 0, so a result
   next to 1 -- from either side, where the answer is as small as the
   input is close to 1 -- carries the whole answer in the polynomial and
   never cancels against a coarse table value.  */

/* ---- the shared slice table ---------------------------------------- */

/* The table type and contents are LMUL-independent, so they are emitted
   once per translation unit even though the kernels below are
   instantiated once per VM32_LMUL.  */
#ifndef VM32_LOGF_TAB_INCLUDED
#define VM32_LOGF_TAB_INCLUDED 1
#include <f32/k_log_tab.inc>
#endif

/* ------------------------------------------------------------------ */
/* Constants.                                                         */
/* ------------------------------------------------------------------ */

/* log2 units.  C1 = log2(e) split as a full-precision f32 plus residual;
   Q0..Q2 are the z^2-series coefficients of log2(1+z) - z*C1.  The *_33
   forms are the same values pre-scaled by 2^33 so that the low part can
   be accumulated in the scale of the packed l2 field.  */
#define VM32_LOGF_L2E32_HI	0x1.715476p+0f		/* log2(e) */
#define VM32_LOGF_L2E32_LO	0x1.4ae0cp-26f		/* log2(e) - HI */
#define VM32_LOGF_L2E32_HI_33	0x1.715476p+33f
#define VM32_LOGF_L2E32_LO_33	0x1.4ae0cp+7f
#define VM32_LOGF_L2Q0		-0x1.715476p-1f		/* -C1/2 */
#define VM32_LOGF_L2Q1		0x1.ec709ep-2f		/*  C1/3 */
#define VM32_LOGF_L2Q2		-0x1.715476p-2f		/* -C1/4 */
#define VM32_LOGF_L2Q0_33	-0x1.715476p+32f
#define VM32_LOGF_L2Q1_33	0x1.ec709ep+31f
#define VM32_LOGF_L2Q2_33	-0x1.715476p+31f

/* Natural log: log2(2^k) scaled by ln2 as the exact pair (HI, LO); the
   series of ln(1+z) has C1 = 1 exactly, so its leading term is z itself
   and there is no residual to carry.  */
#define VM32_LOGF_LN2_HI	0x1.62e4p-1f		/* ln 2, 15 bits */
#define VM32_LOGF_LN2_LO	0x1.7f7d1cp-20f		/* ln 2 - HI */
#define VM32_LOGF_LN2_LO_33	0x1.7f7d1cp+13f
#define VM32_LOGF_LN2Q0		-0x1p-1f		/* -1/2 */
#define VM32_LOGF_LN2Q1		0x1.555556p-2f		/*  1/3 */
#define VM32_LOGF_LN2Q2		-0x1p-2f		/* -1/4 */
#define VM32_LOGF_LN2Q0_33	-0x1p+32f
#define VM32_LOGF_LN2Q1_33	0x1.555556p+31f
#define VM32_LOGF_LN2Q2_33	-0x1p+31f

/* Common log: the same shape with log10(2) and log10(e).  */
#define VM32_LOGF_L10_2_HI	0x1.344p-2f		/* log10(2), 15 bits */
#define VM32_LOGF_L10_2_LO	0x1.3509f8p-18f		/* log10(2) - HI */
#define VM32_LOGF_L10_2_LO_33	0x1.3509f8p+15f
#define VM32_LOGF_L10E_HI	0x1.bcb7b2p-2f		/* log10(e) */
#define VM32_LOGF_L10E_LO	-0x1.5b235ep-27f	/* log10(e) - HI */
#define VM32_LOGF_L10E_LO_33	-0x1.5b235ep+6f
#define VM32_LOGF_L10Q0		-0x1.bcb7b2p-3f		/* -C/2 */
#define VM32_LOGF_L10Q1		0x1.287a76p-3f		/*  C/3 */
#define VM32_LOGF_L10Q2		-0x1.bcb7b2p-4f		/* -C/4 */
#define VM32_LOGF_L10Q0_33	-0x1.bcb7b2p+30f
#define VM32_LOGF_L10Q1_33	0x1.287a76p+30f
#define VM32_LOGF_L10Q2_33	-0x1.bcb7b2p+29f

/* 2^33 and 2^-33: the scale of the packed l2 field.  */
#define VM32_LOGF_S33		0x1p+33f
#define VM32_LOGF_IS33		0x1p-33f

/* ------------------------------------------------------------------ */
/* f32 double-float building blocks (file-local names).               */
/* ------------------------------------------------------------------ */

#undef VM32_LOGF_TWO_SUM
#undef VM32_LOGF_TAB_LD
#undef VM32_LOGF_TAB_RED
#undef VM32_LOGF_DEF_FIXUP
#undef VM32_LOGF_RAISE_UF_SUB
#undef VM32_LOGF_BADNORMAL
#undef VM32_LOGF_BAD1P
#undef VM32_LOGF_ONE

/* (s, e) = a + b exactly (Knuth two-sum; general operand order).  */
#define VM32_LOGF_TWO_SUM(s, e, a, b, vl) do {				\
  vm32_f_t lts_s_ = VM32_VFADD_VV ((a), (b), (vl));			\
  vm32_f_t lts_v_ = VM32_VFSUB_VV (lts_s_, (a), (vl));			\
  (s) = lts_s_;								\
  (e) = VM32_VFADD_VV (							\
    VM32_VFSUB_VV ((a), VM32_VFSUB_VV (lts_s_, lts_v_, (vl)), (vl)),	\
    VM32_VFSUB_VV ((b), lts_v_, (vl)), (vl));				\
} while (0)

/* Load the slice entry selected by off (8 * the top 7 mantissa bits):
   r = the f32 mantissa, l2f = l2 * 2^33 as a float, l1 = the f32 hi part.
   The packed word's low 17 bits carry l2 in two's complement, so the
   sign-extending shift pair turns them into l2 * 2^33 directly.  */
#define VM32_LOGF_TAB_LD(r, l2f, l1, off, vl) do {			\
  vm32_u_t ltw_w_ = VM32_VLUXEI32_U (&__vm32_log2_tab[0].rl2,		\
				     (off), (vl));			\
  (r) = VM32_FROM_BITS (VM32_VAND_VX (ltw_w_, 0xfffe0000u, (vl)));	\
  (l2f) = VM32_VFCVT_F_X_V (VM32_VSRA_VX (				\
    VM32_REINT_U_I (VM32_VSLL_VX (ltw_w_, 15, (vl))), 15, (vl)), (vl));	\
  (l1) = VM32_VLUXEI32 (&__vm32_log2_tab[0].l1, (off), (vl));		\
} while (0)

/* b = bits(u), off = table byte offset, kf = (float) biased exponent,
   m = mantissa of u in [1,2) (exact, no multiply).  */
#define VM32_LOGF_TAB_RED(b, off, kf, m, u, vl) do {			\
  (b) = VM32_BITS (u);							\
  (off) = VM32_VAND_VX (VM32_VSRL_VX ((b), 13, (vl)), 0x3f8u, (vl));	\
  (kf) = VM32_VFCVT_F_X_V (						\
    VM32_REINT_U_I (VM32_VSRL_VX ((b), 23, (vl))), (vl));		\
  (m) = VM32_FROM_BITS (VM32_VOR_VX (					\
    VM32_VAND_VX ((b), 0x007fffffu, (vl)), 0x3f800000u, (vl)));		\
} while (0)

/* ---- binary32 double-float log pair (used by f32/k_hyperinv.inc) ---- */

/* (h, l) = log(x) as a double-float pair for normal positive x.  The
   hyperinv kernels (asinh / acosh / atanh) evaluate the log of a
   double-float argument and need the pair, not a rounded result, so this
   entry point keeps the division-based double-float chain: one exact
   residual division per term instead of the slice table.  */

#undef VM32_LOGF_LOG1P_DD
#undef VM32_LOGF_REDUCE
#undef VM32_LOGF_LOG_DD

/* (ph, pl) = log1p(f) as a double-float pair, f in [-0.4143, 0.4143].
   log1p(f) = 2*atanh(f/(2+f)); the atanh series needs only a few terms
   because |u| <= 0.1716.  u is formed exactly enough (two-sum of 2+f plus
   fma remainder) for a full-precision result; the pair is rounded once by
   the caller.  */
#define VM32_LOGF_LOG1P_DD(ph, pl, f, vl) do {				\
  vm32_f_t lgd_dh_ = VM32_VFADD_VF ((f), 2.0f, (vl));			\
  vm32_f_t lgd_dl_ = VM32_VFSUB_VV (					\
    (f), VM32_VFSUB_VF (lgd_dh_, 2.0f, (vl)), (vl));			\
  vm32_f_t lgd_uh_ = VM32_VFDIV_VV ((f), lgd_dh_, (vl));		\
  vm32_f_t lgd_e_ = VM32_VFNMSAC_VV ((f), lgd_uh_, lgd_dh_, (vl));	\
  lgd_e_ = VM32_VFNMSAC_VV (lgd_e_, lgd_uh_, lgd_dl_, (vl));		\
  vm32_f_t lgd_ul_ = VM32_VFDIV_VV (lgd_e_, lgd_dh_, (vl));		\
  vm32_f_t lgd_u2_ = VM32_VFMUL_VV (lgd_uh_, lgd_uh_, (vl));		\
  /* P = 1/3 + u2*(1/5 + u2*(1/7 + u2*(1/9))) */			\
  vm32_f_t lgd_P_ = VM32_VFMUL_VF (lgd_u2_, 1.0f / 9.0f, (vl));		\
  lgd_P_ = VM32_VFADD_VF (lgd_P_, 1.0f / 7.0f, (vl));			\
  lgd_P_ = VM32_VFMACC_VV (VM32_VFMV_V_F (1.0f / 5.0f, (vl)),		\
			   lgd_u2_, lgd_P_, (vl));			\
  lgd_P_ = VM32_VFMACC_VV (VM32_VFMV_V_F (1.0f / 3.0f, (vl)),		\
			   lgd_u2_, lgd_P_, (vl));			\
  /* rest = u^3 * P; log1p(f) = 2*(u + rest).  */			\
  vm32_f_t lgd_rest_ = VM32_VFMUL_VV (					\
    VM32_VFMUL_VV (lgd_uh_, lgd_u2_, (vl)), lgd_P_, (vl));		\
  vm32_f_t lgd_s_ = VM32_VFADD_VV (lgd_uh_, lgd_rest_, (vl));		\
  vm32_f_t lgd_se_ = VM32_VFSUB_VV (					\
    lgd_rest_, VM32_VFSUB_VV (lgd_s_, lgd_uh_, (vl)), (vl));		\
  (ph) = VM32_VFMUL_VF (lgd_s_, 2.0f, (vl));				\
  (pl) = VM32_VFMUL_VF (VM32_VFADD_VV (lgd_se_, lgd_ul_, (vl)),		\
			2.0f, (vl));					\
} while (0)

/* Reduce normal positive x to x = 2^k * m with m in [sqrt(1/2), sqrt(2)),
   then (ph, pl) = log1p(m-1) and kf = (float) k.  m-1 is exact.  */
#define VM32_LOGF_REDUCE(ph, pl, kf, x, vl) do {			\
  vm32_u_t lrd_ix_ = VM32_BITS (x);					\
  vm32_u_t lrd_k_ = VM32_VSUB_VX (VM32_VSRL_VX (lrd_ix_, 23, (vl)),	\
				  127, (vl));				\
  vm32_f_t lrd_m_ = VM32_FROM_BITS (					\
    VM32_VOR_VX (VM32_VAND_VX (lrd_ix_, 0x7fffff, (vl)),		\
		 0x3f800000, (vl)));					\
  vm32_b_t lrd_hi_ = VM32_VMFGT_VF (lrd_m_, 1.4142135f, (vl));		\
  lrd_m_ = VM32_VFMERGE_VVM (lrd_m_,					\
			     VM32_VFMUL_VF (lrd_m_, 0.5f, (vl)),	\
			     lrd_hi_, (vl));				\
  lrd_k_ = VM32_VMERGE_VVM (lrd_k_, VM32_VADD_VX (lrd_k_, 1, (vl)),	\
			    lrd_hi_, (vl));				\
  (kf) = VM32_VFCVT_F_X_V (VM32_REINT_U_I (lrd_k_), (vl));		\
  VM32_LOGF_LOG1P_DD (ph, pl, VM32_VFSUB_VF (lrd_m_, 1.0f, (vl)),	\
		      (vl));						\
} while (0)

/* The double-float chain below uses the same ln2 pair as the table core
   (VM32_LOGF_LN2_HI / _LO), defined with the other constants above.  */

/* (h, l) = log(x) for normal positive x.  */
#define VM32_LOGF_LOG_DD(h, l, x, vl) do {				\
  vm32_f_t ldd_ph_, ldd_pl_, ldd_kf_;					\
  VM32_LOGF_REDUCE (ldd_ph_, ldd_pl_, ldd_kf_, (x), (vl));		\
  vm32_f_t ldd_base_ = VM32_VFMUL_VF (ldd_kf_, VM32_LOGF_LN2_HI, (vl)); \
  vm32_f_t ldd_s_, ldd_se_;						\
  VM32_LOGF_TWO_SUM (ldd_s_, ldd_se_, ldd_base_, ldd_ph_, (vl));	\
  vm32_f_t ldd_t_ = VM32_VFADD_VV (ldd_se_, ldd_pl_, (vl));		\
  ldd_t_ = VM32_VFMACC_VF (ldd_t_, VM32_LOGF_LN2_LO, ldd_kf_, (vl));	\
  (h) = ldd_s_;								\
  (l) = ldd_t_;								\
} while (0)

/* ---- per-lane scalar epilogue -------------------------------------- */

/* Per-lane scalar epilogue as an out-of-line helper: the vector kernel
   itself must stay small and inlineable so the hot path never pays the
   callee-saved-vector-register spills a scalar libm call forces.  fn
   evaluates a lane in binary64 and narrows -- the value and gated flags
   the previous widen / binary64 kernel / narrow path produced for it.
   Buffer bound: vl <= VLMAX(e32, LMUL <= 4) = VLEN/8 lanes, 600 covers
   VLEN up to 4800.  */
#define VM32_LOGF_DEF_FIXUP(name)					\
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

VM32_LOGF_DEF_FIXUP (log_fixup)
VM32_LOGF_DEF_FIXUP (log2_fixup)
VM32_LOGF_DEF_FIXUP (log10_fixup)
VM32_LOGF_DEF_FIXUP (log1p_fixup)
VM32_LOGF_DEF_FIXUP (log2p1_fixup)

/* Raise FE_UNDERFLOW (with FE_INEXACT) on the lanes whose binary32 result
   is a nonzero subnormal -- the explicit underflow carrier the previous
   wide path used for expm1/log1p/pow: a binary64 evaluation can land
   exactly on the binary32 subnormal grid, where narrowing stays silent,
   while the binary32 operation itself would have been tiny and inexact.
   The masked 2^-126 * 2^-126 multiply flushes to zero with UNDERFLOW and
   the merge leaves inactive lanes untouched.  */
#define VM32_LOGF_RAISE_UF_SUB(yo, vl) do {				\
  vm32_u_t vuf_b_ = VM32_VAND_VX (VM32_BITS (yo), 0x7fffffffu, (vl));	\
  vm32_b_t vuf_m_ = VM32_VMSLTU_VX (					\
    VM32_VSUB_VX (vuf_b_, 1, (vl)), 0x007fffffu, (vl));			\
  vm32_f_t vuf_z_ = VM32_VFMUL_VF_MU (vuf_m_,				\
				      VM32_VFMV_V_F (0.0f, (vl)),	\
				      VM32_VFMV_V_F (0x1p-126f, (vl)),	\
				      0x1p-126f, (vl));			\
  (yo) = VM32_VFADD_VV_MU (vuf_m_, (yo), (yo), vuf_z_, (vl));		\
} while (0)

/* ---- exceptional-lane masks --------------------------------------- */

/* Not a positive normal binary32: zero, subnormal, negative, Inf, NaN.
   One unsigned window test on the raw bit pattern: b - 2^-126's pattern
   leaves [0, 0x7effffff] exactly for the positive normals, and every
   other class (b < 2^-126, b > FLT_MAX, or b >= 0x80000000 with the sign
   bit set) wraps or steps outside it.  */
#define VM32_LOGF_BADNORMAL(x, vl)					\
  VM32_VMSGEU_VX (							\
    VM32_VSUB_VX (VM32_BITS (x), 0x00800000u, (vl)), 0x7f000000u, (vl))

/* Exceptional lanes for log1p/log2p1: x < -0.94 (bits above -0.94 as x
   grows more negative; covers -Inf and negative NaN), |x| < 2^-30 or
   |x| >= 2^60 (the unsigned window test on |x|).  */
#define VM32_LOGF_BAD1P(x, vl)						\
  VM32_VMOR_MM (							\
    VM32_VMSGEU_VX (							\
      VM32_VSUB_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x30800000u, (vl)),					\
      0x2d000000u, (vl)),						\
    VM32_VMSGTU_VX (VM32_BITS (x), 0xbf733333u, (vl)), (vl))

/* Bad lanes are replaced by 0.0 before any arithmetic: the fast path stays
   finite on it (m = 0, z = -1) and the epilogue overwrites those lanes, so
   the filler never reaches the result.  0.0 also needs no constant load.  */
#define VM32_LOGF_SAFE(vl) VM32_VFMV_V_F (0.0f, (vl))

static inline float
VM32_KFN (log_fix) (float x)
{
  return (float) log ((double) x);
}

static inline float
VM32_KFN (log2_fix) (float x)
{
  return (float) log2 ((double) x);
}

static inline float
VM32_KFN (log10_fix) (float x)
{
  return (float) log10 ((double) x);
}

/* ---- log2 ---------------------------------------------------------- */

static inline vm32_f_t
VM32_KFN (log2) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_LOGF_BADNORMAL (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_SAFE (vl), mbad, vl);
  vm32_u_t b, off;
  vm32_f_t kf, m, r, l2f, l1;
  VM32_LOGF_TAB_RED (b, off, kf, m, xs, vl);
  VM32_LOGF_TAB_LD (r, l2f, l1, off, vl);
  /* z = r*m - 1 exact; hi = k + log2(1/r) exact.  */
  vm32_f_t z = VM32_VFMACC_VV (VM32_VFMV_V_F (-1.0f, (vl)), r, m, (vl));
  vm32_f_t hi = VM32_VFADD_VV (kf, l1, (vl));
  /* (w, ee) = z*C1 split exactly.  */
  vm32_f_t w = VM32_VFMUL_VF (z, VM32_LOGF_L2E32_HI, (vl));
  vm32_f_t ee = VM32_VFMSAC_VF (w, VM32_LOGF_L2E32_HI, z, (vl));
  /* lo = l2 + z^2*Q(z) + z*C1_lo + ee, accumulated in 2^33-scaled form.  */
  vm32_f_t z2 = VM32_VFMUL_VV (z, z, (vl));
  vm32_f_t q = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_LOGF_L2Q0_33, (vl)),
			       VM32_LOGF_L2Q1_33, z, (vl));
  q = VM32_VFMACC_VF (q, VM32_LOGF_L2Q2_33, z2, (vl));
  vm32_f_t lo = VM32_VFMACC_VV (l2f, z2, q, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_L2E32_LO_33, z, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_S33, ee, (vl));
  /* one final rounding of hi + w + (se + lo).  */
  vm32_f_t s, se;
  VM32_LOGF_TWO_SUM (s, se, hi, w, vl);
  vm32_f_t res = VM32_VFADD_VV (s, VM32_VFMACC_VF (se, VM32_LOGF_IS33, lo,
						   (vl)), (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (log2_fixup) (x, res, mbad, vl, VM32_KFN (log2_fix));
  return res;
}

/* ---- log10 --------------------------------------------------------- */

static inline vm32_f_t
VM32_KFN (log10) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_LOGF_BADNORMAL (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_SAFE (vl), mbad, vl);
  vm32_u_t b, off;
  vm32_f_t kf, m, r, l2f, l1;
  VM32_LOGF_TAB_RED (b, off, kf, m, xs, vl);
  VM32_LOGF_TAB_LD (r, l2f, l1, off, vl);
  vm32_f_t z = VM32_VFMACC_VV (VM32_VFMV_V_F (-1.0f, (vl)), r, m, (vl));
  vm32_f_t hi = VM32_VFADD_VV (kf, l1, (vl));
  /* (ph, pl) = hi*log10(2) split exactly; (w, ee) = z*log10(e) split.  */
  vm32_f_t ph = VM32_VFMUL_VF (hi, VM32_LOGF_L10_2_HI, (vl));
  vm32_f_t pl = VM32_VFMSAC_VF (ph, VM32_LOGF_L10_2_HI, hi, (vl));
  vm32_f_t w = VM32_VFMUL_VF (z, VM32_LOGF_L10E_HI, (vl));
  vm32_f_t ee = VM32_VFMSAC_VF (w, VM32_LOGF_L10E_HI, z, (vl));
  /* lo = log10(2)*l2 + z^2*Q(z) + z*C1_lo + hi*log10(2)_lo + pl + ee.  */
  vm32_f_t z2 = VM32_VFMUL_VV (z, z, (vl));
  vm32_f_t q = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_LOGF_L10Q0_33, (vl)),
			       VM32_LOGF_L10Q1_33, z, (vl));
  q = VM32_VFMACC_VF (q, VM32_LOGF_L10Q2_33, z2, (vl));
  vm32_f_t lo = VM32_VFMACC_VV (
    VM32_VFMUL_VF (l2f, VM32_LOGF_L10_2_HI, (vl)), z2, q, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_L10E_LO_33, z, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_L10_2_LO_33, hi, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_S33, pl, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_S33, ee, (vl));
  vm32_f_t s, se;
  VM32_LOGF_TWO_SUM (s, se, ph, w, vl);
  vm32_f_t res = VM32_VFADD_VV (s, VM32_VFMACC_VF (se, VM32_LOGF_IS33, lo,
						   (vl)), (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (log10_fixup) (x, res, mbad, vl, VM32_KFN (log10_fix));
  return res;
}

/* ---- log ----------------------------------------------------------- */

static inline vm32_f_t
VM32_KFN (log) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_LOGF_BADNORMAL (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_SAFE (vl), mbad, vl);
  vm32_u_t b, off;
  vm32_f_t kf, m, r, l2f, l1;
  VM32_LOGF_TAB_RED (b, off, kf, m, xs, vl);
  VM32_LOGF_TAB_LD (r, l2f, l1, off, vl);
  vm32_f_t z = VM32_VFMACC_VV (VM32_VFMV_V_F (-1.0f, (vl)), r, m, (vl));
  vm32_f_t hi = VM32_VFADD_VV (kf, l1, (vl));
  /* (ph, pl) = hi*ln2 split exactly; ln(1+z) = z + z^2*Q(z) with C1 = 1,
     so the leading term is z itself and there is no residual to carry.  */
  vm32_f_t ph = VM32_VFMUL_VF (hi, VM32_LOGF_LN2_HI, (vl));
  vm32_f_t pl = VM32_VFMSAC_VF (ph, VM32_LOGF_LN2_HI, hi, (vl));
  vm32_f_t z2 = VM32_VFMUL_VV (z, z, (vl));
  vm32_f_t q = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_LOGF_LN2Q0_33, (vl)),
			       VM32_LOGF_LN2Q1_33, z, (vl));
  q = VM32_VFMACC_VF (q, VM32_LOGF_LN2Q2_33, z2, (vl));
  vm32_f_t lo = VM32_VFMACC_VV (
    VM32_VFMUL_VF (l2f, VM32_LOGF_LN2_HI, (vl)), z2, q, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_LN2_LO_33, hi, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_S33, pl, (vl));
  vm32_f_t s, se;
  VM32_LOGF_TWO_SUM (s, se, ph, z, vl);
  vm32_f_t res = VM32_VFADD_VV (s, VM32_VFMACC_VF (se, VM32_LOGF_IS33, lo,
						   (vl)), (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (log_fixup) (x, res, mbad, vl, VM32_KFN (log_fix));
  return res;
}

/* ---- log1p / log2p1 ------------------------------------------------ */

static inline float
VM32_KFN (log1p_fix) (float x)
{
  return (float) log1p ((double) x);
}

static inline float
VM32_KFN (log2p1_fix) (float x)
{
  return (float) (log1p ((double) x) / 0.69314718055994530942);
}

static inline vm32_f_t
VM32_KFN (log1p) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_LOGF_BAD1P (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_SAFE (vl), mbad, vl);
  /* u = RN(1+x) and the exact residual c = x - (u-1): u + c = 1 + x for
     every lane (u-1 alone rounds once u >= 2, and that error is the size
     of the c/u tail itself).  */
  vm32_f_t u = VM32_VFADD_VF (xs, 1.0f, (vl));
  vm32_f_t c = VM32_VFSUB_VV (xs, VM32_VFSUB_VF (u, 1.0f, (vl)), (vl));
  vm32_u_t b = VM32_BITS (u);
  vm32_u_t off = VM32_VAND_VX (VM32_VSRL_VX (b, 13, (vl)), 0x3f8u, (vl));
  vm32_f_t kf = VM32_VFCVT_F_X_V (
    VM32_REINT_U_I (VM32_VSRL_VX (b, 23, (vl))), (vl));
  /* 2^-k, then m = u*2^-k (exact) and z = r*m - 1 (exact).  */
  vm32_f_t i2e = VM32_FROM_BITS (VM32_VSUB_VV (
    VM32_VMV_V_X (0x7f000000u, (vl)), VM32_VAND_VX (b, 0x7f800000u, (vl)),
    (vl)));
  vm32_f_t m = VM32_VFMUL_VV (u, i2e, (vl));
  vm32_f_t r, l2f, l1;
  VM32_LOGF_TAB_LD (r, l2f, l1, off, vl);
  vm32_f_t z = VM32_VFMACC_VV (VM32_VFMV_V_F (-1.0f, (vl)), r, m, (vl));
  /* c/u = q1/(1+z) with q1 = 2^-k * r * c (exact except for one rounding),
     so log1p (c/u) = q1*(1-z) + O(q1*z^2) -- 2^-38 absolute, far below the
     final rounding, and the polynomial keeps the exact z.  */
  vm32_f_t q1 = VM32_VFMUL_VV (VM32_VFMUL_VV (c, i2e, (vl)), r, (vl));
  vm32_f_t tail = VM32_VFNMSAC_VV (q1, q1, z, (vl));
  vm32_f_t zp = z;
  vm32_f_t hi = VM32_VFADD_VV (kf, l1, (vl));
  vm32_f_t ph = VM32_VFMUL_VF (hi, VM32_LOGF_LN2_HI, (vl));
  vm32_f_t pl = VM32_VFMSAC_VF (ph, VM32_LOGF_LN2_HI, hi, (vl));
  vm32_f_t z2 = VM32_VFMUL_VV (zp, zp, (vl));
  vm32_f_t q = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_LOGF_LN2Q0_33, (vl)),
			       VM32_LOGF_LN2Q1_33, zp, (vl));
  q = VM32_VFMACC_VF (q, VM32_LOGF_LN2Q2_33, z2, (vl));
  vm32_f_t lo = VM32_VFMACC_VV (
    VM32_VFMUL_VF (l2f, VM32_LOGF_LN2_HI, (vl)), z2, q, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_S33, tail, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_LN2_LO_33, hi, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_S33, pl, (vl));
  vm32_f_t s, se;
  VM32_LOGF_TWO_SUM (s, se, ph, zp, vl);
  vm32_f_t res = VM32_VFADD_VV (s, VM32_VFMACC_VF (se, VM32_LOGF_IS33, lo,
						   (vl)), (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      res = VM32_KFN (log1p_fixup) (x, res, mbad, vl, VM32_KFN (log1p_fix));
      /* Only an epilogue lane can hold a subnormal result: the fast path
	 covers |x| >= 2^-30, where |log1p (x)| >= 2^-30, so the carrier
	 never fires on a fast lane and stays off the hot path.  */
      VM32_LOGF_RAISE_UF_SUB (res, vl);
      return res;
    }
  return res;
}

static inline vm32_f_t
VM32_KFN (log2p1) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_LOGF_BAD1P (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_SAFE (vl), mbad, vl);
  vm32_f_t u = VM32_VFADD_VF (xs, 1.0f, (vl));
  vm32_f_t c = VM32_VFSUB_VV (xs, VM32_VFSUB_VF (u, 1.0f, (vl)), (vl));
  vm32_u_t b = VM32_BITS (u);
  vm32_u_t off = VM32_VAND_VX (VM32_VSRL_VX (b, 13, (vl)), 0x3f8u, (vl));
  vm32_f_t kf = VM32_VFCVT_F_X_V (
    VM32_REINT_U_I (VM32_VSRL_VX (b, 23, (vl))), (vl));
  vm32_f_t i2e = VM32_FROM_BITS (VM32_VSUB_VV (
    VM32_VMV_V_X (0x7f000000u, (vl)), VM32_VAND_VX (b, 0x7f800000u, (vl)),
    (vl)));
  vm32_f_t m = VM32_VFMUL_VV (u, i2e, (vl));
  vm32_f_t r, l2f, l1;
  VM32_LOGF_TAB_LD (r, l2f, l1, off, vl);
  vm32_f_t z = VM32_VFMACC_VV (VM32_VFMV_V_F (-1.0f, (vl)), r, m, (vl));
  vm32_f_t q1 = VM32_VFMUL_VV (VM32_VFMUL_VV (c, i2e, (vl)), r, (vl));
  vm32_f_t zp = VM32_VFADD_VV (z, q1, (vl));
  vm32_f_t ez = VM32_VFADD_VV (VM32_VFSUB_VV (z, zp, (vl)), q1, (vl));
  vm32_f_t hi = VM32_VFADD_VV (kf, l1, (vl));
  vm32_f_t w = VM32_VFMUL_VF (zp, VM32_LOGF_L2E32_HI, (vl));
  vm32_f_t ee = VM32_VFMSAC_VF (w, VM32_LOGF_L2E32_HI, zp, (vl));
  vm32_f_t z2 = VM32_VFMUL_VV (zp, zp, (vl));
  vm32_f_t q = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_LOGF_L2Q0_33, (vl)),
			       VM32_LOGF_L2Q1_33, zp, (vl));
  q = VM32_VFMACC_VF (q, VM32_LOGF_L2Q2_33, z2, (vl));
  vm32_f_t lo = VM32_VFMACC_VV (l2f, z2, q, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_L2E32_LO_33, zp, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_L2E32_HI_33, ez, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_S33, ee, (vl));
  vm32_f_t s, se;
  VM32_LOGF_TWO_SUM (s, se, hi, w, vl);
  vm32_f_t res = VM32_VFADD_VV (s, VM32_VFMACC_VF (se, VM32_LOGF_IS33, lo,
						   (vl)), (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (log2p1_fixup) (x, res, mbad, vl, VM32_KFN (log2p1_fix));
  return res;
}
