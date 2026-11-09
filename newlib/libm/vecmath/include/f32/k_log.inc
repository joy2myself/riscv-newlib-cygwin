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

   Fast-path accuracy: the reduction carries a two-part exponent step
   (k*ln2 with a 15-bit hi part, so k*hi is exact for every binary32
   exponent) and log1p(f) is evaluated as an f32 double-float pair, so the
   final add rounds the correctly rounded f32 log at ~0.5 ulp.  log2,
   log10 and log2p1 compose with two-part constants and keep that bound.  */

/* ------------------------------------------------------------------ */
/* f32 double-float building blocks (file-local names).               */
/* ------------------------------------------------------------------ */

#undef VM32_LOGF_TWO_SUM
#undef VM32_LOGF_FAST2_SUM
#undef VM32_LOGF_LOG1P_DD
#undef VM32_LOGF_REDUCE
#undef VM32_LOGF_FIXUP1
#undef VM32_LOGF_RAISE_UF_SUB

/* (s, e) = a + b exactly (Knuth two-sum; general operand order).  */
#define VM32_LOGF_TWO_SUM(s, e, a, b, vl) do {				\
  vm32_f_t lts_s_ = VM32_VFADD_VV ((a), (b), (vl));			\
  vm32_f_t lts_v_ = VM32_VFSUB_VV (lts_s_, (a), (vl));			\
  (s) = lts_s_;								\
  (e) = VM32_VFADD_VV (							\
    VM32_VFSUB_VV ((a), VM32_VFSUB_VV (lts_s_, lts_v_, (vl)), (vl)),	\
    VM32_VFSUB_VV ((b), lts_v_, (vl)), (vl));				\
} while (0)

/* (s, e) = a + b exactly, requires |a| >= |b|.  */
#define VM32_LOGF_FAST2_SUM(s, e, a, b, vl) do {			\
  (s) = VM32_VFADD_VV ((a), (b), (vl));					\
  (e) = VM32_VFSUB_VV ((b), VM32_VFSUB_VV ((s), (a), (vl)), (vl));	\
} while (0)

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

/* Two-part constants: 15 significant bits in the hi part keeps k*hi
   exact for every binary32 exponent k, and the f32 lo part then carries
   the remaining ~2^-43 relative accuracy.  */
#define VM32_LOGF_LN2_HI	0.693145751953125f	/* 0x3f317200 */
#define VM32_LOGF_LN2_LO	1.428606765330187e-06f	/* 0x35bfbe8e */
#define VM32_LOGF_L2E_HI	1.44268798828125f	/* 0x3fb8aa00 */
#define VM32_LOGF_L2E_LO	7.052607543300837e-06f	/* 0x36eca570 */
#define VM32_LOGF_L10E_HI	0.4342803955078125f	/* 0x3ede5a00 */
#define VM32_LOGF_L10E_LO	1.408639582223259e-05f	/* 0x376c549c */
#define VM32_LOGF_L10_2_HI	0.301025390625f		/* 0x3e9a2000 */
#define VM32_LOGF_L10_2_LO	4.605039066518657e-06f	/* 0x369a84fc */

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
#undef VM32_LOGF_DEF_FIXUP
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

/* ---- log / log2 / log10 ------------------------------------------- */

/* Exceptional lanes for the plain log family: anything that is not a
   positive normal binary32 (zero, subnormal, negative, Inf, NaN).  */
#define VM32_LOGF_BADNORMAL(x, vl)					\
  VM32_VMOR_MM (							\
    VM32_VMSGEU_VX (							\
      VM32_VSUB_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x00800000u, (vl)),					\
      0x7f000000u, (vl)),						\
    VM32_VMSGEU_VX (VM32_BITS (x), 0x80000000u, (vl)), (vl))

#define VM32_LOGF_ONE(vl) VM32_VFMV_V_F (1.0f, (vl))

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

static inline vm32_f_t
VM32_KFN (log) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_LOGF_BADNORMAL (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_ONE (vl), mbad, vl);
  vm32_f_t h, l;
  VM32_LOGF_LOG_DD (h, l, xs, vl);
  vm32_f_t r = VM32_VFADD_VV (h, l, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (log_fixup) (x, r, mbad, vl, VM32_KFN (log_fix));
  return r;
}

static inline vm32_f_t
VM32_KFN (log2) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_LOGF_BADNORMAL (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_ONE (vl), mbad, vl);
  vm32_f_t ph, pl, kf;
  VM32_LOGF_REDUCE (ph, pl, kf, xs, vl);
  /* log2(x) = k + log1p(f)*log2(e), the product carried as a pair.  */
  vm32_f_t p = VM32_VFMUL_VF (ph, VM32_LOGF_L2E_HI, (vl));
  vm32_f_t pe = VM32_VFMSAC_VF (p, VM32_LOGF_L2E_HI, ph, (vl));
  vm32_f_t s, se;
  VM32_LOGF_TWO_SUM (s, se, kf, p, (vl));
  pe = VM32_VFADD_VV (pe, VM32_VFMUL_VF (pl, VM32_LOGF_L2E_HI, (vl)),
		      (vl));
  pe = VM32_VFMACC_VF (pe, VM32_LOGF_L2E_LO, ph, (vl));
  vm32_f_t r = VM32_VFADD_VV (s, VM32_VFADD_VV (se, pe, (vl)), (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (log2_fixup) (x, r, mbad, vl, VM32_KFN (log2_fix));
  return r;
}

static inline vm32_f_t
VM32_KFN (log10) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_LOGF_BADNORMAL (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_ONE (vl), mbad, vl);
  vm32_f_t ph, pl, kf;
  VM32_LOGF_REDUCE (ph, pl, kf, xs, vl);
  /* log10(x) = k*log10(2) + log1p(f)*log10(e).  */
  vm32_f_t base = VM32_VFMUL_VF (kf, VM32_LOGF_L10_2_HI, (vl));
  vm32_f_t p = VM32_VFMUL_VF (ph, VM32_LOGF_L10E_HI, (vl));
  vm32_f_t pe = VM32_VFMSAC_VF (p, VM32_LOGF_L10E_HI, ph, (vl));
  vm32_f_t s, se;
  VM32_LOGF_TWO_SUM (s, se, base, p, (vl));
  pe = VM32_VFADD_VV (pe, VM32_VFMUL_VF (pl, VM32_LOGF_L10E_HI, (vl)),
		      (vl));
  pe = VM32_VFMACC_VF (pe, VM32_LOGF_L10E_LO, ph, (vl));
  pe = VM32_VFMACC_VF (pe, VM32_LOGF_L10_2_LO, kf, (vl));
  vm32_f_t r = VM32_VFADD_VV (s, VM32_VFADD_VV (se, pe, (vl)), (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (log10_fixup) (x, r, mbad, vl, VM32_KFN (log10_fix));
  return r;
}

/* ---- log1p / log2p1 ------------------------------------------------ */

/* Exceptional lanes: x < -0.94 (bits above -0.94 as x grows more
   negative; covers -Inf and negative NaN), x >= 2^60 or +Inf/NaN (the
   unsigned window test), and |x| < 2^-30 (tiny, where x*x would slowly
   lose precision).  The benchmark domain is [-0.9, 10], comfortably
   inside.  */
#define VM32_LOGF_BAD1P(x, vl)						\
  VM32_VMOR_MM (							\
    VM32_VMOR_MM (							\
      VM32_VMSGTU_VX (VM32_BITS (x), 0xbf733333u, (vl)),		\
      VM32_VMSLTU_VX (							\
	VM32_VSUB_VX (VM32_BITS (x), 0x5d800000u, (vl)),		\
	0x22800000u, (vl)), (vl)),					\
    VM32_VMSLTU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x30800000u, (vl)), (vl))

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
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_ONE (vl), mbad, vl);
  /* Exact 1+x pair, then log(uh) + log1p(ul/uh); ul/uh <= 2^-25 so the
     series truncates after its quadratic term.  */
  vm32_f_t uh, ul;
  VM32_LOGF_TWO_SUM (uh, ul, VM32_LOGF_ONE (vl), xs, vl);
  vm32_f_t c = VM32_VFDIV_VV (ul, uh, (vl));
  vm32_f_t h, l;
  VM32_LOGF_LOG_DD (h, l, uh, vl);
  vm32_f_t t = VM32_VFADD_VV (l, c, (vl));
  t = VM32_VFNMSAC_VF (t, 0.5f, VM32_VFMUL_VV (c, c, (vl)), (vl));
  vm32_f_t r = VM32_VFADD_VV (h, t, (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    r = VM32_KFN (log1p_fixup) (x, r, mbad, vl, VM32_KFN (log1p_fix));
  VM32_LOGF_RAISE_UF_SUB (r, vl);
  return r;
}

static inline vm32_f_t
VM32_KFN (log2p1) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_LOGF_BAD1P (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_LOGF_ONE (vl), mbad, vl);
  vm32_f_t uh, ul;
  VM32_LOGF_TWO_SUM (uh, ul, VM32_LOGF_ONE (vl), xs, vl);
  vm32_f_t c = VM32_VFDIV_VV (ul, uh, (vl));
  vm32_f_t h, l;
  VM32_LOGF_LOG_DD (h, l, uh, vl);
  /* log2(1+x) = (h + c - c^2/2) * log2(e), evaluated as a double-float:
     s = h + c exactly, then the product with the two-part constant plus
     the small quadratic correction, one final add.  */
  vm32_f_t s, se;
  VM32_LOGF_TWO_SUM (s, se, h, c, vl);
  vm32_f_t pe = VM32_VFMSAC_VF (
    VM32_VFMUL_VF (s, VM32_LOGF_L2E_HI, (vl)), VM32_LOGF_L2E_HI, s, (vl));
  vm32_f_t low = VM32_VFMACC_VF (pe, VM32_LOGF_L2E_HI, se, (vl));
  low = VM32_VFMACC_VF (low, VM32_LOGF_L2E_LO, s, (vl));
  low = VM32_VFMACC_VF (low, VM32_LOGF_L2E_HI,
			VM32_VFMUL_VF (VM32_VFMUL_VV (c, c, (vl)), -0.5f,
				       (vl)), (vl));
  low = VM32_VFADD_VV (low, VM32_VFMUL_VF (l, VM32_LOGF_L2E_HI, (vl)),
		       (vl));
  vm32_f_t r = VM32_VFADD_VV (
    VM32_VFMUL_VF (s, VM32_LOGF_L2E_HI, (vl)), low, (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (log2p1_fixup) (x, r, mbad, vl, VM32_KFN (log2p1_fix));
  return r;
}
