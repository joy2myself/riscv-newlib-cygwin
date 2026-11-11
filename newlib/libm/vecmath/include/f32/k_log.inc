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

/* ---- log2 table core (log2p1) -------------------------------------- */

/* The binary32 twin of the binary64 log2/log2p1 table core: one exact FMA
   residual instead of a division, and no double-float combine.

   u = 2^k * m with m in [1,2).  The top 7 mantissa bits select one of 128
   slices; the entry {r, l1, l2} holds r = q/128 with q = round(128/m_mid),
   so

     z = fma (r, m, -1)      EXACT in binary32: r*m is a multiple of 2^-30
                             and |z| <= 2^-6.79, i.e. at most 23 significant
                             bits (verified per slice against both ends of
                             the slice, see the table below)

   and l1 + l2 = log2(1/r), split so that l1 is a multiple of 2^-12 -- then
   k + l1 is exact for every binary32 exponent (|k| <= 60 needs 6 bits) and
   only the lo part rounds.  Slices 0 and 127 use r = 1 and r = 1/2 exactly:
   l1 = 0 and l1 = 1, so log2(u) next to 1 -- from either side -- carries the
   whole answer in the polynomial instead of cancelling against a coarse
   table value.

   log2(1+z) = z*C1 + z^2*Q(z) with C1 = log2(e) split as an f32 pair and
   Q the z-series of -C1/2 + C1 z/3 - C1 z^2/4 (|z| <= 2^-6.79 makes the
   truncation 2^-31, far below the final rounding).  */

#ifndef VM32_LOGF_LOG2_CORE_TAB
#define VM32_LOGF_LOG2_CORE_TAB 1

static const struct vm32_log2_core_ent
{
  float r, l1, l2, pad;
} __vm32_log2_core_tab[128] =
{
  { 0x1p+0f, -0x1.fcp+6f, 0.0f, 0.0f },                   /*   0  r = 128/128 */
  { 0x1.fcp-1f, -0x1.fbf48p+6f, 0x1.63dd1p-14f, 0.0f },   /*   1  r = 127/128 */
  { 0x1.f8p-1f, -0x1.fbe8cp+6f, 0x1.f7431p-17f, 0.0f },   /*   2  r = 126/128 */
  { 0x1.f4p-1f, -0x1.fbddp+6f, 0x1.2e3932p-15f, 0.0f },   /*   3  r = 125/128 */
  { 0x1.fp-1f, -0x1.fbd1p+6f, -0x1.8d66c4p-14f, 0.0f },   /*   4  r = 124/128 */
  { 0x1.ecp-1f, -0x1.fbc54p+6f, 0x1.d7a3e4p-14f, 0.0f },  /*   5  r = 123/128 */
  { 0x1.e8p-1f, -0x1.fbb9p+6f, -0x1.33568p-14f, 0.0f },   /*   6  r = 122/128 */
  { 0x1.e4p-1f, -0x1.fbadp+6f, 0x1.583f9ap-14f, 0.0f },   /*   7  r = 121/128 */
  { 0x1.ep-1f, -0x1.fba0cp+6f, 0x1.8125b4p-14f, 0.0f },   /*   8  r = 120/128 */
  { 0x1.dcp-1f, -0x1.fb944p+6f, -0x1.6372a4p-15f, 0.0f }, /*   9  r = 119/128 */
  { 0x1.d8p-1f, -0x1.fb87cp+6f, -0x1.3945c4p-14f, 0.0f }, /*  10  r = 118/128 */
  { 0x1.d4p-1f, -0x1.fb7b4p+6f, -0x1.c731ap-19f, 0.0f },  /*  11  r = 117/128 */
  { 0x1.d4p-1f, -0x1.fb7b4p+6f, -0x1.c731ap-19f, 0.0f },  /*  12  r = 117/128 */
  { 0x1.dp-1f, -0x1.fb6e8p+6f, -0x1.291eaap-14f, 0.0f },  /*  13  r = 116/128 */
  { 0x1.ccp-1f, -0x1.fb61cp+6f, -0x1.049af4p-15f, 0.0f }, /*  14  r = 115/128 */
  { 0x1.c8p-1f, -0x1.fb55p+6f, 0x1.ee15p-14f, 0.0f },     /*  15  r = 114/128 */
  { 0x1.c4p-1f, -0x1.fb47cp+6f, -0x1.cfe71cp-14f, 0.0f }, /*  16  r = 113/128 */
  { 0x1.c4p-1f, -0x1.fb47cp+6f, -0x1.cfe71cp-14f, 0.0f }, /*  17  r = 113/128 */
  { 0x1.cp-1f, -0x1.fb3acp+6f, 0x1.30158p-16f, 0.0f },    /*  18  r = 112/128 */
  { 0x1.bcp-1f, -0x1.fb2d8p+6f, 0x1.296a9ap-16f, 0.0f },  /*  19  r = 111/128 */
  { 0x1.b8p-1f, -0x1.fb2p+6f, -0x1.cc2cp-14f, 0.0f },     /*  20  r = 110/128 */
  { 0x1.b8p-1f, -0x1.fb2p+6f, -0x1.cc2cp-14f, 0.0f },     /*  21  r = 110/128 */
  { 0x1.b4p-1f, -0x1.fb128p+6f, -0x1.ee960ep-14f, 0.0f }, /*  22  r = 109/128 */
  { 0x1.bp-1f, -0x1.fb05p+6f, -0x1.3ab7cep-18f, 0.0f },   /*  23  r = 108/128 */
  { 0x1.acp-1f, -0x1.faf74p+6f, -0x1.8f935p-17f, 0.0f },  /*  24  r = 107/128 */
  { 0x1.acp-1f, -0x1.faf74p+6f, -0x1.8f935p-17f, 0.0f },  /*  25  r = 107/128 */
  { 0x1.a8p-1f, -0x1.fae98p+6f, 0x1.c05364p-14f, 0.0f },  /*  26  r = 106/128 */
  { 0x1.a4p-1f, -0x1.fadb8p+6f, 0x1.cd2b14p-14f, 0.0f },  /*  27  r = 105/128 */
  { 0x1.a4p-1f, -0x1.fadb8p+6f, 0x1.cd2b14p-14f, 0.0f },  /*  28  r = 105/128 */
  { 0x1.ap-1f, -0x1.facd4p+6f, -0x1.1c8f12p-22f, 0.0f },  /*  29  r = 104/128 */
  { 0x1.9cp-1f, -0x1.fabfp+6f, 0x1.805f1p-16f, 0.0f },    /*  30  r = 103/128 */
  { 0x1.9cp-1f, -0x1.fabfp+6f, 0x1.805f1p-16f, 0.0f },    /*  31  r = 103/128 */
  { 0x1.98p-1f, -0x1.fab08p+6f, -0x1.044d32p-14f, 0.0f }, /*  32  r = 102/128 */
  { 0x1.94p-1f, -0x1.faa2p+6f, -0x1.18708ap-17f, 0.0f },  /*  33  r = 101/128 */
  { 0x1.94p-1f, -0x1.faa2p+6f, -0x1.18708ap-17f, 0.0f },  /*  34  r = 101/128 */
  { 0x1.9p-1f, -0x1.fa934p+6f, -0x1.e12f34p-15f, 0.0f },  /*  35  r = 100/128 */
  { 0x1.9p-1f, -0x1.fa934p+6f, -0x1.e12f34p-15f, 0.0f },  /*  36  r = 100/128 */
  { 0x1.8cp-1f, -0x1.fa848p+6f, 0x1.3e059ep-15f, 0.0f },  /*  37  r =  99/128 */
  { 0x1.88p-1f, -0x1.fa758p+6f, 0x1.30158p-15f, 0.0f },   /*  38  r =  98/128 */
  { 0x1.88p-1f, -0x1.fa758p+6f, 0x1.30158p-15f, 0.0f },   /*  39  r =  98/128 */
  { 0x1.84p-1f, -0x1.fa664p+6f, -0x1.f1aad2p-15f, 0.0f }, /*  40  r =  97/128 */
  { 0x1.84p-1f, -0x1.fa664p+6f, -0x1.f1aad2p-15f, 0.0f }, /*  41  r =  97/128 */
  { 0x1.8p-1f, -0x1.fa57p+6f, -0x1.a39fbep-20f, 0.0f },   /*  42  r =  96/128 */
  { 0x1.8p-1f, -0x1.fa57p+6f, -0x1.a39fbep-20f, 0.0f },   /*  43  r =  96/128 */
  { 0x1.7cp-1f, -0x1.fa478p+6f, -0x1.07509cp-15f, 0.0f }, /*  44  r =  95/128 */
  { 0x1.78p-1f, -0x1.fa38p+6f, 0x1.9dc2d4p-14f, 0.0f },   /*  45  r =  94/128 */
  { 0x1.78p-1f, -0x1.fa38p+6f, 0x1.9dc2d4p-14f, 0.0f },   /*  46  r =  94/128 */
  { 0x1.74p-1f, -0x1.fa28p+6f, -0x1.93f544p-14f, 0.0f },  /*  47  r =  93/128 */
  { 0x1.74p-1f, -0x1.fa28p+6f, -0x1.93f544p-14f, 0.0f },  /*  48  r =  93/128 */
  { 0x1.7p-1f, -0x1.fa184p+6f, 0x1.f5fe54p-14f, 0.0f },   /*  49  r =  92/128 */
  { 0x1.7p-1f, -0x1.fa184p+6f, 0x1.f5fe54p-14f, 0.0f },   /*  50  r =  92/128 */
  { 0x1.6cp-1f, -0x1.fa08p+6f, 0x1.2ba344p-16f, 0.0f },   /*  51  r =  91/128 */
  { 0x1.6cp-1f, -0x1.fa08p+6f, 0x1.2ba344p-16f, 0.0f },   /*  52  r =  91/128 */
  { 0x1.68p-1f, -0x1.f9f7cp+6f, 0x1.7a9734p-14f, 0.0f },  /*  53  r =  90/128 */
  { 0x1.68p-1f, -0x1.f9f7cp+6f, 0x1.7a9734p-14f, 0.0f },  /*  54  r =  90/128 */
  { 0x1.64p-1f, -0x1.f9e74p+6f, 0x1.955e18p-14f, 0.0f },  /*  55  r =  89/128 */
  { 0x1.64p-1f, -0x1.f9e74p+6f, 0x1.955e18p-14f, 0.0f },  /*  56  r =  89/128 */
  { 0x1.6p-1f, -0x1.f9d68p+6f, 0x1.583f9ap-15f, 0.0f },   /*  57  r =  88/128 */
  { 0x1.6p-1f, -0x1.f9d68p+6f, 0x1.583f9ap-15f, 0.0f },   /*  58  r =  88/128 */
  { 0x1.5cp-1f, -0x1.f9c58p+6f, -0x1.2fad2ap-14f, 0.0f }, /*  59  r =  87/128 */
  { 0x1.5cp-1f, -0x1.f9c58p+6f, -0x1.2fad2ap-14f, 0.0f }, /*  60  r =  87/128 */
  { 0x1.58p-1f, -0x1.f9b48p+6f, 0x1.408c78p-18f, 0.0f },  /*  61  r =  86/128 */
  { 0x1.58p-1f, -0x1.f9b48p+6f, 0x1.408c78p-18f, 0.0f },  /*  62  r =  86/128 */
  { 0x1.58p-1f, -0x1.f9b48p+6f, 0x1.408c78p-18f, 0.0f },  /*  63  r =  86/128 */
  { 0x1.54p-1f, -0x1.f9a34p+6f, 0x1.13eb02p-15f, 0.0f },  /*  64  r =  85/128 */
  { 0x1.54p-1f, -0x1.f9a34p+6f, 0x1.13eb02p-15f, 0.0f },  /*  65  r =  85/128 */
  { 0x1.5p-1f, -0x1.f991cp+6f, 0x1.15db84p-16f, 0.0f },   /*  66  r =  84/128 */
  { 0x1.5p-1f, -0x1.f991cp+6f, 0x1.15db84p-16f, 0.0f },   /*  67  r =  84/128 */
  { 0x1.4cp-1f, -0x1.f98p+6f, -0x1.4ac62cp-15f, 0.0f },   /*  68  r =  83/128 */
  { 0x1.4cp-1f, -0x1.f98p+6f, -0x1.4ac62cp-15f, 0.0f },   /*  69  r =  83/128 */
  { 0x1.4cp-1f, -0x1.f98p+6f, -0x1.4ac62cp-15f, 0.0f },   /*  70  r =  83/128 */
  { 0x1.48p-1f, -0x1.f96e4p+6f, 0x1.de3262p-14f, 0.0f },  /*  71  r =  82/128 */
  { 0x1.48p-1f, -0x1.f96e4p+6f, 0x1.de3262p-14f, 0.0f },  /*  72  r =  82/128 */
  { 0x1.44p-1f, -0x1.f95cp+6f, -0x1.a39fbep-18f, 0.0f },  /*  73  r =  81/128 */
  { 0x1.44p-1f, -0x1.f95cp+6f, -0x1.a39fbep-18f, 0.0f },  /*  74  r =  81/128 */
  { 0x1.44p-1f, -0x1.f95cp+6f, -0x1.a39fbep-18f, 0.0f },  /*  75  r =  81/128 */
  { 0x1.4p-1f, -0x1.f949cp+6f, 0x1.87b432p-14f, 0.0f },   /*  76  r =  80/128 */
  { 0x1.4p-1f, -0x1.f949cp+6f, 0x1.87b432p-14f, 0.0f },   /*  77  r =  80/128 */
  { 0x1.3cp-1f, -0x1.f937p+6f, -0x1.24cea4p-14f, 0.0f },  /*  78  r =  79/128 */
  { 0x1.3cp-1f, -0x1.f937p+6f, -0x1.24cea4p-14f, 0.0f },  /*  79  r =  79/128 */
  { 0x1.3cp-1f, -0x1.f937p+6f, -0x1.24cea4p-14f, 0.0f },  /*  80  r =  79/128 */
  { 0x1.38p-1f, -0x1.f9244p+6f, -0x1.eac382p-20f, 0.0f }, /*  81  r =  78/128 */
  { 0x1.38p-1f, -0x1.f9244p+6f, -0x1.eac382p-20f, 0.0f }, /*  82  r =  78/128 */
  { 0x1.34p-1f, -0x1.f9114p+6f, 0x1.f04a5ap-15f, 0.0f },  /*  83  r =  77/128 */
  { 0x1.34p-1f, -0x1.f9114p+6f, 0x1.f04a5ap-15f, 0.0f },  /*  84  r =  77/128 */
  { 0x1.34p-1f, -0x1.f9114p+6f, 0x1.f04a5ap-15f, 0.0f },  /*  85  r =  77/128 */
  { 0x1.3p-1f, -0x1.f8fep+6f, 0x1.f4a37ep-14f, 0.0f },    /*  86  r =  76/128 */
  { 0x1.3p-1f, -0x1.f8fep+6f, 0x1.f4a37ep-14f, 0.0f },    /*  87  r =  76/128 */
  { 0x1.3p-1f, -0x1.f8fep+6f, 0x1.f4a37ep-14f, 0.0f },    /*  88  r =  76/128 */
  { 0x1.2cp-1f, -0x1.f8ea4p+6f, -0x1.ee4c32p-15f, 0.0f }, /*  89  r =  75/128 */
  { 0x1.2cp-1f, -0x1.f8ea4p+6f, -0x1.ee4c32p-15f, 0.0f }, /*  90  r =  75/128 */
  { 0x1.2cp-1f, -0x1.f8ea4p+6f, -0x1.ee4c32p-15f, 0.0f }, /*  91  r =  75/128 */
  { 0x1.28p-1f, -0x1.f8d68p+6f, 0x1.43a496p-16f, 0.0f },  /*  92  r =  74/128 */
  { 0x1.28p-1f, -0x1.f8d68p+6f, 0x1.43a496p-16f, 0.0f },  /*  93  r =  74/128 */
  { 0x1.28p-1f, -0x1.f8d68p+6f, 0x1.43a496p-16f, 0.0f },  /*  94  r =  74/128 */
  { 0x1.24p-1f, -0x1.f8c28p+6f, 0x1.ea17e8p-14f, 0.0f },  /*  95  r =  73/128 */
  { 0x1.24p-1f, -0x1.f8c28p+6f, 0x1.ea17e8p-14f, 0.0f },  /*  96  r =  73/128 */
  { 0x1.24p-1f, -0x1.f8c28p+6f, 0x1.ea17e8p-14f, 0.0f },  /*  97  r =  73/128 */
  { 0x1.2p-1f, -0x1.f8aep+6f, -0x1.a39fbep-19f, 0.0f },   /*  98  r =  72/128 */
  { 0x1.2p-1f, -0x1.f8aep+6f, -0x1.a39fbep-19f, 0.0f },   /*  99  r =  72/128 */
  { 0x1.2p-1f, -0x1.f8aep+6f, -0x1.a39fbep-19f, 0.0f },   /* 100  r =  72/128 */
  { 0x1.1cp-1f, -0x1.f8994p+6f, -0x1.74f13cp-14f, 0.0f }, /* 101  r =  71/128 */
  { 0x1.1cp-1f, -0x1.f8994p+6f, -0x1.74f13cp-14f, 0.0f }, /* 102  r =  71/128 */
  { 0x1.1cp-1f, -0x1.f8994p+6f, -0x1.74f13cp-14f, 0.0f }, /* 103  r =  71/128 */
  { 0x1.18p-1f, -0x1.f8848p+6f, 0x1.d3b992p-14f, 0.0f },  /* 104  r =  70/128 */
  { 0x1.18p-1f, -0x1.f8848p+6f, 0x1.d3b992p-14f, 0.0f },  /* 105  r =  70/128 */
  { 0x1.18p-1f, -0x1.f8848p+6f, 0x1.d3b992p-14f, 0.0f },  /* 106  r =  70/128 */
  { 0x1.18p-1f, -0x1.f8848p+6f, 0x1.d3b992p-14f, 0.0f },  /* 107  r =  70/128 */
  { 0x1.14p-1f, -0x1.f86f4p+6f, 0x1.ef6fd4p-14f, 0.0f },  /* 108  r =  69/128 */
  { 0x1.14p-1f, -0x1.f86f4p+6f, 0x1.ef6fd4p-14f, 0.0f },  /* 109  r =  69/128 */
  { 0x1.14p-1f, -0x1.f86f4p+6f, 0x1.ef6fd4p-14f, 0.0f },  /* 110  r =  69/128 */
  { 0x1.1p-1f, -0x1.f8598p+6f, -0x1.fb7d64p-15f, 0.0f },  /* 111  r =  68/128 */
  { 0x1.1p-1f, -0x1.f8598p+6f, -0x1.fb7d64p-15f, 0.0f },  /* 112  r =  68/128 */
  { 0x1.1p-1f, -0x1.f8598p+6f, -0x1.fb7d64p-15f, 0.0f },  /* 113  r =  68/128 */
  { 0x1.1p-1f, -0x1.f8598p+6f, -0x1.fb7d64p-15f, 0.0f },  /* 114  r =  68/128 */
  { 0x1.0cp-1f, -0x1.f843cp+6f, 0x1.31d818p-14f, 0.0f },  /* 115  r =  67/128 */
  { 0x1.0cp-1f, -0x1.f843cp+6f, 0x1.31d818p-14f, 0.0f },  /* 116  r =  67/128 */
  { 0x1.0cp-1f, -0x1.f843cp+6f, 0x1.31d818p-14f, 0.0f },  /* 117  r =  67/128 */
  { 0x1.08p-1f, -0x1.f82d8p+6f, 0x1.4b229cp-15f, 0.0f },  /* 118  r =  66/128 */
  { 0x1.08p-1f, -0x1.f82d8p+6f, 0x1.4b229cp-15f, 0.0f },  /* 119  r =  66/128 */
  { 0x1.08p-1f, -0x1.f82d8p+6f, 0x1.4b229cp-15f, 0.0f },  /* 120  r =  66/128 */
  { 0x1.08p-1f, -0x1.f82d8p+6f, 0x1.4b229cp-15f, 0.0f },  /* 121  r =  66/128 */
  { 0x1.04p-1f, -0x1.f817p+6f, 0x1.8697a4p-14f, 0.0f },   /* 122  r =  65/128 */
  { 0x1.04p-1f, -0x1.f817p+6f, 0x1.8697a4p-14f, 0.0f },   /* 123  r =  65/128 */
  { 0x1.04p-1f, -0x1.f817p+6f, 0x1.8697a4p-14f, 0.0f },   /* 124  r =  65/128 */
  { 0x1.04p-1f, -0x1.f817p+6f, 0x1.8697a4p-14f, 0.0f },   /* 125  r =  65/128 */
  { 0x1p-1f, -0x1.f8p+6f, 0.0f, 0.0f },                   /* 126  r =  64/128 */
  { 0x1p-1f, -0x1.f8p+6f, 0.0f, 0.0f },                   /* 127  r =  64/128 */
};

#endif /* VM32_LOGF_LOG2_CORE_TAB */

#define VM32_LOGF_L2E32_HI	0x1.715476p+0f	/* log2(e) */
#define VM32_LOGF_L2E32_LO	0x1.4ae0c0p-26f	/* log2(e) - HI */
#define VM32_LOGF_L2Q0		-0x1.715476p-1f	/* -C1/2 */
#define VM32_LOGF_L2Q1		 0x1.ec709ep-2f	/*  C1/3 */
#define VM32_LOGF_L2Q2		-0x1.715476p-2f	/* -C1/4 */

/* ---- log1p / log2p1 ------------------------------------------------ */

/* Exceptional lanes: x < -0.94 (bits above -0.94 as x grows more
   negative; covers -Inf and negative NaN), x >= 2^60 or +Inf/NaN (the
   unsigned window test), and |x| < 2^-30 (tiny, where x*x would slowly
   lose precision).  The benchmark domain is [-0.9, 10], comfortably
   inside.  */
#define VM32_LOGF_BAD1P(x, vl)						\
  VM32_VMOR_MM (							\
    VM32_VMSGEU_VX (							\
      VM32_VSUB_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x30800000u, (vl)),					\
      0x2d000000u, (vl)),						\
    VM32_VMSGTU_VX (VM32_BITS (x), 0xbf733333u, (vl)), (vl))

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
  /* u = RN(1+x) and the exact residual c = x - (u-1): the two-sum makes
     u + c = 1 + x hold for every lane (u-1 alone rounds once u >= 2, and
     that error is the size of the c/u tail itself).  */
  vm32_f_t u = VM32_VFADD_VF (xs, 1.0f, (vl));
  vm32_f_t c = VM32_VFSUB_VV (xs, VM32_VFSUB_VF (u, 1.0f, (vl)), (vl));
  /* Table reduction: index = top 7 mantissa bits of u (16-byte stride).  */
  vm32_u_t b = VM32_BITS (u);
  vm32_u_t off = VM32_VAND_VX (VM32_VSRL_VX (b, 12, (vl)), 0x7f0u, (vl));
  vm32_f_t r = VM32_VLUXEI32 (&__vm32_log2_core_tab[0].r, off, (vl));
  vm32_f_t l1 = VM32_VLUXEI32 (&__vm32_log2_core_tab[0].l1, off, (vl));
  vm32_f_t l2 = VM32_VLUXEI32 (&__vm32_log2_core_tab[0].l2, off, (vl));
  /* 2^-k, then m = u * 2^-k (exact) and z = r*m - 1 (exact).  */
  vm32_f_t i2e = VM32_FROM_BITS (VM32_VSUB_VV (
    VM32_VMV_V_X (0x7f000000u, (vl)), VM32_VAND_VX (b, 0x7f800000, (vl)),
    (vl)));
  vm32_f_t m = VM32_VFMUL_VV (u, i2e, (vl));
  vm32_f_t z = VM32_VFMACC_VV (VM32_VFMV_V_F (-1.0f, (vl)), r, m, (vl));
  /* Fold the 1+x residual into the polynomial argument: with 2^-k * r,
     1 + zp = 2^-k * r * (1+x) and zp stays O(|x|), so the c/u tail is
     never added to -- and never cancels against -- the z term.  ez is the
     exact rounding residual of that fold (a two-sum tail).  */
  vm32_f_t q0 = VM32_VFMUL_VV (c, i2e, (vl));		/* 2^-k * c, exact */
  vm32_f_t q1 = VM32_VFMUL_VV (q0, r, (vl));
  vm32_f_t zp = VM32_VFADD_VV (z, q1, (vl));
  /* exact fold residual: z - zp is exact by Sterbenz whenever |q1| <= |z|,
     and the q1 = 0 / zp = z cases make both terms zero.  */
  vm32_f_t ez = VM32_VFADD_VV (VM32_VFSUB_VV (z, zp, (vl)), q1, (vl));
  /* hi = (k + 127) + (l1 - 127) exact: the table stores l1 - 127 so the
     biased exponent converts straight to a float with no integer subtract.
     (w, ee) = zp*log2(e) split exactly.  */
  vm32_f_t hi = VM32_VFADD_VV (
    VM32_VFCVT_F_X_V (VM32_REINT_U_I (VM32_VSRL_VX (b, 23, (vl))), (vl)),
    l1, (vl));
  vm32_f_t w = VM32_VFMUL_VF (zp, VM32_LOGF_L2E32_HI, (vl));
  vm32_f_t ee = VM32_VFMSAC_VF (w, VM32_LOGF_L2E32_HI, zp, (vl));
  /* lo = log2(1+zp) - w, one rounding at the end: the z-polynomial plus
     the log2(e) lo part and the two fold residuals.  */
  vm32_f_t zp2 = VM32_VFMUL_VV (zp, zp, (vl));
  vm32_f_t q = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_LOGF_L2Q1, (vl)),
			       VM32_LOGF_L2Q2, zp, (vl));
  q = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_LOGF_L2Q0, (vl)), zp, q, (vl));
  vm32_f_t lo = VM32_VFMACC_VV (l2, zp2, q, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_L2E32_LO, zp, (vl));
  lo = VM32_VFMACC_VF (lo, VM32_LOGF_L2E32_HI, ez, (vl));
  lo = VM32_VFADD_VV (lo, ee, (vl));
  /* (s, se) = hi + w exactly, then one final rounding of s + (se + lo).  */
  vm32_f_t s = VM32_VFADD_VV (hi, w, (vl));
  vm32_f_t sv = VM32_VFSUB_VV (s, hi, (vl));
  vm32_f_t se = VM32_VFADD_VV (
    VM32_VFSUB_VV (hi, VM32_VFSUB_VV (s, sv, (vl)), (vl)),
    VM32_VFSUB_VV (w, sv, (vl)), (vl));
  vm32_f_t r2 = VM32_VFADD_VV (s, VM32_VFADD_VV (se, lo, (vl)), (vl));
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (log2p1_fixup) (x, r2, mbad, vl, VM32_KFN (log2p1_fix));
  return r2;
}
