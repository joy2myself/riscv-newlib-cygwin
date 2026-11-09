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
   through the per-lane scalar binary64 epilogue.  rsqrt and cbrt keep
   using the binary64 kernels (they were already faster than scalar).  */

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
VM32_W1 (cbrt, cbrt)

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
