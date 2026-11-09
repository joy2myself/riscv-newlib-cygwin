/* f32/k_exp.inc -- binary32 exp / exp2 / exp10 / expm1 kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_exp.inc,
   k_expm1.inc) active at VM_LMUL == 2*VM32_LMUL.

   exp and expm1 are native binary32 kernels: range reduction to
   |r| <= ln2/2 with a 15-bit two-part ln2 (k*hi exact, the fma remainder
   carried as a double-float tail), an f32 minimax polynomial, and a
   double-float assembly so the final add rounds once.  The covered domain
   is 2^-30 <= |x| <= 80 (well inside exp's normal-result range); every
   other lane goes through the per-lane scalar binary64 epilogue, which is
   the value and flag behaviour of the previous widen / binary64 kernel /
   narrow path.

   exp2 and exp10 keep using the binary64 kernels (they were already
   faster than scalar libm; no need to move them).  */

/* ------------------------------------------------------------------ */
/* f32 double-float helpers (file-local names).                       */
/* ------------------------------------------------------------------ */

#undef VM32_EXPF_TWO_SUM
#undef VM32_EXPF_FAST2_SUM
#undef VM32_EXPF_MINUS1_DD
#undef VM32_EXPF_EXP_DD
#undef VM32_EXPF_EXPM1_DD

/* (s, e) = a + b exactly (Knuth two-sum).  */
#define VM32_EXPF_TWO_SUM(s, e, a, b, vl) do {				\
  vm32_f_t ets_s_ = VM32_VFADD_VV ((a), (b), (vl));			\
  vm32_f_t ets_v_ = VM32_VFSUB_VV (ets_s_, (a), (vl));			\
  (s) = ets_s_;								\
  (e) = VM32_VFADD_VV (							\
    VM32_VFSUB_VV ((a), VM32_VFSUB_VV (ets_s_, ets_v_, (vl)), (vl)),	\
    VM32_VFSUB_VV ((b), ets_v_, (vl)), (vl));				\
} while (0)

/* (s, e) = a + b exactly, requires |a| >= |b|.  */
#define VM32_EXPF_FAST2_SUM(s, e, a, b, vl) do {			\
  (s) = VM32_VFADD_VV ((a), (b), (vl));					\
  (e) = VM32_VFSUB_VV ((b), VM32_VFSUB_VV ((s), (a), (vl)), (vl));	\
} while (0)

/* (h, l) = (eh + el) - 1, at full double-float precision.  */
#define VM32_EXPF_MINUS1_DD(h, l, eh, el, vl) do {			\
  vm32_f_t e1_s_, e1_e_;						\
  VM32_EXPF_TWO_SUM (e1_s_, e1_e_, (eh), VM32_VFMV_V_F (-1.0f, (vl)),	\
		     (vl));						\
  (h) = e1_s_;								\
  (l) = VM32_VFADD_VV (e1_e_, (el), (vl));				\
} while (0)

#define VM32_EXPF_LN2_HI	0.693145751953125f	/* 0x3f317200 */
#define VM32_EXPF_LN2_LO	1.428606765330187e-06f	/* 0x35bfbe8e */
#define VM32_EXPF_INVLN2	1.4426950408889634f
#define VM32_EXPF_SHIFT		12582912.0f		/* 1.5*2^23 */
#define VM32_EXPF_SHIFTBITS	0x4b400000u

/* exp(x) = (h + l) as a double-float pair, for finite x with
   2^-30 <= |x| <= 80 (salient: the reduction rounds to an integer k with
   |r| <= 0.3471, and 2^k is a normal binary32 so building it from the
   exponent bits cannot raise a flag).  */
#define VM32_EXPF_EXP_DD(h, l, x, vl) do {				\
  vm32_f_t edd_z_ = VM32_VFMUL_VF ((x), VM32_EXPF_INVLN2, (vl));	\
  vm32_f_t edd_kds_ = VM32_VFADD_VF (edd_z_, VM32_EXPF_SHIFT, (vl));	\
  vm32_u_t edd_ki_ = VM32_BITS (edd_kds_);				\
  vm32_f_t edd_kr_ = VM32_VFSUB_VF (edd_kds_, VM32_EXPF_SHIFT, (vl));	\
  /* scale = 2^k from the exponent field: (k + 127) << 23 (mod 2^32).  */\
  vm32_u_t edd_kb_ = VM32_VSLL_VX (					\
    VM32_VADD_VX (VM32_VSUB_VX (edd_ki_, VM32_EXPF_SHIFTBITS, (vl)),	\
		  127, (vl)), 23, (vl));				\
  vm32_f_t edd_scale_ = VM32_FROM_BITS (edd_kb_);			\
  /* r = x - k*ln2 in two pieces; r1 is exact, r2 captures the low part\
     and rl the residual of the final fma, so r = r2 + rl.  */		\
  vm32_f_t edd_r1_ = VM32_VFNMSAC_VF ((x), VM32_EXPF_LN2_HI, edd_kr_,	\
				      (vl));				\
  vm32_f_t edd_r2_ = VM32_VFNMSAC_VF (edd_r1_, VM32_EXPF_LN2_LO,	\
				      edd_kr_, (vl));			\
  vm32_f_t edd_d_ = VM32_VFSUB_VV (edd_r1_, edd_r2_, (vl));		\
  vm32_f_t edd_rl_ = VM32_VFNMSAC_VF (edd_d_, VM32_EXPF_LN2_LO,	\
				      edd_kr_, (vl));			\
  vm32_f_t edd_r_ = edd_r2_;						\
  /* e^r = 1 + r + r^2*Q(r), Q a degree-5 minimax fit on [-0.347,0.347]. */\
  vm32_f_t edd_Q_ = VM32_VFMUL_VF (edd_r_,				\
				   0.00019891103420544327f, (vl));	\
  edd_Q_ = VM32_VFADD_VF (edd_Q_, 0.0013933751387987224f, (vl));	\
  edd_Q_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.0083333108239232109f,	\
					 (vl)), edd_r_, edd_Q_, (vl));	\
  edd_Q_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.041666464010785553f,	\
					 (vl)), edd_r_, edd_Q_, (vl));	\
  edd_Q_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.16666666681725001f,	\
					 (vl)), edd_r_, edd_Q_, (vl));	\
  edd_Q_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.5f, (vl)), edd_r_,	\
			   edd_Q_, (vl));				\
  vm32_f_t edd_u_ = VM32_VFMUL_VV (edd_r_, edd_r_, (vl));		\
  vm32_f_t edd_ue_ = VM32_VFMSAC_VV (edd_u_, edd_r_, edd_r_, (vl));	\
  vm32_f_t edd_rest_ = VM32_VFMUL_VV (edd_u_, edd_Q_, (vl));		\
  vm32_f_t edd_rest_e_ = VM32_VFMSAC_VV (edd_rest_, edd_u_, edd_Q_,	\
					 (vl));				\
  edd_rest_e_ = VM32_VFMACC_VV (edd_rest_e_, edd_Q_, edd_ue_, (vl));	\
  /* p = 1 + r + rest as a pair; 1+r is exact via Fast2Sum.  */		\
  vm32_f_t edd_p1_, edd_p1e_;						\
  VM32_EXPF_FAST2_SUM (edd_p1_, edd_p1e_,				\
		       VM32_VFMV_V_F (1.0f, (vl)), edd_r_, (vl));	\
  vm32_f_t edd_p2_, edd_p2e_;						\
  VM32_EXPF_FAST2_SUM (edd_p2_, edd_p2e_, edd_p1_, edd_rest_, (vl));	\
  vm32_f_t edd_pl_ = VM32_VFADD_VV (edd_p2e_, edd_p1e_, (vl));		\
  edd_pl_ = VM32_VFADD_VV (edd_pl_, edd_rest_e_, (vl));			\
  edd_pl_ = VM32_VFMACC_VV (edd_pl_, edd_rl_, edd_p2_, (vl));		\
  /* exp = scale*p; the product's rounding rides in the low word.  */	\
  vm32_f_t edd_uh_ = VM32_VFMUL_VV (edd_scale_, edd_p2_, (vl));		\
  vm32_f_t edd_ue2_ = VM32_VFMSAC_VV (edd_uh_, edd_scale_, edd_p2_,	\
				      (vl));				\
  (h) = edd_uh_;							\
  (l) = VM32_VFADD_VV (edd_ue2_,					\
		       VM32_VFMUL_VV (edd_scale_, edd_pl_, (vl)),	\
		       (vl));						\
} while (0)

/* (h, l) = expm1(x) for finite x with 2^-30 <= |x| <= 80.  |x| <= 1 uses
   a degree-8 minimax fit of (expm1(x)-x)/x^2, compensated so the final
   x + x^2*R(x) is a double-float; larger |x| uses exp(x) - 1.  */
#define VM32_EXPF_EXPM1_DD(h, l, x, vl) do {				\
  vm32_f_t emd_ax_ = VM32_VFABS_V ((x), (vl));				\
  vm32_b_t emd_sm_ = VM32_VMFLE_VF (emd_ax_, 0.5f, (vl));		\
  vm32_f_t emd_x_ = (x);						\
  vm32_f_t emd_R_ = VM32_VFMUL_VF (emd_x_, 2.7682856169452262e-06f,	\
				   (vl));				\
  emd_R_ = VM32_VFADD_VF (emd_R_, 2.4939733300130981e-05f, (vl));	\
  emd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.00019841073556216572f,	\
					 (vl)), emd_x_, emd_R_, (vl));	\
  emd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.0013888672862062655f,	\
					 (vl)), emd_x_, emd_R_, (vl));	\
  emd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.008333333431502386f,	\
					 (vl)), emd_x_, emd_R_, (vl));	\
  emd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.041666667747145766f,	\
					 (vl)), emd_x_, emd_R_, (vl));	\
  emd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.16666666666589969f,	\
					 (vl)), emd_x_, emd_R_, (vl));	\
  emd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (0.49999999999155825f,	\
					 (vl)), emd_x_, emd_R_, (vl));	\
  vm32_f_t emd_u_ = VM32_VFMUL_VV (emd_x_, emd_x_, (vl));		\
  vm32_f_t emd_ue_ = VM32_VFMSAC_VV (emd_u_, emd_x_, emd_x_, (vl));	\
  vm32_f_t emd_rest_ = VM32_VFMUL_VV (emd_u_, emd_R_, (vl));		\
  vm32_f_t emd_rest_e_ = VM32_VFMSAC_VV (emd_rest_, emd_u_, emd_R_,	\
					 (vl));				\
  emd_rest_e_ = VM32_VFMACC_VV (emd_rest_e_, emd_R_, emd_ue_, (vl));	\
  vm32_f_t emd_s1_, emd_s1e_;						\
  VM32_EXPF_FAST2_SUM (emd_s1_, emd_s1e_, emd_x_, emd_rest_, (vl));	\
  vm32_f_t emd_sl_ = VM32_VFADD_VV (emd_s1e_, emd_rest_e_, (vl));	\
  /* Large |x|: exp(x) - 1, same double-float treatment.  */		\
  vm32_f_t emd_bh_, emd_bl_, emd_bh1_, emd_bl1_;			\
  VM32_EXPF_EXP_DD (emd_bh_, emd_bl_, emd_x_, (vl));			\
  VM32_EXPF_MINUS1_DD (emd_bh1_, emd_bl1_, emd_bh_, emd_bl_, (vl));	\
  (h) = VM32_VFMERGE_VVM (emd_bh1_, emd_s1_, emd_sm_, (vl));		\
  (l) = VM32_VFMERGE_VVM (emd_bl1_, emd_sl_, emd_sm_, (vl));		\
} while (0)

/* ---- per-lane scalar epilogue -------------------------------------- */

#undef VM32_EXPF_DEF_FIXUP
#define VM32_EXPF_DEF_FIXUP(name)					\
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

VM32_EXPF_DEF_FIXUP (exp_fixup)
VM32_EXPF_DEF_FIXUP (expm1_fixup)

#undef VM32_EXPF_RAISE_UF_SUB
#define VM32_EXPF_RAISE_UF_SUB(yo, vl) do {				\
  vm32_u_t vuf_b_ = VM32_VAND_VX (VM32_BITS (yo), 0x7fffffffu, (vl));	\
  vm32_b_t vuf_m_ = VM32_VMSLTU_VX (					\
    VM32_VSUB_VX (vuf_b_, 1, (vl)), 0x007fffffu, (vl));			\
  vm32_f_t vuf_z_ = VM32_VFMUL_VF_MU (vuf_m_,				\
				      VM32_VFMV_V_F (0.0f, (vl)),	\
				      VM32_VFMV_V_F (0x1p-126f, (vl)),	\
				      0x1p-126f, (vl));			\
  (yo) = VM32_VFADD_VV_MU (vuf_m_, (yo), (yo), vuf_z_, (vl));		\
} while (0)

/* ---- entries ------------------------------------------------------- */

/* exp / expm1 use the binary64 kernels (kept for the entries that already
   win).  */
VM32_W1F (exp2, exp2)
VM32_W1F (exp10, exp10)

/* Covered domain: 2^-30 <= |x| <= 80.  Outside it the lane is evaluated
   by the scalar binary64 epilogue.  */
#define VM32_EXPF_BAD(x, vl)						\
  VM32_VMOR_MM (							\
    VM32_VMSGEU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x42a00000u, (vl)),					\
    VM32_VMSLTU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffff, (vl)),	\
		    0x30800000u, (vl)), (vl))

static inline float
VM32_KFN (exp_fix) (float x)
{
  return (float) exp ((double) x);
}

static inline float
VM32_KFN (expm1_fix) (float x)
{
  return (float) expm1 ((double) x);
}

static inline vm32_f_t
VM32_KFN (exp) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  (void) fixup;
  vm32_b_t mbad = VM32_EXPF_BAD (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (1.0f, (vl)), mbad,
				  vl);
  vm32_f_t h, l;
  VM32_EXPF_EXP_DD (h, l, xs, vl);
  vm32_f_t r = VM32_VFADD_VV (h, l, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    return VM32_KFN (exp_fixup) (x, r, mbad, vl, VM32_KFN (exp_fix));
  return r;
}

static inline vm32_f_t
VM32_KFN (expm1) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32_EXPF_BAD (x, vl);
  vm32_f_t xs = VM32_VFMERGE_VVM (x, VM32_VFMV_V_F (0.5f, (vl)), mbad,
				  vl);
  vm32_f_t h, l;
  VM32_EXPF_EXPM1_DD (h, l, xs, vl);
  vm32_f_t r = VM32_VFADD_VV (h, l, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    r = VM32_KFN (expm1_fixup) (x, r, mbad, vl, VM32_KFN (expm1_fix));
  VM32_EXPF_RAISE_UF_SUB (r, vl);
  return r;
}
