/* f32/k_erf.inc -- binary32 erf / erfc kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_erf.inc,
   which pulls k_exp.inc and vecmath_dd.h) active at
   VM_LMUL == 2*VM32_LMUL.

   Shape: a native fast path evaluated in binary64 arithmetic on the
   widened f32mK -> f64m2K group, plus a per-lane fallback to the previous
   widen / binary64 kernel / narrow path for the lanes the fast path does
   not cover.  The fast path is a from-scratch plain-double evaluation:

     |x| < 1      erf(x)  = x * P(x^2)           P of degree 7 (3.7e-11)
     |x| >= 1     erfc(x) = exp(-x^2) * R(t)     R = num/den, t=(2|x|-11)/9
                                                   (4.0e-11 relative)

   exp is a self-contained two-part-constant reduction plus a degree-6
   polynomial (2.5e-9, i.e. 0.04 binary32 ulp) and an integer exponent
   built with a shift, so the whole fast path stays inside this file and
   never calls another kernel.  erf(x) for |x| >= 1 is 1 - erfc(|x|) and
   erfc reflects negative lanes as 2 - erfc(|x|).

   The result is 1 - 2^-31-ish ulp of the correctly rounded binary32 value
   everywhere the fast path applies, so narrowing is the only rounding.
   Per-lane fallback (not per-strip) is required: a lane's result must not
   depend on the strip it happens to land in, or the framework's length
   and alignment sweeps (bitwise identity) would fail.

   The binary64 arithmetic here is bound to the *widened* group
   (2*VM32_LMUL) with file-local macros: the VM_* macros of the including
   translation unit are bound to whatever LMUL it last instantiated
   (vfabi32.c instantiates binary64 at 2/4/8 and then binary32 at 1/2/4),
   so they cannot be reused.  */

/* ------------------------------------------------------------------ */
/* File-local binary64 building blocks, bound to the widened group.    */
/* ------------------------------------------------------------------ */

#undef vm32f_b_t
#undef vm32f_i64_t
#undef VMF_FN
#undef VMF_I64FN
#undef VMF_ADD_VV
#undef VMF_ADD_VF
#undef VMF_SUB_VV
#undef VMF_SUB_VF
#undef VMF_MUL_VV
#undef VMF_MUL_VF
#undef VMF_MACC_VV
#undef VMF_MACC_VF
#undef VMF_NMSAC_VV
#undef VMF_DIV_VV
#undef VMF_RDIV_VF
#undef VMF_MV_F
#undef VMF_MERGE_VFM
#undef VMF_ABS_V
#undef VMF_NEG_V
#undef VMF_SGNJ_VV
#undef VMF_MIN_VF
#undef VMF_MAX_VF
#undef VMF_CVT_X_F_RM
#undef VMF_CVT_F_X
#undef VMF_I64_ADD_VX
#undef VMF_I64_SLL_VX
#undef VMF_I64_TO_F64

#if VM32_LMUL == 1
# define vm32f_b_t vbool32_t
# define vm32f_i64_t vint64m2_t
#elif VM32_LMUL == 2
# define vm32f_b_t vbool16_t
# define vm32f_i64_t vint64m4_t
#elif VM32_LMUL == 4
# define vm32f_b_t vbool8_t
# define vm32f_i64_t vint64m8_t
#else
# error "f32 erf kernels are instantiated at VM32_LMUL 1, 2 or 4"
#endif

#define VMF_FN(op)    VM32_XP3 (__riscv_, op, VM32_XP2 (f64m, VM32_D64M))
#define VMF_I64FN(op) VM32_XP3 (__riscv_, op, VM32_XP2 (i64m, VM32_D64M))

#define VMF_ADD_VV(a, b, vl)      VMF_FN (vfadd_vv_)(a, b, vl)
#define VMF_ADD_VF(a, s, vl)      VMF_FN (vfadd_vf_)(a, s, vl)
#define VMF_SUB_VV(a, b, vl)      VMF_FN (vfsub_vv_)(a, b, vl)
#define VMF_SUB_VF(a, s, vl)      VMF_FN (vfsub_vf_)(a, s, vl)
#define VMF_MUL_VV(a, b, vl)      VMF_FN (vfmul_vv_)(a, b, vl)
#define VMF_MUL_VF(a, s, vl)      VMF_FN (vfmul_vf_)(a, s, vl)
#define VMF_MACC_VV(d, a, b, vl)  VMF_FN (vfmacc_vv_)(d, a, b, vl)
#define VMF_MACC_VF(d, s, b, vl)  VMF_FN (vfmacc_vf_)(d, s, b, vl)
#define VMF_NMSAC_VV(d, a, b, vl) VMF_FN (vfnmsac_vv_)(d, a, b, vl)
#define VMF_DIV_VV(a, b, vl)      VMF_FN (vfdiv_vv_)(a, b, vl)
#define VMF_RDIV_VF(a, s, vl)     VMF_FN (vfrdiv_vf_)(a, s, vl)
#define VMF_MV_F(s, vl)           VMF_FN (vfmv_v_f_)(s, vl)
#define VMF_MERGE_VFM(a, s, m, vl) VMF_FN (vfmerge_vfm_)(a, s, m, vl)
#define VMF_ABS_V(a, vl)          VMF_FN (vfabs_v_)(a, vl)
#define VMF_NEG_V(a, vl)          VMF_FN (vfneg_v_)(a, vl)
#define VMF_SGNJ_VV(a, b, vl)     VMF_FN (vfsgnj_vv_)(a, b, vl)
#define VMF_MIN_VF(a, s, vl)      VMF_FN (vfmin_vf_)(a, s, vl)
#define VMF_MAX_VF(a, s, vl)      VMF_FN (vfmax_vf_)(a, s, vl)
/* f64 -> i64 (round to nearest even) and back; the i64 shift builds 2^k. */
#define VMF_CVT_X_F_RM(a, rm, vl) \
  VM32_XP2 (VMF_I64FN (vfcvt_x_f_v_), _rm)(a, rm, vl)
#define VMF_CVT_F_X(a, vl)        VMF_FN (vfcvt_f_x_v_)(a, vl)
#define VMF_I64_ADD_VX(a, s, vl)  VMF_I64FN (vadd_vx_)(a, s, vl)
#define VMF_I64_SLL_VX(a, s, vl)  VMF_I64FN (vsll_vx_)(a, s, vl)
#define VMF_I64_TO_F64(a)						\
  VM32_XP2 (VM32_XP2 (__riscv_vreinterpret_v_i64m, VM32_D64M),		\
	    VM32_XP2 (_f64m, VM32_D64M))(a)

/* ------------------------------------------------------------------ */
/* Coefficients.                                                       */
/*   data guard: the file is instantiated once per binary32 LMUL.      */
/* ------------------------------------------------------------------ */

#ifndef VM32_ERF_DATA
#define VM32_ERF_DATA

/* P(u) = erf(sqrt(u))/sqrt(u) on u in [0,1], power basis low-to-high,
   Chebyshev fit: max relative error 3.7e-11.  */
static const double vm32f_erf_p[8] =
  {
    1.12837916706154351e+00, -3.76126384680287085e-01, 1.12837824952368451e-01,
    -2.68654307276702427e-02, 5.22103180785802277e-03, -8.48408040553510676e-04,
    1.12659590256066090e-04, -9.66704457148699645e-06
  };

/* R(t) = erfc(x) * exp(x^2), x = (9t+11)/2 in [1,10], t in [-1,1]:
   num/den power basis low-to-high (den[0] = 1); max rel error 4.0e-11.  */
static const double vm32f_erfc_num[6] =
  {
    1.00962218399791359e-01, 2.48400416824180764e-01, 2.32969107812059512e-01,
    9.85698876418816400e-02, 1.58615777391423567e-02, -1.31090261630937407e-08
  };
static const double vm32f_erfc_den[6] =
  {
    1.00000000000000000e+00, 3.25346315839729217e+00, 4.26795153419263418e+00,
    2.82220041515785569e+00, 9.40825381400421157e-01, 1.26512030034250322e-01
  };

/* exp(r) on r in [-ln2/2, ln2/2], power basis low-to-high, degree 6:
   max relative error 2.5e-9 (0.04 binary32 ulp).  */
static const double vm32f_exp_p[7] =
  {
    1.00000000000000000e+00, 1.00000003771621393e+00, 5.00000004711775747e-01,
    1.66664155146532766e-01, 4.16663528967751579e-02, 8.37512639815333505e-03,
    1.39411084339726744e-03
  };

#endif /* VM32_ERF_DATA */

/* ------------------------------------------------------------------ */
/* exp(-xx) for xx in [1, 110].  Plain binary64:                      */
/*   t = -xx*log2e ; k = rint(t) ; r = -xx - k*ln2 (two part)         */
/*   exp(-xx) = exp(r) * 2^k, the scale built from (k+1023) << 52.    */
/* ------------------------------------------------------------------ */

#define VM32F_ERF_LOG2E  1.4426950408889634074e+00
#define VM32F_ERF_LN2HI  6.93147180369123816490e-01
#define VM32F_ERF_LN2LO  1.90821492927058770002e-10

static inline VM32_WD_T
VM32_KFN (erf_expneg) (VM32_WD_T xx, size_t vl)
{
  VM32_WD_T nx = VMF_NEG_V (xx, vl);
  VM32_WD_T t = VMF_MUL_VF (nx, VM32F_ERF_LOG2E, vl);
  vm32f_i64_t ki = VMF_I64FN (vfcvt_x_f_v_)(t, vl);
  VM32_WD_T kf = VMF_CVT_F_X (ki, vl);
  VM32_WD_T r = VMF_MACC_VF (nx, -VM32F_ERF_LN2HI, kf, vl);
  r = VMF_MACC_VF (r, -VM32F_ERF_LN2LO, kf, vl);
  VM32_WD_T p = VMF_MV_F (vm32f_exp_p[6], vl);
  for (int i = 5; i >= 0; i--)
    {
      p = VMF_MUL_VV (p, r, vl);
      p = VMF_ADD_VF (p, vm32f_exp_p[i], vl);
    }
  vm32f_i64_t sc = VMF_I64_ADD_VX (ki, 1023, vl);
  sc = VMF_I64_SLL_VX (sc, 52, vl);
  return VMF_MUL_VV (p, VMF_I64_TO_F64 (sc), vl);
}

/* ------------------------------------------------------------------ */
/* Per-lane scalar fallback for erfc(|x| > 10): the lane is evaluated by
   the scalar double libm and narrowed, i.e. exactly what the old wide
   path produced for it, so value and gated flags are inherited.  Per
   lane, not per strip: a lane's result cannot depend on its neighbours
   or the framework's length/alignment sweeps (bitwise identity) fail.  */
/* ------------------------------------------------------------------ */

static inline float
VM32_KFN (erfc_fix) (float x)
{
  return (float) erfc ((double) x);
}

/* Merge fn(x) into r for the lanes selected by m.  Buffer bound:
   vl <= VLMAX(e32, LMUL <= 4) = VLEN/8 lanes, 600 covers VLEN to 4800.
   noinline: the scalar loop is larger than everything else in the kernel
   put together, and leaving it inlined made GCC outline the whole erfc
   entry -- which then saved/restored 24 callee-saved vector registers on
   every strip (measured: 25.4 -> 19.4 ns/element on muse-pi).  */
__attribute__ ((noinline)) static vm32_f_t
VM32_KFN (erfc_fixup) (vm32_f_t x, vm32_f_t r, vm32_b_t m, size_t vl)
{
  float vfef_x_[600], vfef_r_[600];
  size_t vfef_n_ = (size_t) VM32_VCMP_POPC_M (m, vl);
  VM32_VSE (vfef_x_, VM32_VCOMPRESS (x, m, vl), vfef_n_);
  for (size_t vfef_i_ = 0; vfef_i_ < vfef_n_; vfef_i_++)
    vfef_r_[vfef_i_] = VM32_KFN (erfc_fix) (vfef_x_[vfef_i_]);
  vm32_u_t vfef_idx_ = VM32_VIOTA_M (m, vl);
  vm32_f_t vfef_pick_ = VM32_VLUXEI32 (
    (const float *) vfef_r_, VM32_VSLL_VX (vfef_idx_, 2, vl), vl);
  return VM32_VFMERGE_VVM (r, vfef_pick_, m, vl);
}

#define VM32F_ERFC_FIXUP1(r, x, m, vl) do {				\
  if (VM32_VFIRST_M ((m), (vl)) >= 0)					\
    (r) = VM32_KFN (erfc_fixup) ((x), (r), (m), (vl));			\
} while (0)

/* ------------------------------------------------------------------ */
/* erf / erfc.                                                         */
/* ------------------------------------------------------------------ */

/* |x| > 10: erfc falls back to the scalar evaluation per lane.  */
#define VM32F_ERFC_BAD(x, vl)						\
  VM32_VMSGTU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, (vl)),	\
		  0x41200000u, (vl))
/* NaN: erff is +-1 for every other argument, including +-Inf, so NaN is
   the only lane that needs special handling.  */
#define VM32F_ERF_BAD(x, vl)						\
  VM32_VMSGTU_VX (VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, (vl)),	\
		  0x7f800000u, (vl))

/* R(t) Horner, t in [-1,1].  */
#define VM32F_ERFC_RAT(yo, t, vl) do {					\
  VM32_WD_T vmre_n_ = VMF_MV_F (vm32f_erfc_num[5], (vl));		\
  VM32_WD_T vmre_d_ = VMF_MV_F (vm32f_erfc_den[5], (vl));		\
  for (int vmre_i_ = 4; vmre_i_ >= 0; vmre_i_--)				\
    {									\
      vmre_n_ = VMF_MUL_VV (vmre_n_, (t), (vl));			\
      vmre_n_ = VMF_ADD_VF (vmre_n_, vm32f_erfc_num[vmre_i_], (vl));	\
      vmre_d_ = VMF_MUL_VV (vmre_d_, (t), (vl));			\
      vmre_d_ = VMF_ADD_VF (vmre_d_, vm32f_erfc_den[vmre_i_], (vl));	\
    }									\
  (yo) = VMF_DIV_VV (vmre_n_, vmre_d_, (vl));				\
} while (0)

/* erfc(ac) = exp(-ac^2) * R(t), ac clamped into [1,10] by the caller.  */
#define VM32F_ERFC_EXP(e, acd, vl) do {					\
  VM32_WD_T vmre_xx_ = VMF_MUL_VV ((acd), (acd), (vl));			\
  VM32_WD_T vmre_t_ = VMF_MACC_VF (VMF_MV_F (-11.0 / 9.0, (vl)),	\
				   2.0 / 9.0, (acd), (vl));		\
  VM32_WD_T vmre_r_;							\
  VM32F_ERFC_RAT (vmre_r_, vmre_t_, (vl));				\
  (e) = VMF_MUL_VV (VM32_KFN (erf_expneg) (vmre_xx_, (vl)),		\
		    vmre_r_, (vl));					\
} while (0)

/* Common small-branch polynomial: P(min(x*x,1)).  */
#define VM32F_ERF_POLY(p, xd, vl) do {					\
  VM32_WD_T vmre_u_ = VMF_MIN_VF (VMF_MUL_VV ((xd), (xd), (vl)),	\
				  1.0, (vl));				\
  (p) = VMF_MV_F (vm32f_erf_p[7], (vl));				\
  for (int vmre_i_ = 6; vmre_i_ >= 0; vmre_i_--)				\
    {									\
      (p) = VMF_MUL_VV ((p), vmre_u_, (vl));				\
      (p) = VMF_ADD_VF ((p), vm32f_erf_p[vmre_i_], (vl));		\
    }									\
} while (0)

static inline vm32_f_t
VM32_KFN (erf) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32F_ERF_BAD (x, vl);
  vm32_f_t axf = VM32_VFABS_V (x, vl);
  vm32_b_t msmall = VM32_VMSLTU_VX (
    VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, vl), 0x3f800000u, vl);
  /* tail argument in [1,10] for every lane; clamp keeps it flag-free */
  vm32_f_t acf = VM32_VFMIN_VF (VM32_VFMAX_VF (axf, 1.0f, vl), 10.0f, vl);

  VM32_WD_T xd = VM32_WIDEN (x, vl);
  VM32_WD_T acd = VM32_WIDEN (acf, vl);
  VM32_WD_T p;
  VM32F_ERF_POLY (p, xd, vl);
  VM32_WD_T small = VMF_MUL_VV (xd, p, vl);
  VM32_WD_T e;
  VM32F_ERFC_EXP (e, acd, vl);
  /* erf(|x|) = 1 - erfc(|x|), then the sign of x (erf is odd).  */
  VM32_WD_T tail = VMF_SUB_VV (VMF_MV_F (1.0, vl), e, vl);

  vm32_f_t smallf = VM32_NARROW (small, vl);
  vm32_f_t tailf = VM32_NARROW (tail, vl);
  vm32_f_t tails = VM32_VFSGNJ_VV (tailf, x, vl);
  vm32_f_t r = VM32_VFMERGE_VVM (tails, smallf, msmall, vl);

  /* NaN lanes: propagate through the *masked* add so a signalling NaN
     quiets and raises INVALID (as the reference wants) while the finished
     lanes stay untouched -- an unmasked x+x would overflow on finite max
     and raise a bogus OVERFLOW.  */
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t nanq = VM32_VFADD_VV_MU (mbad, x, x, x, vl);
      r = VM32_VFMERGE_VVM (r, nanq, mbad, vl);
    }
  return r;
}

static inline vm32_f_t
VM32_KFN (erfc) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32F_ERFC_BAD (x, vl);
  vm32_f_t axf = VM32_VFABS_V (x, vl);
  vm32_b_t msmall = VM32_VMSLTU_VX (
    VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, vl), 0x3f800000u, vl);
  vm32_b_t mneg = VM32_VMSGEU_VX (VM32_BITS (x), 0x80000000u, vl);
  vm32_f_t acf = VM32_VFMIN_VF (VM32_VFMAX_VF (axf, 1.0f, vl), 10.0f, vl);

  VM32_WD_T xd = VM32_WIDEN (x, vl);
  VM32_WD_T acd = VM32_WIDEN (acf, vl);
  VM32_WD_T p;
  VM32F_ERF_POLY (p, xd, vl);
  VM32_WD_T small = VMF_NMSAC_VV (VMF_MV_F (1.0, vl), xd, p, vl);

  VM32_WD_T e;
  VM32F_ERFC_EXP (e, acd, vl);
  /* reflect: erfc(x) = x < 0 ? 2 - E : E, carried as E + (2-2E)*q */
  VM32_WD_T q = VMF_MERGE_VFM (VMF_MV_F (0.0, vl), 1.0, mneg, vl);
  VM32_WD_T twoE = VMF_ADD_VV (e, e, vl);
  VM32_WD_T t2 = VMF_SUB_VV (VMF_MV_F (2.0, vl), twoE, vl);
  VM32_WD_T tail = VMF_MACC_VV (e, t2, q, vl);
  /* The lanes the wide path takes must not raise gated flags here: erfc(+Inf)
     is exactly 0 (no UNDERFLOW) while the clamped fast path would hand back
     a subnormal.  Scale those lanes by 2^100 before narrowing (1.0 for the
     fast lanes, an exact multiply), the wide path then sets the flags.  */
  VM32_WD_T ts = VMF_MERGE_VFM (VMF_MV_F (1.0, vl), 0x1p100, mbad, vl);
  tail = VMF_MUL_VV (tail, ts, vl);

  vm32_f_t smallf = VM32_NARROW (small, vl);
  vm32_f_t tailf = VM32_NARROW (tail, vl);
  vm32_f_t r = VM32_VFMERGE_VVM (tailf, smallf, msmall, vl);

  VM32F_ERFC_FIXUP1 (r, x, mbad, vl);
  return r;
}
