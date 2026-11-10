/* f32/k_trig.inc -- native binary32 trigonometric kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_trig.inc)
   active at VM_LMUL == 2*VM32_LMUL.

   Shape: a native binary32 fast path that evaluates the reduced
   argument as a *double-float* (hi,lo) pair and returns hi+lo, so the
   single final binary32 add rounds the correctly rounded result.  Every
   entry keeps a per-lane fallback to the known-good widen / binary64
   kernel / narrow path, merged in lane by lane, which carries the
   special-value and FP-flag semantics (and keeps the result independent
   of the strip length, as the framework's length sweep requires).

   Reduction:
     * sinf/cosf/tanf/sincosf: q = rint(|x| * 2/pi) and a 2-part
       Cody-Waite split whose first part is 17-bit pio2_1, so
       z = |x| - q*pio2_1 is exact and u = z - q*pio2_1t carries the
       residual e = (z-u) - q*pio2_1t.  (u,e) represents the reduced
       argument to ~2^-45; the |u| >= 2^-12 gate covers the rest.
     * tanpif/sincospif: exact q = rint(2x), r = x - q/2 (exact in
       binary32), then p = pi*r as a double-float.  The same kernels
       evaluate sin/cos of (p0,p1); |p| <= pi/4.
     * sinpif/cospif: exact q = rint(x) resp. rint(x+1/2), r = x - q
       resp. x - (q-1/2), then p = pi*r as a double-float with
       |p| <= pi/2, so one wide sin kernel plus the quadrant sign is
       all the fast path needs (VM32_KSIN_HALF).  The wider argument
       costs one more coefficient and a compensated last Horner step,
       but saves the whole second polynomial and the two-output
       quadrant select.

   The kernels (fdlibm float coefficients from newlib math/kf_sin.c and
   kf_cos.c) are evaluated in f32 double-float arithmetic:
       sin(u+e) = u + (e + v*R(zh))       v = u^3,  R = S1 + zh*(...)
       cos(u+e) = (1 - zh/2) - zl/2 + (zh^2*P(zh) - u*e)
   with the rounding of zh = u^2 and of every product carried in a low
   word via fma residuals, so the assembled (hi,lo) pair is good to
   ~2^-45 relative -- the caller's hi+lo then rounds once.  tan is the
   quotient of the two kernels with an fma-compensated division.

   Costs are ~20 vector instructions per kernel; the reduction ~11.  All
   constants are the fdlibm single-precision values.  */

/* ---- fdlibm poly constants (kf_sin.c) ---- */
#define VM32_S1 -1.6666667163e-01f
#define VM32_S2  8.3333337680e-03f
#define VM32_S3 -1.9841270114e-04f
#define VM32_S4  2.7557314297e-06f
#define VM32_S5 -2.5050759689e-08f
#define VM32_S6  1.5896910177e-10f

/* ---- fdlibm poly constants (kf_cos.c) ---- */
#define VM32_C1  4.1666667908e-02f
#define VM32_C2 -1.3888889225e-03f
#define VM32_C3  2.4801587642e-05f
#define VM32_C4 -2.7557314297e-07f
#define VM32_C5  2.0875723372e-09f
#define VM32_C6 -1.1359647598e-11f

/* ---- full-range sin poly, |u| <= pi/2 ----
   sin(u) = u + u^3*R(u^2),  R(z) = Q1 + z*(Q2 + z*(... + z*Q6)).
   R is the minimax fit of (sin(u)/u - 1)/z over z in [0,(pi/2)^2]
   weighted by the relative error of sin; the binary32 values below come
   from a local search over the last bits of the rounded coefficients,
   which absorbs the coefficient quantisation (the fitted poly is within
   0.07 ulp relative over the whole range, against ~1.2 ulp for the
   plain Taylor series at this width).  One poly then covers both the
   sin and the cos branch of a quadrant (cospif, see VM32_KSIN_HALF).  */
#define VM32_Q1 -1.6666667163e-01f	/* 0xbe2aaaab */
#define VM32_Q2  8.3333346993e-03f	/* 0x3c08888a */
#define VM32_Q3 -1.9841230824e-04f	/* 0xb9500ce6 */
#define VM32_Q4  2.7557389330e-06f	/* 0x3638ef3c */
#define VM32_Q5 -2.5052107944e-08f	/* 0xb2d7322b */
#define VM32_Q6  1.6059044372e-10f	/* 0x2f309231 */

/* ---- reduction constants ---- */
#define VM32_INVPIO2 6.366197466850281e-01f	/* 0x3f22f983 */
#define VM32_PIO2_HI 1.57078552246093750e+00f	/* 0x3fc90f80 (17 bits) */
#define VM32_PIO2_LO 1.0804334124259185e-05f	/* 0x37354443 */
#define VM32_PI_HI   3.14159274101257324e+00f	/* 0x40490fdb */
#define VM32_PI_LO  -8.7422776573475858e-08f	/* 0xb3bbbd2e */

/* ---- bit thresholds ---- */
#define VM32_TINY_BITS 0x39800000u	/* 2^-12 */
#define VM32_TRIG_MAX  0x41800000u	/* 16.0 */
#define VM32_PI_MAX    0x44800000u	/* 1024.0 */

/* |x| outside [2^-12, hi] -- integer-only; catches Inf/NaN/subnormal too. */
#define VM32_RANGE_BAD(xa, hi, vl)					\
  VM32_VMOR_MM (VM32_VMSLTU_VX ((xa), VM32_TINY_BITS, (vl)),		\
		VM32_VMSGTU_VX ((xa), (hi), (vl)), (vl))

/* ==================================================================== */
/* double-float kernels                                                 */
/* ==================================================================== */

/* (sh,sl) = sin(u+e), |u| <= pi/4, e the reduced-argument tail.
   zh,zl = u^2 exactly split; v = u^3; R the odd Taylor coefficients;
   C = v*R with its product residual and the v residual folded in; the
   assembly is  (u + (e + C))  with exact two-sum/fast-two-sum so the
   unanswered correction is one f32 add at the caller.  */
#define VM32_KSIN_DD(sho, slo, u, e, vl) do {				\
  vm32_f_t ksd_zh_ = VM32_VFMUL_VV ((u), (u), (vl));			\
  vm32_f_t ksd_zl_ = VM32_VFMSAC_VV (ksd_zh_, (u), (u), (vl));		\
  vm32_f_t ksd_v_ = VM32_VFMUL_VV (ksd_zh_, (u), (vl));			\
  vm32_f_t ksd_ve_ = VM32_VFMSAC_VV (ksd_v_, ksd_zh_, (u), (vl));	\
  ksd_ve_ = VM32_VFMACC_VV (ksd_ve_, ksd_zl_, (u), (vl));		\
  vm32_f_t ksd_R_ = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_S5, (vl)),	\
				    VM32_S6, ksd_zh_, (vl));		\
  ksd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_S4, (vl)), ksd_zh_, ksd_R_, (vl)); \
  ksd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_S3, (vl)), ksd_zh_, ksd_R_, (vl)); \
  ksd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_S2, (vl)), ksd_zh_, ksd_R_, (vl)); \
  ksd_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_S1, (vl)), ksd_zh_, ksd_R_, (vl)); \
  vm32_f_t ksd_C_ = VM32_VFMUL_VV (ksd_v_, ksd_R_, (vl));		\
  vm32_f_t ksd_Ce_ = VM32_VFMSAC_VV (ksd_C_, ksd_v_, ksd_R_, (vl));	\
  ksd_Ce_ = VM32_VFMACC_VV (ksd_Ce_, ksd_ve_, ksd_R_, (vl));		\
  /* the tail enters to first order as e*cos(u) ~ e*(1 - zh/2).  */	\
  vm32_f_t ksd_ee_ = VM32_VFNMSAC_VF ((e), 0.5f,				\
				     VM32_VFMUL_VV ((e), ksd_zh_, (vl)), (vl)); \
  /* s1 + s1e = ee + C exactly (general two-sum).  */			\
  vm32_f_t ksd_s1_ = VM32_VFADD_VV (ksd_ee_, ksd_C_, (vl));		\
  vm32_f_t ksd_v1_ = VM32_VFSUB_VV (ksd_s1_, ksd_ee_, (vl));		\
  vm32_f_t ksd_s1e_ = VM32_VFADD_VV (					\
    VM32_VFSUB_VV (ksd_ee_, VM32_VFSUB_VV (ksd_s1_, ksd_v1_, (vl)), (vl)), \
    VM32_VFSUB_VV (ksd_C_, ksd_v1_, (vl)), (vl));			\
  /* s + se = u + s1 exactly (|u| >= |s1|).  */				\
  vm32_f_t ksd_s_ = VM32_VFADD_VV ((u), ksd_s1_, (vl));			\
  vm32_f_t ksd_se_ = VM32_VFADD_VV (					\
    VM32_VFSUB_VV ((u), ksd_s_, (vl)), ksd_s1_, (vl));			\
  (sho) = ksd_s_;							\
  (slo) = VM32_VFADD_VV (VM32_VFADD_VV (ksd_se_, ksd_s1e_, (vl)),	\
			 ksd_Ce_, (vl));				\
} while (0)

/* (ch,cl) = cos(u+e), |u| <= pi/4.
   (1 - zh/2) and its exact residual pe come from a fast-two-sum with 1;
   the low z and the z^2*P and -u*e terms ride in the correction.  */
#define VM32_KCOS_DD(cho, clo, u, e, vl) do {				\
  vm32_f_t kcd_zh_ = VM32_VFMUL_VV ((u), (u), (vl));			\
  vm32_f_t kcd_zl_ = VM32_VFMSAC_VV (kcd_zh_, (u), (u), (vl));		\
  vm32_f_t kcd_P_ = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_C5, (vl)),	\
				    VM32_C6, kcd_zh_, (vl));		\
  kcd_P_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_C4, (vl)), kcd_zh_, kcd_P_, (vl)); \
  kcd_P_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_C3, (vl)), kcd_zh_, kcd_P_, (vl)); \
  kcd_P_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_C2, (vl)), kcd_zh_, kcd_P_, (vl)); \
  kcd_P_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_C1, (vl)), kcd_zh_, kcd_P_, (vl)); \
  vm32_f_t kcd_t_ = VM32_VFNMSAC_VV (					\
    VM32_VFMUL_VV (kcd_zh_, VM32_VFMUL_VV (kcd_zh_, kcd_P_, (vl)), (vl)), \
    (u), (e), (vl));							\
  vm32_f_t kcd_hz_ = VM32_VFMUL_VF (kcd_zh_, -0.5f, (vl));		\
  vm32_f_t kcd_p_ = VM32_VFADD_VF (kcd_hz_, 1.0f, (vl));		\
  vm32_f_t kcd_pe_ = VM32_VFSUB_VV (kcd_hz_,				\
				    VM32_VFSUB_VF (kcd_p_, 1.0f, (vl)), (vl)); \
  vm32_f_t kcd_c_ = VM32_VFSUB_VV (					\
    kcd_t_, VM32_VFMUL_VF (kcd_zl_, 0.5f, (vl)), (vl));		\
  (cho) = kcd_p_;							\
  (clo) = VM32_VFADD_VV (kcd_c_, kcd_pe_, (vl));			\
} while (0)

/* (sh,sl) = sin(u+e), |u| <= pi/2 (the wider VM32_Q* minimax), e the
   reduced-argument tail.  Same double-float shape as VM32_KSIN_DD: the
   products u^2, u^3 and u^3*R carry their fma residuals, and the tail
   enters to first order as e*cos(u), with cos(u) ~ 1 - zh/2 + zh^2/24
   (the wide range makes the two-term form useless near pi/2).

   The assembly differs from VM32_KSIN_DD in two places.  First, only
   the leading sum u + C needs to be exact (|C| < 0.37|u| over the whole
   range, so the fast-two-sum with u applies), while the two tails are
   O(2^-24) of the result and their own rounding is far below the
   caller's single final add -- so they ride in the low word instead of
   going through a general two-sum.  Second, the last Horner step is
   compensated: the wide range makes u^3*R reach 0.57 of the result, so
   the 0.5 ulp rounding of R itself would show up as 0.3 ulp of output
   (measured 0.64 ulp worst case for the plain chain); carrying R as
   (R, Rl) and folding v*Rl into the low word removes it.  */
#define VM32_KSIN_HALF(sho, slo, u, e, vl) do {				\
  vm32_f_t ksh_zh_ = VM32_VFMUL_VV ((u), (u), (vl));			\
  vm32_f_t ksh_zl_ = VM32_VFMSAC_VV (ksh_zh_, (u), (u), (vl));		\
  vm32_f_t ksh_v_ = VM32_VFMUL_VV (ksh_zh_, (u), (vl));			\
  vm32_f_t ksh_ve_ = VM32_VFMSAC_VV (ksh_v_, ksh_zh_, (u), (vl));	\
  ksh_ve_ = VM32_VFMACC_VV (ksh_ve_, ksh_zl_, (u), (vl));		\
  vm32_f_t ksh_R_ = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_Q5, (vl)),	\
				    VM32_Q6, ksh_zh_, (vl));		\
  ksh_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_Q4, (vl)), ksh_zh_, ksh_R_, (vl)); \
  ksh_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_Q3, (vl)), ksh_zh_, ksh_R_, (vl)); \
  ksh_R_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_Q2, (vl)), ksh_zh_, ksh_R_, (vl)); \
  /* last step carried as (Rh, Rl): Rh + Rl = Q1 + zh*R exactly.  */	\
  vm32_f_t ksh_Rh_ = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_Q1, (vl)),	\
				     ksh_zh_, ksh_R_, (vl));		\
  vm32_f_t ksh_Rl_ = VM32_VFMACC_VV (					\
    VM32_VFRSUB_VF (ksh_Rh_, VM32_Q1, (vl)), ksh_zh_, ksh_R_, (vl));	\
  vm32_f_t ksh_C_ = VM32_VFMUL_VV (ksh_v_, ksh_Rh_, (vl));		\
  vm32_f_t ksh_Ce_ = VM32_VFMSAC_VV (ksh_C_, ksh_v_, ksh_Rh_, (vl));	\
  ksh_Ce_ = VM32_VFMACC_VV (ksh_Ce_, ksh_ve_, ksh_Rh_, (vl));		\
  ksh_Ce_ = VM32_VFMACC_VV (ksh_Ce_, ksh_v_, ksh_Rl_, (vl));		\
  /* tail: e*cos(u); 1-zh/2+zh^2/24 is within 0.02 over the whole range.  */ \
  vm32_f_t ksh_ct_ = VM32_VFRSUB_VF (					\
    VM32_VFMUL_VF (ksh_zh_, 1.0f / 24.0f, (vl)), 0.5f, (vl));		\
  vm32_f_t ksh_ee_ = VM32_VFNMSAC_VV ((e),				\
    VM32_VFMUL_VV ((e), ksh_zh_, (vl)), ksh_ct_, (vl));		\
  /* s + se = u + C exactly (|u| > |C|).  */				\
  vm32_f_t ksh_s_ = VM32_VFADD_VV ((u), ksh_C_, (vl));			\
  (sho) = ksh_s_;							\
  (slo) = VM32_VFADD_VV (						\
    VM32_VFADD_VV (VM32_VFADD_VV (VM32_VFSUB_VV ((u), ksh_s_, (vl)),	\
				  ksh_C_, (vl)), ksh_ee_, (vl)),	\
    ksh_Ce_, (vl));							\
} while (0)

/* ==================================================================== */
/* reductions                                                           */
/* ==================================================================== */

/* (y0,y1,n) = |x| - n*pi/2, with y0+y1 the reduced argument.  */
#define VM32_TRIG_RED(y0o, y1o, no, x, vl) do {				\
  vm32_f_t vmtr__t = VM32_FROM_BITS (					\
    VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, (vl)));			\
  vm32_f_t vmtr__kf = VM32_VFMUL_VF (vmtr__t, VM32_INVPIO2, (vl));	\
  vm32_i_t vmtr__n = VM32_VFCVT_X_F_V_RM (vmtr__kf, __RISCV_FRM_RNE, (vl)); \
  vm32_f_t vmtr__fn = VM32_VFCVT_F_X_V (vmtr__n, (vl));			\
  vm32_f_t vmtr__z = VM32_VFNMSAC_VF (vmtr__t, VM32_PIO2_HI, vmtr__fn, (vl)); \
  vm32_f_t vmtr__u = VM32_VFNMSAC_VF (vmtr__z, VM32_PIO2_LO, vmtr__fn, (vl)); \
  vm32_f_t vmtr__e = VM32_VFNMSAC_VF (VM32_VFSUB_VV (vmtr__z, vmtr__u, (vl)), \
				     VM32_PIO2_LO, vmtr__fn, (vl));	\
  (y0o) = vmtr__u;							\
  (y1o) = vmtr__e;							\
  (no) = vmtr__n;							\
} while (0)

/* True when the reduced pair is too close to a multiple of pi/2 for the
   fast-path split's ~2^-45 tail.  */
#define VM32_RED_BAD(y0, vl)						\
  VM32_VMSLTU_VX (VM32_VAND_VX (VM32_BITS (y0), 0x7fffffffu, (vl)),	\
		  VM32_TINY_BITS, (vl))

/* Fallback: the whole strip through the binary64 kernel.  */
#define VM32_FALLBACK1(name, x, vl)					\
  VM32_NARROW (VM32_D64 (name) (VM32_WIDEN ((x), (vl)), (vl)), (vl))

/* ==================================================================== */
/* quadrant select (applied to the hi and lo words alike)               */
/* ==================================================================== */

#define VM32_TRIG_Q(n, vl) VM32_VAND_VX (VM32_REINT_I_U (n), 3, (vl))

/* sin(|x|):  q&1 -> cos, else sin;  sign flip when q&2.  */
#define VM32_TRIG_SIN2(q, sh, sl, ch, cl, ho, lo, vl) do {		\
  vm32_b_t tq__m0 = VM32_VMSNE_VX (VM32_VAND_VX ((q), 1, (vl)), 0, (vl)); \
  vm32_f_t tq__h = VM32_VFMERGE_VVM ((sh), (ch), tq__m0, (vl));		\
  vm32_f_t tq__l = VM32_VFMERGE_VVM ((sl), (cl), tq__m0, (vl));		\
  vm32_u_t tq__sb = VM32_VSLL_VX (VM32_VAND_VX ((q), 2, (vl)), 30, (vl)); \
  vm32_f_t tq__f = VM32_FROM_BITS (tq__sb);				\
  tq__h = VM32_VFSGNJX_VV (tq__h, tq__f, (vl));				\
  tq__l = VM32_VFSGNJX_VV (tq__l, tq__f, (vl));				\
  (ho) = tq__h;								\
  (lo) = tq__l;								\
} while (0)

/* cos(|x|):  q&1 -> sin, else cos;  sign flip when (q ^ (q>>1)) & 1.  */
#define VM32_TRIG_COS2(q, sh, sl, ch, cl, ho, lo, vl) do {		\
  vm32_b_t tq__m0 = VM32_VMSNE_VX (VM32_VAND_VX ((q), 1, (vl)), 0, (vl)); \
  vm32_f_t tq__h = VM32_VFMERGE_VVM ((ch), (sh), tq__m0, (vl));		\
  vm32_f_t tq__l = VM32_VFMERGE_VVM ((cl), (sl), tq__m0, (vl));		\
  vm32_u_t tq__f = VM32_VAND_VX (					\
    VM32_VXOR_VV ((q), VM32_VSRL_VX ((q), 1, (vl)), (vl)), 1, (vl));	\
  vm32_u_t tq__sb = VM32_VSLL_VX (tq__f, 31, (vl));			\
  vm32_f_t tq__s = VM32_FROM_BITS (tq__sb);				\
  tq__h = VM32_VFSGNJX_VV (tq__h, tq__s, (vl));				\
  tq__l = VM32_VFSGNJX_VV (tq__l, tq__s, (vl));				\
  (ho) = tq__h;								\
  (lo) = tq__l;								\
} while (0)

/* tan(|x|) = q&1 ? -(cos/sin) : sin/cos.  The two double-float words
   are first normalised (fast-two-sum) so the initial quotient is a 0.5 ulp
   approximation, then the quotient is refined with the fma residual.  */
#define VM32_TRIG_TAN2(q, sh, sl, ch, cl, to, x, vl) do {		\
  vm32_b_t tq__m0 = VM32_VMSNE_VX (VM32_VAND_VX ((q), 1, (vl)), 0, (vl)); \
  vm32_f_t tq__nh = VM32_VFMERGE_VVM ((sh), (ch), tq__m0, (vl));	\
  vm32_f_t tq__nl = VM32_VFMERGE_VVM ((sl), (cl), tq__m0, (vl));	\
  vm32_f_t tq__dh = VM32_VFMERGE_VVM ((ch), (sh), tq__m0, (vl));	\
  vm32_f_t tq__dl = VM32_VFMERGE_VVM ((cl), (sl), tq__m0, (vl));	\
  vm32_f_t tq__s = VM32_VFADD_VV (tq__nh, tq__nl, (vl));		\
  vm32_f_t tq__sl2 = VM32_VFSUB_VV (tq__nl,					\
    VM32_VFSUB_VV (tq__s, tq__nh, (vl)), (vl));				\
  vm32_f_t tq__c = VM32_VFADD_VV (tq__dh, tq__dl, (vl));		\
  vm32_f_t tq__cl2 = VM32_VFSUB_VV (tq__dl,				\
    VM32_VFSUB_VV (tq__c, tq__dh, (vl)), (vl));				\
  vm32_f_t tq__q = VM32_VFDIV_VV (tq__s, tq__c, (vl));			\
  vm32_f_t tq__e = VM32_VFNMSAC_VV (tq__s, tq__q, tq__c, (vl));	\
  tq__e = VM32_VFADD_VV (tq__e, tq__sl2, (vl));				\
  tq__e = VM32_VFNMSAC_VV (tq__e, tq__q, tq__cl2, (vl));		\
  tq__e = VM32_VFDIV_VV (tq__e, tq__c, (vl));				\
  tq__q = VM32_VFADD_VV (tq__q, tq__e, (vl));				\
  {									\
    vm32_u_t tq__sb = VM32_VMERGE_VXM (					\
      VM32_VMV_V_X (0, (vl)), 0x80000000u, tq__m0, (vl));		\
    tq__q = VM32_VFSGNJX_VV (tq__q, VM32_FROM_BITS (tq__sb), (vl));	\
  }									\
  tq__q = VM32_VFSGNJX_VV (tq__q, (x), (vl));				\
  (to) = tq__q;								\
} while (0)

/* ==================================================================== */
/* public kernels                                                       */
/* ==================================================================== */

/* Per-lane fast/wide select (length-sweep consistency).  */
#define VM32_FAST_OR_WIDE(fast, wide, bad, vl) do {			\
  if (VM32_VFIRST_M ((bad), (vl)) >= 0)					\
    (fast) = VM32_VFMERGE_VVM ((fast), (wide), (bad), (vl));		\
} while (0)

static inline vm32_f_t
VM32_KFN (sin) (vm32_f_t x, size_t vl)
{
  vm32_u_t ax = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, vl);
  vm32_b_t mbad = VM32_RANGE_BAD (ax, VM32_TRIG_MAX, vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.5f, mbad, vl);
  vm32_f_t y0, y1;
  vm32_i_t n;
  VM32_TRIG_RED (y0, y1, n, xs, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_RED_BAD (y0, vl), vl);
  y0 = VM32_VFMERGE_VFM (y0, 0.5f, mbad, vl);
  y1 = VM32_VFMERGE_VFM (y1, 0.0f, mbad, vl);
  vm32_f_t sh, sl, ch, cl;
  VM32_KSIN_DD (sh, sl, y0, y1, vl);
  VM32_KCOS_DD (ch, cl, y0, y1, vl);
  vm32_u_t q = VM32_TRIG_Q (n, vl);
  vm32_f_t oh, ol;
  VM32_TRIG_SIN2 (q, sh, sl, ch, cl, oh, ol, vl);
  vm32_f_t r = VM32_VFADD_VV (oh, ol, vl);
  r = VM32_VFSGNJX_VV (r, xs, vl);
  VM32_FAST_OR_WIDE (r, VM32_FALLBACK1 (sin, x, vl), mbad, vl);
  return r;
}

static inline vm32_f_t
VM32_KFN (cos) (vm32_f_t x, size_t vl)
{
  vm32_u_t ax = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, vl);
  vm32_b_t mbad = VM32_RANGE_BAD (ax, VM32_TRIG_MAX, vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.5f, mbad, vl);
  vm32_f_t y0, y1;
  vm32_i_t n;
  VM32_TRIG_RED (y0, y1, n, xs, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_RED_BAD (y0, vl), vl);
  y0 = VM32_VFMERGE_VFM (y0, 0.5f, mbad, vl);
  y1 = VM32_VFMERGE_VFM (y1, 0.0f, mbad, vl);
  vm32_f_t sh, sl, ch, cl;
  VM32_KSIN_DD (sh, sl, y0, y1, vl);
  VM32_KCOS_DD (ch, cl, y0, y1, vl);
  vm32_u_t q = VM32_TRIG_Q (n, vl);
  vm32_f_t oh, ol;
  VM32_TRIG_COS2 (q, sh, sl, ch, cl, oh, ol, vl);
  vm32_f_t r = VM32_VFADD_VV (oh, ol, vl);
  VM32_FAST_OR_WIDE (r, VM32_FALLBACK1 (cos, x, vl), mbad, vl);
  return r;
}

static inline vm32_f_t
VM32_KFN (tan) (vm32_f_t x, size_t vl)
{
  vm32_u_t ax = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, vl);
  vm32_b_t mbad = VM32_RANGE_BAD (ax, VM32_TRIG_MAX, vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.5f, mbad, vl);
  vm32_f_t y0, y1;
  vm32_i_t n;
  VM32_TRIG_RED (y0, y1, n, xs, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_RED_BAD (y0, vl), vl);
  y0 = VM32_VFMERGE_VFM (y0, 0.5f, mbad, vl);
  y1 = VM32_VFMERGE_VFM (y1, 0.0f, mbad, vl);
  vm32_f_t sh, sl, ch, cl;
  VM32_KSIN_DD (sh, sl, y0, y1, vl);
  VM32_KCOS_DD (ch, cl, y0, y1, vl);
  vm32_u_t q = VM32_TRIG_Q (n, vl);
  vm32_f_t r;
  VM32_TRIG_TAN2 (q, sh, sl, ch, cl, r, xs, vl);
  VM32_FAST_OR_WIDE (r, VM32_FALLBACK1 (tan, x, vl), mbad, vl);
  return r;
}

/* ---- *pi family ---- */

/* (p0,p1,q,r) = (pi*r as a double-float, 2*rint(2x), r = x - q/2).  */
#define VM32_PI_RED(p0o, p1o, qo, ro, x, vl) do {			\
  vm32_f_t vmpr__t = VM32_VFMUL_VF ((x), 2.0f, (vl));			\
  vm32_i_t vmpr__q = VM32_VFCVT_X_F_V_RM (vmpr__t, __RISCV_FRM_RNE, (vl)); \
  vm32_f_t vmpr__r = VM32_VFNMSAC_VF (					\
    (x), 0.5f, VM32_VFCVT_F_X_V (vmpr__q, (vl)), (vl));			\
  vm32_f_t vmpr__ph = VM32_VFMUL_VF (vmpr__r, VM32_PI_HI, (vl));	\
  vm32_f_t vmpr__pl = VM32_VFMACC_VF (					\
    VM32_VFMSAC_VF (vmpr__ph, VM32_PI_HI, vmpr__r, (vl)),		\
    VM32_PI_LO, vmpr__r, (vl));						\
  (p0o) = vmpr__ph;							\
  (p1o) = vmpr__pl;							\
  (qo) = vmpr__q;							\
  (ro) = vmpr__r;							\
} while (0)

#define VM32_PI_EXACT_BAD(r, vl)					\
  VM32_VMFEQ_VF ((r), 0.0f, (vl))

/* (p0,p1,q,r) = (pi*r as a double-float, q = rint(x+1/2), r = x - (q-1/2)),
   with |r| <= 1/2, so that

       cos(pi*x) = sin(pi*(x+1/2)) = (-1)^q * sin(pi*r)

   needs only the one wide sin kernel.  q is taken from the rounded
   x+1/2: when that rounding crosses a tie, q is off by one and r lands
   at -+(1/2 -+ eps) instead of +-(1/2 -+ eps), which the (-1)^q factor
   turns into the same value, so the argument stays inside the minimax
   range for every lane.  r carries its exact residual re (fast-two-sum;
   |q-1/2| >= |x| whenever the subtraction is inexact, and it is exact
   otherwise, so re is always the true residual), hence (p0,p1) keeps
   the full ~2^-45 tail.  */
#define VM32_PI_RED1(p0o, p1o, qo, ro, x, vl) do {			\
  vm32_i_t vmpr1__q = VM32_VFCVT_X_F_V_RM (				\
    VM32_VFADD_VF ((x), 0.5f, (vl)), __RISCV_FRM_RNE, (vl));		\
  vm32_f_t vmpr1__m = VM32_VFSUB_VF (					\
    VM32_VFCVT_F_X_V (vmpr1__q, (vl)), 0.5f, (vl));			\
  vm32_f_t vmpr1__r = VM32_VFSUB_VV ((x), vmpr1__m, (vl));		\
  vm32_f_t vmpr1__re = VM32_VFSUB_VV (					\
    (x), VM32_VFADD_VV (vmpr1__r, vmpr1__m, (vl)), (vl));		\
  vm32_f_t vmpr1__ph = VM32_VFMUL_VF (vmpr1__r, VM32_PI_HI, (vl));	\
  vm32_f_t vmpr1__pl = VM32_VFMACC_VF (					\
    VM32_VFMACC_VF (VM32_VFMSAC_VF (vmpr1__ph, VM32_PI_HI,		\
				    vmpr1__r, (vl)),			\
		    VM32_PI_LO, vmpr1__r, (vl)),			\
    VM32_PI_HI, vmpr1__re, (vl));					\
  (p0o) = vmpr1__ph;							\
  (p1o) = vmpr1__pl;							\
  (qo) = vmpr1__q;							\
  (ro) = vmpr1__r;							\
} while (0)

/* (p0,p1,q,r) = (pi*r as a double-float, q = rint(x), r = x - q), with
   |r| <= 1/2, so that

       sin(pi*x) = (-1)^q * sin(pi*r)

   needs only the one wide sin kernel.  x - rint(x) is exact in binary32
   (|x| >= 1 puts both operands within a factor of two -- Sterbenz -- and
   |x| < 1 gives q = 0 or a Sterbenz-exact 1/2 <= |x| < 1 case), so no
   residual is needed here.  */
#define VM32_PI_RED0(p0o, p1o, qo, ro, x, vl) do {			\
  vm32_i_t vmpr0__q = VM32_VFCVT_X_F_V_RM ((x), __RISCV_FRM_RNE, (vl));	\
  vm32_f_t vmpr0__r = VM32_VFSUB_VV (					\
    (x), VM32_VFCVT_F_X_V (vmpr0__q, (vl)), (vl));			\
  vm32_f_t vmpr0__ph = VM32_VFMUL_VF (vmpr0__r, VM32_PI_HI, (vl));	\
  vm32_f_t vmpr0__pl = VM32_VFMACC_VF (					\
    VM32_VFMSAC_VF (vmpr0__ph, VM32_PI_HI, vmpr0__r, (vl)),		\
    VM32_PI_LO, vmpr0__r, (vl));					\
  (p0o) = vmpr0__ph;							\
  (p1o) = vmpr0__pl;							\
  (qo) = vmpr0__q;							\
  (ro) = vmpr0__r;							\
} while (0)

/* range mask + sanitized fast-path input for the pi family.  */
#define VM32_PI_PREP(mbado, xso, x, vl) do {				\
  vm32_u_t vmp__ax = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, (vl));	\
  (mbado) = VM32_RANGE_BAD (vmp__ax, VM32_PI_MAX, (vl));		\
  (xso) = VM32_VFMERGE_VFM ((x), 0.5f, (mbado), (vl));			\
} while (0)

/* sinpi rides the same single-kernel structure: the modulo-1 reduction
   leaves |pi*r| <= pi/2 and only the sign of the wide sin kernel depends
   on the quadrant.  */
static inline vm32_f_t
VM32_KFN (sinpi) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad;
  vm32_f_t xs;
  VM32_PI_PREP (mbad, xs, x, vl);
  vm32_f_t p0, p1, r;
  vm32_i_t q;
  VM32_PI_RED0 (p0, p1, q, r, xs, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_PI_EXACT_BAD (r, vl), vl);
  p0 = VM32_VFMERGE_VFM (p0, 0.5f, mbad, vl);
  p1 = VM32_VFMERGE_VFM (p1, 0.0f, mbad, vl);
  vm32_f_t sh, sl;
  VM32_KSIN_HALF (sh, sl, p0, p1, vl);
  vm32_f_t out = VM32_VFSGNJX_VV (
    VM32_VFADD_VV (sh, sl, vl),
    VM32_FROM_BITS (VM32_VSLL_VX (
		      VM32_VAND_VX (VM32_REINT_I_U (q), 1, vl), 31, vl)),
    vl);
  VM32_FAST_OR_WIDE (out, VM32_FALLBACK1 (sinpi, x, vl), mbad, vl);
  return out;
}

/* cospi is the one entry that does not need a sin *and* a cos kernel:
   the modulo-pi reduction below leaves |pi*r| <= pi/2, and the
   quadrant only flips the sign of the single wide sin kernel, so the
   fast path evaluates one polynomial instead of two.  */
static inline vm32_f_t
VM32_KFN (cospi) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad;
  vm32_f_t xs;
  VM32_PI_PREP (mbad, xs, x, vl);
  vm32_f_t p0, p1, r;
  vm32_i_t q;
  VM32_PI_RED1 (p0, p1, q, r, xs, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_PI_EXACT_BAD (r, vl), vl);
  p0 = VM32_VFMERGE_VFM (p0, 0.5f, mbad, vl);
  p1 = VM32_VFMERGE_VFM (p1, 0.0f, mbad, vl);
  vm32_f_t sh, sl;
  VM32_KSIN_HALF (sh, sl, p0, p1, vl);
  vm32_f_t out = VM32_VFSGNJX_VV (
    VM32_VFADD_VV (sh, sl, vl),
    VM32_FROM_BITS (VM32_VSLL_VX (
		      VM32_VAND_VX (VM32_REINT_I_U (q), 1, vl), 31, vl)),
    vl);
  VM32_FAST_OR_WIDE (out, VM32_FALLBACK1 (cospi, x, vl), mbad, vl);
  return out;
}

static inline vm32_f_t
VM32_KFN (tanpi) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad;
  vm32_f_t xs;
  VM32_PI_PREP (mbad, xs, x, vl);
  vm32_f_t p0, p1, r;
  vm32_i_t q2;
  VM32_PI_RED (p0, p1, q2, r, xs, vl);
  /* r == 0 (integers/half-integers: exact results) and |r| == 1/4
     (tanpi = +-1 exactly) are the only exact points; both go wide.  */
  mbad = VM32_VMOR_MM (mbad, VM32_PI_EXACT_BAD (r, vl), vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMFEQ_VF (
			 VM32_VFABS_V (r, vl), 0.25f, vl), vl);
  p0 = VM32_VFMERGE_VFM (p0, 0.5f, mbad, vl);
  p1 = VM32_VFMERGE_VFM (p1, 0.0f, mbad, vl);
  vm32_f_t sh, sl, ch, cl;
  VM32_KSIN_DD (sh, sl, p0, p1, vl);
  VM32_KCOS_DD (ch, cl, p0, p1, vl);
  vm32_u_t q = VM32_TRIG_Q (q2, vl);
  vm32_f_t t;
  VM32_TRIG_TAN2 (q, sh, sl, ch, cl, t, VM32_VFMV_V_F (1.0f, vl), vl);
  VM32_FAST_OR_WIDE (t, VM32_FALLBACK1 (tanpi, x, vl), mbad, vl);
  return t;
}

/* ==================================================================== */
/* fused two-output kernels                                             */
/* ==================================================================== */

#define VM32_SINCOS(syo, cyo, x, vl) do {				\
  vm32_u_t vmsc32__ax = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffu, (vl)); \
  vm32_b_t vmsc32__bad = VM32_RANGE_BAD (vmsc32__ax, VM32_TRIG_MAX, (vl)); \
  vm32_f_t vmsc32__xs = VM32_VFMERGE_VFM ((x), 0.5f, vmsc32__bad, (vl)); \
  vm32_f_t vmsc32__y0, vmsc32__y1;					\
  vm32_i_t vmsc32__n;							\
  VM32_TRIG_RED (vmsc32__y0, vmsc32__y1, vmsc32__n, vmsc32__xs, (vl));	\
  vmsc32__bad = VM32_VMOR_MM (vmsc32__bad,				\
			      VM32_RED_BAD (vmsc32__y0, (vl)), (vl));	\
  vmsc32__y0 = VM32_VFMERGE_VFM (vmsc32__y0, 0.5f, vmsc32__bad, (vl));	\
  vmsc32__y1 = VM32_VFMERGE_VFM (vmsc32__y1, 0.0f, vmsc32__bad, (vl));	\
  {									\
    vm32_f_t vmsc32__sh, vmsc32__sl, vmsc32__ch, vmsc32__cl;		\
    VM32_KSIN_DD (vmsc32__sh, vmsc32__sl, vmsc32__y0, vmsc32__y1, (vl)); \
    VM32_KCOS_DD (vmsc32__ch, vmsc32__cl, vmsc32__y0, vmsc32__y1, (vl)); \
    vm32_u_t vmsc32__q = VM32_TRIG_Q (vmsc32__n, (vl));			\
    vm32_f_t vmsc32__sn, vmsc32__sl2, vmsc32__cn, vmsc32__cl2;		\
    VM32_TRIG_SIN2 (vmsc32__q, vmsc32__sh, vmsc32__sl, vmsc32__ch,	\
		    vmsc32__cl, vmsc32__sn, vmsc32__sl2, (vl));		\
    VM32_TRIG_COS2 (vmsc32__q, vmsc32__sh, vmsc32__sl, vmsc32__ch,	\
		    vmsc32__cl, vmsc32__cn, vmsc32__cl2, (vl));		\
    vm32_f_t vmsc32__s = VM32_VFADD_VV (vmsc32__sn, vmsc32__sl2, (vl));	\
    vm32_f_t vmsc32__c = VM32_VFADD_VV (vmsc32__cn, vmsc32__cl2, (vl));	\
    vmsc32__s = VM32_VFSGNJX_VV (vmsc32__s, vmsc32__xs, (vl));		\
    if (VM32_VFIRST_M (vmsc32__bad, (vl)) >= 0)				\
      {									\
	vm_f_t vmsc32__ws, vmsc32__wc;					\
	VM_SINCOS (vmsc32__ws, vmsc32__wc, VM32_WIDEN ((x), (vl)), (vl)); \
	vm32_f_t vmsc32__rw = VM32_NARROW (vmsc32__ws, (vl));		\
	vm32_f_t vmsc32__cw = VM32_NARROW (vmsc32__wc, (vl));		\
	vmsc32__s = VM32_VFMERGE_VVM (vmsc32__s, vmsc32__rw,		\
				     vmsc32__bad, (vl));		\
	vmsc32__c = VM32_VFMERGE_VVM (vmsc32__c, vmsc32__cw,		\
				     vmsc32__bad, (vl));		\
      }									\
    (syo) = vmsc32__s;							\
    (cyo) = vmsc32__c;							\
  }									\
} while (0)

#define VM32_SINCOSPI(syo, cyo, x, vl) do {				\
  vm32_b_t vmscp32__bad;						\
  vm32_f_t vmscp32__xs;							\
  VM32_PI_PREP (vmscp32__bad, vmscp32__xs, (x), (vl));			\
  vm32_f_t vmscp32__p0, vmscp32__p1, vmscp32__r;			\
  vm32_i_t vmscp32__q2;							\
  VM32_PI_RED (vmscp32__p0, vmscp32__p1, vmscp32__q2, vmscp32__r,	\
	       vmscp32__xs, (vl));					\
  vmscp32__bad = VM32_VMOR_MM (vmscp32__bad,				\
			       VM32_PI_EXACT_BAD (vmscp32__r, (vl)), (vl)); \
  vmscp32__p0 = VM32_VFMERGE_VFM (vmscp32__p0, 0.5f, vmscp32__bad, (vl)); \
  vmscp32__p1 = VM32_VFMERGE_VFM (vmscp32__p1, 0.0f, vmscp32__bad, (vl)); \
  {									\
    vm32_f_t vmscp32__sh, vmscp32__sl, vmscp32__ch, vmscp32__cl;	\
    VM32_KSIN_DD (vmscp32__sh, vmscp32__sl, vmscp32__p0, vmscp32__p1, (vl)); \
    VM32_KCOS_DD (vmscp32__ch, vmscp32__cl, vmscp32__p0, vmscp32__p1, (vl)); \
    vm32_u_t vmscp32__q = VM32_TRIG_Q (vmscp32__q2, (vl));		\
    vm32_f_t vmscp32__sn, vmscp32__sl2, vmscp32__cn, vmscp32__cl2;	\
    VM32_TRIG_SIN2 (vmscp32__q, vmscp32__sh, vmscp32__sl,		\
		    vmscp32__ch, vmscp32__cl, vmscp32__sn,		\
		    vmscp32__sl2, (vl));				\
    VM32_TRIG_COS2 (vmscp32__q, vmscp32__sh, vmscp32__sl,		\
		    vmscp32__ch, vmscp32__cl, vmscp32__cn,		\
		    vmscp32__cl2, (vl));				\
    vm32_f_t vmscp32__s = VM32_VFADD_VV (vmscp32__sn, vmscp32__sl2, (vl)); \
    vm32_f_t vmscp32__c = VM32_VFADD_VV (vmscp32__cn, vmscp32__cl2, (vl)); \
    if (VM32_VFIRST_M (vmscp32__bad, (vl)) >= 0)			\
      {									\
	vm_f_t vmscp32__ws, vmscp32__wc;				\
	VM_SINCOSPI (vmscp32__ws, vmscp32__wc, VM32_WIDEN ((x), (vl)),	\
		     VM_PI_ALL ((vl)), (vl));				\
	vm32_f_t vmscp32__rw = VM32_NARROW (vmscp32__ws, (vl));		\
	vm32_f_t vmscp32__cw = VM32_NARROW (vmscp32__wc, (vl));		\
	vmscp32__s = VM32_VFMERGE_VVM (vmscp32__s, vmscp32__rw,		\
				       vmscp32__bad, (vl));		\
	vmscp32__c = VM32_VFMERGE_VVM (vmscp32__c, vmscp32__cw,		\
				       vmscp32__bad, (vl));		\
      }									\
    (syo) = vmscp32__s;							\
    (cyo) = vmscp32__c;							\
  }									\
} while (0)
