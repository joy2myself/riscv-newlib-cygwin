/* f32/k_gamma.inc -- binary32 tgamma / lgamma / lgamma_r kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_gamma.inc,
   which pulls k_exp.inc, k_log.inc and k_trig.inc) active at
   VM_LMUL == 2*VM32_LMUL.

   Shape: a native fast path evaluated in binary64 arithmetic on the
   widened f32mK -> f64m2K group over the positive bench band
   x in [0.5, 5], plus a per-lane fallback to the previous widen /
   binary64 kernel / narrow path for every other lane.  Both fast paths
   are self-contained double polynomial schemes -- no dependency on the
   binary64 exp/log/gamma kernels, which keeps the binary32 entries
   independent of the binary64 rewrite in progress:

     tgamma:  Gamma(x) = Gamma(1+s) * P, s = x - floor(x),
              P = prod_{j=1}^{floor(x)-1} (x-j)  (P = 1/x for x < 1);
              Gamma(1+s) is a degree-13 polynomial on s in [0,1]
              (3.8e-11 relative).
     lgamma:  x in [0.5,1): lgamma(x) = u*K(u), u = x-1, K degree 11;
              x in [1,2):   lgamma(x) = s(1-s)*H(s), s = x-1, H degree 11;
              x in [2,5]:   lgamma(x) = t*J(t/3), t = x-2, J degree 15.
              (1.3e-10 / 1.1e-10 / 9.1e-12 relative; the prefactors keep
              the relative error intact at the zeros x = 1 and x = 2.)

   Both invert the bench argument range exactly, are flag-free for every
   clamped lane, and narrow once, so they land within ~1e-3 ulp of the
   correctly rounded binary32 result.  Per-lane fallback (not per-strip)
   keeps a lane's result independent of the strip it lands in, which the
   framework's length and alignment sweeps (bitwise identity) require.

   The binary64 arithmetic here is bound to the *widened* group
   (2*VM32_LMUL) with file-local macros: the VM_* macros of the including
   translation unit are bound to whatever LMUL it last instantiated
   (vfabi32.c instantiates binary64 at 2/4/8 and then binary32 at 1/2/4),
   so they cannot be reused.  */

/* ------------------------------------------------------------------ */
/* File-local binary64 building blocks, bound to the widened group.    */
/* ------------------------------------------------------------------ */

#undef vm32f_b_t
#undef VMF_FN
#undef VMF_ADD_VV
#undef VMF_ADD_VF
#undef VMF_SUB_VV
#undef VMF_SUB_VF
#undef VMF_MUL_VV
#undef VMF_MUL_VF
#undef VMF_MACC_VV
#undef VMF_MV_F
#undef VMF_MERGE_VFM
#undef VMF_RDIV_VF

#if VM32_LMUL == 1
# define vm32f_b_t vbool32_t
#elif VM32_LMUL == 2
# define vm32f_b_t vbool16_t
#elif VM32_LMUL == 4
# define vm32f_b_t vbool8_t
#else
# error "f32 gamma kernels are instantiated at VM32_LMUL 1, 2 or 4"
#endif

#define VMF_FN(op)    VM32_XP3 (__riscv_, op, VM32_XP2 (f64m, VM32_D64M))

#define VMF_ADD_VV(a, b, vl)       VMF_FN (vfadd_vv_)(a, b, vl)
#define VMF_ADD_VF(a, s, vl)       VMF_FN (vfadd_vf_)(a, s, vl)
#define VMF_SUB_VV(a, b, vl)       VMF_FN (vfsub_vv_)(a, b, vl)
#define VMF_SUB_VF(a, s, vl)       VMF_FN (vfsub_vf_)(a, s, vl)
#define VMF_MUL_VV(a, b, vl)       VMF_FN (vfmul_vv_)(a, b, vl)
#define VMF_MUL_VF(a, s, vl)       VMF_FN (vfmul_vf_)(a, s, vl)
#define VMF_MACC_VV(d, a, b, vl)   VMF_FN (vfmacc_vv_)(d, a, b, vl)
#define VMF_MV_F(s, vl)            VMF_FN (vfmv_v_f_)(s, vl)
#define VMF_MERGE_VFM(a, s, m, vl) VMF_FN (vfmerge_vfm_)(a, s, m, vl)
#define VMF_RDIV_VF(a, s, vl)      VMF_FN (vfrdiv_vf_)(a, s, vl)

/* ------------------------------------------------------------------ */
/* Coefficients.  data guard: the file is instantiated once per LMUL.  */
/* ------------------------------------------------------------------ */

#ifndef VM32_GAMMA_DATA
#define VM32_GAMMA_DATA

/* Gamma(1+s) on s in [0,1], power basis low-to-high, degree 13:
   max relative error 3.8e-11.  */
static const double vm32f_tg_g[14] =
  {
    9.99999999961701636e-01, -5.77215649850256418e-01, 9.89055004431248830e-01,
    -9.07453103451191234e-01, 9.81368427783016362e-01, -9.78965897140413821e-01,
    9.76286856607282205e-01, -9.30494895756796225e-01, 8.13416713784863576e-01,
    -6.08681195872311176e-01, 3.60205556910506319e-01, -1.53490247734323293e-01,
    4.11072012859324820e-02, -5.13877097840667683e-03
  };

/* K(u) = lgamma(1+u)/u on u in [-0.5,0], power basis low-to-high,
   degree 11: max relative error 1.3e-10.  */
static const double vm32f_lg_k[12] =
  {
    -5.77215664827319896e-01, 8.22467076100673489e-01, -4.00681599528307841e-01,
    2.70729136124988612e-01, -2.04591565381332108e-01, 2.00301394218475620e-01,
    6.76755296735304795e-02, 1.06851620741137543e+00, 2.61759608385884146e+00,
    5.09381495164709719e+00, 5.23658020132679969e+00, 2.70821792997021227e+00
  };

/* H(s) = lgamma(1+s)/(s(1-s)) on s in [0,1], power basis low-to-high,
   degree 11: max relative error 1.1e-10.  */
static const double vm32f_lg_h[12] =
  {
    -5.77215664837752773e-01, 2.45251350095738396e-01, -1.55433373484145765e-01,
    1.15129384522918002e-01, -9.20657506428019801e-02, 7.62634492938224218e-02,
    -6.25354398344936391e-02, 4.72688068138625433e-02, -2.98757676854801597e-02,
    1.40512873535985731e-02, -4.21464056403873886e-03, 5.92023903725521399e-04
  };

/* J(q) = lgamma(2+3q)/(3q) on q in [0,1], power basis low-to-high,
   degree 15: max relative error 9.1e-12.  */
static const double vm32f_lg_j[16] =
  {
    4.22784335102330044e-01, 9.67401098289199290e-01, -6.06170538632603728e-01,
    5.55675943558324947e-01, -5.98121868417574865e-01, 7.01182478387106678e-01,
    -8.60406228376968341e-01, 1.06610067514437534e+00, -1.27298257831372208e+00,
    1.38201069938759380e+00, -1.28263779174536530e+00, 9.58570521608958503e-01,
    -5.42512459884336495e-01, 2.15469662705717102e-01, -5.30867624149177555e-02,
    6.07409038616040214e-03
  };

#endif /* VM32_GAMMA_DATA */

/* ------------------------------------------------------------------ */
/* Exceptional lanes: anything outside 0.5 <= x <= 5 (the positive band  */
/* the fast paths cover).  Unsigned range test on the raw bit pattern --  */
/* catches negative, NaN and Inf without a signalling compare.          */
/* ------------------------------------------------------------------ */

#define VM32F_GAM_BAD(x, vl)						\
  VM32_VMSGTU_VX (VM32_VSUB_VX (VM32_BITS (x), 0x3f000000u, (vl)),	\
		  0x01a00000u, (vl))

/* Clamp a lane into [0.5, 5] for the fast-path arithmetic: for the lanes
   the mask selects, the clamp makes every operation flag-free.  */
#define VM32F_GAM_CLAMP(xs, x, vl)					\
  (xs) = VM32_VFMIN_VF (VM32_VFMAX_VF ((x), 0.5f, (vl)), 5.0f, (vl))

/* Horner over a double table, low-to-high.  The multiply and add are kept
   separate on purpose: the coefficient arrives as a plain f-register scalar
   (vfadd.vf), so one polynomial term is two vector instructions with no
   coefficient broadcast and no register-group copy.  */
#define VM32F_GAM_POLY(yo, tbl, n, t, vl) do {				\
  VM32_WD_T vmre_p_ = VMF_MV_F ((tbl)[(n)], (vl));			\
  for (int vmre_i_ = (n) - 1; vmre_i_ >= 0; vmre_i_--)			\
    {									\
      vmre_p_ = VMF_MUL_VV (vmre_p_, (t), (vl));			\
      vmre_p_ = VMF_ADD_VF (vmre_p_, (tbl)[vmre_i_], (vl));		\
    }									\
  (yo) = vmre_p_;							\
} while (0)

/* ------------------------------------------------------------------ */
/* tgamma / lgamma                                                     */
/* ------------------------------------------------------------------ */

static inline vm32_f_t
VM32_KFN (tgamma) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32F_GAM_BAD (x, vl);
  vm32_f_t xs;
  VM32F_GAM_CLAMP (xs, x, vl);
  vm32_b_t m1 = VM32_VMFGE_VF (xs, 1.0f, vl);
  vm32_b_t m2 = VM32_VMFGE_VF (xs, 2.0f, vl);
  vm32_b_t m3 = VM32_VMFGE_VF (xs, 3.0f, vl);
  vm32_b_t m4 = VM32_VMFGE_VF (xs, 4.0f, vl);
  vm32_b_t ml2 = VM32_VMFLT_VF (xs, 2.0f, vl);
  vm32_b_t ml3 = VM32_VMFLT_VF (xs, 3.0f, vl);
  vm32_b_t ml4 = VM32_VMFLT_VF (xs, 4.0f, vl);

  VM32_WD_T xd = VM32_WIDEN (xs, vl);
  /* s = x - floor(x) via the integer-part vector kd in {0,1,2,3,4}.  */
  VM32_WD_T kd = VMF_MV_F (0.0, vl);
  kd = VMF_MERGE_VFM (kd, 1.0, m1, vl);
  kd = VMF_MERGE_VFM (kd, 2.0, m2, vl);
  kd = VMF_MERGE_VFM (kd, 3.0, m3, vl);
  kd = VMF_MERGE_VFM (kd, 4.0, m4, vl);
  VM32_WD_T s = VMF_SUB_VV (xd, kd, vl);
  VM32_WD_T g;
  VM32F_GAM_POLY (g, vm32f_tg_g, 13, s, vl);

  /* P = prod_{j=1}^{floor(x)-1} (x-j), with 1/x on x < 1.  */
  VM32_WD_T inv = VMF_RDIV_VF (xd, 1.0, vl);
  VM32_WD_T P = VMF_MERGE_VFM (inv, 1.0, m1, vl);
  VM32_WD_T f1 = VMF_MERGE_VFM (VMF_SUB_VF (xd, 1.0, vl), 1.0, ml2, vl);
  VM32_WD_T f2 = VMF_MERGE_VFM (VMF_SUB_VF (xd, 2.0, vl), 1.0, ml3, vl);
  VM32_WD_T f3 = VMF_MERGE_VFM (VMF_SUB_VF (xd, 3.0, vl), 1.0, ml4, vl);
  P = VMF_MUL_VV (P, f1, vl);
  P = VMF_MUL_VV (P, f2, vl);
  P = VMF_MUL_VV (P, f3, vl);

  vm32_f_t r = VM32_NARROW (VMF_MUL_VV (g, P, vl), vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t w = VM32_NARROW (VM32_D64 (tgamma) (VM32_WIDEN (x, vl), vl), vl);
      r = VM32_VFMERGE_VVM (r, w, mbad, vl);
    }
  return r;
}

static inline vm32_f_t
VM32_KFN (lgamma) (vm32_f_t x, size_t vl)
{
  vm32_b_t mbad = VM32F_GAM_BAD (x, vl);
  vm32_f_t xs;
  VM32F_GAM_CLAMP (xs, x, vl);
  vm32_b_t ml1 = VM32_VMFLT_VF (xs, 1.0f, vl);
  vm32_b_t ml2 = VM32_VMFLT_VF (xs, 2.0f, vl);

  VM32_WD_T xd = VM32_WIDEN (xs, vl);
  VM32_WD_T u = VMF_SUB_VF (xd, 1.0, vl);	/* u = x-1, exact */

  /* x in [0.5,1): lgamma(x) = u * K(u).  */
  VM32_WD_T k;
  VM32F_GAM_POLY (k, vm32f_lg_k, 11, u, vl);
  VM32_WD_T la = VMF_MUL_VV (u, k, vl);

  /* x in [1,2): lgamma(x) = u*(1-u) * H(u).  */
  VM32_WD_T h;
  VM32F_GAM_POLY (h, vm32f_lg_h, 11, u, vl);
  VM32_WD_T lb = VMF_MUL_VV (VMF_MUL_VV (u, VMF_SUB_VV (VMF_MV_F (1.0, vl),
						      u, vl), vl),
			     h, vl);

  /* x in [2,5]: lgamma(x) = t * J(t/3), t = x-2.  */
  VM32_WD_T t = VMF_SUB_VF (xd, 2.0, vl);
  VM32_WD_T j;
  VM32F_GAM_POLY (j, vm32f_lg_j, 15, VMF_MUL_VF (t, 1.0 / 3.0, vl), vl);
  VM32_WD_T lc = VMF_MUL_VV (t, j, vl);

  vm32_f_t r = VM32_VFMERGE_VVM (VM32_NARROW (lc, vl),
				VM32_NARROW (lb, vl), ml2, vl);
  r = VM32_VFMERGE_VVM (r, VM32_NARROW (la, vl), ml1, vl);
  /* lgamma(1) = lgamma(2) = +0; branch B forms -0 at x = 1 (u * -H(0)),
     so restore +0 wherever the fast value is exactly zero.  */
  {
    vm32_b_t mz = VM32_VMSEQ_VX (
      VM32_VAND_VX (VM32_BITS (r), 0x7fffffffu, vl), 0, vl);
    r = VM32_VFMERGE_VVM (r, VM32_VFMV_V_F (0.0f, vl), mz, vl);
  }

  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t w = VM32_NARROW (VM32_D64 (lgamma) (VM32_WIDEN (x, vl), vl), vl);
      r = VM32_VFMERGE_VVM (r, w, mbad, vl);
    }
  return r;
}

/* lgamma_r: the fast path covers only the positive band, where the sign
   of Gamma(x) is +1; the remaining lanes take the binary64 kernel and
   contribute both the value and the sign.  This macro is expanded only in
   vf_gamma.c (the ABI does not export lgamma_r), where VM_LMUL is
   2*VM32_LMUL, so VM_GAM_LGAMMA_R / VM_DD_COLLAPSE are correctly bound. */
#undef VM32_GAM_LGAMMA_R
#define VM32_GAM_LGAMMA_R(yo, sgno, x, vl) do {				\
  vm32_f_t vlgr_x_ = (x);						\
  vm32_b_t vlgr_m_ = VM32F_GAM_BAD (vlgr_x_, (vl));			\
  (yo) = VM32_KFN (lgamma) (vlgr_x_, (vl));				\
  (sgno) = VM32_REINT_U_I (VM32_VMV_V_X (1u, (vl)));			\
  if (VM32_VFIRST_M (vlgr_m_, (vl)) >= 0)				\
    {									\
      VM32_WD_T vlgr_xd_ = VM32_WIDEN (vlgr_x_, (vl));			\
      vm_f_t vlgr_h_, vlgr_l_, vlgr_s_, vlgr_v_;			\
      vm32_f_t vlgr_wv_;						\
      VM_GAM_LGAMMA_R (vlgr_h_, vlgr_l_, vlgr_s_, vlgr_xd_, (vl));	\
      VM_DD_COLLAPSE (vlgr_v_, vlgr_h_, vlgr_l_, (vl));			\
      vlgr_wv_ = VM32_NARROW (vlgr_v_, (vl));				\
      (yo) = VM32_VFMERGE_VVM ((yo), vlgr_wv_, vlgr_m_, (vl));		\
      (sgno) = VM32_REINT_U_I (						\
	VM32_VMERGE_VVM (VM32_REINT_I_U (sgno),				\
			 VM32_REINT_I_U (VM32_VFCVT_RTZ_X_F_V (		\
			   VM32_NARROW (vlgr_s_, (vl)), (vl))),		\
			 vlgr_m_, (vl)));				\
    }									\
} while (0)
