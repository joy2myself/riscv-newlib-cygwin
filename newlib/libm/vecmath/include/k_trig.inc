/* k_trig.inc -- sin/cos/tan/sincos and sinpi/cospi/tanpi/sincospi kernels.

   LMUL-generic template: include after vecmath_rvvcompat.h, vecmath_dd.h
   and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.

   sin/cos/tan/sincos port fdlibm (s_sin.c, s_cos.c, s_tan.c, k_sin.c,
   k_cos.c, k_tan.c) with the e_rem_pio2.c reduction:
     * |x| <= pi/4: no reduction (y0 = x, y1 = 0, n = 0).
     * |x| <= 2^19*pi/2: vectorized Cody-Waite, its 2nd and 3rd
       refinement rounds unconditional (branch-free).
     * larger |x|: scalar Payne-Hanek, per lane, in the caller's fixup
       (__ieee754_rem_pio2 is not exported by glibc).
     * inf/NaN: x - x (invalid for inf/sNaN).

   The *pi family (SYCL) needs no Payne-Hanek: q = rint(2x) makes the
   mod-2 reduction exact, and pi*r in double-double feeds the same
   kernels.  */

/* ---- reduction constants (e_rem_pio2.c) ---- */
#define VM_HALF		5.00000000000000000000e-01
#define VM_INVPIO2	6.36619772367581382433e-01
#define VM_PIO2_1	1.57079632673412561417e+00
#define VM_PIO2_1T	6.07710050650619224932e-11
#define VM_PIO2_2	6.07710050630396597660e-11
#define VM_PIO2_2T	2.02226624879595063154e-21
#define VM_PIO2_3	2.02226624871116645580e-21
#define VM_PIO2_3T	8.47842766036889956997e-32

/* dd pi for the *pi family */
#define VM_PI_HI	3.14159265358979311600e+00
#define VM_PI_LO	1.22464679914735320717e-16
#define VM_INVPI_HI	3.18309886183790691216e-01
#define VM_INVPI_LO	-1.96786766751824858819e-17

/* fdlibm polynomial constants */
#define VM_S1 -1.66666666666666324348e-01
#define VM_S2  8.33333333332248946124e-03
#define VM_S3 -1.98412698298579493134e-04
#define VM_S4  2.75573137070700676789e-06
#define VM_S5 -2.50507602534068634195e-08
#define VM_S6  1.58969099521155010221e-10
#define VM_C1  4.16666666666666019037e-02
#define VM_C2 -1.38888888888741095749e-03
#define VM_C3  2.48015872894767294178e-05
#define VM_C4 -2.75573143513906633035e-07
#define VM_C5  2.08757232129817482790e-09
#define VM_C6 -1.13596475577881948265e-11
#define VM_T0  3.33333333333334091986e-01
#define VM_T1  1.33333333333201242699e-01
#define VM_T2  5.39682539762260521377e-02
#define VM_T3  2.18694882948595424599e-02
#define VM_T4  8.86323982359930005737e-03
#define VM_T5  3.59207910759131235356e-03
#define VM_T6  1.45620945432529025516e-03
#define VM_T7  5.88041240820264096874e-04
#define VM_T8  2.46463134818469906812e-04
#define VM_T9  7.81794442939557092300e-05
#define VM_T10 7.14072491382608190305e-05
#define VM_T11 -1.85586374855275456654e-05
#define VM_T12 2.59073051863633712884e-05
#define VM_PIO4_HI 7.85398163397448278999e-01
#define VM_PIO4_LO 3.06161699786838301793e-17

/* Payne-Hanek reduction; used only for |x| > 2^19*pi/2 lanes.  */
extern double __ieee754_rem_pio2 (double, double *);

/* ------------------------------------------------------------------ */
/* VM_TRIG_RED(y0, y1, n, mbig, x, vl): reduced pair, quadrant count,
   and the scalar-fixup mask, per lane.  */
#define VM_TRIG_RED(y0o, y1o, no, mbigo, x, vl) do {				\
  vm_u64_t vmtr__bx = VM_BITS (x);					\
  vm_i64_t vmtr__hx = VM_VSRA_VX (VM_REINT_U_I (vmtr__bx), 32, (vl));	\
  vm_u64_t vmtr__ix = VM_VAND_VX (VM_REINT_I_U (vmtr__hx), 0x7fffffff, (vl)); \
  vm_b_t vmtr__mneg = VM_VMSLT_VX (vmtr__hx, 0, (vl));			\
  vm_b_t vmtr__mfast = VM_VMSLEU_VX (vmtr__ix, 0x3fe921fb, (vl));	\
  vm_b_t vmtr__mmed = VM_VMSLEU_VX (vmtr__ix, 0x413921fb, (vl));	\
  vm_b_t vmtr__minf = VM_VMSGEU_VX (vmtr__ix, 0x7ff00000, (vl));	\
  vm_b_t vmtr__mbig = VM_VMNOT_M (					\
    VM_VMOR_MM (VM_VMOR_MM (vmtr__mfast, vmtr__mmed, (vl)),		\
		vmtr__minf, (vl)), (vl));				\
  /* medium Cody-Waite on |x|.  inf/NaN lanes see 0.5:
     the ftz conversion would otherwise raise INVALID
     on NaN input.  */							\
  vm_f_t vmtr__xs = VM_VFMERGE_VFM ((x), 0.5,				\
				     VM_VMOR_MM (vmtr__minf, vmtr__mbig, (vl)), (vl)); \
  vm_f_t vmtr__t = VM_VFABS_V (vmtr__xs, (vl));				\
  vm_f_t vmtr__kf = VM_VFMACC_VF (VM_VFMV_V_F (VM_HALF, (vl)),		\
				  VM_INVPIO2, vmtr__t, (vl));		\
  vm_i64_t vmtr__ni = VM_VFCVT_RTZ_X_F_V (vmtr__kf, (vl));		\
  vm_f_t vmtr__fn = VM_VFCVT_F_X_V (vmtr__ni, (vl));			\
  vm_f_t vmtr__r = VM_VFNMSAC_VF (vmtr__t, VM_PIO2_1, vmtr__fn, (vl));	\
  vm_f_t vmtr__w = VM_VFMUL_VF (vmtr__fn, VM_PIO2_1T, (vl));		\
  vm_f_t vmtr__ra, vmtr__wa;						\
  {									\
    vm_f_t vmtr__t2, vmtr__w2, vmtr__r2, vmtr__w2f;			\
    /* round 2, unconditional.  fdlibm gates it on i>16; here it is what
       makes the pair exact.  t = r; w = fn*pio2_2;
       r = t-w; w = fn*pio2_2t - ((t-r) - w)  [subtracting w matters!] */ \
    vmtr__t2 = vmtr__r;							\
    vmtr__w2 = VM_VFMUL_VF (vmtr__fn, VM_PIO2_2, (vl));			\
    vmtr__r2 = VM_VFSUB_VV (vmtr__t2, vmtr__w2, (vl));			\
    vmtr__w2f = VM_VFSUB_VV (VM_VFMUL_VF (vmtr__fn, VM_PIO2_2T, (vl)),	\
			     VM_VFSUB_VV (VM_VFSUB_VV (vmtr__t2, vmtr__r2, (vl)), \
					  vmtr__w2, (vl)), (vl));	\
    vmtr__ra = vmtr__r2;						\
    vmtr__wa = vmtr__w2f;						\
    /* round 3 (fdlibm's i>49): rerunning from the rounded
       r2 drops the round-2 residual, so it pays off only once
       the reduced pair has lost > 49 bits to cancellation.  */	\
    {									\
      vm_f_t vmtr__y0a = VM_VFSUB_VV (vmtr__r2, vmtr__w2f, (vl));	\
      /* i = exp(|x|) - exp(y0); the u64 difference reinterprets to the
	 right signed value on underflow.  */				\
      vm_u64_t vmtr__j = VM_VSRL_VX (vmtr__ix, 20, (vl));		\
      vm_u64_t vmtr__ey = VM_VAND_VX (					\
	VM_VSRL_VX (VM_BITS (vmtr__y0a), 52, (vl)), 0x7ff, (vl));	\
      vm_b_t vmtr__m3 = VM_VMSGT_VX (					\
	VM_REINT_U_I (VM_VSUB_VV (vmtr__j, vmtr__ey, (vl))), 49, (vl));	\
      vm_f_t vmtr__t3 = vmtr__r2;					\
      vm_f_t vmtr__w3 = VM_VFMUL_VF (vmtr__fn, VM_PIO2_3, (vl));	\
      vm_f_t vmtr__r3 = VM_VFSUB_VV (vmtr__t3, vmtr__w3, (vl));		\
      vm_f_t vmtr__w3f = VM_VFSUB_VV (VM_VFMUL_VF (vmtr__fn, VM_PIO2_3T, (vl)), \
			     VM_VFSUB_VV (VM_VFSUB_VV (vmtr__t3, vmtr__r3, (vl)), \
					  vmtr__w3, (vl)), (vl));	\
      vmtr__ra = VM_VFMERGE_VVM (vmtr__r2, vmtr__r3, vmtr__m3, (vl));	\
      vmtr__wa = VM_VFMERGE_VVM (vmtr__w2f, vmtr__w3f, vmtr__m3, (vl));	\
    }									\
  }									\
  {									\
    vm_f_t vmtr__y0m = VM_VFSUB_VV (vmtr__ra, vmtr__wa, (vl));		\
    vm_f_t vmtr__y1m = VM_VFSUB_VV (					\
      VM_VFSUB_VV (vmtr__ra, vmtr__y0m, (vl)), vmtr__wa, (vl));		\
    vm_f_t vmtr__y0 = VM_VFMERGE_VVM (					\
      vmtr__y0m, VM_VFNEG_V (vmtr__y0m, (vl)), vmtr__mneg, (vl));	\
    vm_f_t vmtr__y1 = VM_VFMERGE_VVM (					\
      vmtr__y1m, VM_VFNEG_V (vmtr__y1m, (vl)), vmtr__mneg, (vl));	\
    vm_u64_t vmtr__nu = VM_REINT_I_U (vmtr__ni);			\
    vmtr__nu = VM_VMERGE_VVM (						\
      vmtr__nu, VM_VSUB_VV (VM_VMV_V_X (0, (vl)), vmtr__nu, (vl)),	\
      vmtr__mneg, (vl));						\
    /* fast lanes: (x, 0, 0) */						\
    (y0o) = VM_VFMERGE_VVM (vmtr__y0, (x), vmtr__mfast, (vl));		\
    (y1o) = VM_VFMERGE_VFM (vmtr__y1, 0.0, vmtr__mfast, (vl));		\
    vmtr__nu = VM_VMERGE_VVM (vmtr__nu, VM_VMV_V_X (0, (vl)),		\
			      vmtr__mfast, (vl));			\
    /* inf/NaN: x - x (invalid for inf/sNaN), n = 0 */			\
    {									\
      vm_f_t vmtr__xpx = VM_VFSUB_VV_M (vmtr__minf, (x), (x), (vl));	\
      (y0o) = VM_VFMERGE_VVM ((y0o), vmtr__xpx, vmtr__minf, (vl));	\
      (y1o) = VM_VFMERGE_VFM ((y1o), 0.0, vmtr__minf, (vl));		\
      vmtr__nu = VM_VMERGE_VVM (vmtr__nu, VM_VMV_V_X (0, (vl)),		\
				vmtr__minf, (vl));			\
    }									\
    /* large (rare): filler; the CALLER's scalar fixup patches these.  */ \
    (y0o) = VM_VFMERGE_VFM ((y0o), 0.1, vmtr__mbig, (vl));		\
    (y1o) = VM_VFMERGE_VFM ((y1o), 0.0, vmtr__mbig, (vl));		\
    vmtr__nu = VM_VMERGE_VVM (vmtr__nu, VM_VMV_V_X (0, (vl)),		\
			      vmtr__mbig, (vl));			\
    (no) = VM_REINT_U_I (vmtr__nu);					\
    (mbigo) = vmtr__mbig;						\
  }									\
} while (0)

/* ------------------------------------------------------------------ */
/* fdlibm kernels (vector) ---- */

/* __kernel_sin(x, y, 1): sin(x+y), y != 0
   x - ((z*(half*y - v*r) - y) - v*S1) */
static inline vm_f_t
VM_KFN (ksin) (vm_f_t x, vm_f_t y, size_t vl)
{
  /* __kernel_sin returns x for |x| < 2^-27; the merge keeps z from
     underflowing there.  */
  vm_u64_t ix = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 32, vl), 0x7fffffff, vl);
  vm_b_t m_tiny = VM_VMSLTU_VX (ix, 0x3e400000, vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 0.5, m_tiny, vl);
  vm_f_t z = VM_VFMUL_VV (xs, xs, vl);
  vm_f_t v = VM_VFMUL_VV (z, xs, vl);
  vm_f_t r = VM_VFMACC_VV (VM_VFMV_V_F (VM_S5, vl), z,
			   VM_VFMV_V_F (VM_S6, vl), vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_S4, vl), z, r, vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_S3, vl), z, r, vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_S2, vl), z, r, vl);
  vm_f_t t = VM_VFSUB_VV (VM_VFMUL_VF (y, VM_HALF, vl),
			  VM_VFMUL_VV (v, r, vl), vl);
  t = VM_VFSUB_VV (VM_VFMUL_VV (z, t, vl), y, vl);
  t = VM_VFSUB_VV (t, VM_VFMUL_VF (v, VM_S1, vl), vl);
  return VM_VFMERGE_VVM (VM_VFSUB_VV (xs, t, vl), x, m_tiny, vl);
}

/* __kernel_cos(x, y): cos(x+y) */
static inline vm_f_t
VM_KFN (kcos) (vm_f_t x, vm_f_t y, size_t vl)
{
  vm_u64_t ix = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 32, vl), 0x7fffffff, vl);
  vm_b_t m_tinyc = VM_VMSLTU_VX (ix, 0x3e400000, vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 0.5, m_tinyc, vl);
  vm_f_t z = VM_VFMUL_VV (xs, xs, vl);
  vm_f_t r = VM_VFMACC_VV (VM_VFMV_V_F (VM_C5, vl), z,
			   VM_VFMV_V_F (VM_C6, vl), vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_C4, vl), z, r, vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_C3, vl), z, r, vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_C2, vl), z, r, vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_C1, vl), z, r, vl);
  r = VM_VFMUL_VV (z, r, vl);		/* r = z*(C1+...) */
  /* z^2 * poly - x*y, with r already carrying one factor of z.  */
  vm_f_t t = VM_VFNMSAC_VV (VM_VFMUL_VV (z, r, vl), xs, y, vl);
  /* |x| < 0.3: 1 - (0.5*z - t) */
  vm_f_t y1 = VM_VFSUB_VV (VM_VFMV_V_F (1.0, vl),
			   VM_VFSUB_VV (VM_VFMUL_VF (z, 0.5, vl), t, vl), vl);
  /* else: qx = |x|>0.78125 ? 0.28125 : bits((ix-0x200000)<<32);
     1-qx - ((0.5*z-qx) - t) */
  /* Branch 2 needs ix >= 0x200000; on the other lanes the field
     arithmetic wraps (bits() of -inf/NaN) and the discarded y2 would go
     inf-inf (INVALID on subnormal inputs) -- feed those a benign 0.5.
     NOTE: VM_VMERGE_VVM(a, b, m) == m ? b : a -- ix is the TRUE operand.  */
  vm_b_t m03 = VM_VMSGTU_VX (ix, 0x3fd33333, vl);
  vm_u64_t ixs = VM_VMERGE_VVM (VM_VMV_V_X (0x3fe00000, vl), ix, m03, vl);
  vm_u64_t qxb = VM_VSLL_VX (VM_VSUB_VX (ixs, 0x200000, vl), 32, vl);
  vm_f_t qx = VM_FROM_BITS (qxb);
  qx = VM_VFMERGE_VFM (qx, 0.28125,
		       VM_VMSGTU_VX (ix, 0x3fe90000, vl), vl);
  vm_f_t y2 = VM_VFSUB_VV (
    VM_VFSUB_VV (VM_VFMV_V_F (1.0, vl), qx, vl),
    VM_VFSUB_VV (VM_VFSUB_VV (VM_VFMUL_VF (z, 0.5, vl), qx, vl), t, vl), vl);
  vm_f_t yv = VM_VFMERGE_VVM (y1, y2, m03, vl);
  /* |x| < 2^-27: exactly 1.0 (scalar's (int)x == 0 shortcut) */
  return VM_VFMERGE_VFM (yv, 1.0, m_tinyc, vl);
}

/* __kernel_tan(x, y, iy), iy = +-1 */
static inline vm_f_t
VM_KFN (ktan) (vm_f_t x, vm_f_t y, vm_i64_t iy, size_t vl)
{
  vm_u64_t bx = VM_BITS (x);
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (bx), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_b_t m_big = VM_VMSGEU_VX (ix, 0x3fe59428, vl);
  vm_b_t m_tiny = VM_VMSLTU_VX (ix, 0x3e300000, vl);
  vm_b_t m_neg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t m_iy1 = VM_VMSEQ_VX (VM_REINT_I_U (iy), 1, vl);
  /* |x| >= 0.6744: fold to X = pio4-|x| + (pio4lo-y).  The tail keeps the
     mirrored SIGN, not its magnitude: y can carry the opposite sign of x,
     and |y| there costs a whole ulp.  */
  vm_f_t zz = VM_VFRSUB_VF (VM_VFABS_V (x, vl), VM_PIO4_HI, vl);
  vm_f_t yd = VM_VFMERGE_VVM (y, VM_VFNEG_V (y, vl), m_neg, vl);
  vm_f_t ww = VM_VFRSUB_VF (yd, VM_PIO4_LO, vl);
  vm_f_t xr = VM_VFMERGE_VVM (x, VM_VFADD_VV (zz, ww, vl), m_big, vl);
  vm_f_t yr = VM_VFMERGE_VFM (y, 0.0, m_big, vl);
  /* |x| < 2^-28 leaves through fdlibm's early exit, where x*x would
     underflow (spurious UNDERFLOW) -- feed the polynomial a benign 0.5.  */
  vm_f_t xt = VM_VFMERGE_VFM (xr, 0.5, m_tiny, vl);
  vm_f_t yt = VM_VFMERGE_VFM (yr, 0.0, m_tiny, vl);
  vm_f_t z = VM_VFMUL_VV (xt, xt, vl);
  vm_f_t w = VM_VFMUL_VV (z, z, vl);
  vm_f_t r = VM_VFMACC_VV (VM_VFMV_V_F (VM_T9, vl), w,
			   VM_VFMV_V_F (VM_T11, vl), vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_T7, vl), w, r, vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_T5, vl), w, r, vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_T3, vl), w, r, vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (VM_T1, vl), w, r, vl);
  vm_f_t v = VM_VFMACC_VV (VM_VFMV_V_F (VM_T10, vl), w,
			   VM_VFMV_V_F (VM_T12, vl), vl);
  v = VM_VFMACC_VV (VM_VFMV_V_F (VM_T8, vl), w, v, vl);
  v = VM_VFMACC_VV (VM_VFMV_V_F (VM_T6, vl), w, v, vl);
  v = VM_VFMACC_VV (VM_VFMV_V_F (VM_T4, vl), w, v, vl);
  v = VM_VFMACC_VV (VM_VFMV_V_F (VM_T2, vl), w, v, vl);
  v = VM_VFMUL_VV (z, v, vl);
  vm_f_t s = VM_VFMUL_VV (z, xt, vl);
  /* r = y + z*(s*(r+v) + y); r += T0*s */
  r = VM_VFADD_VV (yt,
		   VM_VFMUL_VV (z,
				VM_VFADD_VV (VM_VFMUL_VV (s,
					VM_VFADD_VV (r, v, vl), vl), yt, vl), vl), vl);
  r = VM_VFMACC_VV (r, s, VM_VFMV_V_F (VM_T0, vl), vl);
  w = VM_VFADD_VV (xt, r, vl);
  /* iy == -1: accurate -1/(x+r).  Tiny lanes use fdlibm's own early-exit
     argument w = x + y (and its tail y) instead of the polynomial.  */
  {
    vm_f_t wr = VM_VFMERGE_VVM (w, VM_VFADD_VV (x, y, vl), m_tiny, vl);
    vm_f_t vr = VM_VFMERGE_VVM (r, y, m_tiny, vl);
    /* fdlibm divides on neither the tiny iy == +1 lanes nor the
       |x| >= 0.6744 fold, so park those denominators at 1.0 to keep the
       shared -1/w below from raising a flag.  w == 0 with iy == -1 is the
       tanpi(n + 1/2) pole: there the division MUST run.  */
    vm_b_t m_safe = VM_VMOR_MM (VM_VMAND_MM (m_tiny, m_iy1, vl), m_big, vl);
    vm_f_t wq = VM_VFMERGE_VFM (wr, 1.0, m_safe, vl);
    /* w == 0 (iy == -1) is the tanpi pole: the accurate form below would
       go inf-inf, so give it a benign denominator and patch at the end.  */
    vm_b_t m_wz = VM_VMFEQ_VV (wr, VM_VFMV_V_F (0.0, vl), vl);
    vm_f_t wg = VM_VFMERGE_VFM (wr, 1.0, VM_VMOR_MM (m_safe, m_wz, vl), vl);
    vm_f_t zi = VM_FROM_BITS (VM_VAND_VX (VM_BITS (wg),
					  0xffffffff00000000ULL, vl));
    vm_f_t vi = VM_VFSUB_VV (vr, VM_VFSUB_VV (zi, xr, vl), vl);
    vm_f_t ai = VM_VFRDIV_VF (wq, -1.0, vl);			/* -1/w */
    vm_f_t ti = VM_FROM_BITS (VM_VAND_VX (VM_BITS (
			     VM_VFMERGE_VFM (ai, 1.0, m_wz, vl)),
					  0xffffffff00000000ULL, vl));
    vm_f_t si = VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), ti, zi, vl);
    /* fdlibm: t + a*(s + t*v), t = hi(w) of a = -1/w.  The leading t (not
       a) is what makes the truncation error cancel.  */
    vm_f_t yinv = VM_VFMACC_VV (ti, ai,
				VM_VFMACC_VV (si, ti, vi, vl), vl);
    /* zero w with iy == -1: -1/w is the answer, flag and all.  */
    yinv = VM_VFMERGE_VVM (yinv, ai,
			   VM_VMAND_MM (m_wz, VM_VMNOT_M (m_iy1, vl), vl), vl);
    vm_f_t ygen0 = VM_VFMERGE_VVM (yinv, w, m_iy1, vl);
    /* |x| >= 0.6744: sign*(iy - 2*(x - (w*w/(w+iy) - r))) */
    vm_f_t vf = VM_VFCVT_F_X_V (iy, vl);
    vm_f_t wq2 = VM_VFDIV_VV (VM_VFMUL_VV (w, w, vl),
			      VM_VFADD_VV (w, vf, vl), vl);
    vm_f_t arg = VM_VFSUB_VV (xr, VM_VFSUB_VV (wq2, r, vl), vl);
    vm_f_t expr = VM_VFSUB_VV (vf, VM_VFMUL_VF (arg, 2.0, vl), vl);
    expr = VM_VFMERGE_VVM (expr, VM_VFNEG_V (expr, vl), m_neg, vl);
    ygen0 = VM_VFMERGE_VVM (ygen0, expr, m_big, vl);
    /* |x| < 2^-28 && iy == 1: exactly x (keeps tan(+-0) sign).  */
    return VM_VFMERGE_VVM (ygen0, x, VM_VMAND_MM (m_tiny, m_iy1, vl), vl);
  }
}

/* ------------------------------------------------------------------ */
/* *pi family (SYCL / C23).  No Payne-Hanek is ever needed: with
   q = rint(2x) the reduced r = x - q/2 is exact for every double
   (|r| <= 1/4), so the mod-2 reduction is exact and pi*r is a
   double-double product fed to the same ksin/kcos/ktan kernels.
   |x| >= 2^62 is always an even integer and 2x no longer fits the i64
   round trip, so those lanes are forced to r = 0.

   Zero/pole semantics (OpenCL C 3.0, which SYCL defers to), patched
   after the quadrant select:
     sinpi(+-0) = +-0, sinpi(+n) = +0, sinpi(-n) = -0 (copysign(0,x))
     cospi(+-0) = 1, cospi(n) = (-1)^n, cospi(n+1/2) = +0
     tanpi(+-0) = +-0, tanpi(n) = copysign(0, x)*(-1)^n,
     tanpi(n+1/2) = (-1)^n * +Inf (n = floor(x)), divbyzero
     *pi(+-Inf) = NaN, invalid  */

/* inf/NaN test that never raises (vmfgeu reads the exponent field).  */
#define VM_NONFIN_M(x, vl) VM_VMSGEU_VX (				\
  VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 32, (vl)), 0x7fffffff, (vl)),	\
  0x7ff00000, (vl))

/* Mask of every lane, for the callers that must let the pi*r product
   raise UNDERFLOW on a subnormal r.  */
#define VM_PI_ALL(vl) VM_VMSEQ_VX (VM_VMV_V_X (0, (vl)), 0, (vl))

#define VM_PI_RED(y0o, y1o, ro, qo, fmask, x, vl) do {			\
  vm_f_t vmpi__x = (x);							\
  vm_b_t vmpi__mnf = VM_NONFIN_M (vmpi__x, (vl));			\
  vm_b_t vmpi__m62 = VM_VMSGEU_VX (VM_VAND_VX (			\
    VM_VSRL_VX (VM_BITS (vmpi__x), 32, (vl)), 0x7fffffff, (vl)),	\
    0x43d00000, (vl));		/* |x| >= 2^62: 2x would hit 2^63 */	\
  /* inf/NaN and 2^62+ lanes see 0: the f64->i64 convert would saturate
     otherwise.  */							\
  vm_f_t vmpi__xs = VM_VFMERGE_VFM (vmpi__x, 0.0,			\
				    VM_VMOR_MM (vmpi__mnf, vmpi__m62, (vl)), (vl)); \
  /* q = rint(2x), then r = x - q/2 (exact: |r| <= 1/4, Sterbenz).  */	\
  vm_i64_t vmpi__q = VM_VFCVT_X_F_V_RM (VM_VFMUL_VF (vmpi__xs, 2.0, (vl)), \
					__RISCV_FRM_RNE, (vl));		\
  vm_f_t vmpi__r = VM_VFSUB_VV (vmpi__xs, VM_VFMUL_VF (			\
    VM_VFCVT_F_X_V (vmpi__q, (vl)), 0.5, (vl)), (vl));			\
  /* pi*r comes out of a 2^60-scaled evaluation: every intermediate is
     normal, one rounding happens, and the scale back is exact unless the
     result is subnormal -- precisely when UNDERFLOW belongs.  */	\
  /* The unscaled product below is the UNDERFLOW carrier: it is
     inexact exactly when pi*r landed below DBL_MIN.  Callers whose
     value never depends on pi*r (cospi's +-1/+-0 lanes) mask it off.  */	\
  vm_f_t vmpi__p = VM_VFMUL_VF_MU ((fmask), VM_VFMV_V_F (0.0, (vl)),	\
				   vmpi__r, VM_PI_HI, (vl));		\
  {									\
    vm_f_t vmpi__rs = VM_VFMUL_VF (vmpi__r, 0x1p60, (vl));		\
    vm_f_t vmpi__qs = VM_VFMUL_VF_MU ((fmask), VM_VFMV_V_F (0.0, (vl)),	\
				      vmpi__rs, VM_PI_LO, (vl));	\
    vm_f_t vmpi__hs = VM_VFMACC_VF (vmpi__qs, VM_PI_HI, vmpi__rs, (vl)); \
    vm_f_t vmpi__y0 = VM_VFMUL_VF_MU ((fmask), VM_VFMV_V_F (0.0, (vl)),	\
				      vmpi__hs, 0x1p-60, (vl));		\
    /* tail, relative to y0 (p - y0 is exact by Sterbenz).  Unread
       below 2^-300, where the kernels short-circuit.  */			\
    vm_b_t vmpi__mv = VM_VMSLTU_VX (VM_VAND_VX (VM_VSRL_VX (		\
      VM_BITS (vmpi__r), 32, (vl)), 0x7fffffff, (vl)), 0x2d300000, (vl)); \
    vm_f_t vmpi__rc = VM_VFMERGE_VFM (vmpi__r, 0.5, vmpi__mv, (vl));	\
    vm_f_t vmpi__e = VM_VFMSAC_VF (vmpi__p, VM_PI_HI, vmpi__rc, (vl));	\
    (y0o) = vmpi__y0;							\
    (y1o) = VM_VFMERGE_VFM (						\
      VM_VFADD_VV (VM_VFSUB_VV (vmpi__p, vmpi__y0, (vl)),		\
		   VM_VFMACC_VF (vmpi__e, VM_PI_LO, vmpi__rc, (vl)), (vl)), \
      0.0, vmpi__mv, (vl));						\
  }									\
  (ro) = vmpi__r;							\
  (qo) = vmpi__q;							\
} while (0)

/* Exact-lane patches, shared by the sincospi and tanpi kernels:
     m_int  -- x is an integer (cospi(n) = (-1)^n, sinpi(n) = +-0).
     m_pole -- x = n + 1/2 (cospi is +0, tanpi is (-1)^n * +Inf).  */
#define VM_PI_EXACT_MASKS(minto, mpoleo, ro, q, vl) do {		\
  vm_b_t vmpi__me = VM_VMFEQ_VV ((ro), VM_VFMV_V_F (0.0, (vl)), (vl));	\
  vm_b_t vmpi__mo = VM_VMSNE_VX (VM_VAND_VX ((q), 1, (vl)), 0, (vl));	\
  (minto) = VM_VMAND_MM (vmpi__me, VM_VMNOT_M (vmpi__mo, (vl)), (vl));	\
  (mpoleo) = VM_VMAND_MM (vmpi__me, vmpi__mo, (vl));			\
} while (0)

/* sinpi/cospi pair from one exact reduction.  */
#define VM_SINCOSPI(so, co, x, fmask, vl) do {					\
  vm_f_t vmsp__y0, vmsp__y1, vmsp__r;					\
  vm_i64_t vmsp__q64;							\
  VM_PI_RED (vmsp__y0, vmsp__y1, vmsp__r, vmsp__q64, (fmask), (x), (vl));	\
  {									\
    vm_u64_t vmsp__q = VM_VAND_VX (VM_REINT_I_U (vmsp__q64), 3, (vl));	\
    vm_f_t vmsp__s = VM_KFN (ksin) (vmsp__y0, vmsp__y1, (vl));		\
    vm_f_t vmsp__c = VM_KFN (kcos) (vmsp__y0, vmsp__y1, (vl));		\
    vm_f_t vmsp__ns = VM_VFNEG_V (vmsp__s, (vl));			\
    vm_f_t vmsp__nc = VM_VFNEG_V (vmsp__c, (vl));			\
    vm_f_t vmsp__sn = VM_VFMV_V_F (0.0, (vl));				\
    vm_f_t vmsp__cn = VM_VFMV_V_F (0.0, (vl));				\
    vmsp__sn = VM_VFMERGE_VVM (vmsp__sn, vmsp__s, VM_VMSEQ_VX (vmsp__q, 0, (vl)), (vl)); \
    vmsp__sn = VM_VFMERGE_VVM (vmsp__sn, vmsp__c, VM_VMSEQ_VX (vmsp__q, 1, (vl)), (vl)); \
    vmsp__sn = VM_VFMERGE_VVM (vmsp__sn, vmsp__ns, VM_VMSEQ_VX (vmsp__q, 2, (vl)), (vl)); \
    vmsp__sn = VM_VFMERGE_VVM (vmsp__sn, vmsp__nc, VM_VMSEQ_VX (vmsp__q, 3, (vl)), (vl)); \
    vmsp__cn = VM_VFMERGE_VVM (vmsp__cn, vmsp__c, VM_VMSEQ_VX (vmsp__q, 0, (vl)), (vl)); \
    vmsp__cn = VM_VFMERGE_VVM (vmsp__cn, vmsp__ns, VM_VMSEQ_VX (vmsp__q, 1, (vl)), (vl)); \
    vmsp__cn = VM_VFMERGE_VVM (vmsp__cn, vmsp__nc, VM_VMSEQ_VX (vmsp__q, 2, (vl)), (vl)); \
    vmsp__cn = VM_VFMERGE_VVM (vmsp__cn, vmsp__s, VM_VMSEQ_VX (vmsp__q, 3, (vl)), (vl)); \
    {									\
      vm_b_t vmsp__mint, vmsp__mpole;					\
      VM_PI_EXACT_MASKS (vmsp__mint, vmsp__mpole, vmsp__r, vmsp__q, (vl)); \
      /* sinpi(+-n) = +-0: copysign(0, x), with NO parity factor.  */	\
	vm_f_t vmsp__zi = VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, (vl)), (x), (vl)); \
	vmsp__sn = VM_VFMERGE_VVM (vmsp__sn, vmsp__zi, vmsp__mint, (vl)); \
	vmsp__cn = VM_VFMERGE_VFM (vmsp__cn, 0.0, vmsp__mpole, (vl));	\
      }									\
    {									\
      vm_b_t vmsp__mnf = VM_NONFIN_M ((x), (vl));			\
      vm_f_t vmsp__nn = VM_VFSUB_VV_M (vmsp__mnf, (x), (x), (vl));	\
      vmsp__sn = VM_VFMERGE_VVM (vmsp__sn, vmsp__nn, vmsp__mnf, (vl));	\
      vmsp__cn = VM_VFMERGE_VVM (vmsp__cn, vmsp__nn, vmsp__mnf, (vl));	\
    }									\
    (so) = vmsp__sn;							\
    (co) = vmsp__cn;							\
  }									\
} while (0)

static inline vm_f_t
VM_KFN (tanpi) (vm_f_t x, size_t vl)
{
  vm_f_t y0, y1, r;
  vm_i64_t q64;
  VM_PI_RED (y0, y1, r, q64, VM_PI_ALL (vl), x, vl);
  vm_u64_t q = VM_VAND_VX (VM_REINT_I_U (q64), 3, vl);
  vm_b_t m_odd = VM_VMSNE_VX (VM_VAND_VX (q, 1, vl), 0, vl);
  vm_i64_t iy = VM_REINT_U_I (VM_VMERGE_VXM (VM_VMV_V_X (1, vl), -1,
					     m_odd, vl));
  vm_f_t t = VM_KFN (ktan) (y0, y1, iy, vl);
  {
    vm_b_t m_int, m_pole;
    VM_PI_EXACT_MASKS (m_int, m_pole, r, q, vl);
    {
      /* q&2 is the parity of n = q/2, i.e. (-1)^n.  */
      vm_b_t m_q2 = VM_VMSNE_VX (VM_VAND_VX (q, 2, vl), 0, vl);
      vm_f_t pm = VM_VFMERGE_VFM (VM_VFMV_V_F (1.0, vl), -1.0, m_q2, vl);
      vm_f_t zi = VM_VFMUL_VV (VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, vl), x, vl),
			       pm, vl);
      vm_f_t inf = VM_VINF (vl);
      vm_f_t pv = VM_VFMERGE_VVM (inf, VM_VFNEG_V (inf, vl), m_q2, vl);
      t = VM_VFMERGE_VVM (t, zi, m_int, vl);
      t = VM_VFMERGE_VVM (t, pv, m_pole, vl);
    }
  }
  {
    vm_b_t mnf = VM_NONFIN_M (x, vl);
    t = VM_VFMERGE_VVM (t, VM_VFSUB_VV_M (mnf, x, x, vl), mnf, vl);
  }
  return t;
}

static inline vm_f_t
VM_KFN (sinpi) (vm_f_t x, size_t vl)
{
  vm_f_t s, c;
  VM_SINCOSPI (s, c, x, VM_PI_ALL (vl), vl);
  return s;
}

static inline vm_f_t
VM_KFN (cospi) (vm_f_t x, size_t vl)
{
  vm_f_t s, c;
  /* |x| < 2^-29 stays in quadrant 0, where cospi is exactly 1: keep the
     pi*r product (the only op that can raise UNDERFLOW) off those.  */
  vm_b_t msm = VM_VMSLTU_VX (VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 32, (vl)),
					 0x7fffffff, (vl)), 0x3e300000, (vl));
  VM_SINCOSPI (s, c, x, VM_VMNOT_M (msm, (vl)), vl);
  return c;
}

/* ------------------------------------------------------------------ */
/* public kernels -- one reduction per call.  */

static inline vm_f_t
VM_KFN (sin) (vm_f_t x, size_t vl)
{
  vm_f_t y0, y1;
  vm_i64_t n;
  vm_b_t m_big;
  VM_TRIG_RED (y0, y1, n, m_big, x, vl);
  vm_f_t s = VM_KFN (ksin) (y0, y1, vl);
  vm_f_t c = VM_KFN (kcos) (y0, y1, vl);
  vm_u64_t q = VM_VAND_VX (VM_REINT_I_U (n), 3, vl);
  vm_f_t y = VM_VFMV_V_F (0.0, vl);
  y = VM_VFMERGE_VVM (y, s, VM_VMSEQ_VX (q, 0, vl), vl);
  y = VM_VFMERGE_VVM (y, c, VM_VMSEQ_VX (q, 1, vl), vl);
  y = VM_VFMERGE_VVM (y, VM_VFNEG_V (s, vl), VM_VMSEQ_VX (q, 2, vl), vl);
  y = VM_VFMERGE_VVM (y, VM_VFNEG_V (c, vl), VM_VMSEQ_VX (q, 3, vl), vl);
  return VM_KFN (scalar_fixup) (x, y, m_big, vl, sin);
}

static inline vm_f_t
VM_KFN (cos) (vm_f_t x, size_t vl)
{
  vm_f_t y0, y1;
  vm_i64_t n;
  vm_b_t m_big;
  VM_TRIG_RED (y0, y1, n, m_big, x, vl);
  vm_f_t s = VM_KFN (ksin) (y0, y1, vl);
  vm_f_t c = VM_KFN (kcos) (y0, y1, vl);
  vm_u64_t q = VM_VAND_VX (VM_REINT_I_U (n), 3, vl);
  vm_f_t y = VM_VFMV_V_F (0.0, vl);
  y = VM_VFMERGE_VVM (y, c, VM_VMSEQ_VX (q, 0, vl), vl);
  y = VM_VFMERGE_VVM (y, VM_VFNEG_V (s, vl), VM_VMSEQ_VX (q, 1, vl), vl);
  y = VM_VFMERGE_VVM (y, VM_VFNEG_V (c, vl), VM_VMSEQ_VX (q, 2, vl), vl);
  y = VM_VFMERGE_VVM (y, s, VM_VMSEQ_VX (q, 3, vl), vl);
  return VM_KFN (scalar_fixup) (x, y, m_big, vl, cos);
}

static inline vm_f_t
VM_KFN (tan) (vm_f_t x, size_t vl)
{
  vm_f_t y0, y1;
  vm_i64_t n;
  vm_b_t m_big;
  VM_TRIG_RED (y0, y1, n, m_big, x, vl);
  vm_u64_t q = VM_VAND_VX (VM_REINT_I_U (n), 1, vl);
  vm_b_t m_odd = VM_VMSNE_VX (q, 0, vl);
  vm_i64_t iy = VM_REINT_U_I (VM_VMERGE_VXM (VM_VMV_V_X (1, vl), -1,
					     m_odd, vl));
  return VM_KFN (scalar_fixup) (x, VM_KFN (ktan) (y0, y1, iy, vl),
				m_big, vl, tan);
}

/* sincos: two outputs from one reduction (macro writes sy and cy).  */
#define VM_SINCOS(sy, cy, x, vl) do {					\
  vm_f_t vmsc__y0, vmsc__y1;						\
  vm_i64_t vmsc__n;							\
  vm_b_t vmsc__mbig;							\
  VM_TRIG_RED (vmsc__y0, vmsc__y1, vmsc__n, vmsc__mbig, (x), (vl));	\
  {									\
    vm_f_t vmsc__s = VM_KFN (ksin) (vmsc__y0, vmsc__y1, (vl));		\
    vm_f_t vmsc__c = VM_KFN (kcos) (vmsc__y0, vmsc__y1, (vl));		\
    vm_u64_t vmsc__q = VM_VAND_VX (VM_REINT_I_U (vmsc__n), 3, (vl));	\
    vm_f_t vmsc__sn = VM_VFMV_V_F (0.0, (vl));				\
    vm_f_t vmsc__cn = VM_VFMV_V_F (0.0, (vl));						\
    vmsc__sn = VM_VFMERGE_VVM (vmsc__sn, vmsc__s, VM_VMSEQ_VX (vmsc__q, 0, (vl)), (vl)); \
    vmsc__sn = VM_VFMERGE_VVM (vmsc__sn, vmsc__c, VM_VMSEQ_VX (vmsc__q, 1, (vl)), (vl)); \
    vmsc__sn = VM_VFMERGE_VVM (vmsc__sn, VM_VFNEG_V (vmsc__s, (vl)), VM_VMSEQ_VX (vmsc__q, 2, (vl)), (vl)); \
    vmsc__sn = VM_VFMERGE_VVM (vmsc__sn, VM_VFNEG_V (vmsc__c, (vl)), VM_VMSEQ_VX (vmsc__q, 3, (vl)), (vl)); \
    vmsc__cn = VM_VFMERGE_VVM (vmsc__cn, vmsc__c, VM_VMSEQ_VX (vmsc__q, 0, (vl)), (vl)); \
    vmsc__cn = VM_VFMERGE_VVM (vmsc__cn, VM_VFNEG_V (vmsc__s, (vl)), VM_VMSEQ_VX (vmsc__q, 1, (vl)), (vl)); \
    vmsc__cn = VM_VFMERGE_VVM (vmsc__cn, VM_VFNEG_V (vmsc__c, (vl)), VM_VMSEQ_VX (vmsc__q, 2, (vl)), (vl)); \
    vmsc__cn = VM_VFMERGE_VVM (vmsc__cn, vmsc__s, VM_VMSEQ_VX (vmsc__q, 3, (vl)), (vl)); \
    (sy) = VM_KFN (scalar_fixup) ((x), vmsc__sn, vmsc__mbig, (vl), sin); \
    (cy) = VM_KFN (scalar_fixup) ((x), vmsc__cn, vmsc__mbig, (vl), cos); \
  }									\
} while (0)
