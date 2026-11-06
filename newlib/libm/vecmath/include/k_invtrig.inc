/* k_invtrig.inc -- atan / atan2 kernels and their pi-scaled variants.

   LMUL-generic template: include after vecmath_rvvcompat.h,
   vecmath_dd.h and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.

   Ports fdlibm math/s_atan.c and math/e_atan2.c; s_asin.c supplies the
   asin/acos rationals.  Branch-free: every reduction is evaluated for
   all lanes and selected by mask.  The four atan reductions share the
   shape r = (|x| - c) / (1 + c*|x|) for c = 0.5, 1.0, 1.5 (fdlibm
   writes (2x-1)/(2+x) for c = 0.5, the same quotient), so they use one
   vector division with a per-lane c; only the |x| >= 2.4375 branch
   (r = -1/|x|) needs a second.  atan2(y, x) is atan(|y/x|) plus the
   quadrant fixup, handing atan_k the division's exact residual as a
   tail because the quotient is rounded.  */

/* ---- atan constants (s_atan.c) ---- */
#define VM_ATANHI0 4.63647609000806093515e-01
#define VM_ATANHI1 7.85398163397448278999e-01
#define VM_ATANHI2 9.82793723247329054082e-01
#define VM_ATANHI3 1.57079632679489655800e+00
#define VM_ATANLO0 2.26987774529616870924e-17
#define VM_ATANLO1 3.06161699786838301793e-17
#define VM_ATANLO2 1.39033110312309984516e-17
#define VM_ATANLO3 6.12323399573676603587e-17
#define VM_AT0 3.33333333333329318027e-01
#define VM_AT1 -1.99999999998764832476e-01
#define VM_AT2 1.42857142725034663711e-01
#define VM_AT3 -1.11111104054623557880e-01
#define VM_AT4 9.09088713343650656196e-02
#define VM_AT5 -7.69187620504482999495e-02
#define VM_AT6 6.66107313738753120669e-02
#define VM_AT7 -5.83357013379057348645e-02
#define VM_AT8 4.97687799461593236017e-02
#define VM_AT9 -3.65315727442169155270e-02
#define VM_AT10 1.62858201153657823623e-02

/* ---- atan2 constants (e_atan2.c) ---- */
#define VM_ATAN2_PIO4 7.8539816339744827900e-01
#define VM_ATAN2_PIO2 1.5707963267948965580e+00
#define VM_ATAN2_PI 3.1415926535897931160e+00
#define VM_ATAN2_PILO 1.2246467991473531772e-16

/* Collapse a dd pair without folding an exact signed zero: -0 + +0 is
   +0, and atan(+-0) must stay +-0.  Where lo is exactly zero the
   value *is* hi.  vmfeq is quiet, so a NaN lane raises nothing.  */
#define VM_INVTRIG_COLLAPSE(r, ah, al, vl) do { \
  (r) = VM_VFADD_VV ((ah), (al), (vl)); \
  (r) = VM_VFMERGE_VVM ((r), (ah), VM_VMFEQ_VF ((al), 0.0, (vl)), (vl)); \
} while (0)

/* atan(x + xlo) as a dd pair.  (yh, yl) is the result before its final
   rounding: atan() collapses it and is done, but the *pi variants and
   atan2() cannot, since their own scaling would otherwise add half an
   ulp on top of that rounding.  |xlo| is at most a fraction of an ulp
   of x, and zero for plain atan.  */
#define VM_ATAN_K_DD(yh, yl, x, xlo, vl) do { \
  vm_u64_t bx = VM_BITS (x); \
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (bx), 32, vl); \
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl); \
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl); \
  vm_b_t mbig = VM_VMSGEU_VX (ix, 0x44100000, vl);	/* |x| >= 2^66 */ \
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3e200000, vl);	/* |x| < 2^-29 */ \
  vm_b_t msmall = VM_VMSLTU_VX (ix, 0x3fdc0000, vl);	/* |x| < 0.4375 */ \
  vm_b_t mlt1 = VM_VMSLTU_VX (ix, 0x3fe60000, vl); \
  vm_b_t mlt2 = VM_VMSLTU_VX (ix, 0x3ff30000, vl); \
  vm_b_t mlt3 = VM_VMSLTU_VX (ix, 0x40038000, vl); \
  vm_b_t ml = VM_VMNOT_M (msmall, vl); \
  vm_b_t mid0 = VM_VMAND_MM (ml, mlt1, vl); \
  vm_b_t mid1 = VM_VMAND_MM (ml, VM_VMAND_MM (VM_VMNOT_M (mlt1, vl), mlt2, vl), \
			     vl); \
  vm_b_t mid2 = VM_VMAND_MM (ml, VM_VMAND_MM (VM_VMNOT_M (mlt2, vl), mlt3, vl), \
			     vl); \
  vm_b_t mid3 = VM_VMAND_MM (ml, VM_VMNOT_M (mlt3, vl), vl); \
  vm_f_t ax = VM_VFABS_V (x, vl); \
  /* Two lane classes must not feed the polynomial or the -1/|x|: \
       |x| < 2^-29  -- fdlibm returns x, and both x*x and 1/x would \
                       underflow or overflow; \
       |x| >= 2^66  -- fdlibm returns pi/2 first, and inf - inf in the \
                       polynomial plus an underflowing 1/x would raise \
                       flags fdlibm never does.  */ \
  vm_b_t mguard = VM_VMOR_MM (mtiny, mbig, vl); \
  vm_f_t ag = VM_VFMERGE_VFM (ax, 0.5, mguard, vl); \
  vm_f_t c = VM_VFMV_V_F (0.0, vl); \
  c = VM_VFMERGE_VFM (c, 1.5, mid2, vl); \
  c = VM_VFMERGE_VFM (c, 1.0, mid1, vl); \
  c = VM_VFMERGE_VFM (c, 0.5, mid0, vl); \
  vm_f_t rden = VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), c, ag, vl); \
  vm_f_t rnum = VM_VFSUB_VV (ag, c, vl); \
  vm_f_t r = VM_VFDIV_VV (rnum, rden, vl); \
  /* rr = r - (the exact reduced argument).  Without it the dd could \
     only undo the last rounding, leaving atan's own error plus one \
     more rounding in the *pi variants.  */ \
  vm_f_t rr = VM_VFDIV_VV ( \
    VM_VFMSAC_VV (rnum, r, rden, vl), rden, vl); \
  { \
    /* r = -1/ag on the last branch; r + 1/ag == (r*ag + 1)/ag.  */ \
    vm_f_t r3 = VM_VFRDIV_VF (ag, -1.0, vl); \
    vm_f_t rr3 = VM_VFDIV_VV ( \
      VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), r3, ag, vl), ag, vl); \
    r = VM_VFMERGE_VVM (r, r3, mid3, vl); \
    rr = VM_VFMERGE_VVM (rr, rr3, mid3, vl); \
  } \
  vm_f_t z = VM_VFMUL_VV (r, r, vl); \
  vm_f_t w = VM_VFMUL_VV (z, z, vl); \
  vm_f_t t = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT8, vl), w, \
			   VM_VFMV_V_F (VM_AT10, vl), vl); \
  t = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT6, vl), w, t, vl); \
  t = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT4, vl), w, t, vl); \
  t = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT2, vl), w, t, vl); \
  t = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT0, vl), w, t, vl); \
  vm_f_t s1 = VM_VFMUL_VV (z, t, vl); \
  vm_f_t u = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT7, vl), w, \
			   VM_VFMV_V_F (VM_AT9, vl), vl); \
  u = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT5, vl), w, u, vl); \
  u = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT3, vl), w, u, vl); \
  u = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT1, vl), w, u, vl); \
  vm_f_t ss = VM_VFADD_VV (s1, VM_VFMUL_VV (w, u, vl), vl); \
  vm_f_t hi = VM_VFMV_V_F (0.0, vl); \
  vm_f_t lo = VM_VFMV_V_F (0.0, vl); \
  hi = VM_VFMERGE_VFM (hi, VM_ATANHI3, mid3, vl); \
  hi = VM_VFMERGE_VFM (hi, VM_ATANHI2, mid2, vl); \
  hi = VM_VFMERGE_VFM (hi, VM_ATANHI1, mid1, vl); \
  hi = VM_VFMERGE_VFM (hi, VM_ATANHI0, mid0, vl); \
  lo = VM_VFMERGE_VFM (lo, VM_ATANLO3, mid3, vl); \
  lo = VM_VFMERGE_VFM (lo, VM_ATANLO2, mid2, vl); \
  lo = VM_VFMERGE_VFM (lo, VM_ATANLO1, mid1, vl); \
  lo = VM_VFMERGE_VFM (lo, VM_ATANLO0, mid0, vl); \
  /* Fold the tail in as r' = r + rlo, rlo = xlo * dr/dx.  dr/dx is \
     (1 + c^2)/(1 + c*x)^2, which equals (1 + r^2)/(1 + x^2) for all \
     four reductions -- including r = -1/x, where it is 1/x^2 -- so a \
     single expression covers them all.  r + rlo is split exactly, or \
     the correction (itself only a fraction of an ulp) would be rounded \
     away before reaching the result; it rides in on the `lo` term, so \
     the expression keeps fdlibm's cancellation-free shape and the \
     final two_sum carries it through.  */ \
  vm_f_t rlo = VM_VFMUL_VV ( \
    xlo, VM_VFDIV_VV (VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), r, r, vl), \
		      VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), ag, ag, vl), vl), vl); \
  vm_f_t rt, rte, rte0; \
  VM_TWO_SUM (rt, rte0, r, rlo, (vl)); \
  rte = VM_VFADD_VV (rte0, rr, vl); \
  /* hi - P, with P's own roundings carried as well.  Each subtraction \
     below is exact via two_sum and the residuals ride along in vmak_Plo, \
     so the pair is the computed value to full precision instead of one \
     rounding of it.  */ \
  vm_f_t vmak_A = VM_VFMUL_VV (rt, ss, vl); \
  vm_f_t vmak_Ae = VM_VFNEG_V (VM_VFNMSAC_VV (vmak_A, rt, ss, vl), vl); \
  vm_f_t vmak_B, vmak_Be, vmak_C, vmak_Ce; \
  VM_TWO_SUM (vmak_B, vmak_Be, vmak_A, VM_VFNEG_V (lo, vl), (vl)); \
  VM_TWO_SUM (vmak_C, vmak_Ce, vmak_B, VM_VFNEG_V (rt, vl), (vl)); \
  vm_f_t vmak_Plo = VM_VFSUB_VV ( \
    VM_VFADD_VV (VM_VFADD_VV (vmak_Ce, vmak_Be, vl), vmak_Ae, vl), rte, vl); \
  vm_f_t vmak_y, vmak_yl; \
  VM_TWO_SUM (vmak_y, vmak_yl, hi, VM_VFNEG_V (vmak_C, vl), (vl)); \
  vmak_yl = VM_VFSUB_VV (vmak_yl, vmak_Plo, vl); \
vmak_y = VM_VFMERGE_VVM (vmak_y, VM_VFNEG_V (vmak_y, vl), mneg, vl); \
  vmak_yl = VM_VFMERGE_VVM (vmak_yl, VM_VFNEG_V (vmak_yl, vl), mneg, vl); \
  /* fdlibm returns x here; the tail still has to ride along, or the \
     lane loses the half ulp atan2's quotient rounding put in.  */ \
  { \
    /* Only the |x| < 2^-29 lanes read this; the rest are fed 0.5 so
       the two_sum cannot put inf - inf through itself (INVALID on
       +-Inf, which fdlibm never raises there).  */ \
    vm_b_t vmak_mn = VM_VMNOT_M (mtiny, (vl)); \
    vm_f_t vmak_th, vmak_tl; \
    VM_TWO_SUM (vmak_th, vmak_tl, \
		VM_VFMERGE_VFM (x, 0.5, vmak_mn, (vl)), \
		VM_VFMERGE_VFM (xlo, 0.0, vmak_mn, (vl)), (vl)); \
    /* two_sum folds -0 + +0 into +0; atan(+-0) must stay +-0.  */ \
    vmak_th = VM_VFMERGE_VVM (vmak_th, x, \
			      VM_VMFEQ_VF (x, 0.0, (vl)), (vl)); \
    vmak_y = VM_VFMERGE_VVM (vmak_y, vmak_th, mtiny, vl); \
    vmak_yl = VM_VFMERGE_VVM (vmak_yl, vmak_tl, mtiny, vl); \
  } \
  { \
    /* |x| >= 2^66 (and +-Inf): atanhi[3] + atanlo[3].  */ \
    vm_f_t inf_val = VM_VFMV_V_F (VM_ATANHI3 + VM_ATANLO3, vl); \
    vmak_y = VM_VFMERGE_VVM (vmak_y, VM_VFMERGE_VVM (inf_val, \
						     VM_VFNEG_V (inf_val, vl), mneg, vl), \
						      mbig, vl); \
    vmak_yl = VM_VFMERGE_VFM (vmak_yl, 0.0, mbig, vl); \
  } \
  { \
    /* NaN (low word nonzero) -> x + x, quiet for qNaN, invalid for sNaN */ \
    vm_u64_t lx = VM_VAND_VX (bx, 0xffffffffULL, vl); \
    vm_b_t mnan = VM_VMOR_MM ( \
      VM_VMSGTU_VX (ix, 0x7ff00000, vl), \
      VM_VMAND_MM (VM_VMSEQ_VX (ix, 0x7ff00000, vl), \
		   VM_VMSNE_VX (lx, 0, vl), vl), vl); \
    /* masked: inf - inf would raise INVALID on the +-Inf lanes, which \
       fdlibm never evaluates (it returns pi/2 there).  */ \
    vmak_y = VM_VFMERGE_VVM (vmak_y, VM_VFSUB_VV_M (mnan, x, x, vl), mnan, vl); \
    vmak_yl = VM_VFMERGE_VFM (vmak_yl, 0.0, mnan, vl); \
  } \
  (yh) = vmak_y; \
  (yl) = vmak_yl; \
} while (0)

static inline vm_f_t
VM_KFN (atan_k) (vm_f_t x, vm_f_t xlo, size_t vl)
{
  vm_f_t yh, yl, yr;
  VM_ATAN_K_DD (yh, yl, x, xlo, vl);
  VM_INVTRIG_COLLAPSE (yr, yh, yl, vl);
  return yr;
}


/* atan2(y, x) as a dd pair.  (oh, ol) is the result before its final
   rounding, which atan2pi needs because its own /pi would otherwise
   round a value that is already rounded.  */
#define VM_ATAN2_CORE_DD(oh, ol, osc, y, x, dyt, dxt, vl) do { \
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl); \
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl); \
  vm_u64_t ly = VM_VAND_VX (VM_BITS (y), 0xffffffffULL, vl); \
  vm_u64_t iy = VM_VAND_VX (VM_VSRL_VX (VM_BITS (y), 32, vl), 0x7fffffff, vl); \
  vm_b_t mxs = VM_VMSLT_VX (hx, 0, vl);			/* sign(x) */ \
  vm_b_t mys = VM_VMSLT_VX (VM_REINT_U_I (VM_BITS (y)), 0, vl);	/* sign(y) */ \
  vm_b_t m0 = VM_VMAND_MM (VM_VMNOT_M (mxs, vl), VM_VMNOT_M (mys, vl), vl); \
  vm_b_t mb2 = VM_VMAND_MM (mxs, VM_VMNOT_M (mys, vl), vl); \
  vm_b_t mb1 = VM_VMAND_MM (VM_VMNOT_M (mxs, vl), mys, vl); \
  vm_b_t mb3 = VM_VMAND_MM (mxs, mys, vl); \
  vm_f_t vmc_oh = VM_VFMV_V_F (0.0, vl); \
  vm_f_t vmc_ol = VM_VFMV_V_F (0.0, vl); \
  /* set inside; this keeps it in scope for the (osc) result at the end */ \
  vm_b_t vmc_sc = VM_VMFEQ_VF (VM_VFMV_V_F (0.0, vl), 1.0, vl); \
  { \
    /* NaN in either argument -> x + y */ \
    vm_b_t mx = VM_VMOR_MM (VM_VMSGTU_VX (ix, 0x7ff00000, vl), \
			    VM_VMAND_MM (VM_VMSEQ_VX (ix, 0x7ff00000, vl), \
					 VM_VMSNE_VX (VM_VAND_VX (VM_BITS (x), \
								  0xffffffffULL, vl), \
						      0, vl), vl), vl); \
    vm_b_t my = VM_VMOR_MM (VM_VMSGTU_VX (iy, 0x7ff00000, vl), \
			    VM_VMAND_MM (VM_VMSEQ_VX (iy, 0x7ff00000, vl), \
					 VM_VMSNE_VX (ly, 0, vl), vl), vl); \
    vmc_oh = VM_VFMV_V_F (0.0, vl); \
    { \
      vm_b_t mnan = VM_VMOR_MM (mx, my, vl); \
      vm_b_t myzero = VM_VMSEQ_VX (iy, 0, vl);		/* iy == 0 */ \
      myzero = VM_VMAND_MM (myzero, VM_VMSEQ_VX (ly, 0, vl), vl); \
      vm_b_t mxzero = VM_VMAND_MM (VM_VMSEQ_VX (ix, 0, vl), \
				   VM_VMSEQ_VX (VM_VAND_VX (VM_BITS (x), \
							    0xffffffffULL, vl), \
						0, vl), vl); \
      vm_b_t mxinf = VM_VMAND_MM (VM_VMSEQ_VX (ix, 0x7ff00000, vl), \
				  VM_VMSEQ_VX (VM_VAND_VX (VM_BITS (x), \
							   0xffffffffULL, vl), \
					       0, vl), vl); \
      vm_b_t myinf = VM_VMAND_MM (VM_VMSEQ_VX (iy, 0x7ff00000, vl), \
				  VM_VMSEQ_VX (ly, 0, vl), vl); \
      /* general path: z = atan(|y/x|) with the two magnitude shortcuts */ \
      vm_i64_t k = VM_VSRA_VX ( \
	VM_REINT_U_I (VM_VSUB_VV (iy, ix, vl)), 20, vl); \
      vm_b_t mhi = VM_VMSGT_VX (k, 60, vl); \
      vm_b_t mlo = VM_VMAND_MM (mxs, VM_VMSLT_VX (k, -60, vl), vl); \
      /* fdlibm never forms y/x on the two magnitude-shortcut lanes (it \
	 substitutes a constant), and the quotient overflows or underflows \
	 there; feed those lanes 0/1 so the division stays quiet.  */ \
      /* y = +-Inf joins the shortcut set too: fdlibm answers it before \
	 ever forming y/x, and inf/x would put inf-inf through the residual \
	 fma below (NaN + INVALID, which fdlibm never raises there).  */ \
      vm_b_t mshort = VM_VMOR_MM ( \
	VM_VMOR_MM (VM_VMOR_MM (mhi, mlo, vl), mxinf, vl), myinf, vl); \
      vm_f_t qx = VM_VFMERGE_VFM (x, 1.0, \
				  VM_VMOR_MM (mxzero, mshort, vl), vl); \
      vm_f_t qy = VM_VFMERGE_VFM (y, 0.0, mshort, vl); \
      vm_f_t q = VM_VFDIV_VV (qy, qx, vl); \
      /* What the division threw away, exactly: the fma leaves \
	 qy == q*qx + rem unrounded, so rem/qx is the quotient's own \
	 error.  It is a tail on the magnitude, so it carries the sign \
	 of y/x (+1 exactly when x and y agree in sign). \
 \
	 Both halves sit near the bottom of the exponent range -- the \
	 residual scales with |y|, the tail with |q| -- so they are \
	 computed only where they stay normal and left at zero \
	 elsewhere.  Unmasked they raise UNDERFLOW on lanes whose \
	 quotient is perfectly civil (a subnormal |y|, or a |q| that is \
	 normal but near the boundary), which fdlibm's atan2 never \
	 raises.  The window is [2^-960, 2^900) on both, as a \
	 bit-pattern compare so no NaN can make it signal.  */ \
      vm_u64_t qab = VM_VAND_VX (VM_BITS (q), 0x7fffffffffffffffULL, vl); \
      vm_u64_t yab = VM_VAND_VX (VM_BITS (qy), 0x7fffffffffffffffULL, vl); \
      vm_b_t mtail = VM_VMAND_MM ( \
	VM_VMAND_MM (VM_VMSGEU_VX (qab, (uint64_t) (63ULL << 52), vl), \
		     VM_VMSLTU_VX (qab, (uint64_t) (1923ULL << 52), vl), vl), \
	VM_VMAND_MM (VM_VMSGEU_VX (yab, (uint64_t) (63ULL << 52), vl), \
		     VM_VMSLTU_VX (yab, (uint64_t) (1923ULL << 52), vl), vl), vl); \
      vm_f_t qlo = VM_VFDIV_VV_MU ( \
	mtail, VM_VFMV_V_F (0.0, vl), \
	VM_VFNMSAC_VV_M (mtail, qy, q, qx, vl), qx, vl); \
      /* The window above cannot cover the lanes the tail matters most \
	 on.  A subnormal |qy| puts the residual below the grid (it \
	 scales with |qy|), and a subnormal |q| puts the tail there (it \
	 scales with |q|), and on both the division reports UNDERFLOW on \
	 an ordinary answer.  Recompute just those lanes at 2^600, where \
	 every step is normal: |qy| < 2^-960 forces |q| <= 2^114 (since \
	 |qx| >= 2^-1074), so nothing overflows on the way up.  */ \
      vm_b_t mq = VM_VMSLTU_VX (qab, (uint64_t) (63ULL << 52), vl); \
      vm_b_t msy = VM_VMSLTU_VX (yab, (uint64_t) (63ULL << 52), vl); \
      vm_b_t mneed = VM_VMOR_MM (mq, msy, vl); \
      vm_f_t qy_s = VM_VFMUL_VF_MU (mneed, VM_VFMV_V_F (0.0, vl), qy, \
				    0x1p600, vl); \
      vm_f_t q_s = VM_VFDIV_VV_MU (mneed, VM_VFMV_V_F (0.0, vl), qy_s, qx, \
				   vl); \
      vm_f_t rem_s = VM_VFNMSAC_VV_M (mneed, qy_s, q_s, qx, vl); \
      vm_f_t qlo_s = VM_VFDIV_VV_MU (mneed, VM_VFMV_V_F (0.0, vl), rem_s, \
				     qx, vl); \
      /* Where |q| is still normal the tail comes straight back down: it \
	 is >= 2^-1013 there, so the step is exact and silent.  */ \
      vm_b_t mqlo = VM_VMAND_MM (mneed, VM_VMNOT_M (mq, vl), vl); \
      qlo = VM_VFMERGE_VVM ( \
	qlo, VM_VFMUL_VF_MU (mqlo, VM_VFMV_V_F (0.0, vl), qlo_s, 0x1p-600, \
			     vl), mqlo, vl); \
      /* A caller whose divisor or dividend is itself a rounded value \
	 -- asinpi/acospi hand in a square root -- passes that value's \
	 residual here and it joins the quotient's own tail.  q_true is \
	 (y+dyt)/(x+dxt), so to first order the tail is \
	 qlo + (dyt - q*dxt)/x.  Both arguments are zero vectors for \
	 atan2/atan2pi, where this is then exactly qlo + 0.  The term is \
	 ~2^-53 of q, so it matters only where the tail is live, and for \
	 the two callers the residual is nonzero only there: the sqrt \
	 returns exactly 1 (residual exactly 0) once |x| < 2^-27, and the \
	 remaining lanes have |q| >= 2^-27, inside the window.  */ \
      { \
	vm_b_t mtl = VM_VMOR_MM (mtail, mqlo, vl); \
	vm_f_t vnum = VM_VFSUB_VV (dyt, VM_VFMUL_VV (q, dxt, vl), vl); \
	vm_f_t vcor = VM_VFDIV_VV_MU (mtl, VM_VFMV_V_F (0.0, vl), vnum, qx, \
				      vl); \
	qlo = VM_VFADD_VV (qlo, vcor, vl); \
      } \
      /* A subnormal quotient has no such window -- its tail is below \
	 the grid by construction -- so the whole pair stays at 2^600 \
	 and atan2pi divides the scale out after its own /pi.  Only the \
	 lanes the general path actually answers qualify: the special \
	 values overwrite the pair with unscaled constants, and the \
	 x < 0 quadrants subtract it from pi.  atan(x) == x on all of \
	 them, so the scale rides through VM_ATAN_K_DD's tiny branch \
	 untouched.  */ \
      vm_b_t mspec = VM_VMOR_MM (mshort, VM_VMOR_MM (myzero, mxzero, vl), \
				 vl); \
      mspec = VM_VMOR_MM (mspec, mnan, vl); \
      vmc_sc = VM_VMAND_MM (mq, \
			    VM_VMNOT_M (VM_VMOR_MM (mspec, mxs, vl), vl), \
			    vl); \
      vm_f_t qsg = VM_VFMERGE_VFM (VM_VFMV_V_F (-1.0, vl), 1.0, \
				   VM_VMOR_MM (m0, mb3, vl), vl); \
      vm_f_t aq = VM_VFABS_V (q, vl); \
      vm_f_t atail = VM_VFMUL_VV (qlo, qsg, vl); \
      aq = VM_VFMERGE_VVM (aq, VM_VFABS_V (q_s, vl), vmc_sc, vl); \
      atail = VM_VFMERGE_VVM (atail, VM_VFMUL_VV (qlo_s, qsg, vl), vmc_sc, \
			      vl); \
      vm_f_t azh, azl; \
      VM_ATAN_K_DD (azh, azl, aq, atail, vl); \
      /* atan's pre-rounding bits are carried through the quadrant and \
	 collapsed once at the end; collapsing here instead would add \
	 half an ulp on top of atan's own in the two x < 0 quadrants, \
	 where the result is then subtracted from pi.  */ \
      azh = VM_VFMERGE_VFM (azh, VM_ATAN2_PIO2 + 0.5 * VM_ATAN2_PILO, mhi, \
			    vl); \
      azl = VM_VFMERGE_VFM (azl, 0.0, mhi, vl); \
      azh = VM_VFMERGE_VFM (azh, 0.0, mlo, vl); \
      azl = VM_VFMERGE_VFM (azl, 0.0, mlo, vl); \
      azh = VM_VFMERGE_VVM (azh, VM_VFNEG_V (azh, vl), mb1, vl); \
      azl = VM_VFMERGE_VVM (azl, VM_VFNEG_V (azl, vl), mb1, vl); \
      { \
	/* pi - az and az - pi, in dd: (VM_ATAN2_PI, VM_ATAN2_PILO) is an \
	   exact double-double for pi, so the subtraction need not round \
	   at all -- fdlibm's `pi - (z - pi_lo)` rounds twice and can \
	   spend half an ulp there.  */ \
	vm_f_t vpi = VM_VFMV_V_F (VM_ATAN2_PI, vl); \
	vm_f_t vplo = VM_VFMV_V_F (VM_ATAN2_PILO, vl); \
	vm_f_t u1, u2, v1, v2; \
	VM_TWO_SUM (u1, u2, vpi, VM_VFNEG_V (azh, vl), vl);	/* pi - az */ \
	u2 = VM_VFADD_VV (u2, VM_VFSUB_VV (vplo, azl, vl), vl); \
	VM_TWO_SUM (v1, v2, azh, VM_VFNEG_V (vpi, vl), vl);	/* az - pi */ \
	v2 = VM_VFADD_VV (v2, VM_VFSUB_VV (azl, vplo, vl), vl); \
	azh = VM_VFMERGE_VVM (azh, u1, mb2, vl); \
	azl = VM_VFMERGE_VVM (azl, u2, mb2, vl); \
	azh = VM_VFMERGE_VVM (azh, v1, mb3, vl); \
	azl = VM_VFMERGE_VVM (azl, v2, mb3, vl); \
      } \
      /* y == 0: +-0 for x > 0, +-pi for x < 0 */ \
      azh = VM_VFMERGE_VVM (azh, y, \
			   VM_VMAND_MM (myzero, \
					VM_VMOR_MM (m0, mb1, vl), vl), vl); \
      azh = VM_VFMERGE_VFM (azh, VM_ATAN2_PI, \
			   VM_VMAND_MM (myzero, mb2, vl), vl); \
      azh = VM_VFMERGE_VFM (azh, -VM_ATAN2_PI, \
			   VM_VMAND_MM (myzero, mb3, vl), vl); \
      /* fdlibm answers y == 0 before x == 0, so these must not
	 overwrite it: atan2(+-0, +-0) is +-0 or +-pi, never pi/2.  */ \
      vm_b_t mxnz = VM_VMAND_MM (mxzero, VM_VMNOT_M (myzero, vl), vl); \
      azh = VM_VFMERGE_VFM (azh, VM_ATAN2_PIO2, \
			   VM_VMAND_MM (mxnz, VM_VMNOT_M (mys, vl), vl), vl); \
      azh = VM_VFMERGE_VFM (azh, -VM_ATAN2_PIO2, \
			   VM_VMAND_MM (mxnz, mys, vl), vl); \
      /* x = +-Inf (and y finite) */ \
      { \
	vm_b_t mxinfy = VM_VMAND_MM (mxinf, VM_VMNOT_M (myinf, vl), vl); \
	azh = VM_VFMERGE_VFM (azh, 0.0, \
			     VM_VMAND_MM (mxinfy, m0, vl), vl); \
	azh = VM_VFMERGE_VFM (azh, -0.0, \
			     VM_VMAND_MM (mxinfy, mb1, vl), vl); \
	azh = VM_VFMERGE_VFM (azh, VM_ATAN2_PI, \
			     VM_VMAND_MM (mxinfy, mb2, vl), vl); \
	azh = VM_VFMERGE_VFM (azh, -VM_ATAN2_PI, \
			     VM_VMAND_MM (mxinfy, mb3, vl), vl); \
	/* x = +-Inf, y = +-Inf */ \
	azh = VM_VFMERGE_VFM (azh, VM_ATAN2_PIO4, \
			     VM_VMAND_MM (mxinf, VM_VMAND_MM (myinf, m0, vl), \
					  vl), vl); \
	azh = VM_VFMERGE_VFM (azh, -VM_ATAN2_PIO4, \
			     VM_VMAND_MM (mxinf, VM_VMAND_MM (myinf, mb1, vl), \
					  vl), vl); \
	azh = VM_VFMERGE_VFM (azh, 3.0 * VM_ATAN2_PIO4, \
			     VM_VMAND_MM (mxinf, VM_VMAND_MM (myinf, mb2, vl), \
					  vl), vl); \
	azh = VM_VFMERGE_VFM (azh, -3.0 * VM_ATAN2_PIO4, \
			     VM_VMAND_MM (mxinf, VM_VMAND_MM (myinf, mb3, vl), \
					  vl), vl); \
      } \
      /* y = +-Inf, but only with x finite: fdlibm resolves x == +-Inf \
	 first, and its own y == +-Inf sub-cases (+-3pi/4) must survive.  */ \
      { \
	vm_b_t myinfx = VM_VMAND_MM (myinf, VM_VMNOT_M (mxinf, vl), vl); \
	azh = VM_VFMERGE_VFM (azh, VM_ATAN2_PIO2, \
			     VM_VMAND_MM (myinfx, VM_VMNOT_M (mys, vl), vl), vl); \
	azh = VM_VFMERGE_VFM (azh, -VM_ATAN2_PIO2, \
			     VM_VMAND_MM (myinfx, mys, vl), vl); \
      } \
      /* masked, and as a real addition: x - (-y) would turn \
	 NaN + -Inf into NaN - +Inf and raise INVALID, which fdlibm's \
	 x + y does not.  */ \
      /* every special-case answer above is exact, so it has no low part */ \
      { \
	vm_b_t vmc_spec = VM_VMOR_MM (VM_VMOR_MM (myzero, mxzero, vl), \
				      VM_VMOR_MM (mxinf, myinf, vl), vl); \
	vmc_spec = VM_VMOR_MM (vmc_spec, mnan, vl); \
	vmc_oh = VM_VFMERGE_VVM (azh, VM_VFADD_VV_M (mnan, x, y, vl), mnan, vl); \
	vmc_ol = VM_VFMERGE_VFM (azl, 0.0, vmc_spec, vl); \
      } \
    } \
  } \
  (oh) = vmc_oh; \
  (ol) = vmc_ol; \
  (osc) = vmc_sc; \
} while (0)

static inline vm_f_t
VM_KFN (atan2_core) (vm_f_t y, vm_f_t x, size_t vl)
{
  vm_f_t vh, vlo, vr;
  vm_b_t vsc;
  VM_ATAN2_CORE_DD (vh, vlo, vsc, y, x, VM_VFMV_V_F (0.0, vl),
		    VM_VFMV_V_F (0.0, vl), vl);
  VM_INVTRIG_COLLAPSE (vr, vh, vlo, vl);
  /* The subnormal-quotient lanes hand back a pair still at 2^600.  Undoing
     it is exact wherever the answer came out normal, and rounds (with the
     UNDERFLOW the reference expects there) wherever it did not.  */
  return VM_VFMERGE_VVM (vr, VM_VFMUL_VF_MU (vsc, VM_VFMV_V_F (0.0, vl), vr,
					     0x1p-600, vl), vsc, vl);
}


/* ---- array-level wrappers ----------------------------------------
   atan(x)/pi and atan2(y,x)/pi multiply by 1/pi as a double-double, so
   the scaling itself costs no accuracy.  The VFABI-style *_k forms only
   exist to carry the scalar-fixup argument the ABI wrappers pass.  */

static inline vm_f_t
VM_KFN (atan) (vm_f_t x, size_t vl)
{
  return VM_KFN (atan_k) (x, VM_VFMV_V_F (0.0, vl), vl);
}

/* (ah + al) / pi, correctly rounded.  A fixed 2^900 scale keeps the dd
   corrections out of the subnormal range: they are ~2^-53 of a product
   that can itself be tiny, and unscaled they underflow on exactly the
   lanes whose relative accuracy needs them.  |ah| <= pi for every
   caller, so nothing overflows on the way up.  The step back down is
   exact whenever the result is normal; a lane whose result really is
   subnormal keeps the plain product, the most that grid can hold.  */
#define VM_INVPI_DD(r, ah, al, vl) do { \
  /* The pair is renormalized first: the atan kernels hand out a dd whose
     low part can sit above 2^-53 of the high one when a tail was folded
     in, and the scaling below assumes the invariant.  */ \
  vm_f_t vid_h, vid_l; \
  VM_TWO_SUM (vid_h, vid_l, (ah), (al), (vl)); \
  vm_f_t vid_s = VM_VFMUL_VF (vid_h, 0x1p900, (vl)); \
  vm_f_t vid_p = VM_VFMUL_VF (vid_s, VM_INVPI_HI, (vl)); \
  vm_f_t vid_e = VM_VFMACC_VF (VM_VFNEG_V (vid_p, (vl)), VM_INVPI_HI, \
			       vid_s, (vl)); \
  vm_f_t vid_t = VM_VFMACC_VF (vid_e, VM_INVPI_LO, vid_s, (vl)); \
  vid_t = VM_VFMACC_VF (vid_t, VM_INVPI_HI, \
			VM_VFMUL_VF (vid_l, 0x1p900, (vl)), (vl)); \
  vm_f_t vid_w = VM_VFADD_VV (vid_p, vid_t, (vl)); \
  vm_u64_t vid_b = VM_VAND_VX (VM_BITS (vid_w), 0x7fffffffffffffffULL, \
				(vl)); \
  vm_b_t vid_mn = VM_VMAND_MM ( \
    VM_VMSGEU_VX (vid_b, (uint64_t) (901ULL << 52), (vl)), \
    VM_VMSLTU_VX (vid_b, (uint64_t) (1923ULL << 52), (vl)), (vl)); \
  vm_f_t vid_pn = VM_VFMUL_VF (vid_h, VM_INVPI_HI, (vl)); \
  vm_f_t vid_rb = VM_VFMUL_VF_M (vid_mn, vid_w, 0x1p-900, (vl)); \
  vid_rb = VM_VFMERGE_VVM (vid_rb, vid_pn, \
			   VM_VMNOT_M (vid_mn, (vl)), (vl)); \
  (r) = vid_rb; \
} while (0)

static inline vm_f_t
VM_KFN (atanpi) (vm_f_t x, size_t vl)
{
  /* atan's pre-rounding bits, so the /pi does not land on top of a
     half-ulp rounding that is already there.  */
  vm_f_t ah, al;
  VM_ATAN_K_DD (ah, al, x, VM_VFMV_V_F (0.0, vl), vl);
  /* a * 1/pi as a real dd multiply.  (VM_INVPI_HI, VM_INVPI_LO) is
     1/pi to ~2^-107 already, but a*HI's own rounding has to be kept
     too, which a single fma would throw away.  */
  /* The two corrections below are ~2^-53 of the product, so for a small
     |a| they land in the subnormal range and raise UNDERFLOW on a lane
     whose answer is normal, where the reference raises no flag at all.
     They are masked to |a| >= 2^-29, where they are normal; below that
     atan() has already returned x exactly, so a plain product is
     correctly rounded within the constant's own 2^-54.  */
  vm_f_t w;
  VM_INVPI_DD (w, ah, al, vl);
  /* a * (INVPI_HI + INVPI_LO) with INVPI_LO < 0 splits -0 into (-0, +0),
     whose sum is +0; restore the sign of a on the exact-zero lanes.  */
  return VM_VFMERGE_VVM (w, VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, vl), ah, vl),
			 VM_VMFEQ_VV (w, VM_VFMV_V_F (0.0, vl), vl), vl);
}

static inline vm_f_t
VM_KFN (atan2) (vm_f_t y, vm_f_t x, size_t vl)
{
  return VM_KFN (atan2_core) (y, x, vl);
}

/* atan2pi of (y + dyt) / (x + dxt) -- the two residuals are the caller's
   way of saying "this argument is itself a rounded value".  asinpi and
   acospi are the only users, and they pass the square root's own error.  */
static inline vm_f_t
VM_KFN (atan2pi_t) (vm_f_t y, vm_f_t x, vm_f_t dyt, vm_f_t dxt, size_t vl)
{
  /* The dd core, so the /pi does not round a value atan2 already rounded.
     Masked exactly as in atanpi (see there).  */
  vm_f_t ah, al;
  vm_b_t asc;
  VM_ATAN2_CORE_DD (ah, al, asc, y, x, dyt, dxt, vl);
  vm_f_t w;
  VM_INVPI_DD (w, ah, al, vl);
  /* The subnormal-quotient lanes carry the 2^600 the core left on their
     pair.  Bringing it down is exact -- the point of the scale -- so a
     lane whose answer really does land below DBL_MIN never rounds at
     that magnitude, and nothing raises the UNDERFLOW the reference
     expects of it (pi is irrational, so q/pi is always inexact).
     Detect those lanes as the reference does, |w| < DBL_MIN, and
     restore the flag as force_eval does: a masked operation that
     genuinely underflows, whose zero result is added back untouched.  */
  w = VM_VFMERGE_VVM (w, VM_VFMUL_VF_MU (asc, VM_VFMV_V_F (0.0, vl), w,
					 0x1p-600, vl), asc, vl);
  {
    vm_u64_t wab = VM_VAND_VX (VM_BITS (w), 0x7fffffffffffffffULL, vl);
    vm_b_t muf = VM_VMAND_MM (
      asc, VM_VMSLTU_VX (wab, (uint64_t) (1ULL << 52), vl), vl);
    vm_f_t vuf = VM_VFMUL_VF_MU (
      muf, VM_VFMV_V_F (0.0, vl),
      VM_VFADD_VF_MU (muf, VM_VFMV_V_F (0.0, vl), w, 0x1p-600, vl),
      0x1p-600, vl);
    w = VM_VFMERGE_VVM (w, VM_VFADD_VV (w, vuf, vl), muf, vl);
  }
  return VM_VFMERGE_VVM (w, VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, vl), ah, vl),
			 VM_VMFEQ_VV (w, VM_VFMV_V_F (0.0, vl), vl), vl);
}

static inline vm_f_t
VM_KFN (atan2pi) (vm_f_t y, vm_f_t x, size_t vl)
{
  return VM_KFN (atan2pi_t) (y, x, VM_VFMV_V_F (0.0, vl),
			     VM_VFMV_V_F (0.0, vl), vl);
}

static inline vm_f_t
VM_KFN (atan2_k) (vm_f_t y, vm_f_t x, size_t vl,
		  double (*fixup) (double, double))
{
  (void) fixup;
  return VM_KFN (atan2_core) (y, x, vl);
}

static inline vm_f_t
VM_KFN (atan2pi_k) (vm_f_t y, vm_f_t x, size_t vl,
		    double (*fixup) (double, double))
{
  (void) fixup;
  return VM_KFN (atan2pi) (y, x, vl);
}

/* s_asin.c's rational, used below |x| = 0.5. */
#define VM_ASIN_PS0	1.66666666666666657415e-01
#define VM_ASIN_PS1	-3.25565818622400915405e-01
#define VM_ASIN_PS2	2.01212532134862925881e-01
#define VM_ASIN_PS3	-4.00555345006794114027e-02
#define VM_ASIN_PS4	7.91534994289814532176e-04
#define VM_ASIN_PS5	3.47933107596021167570e-05
#define VM_ASIN_QS1	-2.40339491173441421878e+00
#define VM_ASIN_QS2	2.02094576023350569471e+00
#define VM_ASIN_QS3	-6.88283971605453293030e-01
#define VM_ASIN_QS4	7.70381505559019352791e-02

/* asin(x) = atan(x / sqrt(1-x^2)); acos(x) as atan(s/|x|) reflected into
   the second quadrant for x < 0.

   Two things have to be right to stay under an ulp.  First, 1 - x^2 is
   formed as (1-|x|)*(1+|x|): the first factor is exact by Sterbenz for
   |x| >= 0.5, where x*x would cancel away every significant bit.  Second,
   the square root goes through the dd sqrt and the piece it leaves over
   rides into atan exactly the way atan2's division residual does, or the
   arguments' own error would land on top of atan's.  */
/* q = num/den with the piece the division rounded away, plus the dd sqrt's
   own low part.  The sqrt sits in the denominator for asin and in the
   numerator for acos, and the correction takes the opposite sign in each,
   so there are two forms rather than one.  */
#define VM_ASIN_Q(q, tail, x, sh, sl, vl) do {				\
  (q) = VM_VFDIV_VV ((x), (sh), (vl));					\
  /* x - q*(sh + sl): the division's remainder and the sqrt's own low
     part in one numerator.  */ \
  vm_f_t vmaq_n = VM_VFSUB_VV ( \
    VM_VFNMSAC_VV ((x), (q), (sh), (vl)), \
    VM_VFMUL_VV ((q), (sl), (vl)), (vl)); \
  vm_f_t vmaq_t = VM_VFDIV_VV (vmaq_n, (sh), (vl));			\
  /* One Newton step: the tail is only worth a fraction of an ulp of q, \
     so its own half ulp would otherwise land on the result unchanged. \
     Newton squares that error away.  */ \
  (tail) = VM_VFADD_VV (						\
    vmaq_t, VM_VFDIV_VV (VM_VFNMSAC_VV (vmaq_n, vmaq_t, (sh), (vl)),	\
			 (sh), (vl)), (vl));				\
} while (0)

#define VM_ACOS_Q(q, tail, sh, sl, ax, vl) do {				\
  (q) = VM_VFDIV_VV ((sh), (ax), (vl));					\
  (tail) = VM_VFDIV_VV (						\
    VM_VFADD_VV (VM_VFNMSAC_VV ((sh), (q), (ax), (vl)), (sl), (vl)),	\
    (ax), (vl));							\
} while (0)

static inline vm_f_t
VM_KFN (asin) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_u64_t lx = VM_VAND_VX (VM_BITS (x), 0xffffffffULL, vl);
  vm_b_t mdom = VM_VMSGTU_VX (ix, 0x3ff00000u, vl);	/* |x| > 1 */
  vm_b_t mone = VM_VMAND_MM (VM_VMSEQ_VX (ix, 0x3ff00000u, vl),
			     VM_VMSEQ_VX (lx, 0, vl), vl);	/* |x| == 1 */
  vm_b_t mskip = VM_VMOR_MM (mdom, mone, vl);
  vm_f_t a = VM_VFMERGE_VFM (VM_VFABS_V (x, vl), 0.5, mskip, vl);
  vm_f_t w = VM_VFMUL_VV (VM_VFRSUB_VF (a, 1.0, vl), VM_VFADD_VF (a, 1.0, vl),
			  vl);
  vm_f_t sh, sl, q, tail, yh, yl, r;
  VM_DD_SQRT (sh, sl, w, VM_VFMV_V_F (0.0, vl), vl);
  VM_ASIN_Q (q, tail, VM_VFMERGE_VFM (x, 0.5, mskip, vl), sh, sl, vl);
  /* atan_k is called with the magnitude and the tail signed to match,
     and the sign is restored afterwards -- as atan2 does.  It does not
     take a negative argument with a tail.  */
  vm_f_t qsg = VM_VFMERGE_VFM (
    VM_VFMV_V_F (1.0, vl), -1.0,
    VM_VMSLT_VX (VM_REINT_U_I (VM_BITS (q)), 0, vl), vl);
  VM_ATAN_K_DD (yh, yl, VM_VFABS_V (q, vl), VM_VFMUL_VV (tail, qsg, vl), vl);
  VM_INVTRIG_COLLAPSE (r, yh, yl, vl);
  r = VM_VFSGNJ_VV (r, q, vl);
  {
    /* |x| < 0.5: fdlibm's rational, x + x*P(x^2)/Q(x^2).  The atan
       route above is fine down to 0.5 but loses accuracy on the small
       arguments, where asin is nearly linear and the composition has no
       cancellation error left to hide the extra rounding in.  */
    vm_b_t mhalf = VM_VMSLTU_VX (ix, 0x3fe00000u, vl);
    /* |x| < 2^-26: fdlibm returns x itself, and it is right to --
       x^2/6 is below the last bit there, while x*x would underflow
       on the subnormal inputs and raise UNDERFLOW for nothing.  */
    vm_b_t mh2 = VM_VMAND_MM (mhalf,
			      VM_VMSLTU_VX (ix, 0x3e500000u, vl), vl);
    vm_f_t xs = VM_VFMERGE_VFM (x, 0.25,
				VM_VMOR_MM (VM_VMNOT_M (mhalf, vl), mh2, vl),
				vl);
    vm_f_t t = VM_VFMUL_VV (xs, xs, vl);
    vm_f_t pp = VM_VFMV_V_F (VM_ASIN_PS5, vl);
    vm_f_t qq;
    pp = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_PS4, vl), t, pp, vl);
    pp = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_PS3, vl), t, pp, vl);
    pp = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_PS2, vl), t, pp, vl);
    pp = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_PS1, vl), t, pp, vl);
    pp = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_PS0, vl), t, pp, vl);
    pp = VM_VFMUL_VV (t, pp, vl);
    qq = VM_VFMV_V_F (VM_ASIN_QS4, vl);
    qq = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_QS3, vl), t, qq, vl);
    qq = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_QS2, vl), t, qq, vl);
    qq = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_QS1, vl), t, qq, vl);
    qq = VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), t, qq, vl);
    r = VM_VFMERGE_VVM (
      r, VM_VFMACC_VV (xs, xs, VM_VFDIV_VV (pp, qq, vl), vl), mhalf, vl);
  }
  r = VM_VFMERGE_VVM (
    r, VM_VFSGNJ_VV (VM_VFMV_V_F (VM_ATAN2_PIO2, vl), x, vl), mone, vl);
  r = VM_VFMERGE_VVM (r, x, VM_VMSLTU_VX (ix, 0x3e500000u, vl), vl);
  {
    vm_f_t z = VM_VFSUB_VV_M (mdom, x, x, vl);
    r = VM_VFMERGE_VVM (
      r, VM_VFDIV_VV_MU (mdom, VM_VFMV_V_F (0.0, vl), z, z, vl), mdom, vl);
  }
  return r;
}

static inline vm_f_t
VM_KFN (acos) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_u64_t lx = VM_VAND_VX (VM_BITS (x), 0xffffffffULL, vl);
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t mdom = VM_VMSGTU_VX (ix, 0x3ff00000u, vl);	/* |x| > 1 */
  vm_b_t mone = VM_VMAND_MM (VM_VMSEQ_VX (ix, 0x3ff00000u, vl),
			     VM_VMSEQ_VX (lx, 0, vl), vl);	/* |x| == 1 */
  /* +-0: comparing bits(x) against 0 misses -0, which then reaches the
     reciprocal as a zero divisor (DIVBYZERO) and takes the tail with
     it (OVERFLOW).  */
  vm_b_t mzero = VM_VMSEQ_VX (
    VM_VAND_VX (VM_BITS (x), 0x7fffffffffffffffULL, vl), 0, vl);
  /* |x| < 2^-57 is fdlibm's own shortcut: s/|x| would otherwise
     overflow on the subnormal inputs, and pi/2 - x rounds to pi/2
     there anyway.  */
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3c600000u, vl);
  vm_b_t mskip = VM_VMOR_MM (
    VM_VMOR_MM (VM_VMOR_MM (mdom, mone, vl), mzero, vl), mtiny, vl);
  vm_f_t a = VM_VFMERGE_VFM (VM_VFABS_V (x, vl), 0.5, mskip, vl);
  vm_f_t w = VM_VFMUL_VV (VM_VFRSUB_VF (a, 1.0, vl), VM_VFADD_VF (a, 1.0, vl),
			  vl);
  vm_f_t sh, sl, q, tail, yh, yl, r;
  VM_DD_SQRT (sh, sl, w, VM_VFMV_V_F (0.0, vl), vl);
  VM_ACOS_Q (q, tail, sh, sl, VM_VFMERGE_VFM (a, 1.0, mskip, vl), vl);
  VM_ATAN_K_DD (yh, yl, q, tail, vl);
  {
    /* x < 0: acos(x) = pi - atan(s/|x|), in dd so the subtraction does not
       round away what the branch was for.  */
    vm_f_t u1, u2;
    VM_TWO_SUM (u1, u2, VM_VFMV_V_F (VM_ATAN2_PI, vl), VM_VFNEG_V (yh, vl),
		(vl));
    u2 = VM_VFADD_VV (u2, VM_VFMV_V_F (VM_ATAN2_PILO, vl), vl);
    u2 = VM_VFSUB_VV (u2, yl, vl);
    yh = VM_VFMERGE_VVM (yh, u1, mneg, vl);
    yl = VM_VFMERGE_VVM (yl, u2, mneg, vl);
  }
  VM_INVTRIG_COLLAPSE (r, yh, yl, vl);
  /* the |x| == 1 and x == 0 ends are exact */
  r = VM_VFMERGE_VFM (
    r, 0.0, VM_VMAND_MM (mone, VM_VMSEQ_VX (
			   VM_BITS (x), 0x3ff0000000000000ULL, vl), vl), vl);
  r = VM_VFMERGE_VFM (
    r, VM_ATAN2_PI, VM_VMAND_MM (mone, VM_VMSEQ_VX (
				   VM_BITS (x), 0xbff0000000000000ULL, vl),
				 vl), vl);
  r = VM_VFMERGE_VFM (r, VM_ATAN2_PIO2, mzero, vl);
  r = VM_VFMERGE_VFM (r, VM_ATAN2_PIO2, mtiny, vl);
  {
    vm_f_t z = VM_VFSUB_VV_M (mdom, x, x, vl);
    r = VM_VFMERGE_VVM (
      r, VM_VFDIV_VV_MU (mdom, VM_VFMV_V_F (0.0, vl), z, z, vl), mdom, vl);
  }
  return r;
}

/* asinpi / acospi, both as the atan2pi of a formed argument.  The square
   root's dd is not needed here: acos/asin(1-e) ~ sqrt(2e) is read off a
   value whose error is scaled by the same factor the answer is.  */
static inline vm_f_t
VM_KFN (asinpi) (vm_f_t x, size_t vl)
{
  vm_u64_t ix = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 32, vl), 0x7fffffffULL,
			    vl);
  vm_b_t mdom = VM_VMSGTU_VX (ix, 0x3ff00000u, vl);	/* |x| > 1 */
  vm_f_t a = VM_VFMERGE_VFM (VM_VFABS_V (x, vl), 0.5, mdom, vl);
  /* w = (1-a)(1+a) is a rounded product and sqrt(w) a rounded root, so
     the correction has two halves: wl, the exact residual of the
     product, and w - s*s, the exact residual of the root.  Feeding only
     the second leaves the first's half ulp of q in the answer.  */
  vm_f_t v1m = VM_VFRSUB_VF (a, 1.0, vl);
  vm_f_t v1p = VM_VFADD_VF (a, 1.0, vl);
  vm_f_t w = VM_VFMUL_VV (v1m, v1p, vl);
  vm_f_t wl = VM_VFMACC_VV (VM_VFNEG_V (w, vl), v1m, v1p, vl);
  /* sqrt(0) is exact, so the residual is 0 on the |x| == 1 lanes and the
     division has to be masked there or it is 0/0 and raises INVALID.  */
  vm_f_t s = VM_VFSQRT_V (w, vl);
  vm_f_t ds = VM_VFDIV_VV_MU (
    VM_VMFNE_VF (w, 0.0, vl), VM_VFMV_V_F (0.0, vl),
    VM_VFADD_VV (wl, VM_VFNMSAC_VV (w, s, s, vl), vl),
    VM_VFADD_VV (s, s, vl), vl);
  vm_f_t r = VM_KFN (atan2pi_t) (VM_VFMERGE_VFM (x, 0.5, mdom, vl), s,
				 VM_VFMV_V_F (0.0, vl), ds, vl);
  vm_f_t z = VM_VFSUB_VV_M (mdom, x, x, vl);
  return VM_VFMERGE_VVM (
    r, VM_VFDIV_VV_MU (mdom, VM_VFMV_V_F (0.0, vl), z, z, vl), mdom, vl);
}

static inline vm_f_t
VM_KFN (acospi) (vm_f_t x, size_t vl)
{
  vm_u64_t ix = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 32, vl), 0x7fffffffULL,
			    vl);
  vm_b_t mdom = VM_VMSGTU_VX (ix, 0x3ff00000u, vl);
  vm_f_t a = VM_VFMERGE_VFM (VM_VFABS_V (x, vl), 0.5, mdom, vl);
  vm_f_t v1m = VM_VFRSUB_VF (a, 1.0, vl);
  vm_f_t v1p = VM_VFADD_VF (a, 1.0, vl);
  vm_f_t w = VM_VFMUL_VV (v1m, v1p, vl);
  vm_f_t wl = VM_VFMACC_VV (VM_VFNEG_V (w, vl), v1m, v1p, vl);
  /* Acospi near |x| = 1 is where this matters: the answer is ~sqrt(2e), so
     the root's relative error is the answer's.  Here the root is the
     dividend, so its residual goes in on that side instead.  */
  vm_f_t s = VM_VFSQRT_V (w, vl);
  vm_f_t ds = VM_VFDIV_VV_MU (
    VM_VMFNE_VF (w, 0.0, vl), VM_VFMV_V_F (0.0, vl),
    VM_VFADD_VV (wl, VM_VFNMSAC_VV (w, s, s, vl), vl),
    VM_VFADD_VV (s, s, vl), vl);
  vm_f_t r = VM_KFN (atan2pi_t) (s, VM_VFMERGE_VFM (x, 0.5, mdom, vl), ds,
				 VM_VFMV_V_F (0.0, vl), vl);
  vm_f_t z = VM_VFSUB_VV_M (mdom, x, x, vl);
  return VM_VFMERGE_VVM (
    r, VM_VFDIV_VV_MU (mdom, VM_VFMV_V_F (0.0, vl), z, z, vl), mdom, vl);
}
