/* k_log.inc -- log / log2 / log10 / log1p / log2p1 kernels for libvecmath.

   LMUL-generic template: include after vecmath_rvvcompat.h,
   vecmath_dd.h and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.

   log/log2 vectorize the ARM optimized-routines implementations that are
   live in newlib for rv64 hard-float (libm/common/log.c, log2.c):
   table-gather reduction + polynomial, near-1 special polynomial, and a
   masked scalar fixup for x < 2^-1022 / negative / inf / NaN (covers
   divzero, invalid and subnormal normalization with exact scalar
   semantics and flags).

   log10/log2p1 scale the dd output of log/log1p before the final
   rounding: log10(x) = log(x)*log10(e), log2p1(x) = log1p(x)/ln2.

   Data: own table copies (vm_data.h / tables/), never libm symbols --
   glibc and newlib order the struct fields differently.  */

#include <vm_data.h>

/* log near-1 region: bits(1-2^-4) .. bits(1+0x1.09p-4).  */
#define VM_LOG_LO	0x3fee000000000000ULL
#define VM_LOG_HILO	0x0003090000000000ULL
/* log2 near-1 region: bits(1-0x1.5b51p-5) .. bits(1+0x1.6ab2p-5).  */
#define VM_LOG2_LO	0x3feea4af00000000ULL
#define VM_LOG2_HILO	0x000210aa00000000ULL
/* shared reduction offset (log.c/log2.c OFF).  */
#define VM_LOG_OFF	0x3fe6000000000000ULL

/* log10(e) and 1/ln2 as dd pairs (>= 106 bits).  */
#define VM_LOG10E_HI	4.34294481903251816668e-01
#define VM_LOG10E_LO	1.09831965021676507274e-17
#define VM_INVLN2_HI	1.44269504088896338700e+00
#define VM_INVLN2_LO	2.03552737409310331110e-17

/* ------------------------------------------------------------------ */
/* log(x) -- vectorization of libm/common/log.c (N=128, poly orders
   6 / 12).  fixup must be scalar log().

   VM_LOG_DD returns the result as a double-double pair (yh, yl): the
   main path's (hi, lo) decomposition already carries ~106 bits, so the
   dd tap is exact to ~2^-105 relative, which is what log10 scales by
   log10(e) before the final rounding.  */
#define VM_LOG_DD(yh, yl, x, vl, fixup) do {				\
  const struct vm_log_data *vmlgd__D = &__vecmath_log_data;		\
  vm_u64_t vmlgd__ix = VM_BITS (x);					\
  vm_u64_t vmlgd__top = VM_VSRL_VX (vmlgd__ix, 48, (vl));		\
  vm_b_t vmlgd__mspec = VM_VMSGEU_VX (				\
    VM_VSUB_VX (vmlgd__top, 0x10, (vl)), 0x7fe0, (vl));			\
  vm_b_t vmlgd__mnear = VM_VMSLTU_VX (					\
    VM_VSUB_VX (vmlgd__ix, VM_LOG_LO, (vl)), VM_LOG_HILO, (vl));	\
  vm_f_t vmlgd__xs = VM_VFMERGE_VFM ((x), 1.0, vmlgd__mspec, (vl));	\
  vm_f_t vmlgd__xn = VM_VFMERGE_VFM ((x), 1.0,				\
				     VM_VMNOT_M (vmlgd__mnear, (vl)), (vl)); \
  /* ---- near-1 polynomial (order 12) ---- */				\
  vm_f_t vmlgd__r = VM_VFSUB_VF (vmlgd__xn, 1.0, (vl));			\
  vm_f_t vmlgd__r2 = VM_VFMUL_VV (vmlgd__r, vmlgd__r, (vl));		\
  vm_f_t vmlgd__r3 = VM_VFMUL_VV (vmlgd__r, vmlgd__r2, (vl));		\
  vm_f_t vmlgd__q = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly1[7], (vl)), \
				  vmlgd__D->poly1[8], vmlgd__r, (vl));	\
  vmlgd__q = VM_VFMACC_VF (vmlgd__q, vmlgd__D->poly1[9], vmlgd__r2, (vl)); \
  vmlgd__q = VM_VFMACC_VF (vmlgd__q, vmlgd__D->poly1[10], vmlgd__r3, (vl)); \
  vm_f_t vmlgd__p = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly1[4], (vl)), \
				  vmlgd__D->poly1[5], vmlgd__r, (vl));	\
  vmlgd__p = VM_VFMACC_VF (vmlgd__p, vmlgd__D->poly1[6], vmlgd__r2, (vl)); \
  vmlgd__p = VM_VFMACC_VV (vmlgd__p, vmlgd__r3, vmlgd__q, (vl));	\
  vm_f_t vmlgd__P = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly1[1], (vl)), \
				  vmlgd__D->poly1[2], vmlgd__r, (vl));	\
  vmlgd__P = VM_VFMACC_VF (vmlgd__P, vmlgd__D->poly1[3], vmlgd__r2, (vl)); \
  vmlgd__P = VM_VFMACC_VV (vmlgd__P, vmlgd__r3, vmlgd__p, (vl));	\
  vm_f_t vmlgd__npoly = VM_VFMUL_VV (vmlgd__r3, vmlgd__P, (vl));	\
  vm_f_t vmlgd__w = VM_VFMUL_VF (vmlgd__r, 0x1p27, (vl));		\
  vm_f_t vmlgd__rhi = VM_VFSUB_VV (VM_VFADD_VV (vmlgd__r, vmlgd__w, (vl)), \
				   vmlgd__w, (vl));			\
  vm_f_t vmlgd__rlo = VM_VFSUB_VV (vmlgd__r, vmlgd__rhi, (vl));		\
  vmlgd__w = VM_VFMUL_VF (VM_VFMUL_VV (vmlgd__rhi, vmlgd__rhi, (vl)),	\
			  -0.5, (vl));					\
  vm_f_t vmlgd__nhi = VM_VFADD_VV (vmlgd__r, vmlgd__w, (vl));		\
  vm_f_t vmlgd__nlo = VM_VFADD_VV (VM_VFSUB_VV (vmlgd__r, vmlgd__nhi, (vl)), \
				   vmlgd__w, (vl));			\
  vmlgd__nlo = VM_VFADD_VV (vmlgd__nlo,				\
    VM_VFMUL_VV (VM_VFMUL_VF (vmlgd__rlo, -0.5, (vl)),			\
		 VM_VFADD_VV (vmlgd__rhi, vmlgd__r, (vl)), (vl)), (vl));	\
  /* near-1 dd = (nhi, nlo) + (npoly, 0).  */				\
  vm_f_t vmlgd__zh = VM_VFMV_V_F (0.0, (vl));				\
  vm_f_t vmlgd__dh, vmlgd__dl;						\
  VM_DD_ADD (vmlgd__dh, vmlgd__dl, vmlgd__nhi, vmlgd__nlo,		\
	     vmlgd__npoly, vmlgd__zh, (vl));				\
  /* ---- main path ---- */						\
  vm_u64_t vmlgd__ixb = VM_BITS (vmlgd__xs);				\
  vm_u64_t vmlgd__tmp = VM_VSUB_VX (vmlgd__ixb, VM_LOG_OFF, (vl));	\
  vm_u64_t vmlgd__off = VM_VSLL_VX (VM_VAND_VX (			\
    VM_VSRL_VX (vmlgd__tmp, 45, (vl)), 127, (vl)), 4, (vl));		\
  vm_f_t vmlgd__invc = VM_VLUXEI64 (&vmlgd__D->tab[0].invc, vmlgd__off, (vl)); \
  vm_f_t vmlgd__logc = VM_VLUXEI64 (&vmlgd__D->tab[0].logc, vmlgd__off, (vl)); \
  vm_i64_t vmlgd__k = VM_VSRA_VX (VM_REINT_U_I (vmlgd__tmp), 52, (vl));	\
  vm_u64_t vmlgd__iz = VM_VSUB_VV (vmlgd__ixb,				\
    VM_VAND_VX (vmlgd__tmp, 0xfff0000000000000ULL, (vl)), (vl));	\
  vm_f_t vmlgd__z = VM_FROM_BITS (vmlgd__iz);				\
  vm_f_t vmlgd__kd = VM_VFCVT_F_X_V (vmlgd__k, (vl));			\
  vmlgd__r = VM_VFMACC_VV (VM_VFMV_V_F (-1.0, (vl)),			\
			   vmlgd__invc, vmlgd__z, (vl));		\
  vmlgd__w = VM_VFMACC_VF (vmlgd__logc, vmlgd__D->ln2hi, vmlgd__kd, (vl)); \
  vm_f_t vmlgd__mhi = VM_VFADD_VV (vmlgd__w, vmlgd__r, (vl));		\
  vm_f_t vmlgd__mlo = VM_VFADD_VV (					\
    VM_VFADD_VV (VM_VFSUB_VV (vmlgd__w, vmlgd__mhi, (vl)), vmlgd__r, (vl)), \
    VM_VFMUL_VF (vmlgd__kd, vmlgd__D->ln2lo, (vl)), (vl));		\
  vmlgd__r2 = VM_VFMUL_VV (vmlgd__r, vmlgd__r, (vl));			\
  vmlgd__q = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly[3], (vl)),	\
			   vmlgd__D->poly[4], vmlgd__r, (vl));		\
  vmlgd__q = VM_VFMUL_VV (vmlgd__r2, vmlgd__q, (vl));			\
  vmlgd__p = VM_VFMACC_VF (VM_VFMV_V_F (vmlgd__D->poly[1], (vl)),	\
			   vmlgd__D->poly[2], vmlgd__r, (vl));		\
  vmlgd__p = VM_VFADD_VV (vmlgd__p, vmlgd__q, (vl));			\
  vmlgd__p = VM_VFMUL_VV (VM_VFMUL_VV (vmlgd__r, vmlgd__r2, (vl)),	\
			  vmlgd__p, (vl));				\
  vm_f_t vmlgd__pr = VM_VFADD_VV (					\
    VM_VFMUL_VF (vmlgd__r2, vmlgd__D->poly[0], (vl)), vmlgd__p, (vl));	\
  /* main dd = (mhi, mlo) + (pr, 0).  */				\
  vm_f_t vmlgd__eh, vmlgd__el;						\
  VM_DD_ADD (vmlgd__eh, vmlgd__el, vmlgd__mhi, vmlgd__mlo,		\
	     vmlgd__pr, vmlgd__zh, (vl));				\
  /* merge dd pairs, collapse, fixup special lanes.  */			\
  vm_f_t vmlgd__h = VM_VFMERGE_VVM (vmlgd__eh, vmlgd__dh, vmlgd__mnear, (vl)); \
  vm_f_t vmlgd__l = VM_VFMERGE_VVM (vmlgd__el, vmlgd__dl, vmlgd__mnear, (vl)); \
  vm_f_t vmlgd__yc;							\
  VM_DD_COLLAPSE (vmlgd__yc, vmlgd__h, vmlgd__l, (vl));			\
  (yh) = VM_KFN (scalar_fixup) ((x), vmlgd__yc, vmlgd__mspec, (vl), (fixup)); \
  (yl) = VM_VFMERGE_VFM (vmlgd__l, 0.0, vmlgd__mspec, (vl));		\
} while (0)

static inline vm_f_t
VM_KFN (log_k) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_f_t yh, yl;
  VM_LOG_DD (yh, yl, x, vl, fixup);
  vm_f_t y;
  VM_DD_COLLAPSE (y, yh, yl, vl);
  return y;
}

/* ------------------------------------------------------------------ */
/* log2(x) -- vectorization of libm/common/log2.c (N=64, orders 7/11).
   fixup must be scalar log2().  */
static inline vm_f_t
VM_KFN (log2_k) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  const struct vm_log2_data *D = &__vecmath_log2_data;
  vm_u64_t ix = VM_BITS (x);
  vm_u64_t top16 = VM_VSRL_VX (ix, 48, vl);
  vm_b_t m_special = VM_VMSGEU_VX (VM_VSUB_VX (top16, 0x10, vl), 0x7fe0, vl);
  vm_b_t m_near = VM_VMSLTU_VX (VM_VSUB_VX (ix, VM_LOG2_LO, vl),
				VM_LOG2_HILO, vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 1.0, m_special, vl);
  vm_f_t xn = VM_VFMERGE_VFM (x, 1.0, VM_VMNOT_M (m_near, vl), vl);

  /* ---- near-1 polynomial (LOG2_POLY1_ORDER 11) ---- */
  vm_f_t r = VM_VFSUB_VF (xn, 1.0, vl);
  vm_f_t hi = VM_VFMUL_VF (r, D->invln2hi, vl);
  /* lo = r*InvLn2lo + fma(r, InvLn2hi, -hi) (fused chain like scalar).  */
  vm_f_t lo = VM_VFMACC_VF (VM_VFMACC_VF (VM_VFNEG_V (hi, vl),
					  D->invln2hi, r, vl),
			    D->invln2lo, r, vl);
  vm_f_t r2 = VM_VFMUL_VV (r, r, vl);
  vm_f_t r4 = VM_VFMUL_VV (r2, r2, vl);
  vm_f_t p = VM_VFMUL_VV (r2, VM_VFMACC_VF (VM_VFMV_V_F (D->poly1[0], vl),
					    D->poly1[1], r, vl), vl);
  vm_f_t yn = VM_VFADD_VV (hi, p, vl);
  lo = VM_VFADD_VV (lo, VM_VFADD_VV (VM_VFSUB_VV (hi, yn, vl), p, vl), vl);
  vm_f_t q = VM_VFMACC_VF (VM_VFMV_V_F (D->poly1[8], vl), D->poly1[9], r, vl);
  vm_f_t q2 = VM_VFMACC_VF (VM_VFMV_V_F (D->poly1[6], vl), D->poly1[7], r, vl);
  q2 = VM_VFMACC_VV (q2, r2, q, vl);
  vm_f_t q1 = VM_VFMACC_VF (VM_VFMV_V_F (D->poly1[4], vl), D->poly1[5], r, vl);
  vm_f_t q0 = VM_VFMACC_VF (VM_VFMV_V_F (D->poly1[2], vl), D->poly1[3], r, vl);
  q0 = VM_VFMACC_VV (q0, r2, q1, vl);
  q0 = VM_VFMACC_VV (q0, r4, q2, vl);
  lo = VM_VFMACC_VV (lo, r4, q0, vl);
  yn = VM_VFADD_VV (yn, lo, vl);

  /* ---- main path ---- */
  vm_u64_t ixb = VM_BITS (xs);
  vm_u64_t tmp = VM_VSUB_VX (ixb, VM_LOG_OFF, vl);
  vm_u64_t off = VM_VSLL_VX (VM_VAND_VX (VM_VSRL_VX (tmp, 46, vl), 63, vl),
			     4, vl);
  vm_f_t invc = VM_VLUXEI64 (&D->tab[0].invc, off, vl);
  vm_f_t logc = VM_VLUXEI64 (&D->tab[0].logc, off, vl);
  vm_i64_t k = VM_VSRA_VX (VM_REINT_U_I (tmp), 52, vl);
  /* iz = ix - (tmp & 0xfff0000000000000): strip the exponent delta
     (scalar writes `tmp & 0xfffULL << 52` -- the shift binds tighter,
     so the mask applies to tmp's EXPONENT bits, not its low bits).  */
  vm_u64_t iz = VM_VSUB_VV (ixb, VM_VAND_VX (tmp, 0xfff0000000000000ULL, vl),
			    vl);
  vm_f_t z = VM_FROM_BITS (iz);
  vm_f_t kd = VM_VFCVT_F_X_V (k, vl);
  r = VM_VFMACC_VV (VM_VFMV_V_F (-1.0, vl), invc, z, vl);
  vm_f_t t1 = VM_VFMUL_VF (r, D->invln2hi, vl);
  vm_f_t t2 = VM_VFMACC_VF (VM_VFMACC_VF (VM_VFNEG_V (t1, vl),
					  D->invln2hi, r, vl),
			    D->invln2lo, r, vl);
  vm_f_t t3 = VM_VFADD_VV (kd, logc, vl);
  hi = VM_VFADD_VV (t3, t1, vl);
  lo = VM_VFADD_VV (VM_VFADD_VV (VM_VFSUB_VV (t3, hi, vl), t1, vl), t2, vl);
  r2 = VM_VFMUL_VV (r, r, vl);
  r4 = VM_VFMUL_VV (r2, r2, vl);
  /* p = A0 + r*A1 + r2*(A2 + r*A3) + r4*(A4 + r*A5) */
  p = VM_VFMACC_VF (VM_VFMV_V_F (D->poly[0], vl), D->poly[1], r, vl);
  q = VM_VFMACC_VF (VM_VFMV_V_F (D->poly[2], vl), D->poly[3], r, vl);
  q2 = VM_VFMACC_VF (VM_VFMV_V_F (D->poly[4], vl), D->poly[5], r, vl);
  p = VM_VFMACC_VV (p, r2, q, vl);
  p = VM_VFMACC_VV (p, r4, q2, vl);
  /* y = (lo + r2*p) + hi */
  vm_f_t ym = VM_VFADD_VV (VM_VFMACC_VV (lo, r2, p, vl), hi, vl);

  vm_f_t y = VM_VFMERGE_VVM (ym, yn, m_near, vl);
  return VM_KFN (scalar_fixup) (x, y, m_special, vl, fixup);
}

/* ------------------------------------------------------------------ */
/* log1p(x) -- vectorization of fdlibm common/s_log1p.c.  All branches
   are closed-form; x <= -1 goes through scalar fixup (divzero at -1,
   invalid below), inf/nan computed as x+x inline.  */
#define VM_LN2_HI_L1P 6.93147180369123816490e-01
#define VM_LN2_LO_L1P 1.90821492927058770002e-10
#define VM_LP1 6.666666666666735130e-01
#define VM_LP2 3.999999999940941908e-01
#define VM_LP3 2.857142874366239149e-01
#define VM_LP4 2.222219843214978396e-01
#define VM_LP5 1.818357216161805012e-01
#define VM_LP6 1.531383769920937332e-01
#define VM_LP7 1.479819860511658591e-01

/* Horner: Lp1 + z*(Lp2 + z*(... z*Lp7)); each step is p <- Lp_k + z*p
   (vfmacc with the coefficient as accumulator, not p + z*Lp_k).  */
static inline vm_f_t
VM_KFN (log1p_poly) (vm_f_t z, size_t vl)
{
  vm_f_t p = VM_VFMV_V_F (VM_LP7, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP6, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP5, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP4, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP3, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP2, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_LP1, vl), z, p, vl);
  return p;
}

/* log1p with double-double output.  The fdlibm chain
   k*ln2_hi - ((hfsq - (core + t1)) - f) is not accurate enough as a
   plain double for log2p1, which scales by 1/ln2 and would inflate
   past 1 ulp in the same-binade case; as dd it stays ~2^-105
   relative.  */
#define VM_LOG1P_DD(yh, yl, x, vl, fixup) do {				\
  vm_u64_t vmlp__ix = VM_BITS (x);					\
  vm_i64_t vmlp__hx = VM_VSRA_VX (VM_REINT_U_I (vmlp__ix), 32, (vl));	\
  vm_u64_t vmlp__ab = VM_VAND_VX (vmlp__ix, VM_ABSMASK_U64, (vl));	\
  vm_b_t vmlp__neg = VM_VMSNE_VX (					\
    VM_VAND_VX (vmlp__ix, VM_SIGNBIT_U64, (vl)), 0, (vl));		\
  vm_b_t vmlp__dom = VM_VMAND_MM (vmlp__neg,				\
    VM_VMSGEU_VX (vmlp__ab, 0x3ff0000000000000ULL, (vl)), (vl));	\
  vm_b_t vmlp__infnan = VM_VMSGE_VX (vmlp__hx, (int64_t) 0x7ff00000, (vl)); \
  vm_b_t vmlp__lt54 = VM_VMSLTU_VX (vmlp__ab,				\
    0x3c90000000000000ULL, (vl));					\
  vm_b_t vmlp__lt29 = VM_VMSLTU_VX (vmlp__ab,				\
    0x3e20000000000000ULL, (vl));					\
  vmlp__lt29 = VM_VMAND_MM (vmlp__lt29,					\
    VM_VMNOT_M (vmlp__lt54, (vl)), (vl));				\
  vm_b_t vmlp__reg1 = VM_VMSLT_VX (vmlp__hx, (int64_t) 0x3fda827a, (vl)); \
  vm_b_t vmlp__mid = VM_VMAND_MM (vmlp__reg1,				\
    VM_VMOR_MM (VM_VMSGT_VX (vmlp__hx, 0, (vl)),			\
      VM_VMSLE_VX (vmlp__hx, (int64_t) (int32_t) 0xbfd2bec3, (vl)), (vl)), \
    (vl));								\
  vmlp__mid = VM_VMAND_MM (vmlp__mid, VM_VMNOT_M (			\
    VM_VMOR_MM (vmlp__lt29, vmlp__lt54, (vl)), (vl)), (vl));		\
  vm_b_t vmlp__k = VM_VMNOT_M (VM_VMOR_MM (				\
    VM_VMOR_MM (vmlp__dom, vmlp__infnan, (vl)),				\
    VM_VMOR_MM (VM_VMOR_MM (vmlp__lt54, vmlp__lt29, (vl)),		\
		vmlp__mid, (vl)), (vl)), (vl));				\
  vm_f_t vmlp__zh = VM_VFMV_V_F (0.0, (vl));				\
  /* dd accumulators, default = x (tiny region, exact).  */		\
  vm_f_t vmlp__rh = (x), vmlp__rl = vmlp__zh;				\
  /* ---- tiny2: x - x^2/2 via two_prod ---- */				\
  vm_f_t vmlp__x29 = VM_VFMERGE_VFM ((x), 0.0,				\
    VM_VMNOT_M (vmlp__lt29, (vl)), (vl));				\
  vm_f_t vmlp__ph, vmlp__pl;						\
  VM_TWO_PROD (vmlp__ph, vmlp__pl, vmlp__x29,				\
	       VM_VFMUL_VF (vmlp__x29, 0.5, (vl)), (vl));		\
  vm_f_t vmlp__th, vmlp__tl;						\
  VM_DD_SUB (vmlp__th, vmlp__tl, vmlp__x29, vmlp__zh,			\
	     vmlp__ph, vmlp__pl, (vl));					\
  vmlp__rh = VM_VFMERGE_VVM (vmlp__rh, vmlp__th, vmlp__lt29, (vl));	\
  vmlp__rl = VM_VFMERGE_VVM (vmlp__rl, vmlp__tl, vmlp__lt29, (vl));	\
  /* ---- shared core for mid (k=0) and k paths ---- */			\
  /* mid: f = x sanitized to 0.1 off-region.  */			\
  vm_f_t vmlp__xm = VM_VFMERGE_VFM ((x), 0.1,				\
    VM_VMNOT_M (vmlp__mid, (vl)), (vl));				\
  vm_f_t vmlp__hf_m = VM_VFMUL_VF (VM_VFMUL_VV (vmlp__xm, vmlp__xm, (vl)), \
				   0.5, (vl));				\
  vm_f_t vmlp__sm = VM_VFDIV_VV (vmlp__xm,				\
    VM_VFADD_VF (vmlp__xm, 2.0, (vl)), (vl));				\
  vm_f_t vmlp__zm = VM_VFMUL_VV (vmlp__sm, vmlp__sm, (vl));		\
  vm_f_t vmlp__Rm = VM_VFMUL_VV (vmlp__zm,				\
    VM_KFN (log1p_poly) (vmlp__zm, (vl)), (vl));			\
  vm_f_t vmlp__cm = VM_VFMUL_VV (vmlp__sm,				\
    VM_VFADD_VV (vmlp__hf_m, vmlp__Rm, (vl)), (vl));			\
  /* dd: f - (hfsq - core).  */						\
  VM_DD_SUB (vmlp__th, vmlp__tl, vmlp__hf_m, vmlp__zh,			\
	     vmlp__cm, vmlp__zh, (vl));					\
  vm_f_t vmlp__mh, vmlp__ml;						\
  VM_DD_SUB (vmlp__mh, vmlp__ml, vmlp__xm, vmlp__zh,			\
	     vmlp__th, vmlp__tl, (vl));					\
  vmlp__rh = VM_VFMERGE_VVM (vmlp__rh, vmlp__mh, vmlp__mid, (vl));	\
  vmlp__rl = VM_VFMERGE_VVM (vmlp__rl, vmlp__ml, vmlp__mid, (vl));	\
  /* ---- k path ---- */						\
  vm_f_t vmlp__xk = VM_VFMERGE_VFM ((x), 1.0,				\
    VM_VMNOT_M (vmlp__k, (vl)), (vl));					\
  vm_b_t vmlp__big = VM_VMSGE_VX (VM_VSRA_VX (				\
    VM_REINT_U_I (VM_BITS (vmlp__xk)), 32, (vl)),			\
    (int64_t) 0x43400000, (vl));					\
  vm_f_t vmlp__u = VM_VFMERGE_VVM (					\
    VM_VFADD_VF (vmlp__xk, 1.0, (vl)), vmlp__xk, vmlp__big, (vl));	\
  vm_u64_t vmlp__ub = VM_BITS (vmlp__u);				\
  vm_i64_t vmlp__ki = VM_REINT_U_I (VM_VSUB_VX (			\
    VM_VSRL_VX (vmlp__ub, 52, (vl)), 1023, (vl)));			\
  vm_f_t vmlp__c1 = VM_VFRSUB_VF (					\
    VM_VFSUB_VV (vmlp__u, vmlp__xk, (vl)), 1.0, (vl));			\
  vm_f_t vmlp__c2 = VM_VFSUB_VV (vmlp__xk,				\
    VM_VFSUB_VF (vmlp__u, 1.0, (vl)), (vl));				\
  vm_f_t vmlp__c = VM_VFMERGE_VVM (vmlp__c2, vmlp__c1,			\
    VM_VMSGT_VX (vmlp__ki, 0, (vl)), (vl));				\
  vmlp__c = VM_VFMERGE_VFM (vmlp__c, 0.0, vmlp__big, (vl));		\
  vmlp__c = VM_VFDIV_VV (vmlp__c, vmlp__u, (vl));			\
  vm_u64_t vmlp__huf = VM_VAND_VX (VM_VSRL_VX (vmlp__ub, 32, (vl)),	\
				   0xfffff, (vl));			\
  vm_b_t vmlp__hus = VM_VMSLTU_VX (vmlp__huf, 0x6a09e, (vl));		\
  vm_u64_t vmlp__low = VM_VAND_VX (vmlp__ub, 0xffffffffULL, (vl));	\
  vm_u64_t vmlp__ubn = VM_VMERGE_VVM (					\
    VM_VOR_VV (vmlp__low, VM_VSLL_VX (				\
      VM_VOR_VX (vmlp__huf, 0x3fe00000, (vl)), 32, (vl)), (vl)),	\
    VM_VOR_VV (vmlp__low, VM_VSLL_VX (				\
      VM_VOR_VX (vmlp__huf, 0x3ff00000, (vl)), 32, (vl)), (vl)),	\
    vmlp__hus, (vl));							\
  vmlp__u = VM_FROM_BITS (vmlp__ubn);					\
  vmlp__ki = VM_REINT_U_I (VM_VMERGE_VVM (				\
    VM_VADD_VX (VM_REINT_I_U (vmlp__ki), 1, (vl)),			\
    VM_REINT_I_U (vmlp__ki), vmlp__hus, (vl)));				\
  vm_f_t vmlp__f = VM_VFSUB_VF (vmlp__u, 1.0, (vl));			\
  vm_f_t vmlp__hf = VM_VFMUL_VF (VM_VFMUL_VV (vmlp__f, vmlp__f, (vl)),	\
				 0.5, (vl));				\
  vm_u64_t vmlp__huf2 = VM_VMERGE_VVM (VM_VSRL_VX (			\
    VM_VSUB_VV (VM_VMV_V_X (0x100000, (vl)), vmlp__huf, (vl)), 2, (vl)), \
    vmlp__huf, vmlp__hus, (vl));					\
  vm_f_t vmlp__kd = VM_VFCVT_F_X_V (vmlp__ki, (vl));			\
  /* t1dd = c + kd*ln2_lo (dd).  */					\
  vm_f_t vmlp__t1h, vmlp__t1l;						\
  VM_TWO_PROD (vmlp__t1h, vmlp__t1l, vmlp__kd,				\
	       VM_VFMV_V_F (VM_LN2_LO_L1P, (vl)), (vl));		\
  vm_f_t vmlp__t1dh, vmlp__t1dl;					\
  VM_DD_ADD (vmlp__t1dh, vmlp__t1dl, vmlp__c, vmlp__zh,			\
	     vmlp__t1h, vmlp__t1l, (vl));				\
  vm_f_t vmlp__khi = VM_VFMUL_VF (vmlp__kd, VM_LN2_HI_L1P, (vl));	\
  vm_b_t vmlp__hu0 = VM_VMSEQ_VX (vmlp__huf2, 0, (vl));			\
  vm_b_t vmlp__f0 = VM_VMFEQ_VF (vmlp__f, 0.0, (vl));			\
  vm_b_t vmlp__k0 = VM_VMSEQ_VX (VM_REINT_I_U (vmlp__ki), 0, (vl));	\
  /* ---- branch A (hu == 0) ---- */					\
  vm_f_t vmlp__Ra = VM_VFMUL_VV (vmlp__hf, VM_VFNMSAC_VF (		\
    VM_VFMV_V_F (1.0, (vl)), 0.66666666666666663, vmlp__f, (vl)), (vl)); \
  /* yA_k = khi - ((Ra - t1) - f)  [all dd] */				\
  vm_f_t vmlp__ah, vmlp__al;						\
  VM_DD_SUB (vmlp__ah, vmlp__al, vmlp__Ra, vmlp__zh,			\
	     vmlp__t1dh, vmlp__t1dl, (vl));				\
  VM_DD_SUB (vmlp__ah, vmlp__al, vmlp__ah, vmlp__al,			\
	     vmlp__f, vmlp__zh, (vl));					\
  vm_f_t vmlp__yakh, vmlp__yakl;					\
  VM_DD_SUB (vmlp__yakh, vmlp__yakl, vmlp__khi, vmlp__zh,		\
	     vmlp__ah, vmlp__al, (vl));					\
  /* yA_k0 = f - Ra */							\
  vm_f_t vmlp__ya0h, vmlp__ya0l;					\
  VM_DD_SUB (vmlp__ya0h, vmlp__ya0l, vmlp__f, vmlp__zh,			\
	     vmlp__Ra, vmlp__zh, (vl));					\
  /* yA_f0 = khi + t1 (k != 0), else 0 */				\
  vm_f_t vmlp__yafh, vmlp__yafl;					\
  VM_DD_ADD (vmlp__yafh, vmlp__yafl, vmlp__khi, vmlp__zh,		\
	     vmlp__t1dh, vmlp__t1dl, (vl));				\
  vmlp__yafh = VM_VFMERGE_VFM (vmlp__yafh, 0.0, vmlp__k0, (vl));	\
  vmlp__yafl = VM_VFMERGE_VFM (vmlp__yafl, 0.0, vmlp__k0, (vl));	\
  vm_f_t vmlp__yah = VM_VFMERGE_VVM (vmlp__yakh, vmlp__ya0h,		\
				     vmlp__k0, (vl));			\
  vm_f_t vmlp__yal = VM_VFMERGE_VVM (vmlp__yakl, vmlp__ya0l,		\
				     vmlp__k0, (vl));			\
  vmlp__yah = VM_VFMERGE_VVM (vmlp__yah, vmlp__yafh, vmlp__f0, (vl));	\
  vmlp__yal = VM_VFMERGE_VVM (vmlp__yal, vmlp__yafl, vmlp__f0, (vl));	\
  /* ---- branch B (hu != 0) ---- */					\
  vm_f_t vmlp__s = VM_VFDIV_VV (vmlp__f,				\
    VM_VFADD_VF (vmlp__f, 2.0, (vl)), (vl));				\
  vm_f_t vmlp__z = VM_VFMUL_VV (vmlp__s, vmlp__s, (vl));		\
  vm_f_t vmlp__R = VM_VFMUL_VV (vmlp__z,				\
    VM_KFN (log1p_poly) (vmlp__z, (vl)), (vl));				\
  vm_f_t vmlp__core = VM_VFMUL_VV (vmlp__s,				\
    VM_VFADD_VV (vmlp__hf, vmlp__R, (vl)), (vl));			\
  /* inner = (hfsq - (core + t1)) - f ; y = khi - inner */		\
  vm_f_t vmlp__bh, vmlp__bl;						\
  VM_DD_ADD (vmlp__bh, vmlp__bl, vmlp__core, vmlp__zh,			\
	     vmlp__t1dh, vmlp__t1dl, (vl));				\
  VM_DD_SUB (vmlp__bh, vmlp__bl, vmlp__hf, vmlp__zh,			\
	     vmlp__bh, vmlp__bl, (vl));					\
  VM_DD_SUB (vmlp__bh, vmlp__bl, vmlp__bh, vmlp__bl,			\
	     vmlp__f, vmlp__zh, (vl));					\
  vm_f_t vmlp__ybh, vmlp__ybl;						\
  VM_DD_SUB (vmlp__ybh, vmlp__ybl, vmlp__khi, vmlp__zh,			\
	     vmlp__bh, vmlp__bl, (vl));					\
  /* yB_k0 = f - (hfsq - core) */					\
  vm_f_t vmlp__yb0h, vmlp__yb0l;					\
  VM_DD_SUB (vmlp__yb0h, vmlp__yb0l, vmlp__hf, vmlp__zh,		\
	     vmlp__core, vmlp__zh, (vl));				\
  VM_DD_SUB (vmlp__yb0h, vmlp__yb0l, vmlp__f, vmlp__zh,			\
	     vmlp__yb0h, vmlp__yb0l, (vl));				\
  vm_f_t vmlp__kbh = VM_VFMERGE_VVM (vmlp__ybh, vmlp__yb0h,		\
				     vmlp__k0, (vl));			\
  vm_f_t vmlp__kbl = VM_VFMERGE_VVM (vmlp__ybl, vmlp__yb0l,		\
				     vmlp__k0, (vl));			\
  /* select A/B by hu0 */						\
  vm_f_t vmlp__kh = VM_VFMERGE_VVM (vmlp__kbh, vmlp__yah,		\
				    vmlp__hu0, (vl));			\
  vm_f_t vmlp__kl = VM_VFMERGE_VVM (vmlp__kbl, vmlp__yal,		\
				    vmlp__hu0, (vl));			\
  vmlp__rh = VM_VFMERGE_VVM (vmlp__rh, vmlp__kh, vmlp__k, (vl));	\
  vmlp__rl = VM_VFMERGE_VVM (vmlp__rl, vmlp__kl, vmlp__k, (vl));	\
  /* ---- inf/nan lanes: x + x (masked; quiets sNaN, raises invalid \
     exactly like scalar); fixup for x <= -1.  ---- */			\
  vm_f_t vmlp__xpx = VM_VFADD_VV_M (vmlp__infnan, (x), (x), (vl));	\
  vmlp__rh = VM_VFMERGE_VVM (vmlp__rh, vmlp__xpx, vmlp__infnan, (vl));	\
  vmlp__rl = VM_VFMERGE_VFM (vmlp__rl, 0.0, vmlp__infnan, (vl));	\
  vm_f_t vmlp__yc = VM_VFADD_VV (vmlp__rh, vmlp__rl, (vl));		\
  /* +-0 dd pairs: -0 + 0 collapses to +0; restore the hi sign when the	\
     dd is exactly zero (log1p(+-0) = +-0).  */				\
  vm_b_t vmlp__z0 = VM_VMAND_MM (VM_VMFEQ_VF (vmlp__rh, 0.0, (vl)),	\
				 VM_VMFEQ_VF (vmlp__rl, 0.0, (vl)), (vl)); \
  vmlp__yc = VM_VFMERGE_VVM (vmlp__yc, vmlp__rh, vmlp__z0, (vl));	\
  vmlp__yc = VM_KFN (scalar_fixup) ((x), vmlp__yc, vmlp__dom, (vl), (fixup)); \
  (yh) = vmlp__yc;							\
  (yl) = VM_VFMERGE_VFM (vmlp__rl, 0.0, vmlp__dom, (vl));		\
} while (0)

static inline vm_f_t
VM_KFN (log1p_k) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_f_t yh, yl;
  VM_LOG1P_DD (yh, yl, x, vl, fixup);
  vm_f_t y;
  VM_DD_COLLAPSE (y, yh, yl, vl);
  /* +-0 inputs: log1p(+-0) = +-0; the dd collapse (-0 + 0) would give
     +0.  (Fixup lanes are never zero.)  */
  return VM_VFMERGE_VVM (y, x, VM_VMFEQ_VF (x, 0.0, vl), vl);
}

/* ------------------------------------------------------------------ */
/* log10(x) = log(x) * log10(e), scaled as dd before the final rounding.
   fixup must be scalar log(); the dd scale preserves the fixup's
   +-inf/NaN results (log10e > 0).  */
static inline vm_f_t
VM_KFN (log10) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_f_t lgh, lgl;
  VM_LOG_DD (lgh, lgl, x, vl, fixup);	/* fixup = scalar log */
  vm_u64_t cls = VM_VFCLASS_V (lgh, vl);
  vm_b_t m_fin = VM_VMSNE_VX (VM_VAND_VX (cls, VM_CLS_FINITE, vl), 0, vl);
  vm_f_t lgs = VM_VFMERGE_VFM (lgh, 1.0, VM_VMNOT_M (m_fin, vl), vl);
  vm_f_t lglz = VM_VFMERGE_VFM (lgl, 0.0, VM_VMNOT_M (m_fin, vl), vl);
  vm_f_t rh, rl;
  VM_DD_MUL (rh, rl, lgs, lglz,
	     VM_VFMV_V_F (VM_LOG10E_HI, vl), VM_VFMV_V_F (VM_LOG10E_LO, vl),
	     vl);
  vm_f_t y;
  VM_DD_COLLAPSE (y, rh, rl, vl);
  return VM_VFMERGE_VVM (lgh, y, m_fin, vl);
}

/* ------------------------------------------------------------------ */
/* log2p1(x) = log2(1+x) (SYCL): dd(log1p) * (1/ln2) before the final
   rounding -- never forms 1+x, accurate as x -> 0.  fixup: scalar
   log1p() (divzero at -1, invalid below; the scale by 1/ln2 > 0
   preserves +-inf/NaN pass-through).  */
static inline vm_f_t
VM_KFN (log2p1) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_f_t lph, lpl;
  VM_LOG1P_DD (lph, lpl, x, vl, fixup);
  vm_u64_t cls = VM_VFCLASS_V (lph, vl);
  vm_b_t m_fin = VM_VMSNE_VX (VM_VAND_VX (cls, VM_CLS_FINITE, vl), 0, vl);
  /* tiny lane scaling: for |lp| < 2^-960 the dd cross terms would be
     subnormal (spurious FE_UNDERFLOW); scale the dd pair by 2^60 (exact,
     both parts normal there) and undo after the collapse.  */
  vm_u64_t abl = VM_VAND_VX (VM_BITS (lph), VM_ABSMASK_U64, vl);
  vm_b_t m_tiny = VM_VMSLTU_VX (abl, 0x03f0000000000000ULL, vl);
  m_tiny = VM_VMAND_MM (m_tiny, m_fin, vl);
  vm_b_t m_dir = VM_VMAND_MM (VM_VMNOT_M (m_tiny, vl), m_fin, vl);
  /* direct dd path */
  vm_f_t lphs = VM_VFMERGE_VFM (lph, 1.0, VM_VMNOT_M (m_dir, vl), vl);
  vm_f_t lpls = VM_VFMERGE_VFM (lpl, 0.0, VM_VMNOT_M (m_dir, vl), vl);
  vm_f_t dh, dl;
  VM_DD_MUL (dh, dl, lphs, lpls,
	     VM_VFMV_V_F (VM_INVLN2_HI, vl), VM_VFMV_V_F (VM_INVLN2_LO, vl),
	     vl);
  vm_f_t ydir;
  VM_DD_COLLAPSE (ydir, dh, dl, vl);
  /* scaled dd path */
  vm_f_t tsh = VM_VFMERGE_VFM (VM_VFMUL_VF (lph, 0x1p60, vl), 1.0,
			       VM_VMNOT_M (m_tiny, vl), vl);
  vm_f_t tsl = VM_VFMERGE_VFM (VM_VFMUL_VF (lpl, 0x1p60, vl), 0.0,
			       VM_VMNOT_M (m_tiny, vl), vl);
  VM_DD_MUL (dh, dl, tsh, tsl,
	     VM_VFMV_V_F (VM_INVLN2_HI, vl), VM_VFMV_V_F (VM_INVLN2_LO, vl),
	     vl);
  vm_f_t ysc;
  VM_DD_COLLAPSE (ysc, dh, dl, vl);
  ysc = VM_VFMUL_VF (ysc, 0x1p-60, vl);
  vm_f_t y = VM_VFMERGE_VVM (ydir, ysc, m_tiny, vl);
  /* +-0 results keep their sign (dd +0 adds would lose -0).  */
  y = VM_VFMERGE_VVM (y, VM_VFMUL_VF (lph, VM_INVLN2_HI, vl),
		      VM_VMFEQ_VF (lph, 0.0, vl), vl);
  /* Underflow for tiny results: the 2^-60-scaled path can round to a
     value exactly on the subnormal grid although the true value is not
     representable, so the hardware never sets UF.  Raise it for
     0 < |y| < DBL_MIN lanes -- x/ln2 is not representable on that
     grid, so those results are always inexact.  The masked multiply
     keeps the flag on exactly those lanes and the masked add keeps the
     computation live without changing y (2^-2044 flushes to +-0).  */
  {
    vm_u64_t yb = VM_VAND_VX (VM_BITS (y), VM_ABSMASK_U64, vl);
    vm_b_t m_uf = VM_VMSLTU_VX (VM_VSUB_VX (yb, 1, vl),
				0x000fffffffffffffULL, vl);
    /* _MU: the _m form's inactive lanes are unspecified for
       two-operand results.  */
    vm_f_t uf = VM_VFMUL_VF_MU (m_uf, VM_VFMV_V_F (0.0, vl),
				VM_VFMV_V_F (0x1p-1022, vl), 0x1p-1022, vl);
    y = VM_VFADD_VV_MU (m_uf, y, y, uf, vl);
  }
  /* non-finite log1p results pass through (1/ln2 > 0).  */
  return VM_VFMERGE_VVM (lph, y, m_fin, vl);
}
