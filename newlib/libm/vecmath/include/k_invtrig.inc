/* k_invtrig.inc -- atan / atan2 kernels and their pi-scaled variants.

   LMUL-generic template: include after vecmath_rvvcompat.h, vecmath_dd.h
   and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.  Also needs
   k_trig.inc for the pi constants (VM_INVPI_HI / VM_INVPI_LO).

   Lean rewrite: the base kernels are plain-double ports of fdlibm
   math/s_atan.c, math/e_atan2.c, math/e_asin.c and math/e_acos.c with
   the *final composition* carried as a double-double pair (a handful of
   error-free transforms), so the pi-scaled variants can multiply by 1/pi
   without paying an extra rounding.  The former 100%-dd atan core (4
   divisions, ~300 vector ops) is gone: one division for the reduction,
   one for its quotient residual and, for callers that pass a tail, one
   for the tail.

   The s_atan.c branch structure is evaluated branch-free on the masked
   lane classes; the |x| < 2^-29 and |x| >= 2^66 lanes are fed a benign
   reduction value and patched at the end, exactly as fdlibm answers
   them before doing any arithmetic.  */

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

/* 1/pi as a double-double.  */
#ifndef VM_INVPI_HI
#define VM_INVPI_HI 3.18309886183790691216e-01
#define VM_INVPI_LO -1.96786766751824858819e-17
#endif

/* Collapse a dd pair without folding an exact signed zero: -0 + +0 is
   +0 and atan(+-0) must stay +-0.  Where lo is exactly zero the value
   *is* hi.  vmfeq is quiet, so a NaN lane raises nothing.  */
#define VM_INVTRIG_COLLAPSE(r, ah, al, vl) do {				\
  (r) = VM_VFADD_VV ((ah), (al), (vl));					\
  (r) = VM_VFMERGE_VVM ((r), (ah), VM_VMFEQ_VF ((al), 0.0, (vl)), (vl)); \
} while (0)

/* y = correctly rounded (ah + al)/pi, from the dd pair (ah, al).
   Both parts are scaled by 2^900 first: the dd corrections are ~2^-53 of
   the product, and unscaled they would land in the subnormal range (a
   spurious UNDERFLOW on a lane whose answer is normal).  The scale-up is
   exact for every finite dd part, the scale-back is exact whenever the
   answer is normal, and the final rounding happens once, on the scaled
   pair -- the dd carries 1/pi to 2^-107.  Non-finite parts are parked at
   a benign value; the callers restore their results.  */
#define VM_MUL_INVPI_DD(y, ah, al, vl) do {				\
  vm_u64_t vmpi__ab = VM_VAND_VX (VM_BITS (ah), VM_ABSMASK_U64, (vl));	\
  vm_b_t vmpi__mnf = VM_VMSGEU_VX (vmpi__ab, 0x7ff0000000000000ULL, (vl)); \
  vm_f_t vmpi__a = VM_VFMERGE_VFM ((ah), 0.5, vmpi__mnf, (vl));		\
  vm_f_t vmpi__l = VM_VFMERGE_VFM ((al), 0.0, vmpi__mnf, (vl));		\
  vm_f_t vmpi__hs = VM_VFMUL_VF (vmpi__a, 0x1p900, (vl));		\
  vm_f_t vmpi__ls = VM_VFMUL_VF (vmpi__l, 0x1p900, (vl));		\
  vm_f_t vmpi__p = VM_VFMUL_VF (vmpi__hs, VM_INVPI_HI, (vl));		\
  vm_f_t vmpi__e = VM_VFMACC_VF (VM_VFNEG_V (vmpi__p, (vl)),		\
				 VM_INVPI_HI, vmpi__hs, (vl));		\
  vmpi__e = VM_VFMACC_VF (vmpi__e, VM_INVPI_LO, vmpi__hs, (vl));	\
  vmpi__e = VM_VFMACC_VF (vmpi__e, VM_INVPI_HI, vmpi__ls, (vl));	\
  vm_f_t vmpi__s, vmpi__se;						\
  VM_TWO_SUM (vmpi__s, vmpi__se, vmpi__p, vmpi__e, (vl));		\
  VM_INVTRIG_COLLAPSE ((y), vmpi__s, vmpi__se, (vl));			\
  (y) = VM_VFMUL_VF ((y), 0x1p-900, (vl));				\
  /* A subnormal (or zero) answer here is always inexact -- the value is
     x/pi with x rational and nonzero -- so the reference raises
     UNDERFLOW; do the same explicitly (the scaled product sometimes
     rounds exactly onto the subnormal grid and stays silent).  */	\
  {									\
    vm_u64_t vmpi__yb = VM_VAND_VX (VM_BITS (y), VM_ABSMASK_U64, (vl));	\
    vm_u64_t vmpi__hb = VM_VAND_VX (VM_BITS (vmpi__a), VM_ABSMASK_U64, (vl)); \
    vm_b_t vmpi__mu = VM_VMAND_MM (					\
      VM_VMAND_MM (VM_VMSLTU_VX (vmpi__yb, 0x0010000000000000ULL, (vl)),	\
		   VM_VMSNE_VX (vmpi__hb, 0, (vl)), (vl)),		\
      VM_VMNOT_M (vmpi__mnf, (vl)), (vl));				\
    vm_f_t vmpi__uf = VM_VFMUL_VF_MU (vmpi__mu, VM_VFMV_V_F (0.0, (vl)),	\
				      VM_VFMV_V_F (0x1p-1022, (vl)),	\
				      0x1p-1022, (vl));			\
    (y) = VM_VFADD_VV_MU (vmpi__mu, (y), (y), vmpi__uf, (vl));		\
  }									\
  /* exact zeros keep the sign of the dd (copysign(0, ah)).  */		\
  (y) = VM_VFMERGE_VVM ((y), VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, (vl)), (ah), \
					    (vl)),			\
			VM_VMFEQ_VF ((y), 0.0, (vl)), (vl));		\
} while (0)

/* ------------------------------------------------------------------ */
/* VM_ATAN_DD(yh, yl, x, xlo, vl): atan(x + xlo) as a dd pair, xlo the
   tail of x (zero vector for plain atan).  fdlibm s_atan.c; the final
   composition hi - ((r*S - lo) - r) is evaluated with error-free
   transforms so the pair carries ~106 bits.  */
#define VM_ATAN_DD(yh, yl, x, xlo, vl) do {				\
  vm_u64_t vma__bx = VM_BITS (x);					\
  vm_i64_t vma__hx = VM_VSRA_VX (VM_REINT_U_I (vma__bx), 32, (vl));	\
  vm_u64_t vma__ix = VM_VAND_VX (VM_REINT_I_U (vma__hx), 0x7fffffff, (vl)); \
  vm_u64_t vma__ab = VM_VAND_VX (vma__bx, VM_ABSMASK_U64, (vl));	\
  vm_b_t vma__mneg = VM_VMSLT_VX (vma__hx, 0, (vl));			\
  vm_b_t vma__minf = VM_VMSGEU_VX (vma__ab, 0x7ff0000000000000ULL, (vl)); \
  vm_b_t vma__mnan = VM_VMSGTU_VX (vma__ab, 0x7ff0000000000000ULL, (vl)); \
  vm_b_t vma__mbig = VM_VMSGEU_VX (vma__ix, 0x44100000, (vl));		\
  vm_b_t vma__mtiny = VM_VMSLTU_VX (vma__ix, 0x3e200000, (vl));		\
  vm_b_t vma__msmall = VM_VMSLTU_VX (vma__ix, 0x3fdc0000, (vl));	\
  vm_b_t vma__m0 = VM_VMAND_MM (VM_VMNOT_M (vma__msmall, (vl)),		\
				VM_VMSLTU_VX (vma__ix, 0x3fe60000, (vl)), (vl)); \
  vm_b_t vma__m1 = VM_VMAND_MM (VM_VMNOT_M (vma__msmall, (vl)),		\
    VM_VMAND_MM (VM_VMSGEU_VX (vma__ix, 0x3fe60000, (vl)),		\
		 VM_VMSLTU_VX (vma__ix, 0x3ff30000, (vl)), (vl)), (vl)); \
  vm_b_t vma__m2 = VM_VMAND_MM (VM_VMNOT_M (vma__msmall, (vl)),		\
    VM_VMAND_MM (VM_VMSGEU_VX (vma__ix, 0x3ff30000, (vl)),		\
		 VM_VMSLTU_VX (vma__ix, 0x40038000, (vl)), (vl)), (vl)); \
  vm_b_t vma__m3 = VM_VMAND_MM (VM_VMNOT_M (vma__msmall, (vl)),		\
				VM_VMSGEU_VX (vma__ix, 0x40038000, (vl)), (vl)); \
  vm_b_t vma__mg = VM_VMOR_MM (VM_VMOR_MM (vma__mtiny, vma__mbig, (vl)), \
			       vma__minf, (vl));			\
  vm_f_t vma__ag = VM_VFMERGE_VFM (VM_VFABS_V (x, (vl)), 0.5,		\
				   vma__mg, (vl));			\
  vm_f_t vma__c = VM_VFMV_V_F (0.0, (vl));				\
  vma__c = VM_VFMERGE_VFM (vma__c, 1.5, vma__m2, (vl));			\
  vma__c = VM_VFMERGE_VFM (vma__c, 1.0, vma__m1, (vl));			\
  vma__c = VM_VFMERGE_VFM (vma__c, 0.5, vma__m0, (vl));			\
  /* r = (ag - c)/(1 + c*ag), or -1/ag on the last branch.  */		\
  vm_f_t vma__rnum = VM_VFMERGE_VFM (VM_VFSUB_VV (vma__ag, vma__c, (vl)), \
				     -1.0, vma__m3, (vl));		\
  vm_f_t vma__rden = VM_VFMERGE_VVM (					\
    VM_VFMACC_VV (VM_VFMV_V_F (1.0, (vl)), vma__c, vma__ag, (vl)),	\
    vma__ag, vma__m3, (vl));						\
  vm_f_t vma__r = VM_VFDIV_VV (vma__rnum, vma__rden, (vl));		\
  /* exact residual of that division; without it one rounding of r is
     lost from every result.  */					\
  vm_f_t vma__rr = VM_VFDIV_VV (					\
    VM_VFNMSAC_VV (vma__rnum, vma__r, vma__rden, (vl)),			\
    vma__rden, (vl));							\
  /* tail: dr/dx = (1+r^2)/(1+x^2) for every branch of the reduction.  */ \
  vm_f_t vma__rlo = VM_VFMUL_VV (					\
    (xlo), VM_VFDIV_VV (						\
      VM_VFMACC_VV (VM_VFMV_V_F (1.0, (vl)), vma__r, vma__r, (vl)),	\
      VM_VFMACC_VV (VM_VFMV_V_F (1.0, (vl)), vma__ag, vma__ag, (vl)),	\
      (vl)), (vl));							\
  vm_f_t vma__rt, vma__rte0, vma__rte;					\
  VM_TWO_SUM (vma__rt, vma__rte0, vma__r, vma__rlo, (vl));		\
  vma__rte = VM_VFADD_VV (vma__rte0, vma__rr, (vl));			\
  vm_f_t vma__z = VM_VFMUL_VV (vma__rt, vma__rt, (vl));			\
  /* S = z*(aT0 + z*(aT1 + ... + z*aT10)); one Horner chain (the
     even/odd split only buys scalar ILP).  */				\
  vm_f_t vma__H = VM_VFMACC_VF (VM_VFMV_V_F (VM_AT9, (vl)), VM_AT10,	\
				vma__z, (vl));				\
  vma__H = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT8, (vl)), vma__z, vma__H, (vl)); \
  vma__H = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT7, (vl)), vma__z, vma__H, (vl)); \
  vma__H = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT6, (vl)), vma__z, vma__H, (vl)); \
  vma__H = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT5, (vl)), vma__z, vma__H, (vl)); \
  vma__H = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT4, (vl)), vma__z, vma__H, (vl)); \
  vma__H = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT3, (vl)), vma__z, vma__H, (vl)); \
  vma__H = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT2, (vl)), vma__z, vma__H, (vl)); \
  vma__H = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT1, (vl)), vma__z, vma__H, (vl)); \
  vma__H = VM_VFMACC_VV (VM_VFMV_V_F (VM_AT0, (vl)), vma__z, vma__H, (vl)); \
  vm_f_t vma__ss = VM_VFMUL_VV (vma__z, vma__H, (vl));			\
  vm_f_t vma__hi = VM_VFMV_V_F (0.0, (vl));				\
  vma__hi = VM_VFMERGE_VFM (vma__hi, VM_ATANHI3, vma__m3, (vl));	\
  vma__hi = VM_VFMERGE_VFM (vma__hi, VM_ATANHI2, vma__m2, (vl));	\
  vma__hi = VM_VFMERGE_VFM (vma__hi, VM_ATANHI1, vma__m1, (vl));	\
  vma__hi = VM_VFMERGE_VFM (vma__hi, VM_ATANHI0, vma__m0, (vl));	\
  vm_f_t vma__lo = VM_VFMV_V_F (0.0, (vl));				\
  vma__lo = VM_VFMERGE_VFM (vma__lo, VM_ATANLO3, vma__m3, (vl));	\
  vma__lo = VM_VFMERGE_VFM (vma__lo, VM_ATANLO2, vma__m2, (vl));	\
  vma__lo = VM_VFMERGE_VFM (vma__lo, VM_ATANLO1, vma__m1, (vl));	\
  vma__lo = VM_VFMERGE_VFM (vma__lo, VM_ATANLO0, vma__m0, (vl));	\
  /* y = hi - ((rt*S - lo) - rt), every rounding below carried.  */	\
  vm_f_t vma__A = VM_VFMUL_VV (vma__rt, vma__ss, (vl));			\
  vm_f_t vma__Ae = VM_VFNEG_V (						\
    VM_VFNMSAC_VV (vma__A, vma__rt, vma__ss, (vl)), (vl));		\
  vm_f_t vma__B, vma__Be;						\
  VM_TWO_SUM (vma__B, vma__Be, vma__A, VM_VFNEG_V (vma__lo, (vl)), (vl)); \
  vm_f_t vma__C, vma__Ce;						\
  VM_TWO_SUM (vma__C, vma__Ce, vma__B, VM_VFNEG_V (vma__rt, (vl)), (vl)); \
  vm_f_t vma__Plo = VM_VFSUB_VV (					\
    VM_VFADD_VV (VM_VFADD_VV (vma__Ce, vma__Be, (vl)), vma__Ae, (vl)),	\
    vma__rte, (vl));							\
  VM_TWO_SUM ((yh), (yl), vma__hi, VM_VFNEG_V (vma__C, (vl)), (vl));	\
  (yl) = VM_VFSUB_VV ((yl), vma__Plo, (vl));				\
  (yh) = VM_VFMERGE_VVM ((yh), VM_VFNEG_V ((yh), (vl)), vma__mneg, (vl)); \
  (yl) = VM_VFMERGE_VVM ((yl), VM_VFNEG_V ((yl), (vl)), vma__mneg, (vl)); \
  /* |x| < 2^-29: fdlibm returns x; the tail still rides along.  */	\
  {									\
    vm_b_t vma__mn = VM_VMNOT_M (vma__mtiny, (vl));			\
    vm_f_t vma__th, vma__tl;						\
    VM_TWO_SUM (vma__th, vma__tl,					\
		VM_VFMERGE_VFM (x, 0.5, vma__mn, (vl)),			\
		VM_VFMERGE_VFM ((xlo), 0.0, vma__mn, (vl)), (vl));	\
    /* two_sum folds -0 + +0 into +0; atan(+-0) must stay +-0.  */	\
    vma__th = VM_VFMERGE_VVM (vma__th, x,				\
			      VM_VMFEQ_VF (x, 0.0, (vl)), (vl));		\
    (yh) = VM_VFMERGE_VVM ((yh), vma__th, vma__mtiny, (vl));		\
    (yl) = VM_VFMERGE_VVM ((yl), vma__tl, vma__mtiny, (vl));		\
  }									\
  /* |x| >= 2^66 (and +-inf): atanhi[3] + atanlo[3].  */		\
  {									\
    vm_f_t vma__iv = VM_VFMV_V_F (VM_ATANHI3 + VM_ATANLO3, (vl));	\
    vm_f_t vma__is = VM_VFMERGE_VVM (					\
      vma__iv, VM_VFNEG_V (vma__iv, (vl)), vma__mneg, (vl));		\
    (yh) = VM_VFMERGE_VVM ((yh), vma__is, vma__mbig, (vl));		\
    (yl) = VM_VFMERGE_VFM ((yl), 0.0, vma__mbig, (vl));			\
  }									\
  /* NaN: x + x (quiet for qNaN, invalid for sNaN).  */			\
  {									\
    vm_f_t vma__nn = VM_VFADD_VV_M (vma__mnan, (x), (x), (vl));		\
    (yh) = VM_VFMERGE_VVM ((yh), vma__nn, vma__mnan, (vl));		\
    (yl) = VM_VFMERGE_VFM ((yl), 0.0, vma__mnan, (vl));			\
  }									\
} while (0)

/* atan(x) as a dd pair without any tail term.  */
#define VM_ATAN_ONLY_DD(yh, yl, x, vl)					\
  VM_ATAN_DD ((yh), (yl), (x), VM_VFMV_V_F (0.0, (vl)), (vl))

/* ------------------------------------------------------------------ */
/* atan2(y, x) as a dd pair.  fdlibm e_atan2.c with the quotient's own
   rounding error recovered (y - q*x is exact by fma) and fed to the
   atan kernel as a tail.  */
#define VM_ATAN2_DD(yh, yl, mo, y, x, vl) do {				\
  vm_i64_t vma2__hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, (vl)); \
  vm_u64_t vma2__ix = VM_VAND_VX (VM_REINT_I_U (vma2__hx), 0x7fffffff, (vl)); \
  vm_u64_t vma2__lx = VM_VAND_VX (VM_BITS (x), 0xffffffffULL, (vl));	\
  vm_u64_t vma2__by = VM_BITS (y);					\
  vm_u64_t vma2__ly = VM_VAND_VX (vma2__by, 0xffffffffULL, (vl));	\
  vm_u64_t vma2__iy = VM_VAND_VX (VM_VSRL_VX (vma2__by, 32, (vl)),	\
				  0x7fffffff, (vl));			\
  vm_b_t vma2__mxs = VM_VMSLT_VX (vma2__hx, 0, (vl));			\
  vm_b_t vma2__mys = VM_VMSLT_VX (VM_REINT_U_I (vma2__by), 0, (vl));	\
  vm_b_t vma2__m0 = VM_VMAND_MM (VM_VMNOT_M (vma2__mxs, (vl)),		\
				 VM_VMNOT_M (vma2__mys, (vl)), (vl));	\
  vm_b_t vma2__m1 = VM_VMAND_MM (VM_VMNOT_M (vma2__mxs, (vl)),		\
				 vma2__mys, (vl));			\
  vm_b_t vma2__m2 = VM_VMAND_MM (vma2__mxs,				\
				 VM_VMNOT_M (vma2__mys, (vl)), (vl));	\
  vm_b_t vma2__m3 = VM_VMAND_MM (vma2__mxs, vma2__mys, (vl));		\
  vm_b_t vma2__mnan = VM_VMOR_MM (					\
    VM_VMOR_MM (VM_VMSGTU_VX (vma2__ix, 0x7ff00000, (vl)),		\
		VM_VMAND_MM (VM_VMSEQ_VX (vma2__ix, 0x7ff00000, (vl)),	\
			     VM_VMSNE_VX (vma2__lx, 0, (vl)), (vl)), (vl)), \
    VM_VMOR_MM (VM_VMSGTU_VX (vma2__iy, 0x7ff00000, (vl)),		\
		VM_VMAND_MM (VM_VMSEQ_VX (vma2__iy, 0x7ff00000, (vl)),	\
			     VM_VMSNE_VX (vma2__ly, 0, (vl)), (vl)), (vl)), (vl)); \
  vm_b_t vma2__myzero = VM_VMAND_MM (VM_VMSEQ_VX (vma2__iy, 0, (vl)),	\
				     VM_VMSEQ_VX (vma2__ly, 0, (vl)), (vl)); \
  vm_b_t vma2__mxzero = VM_VMAND_MM (VM_VMSEQ_VX (vma2__ix, 0, (vl)),	\
				     VM_VMSEQ_VX (vma2__lx, 0, (vl)), (vl)); \
  vm_b_t vma2__mxinf = VM_VMAND_MM (VM_VMSEQ_VX (vma2__ix, 0x7ff00000, (vl)), \
				    VM_VMSEQ_VX (vma2__lx, 0, (vl)), (vl)); \
  vm_b_t vma2__myinf = VM_VMAND_MM (VM_VMSEQ_VX (vma2__iy, 0x7ff00000, (vl)), \
				    VM_VMSEQ_VX (vma2__ly, 0, (vl)), (vl)); \
  vm_i64_t vma2__k = VM_VSRA_VX (					\
    VM_REINT_U_I (VM_VSUB_VV (vma2__iy, vma2__ix, (vl))), 20, (vl));	\
  vm_b_t vma2__mhi = VM_VMSGT_VX (vma2__k, 60, (vl));			\
  vm_b_t vma2__mlo = VM_VMAND_MM (vma2__mxs,				\
				  VM_VMSLT_VX (vma2__k, -60, (vl)), (vl)); \
  vm_b_t vma2__mshort = VM_VMOR_MM (					\
    VM_VMOR_MM (VM_VMOR_MM (vma2__mhi, vma2__mlo, (vl)),		\
		VM_VMOR_MM (vma2__mxinf, vma2__myinf, (vl)), (vl)),	\
    VM_VMAND_MM (vma2__mxzero,					\
		 VM_VMNOT_M (vma2__myzero, (vl)), (vl)), (vl));		\
  vm_b_t vma2__mspec = VM_VMOR_MM (					\
    VM_VMOR_MM (vma2__mshort,					\
		VM_VMOR_MM (vma2__myzero, vma2__mxzero, (vl)), (vl)),	\
    vma2__mnan, (vl));							\
  /* the division runs on sanitized operands */			\
  vm_f_t vma2__qx = VM_VFMERGE_VFM (x, 1.0,				\
    VM_VMOR_MM (vma2__mxzero, vma2__mshort, (vl)), (vl));		\
  vm_f_t vma2__qy = VM_VFMERGE_VFM (y, 0.0, vma2__mshort, (vl));	\
  vm_f_t vma2__q = VM_VFDIV_VV (vma2__qy, vma2__qx, (vl));		\
  /* tail = (y - q*x)/x, but only where every step stays normal: the
     fma residual scales with |y| and the division with |q|, so both are
     masked into [2^-900, 2^900); elsewhere the tail is below the
     subnormal grid relative to any answer that can notice it.  */	\
  vm_u64_t vma2__qab = VM_VAND_VX (VM_BITS (vma2__q), VM_ABSMASK_U64, (vl)); \
  vm_u64_t vma2__yab = VM_VAND_VX (VM_BITS (vma2__qy), VM_ABSMASK_U64, (vl)); \
  vm_b_t vma2__mtail = VM_VMAND_MM (					\
    VM_VMAND_MM (							\
      VM_VMSGEU_VX (vma2__qab, 0x07b0000000000000ULL, (vl)),		\
      VM_VMSGEU_VX (vma2__yab, 0x07b0000000000000ULL, (vl)), (vl)),	\
    VM_VMNOT_M (vma2__mspec, (vl)), (vl));				\
  vm_f_t vma2__rem = VM_VFMACC_VV_M (					\
    vma2__mtail, vma2__qy, VM_VFNEG_V (vma2__q, (vl)), vma2__qx, (vl)); \
  vm_f_t vma2__qlo = VM_VFDIV_VV_MU (					\
    vma2__mtail, VM_VFMV_V_F (0.0, (vl)), vma2__rem, vma2__qx, (vl));	\
  /* A subnormal |y| puts the residual below the grid (the fma would even
     raise a spurious UNDERFLOW), yet its quotient tail is still worth up
     to half an ulp.  Recompute just those lanes at 2^600, where every
     step is normal, then bring the tail back down (exact: |q| stays above
     2^-900 there, so the scaled tail is >= 2^-353).  */			\
  {									\
    vm_b_t vma2__mneed = VM_VMAND_MM (					\
      VM_VMAND_MM (VM_VMSLTU_VX (vma2__yab, 0x07b0000000000000ULL, (vl)), \
		   VM_VMSGEU_VX (vma2__qab, 0x07b0000000000000ULL, (vl)), (vl)), \
      VM_VMNOT_M (vma2__mspec, (vl)), (vl));				\
    vm_f_t vma2__qsc = VM_VFMUL_VF_MU (vma2__mneed,			\
				       VM_VFMV_V_F (0.0, (vl)),		\
				       vma2__q, 0x1p600, (vl));		\
    vm_f_t vma2__qys = VM_VFMUL_VF_MU (vma2__mneed,			\
				       VM_VFMV_V_F (0.0, (vl)),		\
				       vma2__qy, 0x1p600, (vl));	\
    vm_f_t vma2__rems = VM_VFMACC_VV_M (				\
      vma2__mneed, vma2__qys, VM_VFNEG_V (vma2__qsc, (vl)), vma2__qx, (vl)); \
    vm_f_t vma2__qlos = VM_VFDIV_VV_MU (				\
      vma2__mneed, VM_VFMV_V_F (0.0, (vl)), vma2__rems, vma2__qx, (vl)); \
    vma2__qlos = VM_VFMUL_VF (vma2__qlos, 0x1p-600, (vl));		\
    vma2__qlo = VM_VFMERGE_VVM (vma2__qlo, vma2__qlos, vma2__mneed, (vl)); \
  }									\
  vm_f_t vma2__aq = VM_VFABS_V (vma2__q, (vl));				\
  vm_f_t vma2__qsg = VM_VFMERGE_VFM (VM_VFMV_V_F (-1.0, (vl)), 1.0,	\
				     VM_VMOR_MM (vma2__m0, vma2__m3, (vl)), (vl)); \
  vm_f_t vma2__atail = VM_VFMUL_VV (vma2__qlo, vma2__qsg, (vl));	\
  /* A subnormal quotient rounds to a coarse grid, so |q| itself can be
     many ulps off -- but atan(q) ~ q there, and the exact quotient can be
     recovered at 2^600 (x > 0 lanes only; for x < 0 the answer is pi to
     within an ulp anyway).  The whole atan is then evaluated in the
     scaled world and the callers divide the scale out.  */		\
  vm_b_t vma2__msc = VM_VMAND_MM (					\
    VM_VMSLTU_VX (vma2__qab, 0x07b0000000000000ULL, (vl)),		\
    VM_VMNOT_M (VM_VMOR_MM (vma2__mspec, vma2__mxs, (vl)), (vl)), (vl)); \
  vm_f_t vma2__azh, vma2__azl;						\
  VM_ATAN_DD (vma2__azh, vma2__azl, vma2__aq, vma2__atail, (vl));	\
  /* |q| > 2^60: pi/2 (fdlibm substitutes pi_o_2 + 0.5*pi_lo).  */	\
  vma2__azh = VM_VFMERGE_VFM (vma2__azh, VM_ATAN2_PIO2, vma2__mhi, (vl)); \
  vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__mhi, (vl));	\
  /* x < 0 and |y/x| < 2^-60: exactly 0 (quadrant patch adds pi).  */	\
  vma2__azh = VM_VFMERGE_VFM (vma2__azh, 0.0, vma2__mlo, (vl));	\
  vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__mlo, (vl));	\
  vma2__azh = VM_VFMERGE_VVM (vma2__azh, VM_VFNEG_V (vma2__azh, (vl)),	\
			      vma2__m1, (vl));				\
  vma2__azl = VM_VFMERGE_VVM (vma2__azl, VM_VFNEG_V (vma2__azl, (vl)),	\
			      vma2__m1, (vl));				\
  /* x < 0 quadrants: pi - az and az - pi, with (pi, pi_lo) an exact dd
     for pi, so neither subtraction rounds.  */				\
  {									\
    vm_f_t vma2__u1, vma2__u2, vma2__v1, vma2__v2;			\
    VM_TWO_SUM (vma2__u1, vma2__u2, VM_VFMV_V_F (VM_ATAN2_PI, (vl)),	\
		VM_VFNEG_V (vma2__azh, (vl)), (vl));			\
    vma2__u2 = VM_VFADD_VV (vma2__u2,					\
      VM_VFSUB_VV (VM_VFMV_V_F (VM_ATAN2_PILO, (vl)), vma2__azl, (vl)), (vl)); \
    VM_TWO_SUM (vma2__v1, vma2__v2, vma2__azh,				\
		VM_VFNEG_V (VM_VFMV_V_F (VM_ATAN2_PI, (vl)), (vl)), (vl)); \
    vma2__v2 = VM_VFADD_VV (vma2__v2,					\
      VM_VFSUB_VV (vma2__azl, VM_VFMV_V_F (VM_ATAN2_PILO, (vl)), (vl)), (vl)); \
    vma2__azh = VM_VFMERGE_VVM (vma2__azh, vma2__u1, vma2__m2, (vl));	\
    vma2__azl = VM_VFMERGE_VVM (vma2__azl, vma2__u2, vma2__m2, (vl));	\
    vma2__azh = VM_VFMERGE_VVM (vma2__azh, vma2__v1, vma2__m3, (vl));	\
    vma2__azl = VM_VFMERGE_VVM (vma2__azl, vma2__v2, vma2__m3, (vl));	\
  }									\
  /* y == 0: +-0 for x > 0, +-pi for x < 0 (answered before x == 0).  */ \
  {									\
    vm_b_t vma2__mz01 = VM_VMAND_MM (vma2__myzero,			\
				     VM_VMOR_MM (vma2__m0, vma2__m1, (vl)), (vl)); \
    vma2__azh = VM_VFMERGE_VVM (vma2__azh, y, vma2__mz01, (vl));	\
    vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__mz01, (vl));	\
    vm_b_t vma2__mz2 = VM_VMAND_MM (vma2__myzero, vma2__m2, (vl));	\
    vma2__azh = VM_VFMERGE_VFM (vma2__azh, VM_ATAN2_PI, vma2__mz2, (vl)); \
    vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__mz2, (vl));	\
    vm_b_t vma2__mz3 = VM_VMAND_MM (vma2__myzero, vma2__m3, (vl));	\
    vma2__azh = VM_VFMERGE_VFM (vma2__azh, -VM_ATAN2_PI, vma2__mz3, (vl)); \
    vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__mz3, (vl));	\
  }									\
  /* x == 0 (y != 0): +-pi/2 */						\
  {									\
    vm_b_t vma2__mxnz = VM_VMAND_MM (vma2__mxzero,			\
				     VM_VMNOT_M (vma2__myzero, (vl)), (vl)); \
    vm_f_t vma2__p2s = VM_VFMERGE_VVM (VM_VFMV_V_F (VM_ATAN2_PIO2, (vl)), \
	VM_VFNEG_V (VM_VFMV_V_F (VM_ATAN2_PIO2, (vl)), (vl)),		\
	VM_VMOR_MM (vma2__m1, vma2__m3, (vl)), (vl));			\
    vma2__azh = VM_VFMERGE_VVM (vma2__azh, vma2__p2s, vma2__mxnz, (vl)); \
    vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__mxnz, (vl));	\
  }									\
  /* x = +-Inf (y finite): 0/-0 for x > 0, pi/-pi for x < 0.  */		\
  {									\
    vm_b_t vma2__mxif = VM_VMAND_MM (vma2__mxinf,			\
				     VM_VMNOT_M (vma2__myinf, (vl)), (vl)); \
    vm_f_t vma2__pv = VM_VFMERGE_VFM (VM_VFMV_V_F (0.0, (vl)), VM_ATAN2_PI, \
	VM_VMOR_MM (vma2__m2, vma2__m3, (vl)), (vl));			\
    vma2__pv = VM_VFMERGE_VVM (vma2__pv, VM_VFNEG_V (vma2__pv, (vl)),	\
	VM_VMOR_MM (vma2__m1, vma2__m3, (vl)), (vl));			\
    vma2__azh = VM_VFMERGE_VVM (vma2__azh, vma2__pv, vma2__mxif, (vl));	\
    vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__mxif, (vl));	\
  }									\
  /* x = +-Inf, y = +-Inf: +-pi/4, +-3pi/4.  */				\
  {									\
    vm_b_t vma2__mxb = VM_VMAND_MM (vma2__mxinf, vma2__myinf, (vl));	\
    vm_f_t vma2__p4 = VM_VFMV_V_F (VM_ATAN2_PIO4, (vl));		\
    vma2__p4 = VM_VFMERGE_VFM (vma2__p4, 3.0 * VM_ATAN2_PIO4,		\
			       VM_VMOR_MM (vma2__m2, vma2__m3, (vl)), (vl)); \
    vma2__p4 = VM_VFMERGE_VVM (vma2__p4, VM_VFNEG_V (vma2__p4, (vl)),	\
	VM_VMOR_MM (vma2__m1, vma2__m3, (vl)), (vl));			\
    vma2__azh = VM_VFMERGE_VVM (vma2__azh, vma2__p4, vma2__mxb, (vl));	\
    vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__mxb, (vl));	\
  }									\
  /* y = +-Inf, x finite: +-pi/2.  */					\
  {									\
    vm_b_t vma2__myif = VM_VMAND_MM (vma2__myinf,			\
				     VM_VMNOT_M (vma2__mxinf, (vl)), (vl)); \
    vm_f_t vma2__p2 = VM_VFMERGE_VVM (VM_VFMV_V_F (VM_ATAN2_PIO2, (vl)), \
	VM_VFNEG_V (VM_VFMV_V_F (VM_ATAN2_PIO2, (vl)), (vl)),		\
	VM_VMOR_MM (vma2__m1, vma2__m3, (vl)), (vl));			\
    vma2__azh = VM_VFMERGE_VVM (vma2__azh, vma2__p2, vma2__myif, (vl)); \
    vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__myif, (vl));	\
  }									\
  /* NaN: x + y */							\
  {									\
    vm_f_t vma2__nn = VM_VFADD_VV_M (vma2__mnan, (x), (y), (vl));	\
    vma2__azh = VM_VFMERGE_VVM (vma2__azh, vma2__nn, vma2__mnan, (vl)); \
    vma2__azl = VM_VFMERGE_VFM (vma2__azl, 0.0, vma2__mnan, (vl));	\
  }									\
  (yh) = vma2__azh;							\
  (yl) = vma2__azl;							\
  (mo) = vma2__msc;							\
} while (0)

/* ------------------------------------------------------------------ */
/* asin / acos: fdlibm e_asin.c / e_acos.c.
   |x| < 0.5  : rational on x^2, asin = x + x*w.
   |x| >= 0.5 : s = sqrt((1-|x|)/2), the same rational on z=(1-|x|)/2.
   The final composition is carried as a dd pair.  */

#define VM_ASIN_PS0 1.66666666666666657415e-01
#define VM_ASIN_PS1 -3.25565818622400915405e-01
#define VM_ASIN_PS2 2.01212532134862925881e-01
#define VM_ASIN_PS3 -4.00555345006794114027e-02
#define VM_ASIN_PS4 7.91534994289814532176e-04
#define VM_ASIN_PS5 3.47933107596021167570e-05
#define VM_ASIN_QS1 -2.40339491173441421878e+00
#define VM_ASIN_QS2 2.02094576023350569471e+00
#define VM_ASIN_QS3 -6.88283971605453293030e-01
#define VM_ASIN_QS4 7.70381505559019352791e-02
#define VM_ASIN_PIO2_HI 1.57079632679489655800e+00
#define VM_ASIN_PIO2_LO 6.12323399573676603587e-17
#define VM_ASIN_PIO4_HI 7.85398163397448278999e-01

/* r = P(z)/Q(z), the shared rational of asin/acos.  */
static inline vm_f_t
VM_KFN (asin_rat) (vm_f_t z, size_t vl)
{
  vm_f_t p = VM_VFMACC_VF (VM_VFMV_V_F (VM_ASIN_PS4, vl), VM_ASIN_PS5, z, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_PS3, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_PS2, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_PS1, vl), z, p, vl);
  p = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_PS0, vl), z, p, vl);
  p = VM_VFMUL_VV (z, p, vl);
  vm_f_t q = VM_VFMACC_VF (VM_VFMV_V_F (VM_ASIN_QS3, vl), VM_ASIN_QS4, z, vl);
  q = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_QS2, vl), z, q, vl);
  q = VM_VFMACC_VV (VM_VFMV_V_F (VM_ASIN_QS1, vl), z, q, vl);
  q = VM_VFMACC_VV (VM_VFMV_V_F (1.0, vl), z, q, vl);
  return VM_VFDIV_VV (p, q, vl);
}

/* asin(x) as a dd pair; x finite, |x| < 1 (special lanes sanitized by
   the caller).  */
#define VM_ASIN_DD(ah, al, x, vl) do {					\
  vm_u64_t vms__ix = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 32, (vl)),	\
				 0x7fffffff, (vl));			\
  vm_b_t vms__mneg = VM_VMSLT_VX (VM_REINT_U_I (VM_BITS (x)), 0, (vl));	\
  vm_b_t vms__mhalf = VM_VMSLTU_VX (vms__ix, 0x3fe00000, (vl));		\
  vm_b_t vms__mge = VM_VMNOT_M (vms__mhalf, (vl));			\
  vm_f_t vms__a = VM_VFABS_V (x, (vl));					\
  /* |x| < 2^-27: fdlibm returns x itself.  x*x would underflow on the
     subnormal inputs (spurious UNDERFLOW), so those lanes are fed a
     benign z and their x*r term is masked to zero below.  */		\
  vm_b_t vms__mtiny = VM_VMSLTU_VX (vms__ix, 0x3e400000, (vl));		\
  vm_b_t vms__mmul = VM_VMNOT_M (vms__mtiny, (vl));			\
  vm_f_t vms__z = VM_VFMUL_VV_MU (vms__mmul, VM_VFMV_V_F (0.25, (vl)),	\
				  x, x, (vl));				\
  vm_f_t vms__z2 = VM_VFMUL_VF (VM_VFRSUB_VF (vms__a, 1.0, (vl)), 0.5, (vl)); \
  vms__z = VM_VFMERGE_VVM (vms__z, vms__z2, vms__mge, (vl));		\
  vm_f_t vms__r = VM_KFN (asin_rat) (vms__z, (vl));			\
  /* |x| < 2^-27: fdlibm returns x itself; zeroing r kills both the x*r
     term and its residual without an underflowing product.  */		\
  vms__r = VM_VFMERGE_VFM (vms__r, 0.0, vms__mtiny, (vl));		\
  /* |x| < 0.5: x + x*r, tracked (Fast2Sum valid: |x*r| << |x|).  */	\
  vm_f_t vms__t = VM_VFMUL_VV (x, vms__r, (vl));			\
  vm_f_t vms__te = VM_VFNEG_V (VM_VFNMSAC_VV (vms__t, x, vms__r, (vl)), (vl)); \
  vm_f_t vms__yh, vms__yl;						\
  VM_FAST2SUM (vms__yh, vms__yl, x, vms__t, (vl));			\
  vms__yl = VM_VFADD_VV (vms__yl, vms__te, (vl));			\
  /* |x| >= 0.5: fdlibm's pio4 split.  */				\
  {									\
    vm_f_t vms__s = VM_VFSQRT_V (vms__z, (vl));				\
    vm_f_t vms__w = VM_FROM_BITS (					\
      VM_VAND_VX (VM_BITS (vms__s), 0xffffffff00000000ULL, (vl)));	\
    /* c = (z - w*w)/(s + w) */						\
    vm_f_t vms__c = VM_VFDIV_VV (					\
      VM_VFMACC_VV (vms__z, vms__w, VM_VFNEG_V (vms__w, (vl)), (vl)),	\
      VM_VFADD_VV (vms__s, vms__w, (vl)), (vl));			\
    /* pp = 2*s*r - (pio2_lo - 2*c) */					\
    vm_f_t vms__pp = VM_VFRSUB_VF (					\
				  VM_VFMUL_VF (vms__c, 2.0, (vl)),	\
				  VM_ASIN_PIO2_LO, (vl));		\
    vms__pp = VM_VFMACC_VV (VM_VFNEG_V (vms__pp, (vl)),		\
			    VM_VFMUL_VF (vms__s, 2.0, (vl)), vms__r, (vl)); \
    /* qq = pio4_hi - 2*w */						\
    vm_f_t vms__qq = VM_VFRSUB_VF (					\
				  VM_VFMUL_VF (vms__w, 2.0, (vl)),	\
				  VM_ASIN_PIO4_HI, (vl));		\
    /* t = pio4_hi - (pp - qq), tracked.  */				\
    vm_f_t vms__B, vms__Be, vms__C, vms__Ce;				\
    VM_TWO_SUM (vms__B, vms__Be, vms__pp, VM_VFNEG_V (vms__qq, (vl)), (vl)); \
    VM_TWO_SUM (vms__C, vms__Ce, VM_VFMV_V_F (VM_ASIN_PIO4_HI, (vl)),	\
		VM_VFNEG_V (vms__B, (vl)), (vl));			\
    vm_f_t vms__bt = VM_VFMERGE_VVM (vms__C,				\
				     VM_VFNEG_V (vms__C, (vl)),		\
				     vms__mneg, (vl));			\
    vm_f_t vms__bl = VM_VFSUB_VV (vms__Ce, vms__Be, (vl));		\
    vms__bl = VM_VFMERGE_VVM (vms__bl,				\
			      VM_VFNEG_V (vms__bl, (vl)), vms__mneg, (vl)); \
    vms__yh = VM_VFMERGE_VVM (vms__bt, vms__yh, vms__mhalf, (vl));	\
    vms__yl = VM_VFMERGE_VVM (vms__bl, vms__yl, vms__mhalf, (vl));	\
  }									\
  (ah) = vms__yh;							\
  (al) = vms__yl;							\
} while (0)

/* acos(x) as a dd pair (|x| < 1).  */
#define VM_ACOS_DD(ah, al, x, vl) do {					\
  vm_u64_t vmc__ix = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 32, (vl)),	\
				 0x7fffffff, (vl));			\
  vm_b_t vmc__mneg = VM_VMSLT_VX (VM_REINT_U_I (VM_BITS (x)), 0, (vl));	\
  vm_b_t vmc__mhalf = VM_VMSLTU_VX (vmc__ix, 0x3fe00000, (vl));		\
  vm_b_t vmc__mgt = VM_VMNOT_M (vmc__mhalf, (vl));			\
  vm_f_t vmc__a = VM_VFABS_V (x, (vl));					\
  vm_b_t vmc__mtiny = VM_VMSLTU_VX (vmc__ix, 0x3e400000, (vl));		\
  vm_b_t vmc__mmul = VM_VMNOT_M (vmc__mtiny, (vl));			\
  vm_f_t vmc__z = VM_VFMUL_VV_MU (vmc__mmul, VM_VFMV_V_F (0.25, (vl)),	\
				  x, x, (vl));				\
  vm_f_t vmc__z2 = VM_VFMUL_VF (VM_VFRSUB_VF (vmc__a, 1.0, (vl)), 0.5, (vl)); \
  vmc__z = VM_VFMERGE_VVM (vmc__z, vmc__z2, vmc__mgt, (vl));		\
  vm_f_t vmc__r = VM_KFN (asin_rat) (vmc__z, (vl));			\
  /* |x| < 0.5: pio2_hi - (x - (pio2_lo - x*r)), tracked.  */		\
  vm_f_t vmc__sh, vmc__sl;						\
  {									\
    vm_f_t vmc__t = VM_VFMUL_VV_MU (vmc__mmul, VM_VFMV_V_F (0.0, (vl)),	\
				    x, vmc__r, (vl));			\
    vm_f_t vmc__u = VM_VFRSUB_VF (vmc__t, VM_ASIN_PIO2_LO, (vl));	\
    vm_f_t vmc__w, vmc__we;						\
    VM_TWO_SUM (vmc__w, vmc__we, x, VM_VFNEG_V (vmc__u, (vl)), (vl));	\
    vm_f_t vmc__s, vmc__se;						\
    VM_TWO_SUM (vmc__s, vmc__se, VM_VFMV_V_F (VM_ASIN_PIO2_HI, (vl)),	\
		VM_VFNEG_V (vmc__w, (vl)), (vl));			\
    vmc__sh = vmc__s;							\
    vmc__sl = VM_VFSUB_VV (vmc__se, vmc__we, (vl));			\
  }									\
  /* |x| >= 0.5 */							\
  {									\
    vm_f_t vmc__s = VM_VFSQRT_V (vmc__z, (vl));				\
    vm_f_t vmc__df = VM_FROM_BITS (					\
      VM_VAND_VX (VM_BITS (vmc__s), 0xffffffff00000000ULL, (vl)));	\
    vm_f_t vmc__c = VM_VFDIV_VV (					\
      VM_VFMACC_VV (vmc__z, vmc__df, VM_VFNEG_V (vmc__df, (vl)), (vl)),	\
      VM_VFADD_VV (vmc__s, vmc__df, (vl)), (vl));			\
    /* x > 0.5: 2*(df + (r*s + c)), tracked.  */			\
    vm_f_t vmc__w2 = VM_VFMACC_VV (vmc__c, vmc__r, vmc__s, (vl));	\
    vm_f_t vmc__g, vmc__ge;						\
    VM_TWO_SUM (vmc__g, vmc__ge, vmc__df, vmc__w2, (vl));		\
    vm_f_t vmc__ph = VM_VFMUL_VF (vmc__g, 2.0, (vl));			\
    vm_f_t vmc__pl = VM_VFMUL_VF (vmc__ge, 2.0, (vl));			\
    /* x < -0.5: pi - 2*(s + (r*s - pio2_lo)), tracked.  */		\
    vm_f_t vmc__w1 = VM_VFMACC_VV (VM_VFNEG_V (				\
				     VM_VFMV_V_F (VM_ASIN_PIO2_LO, (vl)), (vl)), \
				   vmc__r, vmc__s, (vl));		\
    vm_f_t vmc__g1, vmc__g1e;						\
    VM_TWO_SUM (vmc__g1, vmc__g1e, vmc__s, vmc__w1, (vl));		\
    vm_f_t vmc__h1 = VM_VFMUL_VF (vmc__g1, 2.0, (vl));			\
    vm_f_t vmc__h1e = VM_VFMUL_VF (vmc__g1e, 2.0, (vl));		\
    vm_f_t vmc__P, vmc__Pe;						\
    VM_TWO_SUM (vmc__P, vmc__Pe, VM_VFMV_V_F (VM_ATAN2_PI, (vl)),	\
		VM_VFNEG_V (vmc__h1, (vl)), (vl));			\
    vm_f_t vmc__nh = VM_VFSUB_VV (vmc__Pe, vmc__h1e, (vl));		\
    /* select by |x| and sign */					\
    vm_f_t vmc__bh = VM_VFMERGE_VVM (vmc__sh, vmc__ph, vmc__mgt, (vl));	\
    vm_f_t vmc__bl = VM_VFMERGE_VVM (vmc__sl, vmc__pl, vmc__mgt, (vl));	\
    vm_b_t vmc__mnegge = VM_VMAND_MM (vmc__mgt, vmc__mneg, (vl));	\
    (ah) = VM_VFMERGE_VVM (vmc__bh, vmc__P, vmc__mnegge, (vl));	\
    (al) = VM_VFMERGE_VVM (vmc__bl, vmc__nh, vmc__mnegge, (vl));	\
  }									\
} while (0)

static inline double
vm_atan2pi_scalar (double y, double x)
{
  return atan2pi (y, x);
}
#define atan2pi_scalar vm_atan2pi_scalar

static inline double
vm_atan2_scalar (double y, double x)
{
  return atan2 (y, x);
}
#define atan2_scalar vm_atan2_scalar

/* Scalar fixup for atan2pi lanes whose |y/x| quotient is subnormal: the
   2^600 evaluation recovers the exact quotient despite the coarse grid,
   then one rounding divides the scale out (the final multiply raises
   UNDERFLOW exactly when the reference expects it; an exactly-on-grid
   subnormal result gets it forced here).  */
static inline double
vm_atan2pi_msc (double y, double x)
{
  double q = y / x;
  double qs = q * 0x1p600;
  double ys = y * 0x1p600;
  double rem = fma (-qs, x, ys);
  double qlo = rem / x;
  double v = qs + qlo;
  double ve = (qs - v) + qlo;
  double p = v * 3.18309886183790691216e-01;
  double e = fma (v, 3.18309886183790691216e-01, -p)
	     + v * -1.96786766751824858819e-17
	     + ve * 3.18309886183790691216e-01;
  double r = (p + e) * 0x1p-600;
  if (r == 0.0)
    return __builtin_copysign (0.0, v);	/* keep the sign of y/x */
  if (fabs (r) < 0x1p-1022)
    {
      double z = 0x1p-1022 * 0x1p-1022;	/* flushes to +0, raises UF */
      r = r + z;
    }
  return r;
}
#define atan2pi_msc vm_atan2pi_msc

/* ------------------------------------------------------------------ */
/* public kernels */

static inline vm_f_t
VM_KFN (atan) (vm_f_t x, size_t vl)
{
  vm_f_t yh, yl, y;
  VM_ATAN_ONLY_DD (yh, yl, x, vl);
  VM_INVTRIG_COLLAPSE (y, yh, yl, vl);
  return y;
}

/* atan(x)/pi: one rounding on the finished pi-scaled value.  */
static inline vm_f_t
VM_KFN (atanpi) (vm_f_t x, size_t vl)
{
  vm_f_t yh, yl, y;
  VM_ATAN_ONLY_DD (yh, yl, x, vl);
  VM_MUL_INVPI_DD (y, yh, yl, vl);
  /* atan(+-inf)/pi = +-0.5 (the dd carries the rounded pi/2, whose scale
     is only 0.17 ulp from the exact limit), NaN passes through.  */
  {
    vm_u64_t ab = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, (vl));
    vm_b_t minf = VM_VMSEQ_VX (ab, 0x7ff0000000000000ULL, (vl));
    vm_b_t mnan = VM_VMSGTU_VX (ab, 0x7ff0000000000000ULL, (vl));
    vm_f_t half = VM_VFSGNJ_VV (VM_VFMV_V_F (0.5, (vl)), x, (vl));
    y = VM_VFMERGE_VVM (y, half, minf, (vl));
    y = VM_VFMERGE_VVM (y, yh, mnan, (vl));
  }
  /* atan(+-0)/pi = +-0 (the dd product folds -0).  */
  return VM_VFMERGE_VVM (y, VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, (vl)), x, (vl)), \
			 VM_VMFEQ_VF (x, 0.0, (vl)), (vl));
}

static inline vm_f_t
VM_KFN (atan2_core) (vm_f_t y, vm_f_t x, size_t vl)
{
  vm_f_t yh, yl, r;
  vm_b_t msc;
  VM_ATAN2_DD (yh, yl, msc, y, x, vl);
  VM_INVTRIG_COLLAPSE (r, yh, yl, vl);
  /* divide the 2^600 out of the scaled-quotient lanes: exact wherever
     the answer is normal, and the rounding there is the UNDERFLOW the
     reference expects of a subnormal answer.  */
  r = VM_VFMERGE_VVM (r, VM_VFMUL_VF_MU (msc, VM_VFMV_V_F (0.0, (vl)), r,
					 0x1p-600, (vl)), msc, (vl));
  return r;
}

static inline vm_f_t
VM_KFN (atan2) (vm_f_t y, vm_f_t x, size_t vl)
{
  vm_f_t yh, yl, r;
  vm_b_t msc;
  VM_ATAN2_DD (yh, yl, msc, y, x, vl);
  (void) msc;
  VM_INVTRIG_COLLAPSE (r, yh, yl, vl);
  return r;
}

/* atan2(y,x)/pi: the quadrant dd goes through one final pi scale.  */
static inline vm_f_t
VM_KFN (atan2pi) (vm_f_t y, vm_f_t x, size_t vl)
{
  vm_f_t yh, yl, yr;
  vm_b_t msc;
  vm_u64_t abx = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, (vl));
  vm_u64_t aby = VM_VAND_VX (VM_BITS (y), VM_ABSMASK_U64, (vl));
  vm_b_t mnan = VM_VMOR_MM (VM_VMSGTU_VX (abx, 0x7ff0000000000000ULL, (vl)), \
			     VM_VMSGTU_VX (aby, 0x7ff0000000000000ULL, (vl)), (vl));
  VM_ATAN2_DD (yh, yl, msc, y, x, vl);
  VM_MUL_INVPI_DD (yr, yh, yl, vl);
  yr = VM_VFMERGE_VVM (yr, yh, mnan, (vl));
  yr = VM_VFMERGE_VVM (yr, yh, mnan, (vl));
  yr = VM_VFMERGE_VVM (yr, VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, (vl)), y, (vl)), \
		       VM_VMFEQ_VF (yr, 0.0, (vl)), (vl));
  /* adversarial subnormal-quotient lanes: exact scalar reconstruction */
  return VM_KFN (scalar_fixup2) (y, x, yr, msc, vl, atan2pi_msc);
}

static inline vm_f_t
VM_KFN (asin) (vm_f_t x, size_t vl)
{
  vm_u64_t ab = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, vl);
  vm_b_t mspec = VM_VMSGEU_VX (ab, 0x3ff0000000000000ULL, vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 0.5, mspec, vl);
  vm_f_t yh, yl, y;
  VM_ASIN_DD (yh, yl, xs, vl);
  VM_INVTRIG_COLLAPSE (y, yh, yl, vl);
  /* asin(+-0) = +-0 (the collapse folds -0).  */
  y = VM_VFMERGE_VVM (y, x, VM_VMFEQ_VF (x, 0.0, vl), vl);
  return VM_KFN (scalar_fixup) (x, y, mspec, vl, asin);
}

static inline vm_f_t
VM_KFN (acos) (vm_f_t x, size_t vl)
{
  vm_u64_t ab = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, vl);
  vm_b_t mspec = VM_VMSGEU_VX (ab, 0x3ff0000000000000ULL, vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 0.0, mspec, vl);
  vm_f_t yh, yl, y;
  VM_ACOS_DD (yh, yl, xs, vl);
  VM_INVTRIG_COLLAPSE (y, yh, yl, vl);
  return VM_KFN (scalar_fixup) (x, y, mspec, vl, acos);
}

static inline vm_f_t
VM_KFN (asinpi) (vm_f_t x, size_t vl)
{
  vm_u64_t ab = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, vl);
  vm_b_t mspec = VM_VMSGEU_VX (ab, 0x3ff0000000000000ULL, vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 0.5, mspec, vl);
  vm_f_t yh, yl, y;
  VM_ASIN_DD (yh, yl, xs, vl);
  VM_MUL_INVPI_DD (y, yh, yl, vl);
  y = VM_VFMERGE_VVM (y, VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, (vl)), x, (vl)), \
		      VM_VMFEQ_VF (x, 0.0, (vl)), (vl));
  /* scalar fixup supplies asinpi() for |x| >= 1 and non-finite.  */
  return VM_KFN (scalar_fixup) (x, y, mspec, vl, asinpi);
}

static inline vm_f_t
VM_KFN (acospi) (vm_f_t x, size_t vl)
{
  vm_u64_t ab = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, vl);
  vm_b_t mspec = VM_VMSGEU_VX (ab, 0x3ff0000000000000ULL, vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 0.0, mspec, vl);
  vm_f_t yh, yl, y;
  VM_ACOS_DD (yh, yl, xs, vl);
  VM_MUL_INVPI_DD (y, yh, yl, vl);
  return VM_KFN (scalar_fixup) (x, y, mspec, vl, acospi);
}
