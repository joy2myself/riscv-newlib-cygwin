/* k_invtrig.inc -- atan / atan2 kernels and their pi-scaled variants.

   LMUL-generic template: include after vecmath_rvvcompat.h, vecmath_dd.h
   and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.  Also needs
   k_trig.inc for the pi constants (VM_INVPI_HI / VM_INVPI_LO).

   SHAPE (CORE-MATH atan/atan2, "table + tangent subtraction"): reduce
   q = min/max in [0,1] to the nearest node t0 = i/128 of a 128-node
   table, form z = (q - t0)/(1 + q*t0) -- two fma and one reciprocal --
   and correct with a degree-7 polynomial.  |z| <= 1/252 makes four
   coefficients enough; the table carries atan(t0) as a double-double
   pair, so the base is 106 bits and the whole kernel costs a single
   final rounding.  The four quadrants (atan2's pi/2 and pi offsets) are
   folded into the table, so they cost one index addend and no dd add;
   the pi-scaled entries use a pre-divided table and polynomial, so
   atanpi/atan2pi also pay exactly one rounding (the old path paid two,
   which is why those entries sat at exactly 1.0 ulp).

   Lanes whose operands leave [2^-500, 2^500) -- zero, subnormal, inf,
   NaN, or simply huge -- plus the tiny-ratio lanes whose quotient
   residual falls under the subnormal grid are answered by the previous
   fdlibm-shaped dd kernel, kept below and reached through a vfirst
   guard.  Inside the window every intermediate is normal, so the fast
   path can neither overflow nor underflow nor cancel into a subnormal,
   and it is 0.5-0.9 ulp; the retained kernel keeps the exact
   special-value patches and flags the reference expects on the rest.

   asin/acos keep their fdlibm shape (rational + sqrt); their final
   composition is carried as a double-double pair so the pi-scaled
   variants can divide by pi without paying an extra rounding.  */

/* ---- fdlibm s_atan.c / e_atan2.c constants, used only by the legacy
   fallback kernel below ---- */
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
#define VM_ATAN2_PIO4 7.8539816339744827900e-01
#define VM_ATAN2_PIO2 1.5707963267948965580e+00

/* ---- atan2 constants ---- */
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


/* ---- reduction tables (generated; see the header) ---- */
#include <vm_atan_table.h>

/* fast-path operand window: |x| and |y| in [2^-500, 2^500).  */
#define VM_AT_LO 0x20b0000000000000ULL		/* bits(2^-500) */
#define VM_AT_SPAN 0x3e80000000000000ULL	/* bits(2^500)-bits(2^-500) */
#define VM_AT_KBIAS 0x4330000000000000ULL	/* bits(2^52) */
#define VM_AT_K0 (0x1p52)
#define VM_AT_K129 (0x1p52 + 129.0)

/* ------------------------------------------------------------------ */
/* Table-driven atan core (CORE-MATH shape).

   VM_ATAN2_TAB(rh, rl, msp, y, x, vl, THI, TLO, C0, C1, C2, C3, C0R,
		ONE)
   returns atan2(y,x) -- or atan2(y,x)/pi with the _pi table and
   polynomial -- as a double-double pair, and sets msp to the lanes the
   caller must hand to the legacy kernel below.

   ONE marks the atan-only instantiation: the caller passes x = 1.0, so
   there is no x < 0 quadrant and the quarter index loses its 258 term.

   Table layout (see vm_atan_table.h): four quarters of 129 entries,
   [i + 129*(|x|<|y|) + 258*(x<0)] with the correction added for the
   even quarters and subtracted for the odd ones.  The quarter offset is
   folded into the 2^52 magic constant that rounds q*128, so the index
   and t0 = i/128 come out of two instructions.

   Everything the reduction touches is exact or a single rounding:
     num = b - t0*a, den = a + t0*b       (one fma each)
     z   = num/den and its fma residual   (dd quotient, 2^-56)
     corr= (z + zl)*p                     (one fma, p degree 3 in z^2)
     r   = hi + (lo + corr)               (one 2Sum, then one add)
   so the only rounding left is the caller's collapse of (rh, rl).  */
#define VM_ATAN2_TAB(rh, rl, msp, y, x, vl, THI, TLO, C0, C1, C2, C3, C0R, \
		       ONE) \
do {									\
  vm_u64_t vmt__abx = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, (vl));	\
  vm_u64_t vmt__aby = VM_VAND_VX (VM_BITS (y), VM_ABSMASK_U64, (vl));	\
  (msp) = VM_VMOR_MM (							\
    VM_VMSGEU_VX (VM_VSUB_VX (vmt__abx, VM_AT_LO, (vl)), VM_AT_SPAN, (vl)), \
    VM_VMSGEU_VX (VM_VSUB_VX (vmt__aby, VM_AT_LO, (vl)), VM_AT_SPAN, (vl)), \
    (vl));								\
  vm_f_t vmt__xs = VM_VFMERGE_VFM ((x), 1.0, (msp), (vl));		\
  vm_f_t vmt__ys = VM_VFMERGE_VFM ((y), 0.0, (msp), (vl));		\
  vm_f_t vmt__ax = VM_VFABS_V (vmt__xs, (vl));				\
  vm_f_t vmt__ay = VM_VFABS_V (vmt__ys, (vl));				\
  vm_b_t vmt__msw = VM_VMFLT_VV (vmt__ax, vmt__ay, (vl));		\
  vm_f_t vmt__a = VM_VFMERGE_VVM (vmt__ax, vmt__ay, vmt__msw, (vl));	\
  vm_f_t vmt__b = VM_VFMERGE_VVM (vmt__ay, vmt__ax, vmt__msw, (vl));	\
  vm_b_t vmt__mx = (msp);						\
  vm_b_t vmt__mneg = vmt__msw;						\
  if (!(ONE))								\
    {									\
      vmt__mx = VM_VMFLT_VF (vmt__xs, 0.0, (vl));			\
      vmt__mneg = VM_VMXOR_MM (vmt__msw, vmt__mx, (vl));		\
    }									\
  /* q = b/a to ~2^-14: only the table index needs it, and an index off	\
     by one still leaves |z| <= 1/252.  */				\
  vm_f_t vmt__ra = VM_VFREC7_V (vmt__a, (vl));				\
  vmt__ra = VM_VFMACC_VV (vmt__ra, vmt__ra,				\
    VM_VFNMSAC_VV (VM_VFMV_V_F (1.0, (vl)), vmt__a, vmt__ra, (vl)), (vl)); \
  vm_f_t vmt__q = VM_VFMUL_VV (vmt__b, vmt__ra, (vl));			\
  /* i = round(q*128) with the quarter offset carried by the magic	\
     constant; s - k is then exactly i, so t0 = i/128 is exact.  */	\
  vm_f_t vmt__k = VM_VFMERGE_VFM (VM_VFMV_V_F (VM_AT_K0, (vl)), \
				 VM_AT_K129, vmt__msw, (vl)); \
  if (!(ONE))								\
    vmt__k = VM_VFADD_VF_MU (vmt__mx, vmt__k, vmt__k, 258.0, (vl));	\
  vm_f_t vmt__s = VM_VFMACC_VF (vmt__k, 128.0, vmt__q, (vl));		\
  vm_f_t vmt__t0 = VM_VFMUL_VF (VM_VFSUB_VV (vmt__s, vmt__k, (vl)),	\
				0x1p-7, (vl));				\
  vm_u64_t vmt__off = VM_VSLL_VX (					\
    VM_VSUB_VX (VM_BITS (vmt__s), VM_AT_KBIAS, (vl)), 3, (vl));		\
  vm_f_t vmt__hi = VM_VLUXEI64 (THI, vmt__off, (vl));			\
  vm_f_t vmt__lo = VM_VLUXEI64 (TLO, vmt__off, (vl));			\
  /* z = (b - t0 a)/(a + t0 b) as a dd.  den's own rounding is half the \
     reduction's error budget and appears whole on the lanes where the \
     table base and the correction cancel, so it is recovered with an \
     fma residual and carried in the quotient dd.  */ \
  vm_f_t vmt__num = VM_VFNMSAC_VV (vmt__b, vmt__t0, vmt__a, (vl)); \
  vm_f_t vmt__den = VM_VFMACC_VV (vmt__a, vmt__t0, vmt__b, (vl)); \
  vm_f_t vmt__dlo = VM_VFMACC_VV (VM_VFSUB_VV (vmt__a, vmt__den, (vl)), \
				  vmt__t0, vmt__b, (vl)); \
  vm_f_t vmt__rd; \
  VM_RCP_FAST (vmt__rd, vmt__den, (vl)); \
  vm_f_t vmt__z = VM_VFMUL_VV (vmt__num, vmt__rd, (vl)); \
  /* |z| below 2^-340: z^2, z^3 and the quotient residual all fall past \
     the subnormal grid.  The reference (much wider exponent range) \
     expects no underflow there and the dropped terms are < 2^-680 \
     relative, so those lanes run with w = zl = 0.  */ \
  vm_b_t vmt__mtiny = VM_VMFLT_VF (vmt__q, 0x1p-340, (vl)); \
  vm_b_t vmt__mt = VM_VMNOT_M (vmt__mtiny, (vl)); \
  vm_f_t vmt__zl = VM_VFNMSAC_VV (vmt__num, vmt__z, vmt__den, (vl)); \
  vmt__zl = VM_VFNMSAC_VV (vmt__zl, vmt__z, vmt__dlo, (vl)); \
  vmt__zl = VM_VFMUL_VV_MU (vmt__mt, VM_VFMV_V_F (0.0, (vl)), vmt__zl, \
			    vmt__rd, (vl)); \
  /* corr = atan(z + zl) = z*C0 + (z*w)*Q + zl*(1 - w) + O(zl^2), with \
     Q = C1 + w*(C2 + w*C3), w = z^2.  The small terms ride on the fma \
     addends, so corr costs a single rounding -- evaluating the \
     polynomial as C0 + C1*w + ... would round the leading 1 and cost \
     up to a whole extra ulp where base and correction cancel.  The \
     zl*(1 - w) form matters: zl is only 2^-29 relative (VM_RCP_FAST) \
     and dropping its w*zl part is a 4-ulp error on the small-q lanes. \
     For the pi-scaled entries C0 is rounded, so its residual C0R \
     rides on the same addend and the 1/pi is carried to 106 bits.  */ \
  vm_f_t vmt__w = VM_VFMUL_VV_MU (vmt__mt, VM_VFMV_V_F (0.0, (vl)), \
				  vmt__z, vmt__z, (vl)); \
  vm_f_t vmt__Q = VM_VFMACC_VV (VM_VFMV_V_F (C2, (vl)), vmt__w, \
				VM_VFMV_V_F (C3, (vl)), (vl)); \
  vmt__Q = VM_VFMACC_VV (VM_VFMV_V_F (C1, (vl)), vmt__w, vmt__Q, (vl)); \
  vm_f_t vmt__t1 = VM_VFNMSAC_VV (vmt__zl, vmt__zl, vmt__w, (vl)); \
  if ((C0) != 1.0) \
    { \
      vmt__t1 = VM_VFMUL_VF (vmt__t1, C0, (vl)); \
      vmt__t1 = VM_VFMACC_VF_MU (vmt__mt, vmt__t1, C0R, vmt__z, (vl)); \
    } \
  vm_f_t vmt__corr = VM_VFMACC_VV ( \
    vmt__t1, VM_VFMUL_VV (vmt__z, vmt__w, (vl)), vmt__Q, (vl)); \
  vmt__corr = VM_VFMACC_VF (vmt__corr, C0, vmt__z, (vl)); \
  vmt__corr = VM_VFMERGE_VVM (vmt__corr, VM_VFNEG_V (vmt__corr, (vl)),	\
			      vmt__mneg, (vl));				\
  /* base + (lo + corr): one 2Sum, one rounding left for the caller.  */ \
  VM_TWO_SUM ((rh), (rl), vmt__hi,					\
	      VM_VFADD_VV (vmt__lo, vmt__corr, (vl)), (vl));		\
  /* the tiny-ratio lanes are answered by the legacy kernel: their	\
     quotient residual is below the subnormal grid, and that kernel	\
     already carries the exact-tiny patch the reference expects.  */	\
  (msp) = VM_VMOR_MM ((msp), vmt__mtiny, (vl));				\
} while (0)

/* atan(z) = z*(C0 + C1 z^2 + C2 z^4 + C3 z^6) on |z| <= 1/252: minimax
   (relative) fit, max relative error 2^-74.  The _pi set is the same
   polynomial pre-divided by pi, so the pi-scaled entries fold the 1/pi
   into the correction instead of scaling afterwards.  */
#define VM_AT_C0 0x1.0000000000000p+0
#define VM_AT_C1 -0x1.5555555555553p-2
#define VM_AT_C2 0x1.999999986aab1p-3
#define VM_AT_C3 -0x1.2490737a3b069p-3
#define VM_ATP_C0 0x1.45f306dc9c883p-2
#define VM_ATP_C0R -0x1.6b01ec5417056p-56
#define VM_ATP_C1 -0x1.b2995e7b7b601p-4
#define VM_ATP_C2 0x1.04c26be2ef92bp-4
#define VM_ATP_C3 -0x1.74811f8f4bb12p-5

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
vm_atan_scalar (double x)
{
  return atan (x);
}
#define atan_scalar vm_atan_scalar

static inline double
vm_atanpi_scalar (double x)
{
  return atanpi (x);
}
#define atanpi_scalar vm_atanpi_scalar

static inline double
vm_atan2_scalar (double y, double x)
{
  return atan2 (y, x);
}
#define atan2_scalar vm_atan2_scalar

static inline double
vm_atan2pi_scalar (double y, double x)
{
  return atan2pi (y, x);
}
#define atan2pi_scalar vm_atan2pi_scalar

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
/* Legacy dd kernel, kept as the out-of-line fallback for lanes outside
   the fast path's operand window.

   The table kernel is only flag-safe and 0.5-ulp accurate when both
   operands sit in [2^-500, 2^500): inside, every intermediate is a
   normal number and the ratio is bounded by 2^+-1000, so neither the
   reduction nor the quotient can underflow, overflow or cancel into a
   subnormal.  Outside that window live zero, subnormal, inf and NaN
   operands plus the extreme-ratio pairs, and the reference expects no
   spurious flag there -- exactly the cases the old fdlibm-shaped dd
   kernel (with its guard values and masked patches) was tuned for.  It
   is therefore retained verbatim and reached only when a lane needs it,
   through a vfirst guard so the hot path pays nothing.

   The old kernels are the ones this file used before the rewrite; their
   accuracy on those lanes is the accuracy the library had.  */

/* VM_ATAN_DD(yh, yl, x, xlo, vl): atan(x + xlo) as a dd pair, xlo the
   tail of x (zero vector for plain atan).  fdlibm s_atan.c; the final
   composition hi - ((r*S - lo) - r) is evaluated with error-free
   transforms so the pair carries ~106 bits.  */
/* fastq is a compile-time flag: the standalone atan/atanpi kernels pass 1
   and get the single-reciprocal quotient, where it is a clear win at
   LMUL 1-2 (and only there -- above LMUL 2 the extra live vector costs
   more than the division it saves, measured on both RVV boards).
   atan2/atan2pi inline the atan core with a non-zero tail and keep the
   two-division form, which schedules better in that larger expression.  */
#define VM_ATAN_DD(yh, yl, x, xlo, vl, fastq) do {				\
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
  /* One reciprocal serves both divisions at LMUL 1-2: r = rnum*rt and \
     the residual (rnum - r*rden)*rt.  The 2^-600 scale keeps the \
     reciprocal in range even for rden = |x| on the -1/ag branch (up to \
     the largest finite double), and the Newton residual absorbs the \
     extra rounding, so the dd pair is as accurate as a true dd divide. \
     Above LMUL 2 the extra live vector costs more than the division it \
     saves, so those instantiations keep two true divisions.  */ \
  vm_f_t vma__r, vma__rr; \
  if ((fastq) && VM_LMUL <= 2) \
    { \
      vm_f_t vq__rt; \
      VM_RCP_FULL (vq__rt, VM_VFMUL_VF (vma__rden, 0x1p-600, (vl)), (vl)); \
      vq__rt = VM_VFMUL_VF (vq__rt, 0x1p-600, (vl)); \
      vma__r = VM_VFMUL_VV (vma__rnum, vq__rt, (vl)); \
      vma__rr = VM_VFMUL_VV ( \
        VM_VFNMSAC_VV (vma__rnum, vma__r, vma__rden, (vl)), \
        vq__rt, (vl)); \
    } \
  else \
    { \
      vma__r = VM_VFDIV_VV (vma__rnum, vma__rden, (vl)); \
      vma__rr = VM_VFDIV_VV ( \
        VM_VFNMSAC_VV (vma__rnum, vma__r, vma__rden, (vl)), \
        vma__rden, (vl)); \
    } \
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
  VM_ATAN_DD ((yh), (yl), (x), VM_VFMV_V_F (0.0, (vl)), (vl), 1)

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
  VM_ATAN_DD (vma2__azh, vma2__azl, vma2__aq, vma2__atail, (vl), 0);	\
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



/* Out-of-line fallbacks: recompute the masked lanes with the legacy
   kernel and merge.  Passing vector groups by value spills the caller's
   live registers, so the call site tests vfirst first.  */
__attribute__ ((noinline)) static vm_f_t
VM_KFN (atan_ref) (vm_f_t x, vm_f_t r, vm_b_t m, size_t vl)
{
  vm_f_t yh, yl, y;
  VM_ATAN_ONLY_DD (yh, yl, x, vl);
  VM_INVTRIG_COLLAPSE (y, yh, yl, vl);
  return VM_VFMERGE_VVM (r, y, m, vl);
}

__attribute__ ((noinline)) static vm_f_t
VM_KFN (atanpi_ref) (vm_f_t x, vm_f_t r, vm_b_t m, size_t vl)
{
  vm_f_t yh, yl, y;
  VM_ATAN_ONLY_DD (yh, yl, x, vl);
  VM_MUL_INVPI_DD (y, yh, yl, vl);
  {
    vm_u64_t ab = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, (vl));
    vm_b_t minf = VM_VMSEQ_VX (ab, 0x7ff0000000000000ULL, (vl));
    vm_b_t mnan = VM_VMSGTU_VX (ab, 0x7ff0000000000000ULL, (vl));
    vm_f_t half = VM_VFSGNJ_VV (VM_VFMV_V_F (0.5, (vl)), x, (vl));
    y = VM_VFMERGE_VVM (y, half, minf, (vl));
    y = VM_VFMERGE_VVM (y, yh, mnan, (vl));
  }
  y = VM_VFMERGE_VVM (y, VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, (vl)), x, (vl)),
		      VM_VMFEQ_VF (x, 0.0, (vl)), (vl));
  return VM_VFMERGE_VVM (r, y, m, vl);
}

__attribute__ ((noinline)) static vm_f_t
VM_KFN (atan2_ref) (vm_f_t y, vm_f_t x, vm_f_t r, vm_b_t m, size_t vl)
{
  vm_f_t yh, yl, y2;
  vm_b_t msc;
  VM_ATAN2_DD (yh, yl, msc, y, x, vl);
  (void) msc;
  VM_INVTRIG_COLLAPSE (y2, yh, yl, vl);
  return VM_VFMERGE_VVM (r, y2, m, vl);
}

__attribute__ ((noinline)) static vm_f_t
VM_KFN (atan2pi_ref) (vm_f_t y, vm_f_t x, vm_f_t r, vm_b_t m, size_t vl)
{
  vm_f_t yh, yl, yr;
  vm_b_t msc;
  vm_u64_t abx = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, (vl));
  vm_u64_t aby = VM_VAND_VX (VM_BITS (y), VM_ABSMASK_U64, (vl));
  vm_b_t mnan = VM_VMOR_MM (VM_VMSGTU_VX (abx, 0x7ff0000000000000ULL, (vl)),
			     VM_VMSGTU_VX (aby, 0x7ff0000000000000ULL, (vl)),
			     (vl));
  VM_ATAN2_DD (yh, yl, msc, y, x, vl);
  VM_MUL_INVPI_DD (yr, yh, yl, vl);
  yr = VM_VFMERGE_VVM (yr, yh, mnan, (vl));
  yr = VM_VFMERGE_VVM (yr, VM_VFSGNJ_VV (VM_VFMV_V_F (0.0, (vl)), y, (vl)),
		       VM_VMFEQ_VF (yr, 0.0, (vl)), (vl));
  if (VM_VFIRST_M (msc, vl) < 0)
    return VM_VFMERGE_VVM (r, yr, m, vl);
  yr = VM_FIXUP2 (y, x, yr, msc, vl, atan2pi_msc);
  return VM_VFMERGE_VVM (r, yr, m, vl);
}


/* ------------------------------------------------------------------ */
/* public kernels */

/* atan(x): the atan2 core with x = 1 (the reduction then gives
   q = min(|x|,1)/max(|x|,1) and the swapped quarter supplies the
   pi/2 - atan(1/|x|) branch), odd sign applied on the collapsed pair.  */
static inline vm_f_t
VM_KFN (atan) (vm_f_t x, size_t vl)
{
  vm_f_t yh, yl, r;
  vm_b_t msp;
  VM_ATAN2_TAB (yh, yl, msp, x, VM_VFMV_V_F (1.0, (vl)), vl,
		vm_atan_tab_hi, vm_atan_tab_lo,
		VM_AT_C0, VM_AT_C1, VM_AT_C2, VM_AT_C3, 0.0, 1);
  r = VM_VFSGNJ_VV (VM_VFADD_VV (yh, yl, vl), x, vl);
  if (VM_VFIRST_M (msp, vl) < 0)
    return r;
  return VM_KFN (atan_ref) (x, r, msp, vl);
}

/* atan(x)/pi: same kernel against the pre-divided table, so the pi
   scaling costs no extra rounding at all.  */
static inline vm_f_t
VM_KFN (atanpi) (vm_f_t x, size_t vl)
{
  vm_f_t yh, yl, r;
  vm_b_t msp;
  VM_ATAN2_TAB (yh, yl, msp, x, VM_VFMV_V_F (1.0, (vl)), vl,
		vm_atanpi_tab_hi, vm_atanpi_tab_lo,
		VM_ATP_C0, VM_ATP_C1, VM_ATP_C2, VM_ATP_C3, VM_ATP_C0R, 1);
  r = VM_VFSGNJ_VV (VM_VFADD_VV (yh, yl, vl), x, vl);
  if (VM_VFIRST_M (msp, vl) < 0)
    return r;
  return VM_KFN (atanpi_ref) (x, r, msp, vl);
}

static inline vm_f_t
VM_KFN (atan2) (vm_f_t y, vm_f_t x, size_t vl)
{
  vm_f_t yh, yl, r;
  vm_b_t msp;
  VM_ATAN2_TAB (yh, yl, msp, y, x, vl, vm_atan_tab_hi, vm_atan_tab_lo,
		VM_AT_C0, VM_AT_C1, VM_AT_C2, VM_AT_C3, 0.0, 0);
  r = VM_VFSGNJ_VV (VM_VFADD_VV (yh, yl, vl), y, vl);
  if (VM_VFIRST_M (msp, vl) < 0)
    return r;
  return VM_KFN (atan2_ref) (y, x, r, msp, vl);
}

/* atan2(y,x)/pi: the quadrant constants are pre-divided by pi in the
   table (0, 1/2, 1 instead of 0, pi/2, pi), so this is the same kernel
   against the _pi table.  */
static inline vm_f_t
VM_KFN (atan2pi) (vm_f_t y, vm_f_t x, size_t vl)
{
  vm_f_t yh, yl, r;
  vm_b_t msp;
  VM_ATAN2_TAB (yh, yl, msp, y, x, vl, vm_atanpi_tab_hi, vm_atanpi_tab_lo,
		VM_ATP_C0, VM_ATP_C1, VM_ATP_C2, VM_ATP_C3, VM_ATP_C0R, 0);
  r = VM_VFSGNJ_VV (VM_VFADD_VV (yh, yl, vl), y, vl);
  if (VM_VFIRST_M (msp, vl) < 0)
    return r;
  return VM_KFN (atan2pi_ref) (y, x, r, msp, vl);
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
  return VM_FIXUP1 (x, y, mspec, vl, asin);
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
  return VM_FIXUP1 (x, y, mspec, vl, acos);
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
  return VM_FIXUP1 (x, y, mspec, vl, asinpi);
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
  return VM_FIXUP1 (x, y, mspec, vl, acospi);
}
