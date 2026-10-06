/* k_exp.inc -- exp / exp2 / exp10 kernels for libvecmath.

   LMUL-generic template: include after vecmath_rvvcompat.h (and
   vecmath_local.h / k_fixup.inc) with VM_LMUL defined.

   Vectorized ARM optimized-routines exp/exp2 (libm/common/exp.c and
   exp2.c): branchless and table-driven.  Shift-trick round-to-int;
   vluxei64 gathers into __exp_data.tab for the (tail, scale-bits)
   pairs; exponent injection sbits = T[2i+1] + (ki << (52 - TABLE_BITS));
   a masked scalar-fixup epilogue handles the rare |x| >= 512 lanes,
   reproducing scalar exp()/exp2() bit-for-bit, flags included.

   exp10 (SYCL/GNU) splits x*log2(10) into (thi, tlo) and feeds the exp2
   core.  The table is libvecmath's own __vecmath_exp_data (vm_data.h),
   emitted by test/gen/extract_table.py; never link libm's __exp_data,
   which orders the struct fields differently.  */

#include <vm_data.h>
/* VM_EXP_DD needs VM_TWO_SUM.  vecmath_dd.h carries no include guard on
   purpose -- it is re-includable at a different VM_LMUL.  */
#include <vecmath_dd.h>

/* log2(10) = L2HI + L2LO + O(2^-105), used by exp10.  */
#define VM_L2HI_10 3.32192809488736218171e+00
#define VM_L2LO_10 1.66161751697359200641e-16

/* Shared exp2 main path: computes 2^(x + xlo) for lanes with
   |x| < 512 (finite); no special-lane handling -- callers must mask or
   sanitize.  xlo is the dd tail correction (zero vector for exp2).  */
static inline vm_f_t
VM_KFN (exp2_core) (vm_f_t x, vm_f_t xlo, size_t vl)
{
  const struct vm_exp_data *D = &__vecmath_exp_data;
  /* kd = x + Shift; ki = bits(kd); kd -= Shift  (Shift-trick toint).  */
  vm_f_t kd = VM_VFADD_VF (x, D->exp2_shift, vl);
  vm_u64_t ki = VM_BITS (kd);
  kd = VM_VFSUB_VF (kd, D->exp2_shift, vl);
  /* r = (x - kd) + xlo.  */
  vm_f_t r = VM_VFADD_VV (VM_VFSUB_VV (x, kd, vl), xlo, vl);
  /* Table gathers: off = (ki mod 128) * 16 bytes.  */
  vm_u64_t off = VM_VSLL_VX (VM_VAND_VX (ki, 127, vl), 4, vl);
  const double *tabd = (const double *) &D->tab[0];
  vm_f_t tail = VM_VLUXEI64 (tabd, off, vl);
  vm_u64_t sbits = VM_VADD_VV (
    VM_VLUXEI64_U ((const uint64_t *) &D->tab[1], off, vl),
    VM_VSLL_VX (ki, 52 - 7, vl), vl);
  /* tmp = tail + r*C1 + r2*(C2 + r*C3) + r4*(C4 + r*C5)  (order 5).  */
  vm_f_t r2 = VM_VFMUL_VV (r, r, vl);
  vm_f_t p = VM_VFMACC_VF (VM_VFMV_V_F (D->exp2_poly[1], vl),
			   D->exp2_poly[2], r, vl);
  vm_f_t q = VM_VFMACC_VF (VM_VFMV_V_F (D->exp2_poly[3], vl),
			   D->exp2_poly[4], r, vl);
  vm_f_t r4 = VM_VFMUL_VV (r2, r2, vl);
  p = VM_VFMUL_VV (r2, p, vl);
  q = VM_VFMUL_VV (r4, q, vl);
  vm_f_t rc1 = VM_VFMUL_VF (r, D->exp2_poly[0], vl);
  vm_f_t tmp = VM_VFADD_VV (tail,
			    VM_VFADD_VV (rc1, VM_VFADD_VV (p, q, vl), vl),
			    vl);
  vm_f_t scale = VM_FROM_BITS (sbits);
  /* y = scale + scale*tmp (fused, mirrors scalar -ffp-contract=fast).  */
  return VM_VFMACC_VV (scale, scale, tmp, vl);
}

/* exp main path on an already-sanitized input: exp(xs) ~ scale*(1+tmp).
   Lanes that would break the reduction or the polynomial (|x| >= 512,
   inf, NaN, |x| < 2^-54) were replaced by a benign value and are
   restored by the callers.  Shared with VM_EXP_DD, whose main path must
   stay bit-identical.  */
#define VM_EXP_MAIN(scale, tmp, xs, vl) do {				\
  const struct vm_exp_data *vme_D = &__vecmath_exp_data;		\
  /* z = x * InvLn2N; kd = round via Shift trick.  */			\
  vm_f_t vme_z = VM_VFMUL_VF ((xs), vme_D->invln2N, (vl));		\
  vm_f_t vme_kd = VM_VFADD_VF (vme_z, vme_D->shift, (vl));		\
  vm_u64_t vme_ki = VM_BITS (vme_kd);					\
  vme_kd = VM_VFSUB_VF (vme_kd, vme_D->shift, (vl));			\
  /* r = x + kd*NegLn2hiN + kd*NegLn2loN (fused like scalar gcc).  */	\
  vm_f_t vme_r = VM_VFMACC_VF ((xs), vme_D->negln2hiN, vme_kd, (vl));	\
  vme_r = VM_VFMACC_VF (vme_r, vme_D->negln2loN, vme_kd, (vl));		\
  vm_u64_t vme_off = VM_VSLL_VX (VM_VAND_VX (vme_ki, 127, (vl)), 4, (vl)); \
  const double *vme_tabd = (const double *) &vme_D->tab[0];		\
  vm_f_t vme_tail = VM_VLUXEI64 (vme_tabd, vme_off, (vl));		\
  vm_u64_t vme_sbits = VM_VADD_VV (					\
    VM_VLUXEI64_U ((const uint64_t *) &vme_D->tab[1], vme_off, (vl)),	\
    VM_VSLL_VX (vme_ki, 52 - 7, (vl)), (vl));				\
  /* tmp = tail + r + r2*(C2 + r*C3) + r4*(C4 + r*C5)  (order 5).  */	\
  vm_f_t vme_r2 = VM_VFMUL_VV (vme_r, vme_r, (vl));			\
  vm_f_t vme_p = VM_VFMACC_VF (VM_VFMV_V_F (vme_D->poly[0], (vl)),	\
			       vme_D->poly[1], vme_r, (vl));		\
  vm_f_t vme_q = VM_VFMACC_VF (VM_VFMV_V_F (vme_D->poly[2], (vl)),	\
			       vme_D->poly[3], vme_r, (vl));		\
  vm_f_t vme_r4 = VM_VFMUL_VV (vme_r2, vme_r2, (vl));			\
  vme_p = VM_VFMUL_VV (vme_r2, vme_p, (vl));				\
  vme_q = VM_VFMUL_VV (vme_r4, vme_q, (vl));				\
  (tmp) = VM_VFADD_VV (VM_VFADD_VV (vme_tail, vme_r, (vl)),		\
		       VM_VFADD_VV (vme_p, vme_q, (vl)), (vl));		\
  (scale) = VM_FROM_BITS (vme_sbits);					\
} while (0)

/* exp(x): vectorization of libm/common/exp.c.  fixup receives the rare
   guard lanes (|x| >= 512, inf, NaN) and must be scalar exp().  */
static inline vm_f_t
VM_KFN (exp) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_u64_t abstop = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 52, vl), 0x7ff, vl);
  /* |x| < 2^-54 (incl +-0): scalar returns 1.0 + x.  */
  vm_b_t m_tiny = VM_VMSLTU_VX (abstop, 0x3c9, vl);
  /* |x| >= 512 (incl inf/nan): scalar specialcase path.  */
  vm_b_t m_guard = VM_VMSGEU_VX (abstop, 0x408, vl);
  /* Sanitize guard and tiny lanes: the main path must not see inf/NaN
     (spurious invalid) or |x| < 2^-54 (r*r raises spurious FE_UNDERFLOW;
     scalar exp returns 1+x early there).  Both are restored below.  */
  vm_f_t xs = VM_VFMERGE_VFM (x, 1.0, VM_VMOR_MM (m_guard, m_tiny, vl), vl);
  vm_f_t scale, tmp, y;
  VM_EXP_MAIN (scale, tmp, xs, vl);
  y = VM_VFMACC_VV (scale, scale, tmp, vl);
  /* Tiny lanes: exactly 1.0 + x, flag-identical to scalar (guard lanes
     are overwritten by fixup below).  */
  y = VM_VFMERGE_VVM (y, VM_VFADD_VF (x, 1.0, vl), m_tiny, vl);
  return VM_KFN (scalar_fixup) (x, y, m_guard, vl, fixup);
}

/* Double-double exp: (hi, lo) = exp(x) with the final scaling rounding
   removed, keeping the polynomial's ~2^-60 relative accuracy instead of
   0.5 ulp, which callers that square or multiply (sinh/cosh) amplify.
   The kernel evaluates scale + scale*tmp as one fma with scale a power
   of two, so q + r == scale*tmp exactly with q = RN(scale*tmp).  Lanes
   the main path does not cover get lo = 0.  */
#define VM_EXP_DD(hi, lo, x, vl, fixup) do {				\
  vm_u64_t vmed_abstop = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 52, (vl)), \
				     0x7ff, (vl));			\
  vm_b_t vmed_mtiny = VM_VMSLTU_VX (vmed_abstop, 0x3c9, (vl));		\
  vm_b_t vmed_mguard = VM_VMSGEU_VX (vmed_abstop, 0x408, (vl));		\
  vm_f_t vmed_xs = VM_VFMERGE_VFM ((x), 1.0,				\
				   VM_VMOR_MM (vmed_mguard, vmed_mtiny, (vl)), \
				   (vl));				\
  vm_f_t vmed_scale, vmed_tmp;						\
  VM_EXP_MAIN (vmed_scale, vmed_tmp, vmed_xs, (vl));			\
  vm_f_t vmed_q = VM_VFMUL_VV (vmed_scale, vmed_tmp, (vl));		\
  vm_f_t vmed_e1 = VM_VFNEG_V (						\
    VM_VFNMSAC_VV (vmed_q, vmed_scale, vmed_tmp, (vl)), (vl));		\
  vm_f_t vmed_s, vmed_e, vmed_t;					\
  VM_TWO_SUM (vmed_s, vmed_e, vmed_scale, vmed_q, (vl));		\
  vmed_t = VM_VFADD_VV (vmed_e, vmed_e1, (vl));				\
  VM_TWO_SUM ((hi), (lo), vmed_s, vmed_t, (vl));			\
  /* |x| < 2^-54: scalar exp returns 1+x, so carry (1,x) exactly.  */	\
  do {									\
    vm_f_t vmed_th, vmed_tl;						\
    VM_TWO_SUM (vmed_th, vmed_tl, VM_VFMV_V_F (1.0, (vl)), (x), (vl));	\
    (hi) = VM_VFMERGE_VVM ((hi), vmed_th, vmed_mtiny, (vl));		\
    (lo) = VM_VFMERGE_VVM ((lo), vmed_tl, vmed_mtiny, (vl));		\
  } while (0);								\
  (hi) = VM_KFN (scalar_fixup) ((x), (hi), vmed_mguard, (vl), fixup);	\
  (lo) = VM_VFMERGE_VFM ((lo), 0.0, vmed_mguard, (vl));			\
} while (0)

/* exp2(x): wrapper adding tiny/guard handling around exp2_core.
   fixup must be scalar exp2().  */
static inline vm_f_t
VM_KFN (exp2) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_u64_t abstop = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 52, vl), 0x7ff, vl);
  vm_b_t m_tiny = VM_VMSLTU_VX (abstop, 0x3c9, vl);
  vm_b_t m_guard = VM_VMSGEU_VX (abstop, 0x408, vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 1.0, VM_VMOR_MM (m_guard, m_tiny, vl), vl);
  vm_f_t z = VM_VFMV_V_F (0.0, vl);
  vm_f_t y = VM_KFN (exp2_core) (xs, z, vl);
  y = VM_VFMERGE_VVM (y, VM_VFADD_VF (x, 1.0, vl), m_tiny, vl);
  return VM_KFN (scalar_fixup) (x, y, m_guard, vl, fixup);
}

/* exp10(x) = 10^x: t = x*log2(10) as (thi, tlo); exp2_core(thi, tlo).
   Guard at |x| >= 256 (abstop >= 0x407): the exp2 core main path stays
   valid to |t| < ~1024, and every lane beyond the guard goes through
   scalar fixup (newlib exp10 == pow(10,x)).  */
static inline vm_f_t
VM_KFN (exp10) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_u64_t abstop = VM_VAND_VX (VM_VSRL_VX (VM_BITS (x), 52, vl), 0x7ff, vl);
  /* tiny threshold 2^-56, not 2^-54: 10^x = 1 - x*ln10 and the double
     grid below 1 has spacing 2^-53, so |x| between 2^-55.2 and 2^-54
     must round to 1-2^-53, not to 1.  */
  vm_b_t m_tiny = VM_VMSLTU_VX (abstop, 0x3c7, vl);
  vm_b_t m_guard = VM_VMSGEU_VX (abstop, 0x407, vl);
  vm_f_t xs = VM_VFMERGE_VFM (x, 1.0, VM_VMOR_MM (m_guard, m_tiny, vl), vl);
  /* thi = RN(x*L2HI); e = exact residual via vfnmsac; tlo = e + x*L2LO. */
  vm_f_t thi = VM_VFMUL_VF (xs, VM_L2HI_10, vl);
  vm_f_t e = VM_VFNEG_V (VM_VFNMSAC_VF (thi, VM_L2HI_10, xs, vl), vl);
  vm_f_t tlo = VM_VFMACC_VF (e, VM_L2LO_10, xs, vl);
  vm_f_t y = VM_KFN (exp2_core) (thi, tlo, vl);
  /* 10^x for |x| < 2^-54 is exactly 1.0 (2^-54*ln10 < 2^-53), matching
     scalar pow(10,x); avoids core-path underflow flags on tiny lanes.  */
  y = VM_VFMERGE_VFM (y, 1.0, m_tiny, vl);
  return VM_KFN (scalar_fixup) (x, y, m_guard, vl, fixup);
}
