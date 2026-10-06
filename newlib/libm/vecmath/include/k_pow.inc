/* k_pow.inc -- pow kernel for libvecmath.

   LMUL-generic template: include after vecmath_rvvcompat.h,
   vecmath_local.h (k_fixup.inc) with VM_LMUL defined.

   Vectorization of the ARM optimized-routines pow that is the live
   newlib pow on rv64 hard-float (libm/common/pow.c):
     log_inline  -> table-gather log with ~15 extra tail bits
                    (POW_LOG_TABLE_BITS 7, poly order 8, logctail),
     ehi/elo     -> y*log(x) as a two-product split,
     exp_inline  -> the shared exp core with the tail folded into r,
                    sign_bias == 0.

   Every x < 0 lane goes to the scalar fixup, so the whole special-case
   matrix -- signed zero/inf/NaN rules, integer-y sign handling,
   |y| < 2^-65 or >= 2^63 -- is handled by the scalar pow with identical
   results and FP flags.  Lanes whose ehi needs the exp specialcase
   (|y*log(x)| >= 512: the overflow/underflow tails) are fixup lanes.
   Worst-case error 0.54 ulp per the optimized-routines analysis.  */

#include <vm_data.h>

/* pow's log reduction offset.  */
#define VM_POW_LOG_OFF	0x3fe6955500000000ULL

static inline vm_f_t
VM_KFN (pow_k) (vm_f_t x, vm_f_t y, size_t vl,
		double (*fixup) (double, double))
{
  const struct vm_pow_log_data *D = &__vecmath_pow_log_data;
  const struct vm_exp_data *E = &__vecmath_exp_data;
  vm_u64_t ix = VM_BITS (x);
  vm_u64_t topx = VM_VSRL_VX (ix, 52, vl);
  vm_u64_t topy = VM_VSRL_VX (VM_BITS (y), 52, vl);
  /* Guard from scalar pow: (topx - 1) >= 0x7fe  (x subnormal/zero/
     inf/NaN/negative)  ||  ((topy & 0x7ff) - 0x3be) >= 0x80
     (|y| < 2^-65 or |y| >= 2^63 or NaN).  */
  vm_b_t m_spx = VM_VMSGEU_VX (VM_VSUB_VX (topx, 1, vl), 0x7fe, vl);
  vm_b_t m_spy = VM_VMSGEU_VX (
    VM_VSUB_VX (VM_VAND_VX (topy, 0x7ff, vl), 0x3be, vl), 0x80, vl);
  vm_b_t m_spec = VM_VMOR_MM (m_spx, m_spy, vl);
  /* Sanitize both inputs for the main path.  */
  vm_f_t xs = VM_VFMERGE_VFM (x, 2.0, m_spec, vl);
  vm_f_t ys = VM_VFMERGE_VFM (y, 1.5, m_spec, vl);

  /* ---- log_inline(xs): x = 2^k z, table (invc, pad, logc, logctail) -- */
  vm_u64_t ixs = VM_BITS (xs);
  vm_u64_t tmp = VM_VSUB_VX (ixs, VM_POW_LOG_OFF, vl);
  vm_u64_t off = VM_VSLL_VX (VM_VAND_VX (VM_VSRL_VX (tmp, 45, vl), 127, vl),
			     5, vl);	/* stride 32 bytes  */
  vm_f_t invc = VM_VLUXEI64 (&D->tab[0].invc, off, vl);
  vm_f_t logc = VM_VLUXEI64 (&D->tab[0].logc, off, vl);
  vm_f_t logct = VM_VLUXEI64 (&D->tab[0].logctail, off, vl);
  vm_i64_t k = VM_VSRA_VX (VM_REINT_U_I (tmp), 52, vl);
  vm_u64_t iz = VM_VSUB_VV (ixs,
			    VM_VAND_VX (tmp, 0xfff0000000000000ULL, vl), vl);
  vm_f_t z = VM_FROM_BITS (iz);
  vm_f_t kd = VM_VFCVT_F_X_V (k, vl);
  /* r = fma(z, invc, -1)  (exact: |r| < 1/N).  */
  vm_f_t r = VM_VFMACC_VV (VM_VFMV_V_F (-1.0, vl), invc, z, vl);
  /* t1 = kd*Ln2hi + logc; t2 = t1 + r; lo1 = kd*Ln2lo + logctail;
     lo2 = t1 - t2 + r.  */
  vm_f_t t1 = VM_VFMACC_VF (logc, D->ln2hi, kd, vl);
  vm_f_t t2 = VM_VFADD_VV (t1, r, vl);
  vm_f_t lo1 = VM_VFMACC_VF (logct, D->ln2lo, kd, vl);
  vm_f_t lo2 = VM_VFADD_VV (VM_VFSUB_VV (t1, t2, vl), r, vl);
  /* ar = A0*r; ar2 = r*ar; ar3 = r*ar2; hi = t2 + ar2;
     lo3 = fma(ar, r, -ar2); lo4 = t2 - hi + ar2.  */
  vm_f_t ar = VM_VFMUL_VF (r, D->poly[0], vl);
  vm_f_t ar2 = VM_VFMUL_VV (r, ar, vl);
  vm_f_t ar3 = VM_VFMUL_VV (r, ar2, vl);
  vm_f_t hi = VM_VFADD_VV (t2, ar2, vl);
  vm_f_t lo3 = VM_VFMACC_VV (VM_VFNEG_V (ar2, vl), ar, r, vl);
  vm_f_t lo4 = VM_VFADD_VV (VM_VFSUB_VV (t2, hi, vl), ar2, vl);
  vm_f_t pa, lo;
  /* p = ar3 * (A1 + r*A2 + ar2*(A3 + r*A4 + ar2*(A5 + r*A6))).  */
  {
    vm_f_t q1 = VM_VFMACC_VF (VM_VFMV_V_F (D->poly[5], vl), D->poly[6],
			      r, vl);
    vm_f_t q2 = VM_VFMACC_VF (VM_VFMV_V_F (D->poly[3], vl), D->poly[4],
			      r, vl);
    q2 = VM_VFMACC_VV (q2, ar2, q1, vl);
    vm_f_t q0 = VM_VFMACC_VF (VM_VFMV_V_F (D->poly[1], vl), D->poly[2],
			      r, vl);
    q0 = VM_VFMACC_VV (q0, ar2, q2, vl);
    pa = VM_VFMUL_VV (ar3, q0, vl);
  }
  /* lo = lo1 + lo2 + lo3 + lo4 + p (scalar association order).  */
  lo = VM_VFADD_VV (VM_VFADD_VV (VM_VFADD_VV (VM_VFADD_VV (lo1, lo2, vl),
					       lo3, vl), lo4, vl), pa, vl);
  vm_f_t ylog = VM_VFADD_VV (hi, lo, vl);
  vm_f_t ltail = VM_VFADD_VV (VM_VFSUB_VV (hi, ylog, vl), lo, vl);

  /* ---- ehi/elo = ys * (ylog + ltail) ---- */
  vm_f_t ehi = VM_VFMUL_VV (ys, ylog, vl);
  vm_f_t eres = VM_VFNEG_V (VM_VFNMSAC_VV (ehi, ys, ylog, vl), vl);
  vm_f_t elo = VM_VFMACC_VV (eres, ys, ltail, vl);

  /* ---- exp_inline(ehi, elo, sign_bias = 0) ---- */
  vm_u64_t ab = VM_VAND_VX (VM_VSRL_VX (VM_BITS (ehi), 52, vl), 0x7ff, vl);
  vm_b_t m_tiny = VM_VMSLTU_VX (ab, 0x3c9, vl);
  vm_b_t m_eg = VM_VMSGEU_VX (ab, 0x408, vl);
  vm_f_t es = VM_VFMERGE_VFM (ehi, 1.0, VM_VMOR_MM (m_eg, m_tiny, vl), vl);
  vm_f_t zz = VM_VFMUL_VF (es, E->invln2N, vl);
  vm_f_t kd2 = VM_VFADD_VF (zz, E->shift, vl);
  vm_u64_t ki = VM_BITS (kd2);
  kd2 = VM_VFSUB_VF (kd2, E->shift, vl);
  vm_f_t rr = VM_VFMACC_VF (es, E->negln2hiN, kd2, vl);
  rr = VM_VFMACC_VF (rr, E->negln2loN, kd2, vl);
  rr = VM_VFADD_VV (rr, elo, vl);	/* r += xtail  */
  {
    vm_u64_t o2 = VM_VSLL_VX (VM_VAND_VX (ki, 127, vl), 4, vl);
    const double *tabd = (const double *) &E->tab[0];
    vm_f_t tail = VM_VLUXEI64 (tabd, o2, vl);
    vm_u64_t sbits = VM_VADD_VV (
      VM_VLUXEI64_U ((const uint64_t *) &E->tab[1], o2, vl),
      VM_VSLL_VX (ki, 52 - 7, vl), vl);
    vm_f_t r2 = VM_VFMUL_VV (rr, rr, vl);
    vm_f_t p1 = VM_VFMACC_VF (VM_VFMV_V_F (E->poly[0], vl), E->poly[1], rr, vl);
    vm_f_t p2 = VM_VFMACC_VF (VM_VFMV_V_F (E->poly[2], vl), E->poly[3], rr, vl);
    vm_f_t r4 = VM_VFMUL_VV (r2, r2, vl);
    p1 = VM_VFMUL_VV (r2, p1, vl);
    p2 = VM_VFMUL_VV (r4, p2, vl);
    vm_f_t tmp2 = VM_VFADD_VV (VM_VFADD_VV (tail, rr, vl),
			       VM_VFADD_VV (p1, p2, vl), vl);
    vm_f_t scale = VM_FROM_BITS (sbits);
    vm_f_t res = VM_VFMACC_VV (scale, scale, tmp2, vl);
    /* tiny ehi: scalar returns 1 + ehi (xtail dropped).  */
    res = VM_VFMERGE_VVM (res, VM_VFADD_VF (ehi, 1.0, vl), m_tiny, vl);
    /* guard lanes (|ehi| >= 512: pow's overflow/underflow tails) go to
       the scalar fixup together with the input-guard lanes.  */
    m_spec = VM_VMOR_MM (m_spec, m_eg, vl);
    return VM_KFN (scalar_fixup2) (x, y, res, m_spec, vl, fixup);
  }
}
