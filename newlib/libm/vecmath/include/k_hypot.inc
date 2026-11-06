/* k_hypot.inc -- hypot kernel.

   LMUL-generic template: include after vecmath_rvvcompat.h, vecmath_dd.h
   and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.

   fdlibm's e_hypot.c reaches <= 1 ulp by Dekker-splitting x and y so the
   squares and their sum survive; a dd does the same job in one step and
   leaves only the final rounding:

       a = max(|x|,|y|), b = min(|x|,|y|)
       (a^2, e_a) = two_prod(a, a)      exact
       (b^2, e_b) = two_prod(b, b)      exact
       (s, e)     = dd_add(...)         ~2^-106
       sqrt(s, e)                       ~2^-106
       collapse                         correctly rounded

   That only works once both squares sit comfortably inside the format, so
   the pair is first scaled by 2^(1023 - exponent(a)), which puts a near 1
   whatever the input, subnormal a included.  The scale-back is an exact
   power of two unless the result is itself subnormal, where it rounds once
   onto the subnormal grid and raises UNDERFLOW, as IEEE asks.

   Only zero and the non-finite inputs go to the scalar hypot: routing the
   subnormal lanes there would make the flags depend on which scalar libm
   is linked, and glibc and newlib disagree on exactly those inputs.  */

static inline vm_f_t
VM_KFN (hypot2) (vm_f_t x, vm_f_t y, size_t vl,
		 double (*fixup) (double, double))
{
  vm_f_t xa = VM_VFABS_V (x, vl);
  vm_f_t ya = VM_VFABS_V (y, vl);
  /* magnitudes compare as unsigned bit patterns */
  vm_b_t msw = VM_VMSLTU_VV (VM_BITS (xa), VM_BITS (ya), vl);
  vm_f_t a = VM_VFMERGE_VVM (xa, ya, msw, vl);
  vm_f_t b = VM_VFMERGE_VVM (ya, xa, msw, vl);
  vm_u64_t hm = VM_VSRL_VX (VM_BITS (a), 32, vl);
  vm_b_t mguard = VM_VMOR_MM (VM_VMSGEU_VX (hm, 0x7ff00000u, vl),
			      VM_VMSEQ_VX (VM_BITS (a), 0, vl), vl);
  vm_f_t as = VM_VFMERGE_VFM (a, 1.0, mguard, vl);
  vm_f_t bs = VM_VFMERGE_VFM (b, 0.0, mguard, vl);
  /* fdlibm's "far" shortcut, and not only as an optimisation: for a >> b
     the dd sqrt's Newton correction (a^2 - s^2)/2/s is ~b^2/2/a, which
     past a ratio of 2^60 leaves the format and raises UNDERFLOW on a lane
     whose answer is simply a.  */
  vm_b_t mfar = VM_VMSGTU_VX (
    VM_VSUB_VV (VM_VSRL_VX (VM_BITS (a), 32, vl),
		VM_VSRL_VX (VM_BITS (b), 32, vl), vl),
    (uint64_t) (60ULL << 20), vl);
  bs = VM_VFMERGE_VFM (bs, 0.0, mfar, vl);
  /* Scale a to ~1.  Both ends of the exponent range need their scale built
     by hand, because the bit arithmetic would produce 0.0 there.  */
  vm_u64_t ea = VM_VAND_VX (VM_VSRL_VX (VM_BITS (as), 52, vl), 0x7ff, vl);
  vm_f_t S = VM_VFMERGE_VFM (
    VM_FROM_BITS (VM_VSLL_VX (
      VM_VSUB_VV (VM_VMV_V_X (2046, vl), ea, vl), 52, vl)),
    0x1p-1023, VM_VMSEQ_VX (ea, 2046, vl), vl);
  vm_f_t R = VM_VFMERGE_VFM (VM_FROM_BITS (VM_VSLL_VX (ea, 52, vl)),
			     0x1p-1023, VM_VMSEQ_VX (ea, 0, vl), vl);
  vm_f_t ap = VM_VFMUL_VV (as, S, vl);
  vm_f_t bp = VM_VFMUL_VV (bs, S, vl);
  /* A b' this small cannot move the result -- its square is 2^-960 of a'
     -- and leaving it in would put a subnormal through the residual of
     two_prod, which is 2^-53 of the product again.  */
  vm_b_t mbt = VM_VMSGEU_VX (VM_VSRL_VX (VM_BITS (bp), 32, vl),
			     0x21f00000u, vl);
  bp = VM_VFMERGE_VFM (bp, 0.0, VM_VMNOT_M (mbt, vl), vl);
  vm_f_t ph, pl, qh, ql, ah, al, sh, sl, r;
  VM_TWO_PROD (ph, pl, ap, ap, vl);
  VM_TWO_PROD (qh, ql, bp, bp, vl);
  VM_DD_ADD (ah, al, ph, pl, qh, ql, vl);
  VM_DD_SQRT (sh, sl, ah, al, vl);
  /* Collapse in the scaled domain and scale once.  For a normal result
     that single multiply is exact, so no flag can leak out of the low
     part.  A SUBNORMAL result is the one case where that shortcut is
     wrong: rounding to 53 bits first and then onto the subnormal grid can
     come out exact even though the true value is not, and IEEE wants
     UNDERFLOW exactly then.  Those lanes round the pair back separately,
     so the tiny, inexact final rounding actually happens.  */
  vm_f_t rc;
  VM_DD_COLLAPSE (rc, sh, sl, vl);
  r = VM_VFMUL_VV (rc, R, vl);
  {
    vm_u64_t rb = VM_VAND_VX (VM_BITS (r), 0x7fffffffffffffffULL, vl);
    vm_b_t msub = VM_VMAND_MM (VM_VMSNE_VX (rb, 0, vl),
			       VM_VMSLTU_VX (rb, 0x0010000000000000ULL, vl), vl);
    vm_f_t t1 = VM_VFMUL_VV_M (msub, sh, R, vl);
    vm_f_t t2 = VM_VFMUL_VV_M (msub, sl, R, vl);
    t1 = VM_VFMERGE_VFM (t1, 0.0, VM_VMNOT_M (msub, vl), vl);
    t2 = VM_VFMERGE_VFM (t2, 0.0, VM_VMNOT_M (msub, vl), vl);
    r = VM_VFMERGE_VVM (r, VM_VFADD_VV (t1, t2, vl), msub, vl);
  }
  r = VM_VFMERGE_VVM (r, a, mfar, vl);
  return VM_KFN (scalar_fixup2) (x, y, r, mguard, vl, fixup);
}
