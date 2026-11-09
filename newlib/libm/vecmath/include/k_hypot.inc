/* k_hypot.inc -- hypot kernel.

   LMUL-generic template: include after vecmath_rvvcompat.h, vecmath_dd.h
   and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.

   fdlibm's e_hypot.c reaches <= 1 ulp by Dekker-splitting x and y so the
   squares and their sum survive; a dd does the same job in one step and
   leaves only the final rounding:

       a = max(|x|,|y|), b = min(|x|,|y|)
       (a^2, e_a) = two_prod(a, a)      exact
       (b^2, e_b) = two_prod(b, b)      exact
       (s, e)     = dd_add(...)         ~2^-106     (fast two-sum: a>=b)
       sqrt(s, e)                       ~2^-106
       collapse                         correctly rounded

   That only works once both squares sit comfortably inside the format, so
   the pair is first scaled by 2^(1023 - exponent(a)), which puts a near 1
   whatever the input, subnormal a included.  The scale-back is an exact
   power of two unless the result is itself subnormal, where it rounds once
   onto the subnormal grid; that lane is finished with one fma so the
   final rounding is still a single round (the plain "add the two scaled
   halves" form was measured at exactly 1.0 ulp there).

   Only zero and the non-finite inputs go to the scalar hypot: routing the
   subnormal lanes there would make the flags depend on which scalar libm
   is linked, and glibc and newlib disagree on exactly those inputs.  */

/* (oh, ol) = sqrt(ah + al), ~2^-106; one division (by a power of two
   replaced with a multiply).  */
#define VM_HYP_DDSQRT(rh, rl, ah, al, vl) do {				\
  vm_f_t vhs_s = VM_VFSQRT_V ((ah), (vl));				\
  vm_f_t vhs_p, vhs_e;							\
  VM_TWO_PROD (vhs_p, vhs_e, vhs_s, vhs_s, (vl));			\
  VM_DD_SUB (vhs_p, vhs_e, (ah), (al), vhs_p, vhs_e, (vl));		\
  vhs_p = VM_VFDIV_VV (VM_VFMUL_VF (vhs_p, 0.5, (vl)), vhs_s, (vl));	\
  VM_FAST2SUM ((rh), (rl), vhs_s, vhs_p, (vl));				\
} while (0)

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
  vm_f_t ph, pl, qh, ql, sh, sl, rh, rl, r;
  VM_TWO_PROD (ph, pl, ap, ap, vl);
  VM_TWO_PROD (qh, ql, bp, bp, vl);
  /* dd add, using a >= b (so |ph| >= |qh|) to make the first sum a
     fast2sum; the pair is left unnormalized for the sqrt, which only
     needs ~106 bits, not the normalized invariant.  */
  sh = VM_VFADD_VV (ph, qh, vl);
  sl = VM_VFADD_VV (VM_VFADD_VV (VM_VFSUB_VV (ph, sh, vl), qh, vl),
		    VM_VFADD_VV (pl, ql, vl), vl);
  VM_HYP_DDSQRT (rh, rl, sh, sl, vl);
  /* Collapse in the scaled domain and scale once.  For a normal result
     that single multiply is exact, so no flag can leak out of the low
     part.  A SUBNORMAL result is the one case where that shortcut is
     wrong: rounding to 53 bits first and then onto the subnormal grid can
     come out exact even though the true value is not, and IEEE wants
     UNDERFLOW exactly then.  Those lanes finish with one fma on the two
     scaled halves, i.e. a single round onto the subnormal grid.  */
  vm_f_t rc = VM_VFADD_VV (rh, rl, vl);
  r = VM_VFMUL_VV (rc, R, vl);
  {
    vm_u64_t rb = VM_VAND_VX (VM_BITS (r), 0x7fffffffffffffffULL, vl);
    vm_b_t msub = VM_VMAND_MM (VM_VMSNE_VX (rb, 0, vl),
			       VM_VMSLTU_VX (rb, 0x0010000000000000ULL, vl), vl);
    if (VM_VFIRST_M (msub, vl) >= 0)
      {
	/* rh is in [1,2) here, so rh*R = M*2^-1075 with M the 53-bit
	   mantissa: M even -> exactly on the subnormal grid, M odd -> on a
	   midpoint.  rh*R is then one round away from the answer; a
	   midpoint is resolved by the sign of the dd low part (|rl*R| is
	   under a quarter grid step).  The rl*R multiply is what raises
	   UNDERFLOW for the inexact subnormal lanes.  */
	vm_f_t lo = VM_VFMUL_VV_M (msub, rl, R, vl);
	vm_f_t base = VM_VFMUL_VV_M (msub, rh, R, vl);
	vm_f_t step = VM_VFSGNJ_VV (
	  VM_VFMV_V_F (0x0.0000000000001p-1022, vl), rl, vl);	/* +-2^-1074 */
	/* rh*R sits on a subnormal midpoint iff M*2^(kappa-1) is a
	   half-integer, i.e. the 53-bit mantissa M of rh has its lowest set
	   bit at position -kappa (= 1023 - exponent(rh)).  All integer
	   except the bounded power-of-two conversion; kp is clamped for the
	   lanes the branch does not read.  */
	vm_u64_t expf = VM_VAND_VX (VM_VSRL_VX (VM_BITS (rh), 52, vl), 0x7ff, vl);
	expf = VM_VMERGE_VVM (VM_VMV_V_X (1023, vl), expf, msub, vl);
	vm_u64_t mv = VM_VOR_VX (
	  VM_VAND_VX (VM_BITS (rh), 0x000fffffffffffffULL, vl),
	  0x0010000000000000ULL, vl);
	vm_u64_t kp = VM_VSUB_VV (VM_VMV_V_X (1023, vl), expf, vl);	/* -kappa */
	vm_u64_t hi = VM_VSRL_VV (mv, kp, vl);
	vm_u64_t powi = VM_VFCVT_XU_F_V_RM (
	  VM_FROM_BITS (VM_VSLL_VX (VM_VADD_VX (kp, 1023, vl), 52, vl)),
	  __RISCV_FRM_RTZ, vl);
	vm_b_t tie = VM_VMAND_MM (
	  msub,
	  VM_VMSEQ_VX (VM_VSUB_VV (mv, VM_VMUL_VV (hi, powi, vl), vl), 0, vl),
	  vl);
	tie = VM_VMAND_MM (
	  tie, VM_VMSNE_VX (VM_VAND_VX (hi, 1, vl), 0, vl), vl);
	tie = VM_VMAND_MM (
	  tie, VM_VMFNE_VV (rl, VM_VFMV_V_F (0.0, vl), vl), vl);
	/* base is the round-half-to-even image of the midpoint, i.e. the
	   even one of {k, k+1}; the true value is k+1 for rl > 0 and k for
	   rl < 0, so step is only needed when base is not already it.
	   k's parity is bit 1 of hi.  */
	vm_b_t rlp = VM_VMFGE_VF (rl, 0.0, vl);
	vm_b_t kodd = VM_VMSNE_VX (
	  VM_VAND_VX (VM_VSRL_VX (hi, 1, vl), 1, vl), 0, vl);
	vm_b_t adj = VM_VMAND_MM (tie, VM_VMXOR_MM (rlp, kodd, vl), vl);
	r = VM_VFMERGE_VVM (r, base, msub, vl);
	r = VM_VFMERGE_VVM (
	  r, VM_VFADD_VV_M (adj, base, step, vl), adj, vl);
	/* lo is +-0 (|rl*R| is under a quarter grid step); adding it keeps
	   the value and is what raises UNDERFLOW on the inexact lanes.  */
	r = VM_VFMERGE_VVM (r, VM_VFADD_VV_M (msub, r, lo, vl), msub, vl);
      }
  }
  r = VM_VFMERGE_VVM (r, a, mfar, vl);
  return VM_KFN (scalar_fixup2) (x, y, r, mguard, vl, fixup);
}
