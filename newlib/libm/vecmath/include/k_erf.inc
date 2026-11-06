/* k_erf.inc -- erf / erfc as double-double: two regions, both in dd so
   that the collapse's half ulp is the only rounding that matters.

     |x| < 1    erf(x) = x * P(x^2), P of degree 15 on [0,1]   (5.6e-25)
     |x| >= 1   erfc(x) = exp(-x^2) * H(x) / x, H = RP/RQ on [1,40]
                (8.6e-23), and erf(x) = 1 - erfc(x)

   Tables from test/gen/fit_erf.py: P is the Chebyshev truncation of
   erf(sqrt(t))/sqrt(t), RP/RQ the Sanathanan-Koerner fixed point of the
   relative-error weighting w = 1/Q, measured off the fit grid.

   RP/RQ must be evaluated in the Chebyshev basis (VM_ERF_CHEB): near
   x = 1 both pass through ~1e-13 while their ratio is 0.428, so the
   power basis loses the fit.  H's argument t = (2x-41)/39 is carried as
   a dd because 2x - 41 cancels at x = 20.5.

   Past x = 40 erfc is below 2^-1075, so exp(-x^2) underflows to 0 and the
   answers are exactly 0 and +-1 with no special case; only the rational's
   argument is clamped, so it cannot be extrapolated to inf and turn the
   zero into a NaN.  erfc also crosses the whole subnormal range, so the
   tail is evaluated scaled by 2^256 (VM_ERFC_TAIL), and every range test
   is an integer compare on the magnitude bits, because vmfgt/vmflt/vmfge
   are signalling and raise INVALID on a quiet NaN.

   Nothing needs a helper: erf/erfc are newlib's own scalar fallbacks when
   the vector unit is absent.  */

#ifndef VM_ERF_DATA
#define VM_ERF_DATA

static const double vm_erf_p[16][2] = {
  { 1.12837916709551256e+00, 1.53354591297935063e-17 },
  { -3.76126389031837538e-01, 1.33921451401565852e-17 },
  { 1.12837916709551261e-01, -4.03868681168630800e-18 },
  { -2.68661706451312522e-02, 1.17214881305522062e-18 },
  { 5.22397762544217492e-03, 3.38769507086966561e-19 },
  { -8.54832702344950675e-04, 2.67211259026152065e-20 },
  { 1.20553329816950656e-04, -9.95900682906516864e-23 },
  { -1.49256503538115455e-05, -1.65133152285990302e-22 },
  { 1.64621142064921003e-06, 1.05822534817324708e-22 },
  { -1.63658406628194321e-07, -1.05142264064748862e-23 },
  { 1.48071179408434930e-08, -7.80927831070252967e-25 },
  { -1.22895318115311343e-09, -2.41980655533151607e-26 },
  { 9.41257660857208957e-11, -3.40765663238783948e-27 },
  { -6.63938464665002211e-12, 1.12849921617339364e-28 },
  { 4.11757524819339381e-13, 5.77386054668165552e-30 },
  { -1.74696350739425451e-14, -1.13393237893297267e-32 },
};

static const double vm_erfc_rp[13][2] = {
  { 5.63996056489841968e-01, 5.25398304659511822e-17 },
  { 1.03470306545963009e+00, 1.07695849500308731e-16 },
  { 7.97832477371933413e-01, 2.00090894820633220e-17 },
  { 5.15535692197734097e-01, -1.53783664666705836e-17 },
  { 2.77653878888470318e-01, -2.34385511522129753e-17 },
  { 1.23610693635713947e-01, -1.25224888682149049e-18 },
  { 4.49479915706404304e-02, 5.49772298179890601e-19 },
  { 1.31221036275803674e-02, -8.20210652388352269e-19 },
  { 2.99965990415393879e-03, -1.95445023412875275e-19 },
  { 5.16956151720724365e-04, 4.63295362634888077e-20 },
  { 6.31385928433697910e-05, 6.59356590673936525e-21 },
  { 4.87009118409009250e-06, 3.04711420459967005e-22 },
  { 1.78345829827900262e-07, 2.58941053016642617e-24 },
};

static const double vm_erfc_rq[13][2] = {
  { 1.00000000000000000e+00, 0.00000000000000000e+00 },
  { 1.83458172231892913e+00, -3.94227646928806813e-17 },
  { 1.41457310928777247e+00, -8.76307089313642390e-17 },
  { 9.14029624748139646e-01, 5.30761905215468639e-17 },
  { 4.92254205361056907e-01, 1.69711438792688604e-17 },
  { 2.19140811507456124e-01, 4.09910697020747429e-18 },
  { 7.96815295324800749e-02, -6.35616446448699710e-18 },
  { 2.32611374269699467e-02, 1.61900033544045064e-18 },
  { 5.31717515000608518e-03, -2.65757522481010568e-19 },
  { 9.16319332017046743e-04, 3.33573401033305007e-20 },
  { 1.11911904667684453e-04, 4.97749944986971605e-21 },
  { 8.63201187350148997e-06, -6.06887072628454405e-22 },
  { 3.16109752871401652e-07, 2.30632892969860648e-23 },
};

#endif /* VM_ERF_DATA */

/* dd Horner.  Tables run low-to-high (entry 0 is the constant term), so
   the fit's own error is all that is left.  */
#define VM_ERF_POLY(rh, rl, tbl, n, th, tl, vl) do {			\
  vm_f_t veph = VM_VFMV_V_F ((tbl)[(n) - 1][0], (vl));			\
  vm_f_t vep = VM_VFMV_V_F ((tbl)[(n) - 1][1], (vl));			\
  for (int vepi = (n) - 2; vepi >= 0; vepi--)				\
    {									\
      vm_f_t veph2, vep2;						\
      VM_DD_MUL (veph2, vep2, veph, vep, (th), (tl), (vl));		\
      VM_DD_ADD_D (veph, vep, veph2, vep2, (tbl)[vepi][0], (vl));	\
      vep = VM_VFADD_VF (vep, (tbl)[vepi][1], (vl));			\
    }									\
  (rh) = veph;								\
  (rl) = vep;								\
} while (0)

/* Clenshaw over a Chebyshev-basis table, in dd.  The rational below MUST
   be evaluated this way and not in the power basis: near x = 1 P(t) and
   Q(t) are ~1e-11 while their power-basis terms are O(1) and alternating,
   and the power basis loses the fit there.  */
#define VM_ERF_CHEB(rh, rl, tbl, n, th, tl, vl) do {			\
  vm_f_t vcb1h = VM_VFMV_V_F (0.0, (vl));				\
  vm_f_t vcb1l = VM_VFMV_V_F (0.0, (vl));				\
  vm_f_t vcb2h = VM_VFMV_V_F (0.0, (vl));				\
  vm_f_t vcb2l = VM_VFMV_V_F (0.0, (vl));				\
  vm_f_t vcth, vctl;							\
  VM_DD_ADD (vcth, vctl, (th), (tl), (th), (tl), (vl));			\
  for (int vci = (n) - 1; vci >= 1; vci--)				\
    {									\
      vm_f_t vcmh, vcml, vcth2, vctl2, vcnh, vcnl;			\
      VM_DD_MUL (vcmh, vcml, vcth, vctl, vcb1h, vcb1l, (vl));		\
      VM_DD_SUB (vcth2, vctl2, vcmh, vcml, vcb2h, vcb2l, (vl));		\
      VM_DD_ADD_D (vcnh, vcnl, vcth2, vctl2, (tbl)[vci][0], (vl));	\
      vcnl = VM_VFADD_VF (vcnl, (tbl)[vci][1], (vl));			\
      vcb2h = vcb1h;							\
      vcb2l = vcb1l;							\
      vcb1h = vcnh;							\
      vcb1l = vcnl;							\
    }									\
  {									\
    /* the recurrence runs on 2t; the closing step takes plain t.  */	\
    vm_f_t vcmh, vcml, vcoh, vcol;					\
    vm_f_t vcjh = VM_VFMUL_VF (vcth, 0.5, (vl));			\
    vm_f_t vcjl = VM_VFMUL_VF (vctl, 0.5, (vl));			\
    VM_DD_MUL (vcmh, vcml, vcjh, vcjl, vcb1h, vcb1l, (vl));		\
    VM_DD_SUB (vcoh, vcol, vcmh, vcml, vcb2h, vcb2l, (vl));		\
    VM_DD_ADD_D ((rh), (rl), vcoh, vcol, (tbl)[0][0], (vl));		\
    (rl) = VM_VFADD_VF ((rl), (tbl)[0][1], (vl));			\
  }									\
} while (0)

/* ---- magnitudes as raw bits ------------------------------------------
   Every range test below is an INTEGER compare on the magnitude's bit
   pattern, never vmfgt/vmflt/vmfge: those are signalling and raise
   INVALID on a quiet NaN (scalar RISC-V does the same).  For a
   non-negative double the bit pattern orders like the value, and any
   NaN's pattern is larger than every finite one -- so each mask below is
   written so that a NaN lands where its propagation is wanted.  */
#define VM_ERF_B1    0x3ff0000000000000ULL	/* 1.0 */
#define VM_ERF_B25   0x4039000000000000ULL	/* 25.0 */
#define VM_ERF_B40   0x4044000000000000ULL	/* 40.0 */
#define VM_ERF_BTINY 0x3df0000000000000ULL	/* 2^-32 */
/* The largest double whose ROUNDED x*2/sqrt(pi) is still subnormal -- that
   is, the largest x whose erf answer is subnormal.  Above it the answer is
   normal and must not be allowed to underflow; at or below it, UNDERFLOW is
   exactly what the reference wants.  */
#define VM_ERF_BSUB  0x000e2dfc48da77b4ULL
#define VM_ERF_BINF  0x7ff0000000000000ULL	/* +Inf */

/* The tail is evaluated scaled by 2^256; see VM_ERFC_TAIL.  */
#define VM_ERF_S     0x1p256
#define VM_ERF_SINV  0x1p-256
#define VM_ERF_S2    0x1p257
/* 256*ln2, split: one rounded double would inject a 1.4e-14 relative error
   straight into the exponent.  */
#define VM_ERF_SLN2H 1.77445678223345993e+02
#define VM_ERF_SLN2L 5.93675984344652687e-15

/* erfc(ax) for the lanes that need it, as a dd SCALED by 2^256.

   A dd pair carries ~53 bits below its hi part, so below 2^-969 that lo
   part is itself subnormal, and merely forming it raises UNDERFLOW on
   lanes whose answer is an ordinary normal double.  Everything here is
   therefore evaluated 2^256 up (hi >= 2^-766, lo >= 2^-819) and the caller
   scales back with one multiply, which rounds and raises UNDERFLOW exactly
   when the answer really is subnormal or zero.  The exp's ARGUMENT is
   shifted by 256*ln2 rather than its result scaled, because exp()'s own
   internal 2^(k/128) would otherwise go subnormal for x > 26.7.

   The argument is clamped into [1, 40] -- this macro runs on every lane,
   including the |x| < 1 ones whose answer is thrown away, where a divisor
   of 0 would be 0/0 (INVALID) plus DIVBYZERO.  mceil is the caller's own
   ceiling: 25 wherever erfc is discarded (erf) or the answer is exactly 2
   (x < 0), and it must be built NaN-free, or a NaN would be clamped away
   before it could propagate.  +-Inf is turned into 1 for the arithmetic:
   erfc(+-Inf) is exactly 0 or 2, and exp(-40^2) on the way there would
   raise UNDERFLOW on a lane whose answer is exact.  */
#define VM_ERFC_TAIL(eh, el, ax, mceil, vl) do {			\
  vm_u64_t vmef_bx = VM_BITS (ax);					\
  vm_b_t vmef_minf = VM_VMSEQ_VX (vmef_bx, VM_ERF_BINF, (vl));		\
  vm_b_t vmef_mnan = VM_VMSGTU_VX (vmef_bx, VM_ERF_BINF, (vl));		\
  vm_f_t vmef_axc = VM_VFMERGE_VFM ((ax), 1.0,				\
				    VM_VMSLTU_VX (vmef_bx, VM_ERF_B1, (vl)), (vl)); \
  vmef_axc = VM_VFMERGE_VFM (vmef_axc, 40.0,				\
			     VM_VMSGTU_VX (vmef_bx, VM_ERF_B40, (vl)), (vl)); \
  vmef_axc = VM_VFMERGE_VFM (vmef_axc, 1.0, vmef_minf, (vl));		\
  vmef_axc = VM_VFMERGE_VVM (vmef_axc, (ax), vmef_mnan, (vl));		\
  vmef_axc = VM_VFMERGE_VFM (vmef_axc, 25.0, (mceil), (vl));		\
  vm_f_t vmef_th, vmef_tl;						\
  VM_TWO_SUM (vmef_th, vmef_tl, VM_VFMUL_VF (vmef_axc, 2.0, (vl)),	\
	      VM_VFMV_V_F (-41.0, (vl)), (vl));				\
  VM_DD_DIV (vmef_th, vmef_tl, vmef_th, vmef_tl,			\
	     VM_VFMV_V_F (39.0, (vl)), VM_VFMV_V_F (0.0, (vl)), (vl));	\
  vm_f_t vmef_ph, vmef_pl, vmef_qh, vmef_ql;				\
  VM_ERF_CHEB (vmef_ph, vmef_pl, vm_erfc_rp, 13, vmef_th, vmef_tl, (vl)); \
  VM_ERF_CHEB (vmef_qh, vmef_ql, vm_erfc_rq, 13, vmef_th, vmef_tl, (vl)); \
  vm_f_t vmef_hh, vmef_hl;						\
  VM_DD_DIV (vmef_hh, vmef_hl, vmef_ph, vmef_pl, vmef_qh, vmef_ql, (vl)); \
  vm_f_t vmef_x2h, vmef_x2l, vmef_ah, vmef_al, vmef_eh, vmef_el;	\
  vm_f_t vmef_ch, vmef_cl, vmef_d;					\
  /* x^2 from the clamped argument -- see the +-Inf note above.  */	   \
  VM_TWO_PROD (vmef_x2h, vmef_x2l, vmef_axc, vmef_axc, (vl));		\
  /* exp(ah) * exp(al + C_lo - x2l), where ah captures -x2h + C_hi         \
     exactly -- VM_TWO_SUM is exact for any magnitudes.  What is left is   \
     <= 2^-43, so the linear factor is good to 2^-87.  */                  \
  VM_TWO_SUM (vmef_ah, vmef_al, VM_VFNEG_V (vmef_x2h, (vl)),		\
	      VM_VFMV_V_F (VM_ERF_SLN2H, (vl)), (vl));			\
  VM_EXP_DD (vmef_eh, vmef_el, vmef_ah, (vl), exp);			\
  vmef_d = VM_VFADD_VF (vmef_al, VM_ERF_SLN2L, (vl));			\
  vmef_d = VM_VFSUB_VV (vmef_d, vmef_x2l, (vl));			\
  /* The factor has to be a dd: as a plain double, 1+d rounds to a         \
     relative 2^-53, and that single rounding is half an ulp of the        \
     answer.  VM_TWO_SUM keeps the pair.  */                               \
  VM_TWO_SUM (vmef_ch, vmef_cl, VM_VFMV_V_F (1.0, (vl)), vmef_d, (vl));	\
  VM_DD_MUL (vmef_eh, vmef_el, vmef_eh, vmef_el, vmef_ch, vmef_cl, (vl)); \
  VM_DD_MUL (vmef_eh, vmef_el, vmef_eh, vmef_el, vmef_hh, vmef_hl, (vl)); \
  VM_DD_DIV ((eh), (el), vmef_eh, vmef_el, vmef_axc,			\
	     VM_VFMV_V_F (0.0, (vl)), (vl));				\
  /* x = +-Inf: erfc is exactly 0 (and erf +-1, erfc(-Inf) = 2).  Reaching  \
     it through the arithmetic would need exp(-Inf) and an Inf/Inf, and the  \
     dd pair the exp hands back is not worth trusting there.  */	    \
  {									\
    (eh) = VM_VFMERGE_VFM ((eh), 0.0, vmef_minf, (vl));			\
    (el) = VM_VFMERGE_VFM ((el), 0.0, vmef_minf, (vl));			\
  }									\
} while (0)

/* erf(x) for |x| < 1, as a dd, before the sign.  x0 is the argument with
   the lanes that do not use this branch already zeroed (see the entries):
   x0*x0 and x0*P must be exactly zero there, or inf*inf (INVALID plus
   OVERFLOW) or a huge x*x (OVERFLOW) would raise on a lane whose answer
   comes from the tail.  mtiny marks |x| < 2^-32, where P(x^2) is its
   constant term to 2^-64 and the dd multiply is not usable.  */
#define VM_ERF_SMALL(ph, pl, x0, mtiny, vl) do {			\
  /* The polynomial sees zero on the mtiny lanes, not their tiny argument: \
     x^2 of 1e-200 and its rounding raise UNDERFLOW on a lane whose answer \
     does not come from here; the merge below replaces it anyway.  */      \
  vm_f_t vmes_x = VM_VFMERGE_VFM ((x0), 0.0, (mtiny), (vl));		\
  vm_f_t vmes_th, vmes_tl;						\
  VM_TWO_PROD (vmes_th, vmes_tl, vmes_x, vmes_x, (vl));			\
  VM_ERF_POLY (ph, pl, vm_erf_p, 16, vmes_th, vmes_tl, (vl));		\
  VM_DD_MUL (ph, pl, ph, pl, vmes_x, VM_VFMV_V_F (0.0, (vl)), (vl));	\
  {									\
    /* |x| < 2^-32: the answer is x*(2/sqrt(pi)) to 2^-64, computed in the \
       same 2^256 domain as the tail: the dd multiply above cannot compute \
       it directly, since x*P_hi's residual lands subnormal and raises     \
       UNDERFLOW on a lane whose answer is not tiny at all.  */            \
    /* The scale is per lane: a NORMAL x can still have a spurious         \
       subnormal x*C residual, so those lanes run 2^256 up; a SUBNORMAL x  \
       has a subnormal answer, where UNDERFLOW is wanted and a scale would \
       only hide it, so those stay at 1.0.  Either way the residual is     \
       folded back with fma, leaving the collapse as the only rounding.  */ \
    vm_b_t vmez_sub = VM_VMSLEU_VX (				   \
	VM_BITS (VM_VFABS_V ((x0), (vl))), VM_ERF_BSUB, (vl));		   \
    vm_f_t vmez_s = VM_VFMERGE_VFM (VM_VFMV_V_F (VM_ERF_S, (vl)),		   \
				    1.0, vmez_sub, (vl));			   \
    vm_f_t vmez_si = VM_VFMERGE_VFM (VM_VFMV_V_F (VM_ERF_SINV, (vl)),	   \
				     1.0, vmez_sub, (vl));			   \
    vm_f_t vmez_x = VM_VFMUL_VV ((x0), vmez_s, (vl));			   \
    vm_f_t vmez_p, vmez_e, vmez_f, vmez_r;				\
    VM_TWO_PROD (vmez_p, vmez_e, vmez_x,				\
		 VM_VFMV_V_F (vm_erf_p[0][0], (vl)), (vl));		\
    vmez_f = VM_VFMACC_VF (vmez_e, vm_erf_p[0][1], vmez_x, (vl));	\
    vmez_r = VM_VFMUL_VV (VM_VFADD_VV (vmez_p, vmez_f, (vl)),		\
			  vmez_si, (vl));				\
    (ph) = VM_VFMERGE_VVM ((ph), vmez_r, (mtiny), (vl));		\
    (pl) = VM_VFMERGE_VFM ((pl), 0.0, (mtiny), (vl));			\
  }									\
} while (0)

static inline vm_f_t
VM_KFN (erf) (vm_f_t x, size_t vl)
{
  vm_f_t vax = VM_VFABS_V (x, vl);
  vm_u64_t vbx = VM_BITS (vax);
  /* |x| >= 1, and true for a NaN as well -- whose answer the tail then
     propagates.  */
  vm_b_t vbig = VM_VMSGEU_VX (vbx, VM_ERF_B1, vl);
  vm_b_t vtiny = VM_VMSLTU_VX (vbx, VM_ERF_BTINY, vl);
  vm_b_t vmnan = VM_VMSGTU_VX (vbx, VM_ERF_BINF, vl);
  vm_f_t vzero = VM_VFMV_V_F (0.0, vl);
  vm_f_t vph, vpl, vsmall;

  VM_ERF_SMALL (vph, vpl, VM_VFMERGE_VFM (x, 0.0, vbig, vl), vtiny, vl);
  VM_DD_COLLAPSE (vsmall, vph, vpl, vl);
  /* the dd collapse folds -0 + +0 into +0; erf(+-0) = +-0.  */
  vsmall = VM_VFMERGE_VVM (vsmall, VM_VFSGNJ_VV (vzero, x, vl),
			   VM_VMFEQ_VF (vsmall, 0.0, vl), vl);

  /* 1 - erfc(|x|), then handed the sign of x: erf is odd.  Past |x| = 25
     erfc < 1e-272, so 1 - erfc is exactly 1 -- and taking the true
     argument there would drive exp() subnormal and raise UNDERFLOW on a
     lane whose answer is exactly +-1.  */
  {
    vm_f_t vth, vtl, voh, vol, vtail;
    vm_b_t vc25 = VM_VMAND_MM (VM_VMSGTU_VX (vbx, VM_ERF_B25, vl),
			       VM_VMNOT_M (vmnan, vl), vl);
    VM_ERFC_TAIL (vth, vtl, vax, vc25, vl);
    VM_DD_SUB (voh, vol, VM_VFMV_V_F (VM_ERF_S, vl), vzero, vth, vtl, vl);
    VM_DD_COLLAPSE (vtail, voh, vol, vl);
    vtail = VM_VFMUL_VF (vtail, VM_ERF_SINV, vl);
    return VM_VFMERGE_VVM (vsmall, VM_VFSGNJ_VV (vtail, x, vl), vbig, vl);
  }
}

static inline vm_f_t
VM_KFN (erfc) (vm_f_t x, size_t vl)
{
  vm_f_t vax = VM_VFABS_V (x, vl);
  vm_u64_t vbx = VM_BITS (vax);
  vm_b_t vbig = VM_VMSGEU_VX (vbx, VM_ERF_B1, vl);
  vm_b_t vtiny = VM_VMSLTU_VX (vbx, VM_ERF_BTINY, vl);
  vm_b_t vmnan = VM_VMSGTU_VX (vbx, VM_ERF_BINF, vl);
  /* x < 0 is the sign bit of x, as an unsigned compare -- vmslt_vx is
     the signed-integer form and will not take a u64 vector.  */
  vm_b_t vneg = VM_VMSGEU_VX (VM_BITS (x), 0x8000000000000000ULL, vl);
  vm_f_t vzero = VM_VFMV_V_F (0.0, vl);
  vm_f_t vph, vpl, vsmall;

  /* |x| < 1: 1 - x*P(x^2).  P's own error is 5.6e-25, so this subtraction
     carries no cancellation worth worrying about until well past 1.

     Subnormal x is zeroed as well: erfc(x) = 1 - 1.13*x, and 1.13*x is
     itself subnormal there, so the true answer is exactly 1 -- but forming
     that product rounds and raises UNDERFLOW on a lane whose answer is 1.
     Zeroing makes the product exactly 0 and the answer exactly 1.  */
  {
    vm_b_t vsub = VM_VMSLTU_VX (vbx, 0x0010000000000000ULL, vl);
    VM_ERF_SMALL (vph, vpl,
		  VM_VFMERGE_VFM (x, 0.0, VM_VMOR_MM (vbig, vsub, vl), vl),
		  vtiny, vl);
  }
  {
    vm_f_t voh, vol;
    VM_DD_SUB (voh, vol, VM_VFMV_V_F (1.0, vl), vzero, vph, vpl, vl);
    VM_DD_COLLAPSE (vsmall, voh, vol, vl);
  }

  {
    vm_f_t vth, vtl, vpos, vneg2, vnh, vnl, vtail;
    /* x < 0: erfc(-a) = 2 - erfc(a), which is exactly 2 for a > 25, so the
       tail's argument is clamped there rather than evaluated.  erfc(30)
       underflows, and that UNDERFLOW would be a flag the reference does
       not have.  */
    vm_b_t vc25 = VM_VMAND_MM (vneg,
			       VM_VMAND_MM (VM_VMSGTU_VX (vbx, VM_ERF_B25, vl),
					    VM_VMNOT_M (vmnan, vl), vl), vl);
    VM_ERFC_TAIL (vth, vtl, vax, vc25, vl);
    /* the tail hands back (erfc(ax) * 2^256); collapse up there and scale
       back once, so the scale-back is the only rounding.  */
    VM_DD_COLLAPSE (vpos, vth, vtl, vl);
    vpos = VM_VFMUL_VF (vpos, VM_ERF_SINV, vl);
    /* erfc(-x) = 2 - erfc(x), the 2 being 2*2^256 up there */
    VM_DD_SUB (vnh, vnl, VM_VFMV_V_F (VM_ERF_S2, vl), vzero, vth, vtl, vl);
    VM_DD_COLLAPSE (vneg2, vnh, vnl, vl);
    vneg2 = VM_VFMUL_VF (vneg2, VM_ERF_SINV, vl);
    vtail = VM_VFMERGE_VVM (vneg2, vpos, VM_VMNOT_M (vneg, vl), vl);
    return VM_VFMERGE_VVM (vsmall, vtail, vbig, vl);
  }
}
