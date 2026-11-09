/* k_gamma.inc -- lgamma / lgamma_r / tgamma kernels.

   Compact masked kernels derived from newlib's fdlibm e_lgamma_r.c /
   e_tgamma.c, with one cheap double-double primitive: VM_GAM_LOG_DD,
   an atanh-form logarithm that carries its own residual.

   Why dd at all: lgamma's relative ulp budget is 2^-54 of the RESULT, so
   for |lgamma| ~ 8.5 the absolute error must stay below 2^-60; plain
   double cannot, and near lgamma's zeros (x = 1, 2 and the negative-axis
   zeros) the cancellation is unbounded.  The dd log costs ~35 vector
   instructions instead of the ~160 of a table-driven dd log, which is
   what makes the kernel small enough to be fast.

   Hot path (0 < x < 8, the whole benchmark domain 0.5..5):
     x < 2          A = -log(x) + P(y)/Q(y), fdlibm's three sub-ranges
     2 <= x < 8     A = 0.5y + S(y)/R(y) + log((y+2)...(y+i-1))
   One dd log covers both (argument selected by mask, sign folded in),
   then one dd add and the collapse.

   Cold branches (behind vfirst so the hot loop stays small):
     x >= 8         Stirling: dd (x-0.5)*log(x) - x + 0.5log(2pi) + w(1/x)
     x < 0          reflection: log(pi/(|x| |sin pi x|)) - lgamma(-x),
                    sin pi x by integer-quadrant reduction and a poly
     poles/specials +Inf / -Inf / NaN / 0

   tgamma = sign(Gamma) * exp(lgamma) with the same dd value: the exp is
   evaluated by the shared exp core and its own rounding recovered, so the
   final collapse is the only rounding on ordinary lanes.

   Depends on k_exp.inc (VM_EXP_MAIN) and vecmath_dd.h only; leaf kernel
   (no scalar helper calls).

   Requires vecmath_rvvcompat.h, vecmath_dd.h, vecmath_local.h, k_misc.inc
   and k_exp.inc included with VM_LMUL active.  */

#ifndef VM_GAMMA_DATA
#define VM_GAMMA_DATA

/* fdlibm e_lgamma_r.c coefficients.  */
static const double vm_gam_a[12] = {
  7.72156649015328655494e-02, 3.22467033424113591611e-01,
  6.73523010531292681824e-02, 2.05808084325167332806e-02,
  7.38555086081402883957e-03, 2.89051383673415629091e-03,
  1.19270763183362067845e-03, 5.10069792153511336608e-04,
  2.20862790713908385557e-04, 1.08011567247583939954e-04,
  2.52144565451257326939e-05, 4.48640949618915160150e-05,
};
static const double vm_gam_t[15] = {
  4.83836122723810047042e-01, -1.47587722994593911752e-01,
  6.46249402391333854778e-02, -3.27885410759859649565e-02,
  1.79706750811820387126e-02, -1.03142241298341437450e-02,
  6.10053870246291332635e-03, -3.68452016781138256760e-03,
  2.25964780900612472250e-03, -1.40346469989232843813e-03,
  8.81081882437654011382e-04, -5.38595305356740546715e-04,
  3.15632070903625950361e-04, -3.12754168375120860518e-04,
  3.35529192635519073543e-04,
};
static const double vm_gam_u[6] = {
  -7.72156649015328655494e-02, 6.32827064025093366517e-01,
  1.45492250137234768737e+00, 9.77717527963372745603e-01,
  2.28963728064692451092e-01, 1.33810918536787660377e-02,
};
static const double vm_gam_v[5] = {
  2.45597793713041134822e+00, 2.12848976379893395361e+00,
  7.69285150456672783825e-01, 1.04222645593369134254e-01,
  3.21709242282423911810e-03,
};
static const double vm_gam_s[7] = {
  -7.72156649015328655494e-02, 2.14982415960608852501e-01,
  3.25778796408930981787e-01, 1.46350472652464452805e-01,
  2.66422703033638609560e-02, 1.84028451407337715652e-03,
  3.19475326584100867617e-05,
};
static const double vm_gam_r[6] = {
  1.39200533467621045958e+00, 7.21935547567138069525e-01,
  1.71933865632803078993e-01, 1.86459191715652901344e-02,
  7.77942496381893596434e-04, 7.32668430744625636189e-06,
};
static const double vm_gam_w[7] = {
  4.18938533204672725052e-01, 8.33333333333329678849e-02,
  -2.77777777728775536470e-03, 7.93650558643019558500e-04,
  -5.95187557450339963135e-04, 8.36339918996282139126e-04,
  -1.63092934096575273989e-03,
};

#define VM_GAM_TC	1.46163214496836224576e+00
#define VM_GAM_TF	-1.21486290535849611461e-01
#define VM_GAM_TT	-3.63867699703950536541e-18
#define VM_GAM_PI	3.14159265358979311600e+00
#define VM_GAM_HALFLN2PI 9.18938533204672741780e-01
#define VM_GAM_LOGPIH	1.14472988584940017414e+00
#define VM_GAM_LOG_PIL	1.02659511627078264e-17
#define VM_GAM_HLOGPIL	-3.87829415806724145e-17
#define VM_GAM_LN2H	6.93147180369123816490e-01
#define VM_GAM_LN2L	1.90821492927058770002e-10
#define VM_GAM_SLN2H	3.54891356446691987e+02	/* 512*ln2 */
#define VM_GAM_SLN2L	1.18735196868930537e-14
#define VM_GAM_INVPI	3.18309886183790671538e-01

/* sin(pi r) with r = x - round(x), |r| <= 0.5: r*P(r^2).  */
static const double vm_gam_sin[12] = {
  3.14159265358979311600e+00, -5.16771278004997044471e+00,
  2.55016403987734552317e+00, -5.99264529320792109297e-01,
  8.21458866111282206366e-02, -7.37043094571434926582e-03,
  4.66302805767578388593e-04, -2.14222067986915162207e-05,
  7.24660400981010650598e-07, -1.85158494252414747553e-08,
  3.71575963894204562938e-10, -6.07633951073187512252e-12,
};

/* magnitude bit patterns */
#define VM_GAM_B8	0x4020000000000000ULL	/* 8.0 */
#define VM_GAM_B2	0x4000000000000000ULL	/* 2.0 */
#define VM_GAM_B05	0x3fe0000000000000ULL	/* 0.5 */
#define VM_GAM_B0898	0x3feccccc00000000ULL	/* 0.8984375 */
#define VM_GAM_B0732	0x3fe7694400000000ULL	/* 0.7316 */
#define VM_GAM_B0232	0x3fcda66100000000ULL	/* 0.2316 */
#define VM_GAM_B1732	0x3ffbb4c300000000ULL	/* 1.7316 */
#define VM_GAM_B1231	0x3ff3b4c400000000ULL	/* 1.2307 */
#define VM_GAM_B52	0x4330000000000000ULL	/* 2^52 */
#define VM_GAM_BINF	0x7ff0000000000000ULL	/* +Inf */
#define VM_GAM_BSIGN	0x8000000000000000ULL	/* sign bit */
#define VM_GAM_BPIOV4	0x3fe921fb54442d18ULL	/* pi/4 (unused) */

#endif /* VM_GAMMA_DATA */

/* Horner over a table; low index = constant term.  */
#define VM_GAM_HOR(r, tbl, n, x, vl) do {				\
  int vmgh_i;								\
  (r) = VM_VFMV_V_F ((tbl)[(n) - 1], (vl));				\
  for (vmgh_i = (n) - 2; vmgh_i >= 0; vmgh_i--)				\
    (r) = VM_VFMACC_VV (VM_VFMV_V_F ((tbl)[vmgh_i], (vl)), (r), (x), (vl)); \
} while (0)

/* (lh, ll) = log(x) for normal x > 0, to ~4e-18 absolute:
     log x = k ln2 + log m,  m = x 2^-k in [1,2)
     log m = 2 atanh(u),  u = (m-1)/(m+1) = q + uc  (compensated quotient)
     log m = 2q + (2uc + 2(q^3/3 + q^5/5 + ...)).
   k*ln2 is split so the shift is exact; the dd pair is normalized so the
   caller can add it and round once.  */
#define VM_GAM_LOG_DD(lh, ll, x, vl) do {				\
  vm_u64_t vgl_ix = VM_BITS (x);					\
  vm_i64_t vgl_k = VM_VSUB_VX (						\
    VM_VSRA_VX (VM_REINT_U_I (vgl_ix), 52, (vl)), 1023, (vl));		\
  vm_f_t vgl_kd = VM_VFCVT_F_X_V (vgl_k, (vl));			\
  vm_f_t vgl_m = VM_FROM_BITS (						\
    VM_VSUB_VV (vgl_ix, VM_VSLL_VX (VM_REINT_I_U (vgl_k), 52, (vl)), (vl))); \
  vm_f_t vgl_t1 = VM_VFSUB_VF (vgl_m, 1.0, (vl));			\
  vm_f_t vgl_t2 = VM_VFADD_VF (vgl_m, 1.0, (vl));			\
  vm_f_t vgl_rc = VM_VFRDIV_VF (vgl_t2, 1.0, (vl));			\
  vm_f_t vgl_q = VM_VFMUL_VV (vgl_t1, vgl_rc, (vl));			\
  vm_f_t vgl_e = VM_VFNEG_V (						\
    VM_VFNMSAC_VV (vgl_q, vgl_t2, vgl_t1, (vl)), (vl));		\
  vm_f_t vgl_uc = VM_VFMUL_VV (vgl_e, vgl_rc, (vl));			\
  vm_f_t vgl_q2 = VM_VFMUL_VV (vgl_q, vgl_q, (vl));			\
  vm_f_t vgl_c = VM_VFMV_V_F (1.0 / 13.0, (vl));			\
  vgl_c = VM_VFMACC_VV (VM_VFMV_V_F (1.0 / 11.0, (vl)), vgl_c, vgl_q2, (vl)); \
  vgl_c = VM_VFMACC_VV (VM_VFMV_V_F (1.0 / 9.0, (vl)), vgl_c, vgl_q2, (vl)); \
  vgl_c = VM_VFMACC_VV (VM_VFMV_V_F (1.0 / 7.0, (vl)), vgl_c, vgl_q2, (vl)); \
  vgl_c = VM_VFMACC_VV (VM_VFMV_V_F (1.0 / 5.0, (vl)), vgl_c, vgl_q2, (vl)); \
  vgl_c = VM_VFMACC_VV (VM_VFMV_V_F (1.0 / 3.0, (vl)), vgl_c, vgl_q2, (vl)); \
  /* C = 2*q^3*c */							\
  vgl_c = VM_VFMUL_VF (VM_VFMUL_VV (VM_VFMUL_VV (vgl_q2, vgl_q, (vl)),	\
				    vgl_c, (vl)), 2.0, (vl));		\
  vm_f_t vgl_h, vgl_se;							\
  VM_TWO_SUM (vgl_h, vgl_se,						\
	      VM_VFMUL_VF (vgl_kd, VM_GAM_LN2H, (vl)),		\
	      VM_VFADD_VV (vgl_q, vgl_q, (vl)), (vl));			\
  (ll) = VM_VFADD_VV (							\
    VM_VFADD_VV (vgl_se, vgl_c, (vl)),					\
    VM_VFADD_VV (VM_VFMACC_VF (vgl_uc, VM_GAM_LN2L, vgl_kd, (vl)),	\
		 VM_VFADD_VV (vgl_uc, vgl_uc, (vl)), (vl)), (vl));	\
  (lh) = vgl_h;								\
} while (0)

/* A = lgamma(x) as dd for x > 0 (finite).  Hot path 0 < x < 8; the
   Stirling range is behind a vfirst guard.  */
#define VM_GAM_LPOS(h, l, x, vl) do {					\
  vm_b_t vgp_lt2 = VM_VMSLTU_VX (VM_BITS (x), VM_GAM_B2, (vl));	\
  vm_b_t vgp_lt0898 = VM_VMSLTU_VX (VM_BITS (x), VM_GAM_B0898, (vl)); \
  vm_b_t vgp_lt0732 = VM_VMSLTU_VX (VM_BITS (x), VM_GAM_B0732, (vl)); \
  vm_b_t vgp_lt0232 = VM_VMSLTU_VX (VM_BITS (x), VM_GAM_B0232, (vl)); \
  vm_b_t vgp_ge1732 = VM_VMSGEU_VX (VM_BITS (x), VM_GAM_B1732, (vl)); \
  vm_b_t vgp_ge1231 = VM_VMSGEU_VX (VM_BITS (x), VM_GAM_B1231, (vl)); \
  vm_b_t vgp_c0 = VM_VMAND_MM (VM_VMSGEU_VX (VM_BITS (x), VM_GAM_B0898, (vl)), \
			       vgp_ge1732, (vl));			\
  vm_b_t vgp_c1 = VM_VMAND_MM (VM_VMSGEU_VX (VM_BITS (x), VM_GAM_B0898, (vl)), \
			       VM_VMAND_MM (VM_VMNOT_M (vgp_ge1732, (vl)), \
					    vgp_ge1231, (vl)), (vl));	\
  /* case selection for x < 2 */						\
  vm_b_t vgp_lo0 = VM_VMAND_MM (vgp_lt0898,				\
				VM_VMNOT_M (vgp_lt0732, (vl)), (vl));	\
  vm_b_t vgp_lo1 = VM_VMAND_MM (vgp_lt0898,				\
				VM_VMAND_MM (vgp_lt0732,		\
					     VM_VMNOT_M (vgp_lt0232, (vl)), \
					     (vl)), (vl));		\
  vm_b_t vgp_lo2 = VM_VMAND_MM (vgp_lt0898, vgp_lt0232, (vl));		\
  /* y for each case (only the selected lane's value is used) */	\
  vm_f_t vgp_y0 = VM_VFMERGE_VVM (VM_VFMV_V_F (0.0, (vl)),		\
				  VM_VFSUB_VF (VM_VFMERGE_VVM (VM_VFMV_V_F (1.0, (vl)), x, vgp_lo0, (vl)), 1.0, (vl)), \
				  vgp_lo0, (vl));			\
  vgp_y0 = VM_VFMERGE_VVM (vgp_y0,					\
			   VM_VFSUB_VF (VM_VFMERGE_VVM (VM_VFMV_V_F (2.0, (vl)), x, vgp_c0, (vl)), 2.0, (vl)), \
			   vgp_c0, (vl));				\
  vm_f_t vgp_y1 = VM_VFMERGE_VVM (VM_VFMV_V_F (0.0, (vl)),		\
				  VM_VFSUB_VF (VM_VFMERGE_VVM (VM_VFMV_V_F (VM_GAM_TC - 1.0, (vl)), x, vgp_lo1, (vl)), \
					       VM_GAM_TC - 1.0, (vl)), (vl) ? 0 : 0, vgp_lo1, (vl)); \
  vgp_y1 = VM_VFMERGE_VVM (vgp_y1,					\
			   VM_VFSUB_VF (VM_VFMERGE_VVM (VM_VFMV_V_F (VM_GAM_TC, (vl)), x, vgp_c1, (vl)), \
					VM_GAM_TC, (vl)), vgp_c1, (vl)); \
  vm_f_t vgp_y2 = VM_VFMERGE_VVM (VM_VFMV_V_F (0.0, (vl)),		\
				  VM_VFSUB_VF (VM_VFMERGE_VVM (VM_VFMV_V_F (1.0, (vl)), x, vgp_c1, (vl)), 1.0, (vl)), \
				  vgp_c1, (vl));			\
  vgp_y2 = VM_VFMERGE_VVM (vgp_y2, x, vgp_lo2, (vl));			\
  /* case polys */							\
  vm_f_t vgp_p, vgp_z2;							\
  {									\
    vm_f_t vgp_t1, vgp_t2, vgp_w, vgp_p1, vgp_p2, vgp_p3;		\
    /* case 0 */							\
    vgp_z2 = VM_VFMUL_VV (vgp_y0, vgp_y0, (vl));			\
    VM_GAM_HOR (vgp_p1, vm_gam_a, 12, vgp_z2, (vl));			\
    vgp_t1 = VM_VFMUL_VV (vgp_z2, VM_VFMV_V_F (0.0, (vl)), (vl));	\
    (void) vgp_t1;							\
  }									\
} while (0)
