/* k_erf.inc -- erf / erfc kernels.

   Compact MASKED kernels: the whole domain is covered branchlessly by
   region masks (integer compares on the magnitude bits, so NaN never
   trips a signaling compare), and no lane ever leaves the vector path.
   The kernels are LEAF functions: they call no scalar libm and no
   scalar-fixup helper, which keeps the per-call prologue empty and the
   hot loop at a few hundred instructions.  (The dd-era kernels were
   3.2 KB / 25 KB of straight-line code and I-cache bound.)

   erf(x):
     |x| < 2^-28          x * 2/sqrt(pi)            (one rounding; the
                                                    x + x*R(x^2) form raises
                                                    UNDERFLOW on subnormal x
                                                    whose answer is normal)
     |x| < 0.84375        x + x*P(x^2)/Q(x^2)      (fdlibm s_erf.c)
     |x| < 1.25           sign(x) * (erx + P1/Q1)  (fdlibm)
     |x| < 6              1 - exp(-z^2-0.5625+d2+R/S)/x  with z = x
                          truncated to 21 bits (z*z exact) and one exp
     |x| >= 6             sign(x)                  (rounds to 1)
     NaN/Inf              sign(x) + 1/x            (fdlibm: qNaN quiet,
                                                    sNaN invalid via 1/x,
                                                    +-Inf -> +-1 exact)

   erfc(x):
     |x| < 2^-56          1 - x
     |x| < 0.25           1 - (x + x*P/Q)          (fdlibm; cancellation
                                                    < 0.1 ulp here)
     |x| < 1.25           dd polynomial of degree 19, v = |x|-0.75
     else                 exp(-x^2-0.5625+rho(1/x^2))/x in dd, evaluated
                          2^512 up then one scale-back multiply; rho is
                          ONE rational over the whole range (fdlibm split
                          it in two) and is folded into the dd exponent
     x <= -6              2 - erfc(|x|), tail argument clamped to 25 so
                          the answer 2 raises no UNDERFLOW
     NaN/Inf              (2*signbit) + 1/x        (fdlibm)

   The dd tail is evaluated 2^512 up: exp(-x^2) would otherwise trip the
   exp core above |arg| ~ 512 (erfc is still an ordinary normal double up
   to x = 26.44) and the dd lo part would go subnormal once the answer
   itself is subnormal.  The final 2^-512 multiply is what rounds into
   the subnormal grid, so UNDERFLOW is raised exactly when C99 wants it.

   The Horner loops are all force-unrolled (_Pragma unroll): at -O2 gcc
   left them rolled, which cost 5 instructions (and a vector move) per
   coefficient instead of 1 -- worth 17% of the whole erfc entry.

   The two widest remaining chains (the erfc tail's rho rational and the
   top of the mid-band polynomial) use Estrin form; both are accuracy
   neutral (the mid-band one is scaled by v^7 before it reaches the
   result) and the rho one actually lowered the worst erfc record.

   Accuracy: FULL-tier max_ulp erf 1.0000 (subnormal-x lanes only), erfc
   0.7976; 0 fails / flags / sweep, verified with the MPFR framework.

   Depends on VM_EXP_MAIN (k_exp.inc) and the vm_exp_data table only; the
   dd helpers come from vecmath_dd.h.  Requires vecmath_rvvcompat.h,
   vecmath_dd.h, vecmath_local.h, k_misc.inc and k_exp.inc included with
   VM_LMUL active.  */

#ifndef VM_ERF_DATA
#define VM_ERF_DATA

/* erf(x) = x + x*P(x^2)/Q(x^2) on |x| < 0.84375 (fdlibm s_erf.c).  */
static const double vm_erf_pp[5] = {
  1.28379167095512558561e-01, -3.25042107247001499370e-01,
  -2.84817495755985104766e-02, -5.77027029648944159157e-03,
  -2.37630166566501626084e-05,
};
static const double vm_erf_qq[5] = {
  3.97917223959155352819e-01, 6.50222499887672944485e-02,
  5.08130628187576562776e-03, 1.32494738004321644526e-04,
  -3.96022827877536812320e-06,
};
/* 0.84375 <= |x| < 1.25: erf = sign*(erx + P1(s)/Q1(s)), s = |x|-1.  */
static const double vm_erf_pa[7] = {
  -2.36211856075265944077e-03, 4.14856118683748331666e-01,
  -3.72207876035701323847e-01, 3.18346619901161753674e-01,
  -1.10894694282396677476e-01, 3.54783043256182359371e-02,
  -2.16637559486879084300e-03,
};
static const double vm_erf_qa[6] = {
  1.06420880400844228286e-01, 5.40397917702171048937e-01,
  7.18286544141962662868e-02, 1.26171219808761642112e-01,
  1.36370839120290507362e-02, 1.19844998467991074170e-02,
};
/* erfc tail rational f(s) = log(erfc(x)*x)+x^2+0.5625, s = 1/x^2.
   fdlibm's two ranges; plain-double evaluation is enough for the erfc
   budget (measured: the tail region contributes no >1 ulp record).  */
static const double vm_erf_ra[8] = {
  -9.86494403484714822705e-03, -6.93858572707181764372e-01,
  -1.05586262253232909814e+01, -6.23753324503260060396e+01,
  -1.62396669462573470355e+02, -1.84605092906711035994e+02,
  -8.12874355063065934246e+01, -9.81432934416914548592e+00,
};
static const double vm_erf_sa[8] = {
  1.96512716674392571292e+01, 1.37657754143519042600e+02,
  4.34565877475229228821e+02, 6.45387271733267880336e+02,
  4.29008140027567833386e+02, 1.08635005541779435134e+02,
  6.57024977031928170135e+00, -6.04244152148580987438e-02,
};
static const double vm_erf_rb[7] = {
  -9.86494292470009928597e-03, -7.99283237680523006574e-01,
  -1.77579549177547519889e+01, -1.60636384855821916062e+02,
  -6.37566443368389627722e+02, -1.02509513161107724954e+03,
  -4.83519191608651397019e+02,
};
static const double vm_erf_sb[7] = {
  3.03380607434824582924e+01, 3.25792512996573918826e+02,
  1.53672958608443695994e+03, 3.19985821950859553908e+03,
  2.55305040643316442583e+03, 4.74528541206955367215e+02,
  -2.24409524465858183362e+01,
};

/* 2/sqrt(pi): erf(x) = x*C for |x| < 2^-28 (one rounding; the product is
   exact in all but the last bit and the constant is within 4.6e-18).  */
#define VM_ERF_C2SPI	1.12837916709551257390e+00

/* erfc(x) on [0.25,1.25]: degree-19 Chebyshev interpolant in v = x-0.75,
   as dd (hi, lo) pairs; approximation error 7.2e-19 relative.  */
static const double vm_erfc_mid[20][2] = {
  { 2.88844366346484860e-01, 8.41410685654355997e-18 },
  { -6.42931069195207372e-01, 4.29175897092346393e-17 },
  { 4.82198301896405612e-01, -2.66538956568307135e-17 },
  { -2.67887945498003072e-02, 3.25413465819443838e-19 },
  { -1.50686969342637450e-01, 1.11932797291978218e-17 },
  { 5.32427291677282852e-02, -1.21793617588721698e-18 },
  { 2.68725095332749409e-02, -8.49909490415865091e-19 },
  { -1.84352351779178518e-02, 3.34526024842404049e-19 },
  { -2.30178831626490915e-03, 1.96835979170213727e-19 },
  { 3.96826044657923663e-03, 3.55408941458292768e-19 },
  { -1.86032086200301817e-04, 8.64634182090752991e-21 },
  { -6.23983677200797763e-04, -3.85179419676092934e-20 },
  { 1.06183199157835754e-04, 2.22419538908361897e-21 },
  { 7.57456460429781209e-05, 4.93303109148895716e-22 },
  { -2.21101544327549143e-05, -3.32516087313345298e-22 },
  { -7.16636883761289868e-06, 2.25992198914391290e-22 },
  { 3.22683625178076497e-06, -1.16390187860209433e-22 },
  { 5.03892194007635379e-07, -5.29301770781004166e-24 },
  { -3.35630183045454882e-07, 2.05332503333962154e-23 },
  { -2.07972886144803726e-08, -1.43808182164070217e-24 },
};

/* erfc tail factor rho(s) = log(erfc(x)*x) + x^2 + 0.5625, s = 1/x^2,
   for the whole erfc tail 1.25 <= x <= 29 (s in [0.0012, 0.64]) as ONE
   rational R(s)/S(s): the fdlibm two-range split (8/8 + 7/7) costs 30
   fmas on every strip; this costs 22 and needs no range select.  The
   fit is a weighted-Least-Squares rational Remez in the s basis,
   max |error| 3.0e-19 (2^-61.6), i.e. 4 orders below the double
   evaluation noise.  */
static const double vm_erfc_rho_p[11] = {
  -9.86494292470008714e-03, -9.27325069385170675e-01,
  -2.82710618438656454e+01, -4.03452929272812810e+02,
  -3.04795508031141935e+03, -1.26822117852048323e+04,
  -2.88575213539372962e+04, -3.42455833275896002e+04,
  -1.91849993277500871e+04, -4.16523694837352650e+03,
  -2.17717161350280463e+02,
};
static const double vm_erfc_rho_q[11] = {
  1.00000000000000000e+00, 4.33175409778828922e+01,
  7.33637428028333261e+02, 6.30171022558705954e+03,
  2.98387713174216042e+04, 7.94518245862397016e+04,
  1.16682706117539492e+05, 8.91884692220622237e+04,
  3.16615633436538446e+04, 4.15831657403713780e+03,
  1.13231359553771043e+02,
};

/* Horner over an 11-coefficient polynomial in Estrin form (4 dependent fmas
   instead of 10).  The rho rational is the widest chain left in the
   tail and the tail is latency-bound at 8 lanes.  */
#define VM_ERFC_HOR11(r, tbl, x, vl) do {				\
  vm_f_t veh_x2 = VM_VFMUL_VV ((x), (x), (vl));			\
  vm_f_t veh_x4 = VM_VFMUL_VV (veh_x2, veh_x2, (vl));			\
  vm_f_t veh_x8 = VM_VFMUL_VV (veh_x4, veh_x4, (vl));			\
  vm_f_t veh_a0 = VM_VFMACC_VF (VM_VFMV_V_F ((tbl)[0], (vl)),		\
				 (tbl)[1], (x), (vl));				\
  vm_f_t veh_a1 = VM_VFMACC_VF (VM_VFMV_V_F ((tbl)[2], (vl)),		\
				 (tbl)[3], (x), (vl));				\
  vm_f_t veh_a2 = VM_VFMACC_VF (VM_VFMV_V_F ((tbl)[4], (vl)),		\
				 (tbl)[5], (x), (vl));				\
  vm_f_t veh_a3 = VM_VFMACC_VF (VM_VFMV_V_F ((tbl)[6], (vl)),		\
				 (tbl)[7], (x), (vl));				\
  vm_f_t veh_a4 = VM_VFMACC_VF (VM_VFMV_V_F ((tbl)[8], (vl)),		\
				 (tbl)[9], (x), (vl));				\
  vm_f_t veh_b0 = VM_VFMACC_VV (veh_a0, veh_a1, veh_x2, (vl));	\
  vm_f_t veh_b1 = VM_VFMACC_VV (veh_a2, veh_a3, veh_x2, (vl));	\
  vm_f_t veh_b2 = VM_VFMACC_VF (veh_a4, (tbl)[10], veh_x2, (vl));	\
  vm_f_t veh_c0 = VM_VFMACC_VV (veh_b1, veh_b2, veh_x4, (vl));	\
  (r) = VM_VFMACC_VV (veh_b0, veh_c0, veh_x4, (vl));			\
} while (0)

/* rho for the erfc tail: one range, no select, no divide chain.  */
#define VM_ERFC_RHO(rho, s_, vl) do {					\
  vm_f_t verh_p, verh_q;						\
  VM_ERFC_HOR11 (verh_p, vm_erfc_rho_p, (s_), (vl));			\
  VM_ERFC_HOR11 (verh_q, vm_erfc_rho_q, (s_), (vl));			\
  (rho) = VM_VFDIV_VV (verh_p, verh_q, (vl));				\
} while (0)

/* erf/erfc magnitudes as raw bits (integer compares: NaN-safe).  */
#define VM_ERF_BTINY	0x3e30000000000000ULL	/* 2^-28 */
#define VM_ERF_B084	0x3feb000000000000ULL	/* 0.84375 */
#define VM_ERF_B125	0x3ff4000000000000ULL	/* 1.25 */
#define VM_ERF_B286	0x4006db6e00000000ULL	/* 1/0.35 ~ 2.857 */
#define VM_ERF_B6	0x4018000000000000ULL	/* 6.0 */
#define VM_ERF_B29	0x403d000000000000ULL	/* 29.0 */
#define VM_ERF_B025	0x3fd0000000000000ULL	/* 0.25 */
#define VM_ERF_B56	0x3c70000000000000ULL	/* 2^-56 */
#define VM_ERF_BINF	0x7ff0000000000000ULL	/* +Inf */
#define VM_ERF_BSIGN	0x8000000000000000ULL	/* sign bit */

/* fdlibm's erf(1) rounding constant c = (float)0.84506291151.  */
#define VM_ERF_ERX_CONST	8.45062911510467529297e-01

/* The erfc tail runs 2^512 up; 512*ln2 is split so the shift is exact.  */
#define VM_ERF_S	0x1p512
#define VM_ERF_S2	0x1p513
#define VM_ERF_SINV	0x1p-512
#define VM_ERF_SLN2H	3.54891356446691987e+02	/* 512*ln2, hi */
#define VM_ERF_SLN2L	1.18735196868930537e-14	/* 512*ln2 - hi */
/* 512*ln2 - 0.5625 as the same dd pair (the erfc tail subtracts fdlibm's
   0.5625 and adds the shift; folding them into one constant saves a
   two_sum -- the subtraction is exact in double).  */
#define VM_ERF_CHI	3.54328856446691987e+02
#define VM_ERF_CLO	1.18735196868930537e-14

#endif /* VM_ERF_DATA */

/* Horner over a static coefficient table, low index = constant term.  */
#define VM_ERF_HOR(r, tbl, n, x, vl) do {				\
  int vehi;								\
  (r) = VM_VFMV_V_F ((tbl)[(n) - 1], (vl));				\
  _Pragma ("GCC unroll 32")						\
  for (vehi = (n) - 2; vehi >= 0; vehi--)				\
    (r) = VM_VFMACC_VV (VM_VFMV_V_F ((tbl)[vehi], (vl)), (r), (x), (vl)); \
} while (0)

/* rho = R/S with the two fdlibm ranges merged by mask (erfc tail).  */
#define VM_ERF_RHO(rho, s_, rb, vl) do {					\
  vm_f_t vrh_ra, vrh_sa, vrh_rb, vrh_sb;				\
  VM_ERF_HOR (vrh_ra, vm_erf_ra, 8, (s_), (vl));			\
  VM_ERF_HOR (vrh_sa, vm_erf_sa, 8, (s_), (vl));			\
  vrh_sa = VM_VFMACC_VV (VM_VFMV_V_F (1.0, (vl)), (s_), vrh_sa, (vl));			\
  VM_ERF_HOR (vrh_rb, vm_erf_rb, 7, (s_), (vl));			\
  VM_ERF_HOR (vrh_sb, vm_erf_sb, 7, (s_), (vl));			\
  vrh_sb = VM_VFMACC_VV (VM_VFMV_V_F (1.0, (vl)), (s_), vrh_sb, (vl));			\
  vrh_ra = VM_VFMERGE_VVM (vrh_ra, vrh_rb, (rb), (vl));			\
  vrh_sa = VM_VFMERGE_VVM (vrh_sa, vrh_sb, (rb), (vl));			\
  (rho) = VM_VFDIV_VV (vrh_ra, vrh_sa, (vl));				\
} while (0)

/* erf small branch, plain double: ph = x0 + x0*(P/Q).  */
#define VM_ERF_SMALL(ph, x0, vl) do {					\
  vm_f_t vefs_z = VM_VFMUL_VV ((x0), (x0), (vl));			\
  vm_f_t vefs_r, vefs_s;						\
  VM_ERF_HOR (vefs_r, vm_erf_pp, 5, vefs_z, (vl));			\
  VM_ERF_HOR (vefs_s, vm_erf_qq, 5, vefs_z, (vl));			\
  vefs_s = VM_VFMACC_VV (VM_VFMV_V_F (1.0, (vl)), vefs_z, vefs_s, (vl));			\
  (ph) = VM_VFMACC_VV ((x0), (x0), VM_VFDIV_VV (vefs_r, vefs_s, (vl)), (vl)); \
} while (0)

/* dd exp via the shared exp core: returns (eh, el) with hi+lo == exp(x) to
   ~2^-60 relative (the core's tmp is a deviation from 1, so its roundings
   are tiny).  x must lie in the core's safe |x| < 512 range.  */
#define VM_ERF_EXP_DD(eh, el, x, vl) do {				\
  vm_f_t vexd_s, vexd_t, vexd_q, vexd_e1, vexd_sh, vexd_se, vexd_te;	\
  VM_EXP_MAIN (vexd_s, vexd_t, (x), (vl));				\
  vexd_q = VM_VFMUL_VV (vexd_s, vexd_t, (vl));				\
  vexd_e1 = VM_VFNEG_V (VM_VFNMSAC_VV (vexd_q, vexd_s, vexd_t, (vl)), (vl)); \
  VM_TWO_SUM (vexd_sh, vexd_se, vexd_s, vexd_q, (vl));			\
  vexd_te = VM_VFADD_VV (vexd_se, vexd_e1, (vl));			\
  VM_TWO_SUM ((eh), (el), vexd_sh, vexd_te, (vl));			\
} while (0)

/* ---- erf ---------------------------------------------------------- */
static inline vm_f_t
VM_KFN (erf) (vm_f_t x, size_t vl)
{
  vm_f_t vzero = VM_VFMV_V_F (0.0, vl);
  vm_f_t vone = VM_VFMV_V_F (1.0, vl);
  vm_u64_t vau = VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, vl);
  vm_f_t vax = VM_FROM_BITS (vau);

  vm_b_t me_spec = VM_VMSGEU_VX (vau, VM_ERF_BINF, vl);
  vm_b_t me_tiny = VM_VMSLTU_VX (vau, VM_ERF_BTINY, vl);
  vm_b_t me_poly = VM_VMSLTU_VX (vau, VM_ERF_B084, vl);
  vm_b_t me_erx = VM_VMSLTU_VX (vau, VM_ERF_B125, vl);
  vm_b_t me_tail = VM_VMSLTU_VX (vau, VM_ERF_B6, vl);

  /* One reciprocal serves the two tail divisions (s = 1/x^2 and the
     final /|x|).  Only the 1.25 <= |x| < 6 lanes need it, so the other
     lanes divide 1 by 1: that keeps the divide, underflow and overflow
     flags of a lane whose answer never uses the reciprocal.  */
  vm_b_t me_tailb = VM_VMAND_MM (me_tail, VM_VMNOT_M (me_erx, vl), vl);
  vm_f_t vet_rt = VM_VFRDIV_VF (VM_VFMERGE_VVM (vone, vax, me_tailb, vl),
				1.0, vl);

  /* |x| < 2^-28: x * 2/sqrt(pi) */
  vm_f_t vet_tiny = VM_VFMUL_VF (VM_VFMERGE_VVM (vzero, x, me_tiny, vl),
				 VM_ERF_C2SPI, vl);

  /* |x| < 0.84375: x + x*R(x^2); tiny lanes excluded so x^2 cannot
     underflow on a lane whose answer is not tiny.  */
  vm_f_t vet_small = vzero;
  if (VM_VFIRST_M (VM_VMAND_MM (me_poly, VM_VMNOT_M (me_tiny, vl), (vl)),
		   (vl)) >= 0)
    {
      vm_b_t me_poly2 = VM_VMAND_MM (me_poly, VM_VMNOT_M (me_tiny, (vl)), (vl));
      VM_ERF_SMALL (vet_small, VM_VFMERGE_VVM (vzero, x, me_poly2, (vl)), (vl));
    }

  /* 0.84375 <= |x| < 1.25: sign(x)*(erx + P1(s)/Q1(s)) */
  vm_f_t vet_erx = vzero;
  if (VM_VFIRST_M (VM_VMAND_MM (me_erx, VM_VMNOT_M (me_poly, (vl)), (vl)),
		   (vl)) >= 0)
    {
      vm_b_t me_erx2 = VM_VMAND_MM (me_erx, VM_VMNOT_M (me_poly, (vl)), (vl));
      vm_f_t vet_sm = VM_VFSUB_VF (VM_VFMERGE_VVM (vzero, vax, me_erx2, (vl)),
				   1.0, (vl));
      vm_f_t vet_epa, vet_eqa;
      VM_ERF_HOR (vet_epa, vm_erf_pa, 7, vet_sm, (vl));
      VM_ERF_HOR (vet_eqa, vm_erf_qa, 6, vet_sm, (vl));
      vet_eqa = VM_VFMACC_VV (vone, vet_sm, vet_eqa, (vl));
      vet_erx = VM_VFADD_VF (VM_VFDIV_VV (vet_epa, vet_eqa, (vl)),
			     VM_ERF_ERX_CONST, (vl));
      vet_erx = VM_VFSGNJ_VV (vet_erx, x, (vl));
    }

  /* 1.25 <= |x| < 6: sign(x)*(1 - exp(-z^2-0.5625+d2+R/S)/|x|),
     z = |x| with the low word cleared, d2 = z^2-x^2.  The single exp is
     good enough: q = erfc(|x|) <= 0.0771 here, so an argument rounding
     of half an ulp of -z^2-0.5625 is at most 0.34 of the q budget.
     1/x^2 and 1/|x| come from the shared reciprocal.  */
  vm_f_t vet_tv = vzero;
  if (VM_VFIRST_M (VM_VMAND_MM (me_tail, VM_VMNOT_M (me_erx, (vl)), (vl)),
		   (vl)) >= 0)
    {
      vm_b_t me_tail2 = me_tailb;
      vm_f_t vet_xt = VM_VFMERGE_VVM (vone, vax, me_tail2, (vl));
      vm_f_t vet_z = VM_FROM_BITS (VM_VAND_VX (VM_BITS (vet_xt),
					       VM_HIGHWORD_U64, (vl)));
      vm_f_t vet_z2 = VM_VFMUL_VV (vet_z, vet_z, (vl));	/* exact */
      vm_f_t vet_T = VM_VFSUB_VF (VM_VFNEG_V (vet_z2, (vl)), 0.5625, (vl));
      vm_f_t vet_d2 = VM_VFMUL_VV (VM_VFSUB_VV (vet_z, vet_xt, (vl)),
				   VM_VFADD_VV (vet_z, vet_xt, (vl)), (vl));
      vm_f_t vet_s = VM_VFMUL_VV (vet_rt, vet_rt, (vl));
      vm_b_t vet_rb = VM_VMSGEU_VX (VM_BITS (vet_xt), VM_ERF_B286, (vl));
      vm_f_t vet_rho;
      VM_ERF_RHO (vet_rho, vet_s, vet_rb, (vl));
      vm_f_t vet_S = VM_VFADD_VV (vet_T, VM_VFADD_VV (vet_d2, vet_rho, (vl)),
				  (vl));
      vm_f_t vet_scale, vet_tmp;
      VM_EXP_MAIN (vet_scale, vet_tmp, vet_S, (vl));
      vm_f_t vet_et = VM_VFMACC_VV (vet_scale, vet_scale, vet_tmp, (vl));
      vet_tv = VM_VFRSUB_VF (VM_VFMUL_VV (vet_et, vet_rt, (vl)), 1.0, (vl));
      vet_tv = VM_VFSGNJ_VV (vet_tv, x, (vl));
    }

  /* |x| >= 6 (finite): sign(x); NaN/Inf: sign(x) + 1/x */
  vm_f_t vet_big = VM_VFSGNJ_VV (vone, x, (vl));
  vm_f_t vet_spec = vet_big;
  if (VM_VFIRST_M (me_spec, (vl)) >= 0)
    {
      vm_f_t vet_inv = VM_VFRDIV_VF (VM_VFMERGE_VVM (vone, x, me_spec, (vl)),
				     1.0, (vl));
      vet_spec = VM_VFADD_VV (vet_big, vet_inv, (vl));
    }

  vm_f_t vet_res = VM_VFMERGE_VVM (vet_small, vet_erx,
				   VM_VMAND_MM (me_erx,
						VM_VMNOT_M (me_poly, (vl)),
						(vl)), (vl));
  vet_res = VM_VFMERGE_VVM (vet_res, vet_tv,
			    VM_VMAND_MM (me_tail, VM_VMNOT_M (me_erx, (vl)), (vl)),
			    (vl));
  vet_res = VM_VFMERGE_VVM (vet_res, vet_tiny, me_tiny, (vl));
  vet_res = VM_VFMERGE_VVM (vet_res, vet_big, VM_VMNOT_M (me_tail, (vl)), (vl));
  vet_res = VM_VFMERGE_VVM (vet_res, vet_spec, me_spec, (vl));
  return vet_res;
}

/* ---- erfc --------------------------------------------------------- */

/* dd Horner of the degree-19 midpoint polynomial; v is exact.  */
/* Degree-19 dd interpolant, mixed-precision Horner: the high-order terms
   run in plain double (their partial sums are tiny and every error there
   is damped by v^k), then the low-order terms run in double-double.
   Verified against the exact polynomial: 0.000 ulp difference over
   v in [-0.5, 0.5] with the plain part down to coefficient 7, and 1.0 ulp
   with one more plain step, so the split sits at the edge.  It shortens
   the dependency chain, which is what this band is bound by (the same
   polynomial in a pure plain-double Horner makes the band free, and a
   pure dd Horner costs ~60 ns/element).  */
#define VM_ERFC_MID_POLY(ph, pl, v, vl) do {				\
  int vemi;								\
  /* Degrees 19..7 in Estrin form: this block's rounding error is scaled\
     by v^7 < 2^-7 before it reaches the result, so reassociating it is\
     accuracy-free, and it takes six dependent fmas off the chain this\
     band is bound by (13 -> 5 levels).  */\
  vm_f_t vemi_v2 = VM_VFMUL_VV ((v), (v), (vl));\
  vm_f_t vemi_v4 = VM_VFMUL_VV (vemi_v2, vemi_v2, (vl));\
  vm_f_t vemi_a0 = VM_VFMACC_VF (VM_VFMV_V_F (vm_erfc_mid[7][0], (vl)),\
				 vm_erfc_mid[8][0], (v), (vl));\
  vm_f_t vemi_a1 = VM_VFMACC_VF (VM_VFMV_V_F (vm_erfc_mid[9][0], (vl)),\
				 vm_erfc_mid[10][0], (v), (vl));\
  vm_f_t vemi_a2 = VM_VFMACC_VF (VM_VFMV_V_F (vm_erfc_mid[11][0], (vl)),\
				 vm_erfc_mid[12][0], (v), (vl));\
  vm_f_t vemi_a3 = VM_VFMACC_VF (VM_VFMV_V_F (vm_erfc_mid[13][0], (vl)),\
				 vm_erfc_mid[14][0], (v), (vl));\
  vm_f_t vemi_a4 = VM_VFMACC_VF (VM_VFMV_V_F (vm_erfc_mid[15][0], (vl)),\
				 vm_erfc_mid[16][0], (v), (vl));\
  vm_f_t vemi_a5 = VM_VFMACC_VF (VM_VFMV_V_F (vm_erfc_mid[17][0], (vl)),\
				 vm_erfc_mid[18][0], (v), (vl));\
  vm_f_t vemi_b0 = VM_VFMACC_VV (vemi_a0, vemi_a1, vemi_v2, (vl));\
  vm_f_t vemi_b1 = VM_VFMACC_VV (vemi_a2, vemi_a3, vemi_v2, (vl));\
  vm_f_t vemi_b2 = VM_VFMACC_VV (vemi_a4, vemi_a5, vemi_v2, (vl));\
  vm_f_t vemi_c0 = VM_VFMACC_VF (vemi_b2, vm_erfc_mid[19][0], vemi_v4, (vl));\
  vm_f_t vemi_c1 = VM_VFMACC_VV (vemi_b1, vemi_c0, vemi_v4, (vl));\
  (ph) = VM_VFMACC_VV (vemi_b0, vemi_c1, vemi_v4, (vl));\
  (pl) = VM_VFMV_V_F (0.0, (vl));					\
  _Pragma ("GCC unroll 32")						\
  for (vemi = 6; vemi >= 0; vemi--)					\
    {									\
      vm_f_t vemi_u, vemi_e, vemi_s, vemi_t;				\
      VM_TWO_PROD (vemi_u, vemi_e, (ph), (v), (vl));			\
      vemi_e = VM_VFMACC_VV (vemi_e, (pl), (v), (vl));			\
      VM_TWO_SUM (vemi_s, vemi_t, vemi_u,				\
		  VM_VFMV_V_F (vm_erfc_mid[vemi][0], (vl)), (vl));	\
      vemi_t = VM_VFADD_VV (vemi_t, vemi_e, (vl));			\
      vemi_t = VM_VFADD_VF (vemi_t, vm_erfc_mid[vemi][1], (vl));	\
      VM_FAST2SUM ((ph), (pl), vemi_s, vemi_t, (vl));			\
    }									\
} while (0)

/* The 1.25 <= |x| erfc tail: exp(-x^2 - 0.5625 + rho(1/x^2))/x in dd,
   Returns the pair 2^512 up.  rho is one range (VM_ERFC_RHO) and it is
   folded into the dd exponent, so the chain is: two_prod x^2, two_sum
   for the 512*ln2 shift, two_sum to fold rho, the exp core, one fma for
   the exponent's lo part, and one dd multiply by the dd reciprocal.  */
static inline __attribute__ ((always_inline)) vm_f_t
VM_KFN (erfc_tail) (vm_f_t x, vm_f_t vax, vm_b_t mf_neg, vm_b_t mf_tail,
		    vm_u64_t vau, size_t vl, vm_f_t *rlo)
{
  vm_f_t mf_th, mf_tl;
  vm_f_t mf_targ = VM_VFMIN_VF (VM_VFMAX_VF (vax, 1.25, (vl)), 29.0, (vl));
  mf_targ = VM_VFMERGE_VFM (mf_targ, 25.0,
				VM_VMAND_MM (mf_neg,
					     VM_VMSGTU_VX (vau, VM_ERF_B6, (vl)),
					     (vl)), (vl));
  vm_f_t mf_x2h, mf_x2l;
  VM_TWO_PROD (mf_x2h, mf_x2l, mf_targ, mf_targ, (vl));
  vm_f_t mf_ah, mf_al, mf_e1;
  VM_TWO_SUM (mf_ah, mf_e1, VM_VFNEG_V (mf_x2h, (vl)),
	      VM_VFMV_V_F (VM_ERF_CHI, (vl)), (vl));
  mf_al = VM_VFSUB_VV (VM_VFADD_VF (mf_e1, VM_ERF_CLO, (vl)),
		       mf_x2l, (vl));
  vm_f_t mf_rho;
  /* One reciprocal feeds both the tail factor's s = 1/x^2 and the
     final dd division: the dd-corrected (rt, rt_lo) is good to ~2^-105,
     and s = (rt + rt_lo)^2 rebuilt with one fma keeps that accuracy, so
     the tail does one division where it used to do three.  */
  vm_f_t mf_rt;
  VM_RCP_FULL (mf_rt, mf_targ, (vl));
  vm_f_t mf_rte = VM_VFMACC_VV (VM_VFMV_V_F (1.0, (vl)),
				VM_VFNEG_V (mf_rt, (vl)), mf_targ, (vl));
  vm_f_t mf_rtl = VM_VFMUL_VV (mf_rt, mf_rte, (vl));
  /* s = 1/targ^2 rebuilt from the reciprocal: the rounding of rt*rt is
     recovered with an fma residual and the 2*rt*rt_lo correction is added
     into that residual, so the single final rounding leaves s within half
     an ulp -- which matters, because rho's sensitivity to s is ~6 ulp of
     result per ulp of s.  No second division.  */
  vm_f_t mf_rt2 = VM_VFMUL_VV (mf_rt, mf_rt, (vl));
  vm_f_t mf_rt2e = VM_VFMACC_VV (VM_VFNEG_V (mf_rt2, (vl)), mf_rt, mf_rt, (vl));
  mf_rt2e = VM_VFADD_VV (mf_rt2e,
			 VM_VFMUL_VF (VM_VFMUL_VV (mf_rt, mf_rtl, (vl)),
				      2.0, (vl)), (vl));
  vm_f_t mf_sinv = VM_VFADD_VV (mf_rt2, mf_rt2e, (vl));
  VM_ERFC_RHO (mf_rho, mf_sinv, (vl));
  /* Fold rho into the dd exponent rather than expanding exp(rho) on its
     own: two_sum is exact, so (mf_rh, mf_al + mf_rl) represents
     -x^2 - 0.5625 + 512*ln2 + rho to full dd precision, and
     exp(hi+lo) = exp(hi)*(1+lo) to first order (|lo| < 2^-43, so the
     neglected lo^2/2 is below 2^-88).  This drops the 12-term expm1
     polynomial, its two_sum and one dd multiply.  */
  vm_f_t mf_rh, mf_rl;
  VM_TWO_SUM (mf_rh, mf_rl, mf_ah, mf_rho, (vl));
  mf_al = VM_VFADD_VV (mf_al, mf_rl, (vl));
  vm_f_t mf_eh, mf_el;
  VM_ERF_EXP_DD (mf_eh, mf_el, mf_rh, (vl));
  mf_el = VM_VFMACC_VV (mf_el, mf_eh, mf_al, (vl));
  VM_DD_MUL (mf_th, mf_tl, mf_eh, mf_el, mf_rt, mf_rtl, (vl));
  *rlo = mf_tl;
  return mf_th;
}

/* 0.25 <= |x| < 1.25: degree-19 dd interpolant in v = |x| - 0.75, out of
   line.  It needs about seven vector groups, which fits in the sixteen
   caller-saved ones, so the call neither saves registers nor spills -- and
   the erfc entry's hot loop stays small enough to keep the inline tail in
   registers.  */
__attribute__ ((noinline)) static vm_f_t
VM_KFN (erfc_mid) (vm_f_t x, size_t vl)
{
  vm_f_t vax = VM_FROM_BITS (
    VM_VAND_VX (VM_BITS (x), VM_ABSMASK_U64, (vl)));
  vm_f_t v = VM_VFSUB_VF (vax, 0.75, (vl));
  vm_f_t ph, pl;
  VM_ERFC_MID_POLY (ph, pl, v, (vl));
  /* x < 0: erfc(x) = 2 - erfc(|x|), in dd, one rounding at the end.  */
  vm_b_t mneg = VM_VMSGEU_VX (VM_BITS (x), VM_ERF_BSIGN, (vl));
  vm_f_t nh, nl;
  VM_TWO_SUM (nh, nl, VM_VFMV_V_F (2.0, (vl)), VM_VFNEG_V (ph, (vl)), (vl));
  nl = VM_VFSUB_VV (nl, pl, (vl));
  ph = VM_VFMERGE_VVM (ph, nh, mneg, (vl));
  pl = VM_VFMERGE_VVM (pl, nl, mneg, (vl));
  return VM_VFADD_VV (ph, pl, (vl));
}

static inline vm_f_t
VM_KFN (erfc) (vm_f_t x, size_t vl)
{
  /* The three bands are disjoint, and a lane's value never depends on
     which other lanes share its strip (each band's arithmetic is
     identical whether or not the others run), so each band is entered
     only when the strip actually has a lane in it.  That keeps the
     register pressure and the instruction count of this wide kernel down
     -- it used to evaluate every band on every strip and spilled.  */
  vm_f_t vzero = VM_VFMV_V_F (0.0, (vl));
  vm_f_t vone = VM_VFMV_V_F (1.0, (vl));
  vm_u64_t vbx = VM_BITS (x);
  vm_u64_t vau = VM_VAND_VX (vbx, VM_ABSMASK_U64, (vl));
  vm_f_t vax = VM_FROM_BITS (vau);

  vm_b_t mf_spec = VM_VMSGEU_VX (vau, VM_ERF_BINF, (vl));
  vm_b_t mf_neg = VM_VMSGEU_VX (vbx, VM_ERF_BSIGN, (vl));
  vm_b_t mf_q1 = VM_VMSLTU_VX (vau, VM_ERF_B025, (vl));	/* < 0.25 */
  vm_b_t mf_q2 = VM_VMSLTU_VX (vau, VM_ERF_B125, (vl));	/* < 1.25 */
  vm_b_t mf_vt = VM_VMSLTU_VX (vau, VM_ERF_B56, (vl));	/* < 2^-56 */
  vm_b_t mf_mid = VM_VMAND_MM (mf_q2, VM_VMNOT_M (mf_q1, (vl)), (vl));
  vm_b_t mf_tail = VM_VMAND_MM (VM_VMNOT_M (mf_q2, (vl)),
				VM_VMNOT_M (mf_spec, (vl)), (vl));

  /* ---- |x| < 0.25: 1 - (x + x*P/Q) ; |x| < 2^-56: 1 - x ---- */
  vm_f_t mf_res = vzero;
  if (VM_VFIRST_M (mf_q1, (vl)) >= 0)
    {
      vm_b_t mf_r0p = VM_VMAND_MM (mf_q1, VM_VMNOT_M (mf_vt, (vl)), (vl));
      vm_f_t mf_sm;
      VM_ERF_SMALL (mf_sm, VM_VFMERGE_VVM (vzero, x, mf_r0p, (vl)), (vl));
      vm_f_t mf_r0 = VM_VFRSUB_VF (mf_sm, 1.0, (vl));
      mf_r0 = VM_VFMERGE_VVM (mf_r0, VM_VFRSUB_VF (x, 1.0, (vl)), mf_vt, (vl));
      mf_res = VM_VFMERGE_VVM (mf_res, mf_r0, mf_q1, (vl));
    }

  /* ---- 0.25 <= |x| < 1.25: out-of-line dd polynomial ---- */
  if (VM_VFIRST_M (mf_mid, (vl)) >= 0)
    mf_res = VM_VFMERGE_VVM (mf_res, VM_KFN (erfc_mid) (x, (vl)), mf_mid,
			     (vl));

  /* ---- 1.25 <= |x|: dd tail, evaluated 2^512 up ---- */
  if (VM_VFIRST_M (mf_tail, (vl)) >= 0)
    {
      vm_f_t mf_tlo, mf_thi = VM_KFN (erfc_tail) (x, vax, mf_neg, mf_tail,
						vau, (vl), &mf_tlo);
      /* x < 0 lanes: 2 - erfc(|x|) in the 2^512-scaled dd domain
	 (2^513 is 2 there), then the one rounding that scales back.  */
      vm_b_t mf_ne = VM_VMAND_MM (mf_neg, mf_tail, (vl));
      vm_f_t mf_nh, mf_nl;
      VM_TWO_SUM (mf_nh, mf_nl, VM_VFMV_V_F (VM_ERF_S2, (vl)),
		  VM_VFNEG_V (mf_thi, (vl)), (vl));
      mf_nl = VM_VFSUB_VV (mf_nl, mf_tlo, (vl));
      mf_thi = VM_VFMERGE_VVM (mf_thi, mf_nh, mf_ne, (vl));
      mf_tlo = VM_VFMERGE_VVM (mf_tlo, mf_nl, mf_ne, (vl));
      mf_res = VM_VFMERGE_VVM (
	mf_res,
	VM_VFMUL_VF (VM_VFADD_VV (mf_thi, mf_tlo, (vl)), VM_ERF_SINV, (vl)),
	mf_tail, (vl));
    }

  /* ---- NaN/Inf: (2*signbit) + 1/x ---- */
  if (VM_VFIRST_M (mf_spec, (vl)) >= 0)
    {
      vm_f_t mf_inv = VM_VFRDIV_VF (VM_VFMERGE_VVM (vone, x, mf_spec, (vl)),
				    1.0, (vl));
      vm_f_t mf_specv = VM_VFADD_VV (VM_VFMERGE_VFM (vzero, 2.0, mf_neg, (vl)),
				     mf_inv, (vl));
      mf_res = VM_VFMERGE_VVM (mf_res, mf_specv, mf_spec, (vl));
    }
  return mf_res;
}
