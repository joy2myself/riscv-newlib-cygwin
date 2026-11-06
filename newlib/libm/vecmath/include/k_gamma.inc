/* k_gamma.inc -- tgamma / lgamma / lgamma_r as double-double.

   LMUL-generic template: include after vecmath_rvvcompat.h, vecmath_dd.h,
   vecmath_local.h, k_log.inc, k_exp.inc and k_trig.inc, with VM_LMUL
   defined.  Uses VM_LOG_DD, VM_LOG1P_DD, VM_EXP_DD, VM_PI_RED; constants
   from test/gen/fit_gamma.py.

   lgamma(x) for x > 0; C1 = 1-gamma, G(z) = (logGamma(1+z) + log1p(z) -
   z*C1)/z^2 on [-0.5, 0.5] (degree 24):
     x <= 1.5         u = (x<0.5 ? x : x-1)
                      -log1p(u) + u*C1 + u^2*G(u) + (x<0.5 ? -log(x) : 0)
     1.5 < x <= 2.5   w = x-2 ; w*C1 + w^2*G(w)
     2.5 < x < 10     P = prod(x+k, k=0..7) ; stirling(x+8) - log(P)
     x >= 10          stirling(x)

   x < 0 is the reflection lgamma(x) = log(pi) - log|sin(pi x)| - lgamma(1-x),
   whose |sin(pi x)| must be dd.  tgamma = sign(Gamma(x)) * exp(lgamma(x));
   C99 F.10.4.1 gives tgamma(+-0) = +-Inf with divide-by-zero and
   tgamma(negative integer) = NaN with invalid.  */

#ifndef VM_GAMMA_DATA
#define VM_GAMMA_DATA

static const double vm_gam_G[25][2] = {
  { 3.22467033424113203e-01, 1.52033618404275648e-17 },
  { -6.73523010531981020e-02, 6.87671237495866309e-18 },
  { 2.05808084277845464e-02, 1.46281996988265948e-18 },
  { -7.38555102867398568e-03, 3.95551278202764788e-19 },
  { 2.89051033074152336e-03, -4.69340830849864829e-20 },
  { -1.19275391170325907e-03, -7.22187508754371579e-20 },
  { 5.09669524743040065e-04, 2.02600435422150421e-20 },
  { -2.23154758453683984e-04, -7.52509155530826312e-21 },
  { 9.94575127819149590e-05, 2.08945775587849618e-21 },
  { -4.49262367348008467e-05, 1.07524591661951034e-21 },
  { 2.05072127727895729e-05, -1.05447899532642156e-21 },
  { -9.43948834081242959e-06, -3.03795285078814166e-22 },
  { 4.37486683989970052e-06, 2.16937442481511707e-23 },
  { -2.03921491293556333e-06, -3.72651066530606524e-23 },
  { 9.55140632925711910e-07, -9.26178506598952087e-23 },
  { -4.49254152906803301e-07, 2.03663069769047819e-23 },
  { 2.12076442558448890e-07, -2.17435729003077758e-25 },
  { -1.00390272091961978e-07, 5.29850193624943312e-24 },
  { 4.76732277737600733e-08, -9.19573829295848571e-25 },
  { -2.28727595228690857e-08, 1.01594115370066011e-24 },
  { 1.09289172310213077e-08, 2.27646495152011930e-25 },
  { -4.79039305392753769e-09, -8.19515938424527575e-26 },
  { 2.27535080707918358e-09, -2.04085525224513933e-25 },
  { -1.72668381232436087e-09, -7.52699929899111677e-26 },
  { 8.43448878952091817e-10, -2.30496904423607215e-26 },
};

static const double vm_gam_S[12][2] = {
  { 8.33333333333333287e-02, 4.62592926927148533e-18 },
  { -2.77777777777777788e-03, 1.06010879087471541e-19 },
  { 7.93650793650793650e-04, 6.88382331736828211e-22 },
  { -5.95238095238095292e-04, 5.36938218754726024e-20 },
  { 8.41750841750841714e-04, 3.68701748892376936e-20 },
  { -1.91752691752691763e-03, 1.06757027768724749e-19 },
  { 6.41025641025641003e-03, 2.22400445638052172e-19 },
  { -2.95506535947712423e-02, 4.86176095750885531e-19 },
  { 1.79644372368830574e-01, -6.40160048271094580e-19 },
  { -1.39243221690590113e+00, 1.58370569892303027e-17 },
  { 1.34028640441683926e+01, -6.15411410199396641e-16 },
  { -1.56848284626002027e+02, 9.39182314171538895e-15 },
};

static const double vm_gam_PS[11][2] = {
  { -1.66666666666666657e-01, -9.25185853854297066e-18 },
  { 8.33333333333333322e-03, 1.15648231731787138e-19 },
  { -1.98412698412698413e-04, -1.72095582934207053e-22 },
  { 2.75573192239858925e-06, -1.85839327404647208e-22 },
  { -2.50521083854417202e-08, 1.44881407093591197e-24 },
  { 1.60590438368216133e-10, 1.25852945887520981e-26 },
  { -7.64716373181981641e-13, -7.03872877733453001e-30 },
  { 2.81145725434552060e-15, 1.65088427308614326e-31 },
  { -8.22063524662432950e-18, -2.21418941196042654e-34 },
  { 1.95729410633912626e-20, -1.36435038300879085e-36 },
  { -3.86817017063068413e-23, 8.84317765548234385e-40 },
};

static const double vm_gam_PC[11][2] = {
  { -5.00000000000000000e-01, 0.00000000000000000e+00 },
  { 4.16666666666666644e-02, 2.31296463463574266e-18 },
  { -1.38888888888888894e-03, 5.30054395437357706e-20 },
  { 2.48015873015873016e-05, 2.15119478667758816e-23 },
  { -2.75573192239858883e-07, -2.37677146222502973e-23 },
  { 2.08767569878681002e-09, -1.20734505911325997e-25 },
  { -1.14707455977297245e-11, -2.06555127528307454e-28 },
  { 4.77947733238738525e-14, 4.39920548583408126e-31 },
  { -1.56192069685862253e-16, -1.19106796602737540e-32 },
  { 4.11031762331216484e-19, 1.44129733786595271e-36 },
  { -8.89679139245057408e-22, 7.91140261487237622e-38 },
};

static const double vm_gam_logtab[32][3] = {
  { 9.84615384615384670e-01, 1.55041865359651990e-02, -3.27832102289241372e-19 },
  { 9.55223880597014907e-01, 4.58095360312942221e-02, 1.68236390497450161e-19 },
  { 9.27536231884057982e-01, 7.52234212375875178e-02, -4.19588072031643362e-18 },
  { 9.01408450704225372e-01, 1.03796793681643545e-01, -3.19589322261744496e-18 },
  { 8.76712328767123239e-01, 1.31576357788719317e-01, 1.11230008797295896e-17 },
  { 8.53333333333333388e-01, 1.58605030176638517e-01, 2.58338649229855793e-18 },
  { 8.31168831168831224e-01, 1.84922338494011934e-01, -7.38467944050343460e-18 },
  { 8.10126582278481000e-01, 2.10564769107349642e-01, 1.13631059690613693e-17 },
  { 7.90123456790123413e-01, 2.35566071312766967e-01, -2.39433714951873392e-18 },
  { 7.71084337349397630e-01, 2.59957524436925991e-01, 2.41675163417429644e-17 },
  { 7.52941176470588225e-01, 2.83768173130644619e-01, -6.44886800345210525e-18 },
  { 7.35632183908045967e-01, 3.07025035294911874e-01, 1.55787160771249324e-18 },
  { 7.19101123595505598e-01, 3.29753286372468035e-01, -2.56335549994319656e-17 },
  { 7.03296703296703352e-01, 3.51976423157178087e-01, 2.00058530133673765e-17 },
  { 6.88172043010752743e-01, 3.73716409793584003e-01, -2.44991738247711115e-18 },
  { 6.73684210526315774e-01, 3.94993808240868993e-01, 7.43768076936232424e-18 },
  { 6.59793814432989678e-01, 4.15827895143710990e-01, -5.79344080121482218e-18 },
  { 6.46464646464646520e-01, 4.36236766774917961e-01, 2.41828873165900646e-17 },
  { 6.33663366336633671e-01, 4.56237433481587573e-01, 9.07916350878553033e-18 },
  { 6.21359223300970820e-01, 4.75845904869963976e-01, 2.50430698450403127e-17 },
  { 6.09523809523809579e-01, 4.95077266797851412e-01, 1.25087307520943320e-17 },
  { 5.98130841121495282e-01, 5.13945751102234394e-01, -2.45370740219152652e-18 },
  { 5.87155963302752326e-01, 5.32464798869471734e-01, 5.45962273071397454e-17 },
  { 5.76576576576576572e-01, 5.50647117952662302e-01, -1.37206774786850453e-17 },
  { 5.66371681415929196e-01, 5.68504735352668766e-01, -4.03895582216683167e-17 },
  { 5.56521739130434789e-01, 5.86049045003578239e-01, -4.27266963844734176e-17 },
  { 5.47008547008547064e-01, 6.03290851438084141e-01, 2.03483972028783465e-17 },
  { 5.37815126050420145e-01, 6.20240409751857569e-01, 3.47647625634366849e-18 },
  { 5.28925619834710758e-01, 6.36907462237069177e-01, 2.64739831190235584e-17 },
  { 5.20325203252032575e-01, 6.53301272012745571e-01, -3.78647527923636550e-17 },
  { 5.12000000000000011e-01, 6.69430653942629239e-01, 7.42065772756174628e-18 },
  { 5.03937007874015741e-01, 6.85304003098919479e-01, -4.82096651919985849e-17 },
};

static const double vm_gam_expP[15][2] = {
  { 1.00000000000000000e+00, -6.32968935083353042e-26 },
  { 1.00000000000000000e+00, 2.52969944154442632e-22 },
  { 5.00000000000000000e-01, 6.74512678614885528e-23 },
  { 1.66666666666666657e-01, 9.17323379512956752e-18 },
  { 4.16666666666666644e-02, 2.30117218771285506e-18 },
  { 8.33333333333334016e-03, 2.45972278367769266e-19 },
  { 1.38888888888888959e-03, 8.18600338919519492e-20 },
  { 1.98412698412418173e-04, -1.99419120366859650e-21 },
  { 2.48015873015616161e-05, -7.68350482620559677e-22 },
  { 2.75573192810125551e-06, -1.62220063269151526e-22 },
  { 2.75573192696029376e-07, 2.60037653751342004e-23 },
  { 2.50520462435594069e-08, -5.43899810450729544e-25 },
  { 2.08767121113632768e-09, -1.03892160204250539e-25 },
  { 1.60935230499617699e-10, -6.04736090955979553e-27 },
  { 1.14937303015795082e-11, 2.58474497833493001e-28 },
};

#endif /* VM_GAMMA_DATA */

/* ---- scalars ---------------------------------------------------------- */
#define VM_GAM_C1H	4.22784335098467134e-01
#define VM_GAM_C1L	4.94291515243064487e-18
#define VM_GAM_LOGPIH	1.14472988584940016e+00
#define VM_GAM_LOG_PIL	1.02659511627078264e-17
/* log(2*pi)/2 as a proper dd: generate both parts, never type the lo.  */
#define VM_GAM_HLOG2PIH	9.18938533204672781e-01
#define VM_GAM_HLOG2PIL	-3.87829415806724145e-17


/* ---- a log good to ~2^-75, which VM_LOG_DD is not --------------------
   VM_LOG_DD's low part carries only ~7 bits past the double, too little for
   Stirling (log(z) times (z-1/2)) or the reflection.  This one keeps logc as
   a dd; the rest is an 11-term series in r = m*invc_i - 1, |r| <= 1/64
   (1.5e-23).  */
/* ln2 must be split with few significant bits in the hi part, or k*ln2hi
   rounds: 21 bits makes the product exact for |k| <= 2^32.  */
#define VM_GAM_LN2H	6.93147180369123816e-01
#define VM_GAM_LN2L	1.90821492927058770e-10
#define VM_GAM_LOG2EH	1.44269504088896341e+00
#define VM_GAM_LOG2EL	2.03552737409310321e-17

#define VM_GAM_LOG_DD(yh, yl, x, vl, fixup) do {			\
  vm_u64_t vglg_ix0 = VM_BITS (x);					\
  /* A subnormal argument is rescaled by 2^600 (undone in k) so the exponent \
     extraction sees a normal number; the rest goes to the scalar log.  */ \
  vm_b_t vglg_msub = VM_VMSLTU_VX (vglg_ix0, 0x0010000000000000ULL, (vl)); \
  vm_b_t vglg_mspec = VM_VMOR_MM (					\
    VM_VMOR_MM (VM_VMSGEU_VX (vglg_ix0, VM_GAM_BSIGN, (vl)),		\
	       VM_VMSGTU_VX (vglg_ix0, VM_GAM_BINF, (vl)), (vl)),	\
    VM_VMSGEU_VX (vglg_ix0, VM_GAM_BINF, (vl)), (vl));			\
  /* the rescale runs on every lane: 1.0 where it is not wanted, so a 1e308 \
     argument cannot overflow it */ \
  vm_f_t vglg_x = VM_VFMERGE_VVM ((x),				\
	VM_VFMUL_VF (VM_VFMERGE_VFM ((x), 1.0,				\
				     VM_VMNOT_M (vglg_msub, (vl)), (vl)), \
		      0x1p600, (vl)), vglg_msub, (vl));			\
  vm_u64_t vglg_ix = VM_BITS (vglg_x);					\
  /* k as u64: the exponent minus 1023 wraps negative for x < 1, and the \
     signed reinterpret is two's-complement arithmetic.  */ \
  vm_u64_t vglg_k = VM_VSUB_VX (					\
    VM_VSRL_VX (vglg_ix, 52, (vl)), 1023, (vl));			\
  vglg_k = VM_VMERGE_VVM (vglg_k, VM_VSUB_VX (vglg_k, 600, (vl)),	\
			  vglg_msub, (vl));				\
  vm_f_t vglg_m = VM_FROM_BITS (VM_VOR_VX (				\
    VM_VAND_VX (vglg_ix, 0x000fffffffffffffULL, (vl)),			\
    0x3ff0000000000000ULL, (vl)));					\
  /* the special lanes must not index the table; m in [1,2) keeps (m-1)*32 \
     in [0,32) and the truncation in [0,31] */ \
  vglg_m = VM_VFMERGE_VFM (vglg_m, 1.0, vglg_mspec, (vl));		\
  vm_i64_t vglg_i = VM_VFCVT_RTZ_X_F_V (				\
    VM_VFMUL_VF (VM_VFSUB_VF (vglg_m, 1.0, (vl)), 32.0, (vl)), (vl));	\
  {									\
    const double *vglg_t = &vm_gam_logtab[0][0];			\
    /* 24 bytes per entry: i*8 + i*16 */				\
    vm_u64_t vglg_iu = VM_REINT_I_U (vglg_i);				\
    vm_u64_t vglg_off = VM_VADD_VV (VM_VSLL_VX (vglg_iu, 3, (vl)),	\
				    VM_VSLL_VX (vglg_iu, 4, (vl)), (vl)); \
    vm_f_t vglg_invc = VM_VLUXEI64 (vglg_t, vglg_off, (vl));		\
    vm_f_t vglg_lh = VM_VLUXEI64 (vglg_t + 1, vglg_off, (vl));		\
    vm_f_t vglg_ll = VM_VLUXEI64 (vglg_t + 2, vglg_off, (vl));		\
    vm_f_t vglg_p, vglg_e, vglg_rh, vglg_rl;				\
    VM_TWO_PROD (vglg_p, vglg_e, vglg_m, vglg_invc, (vl));		\
    VM_TWO_SUM (vglg_rh, vglg_rl, vglg_p, VM_VFMV_V_F (-1.0, (vl)), (vl)); \
    vglg_rl = VM_VFADD_VV (vglg_rl, vglg_e, (vl));			\
    /* log1p(r) = r * sum_{j=0..10} (-1)^j r^j / (j+1) */		\
    vm_f_t vglg_sh = VM_VFMV_V_F (1.0 / 11.0, (vl));			\
    vm_f_t vglg_sl = VM_VFMV_V_F (0.0, (vl));				\
    for (int vglg_j = 9; vglg_j >= 0; vglg_j--)				\
      {									\
	vm_f_t vglg_a, vglg_b;						\
	VM_DD_MUL (vglg_a, vglg_b, vglg_sh, vglg_sl, vglg_rh, vglg_rl, (vl)); \
	VM_DD_ADD_D (vglg_sh, vglg_sl, vglg_a, vglg_b,			\
		     ((vglg_j & 1) ? -1.0 : 1.0) / (double) (vglg_j + 1), (vl)); \
      }									\
    VM_DD_MUL (vglg_sh, vglg_sl, vglg_sh, vglg_sl, vglg_rh, vglg_rl, (vl)); \
    /* log(x) = k*ln2 + logc + log1p(r) */				\
    {									\
      vm_f_t vglg_kd = VM_VFCVT_F_X_V (VM_REINT_U_I (vglg_k), (vl));			\
      vm_f_t vglg_th, vglg_tl;						\
      VM_DD_MUL_D (vglg_th, vglg_tl, vglg_kd,				\
		   VM_VFMV_V_F (0.0, (vl)), VM_GAM_LN2H, (vl));		\
      vglg_tl = VM_VFMACC_VF (vglg_tl, VM_GAM_LN2L, vglg_kd, (vl));	\
      VM_DD_ADD (vglg_th, vglg_tl, vglg_th, vglg_tl, vglg_lh,		\
		 VM_VFMV_V_F (0.0, (vl)), (vl));			\
      vglg_tl = VM_VFADD_VV (vglg_tl, vglg_ll, (vl));			\
      VM_DD_ADD (vglg_th, vglg_tl, vglg_th, vglg_tl, vglg_sh, vglg_sl, (vl)); \
      (yh) = VM_VFMERGE_VVM (vglg_th, VM_KFN (scalar_fixup) ((x),	\
	       vglg_th, vglg_mspec, (vl), (fixup)), vglg_mspec, (vl));	\
      (yl) = VM_VFMERGE_VFM (vglg_tl, 0.0, vglg_mspec, (vl));		\
    }									\
  }									\
} while (0)

/* ---- exp to ~2^-75, which VM_EXP_DD is not --------------------------
   VM_EXP_DD's `tmp` is evaluated in plain double (~2^-53); for tgamma the
   result's whole relative error IS exp's.  Here r = x-k*ln2 is a dd and
   exp(r) is a degree-14 dd polynomial on |r| <= ln2/2 (8.2e-24).  */
#define VM_GAM_EXP_DD(eh, el, x, vl, fixup) do {			\
  /* The guard is on the magnitude: a negative x's raw pattern is larger \
     than every finite one, so a test on it would send negatives to exp.  */ \
  vm_u64_t vgex_bx = VM_BITS (VM_VFABS_V ((x), (vl)));			\
  /* |x| >= 709 over/underflows or hits the scale's floor; the scalar is \
     authoritative there, flags included */ \
  vm_b_t vgex_mspec = VM_VMSGEU_VX (vgex_bx, VM_GAM_B709, (vl));	\
  vm_f_t vgex_xs = VM_VFMERGE_VFM ((x), 0.0, vgex_mspec, (vl));		\
  vm_i64_t vgex_k = VM_VFCVT_X_F_V_RM (					\
    VM_VFMUL_VF (vgex_xs, VM_GAM_LOG2EH, (vl)), __RISCV_FRM_RNE, (vl));	\
  vm_f_t vgex_ph, vgex_pl, vgex_rl;					\
  /* r = x - k*ln2 as a dd: ln2 is split, or the product rounds at 2^-53 of \
     k*ln2 */ \
  VM_TWO_SUM (vgex_ph, vgex_pl, vgex_xs,				\
	      VM_VFMUL_VF (VM_VFCVT_F_X_V (vgex_k, (vl)),		\
			   -VM_GAM_LN2H, (vl)), (vl));			\
  vgex_rl = VM_VFMUL_VF (VM_VFCVT_F_X_V (vgex_k, (vl)),			\
			 VM_GAM_LN2L, (vl));				\
  vgex_pl = VM_VFSUB_VV (vgex_pl, vgex_rl, (vl));			\
  VM_GAM_POLY (vgex_ph, vgex_pl, vm_gam_expP, 15, vgex_ph, vgex_pl, (vl)); \
  /* 2^k is exact, and so is the rescale unless the result is subnormal */ \
  {									\
    vm_f_t vgex_sc = VM_FROM_BITS (VM_VSLL_VX (VM_VADD_VX (		\
      VM_REINT_I_U (vgex_k), 1023, (vl)), 52, (vl)));			\
    VM_DD_MUL (vgex_ph, vgex_pl, vgex_ph, vgex_pl, vgex_sc,		\
	       VM_VFMV_V_F (0.0, (vl)), (vl));				\
  }									\
  (eh) = VM_VFMERGE_VVM (vgex_ph, VM_KFN (scalar_fixup) ((x), (eh),	\
	 vgex_mspec, (vl), (fixup)), vgex_mspec, (vl));			\
  (el) = VM_VFMERGE_VFM (vgex_pl, 0.0, vgex_mspec, (vl));		\
} while (0)

/* ---- range bounds, as magnitude bit patterns --------------------------
   Integer compares, never vmfgt/vmflt (signalling; INVALID on a quiet NaN);
   any NaN's magnitude bits exceed every finite pattern.  */
#define VM_GAM_BHALF	0x3fe0000000000000ULL	/* 0.5 */
#define VM_GAM_B15	0x3ff8000000000000ULL	/* 1.5 */
#define VM_GAM_B25	0x4004000000000000ULL	/* 2.5 */
#define VM_GAM_B10	0x4024000000000000ULL	/* 10.0 */
#define VM_GAM_B12	0x4028000000000000ULL	/* 12.0 */
#define VM_GAM_B2P100	0x4630000000000000ULL	/* 2^100 */
#define VM_GAM_B2P60	0x43b0000000000000ULL	/* 2^60 (a magnitude) */
#define VM_GAM_BSIGN	0x8000000000000000ULL	/* the sign bit */
#define VM_GAM_BINF	0x7ff0000000000000ULL	/* +Inf */
/* 709, not 710: at x = -709 the reduction gives k = -1023 and the scale
   (k+1023)<<52 is exactly zero.  */
#define VM_GAM_B1E305	0x7f423a516e82d9baULL	/* 1e305 */
/* Where region A stops trusting VM_LOG1P_DD for its log1p term: it is
   relatively exact as u -> 0, VM_GAM_LOG_DD only to ~1.5e-23 absolute.  */
#define VM_GAM_B2M8	0x3f70000000000000ULL	/* 2^-8 */
#define VM_GAM_B2M100	0x39b0000000000000ULL	/* 2^-100 */
#define VM_GAM_B2M64	0x3bf0000000000000ULL	/* 2^-64 */
#define VM_GAM_B2M500	0x20b0000000000000ULL	/* 2^-500 */
#define VM_GAM_B2M32	0x3df0000000000000ULL	/* 2^-32 */
#define VM_GAM_B709	0x4086240000000000ULL	/* 708.5 */

/* |v| <= thr for a non-negative threshold (the magnitude's bits order
   like the value).  */
#define VM_GAM_ABSLE_V(v, thr, vl)					\
  VM_VMSLEU_VX (VM_BITS (VM_VFABS_V ((v), (vl))), (thr), (vl))

/* ---- dd polynomial, low-to-high table (the shape of VM_ERF_POLY) ----- */
#define VM_GAM_POLY(rh, rl, tbl, n, th, tl, vl) do {			\
  vm_f_t vgp_h = VM_VFMV_V_F ((tbl)[(n) - 1][0], (vl));			\
  vm_f_t vgp_l = VM_VFMV_V_F ((tbl)[(n) - 1][1], (vl));			\
  for (int vgp_i = (n) - 2; vgp_i >= 0; vgp_i--)				\
    {									\
      vm_f_t vgp_h2, vgp_l2;						\
      VM_DD_MUL (vgp_h2, vgp_l2, vgp_h, vgp_l, (th), (tl), (vl));	\
      VM_DD_ADD_D (vgp_h, vgp_l, vgp_h2, vgp_l2, (tbl)[vgp_i][0], (vl)); \
      vgp_l = VM_VFADD_VF (vgp_l, (tbl)[vgp_i][1], (vl));		\
    }									\
  (rh) = vgp_h;								\
  (rl) = vgp_l;								\
} while (0)

/* ---- sin and cos of a dd argument with |y| <= pi/4, as dd -------------
   sin(y) = y*(1 + t*PS(t)), cos(y) = 1 + t*PC(t), t = y^2, 11 terms each.  */
#define VM_GAM_SINCOS_DD(sh, sl, ch, cl, yh, yl, vl) do {		\
  vm_f_t vgs_tH, vgs_tL, vgs_pH, vgs_pL, vgs_uH, vgs_uL, vgs_vH, vgs_vL; \
  VM_DD_MUL (vgs_tH, vgs_tL, (yh), (yl), (yh), (yl), (vl));		\
  /* |y| < 2^-32: y^2 is below 2^-64 and the series is y itself; the Horner \
     would underflow its own low parts there.  */ \
  {									\
    vm_b_t vgs_mt = VM_VMSLTU_VX (					\
      VM_BITS (VM_VFABS_V ((yh), (vl))), VM_GAM_B2M32, (vl));		\
    vgs_tH = VM_VFMERGE_VFM (vgs_tH, 0.0, vgs_mt, (vl));		\
    vgs_tL = VM_VFMERGE_VFM (vgs_tL, 0.0, vgs_mt, (vl));		\
  }									\
  VM_GAM_POLY (vgs_pH, vgs_pL, vm_gam_PS, 11, vgs_tH, vgs_tL, (vl));	\
  VM_DD_MUL (vgs_uH, vgs_uL, vgs_tH, vgs_tL, vgs_pH, vgs_pL, (vl));	\
  VM_DD_MUL (vgs_vH, vgs_vL, vgs_uH, vgs_uL, (yh), (yl), (vl));		\
  VM_DD_ADD ((sh), (sl), (yh), (yl), vgs_vH, vgs_vL, (vl));		\
  VM_GAM_POLY (vgs_pH, vgs_pL, vm_gam_PC, 11, vgs_tH, vgs_tL, (vl));	\
  VM_DD_MUL (vgs_uH, vgs_uL, vgs_tH, vgs_tL, vgs_pH, vgs_pL, (vl));	\
  VM_DD_ADD_D ((ch), (cl), vgs_uH, vgs_uL, 1.0, (vl));			\
} while (0)

/* ---- Stirling: lgamma(z) for z >= 10, as a dd ------------------------ */
#define VM_GAM_STIRLING(h, l, zh, zl, vl) do {				\
  /* The series' 1/z is clamped: at z = 1e308 it is subnormal, and forming it \
     raises UNDERFLOW on a lane whose answer is an overflowing +Inf.  */ \
  vm_b_t vgt_mbig = VM_VMSGTU_VX (VM_BITS (zh), VM_GAM_B2P100, (vl));	\
  vm_f_t vgt_zs = VM_VFMERGE_VFM ((zh), 0x1p100, vgt_mbig, (vl));	\
  vm_f_t vgt_zl = VM_VFMERGE_VFM ((zl), 0.0, vgt_mbig, (vl));		\
  vm_f_t vgt_lH, vgt_lL, vgt_mH, vgt_mL, vgt_pH, vgt_pL;		\
  /* log(z) as a dd: VM_LOG_DD takes a single double, and dropping zl costs \
     (z-1/2)*2^-53 absolute.  log(zh+zl) = log(zh) + zl/zh.  */ \
  VM_GAM_LOG_DD (vgt_lH, vgt_lL, (zh), (vl), log);				\
  vgt_lL = VM_VFADD_VV (vgt_lL, VM_VFDIV_VV ((zl), (zh), (vl)), (vl));	\
  VM_DD_ADD_D (vgt_mH, vgt_mL, (zh), (zl), -0.5, (vl));			\
  VM_DD_MUL (vgt_pH, vgt_pL, vgt_mH, vgt_mL, vgt_lH, vgt_lL, (vl));	\
  VM_DD_SUB (vgt_pH, vgt_pL, vgt_pH, vgt_pL, (zh), (zl), (vl));		\
  VM_DD_ADD_D (vgt_pH, vgt_pL, vgt_pH, vgt_pL, VM_GAM_HLOG2PIH, (vl));	\
  vgt_pL = VM_VFADD_VF (vgt_pL, VM_GAM_HLOG2PIL, (vl));			\
  {									\
    vm_f_t vgt_iH, vgt_iL, vgt_qH, vgt_qL, vgt_wH, vgt_wL, vgt_tH, vgt_tL; \
    VM_DD_DIV (vgt_iH, vgt_iL, VM_VFMV_V_F (1.0, (vl)),			\
	       VM_VFMV_V_F (0.0, (vl)), vgt_zs, vgt_zl, (vl));		\
    VM_DD_MUL (vgt_qH, vgt_qL, vgt_iH, vgt_iL, vgt_iH, vgt_iL, (vl));	\
    VM_GAM_POLY (vgt_wH, vgt_wL, vm_gam_S, 12, vgt_qH, vgt_qL, (vl));	\
    VM_DD_MUL (vgt_tH, vgt_tL, vgt_wH, vgt_wL, vgt_iH, vgt_iL, (vl));	\
    VM_DD_ADD ((h), (l), vgt_pH, vgt_pL, vgt_tH, vgt_tL, (vl));		\
  }									\
} while (0)

/* ---- lgamma(x) for x > 0, as a dd ------------------------------------
   The argument must be finite and > 0 on every lane; the callers sanitize.
   +0 is allowed through -- it is a pole, and -log(0) gives its +Inf.  */
#define VM_GAM_LGAM_POS(h, l, yh, yl, vl) do {				\
  vm_f_t vgl_abH, vgl_abL, vgl_bH, vgl_bL, vgl_cH, vgl_cL, vgl_dH, vgl_dL; \
  /* The low part is dropped when below 2^-100: only ever the rounding \
     residual of a shifted argument, but on the reflection's 1-x it can be \
     5e-324, and dividing that underflows on a lane whose answer is 744.  */ \
  vm_f_t vgl_yl = VM_VFMERGE_VFM ((yl), 0.0,			\
				  VM_VMSLTU_VX (VM_BITS (VM_VFABS_V ((yl), (vl))), \
						VM_GAM_B2M100, (vl)), (vl)); \
  vm_f_t vgl_h0, vgl_l0;						\
  /* y is a dd because the reflection hands in 1-x, which is not always \
     representable; every region builds y+k with a two_sum.  */ \
  /* --- regions A and B: u = x - (x < 0.5) --- */			\
  {									\
    vm_b_t vgl_sm = VM_VMSLTU_VX (VM_BITS (yh), VM_GAM_BHALF, (vl));	\
    /* y-1 is exact by Sterbenz for y in [1,2], which is where it is used */ \
    vm_f_t vgl_u = VM_VFMERGE_VVM (VM_VFSUB_VF ((yh), 1.0, (vl)),	\
				   (yh), vgl_sm, (vl));			\
    vm_f_t vgl_ul = vgl_yl;						\
    /* the Horner only needs |u| <= 0.5; a lane from another region would \
       overflow a degree-24 polynomial.  */ \
    vm_f_t vgl_uc = VM_VFMERGE_VFM (vgl_u, 0.0,				\
	VM_VMNOT_M (VM_GAM_ABSLE_V (vgl_u, VM_GAM_BHALF, (vl)), (vl)), (vl)); \
    /* AND zero it on the x < 0.5 half when |u| is tiny: there u = x, the \
       answer is -log(x) >= 44 and the other terms underflow.  The other half \
       has u = x-1, where the answer IS -gamma*u, and cannot be.  */ \
    {									\
      vm_b_t vgl_mz = VM_VMAND_MM (vgl_sm,				\
	VM_VMSLTU_VX (VM_BITS (VM_VFABS_V (vgl_u, (vl))),		\
		      VM_GAM_B2M64, (vl)), (vl));			\
      vgl_uc = VM_VFMERGE_VFM (vgl_uc, 0.0, vgl_mz, (vl));		\
      vgl_ul = VM_VFMERGE_VFM (vgl_ul, 0.0, vgl_mz, (vl));		\
    }									\
    vgl_ul = VM_VFMERGE_VFM (vgl_ul, 0.0,				\
	VM_VMNOT_M (VM_GAM_ABSLE_V (vgl_u, VM_GAM_BHALF, (vl)), (vl)), (vl)); \
    vm_f_t vgl_l1H, vgl_l1L, vgl_lxH, vgl_lxL, vgl_u2H, vgl_u2L;	\
    vm_f_t vgl_uz, vgl_ulz;						\
    vm_f_t vgl_gH, vgl_gL, vgl_pH, vgl_pL, vgl_tH, vgl_tL;		\
    VM_LOG1P_DD (vgl_l1H, vgl_l1L, vgl_uc, (vl), log1p);		\
    vgl_l1L = VM_VFADD_VV (vgl_l1L, VM_VFDIV_VV (vgl_ul,		\
	VM_VFADD_VF (vgl_uc, 1.0, (vl)), (vl)), (vl));			\
    /* u^2 underflows for |u| below 2^-500 and is negligible there: feed zero \
       rather than raise UNDERFLOW on a lane whose answer is 700.  */ \
    {									\
      vm_b_t vgl_mv = VM_VMSLTU_VX (					\
	VM_BITS (VM_VFABS_V (vgl_uc, (vl))), VM_GAM_B2M500, (vl));	\
      vgl_uz = VM_VFMERGE_VFM (vgl_uc, 0.0, vgl_mv, (vl));		\
      vgl_ulz = VM_VFMERGE_VFM (vgl_ul, 0.0, vgl_mv, (vl));		\
      VM_DD_MUL (vgl_u2H, vgl_u2L, vgl_uz, vgl_ulz, vgl_uz, vgl_ulz, (vl)); \
    }									\
    /* G takes u itself -- u^2 is only the multiplier in front of it */	\
    /* |u| < 2^-32: u^2*G(u) is below 2^-64, and its Horner would underflow \
       on the coefficients' low parts.  Feed it zero.  */ \
    {									\
      vm_b_t vgl_mt = VM_VMSLTU_VX (					\
	VM_BITS (VM_VFABS_V (vgl_uc, (vl))), VM_GAM_B2M32, (vl));	\
      vm_f_t vgl_gc = VM_VFMERGE_VFM (vgl_uc, 0.0, vgl_mt, (vl));	\
      VM_GAM_POLY (vgl_gH, vgl_gL, vm_gam_G, 25, vgl_gc,		\
		   VM_VFMV_V_F (0.0, (vl)), (vl));			\
    }									\
    VM_DD_MUL (vgl_pH, vgl_pL, vgl_u2H, vgl_u2L, vgl_gH, vgl_gL, (vl));	\
    /* with the CLAMPED low part: on a negative subnormal x it is 5e-324, and \
       C1 times that raises UNDERFLOW on a lane whose answer is 744.  */ \
    VM_DD_MUL_D (vgl_tH, vgl_tL, vgl_uc, vgl_ul, VM_GAM_C1H, (vl));	\
    vgl_tL = VM_VFMACC_VF (vgl_tL, VM_GAM_C1L, vgl_uc, (vl));		\
    VM_DD_ADD (vgl_pH, vgl_pL, vgl_pH, vgl_pL, vgl_tH, vgl_tL, (vl));	\
    /* On the x >= 0.5 half u = x-1, so log1p(u) IS log(yh) and has to come \
       from VM_GAM_LOG_DD, not VM_LOG1P_DD.  */ \
    VM_GAM_LOG_DD (vgl_lxH, vgl_lxL, (yh), (vl), log);			\
    vgl_lxL = VM_VFADD_VV (vgl_lxL, VM_VFDIV_VV (vgl_yl, (yh), (vl)), (vl)); \
    {									\
      vm_b_t vgl_mlx = VM_VMAND_MM (VM_VMNOT_M (vgl_sm, (vl)),		\
	VM_VMSGEU_VX (VM_BITS (VM_VFABS_V (vgl_uc, (vl))),		\
		      VM_GAM_B2M8, (vl)), (vl));			\
      vgl_l1H = VM_VFMERGE_VVM (vgl_l1H, vgl_lxH, vgl_mlx, (vl));	\
      vgl_l1L = VM_VFMERGE_VVM (vgl_l1L, vgl_lxL, vgl_mlx, (vl));	\
    }									\
    VM_DD_SUB (vgl_pH, vgl_pL, vgl_pH, vgl_pL, vgl_l1H, vgl_l1L, (vl));	\
    /* Only the x < 0.5 half carries a further -log(x): the other half is \
       logGamma(1+u) itself.  */ \
    VM_DD_SUB (vgl_tH, vgl_tL, vgl_pH, vgl_pL, vgl_lxH, vgl_lxL, (vl));	\
    vgl_abH = VM_VFMERGE_VVM (vgl_pH, vgl_tH, vgl_sm, (vl));		\
    vgl_abL = VM_VFMERGE_VVM (vgl_pL, vgl_tL, vgl_sm, (vl));		\
    /* --- region B: w = x-2 --- */					\
    {									\
      vm_f_t vgl_w = VM_VFSUB_VF ((yh), 2.0, (vl));			\
      vm_f_t vgl_wl = vgl_yl;						\
      vm_f_t vgl_wc = VM_VFMERGE_VFM (vgl_w, 0.0,			\
	  VM_VMNOT_M (VM_GAM_ABSLE_V (vgl_w, VM_GAM_BHALF, (vl)), (vl)), (vl)); \
      vgl_wl = VM_VFMERGE_VFM (vgl_wl, 0.0,				\
	  VM_VMNOT_M (VM_GAM_ABSLE_V (vgl_w, VM_GAM_BHALF, (vl)), (vl)), (vl)); \
      vm_f_t vgl_w2H, vgl_w2L, vgl_hH, vgl_hL, vgl_uH, vgl_uL;		\
      VM_DD_MUL (vgl_w2H, vgl_w2L, vgl_wc, vgl_wl, vgl_wc, vgl_wl, (vl)); \
      {									\
	vm_b_t vgl_mt = VM_VMSLTU_VX (					\
	  VM_BITS (VM_VFABS_V (vgl_wc, (vl))), VM_GAM_B2M32, (vl));	\
	vm_f_t vgl_hc = VM_VFMERGE_VFM (vgl_wc, 0.0, vgl_mt, (vl));	\
	VM_GAM_POLY (vgl_hH, vgl_hL, vm_gam_G, 25, vgl_hc,		\
		     VM_VFMV_V_F (0.0, (vl)), (vl));			\
      }									\
      VM_DD_MUL (vgl_uH, vgl_uL, vgl_w2H, vgl_w2L, vgl_hH, vgl_hL, (vl)); \
      VM_DD_MUL_D (vgl_tH, vgl_tL, vgl_wc, vgl_wl, VM_GAM_C1H, (vl));	\
      vgl_tL = VM_VFMACC_VF (vgl_tL, VM_GAM_C1L, vgl_wc, (vl));		\
      VM_DD_ADD (vgl_bH, vgl_bL, vgl_uH, vgl_uL, vgl_tH, vgl_tL, (vl));	\
    }									\
  }									\
  /* --- region C: 2.5 < x < 10, via x+8 --- */				\
  {									\
    /* clamped from below as well: this region runs on every lane, and a tiny \
       x would underflow the product and overflow Stirling's series.  */ \
    vm_f_t vgl_xc = VM_VFMERGE_VFM ((yh), 12.0,				\
	VM_VMSGTU_VX (VM_BITS (yh), VM_GAM_B12, (vl)), (vl));		\
    vgl_xc = VM_VFMERGE_VFM (vgl_xc, 3.0,				\
	VM_VMSLTU_VX (VM_BITS (yh), VM_GAM_B25, (vl)), (vl));		\
    vm_f_t vgl_pH = VM_VFMV_V_F (1.0, (vl));				\
    vm_f_t vgl_pL = VM_VFMV_V_F (0.0, (vl));				\
    /* the factors have to be dd: x+k rounds for fractional x, and eight of \
       those add up to ~1e-16 relative in P.  */ \
    for (int vgl_k = 0; vgl_k < 8; vgl_k++)				\
      {									\
	vm_f_t vgl_fh, vgl_fl;						\
	VM_TWO_SUM (vgl_fh, vgl_fl, vgl_xc,				\
		    VM_VFMV_V_F ((double) vgl_k, (vl)), (vl));		\
	vgl_fl = VM_VFADD_VV (vgl_fl, vgl_yl, (vl));			\
	VM_DD_MUL (vgl_pH, vgl_pL, vgl_pH, vgl_pL, vgl_fh, vgl_fl, (vl)); \
      }									\
    {									\
      vm_f_t vgl_zH, vgl_zL, vgl_sH, vgl_sL, vgl_gH, vgl_gL;		\
      VM_TWO_SUM (vgl_zH, vgl_zL, vgl_xc, VM_VFMV_V_F (8.0, (vl)), (vl)); \
      vgl_zL = VM_VFADD_VV (vgl_zL, vgl_yl, (vl));			\
      VM_GAM_STIRLING (vgl_sH, vgl_sL, vgl_zH, vgl_zL, (vl));		\
      /* P is a dd; log(P) needs its low part too, or it loses 8*2^-53 \
	 relative.  */ \
      VM_GAM_LOG_DD (vgl_gH, vgl_gL, vgl_pH, (vl), log);			\
      vgl_gL = VM_VFADD_VV (vgl_gL,					\
			    VM_VFDIV_VV (vgl_pL, vgl_pH, (vl)), (vl));	\
      VM_DD_SUB (vgl_cH, vgl_cL, vgl_sH, vgl_sL, vgl_gH, vgl_gL, (vl));	\
    }									\
  }									\
  /* --- region D: x >= 10 --- */					\
  /* x >= 10 only, clamped from below for the same reason */		\
  VM_GAM_STIRLING (vgl_dH, vgl_dL,					\
		   VM_VFMERGE_VFM ((yh), 20.0,				\
		      VM_VMSLTU_VX (VM_BITS (yh), VM_GAM_B10, (vl)), (vl)), \
		   VM_VFMERGE_VFM (vgl_yl, 0.0,				\
		      VM_VMSLTU_VX (VM_BITS (yh), VM_GAM_B10, (vl)), (vl)), (vl)); \
  /* --- merge: <=1.5, <=2.5, <10, else --- */				\
  {									\
    vm_f_t vgl_r1H, vgl_r1L, vgl_r2H, vgl_r2L;				\
    vm_b_t vgl_m10 = VM_VMSLTU_VX (VM_BITS (yh), VM_GAM_B10, (vl));	\
    vm_b_t vgl_m25 = VM_VMSLEU_VX (VM_BITS (yh), VM_GAM_B25, (vl));	\
    vm_b_t vgl_m15 = VM_VMSLEU_VX (VM_BITS (yh), VM_GAM_B15, (vl));	\
    vgl_r1H = VM_VFMERGE_VVM (vgl_dH, vgl_cH, vgl_m10, (vl));		\
    vgl_r1L = VM_VFMERGE_VVM (vgl_dL, vgl_cL, vgl_m10, (vl));		\
    vgl_r2H = VM_VFMERGE_VVM (vgl_r1H, vgl_bH, vgl_m25, (vl));		\
    vgl_r2L = VM_VFMERGE_VVM (vgl_r1L, vgl_bL, vgl_m25, (vl));		\
    (h) = VM_VFMERGE_VVM (vgl_r2H, vgl_abH, vgl_m15, (vl));		\
    (l) = VM_VFMERGE_VVM (vgl_r2L, vgl_abL, vgl_m15, (vl));		\
  }									\
} while (0)

/* ---- lgamma, with the sign of Gamma(x) on the way out ----------------
   mneg is set exactly on the lanes where Gamma(x) < 0.  */
#define VM_GAM_LGAM(h, l, mneg, x, vl) do {				\
  /* vgm_bx is the magnitude's bits: a test on the raw pattern would call \
     every negative lane a NaN, so the sign is its own test.  */ \
  vm_u64_t vgm_bx = VM_BITS (VM_VFABS_V ((x), (vl)));			\
  vm_b_t vgm_mbit = VM_VMSGEU_VX (VM_BITS (x), VM_GAM_BSIGN, (vl));	\
  vm_b_t vgm_mnan = VM_VMSGTU_VX (vgm_bx, VM_GAM_BINF, (vl));		\
  vm_b_t vgm_minf = VM_VMSEQ_VX (vgm_bx, VM_GAM_BINF, (vl));		\
  /* The positive chain runs on every lane, its argument 1.0 wherever the \
     answer comes from the reflection, +-Inf or a NaN.  +0 is left alone.  */ \
  vm_b_t vgm_mbad = VM_VMOR_MM (vgm_mbit,				\
	VM_VMOR_MM (vgm_minf, vgm_mnan, (vl)), (vl));			\
  /* Poles and overflow go to the scalar: an infinite value in the \
     double-double makes two_sum(Inf, -Inf) NaN.  */ \
  vm_b_t vgm_mfix;							\
  {									\
    vm_f_t vgm_r0, vgm_y0, vgm_y1;					\
    vm_i64_t vgm_q0;							\
    vm_b_t vgm_mi, vgm_mp;						\
    VM_PI_RED (vgm_y0, vgm_y1, vgm_r0, vgm_q0, VM_PI_ALL ((vl)), (x), (vl)); \
    VM_PI_EXACT_MASKS (vgm_mi, vgm_mp, vgm_r0,			\
		       VM_VAND_VX (VM_REINT_I_U (vgm_q0), 3, (vl)), (vl)); \
    vgm_mfix = VM_VMOR_MM (						\
      VM_VMAND_MM (vgm_mi,						\
		   VM_VMSGEU_VX (VM_BITS (x), VM_GAM_BSIGN, (vl)), (vl)), \
      VM_VMOR_MM (VM_VMFEQ_VF ((x), 0.0, (vl)),			\
		  VM_VMSGEU_VX (vgm_bx, VM_GAM_B1E305, (vl)), (vl)), (vl)); \
  }									\
  vgm_mbad = VM_VMOR_MM (vgm_mbad, vgm_mfix, (vl));			\
  vm_f_t vgm_xp = VM_VFMERGE_VFM ((x), 1.0, vgm_mbad, (vl));		\
  vm_f_t vgm_ah, vgm_al, vgm_ph, vgm_pl;				\
  vm_b_t vgm_mneg;							\
  VM_GAM_LGAM_POS (vgm_ah, vgm_al, vgm_xp, VM_VFMV_V_F (0.0, (vl)), (vl));			\
  /* reflection */							\
  {									\
    /* |x| >= 2^60 is an integer, hence a pole, and 1-x would overflow \
       Stirling into +Inf - Inf = NaN.  */ \
    /* Both get -1.5, not -3, so the sanitized value is not itself a pole \
       (log|sin| = -Inf would raise INVALID in two_sum).  */ \
    vm_f_t vgm_xr = VM_VFMERGE_VFM ((x), -1.5,				\
	VM_VMOR_MM (VM_VMSGEU_VX (vgm_bx, VM_GAM_B2P60, (vl)),		\
		    VM_VMOR_MM (vgm_mfix,					\
			       VM_VMOR_MM (vgm_minf, vgm_mnan, (vl)), (vl)), (vl)), (vl)); \
    vm_f_t vgm_y0, vgm_y1, vgm_r;					\
    vm_i64_t vgm_q64;							\
    /* |x| < 2^-32 takes the log(pi)+log|x| branch below, so the reduction's \
       underflow carrier is not needed; leaving it on raises UNDERFLOW.  */ \
    vm_b_t vgm_mt = VM_VMSLTU_VX (					\
      VM_BITS (VM_VFABS_V ((x), (vl))), VM_GAM_B2M32, (vl));		\
    VM_PI_RED (vgm_y0, vgm_y1, vgm_r, vgm_q64,			\
	       VM_VMNOT_M (vgm_mt, (vl)), vgm_xr, (vl));			\
    vm_u64_t vgm_q = VM_VAND_VX (VM_REINT_I_U (vgm_q64), 3, (vl));	\
    vm_f_t vgm_sH, vgm_sL, vgm_cH, vgm_cL, vgm_mH, vgm_mL;		\
    VM_GAM_SINCOS_DD (vgm_sH, vgm_sL, vgm_cH, vgm_cL, vgm_y0, vgm_y1, (vl)); \
    {									\
      vm_b_t vgm_modd;							\
      vm_f_t vgm_lgH, vgm_lgL, vgm_rH, vgm_rL;				\
      vgm_modd = VM_VMSNE_VX (VM_VAND_VX (vgm_q, 1, (vl)), 0, (vl));	\
      vgm_mH = VM_VFMERGE_VVM (vgm_sH, vgm_cH, vgm_modd, (vl));		\
      vgm_mL = VM_VFMERGE_VVM (vgm_sL, vgm_cL, vgm_modd, (vl));		\
      {									\
	/* |sin(pi x)| */						\
	vm_b_t vgm_msm = VM_VMSGEU_VX (VM_BITS (vgm_mH), VM_GAM_BSIGN, (vl)); \
	vgm_mH = VM_VFMERGE_VVM (vgm_mH, VM_VFNEG_V (vgm_mH, (vl)),	\
				 vgm_msm, (vl));			\
	vgm_mL = VM_VFMERGE_VVM (vgm_mL, VM_VFNEG_V (vgm_mL, (vl)),	\
				 vgm_msm, (vl));			\
      }									\
      /* log of the dd magnitude, not just its hi part: dropping mL costs \
	 2^-53 relative, 1.1e-16 absolute in log|sin| and hence in lgamma.  */ \
      VM_GAM_LOG_DD (vgm_lgH, vgm_lgL, vgm_mH, (vl), log);			\
      /* |sin| is 0 at an integer, where the division would be 0/0 and raise \
	 INVALID; those lanes are poles whose value the scalar replaces.  */ \
      {									\
	vm_f_t vgm_dn = VM_VFMERGE_VFM (vgm_mH, 1.0,			\
			    VM_VMFEQ_VF (vgm_mH, 0.0, (vl)), (vl));	\
	vgm_lgL = VM_VFADD_VV (vgm_lgL,				\
			       VM_VFDIV_VV (vgm_mL, vgm_dn, (vl)), (vl)); \
      }									\
      /* |x| < 2^-32: sin(pi x) IS pi x to 1e-19, and forming pi*x would be \
	 subnormal for a subnormal x.  log(pi) + log|x| is exact.  */ \
      {									\
	/* 1.0 on the non-finite lanes: log() of an infinity is still evaluated \
	   here, and log(-Inf) is NaN with INVALID.  */ \
	vm_f_t vgm_ax = VM_VFMERGE_VFM (VM_VFABS_V ((x), (vl)), 1.0,	\
					VM_VMOR_MM (vgm_minf, vgm_mnan, (vl)), (vl)); \
	vm_f_t vgm_tH, vgm_tL;						\
	VM_GAM_LOG_DD (vgm_tH, vgm_tL, vgm_ax, (vl), log);		\
	VM_DD_ADD_D (vgm_tH, vgm_tL, vgm_tH, vgm_tL,			\
		     VM_GAM_LOGPIH, (vl));				\
	vgm_tL = VM_VFADD_VF (vgm_tL, VM_GAM_LOG_PIL, (vl));		\
	vgm_lgH = VM_VFMERGE_VVM (vgm_lgH, vgm_tH, vgm_mt, (vl));	\
	vgm_lgL = VM_VFMERGE_VVM (vgm_lgL, vgm_tL, vgm_mt, (vl));	\
      }									\
      /* lgamma(1-x), the argument always > 1 */			\
      /* 1-x as a dd: the addition rounds for some x, and that rounding lands \
	 straight in the result through psi(1-x).  */ \
      {									\
	vm_f_t vgm_omH, vgm_omL;					\
	/* -xr, not -x: two_sum(-Inf, 1) leaves Inf - Inf in its residual and \
	   raises INVALID on a lane whose answer is +Inf with no flags.  */ \
	VM_TWO_SUM (vgm_omH, vgm_omL, VM_VFNEG_V (vgm_xr, (vl)),		\
		    VM_VFMV_V_F (1.0, (vl)), (vl));			\
	/* +-Inf makes 1-x infinite, and the positive chain would take it \
	   through log(-Inf): NaN and INVALID on a lane that owes none.  */ \
	/* For x >= 0 the reflection is not the answer and 1-x is negative, \
	   so the positive chain would take log of a negative there.  */ \
	{								\
	  vm_b_t vgm_modd = VM_VMOR_MM (VM_VMNOT_M (vgm_mbit, (vl)),	\
			     VM_VMOR_MM (vgm_mfix, vgm_mnan, (vl)), (vl)); \
	  vgm_omH = VM_VFMERGE_VFM (vgm_omH, 2.0, vgm_modd, (vl));	\
	  vgm_omL = VM_VFMERGE_VFM (vgm_omL, 0.0, vgm_modd, (vl));	\
	}								\
	VM_GAM_LGAM_POS (vgm_rH, vgm_rL, vgm_omH, vgm_omL, (vl));	\
      }									\
      /* log(pi) - log|sin| - lgamma(1-x) */				\
      VM_DD_SUB (vgm_ph, vgm_pl, VM_VFMV_V_F (VM_GAM_LOGPIH, (vl)),	\
		 VM_VFMV_V_F (VM_GAM_LOG_PIL, (vl)), vgm_lgH, vgm_lgL, (vl)); \
      VM_DD_SUB (vgm_ph, vgm_pl, vgm_ph, vgm_pl, vgm_rH, vgm_rL, (vl));	\
      /* the sign of Gamma(x) = the sign of sin(pi x) */		\
      {									\
	vm_b_t vgm_mq0 = VM_VMSEQ_VX (vgm_q, 0, (vl));			\
	vm_b_t vgm_mq2 = VM_VMSEQ_VX (vgm_q, 2, (vl));			\
	vm_b_t vgm_mq3 = VM_VMSEQ_VX (vgm_q, 3, (vl));			\
	vm_b_t vgm_rsn = VM_VMSGEU_VX (VM_BITS (vgm_r), VM_GAM_BSIGN, (vl)); \
	vgm_mneg = VM_VMOR_MM (				\
	  VM_VMAND_MM (vgm_mq0, vgm_rsn, (vl)),				\
	  VM_VMOR_MM (VM_VMAND_MM (vgm_mq2, VM_VMNOT_M (vgm_rsn, (vl)), (vl)), \
		      vgm_mq3, (vl)), (vl));				\
      }									\
    }									\
  }									\
  /* the reflection result applies to every negative lane, poles included \
     (where log|sin| = -Inf is exactly the +Inf that belongs there) */	\
  (h) = VM_VFMERGE_VVM (vgm_ah, vgm_ph, vgm_mbit, (vl));		\
  (l) = VM_VFMERGE_VVM (vgm_al, vgm_pl, vgm_mbit, (vl));		\
  /* +-Inf: lgamma is +Inf and the sign is +1 */ \
  (h) = VM_VFMERGE_VVM ((h), VM_VINF (vl), vgm_minf, (vl));		\
  (l) = VM_VFMERGE_VFM ((l), 0.0, vgm_minf, (vl));			\
  /* Gamma > 0 for every x > 0, so the reflection's sign applies only on \
     negative lanes, never at an Inf, NaN or pole.  -0 is the exception.  */ \
  (mneg) = VM_VMAND_MM (vgm_mneg,					\
	VM_VMAND_MM (vgm_mbit,						\
		     VM_VMNOT_M (VM_VMOR_MM (vgm_mfix,			\
			VM_VMOR_MM (vgm_minf, vgm_mnan, (vl)), (vl)),	\
				(vl)), (vl)), (vl)), (vl);		\
  (mneg) = VM_VMOR_MM ((mneg), VM_VMAND_MM (vgm_mbit,		\
					    VM_VMFEQ_VF ((x), 0.0, (vl)), (vl)), (vl)); \
  /* NaN propagates, and an sNaN raises through the masked add */ \
  (h) = VM_VFMERGE_VVM ((h), VM_VFADD_VV_M (vgm_mnan, (x), (x), (vl)),	\
			vgm_mnan, (vl));				\
  (l) = VM_VFMERGE_VFM ((l), 0.0, vgm_mnan, (vl));			\
  /* the poles and the overflow: the scalar owns the value and the flags */ \
  (h) = VM_KFN (scalar_fixup) ((x), (h), vgm_mfix, (vl), lgamma);	\
  (l) = VM_VFMERGE_VFM ((l), 0.0, vgm_mfix, (vl));			\
} while (0)

/* ---- the three entry kernels ---------------------------------------- */

/* lgamma(x): the value alone.  */
#define VM_GAM_LGAMMA(h, l, x, vl) do {					\
  vm_b_t vgz_m;								\
  VM_GAM_LGAM (h, l, vgz_m, (x), (vl));					\
} while (0)

/* lgamma_r(x, &sign): the value and sign(Gamma(x)) as +-1.  */
#define VM_GAM_LGAMMA_R(h, l, sgn, x, vl) do {				\
  vm_b_t vgz_m;								\
  vm_f_t vgz_s;								\
  VM_GAM_LGAM (h, l, vgz_m, (x), (vl));					\
  vgz_s = VM_VFMERGE_VFM (VM_VFMV_V_F (1.0, (vl)), -1.0, vgz_m, (vl));	\
  (sgn) = vgz_s;							\
} while (0)

/* tgamma(x) = sign(Gamma(x)) * exp(lgamma(x)).
   The poles need care lgamma does not: C99 wants tgamma(negative integer) =
   NaN with INVALID, where exp(lgamma) would give +Inf with DIVBYZERO.  Those
   lanes get a benign argument and a patched masked 0*Inf.  */
#define VM_GAM_TGAMMA(h, l, x, vl) do {					\
  vm_f_t vgt_y0, vgt_y1, vgt_rr, vgt_xs;				\
  vm_i64_t vgt_q64;							\
  vm_b_t vgt_mint, vgt_mpole;						\
  VM_PI_RED (vgt_y0, vgt_y1, vgt_rr, vgt_q64, VM_PI_ALL ((vl)), (x), (vl)); \
  VM_PI_EXACT_MASKS (vgt_mint, vgt_mpole, vgt_rr,			\
		     VM_VAND_VX (VM_REINT_I_U (vgt_q64), 3, (vl)), (vl)); \
  /* -0 is an integer by every bit test here, but its tgamma(-0) = -Inf comes \
     out of the ordinary path.  Only |x| >= 1 is a C99 domain error.  */ \
  /* also finite: PI_RED sanitizes a non-finite argument to 0, so its \
     quadrant test would call every Inf and NaN an integer.  */ \
  vgt_mpole = VM_VMAND_MM (vgt_mint,					\
	VM_VMAND_MM (VM_VMNOT_M (VM_VMSGTU_VX (					\
				     VM_BITS (VM_VFABS_V ((x), (vl))),	\
				     VM_GAM_BINF, (vl)), (vl)),		\
		      VM_VMAND_MM (VM_VMSGEU_VX (VM_BITS (x), VM_GAM_BSIGN, (vl)), \
		       VM_VMNOT_M (VM_VMFEQ_VF ((x), 0.0, (vl)), (vl)), (vl)), (vl)), (vl)); \
  vgt_xs = VM_VFMERGE_VFM ((x), -1.5, vgt_mpole, (vl));			\
  {									\
    vm_b_t vgt_mneg;							\
    vm_f_t vgt_ph, vgt_pl, vgt_eH, vgt_eL, vgt_cH, vgt_cL, vgt_np;	\
    VM_GAM_LGAM (vgt_ph, vgt_pl, vgt_mneg, vgt_xs, (vl));		\
    VM_GAM_EXP_DD (vgt_eH, vgt_eL, vgt_ph, (vl), exp);			\
    /* exp(lgh + lgl) = exp(lgh)*exp(lgl), and lgl is 2^-53 of lgh */	\
    VM_TWO_SUM (vgt_cH, vgt_cL, VM_VFMV_V_F (1.0, (vl)), vgt_pl, (vl));	\
    /* An infinite or NaN exp must not go through the dd multiply: its \
       residual is Inf - Inf, so the collapse would hand back NaN.  */ \
    {									\
      vm_b_t vgt_mnf = VM_VMSGEU_VX (VM_BITS (vgt_eH), VM_GAM_BINF, (vl)); \
      vm_f_t vgt_dH, vgt_dL, vgt_mH, vgt_mL;				\
      /* the multiply must not even run on those lanes: dd_mul of an infinity \
	 raises UNDERFLOW and INVALID on a lane whose answer is +Inf.  */ \
      vgt_mH = VM_VFMERGE_VFM (vgt_eH, 1.0, vgt_mnf, (vl));		\
      vgt_mL = VM_VFMERGE_VFM (vgt_eL, 0.0, vgt_mnf, (vl));		\
      VM_DD_MUL (vgt_dH, vgt_dL, vgt_mH, vgt_mL, vgt_cH, vgt_cL, (vl));	\
      vgt_eH = VM_VFMERGE_VVM (vgt_dH, vgt_eH, vgt_mnf, (vl));		\
      vgt_eL = VM_VFMERGE_VVM (vgt_dL, vgt_eL, vgt_mnf, (vl));		\
    }									\
    (h) = VM_VFMERGE_VVM (vgt_eH, VM_VFNEG_V (vgt_eH, (vl)), vgt_mneg, (vl)); \
    (l) = VM_VFMERGE_VVM (vgt_eL, VM_VFNEG_V (vgt_eL, (vl)), vgt_mneg, (vl)); \
    /* the negative-integer poles: NaN + INVALID */			\
    vgt_np = VM_VFMUL_VV_M (vgt_mpole, VM_VINF (vl),			\
			    VM_VFMV_V_F (0.0, (vl)), (vl));		\
    (h) = VM_VFMERGE_VVM ((h), vgt_np, vgt_mpole, (vl));		\
    (l) = VM_VFMERGE_VFM ((l), 0.0, vgt_mpole, (vl));			\
  }									\
} while (0)

/* ---- entry kernels (the VFABI wrappers call these) ------------------- */
static inline vm_f_t
VM_KFN (lgamma) (vm_f_t x, size_t vl)
{
  vm_f_t h, l, r;
  VM_GAM_LGAMMA (h, l, x, vl);
  VM_DD_COLLAPSE (r, h, l, vl);
  return r;
}

static inline vm_f_t
VM_KFN (tgamma) (vm_f_t x, size_t vl)
{
  vm_f_t h, l, r;
  VM_GAM_TGAMMA (h, l, x, vl);
  VM_DD_COLLAPSE (r, h, l, vl);
  return r;
}
