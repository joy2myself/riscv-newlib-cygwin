/* f32/k_pow.inc -- binary32 pow kernel.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_pow.inc,
   and k_log.inc / k_exp.inc which it uses) active at
   VM_LMUL == 2*VM32_LMUL.

   Native binary32 kernel: powf(x,y) = exp2(y*log2(x)), with log2 and
   exp2 evaluated in binary32 double-float (two-word) arithmetic -- no
   binary64 register group, no widening/narrowing and no division on the
   hot path.  Lanes outside the covered domain (x not a positive normal,
   |y| outside [2^-32, 2048), |y*log2 x| >= 96, i.e. a result outside
   [2^-96, 2^96]) are evaluated one by one by the scalar fixup threaded
   in from the entry point -- the scalar pow the binary64 kernel also
   uses for its own guard lanes -- so their values and gated flags are
   inherited from the previously verified path.  y == 1 (a bit-exact
   class: pow(x,1) == x) is merged from x directly, and |y| < 2^-32 needs
   no fixup lane at all: t rounds to 1.0 exactly there.

   Accuracy: log2 is reduced with a 128-bin table indexed by the top
   seven mantissa bits (invc has 16 significant bits, so m*invc and
   r = m*invc - 1 are exact) and the log is assembled as a normalised
   double-float pair with ~2^-38 absolute error.  t = y*log2(x) is a
   two-product pair, exp2 uses the k_exp minimax polynomial with a
   compensated 1 + r assembly.  Measured on the FULL blob: max 0.5569 ulp
   (FULL was 0.5000 through the widen / binary64 / narrow path), every
   lane within one binary32 ulp of the correctly rounded value.

   The covered domain also keeps every intermediate a normal number:
   |t| < 96 bounds 2^n to [2^-96, 2^96], so the scale construction, the
   final product and the low-word products cannot raise over/underflow,
   and the fast path leaves the gated FP flags untouched.  Subnormal
   results (|t| >= 96) take the fixup lane, whose narrowing raises
   UNDERFLOW; VM32_RAISE_UF_SUBNORMAL then also covers the exact-grid
   case, as the previous wide path did.  */

/* ---- log2 reduction table: 128 bins, one per top mantissa value ---- */
/* z = 1 + j/128 exactly; invc ~ 1/z_centre with 16 significant bits
   (so z*invc is exact in binary32); logc = -log2(invc) split into a
   24-bit hi and a 24-bit lo part.  */
#ifndef VM32_POWF_LOG_TAB_DEFINED
#define VM32_POWF_LOG_TAB_DEFINED
/* 1/c_j, 16 significant bits */
static const float VM32_POWF_INVC[128] = {
  9.961090088e-01f, 9.884185791e-01f, 9.808502197e-01f, 9.733886719e-01f,
  9.660339355e-01f, 9.588012695e-01f, 9.516754150e-01f, 9.446563721e-01f,
  9.377288818e-01f, 9.309082031e-01f, 9.241943359e-01f, 9.175567627e-01f,
  9.110260010e-01f, 9.045867920e-01f, 8.982391357e-01f, 8.919830322e-01f,
  8.858184814e-01f, 8.797302246e-01f, 8.737182617e-01f, 8.677978516e-01f,
  8.619537354e-01f, 8.561859131e-01f, 8.504943848e-01f, 8.448791504e-01f,
  8.393402100e-01f, 8.338775635e-01f, 8.284759521e-01f, 8.231506348e-01f,
  8.178863525e-01f, 8.126983643e-01f, 8.075714111e-01f, 8.025054932e-01f,
  7.975006104e-01f, 7.925720215e-01f, 7.876892090e-01f, 7.828674316e-01f,
  7.781219482e-01f, 7.734069824e-01f, 7.687683105e-01f, 7.641754150e-01f,
  7.596435547e-01f, 7.551574707e-01f, 7.507324219e-01f, 7.463531494e-01f,
  7.420349121e-01f, 7.377471924e-01f, 7.335205078e-01f, 7.293395996e-01f,
  7.252197266e-01f, 7.211303711e-01f, 7.170867920e-01f, 7.130889893e-01f,
  7.091369629e-01f, 7.052307129e-01f, 7.013702393e-01f, 6.975402832e-01f,
  6.937713623e-01f, 6.900329590e-01f, 6.863250732e-01f, 6.826629639e-01f,
  6.790466309e-01f, 6.754608154e-01f, 6.719207764e-01f, 6.684112549e-01f,
  6.649322510e-01f, 6.614990234e-01f, 6.580963135e-01f, 6.547241211e-01f,
  6.513977051e-01f, 6.481018066e-01f, 6.448364258e-01f, 6.416015625e-01f,
  6.383972168e-01f, 6.352386475e-01f, 6.320953369e-01f, 6.289978027e-01f,
  6.259155273e-01f, 6.228637695e-01f, 6.198577881e-01f, 6.168670654e-01f,
  6.139068604e-01f, 6.109771729e-01f, 6.080780029e-01f, 6.051940918e-01f,
  6.023559570e-01f, 5.995330811e-01f, 5.967407227e-01f, 5.939636230e-01f,
  5.912170410e-01f, 5.885009766e-01f, 5.858154297e-01f, 5.831451416e-01f,
  5.805053711e-01f, 5.778808594e-01f, 5.752868652e-01f, 5.727081299e-01f,
  5.701599121e-01f, 5.676269531e-01f, 5.651245117e-01f, 5.626373291e-01f,
  5.601806641e-01f, 5.577392578e-01f, 5.553131104e-01f, 5.529174805e-01f,
  5.505371094e-01f, 5.481872559e-01f, 5.458374023e-01f, 5.435180664e-01f,
  5.412292480e-01f, 5.389404297e-01f, 5.366821289e-01f, 5.344543457e-01f,
  5.322265625e-01f, 5.300140381e-01f, 5.278320312e-01f, 5.256652832e-01f,
  5.235137939e-01f, 5.213775635e-01f, 5.192718506e-01f, 5.171661377e-01f,
  5.150909424e-01f, 5.130310059e-01f, 5.109710693e-01f, 5.089416504e-01f,
  5.069274902e-01f, 5.049285889e-01f, 5.029449463e-01f, 5.009765625e-01f,
};
/* log2(c_j) = -log2(invc_j), hi part */
static const float VM32_POWF_LOGC_HI[128] = {
  5.624463316e-03f, 1.680596545e-02f, 2.789524756e-02f, 3.891211003e-02f,
  4.985422641e-02f, 6.069627404e-02f, 7.145849615e-02f, 8.213846385e-02f,
  9.275722504e-02f, 1.032891870e-01f, 1.137318462e-01f, 1.241306886e-01f,
  1.344358623e-01f, 1.446691602e-01f, 1.548285186e-01f, 1.649118215e-01f,
  1.749169976e-01f, 1.848669201e-01f, 1.947599500e-01f, 2.045690864e-01f,
  2.143176645e-01f, 2.240040004e-01f, 2.336263806e-01f, 2.431830913e-01f,
  2.526724041e-01f, 2.620925307e-01f, 2.714682817e-01f, 2.807716429e-01f,
  2.900277078e-01f, 2.992081046e-01f, 3.083382547e-01f, 3.174168169e-01f,
  3.264424801e-01f, 3.353860676e-01f, 3.443015814e-01f, 3.531600535e-01f,
  3.619318306e-01f, 3.707002997e-01f, 3.793792129e-01f, 3.880242407e-01f,
  3.966054618e-01f, 4.051505923e-01f, 4.136292934e-01f, 4.220696688e-01f,
  4.304410219e-01f, 4.388015568e-01f, 4.470908046e-01f, 4.553373754e-01f,
  4.635099173e-01f, 4.716680050e-01f, 4.797803462e-01f, 4.878459573e-01f,
  4.958637953e-01f, 5.038328171e-01f, 5.117518902e-01f, 5.196515322e-01f,
  5.274677873e-01f, 5.352628231e-01f, 5.430360436e-01f, 5.507546067e-01f,
  5.584174395e-01f, 5.660560131e-01f, 5.736369491e-01f, 5.811920762e-01f,
  5.887207389e-01f, 5.961890817e-01f, 6.036293507e-01f, 6.110409498e-01f,
  6.183894873e-01f, 6.257076263e-01f, 6.329948306e-01f, 6.402504444e-01f,
  6.474737525e-01f, 6.546294093e-01f, 6.617859006e-01f, 6.688731313e-01f,
  6.759601235e-01f, 6.830114126e-01f, 6.899908185e-01f, 6.969684958e-01f,
  7.039083242e-01f, 7.108095884e-01f, 7.176716924e-01f, 7.245302200e-01f,
  7.313117981e-01f, 7.380887270e-01f, 7.448238730e-01f, 7.515535355e-01f,
  7.582402229e-01f, 7.648832798e-01f, 7.714819312e-01f, 7.780730724e-01f,
  7.846186757e-01f, 7.911559939e-01f, 7.976465821e-01f, 8.041279912e-01f,
  8.105614781e-01f, 8.169850111e-01f, 8.233593106e-01f, 8.297228217e-01f,
  8.360359073e-01f, 8.423372507e-01f, 8.486266136e-01f, 8.548638821e-01f,
  8.610882759e-01f, 8.672593236e-01f, 8.734568357e-01f, 8.796001077e-01f,
  8.856883049e-01f, 8.918022513e-01f, 8.978602290e-01f, 9.038614035e-01f,
  9.098875523e-01f, 9.158975482e-01f, 9.218491912e-01f, 9.277836084e-01f,
  9.337005615e-01f, 9.395995736e-01f, 9.454380870e-01f, 9.513002634e-01f,
  9.571009278e-01f, 9.628821015e-01f, 9.686864614e-01f, 9.744278193e-01f,
  9.801486731e-01f, 9.858487248e-01f, 9.915276170e-01f, 9.971849918e-01f,
};
/* log2(c_j), lo part */
static const float VM32_POWF_LOGC_LO[128] = {
  -1.134979125e-10f, 4.692491951e-10f, 1.305284353e-10f, -2.798176274e-11f,
  -1.521185622e-09f, 1.610767186e-09f, -3.201185317e-09f, 6.048253010e-10f,
  1.874559397e-09f, -2.680613287e-09f, 9.136991164e-10f, -3.693584993e-09f,
  2.986844549e-09f, 3.524697645e-09f, -4.589071789e-09f, 6.704997890e-09f,
  -1.489653678e-10f, -4.330559467e-09f, 7.151378795e-11f, -5.973587403e-09f,
  -5.555077287e-09f, -4.459578928e-09f, 3.535895132e-09f, 7.299397087e-09f,
  -6.286404286e-09f, -7.121125467e-09f, -8.361149106e-09f, -1.287924167e-08f,
  -3.748844346e-09f, -2.318363002e-10f, 1.527515670e-09f, 1.039612396e-08f,
  -8.216403558e-09f, -1.249479986e-08f, 1.971707242e-09f, 1.473138500e-08f,
  -9.654086419e-09f, 5.823856952e-09f, 1.412847972e-08f, 9.927116018e-09f,
  7.276764524e-09f, -1.408148353e-08f, 1.144426331e-08f, -1.083148127e-09f,
  6.954689269e-09f, 1.280512052e-08f, -1.111073722e-08f, -8.085804915e-09f,
  9.541621715e-09f, -1.385083337e-08f, 3.865261000e-09f, 9.957502378e-09f,
  2.748571371e-09f, -2.777596819e-08f, -1.033683716e-08f, 2.720718051e-08f,
  1.775213754e-08f, -1.162941077e-09f, -9.807614276e-09f, 3.891294398e-09f,
  6.141243958e-09f, 2.865632398e-09f, 5.499539046e-09f, -7.397350732e-09f,
  2.091501194e-09f, -1.345488254e-08f, 4.023489986e-09f, 1.394563931e-08f,
  -2.940154076e-08f, 1.284363815e-08f, 2.256932952e-08f, -4.734561188e-09f,
  -1.950658124e-08f, -2.102098273e-09f, 2.241101882e-08f, -1.376303516e-08f,
  5.093814703e-09f, 2.545937328e-08f, 1.565657115e-08f, -2.370661711e-08f,
  -2.126632381e-08f, 2.694958923e-08f, 1.179881193e-09f, -2.846744174e-08f,
  9.676107027e-09f, 6.983080336e-09f, -1.050278087e-08f, -1.729552146e-08f,
  1.817872253e-08f, 5.803335146e-09f, -2.892575779e-08f, 1.551525308e-08f,
  3.570145513e-09f, 1.532693972e-08f, -1.868589372e-08f, 2.029057278e-08f,
  9.662459277e-09f, -1.198352528e-08f, 1.823900320e-08f, -6.351073001e-10f,
  1.319203413e-09f, 2.208201799e-08f, 2.444412672e-08f, 2.965342460e-08f,
  5.947832893e-09f, -1.779594605e-08f, 3.540909788e-09f, -3.658811476e-09f,
  -1.472367384e-08f, 2.612521222e-08f, 1.757395118e-08f, -2.351938022e-08f,
  2.807435884e-08f, -2.507916363e-08f, 1.076537526e-09f, 2.908439889e-08f,
  -1.753488732e-08f, 2.079055506e-08f, -1.201592958e-08f, 1.607598676e-08f,
  -4.154253830e-09f, -2.672600097e-08f, 2.389445619e-08f, 1.327840771e-08f,
  1.969089247e-08f, 5.491972210e-09f, -9.471688323e-09f, -7.443601735e-09f,
};
#endif

/* two-part constants: 15/24 significant bits in the hi part plus the f32
   tail, so a leading product carried by an fma pair is exact and the
   constant itself is represented to ~2^-48.  */
#define VM32_POWF_L2E	 1.442695022e+00f	/* fl32(log2 e) 0x3fb8aa3b */
#define VM32_POWF_L2E_LO 1.925963034e-08f	/* 0x32a57060 */
#define VM32_POWF_LN2	 6.931471825e-01f	/* fl32(ln2)    0x3f317218 */
#define VM32_POWF_LN2_LO (-1.904654212e-09f)	/* 0xb102e308 */
#define VM32_POWF_TA	 (-7.213475108e-01f)	/* -log2(e)/2 */
#define VM32_POWF_TB	 4.808983505e-01f	/*  log2(e)/3 */
#define VM32_POWF_TC	 (-3.606737554e-01f)	/* -log2(e)/4 */
/* e^r = 1 + r + r^2*Q(r), |r| <= 0.3466 (k_exp's minimax fit).  */
#define VM32_POWF_E0	 1.989110315e-04f
#define VM32_POWF_E1	 1.393375103e-03f
#define VM32_POWF_E2	 8.333310485e-03f
#define VM32_POWF_E3	 4.166646302e-02f
#define VM32_POWF_E4	 1.666666716e-01f
#define VM32_POWF_E5	 5.000000000e-01f
#define VM32_POWF_SHIFT	 12582912.0f		/* 1.5*2^23 */
#define VM32_POWF_SHIFTB 0x4b400000u

/* ---- per-lane scalar epilogue -------------------------------------- */

/* Lanes the fast path does not cover are compressed out, evaluated one by
   one by the scalar pow the entry passes in (a binary64 evaluation, then
   one narrowing), and merged back.  Values and gated FP flags are those of
   the previous widen / binary64 kernel / narrow path, which used the same
   scalar pow for the lanes the binary64 kernel could not vectorise.

   Out of line and guarded by vfirst at the call site: the 7 KB of buffer
   and the vector-register spills a call forces would otherwise be charged
   to every strip, even though a strip with a covered domain takes no lane
   here at all (same shape as k_log.inc / k_exp.inc).

   Buffer bound: vl <= VLMAX(e32, LMUL <= 8) = VLEN/4 lanes; 600 covers
   VLEN up to 2400.  */
__attribute__ ((noinline)) static vm32_f_t
VM32_KFN (pow_fixup) (vm32_f_t x, vm32_f_t y, vm32_f_t r, vm32_b_t m,
		      size_t vl, double (*fn) (double, double))
{
  float xb[600], yb[600], rb[600];
  size_t n = (size_t) VM32_VCMP_POPC_M (m, vl);
  VM32_VSE ((float *) xb, VM32_VCOMPRESS (x, m, vl), n);
  VM32_VSE ((float *) yb, VM32_VCOMPRESS (y, m, vl), n);
  for (size_t i = 0; i < n; i++)
    rb[i] = (float) fn ((double) xb[i], (double) yb[i]);
  vm32_u_t idx = VM32_VIOTA_M (m, vl);
  vm32_f_t pick = VM32_VLUXEI32 ((const float *) rb,
				 VM32_VSLL_VX (idx, 2, vl), vl);
  return VM32_VFMERGE_VVM (r, pick, m, vl);
}

static inline __attribute__ ((always_inline)) vm32_f_t
VM32_KFN (pow) (vm32_f_t x, vm32_f_t y, size_t vl,
		double (*fixup) (double, double))
{
  vm32_u_t ix = VM32_BITS (x);
  vm32_u_t iy = VM32_BITS (y);
  /* ---- 1. lane classification ------------------------------------ */
  /* x positive normal: (ix - 2^-126 bits) < 0x7f000000 covers x < 0,
     +-0, subnormals, +-Inf and NaN in one unsigned compare; y keeps its
     lane only for 2^-32 <= |y| < 2048.  The evaluation below is finite
     for *any* input bit pattern (m is rebuilt in [1,2), the exponent
     field is read as an integer and no division or sqrt is involved), so
     a covered-out lane only needs its y replaced by zero: then t is
     exactly 0, 2^t is exactly 1 without raising a flag, and the scalar
     fixup overwrites the lane at the end.  */
  vm32_b_t ngx = VM32_VMNOT_M (
    VM32_VMSLTU_VX (VM32_VSUB_VX (ix, 0x00800000u, vl), 0x7f000000u, vl),
    vl);
  vm32_b_t gy = VM32_VMSLTU_VX (
    VM32_VSUB_VX (VM32_VAND_VX (iy, 0x7fffffffu, vl), 0x2f800000u, vl),
    0x15800000u, vl);
  vm32_b_t yone = VM32_VMSEQ_VX (iy, 0x3f800000u, vl);	/* y == 1 */
  vm32_b_t bad = VM32_VMOR_MM (ngx, VM32_VMNOT_M (gy, vl), vl);
  vm32_f_t ys = VM32_VFMERGE_VFM (y, 0.0f, VM32_VMNOT_M (gy, vl), vl);

  /* ---- 2. log2(x) as a normalised (hi, lo) pair ------------------ */
  vm32_u_t off = VM32_VAND_VX (VM32_VSRL_VX (ix, 14, vl), 0x1fcu, vl);
  /* m = 1 + mantissa (exact), k = unbiased exponent.  */
  vm32_f_t m = VM32_FROM_BITS (
    VM32_VOR_VX (VM32_VAND_VX (ix, 0x7fffffu, vl), 0x3f800000u, vl));
  vm32_f_t kf = VM32_VFCVT_F_X_V (
    VM32_REINT_U_I (VM32_VSUB_VX (VM32_VSRL_VX (ix, 23, vl), 127, vl)), vl);
  vm32_f_t invc = VM32_VLUXEI32 (VM32_POWF_INVC, off, vl);
  vm32_f_t lch = VM32_VLUXEI32 (VM32_POWF_LOGC_HI, off, vl);
  vm32_f_t lcl = VM32_VLUXEI32 (VM32_POWF_LOGC_LO, off, vl);
  /* r = m*invc - 1 is exact: invc has 16 bits and |m*invc - 1| < 2^-8.  */
  vm32_f_t pr = VM32_VFMUL_VV (m, invc, vl);
  vm32_f_t pre = VM32_VFMSAC_VV (pr, invc, m, vl);	/* invc*m - pr */
  vm32_f_t r = VM32_VFSUB_VF (pr, 1.0f, vl);
  /* r*log2(e) as an exact pair, with the constant tail folded in.  */
  vm32_f_t p = VM32_VFMUL_VF (r, VM32_POWF_L2E, vl);
  vm32_f_t pe = VM32_VFMSAC_VF (p, VM32_POWF_L2E, r, vl);
  pe = VM32_VFMACC_VF (pe, VM32_POWF_L2E_LO, r, vl);
  pe = VM32_VFMACC_VF (pe, VM32_POWF_L2E, pre, vl);
  /* rest = log2(1+r) - r*log2(e) = r^2*(TA + TB r + TC r^2).  */
  vm32_f_t r2 = VM32_VFMUL_VV (r, r, vl);
  vm32_f_t Q = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_POWF_TB, vl),
			       VM32_POWF_TC, r, vl);
  Q = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_POWF_TA, vl), Q, r, vl);
  vm32_f_t rest = VM32_VFMUL_VV (r2, Q, vl);
  /* kf + logc: a fast-two-sum (|kf| >= |logc| always).  */
  vm32_f_t u = VM32_VFADD_VV (kf, lch, vl);
  vm32_f_t ue = VM32_VFSUB_VV (lch, VM32_VFSUB_VV (u, kf, vl), vl);
  vm32_f_t hi = VM32_VFADD_VV (u, p, vl);
  vm32_f_t ehi = VM32_VFSUB_VV (p, VM32_VFSUB_VV (hi, u, vl), vl);
  vm32_f_t lo = VM32_VFADD_VV (VM32_VFADD_VV (ehi, ue, vl), pe, vl);
  lo = VM32_VFADD_VV (lo, lcl, vl);
  lo = VM32_VFADD_VV (lo, rest, vl);
  lo = VM32_VFNMSAC_VV (lo, p, pre, vl);		/* -p*pre */
  /* Renormalise: |lo| <= ulp(hi)/2 afterwards, which is what keeps
     y*lo inside the linear term of 2^(f+lo) below.  */
  vm32_f_t s = VM32_VFADD_VV (hi, lo, vl);
  vm32_f_t w = VM32_VFSUB_VV (s, lo, vl);
  vm32_f_t wp = VM32_VFSUB_VV (s, w, vl);
  vm32_f_t e = VM32_VFADD_VV (VM32_VFSUB_VV (hi, w, vl),
			      VM32_VFSUB_VV (lo, wp, vl), vl);

  /* ---- 3. t = ys*(s + e) ----------------------------------------- */
  vm32_f_t th = VM32_VFMUL_VV (ys, s, vl);
  vm32_f_t te = VM32_VFMSAC_VV (th, ys, s, vl);		/* ys*s - th */
  vm32_f_t tl = VM32_VFMACC_VV (te, ys, e, vl);		/* te + ys*e */
  vm32_b_t nbig = VM32_VMNOT_M (VM32_VMSLTU_VX (
    VM32_VAND_VX (VM32_BITS (th), 0x7fffffffu, vl), 0x42c00000u, vl), vl);
  th = VM32_VFMERGE_VFM (th, 0.0f, nbig, vl);
  tl = VM32_VFMERGE_VFM (tl, 0.0f, nbig, vl);
  bad = VM32_VMOR_MM (bad, nbig, vl);

  /* ---- 4. 2^(th + tl) ------------------------------------------- */
  vm32_f_t kds = VM32_VFADD_VF (th, VM32_POWF_SHIFT, vl);
  vm32_f_t nf = VM32_VFSUB_VF (kds, VM32_POWF_SHIFT, vl);
  vm32_f_t scale = VM32_FROM_BITS (
    VM32_VSLL_VX (VM32_VSUB_VX (VM32_BITS (kds), 0x4b3fff81u, vl), 23, vl));
  vm32_f_t f = VM32_VFSUB_VV (th, nf, vl);
  /* fp = f*ln2 as an exact pair, with tl folded into the low word.  */
  vm32_f_t fp = VM32_VFMUL_VF (f, VM32_POWF_LN2, vl);
  vm32_f_t fpe = VM32_VFMSAC_VF (fp, VM32_POWF_LN2, f, vl);
  vm32_f_t rl = VM32_VFMACC_VF (fpe, VM32_POWF_LN2, tl, vl);
  rl = VM32_VFMACC_VF (rl, VM32_POWF_LN2_LO, f, vl);
  /* 2^f = 1 + fp + fp^2*Q(fp), assembled at double-float precision.  */
  vm32_f_t uu = VM32_VFMUL_VV (fp, fp, vl);
  vm32_f_t q = VM32_VFMACC_VF (VM32_VFMV_V_F (VM32_POWF_E1, vl),
			       VM32_POWF_E0, fp, vl);
  q = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_POWF_E2, vl), q, fp, vl);
  q = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_POWF_E3, vl), q, fp, vl);
  q = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_POWF_E4, vl), q, fp, vl);
  q = VM32_VFMACC_VV (VM32_VFMV_V_F (VM32_POWF_E5, vl), q, fp, vl);
  vm32_f_t rs = VM32_VFMUL_VV (uu, q, vl);
  vm32_f_t p1 = VM32_VFADD_VF (fp, 1.0f, vl);
  vm32_f_t p1e = VM32_VFSUB_VV (fp, VM32_VFSUB_VF (p1, 1.0f, vl), vl);
  vm32_f_t p2 = VM32_VFADD_VV (p1, rs, vl);
  vm32_f_t p2e = VM32_VFSUB_VV (rs, VM32_VFSUB_VV (p2, p1, vl), vl);
  vm32_f_t pl = VM32_VFADD_VV (p2e, p1e, vl);
  pl = VM32_VFMACC_VV (pl, rl, p2, vl);
  vm32_f_t uh = VM32_VFMUL_VV (scale, p2, vl);
  vm32_f_t ue2 = VM32_VFMSAC_VV (uh, scale, p2, vl);
  vm32_f_t t1 = VM32_VFMACC_VV (ue2, scale, pl, vl);
  vm32_f_t res = VM32_VFADD_VV (uh, t1, vl);
  /* y == 1 is a bit-exact class: pow(x,1) == x.  */
  res = VM32_VFMERGE_VVM (res, x, yone, vl);

  /* ---- 5. per-lane fallback and subnormal underflow carrier ------ */
  if (VM32_VFIRST_M (bad, vl) >= 0)
    {
      res = VM32_KFN (pow_fixup) (x, y, res, bad, vl, fixup);
      /* pow raises FE_UNDERFLOW for a subnormal result even when the
	 binary64 evaluation lands exactly on the binary32 subnormal grid
	 and the narrowing conversion therefore stays silent.  Every
	 subnormal result is a fixup lane here (|t| < 96 keeps the fast
	 path's results normal), so the masked carrier runs on the cold
	 path only -- on the hot path it would also cost a policy-changing
	 vsetvli.  */
      VM32_RAISE_UF_SUBNORMAL (res, vl);
    }
  return res;
}
