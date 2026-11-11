/* f32/k_invtrig.inc -- binary32 inverse trigonometric kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_invtrig.inc)
   active at VM_LMUL == 2*VM32_LMUL.

   SHAPE: native binary32 fast path plus a per-strip fallback.  The fast
   path is a branch-free transcription of newlib's scalar float fdlibm
   sources (math/sf_atan.c, math/ef_atan2.c, math/ef_asin.c,
   math/ef_acos.c); it only runs on strips where every lane is inside
   the range for which that algorithm's error bound holds, which is the
   whole benchmark domain.  A strip with any exceptional lane (Inf/NaN,
   outside the fast range, tiny or zero argument) is recomputed *as a
   whole* by the existing widen-to-binary64 path, so special values,
   signed zeros and per-lane exception flags are inherited bit-for-bit
   from the already-verified binary64 kernels.  The fallback is never
   reached by the benchmark's inputs, so its (heavy) cost does not
   matter.

   The fast path is binary32 because the binary64 kernels carry every
   intermediate as a double-double pair and so cost ~10x the scalar f32
   libm call; fdlibm's float algorithms are 0.5-0.9 ulp as written.

   The *pi-scaled* entries divide by pi, and a second rounding there
   (0.5 + 0.5 ulp) would not reliably fit the budget.  They therefore
   use the same cores carried as a two-float (hi, lo) pair -- the final
   rounding of the kernel body is captured with Fast2Sum -- and scale by
   1/pi as a hi+lo pair with an fma-compensated product
   (VM32I_DIVPI), so the scaling costs exactly one final rounding.  The
   non-scaled entries return just the hi part.

   acospi is the same story one step further: 0.5 - asin(x)/pi cancels,
   so the pair is what has to survive the subtraction, and the two
   kernels are each used in the branch where the *pair* is accurate --
   asin's small branch (error relative to the small x*r term) below 1/2,
   acos's sqrt branch (error relative to acos itself, sqrt rounding
   recovered by c) above it, with the x < 0 half reached by 1 - u
   instead of a pi-sized subtraction.  Both halves fold their offset
   into a single fma, so the final add rounds once.  */

/* ==================================================================== */
/* shared helpers                                                       */
/* ==================================================================== */

/* s + e = a + b exactly (2Sum; valid for any magnitudes).  */
#define VM32I_TWO_SUM(so, eo, a, b, vl) do {				\
  (so) = VM32_VFADD_VV ((a), (b), (vl));				\
  (eo) = VM32_VFADD_VV (VM32_VFSUB_VV ((a), (so), (vl)), (b), (vl));	\
} while (0)

/* acc = c + x*acc  (one Horner step; c is a scalar coefficient).  */
#define VM32I_HORNER(acc, c, xv, vl) \
  ((acc) = VM32_VFMACC_VV (VM32_VFMV_V_F ((c), (vl)), (xv), (acc), (vl)))

/* ==================================================================== */
/* plain binary64 compute layer (VM32D_)                                */
/* ==================================================================== */
/* Two entries -- atan2 and atan2pi -- cannot clear the budget with an
   fdlibm *float* algorithm: atan2 stacks the quotient rounding, atan's
   own error and the quadrant fixup rounding.  They are evaluated instead
   in *plain* binary64 at the same lane count (an f64m2K group has
   exactly the lanes of the f32mK group), where the algorithm's ~1e-16
   relative error is far below a binary32 ulp, so the final narrowing is
   the only rounding.  That is the same "drop the double-double" move the
   binary64 playbook prescribes, applied at the f32 entry: the existing
   wide path for these entries is the full dd kernel, this is the plain
   f64 one under it.  (acospi used to live here as well, as a plain-f64
   acos times 1/pi; its native path below is both exact enough and ~2x
   cheaper, so only the f64 *fallback* remains.)

   The binary64 compat macros of <vecmath_rvvcompat.h> are not active at
   this LMUL from the including TU (vfabi32.c instantiates them at its
   own LMULs), so the handful of operations needed here are re-defined,
   keyed off VM32_LMUL, exactly like <f32/wrap.h> does for the widening.
   Re-include-safe: every macro is #undef'ed first.  Masks need no
   binary64 twin -- the f64 group has the same lane count as the f32
   group, so vbool<...> is shared.  */

#undef vm32d_f_t
#undef vm32d_u64_t
#undef VM32D_M
#undef VM32D_FN
#undef VM32D_U64FN
#undef VM32D_VFADD_VV
#undef VM32D_VFSUB_VV
#undef VM32D_VFSUB_VF
#undef VM32D_VFRSUB_VF
#undef VM32D_VFMUL_VV
#undef VM32D_VFMUL_VF
#undef VM32D_VFDIV_VV
#undef VM32D_VFMACC_VV
#undef VM32D_VFMACC_VF
#undef VM32D_VFMSAC_VV
#undef VM32D_VFNMSAC_VV
#undef VM32D_VAND_VX
#undef VM32D_VFSQRT_V
#undef VM32D_VFABS_V
#undef VM32D_VFNEG_V
#undef VM32D_VFMV_V_F
#undef VM32D_BITS
#undef VM32D_FROM_BITS
#undef VM32D_VMERGE_VVM
#undef VM32D_VFMERGE_VVM

#if VM32_LMUL == 1
# define vm32d_f_t vfloat64m2_t
# define vm32d_u64_t vuint64m2_t
# define VM32D_M 2
#elif VM32_LMUL == 2
# define vm32d_f_t vfloat64m4_t
# define vm32d_u64_t vuint64m4_t
# define VM32D_M 4
#elif VM32_LMUL == 4
# define vm32d_f_t vfloat64m8_t
# define vm32d_u64_t vuint64m8_t
# define VM32D_M 8
#else
# error "binary32 invtrig kernels are instantiated at VM32_LMUL 1, 2 or 4"
#endif

#define VM32D_FN(op) \
  VM32_XP2 (VM32_XP2 (__riscv_, op), VM32_XP2 (_f64m, VM32D_M))
#define VM32D_U64FN(op) \
  VM32_XP2 (VM32_XP2 (__riscv_, op), VM32_XP2 (_u64m, VM32D_M))

#define VM32D_VFADD_VV(a, b, vl) VM32D_FN (vfadd_vv) (a, b, vl)
#define VM32D_VFSUB_VV(a, b, vl) VM32D_FN (vfsub_vv) (a, b, vl)
#define VM32D_VFSUB_VF(a, s, vl) VM32D_FN (vfsub_vf) (a, s, vl)
#define VM32D_VFRSUB_VF(a, s, vl) VM32D_FN (vfrsub_vf) (a, s, vl)
#define VM32D_VFMUL_VV(a, b, vl) VM32D_FN (vfmul_vv) (a, b, vl)
#define VM32D_VFMUL_VF(a, s, vl) VM32D_FN (vfmul_vf) (a, s, vl)
#define VM32D_VFDIV_VV(a, b, vl) VM32D_FN (vfdiv_vv) (a, b, vl)
#define VM32D_VFMACC_VV(d, a, b, vl) VM32D_FN (vfmacc_vv) (d, a, b, vl)
#define VM32D_VFMACC_VF(d, s, b, vl) VM32D_FN (vfmacc_vf) (d, s, b, vl)
#define VM32D_VFMSAC_VV(d, a, b, vl) VM32D_FN (vfmsac_vv) (d, a, b, vl)
#define VM32D_VFNMSAC_VV(d, a, b, vl) VM32D_FN (vfnmsac_vv) (d, a, b, vl)
#define VM32D_VAND_VX(a, s, vl) VM32D_U64FN (vand_vx) (a, s, vl)
#define VM32D_VFSQRT_V(a, vl) VM32D_FN (vfsqrt_v) (a, vl)
#define VM32D_VFABS_V(a, vl) VM32D_FN (vfabs_v) (a, vl)
#define VM32D_VFNEG_V(a, vl) VM32D_FN (vfneg_v) (a, vl)
#define VM32D_VFMV_V_F(s, vl) VM32D_FN (vfmv_v_f) (s, vl)
#define VM32D_BITS(v) \
  VM32_XP2 (VM32_XP3 (__riscv_vreinterpret_v_f64m, VM32D_M, _u64m), \
	    VM32D_M) (v)
#define VM32D_FROM_BITS(u) \
  VM32_XP2 (VM32_XP3 (__riscv_vreinterpret_v_u64m, VM32D_M, _f64m), \
	    VM32D_M) (u)
/* f64 vmerge: no explicit vfmerge_vvm in GCC, so go through u64 bits.  */
#define VM32D_VMERGE_VVM(a, b, m, vl) VM32D_U64FN (vmerge_vvm) (a, b, m, vl)
#define VM32D_VFMERGE_VVM(a, b, m, vl) \
  VM32D_FROM_BITS (VM32D_VMERGE_VVM (VM32D_BITS (a), VM32D_BITS (b), m, vl))

/* ==================================================================== */
/* binary64 atan (fdlibm s_atan.c, double coefficients), for atan2       */
/* ==================================================================== */

#define VM32D_ATANHI0 4.63647609000806093515e-01
#define VM32D_ATANHI1 7.85398163397448278999e-01
#define VM32D_ATANHI2 9.82793723247329054082e-01
#define VM32D_ATANHI3 1.57079632679489655800e+00
#define VM32D_ATANLO0 2.26987774529616870924e-17
#define VM32D_ATANLO1 3.06161699786838301793e-17
#define VM32D_ATANLO2 1.39033110312309984516e-17
#define VM32D_ATANLO3 6.12323399573676603587e-17
#define VM32D_AT0 3.33333333333329318027e-01
#define VM32D_AT1 -1.99999999998764832476e-01
#define VM32D_AT2 1.42857142725034663711e-01
#define VM32D_AT3 -1.11111104054623557880e-01
#define VM32D_AT4 9.09088713343650656196e-02
#define VM32D_AT5 -7.69187620504482999495e-02
#define VM32D_AT6 6.66107313738753120669e-02
#define VM32D_AT7 -5.83357013379057348645e-02
#define VM32D_AT8 4.97687799461593236017e-02
#define VM32D_AT9 -3.65315727442169155270e-02
#define VM32D_AT10 1.62858201153657823623e-02

/* atan(q) in binary64 for finite non-negative q; the f32 magnitude
   bit pattern of q drives the region selects (a 1-ulp region slip at a
   boundary is harmless: every reduction is exact and the polynomial
   covers the union of the reduced ranges).  */
static inline vm32d_f_t
VM32_KFN (invtrig_d_atan) (vm32d_f_t q, vm32_u_t iq, size_t vl)
{
  vm32_b_t m0 = VM32_VMSLTU_VX (iq, 0x3ee00000u, vl);	/* < 0.4375 */
  vm32_b_t m1 = VM32_VMSLTU_VX (iq, 0x3f300000u, vl);	/* < 0.6875 */
  vm32_b_t m2 = VM32_VMSLTU_VX (iq, 0x3f980000u, vl);	/* < 1.1875 */
  vm32_b_t m3 = VM32_VMSLTU_VX (iq, 0x401c0000u, vl);	/* < 2.4375 */

  vm32d_f_t c = VM32D_VFMV_V_F (1.5, vl);
  c = VM32D_VFMERGE_VVM (c, VM32D_VFMV_V_F (1.0, vl), m2, vl);
  c = VM32D_VFMERGE_VVM (c, VM32D_VFMV_V_F (0.5, vl), m1, vl);
  c = VM32D_VFMERGE_VVM (c, VM32D_VFMV_V_F (0.0, vl), m0, vl);
  vm32d_f_t r = VM32D_VFDIV_VV (
    VM32D_VFSUB_VV (q, c, vl),
    VM32D_VFMACC_VV (VM32D_VFMV_V_F (1.0, vl), c, q, vl), vl);
  r = VM32D_VFMERGE_VVM (r,
			 VM32D_VFDIV_VV (VM32D_VFMV_V_F (-1.0, vl), q, vl),
			 VM32_VMNOT_M (m3, vl), vl);

  vm32d_f_t z = VM32D_VFMUL_VV (r, r, vl);
  vm32d_f_t w = VM32D_VFMUL_VV (z, z, vl);
  vm32d_f_t o = VM32D_VFMV_V_F (VM32D_AT10, vl);
  o = VM32D_VFMACC_VV (VM32D_VFMV_V_F (VM32D_AT8, vl), w, o, vl);
  o = VM32D_VFMACC_VV (VM32D_VFMV_V_F (VM32D_AT6, vl), w, o, vl);
  o = VM32D_VFMACC_VV (VM32D_VFMV_V_F (VM32D_AT4, vl), w, o, vl);
  o = VM32D_VFMACC_VV (VM32D_VFMV_V_F (VM32D_AT2, vl), w, o, vl);
  o = VM32D_VFMACC_VV (VM32D_VFMV_V_F (VM32D_AT0, vl), w, o, vl);
  vm32d_f_t e = VM32D_VFMV_V_F (VM32D_AT9, vl);
  e = VM32D_VFMACC_VV (VM32D_VFMV_V_F (VM32D_AT7, vl), w, e, vl);
  e = VM32D_VFMACC_VV (VM32D_VFMV_V_F (VM32D_AT5, vl), w, e, vl);
  e = VM32D_VFMACC_VV (VM32D_VFMV_V_F (VM32D_AT3, vl), w, e, vl);
  e = VM32D_VFMACC_VV (VM32D_VFMV_V_F (VM32D_AT1, vl), w, e, vl);
  vm32d_f_t poly = VM32D_VFADD_VV (VM32D_VFMUL_VV (z, o, vl),
				   VM32D_VFMUL_VV (w, e, vl), vl);
  vm32d_f_t t = VM32D_VFMUL_VV (r, poly, vl);

  /* atanhi + atanlo is the accurately split atan(c) (f32 pair, so
     ~3e-15 -- far below a binary32 ulp).  */
  vm32d_f_t hi = VM32D_VFMV_V_F (VM32D_ATANHI3 + VM32D_ATANLO3, vl);
  hi = VM32D_VFMERGE_VVM (hi,
			  VM32D_VFMV_V_F (VM32D_ATANHI2 + VM32D_ATANLO2, vl),
			  m3, vl);
  hi = VM32D_VFMERGE_VVM (hi,
			  VM32D_VFMV_V_F (VM32D_ATANHI1 + VM32D_ATANLO1, vl),
			  m2, vl);
  hi = VM32D_VFMERGE_VVM (hi,
			  VM32D_VFMV_V_F (VM32D_ATANHI0 + VM32D_ATANLO0, vl),
			  m1, vl);
  hi = VM32D_VFMERGE_VVM (hi, VM32D_VFMV_V_F (0.0, vl), m0, vl);
  return VM32D_VFSUB_VV (hi, VM32D_VFSUB_VV (t, r, vl), vl);
}

/* 1/pi and 2/pi as hi+lo float pairs (each sums to the constant to
   2^-48).  acospi folds the pi scaling into the same fma that applies
   its 1/2 or 0/1 offset; the sqrt branch needs 2/pi because acos itself
   is 2*(df + w1) there, so the factor two never rounds.  */
#define VM32I_INVPI_HI 3.18309873342514038086e-01f	/* 0x3ea2f983 */
#define VM32I_INVPI_LO 1.2841276486597053e-08f		/* 0x325c9c88 */
#define VM32I_TWOPI_HI 6.366197466850281e-01f		/* 0x3f22f983 */
#define VM32I_TWOPI_LO 2.5682552973194106e-08f		/* 0x32dc9c88 */

/* pi/2, pi/4 and pi as hi+lo float pairs (each sums to the exact
   constant to ~2^-48).  */
#define VM32I_PIO2_HI 1.57079637050628662109e+00f	/* 0x3fc90fdb */
#define VM32I_PIO2_LO -4.371138828673793e-08f		/* 0xb33bbd2e */
#define VM32I_PIO4_HI 7.85398185253143310547e-01f	/* 0x3f490fdb */
#define VM32I_PIO4_LO -2.1855694143368964e-08f		/* 0xb2bbbd2e */
#define VM32I_PI_HI 3.14159274101257324219e+00f	/* 0x40490fdb */
#define VM32I_PI_LO -8.742277657347586e-08f		/* 0xb3bbbd2e */

/* RN((h + l) / pi), with the hi+lo product kept to 2^-48 so that the
   final add is the only rounding.  */
static inline vm32_f_t
VM32_KFN (invtrig_divpi) (vm32_f_t h, vm32_f_t l, size_t vl)
{
  vm32_f_t p = VM32_VFMUL_VF (h, VM32I_INVPI_HI, vl);
  /* e = h*HI - p (exact residual of the product), then the tail.  */
  vm32_f_t e = VM32_VFMSAC_VF (p, VM32I_INVPI_HI, h, vl);
  e = VM32_VFMACC_VF (e, VM32I_INVPI_LO, h, vl);
  e = VM32_VFMACC_VF (e, VM32I_INVPI_HI, l, vl);
  return VM32_VFADD_VV (p, e, vl);
}

/* ==================================================================== */
/* atan: sf_atan.c                                                      */
/* ==================================================================== */

/* fdlibm float atan constants (hex patterns from sf_atan.c).  */
#define VM32I_ATANHI0 4.6364760399e-01f	/* atan(0.5)hi 0x3eed6338 */
#define VM32I_ATANHI1 7.8539812565e-01f	/* atan(1.0)hi 0x3f490fda */
#define VM32I_ATANHI2 9.8279368877e-01f	/* atan(1.5)hi 0x3f7b985e */
#define VM32I_ATANHI3 1.5707962513e+00f	/* atan(inf)hi 0x3fc90fda */
#define VM32I_ATANLO0 5.0121582440e-09f	/* atan(0.5)lo 0x31ac3769 */
#define VM32I_ATANLO1 3.7748947079e-08f	/* atan(1.0)lo 0x33222168 */
#define VM32I_ATANLO2 3.4473217170e-08f	/* atan(1.5)lo 0x33140fb4 */
#define VM32I_ATANLO3 7.5497894159e-08f	/* atan(inf)lo 0x33a22168 */
#define VM32I_AT0 3.3333334327e-01f	/* 0x3eaaaaaa */
#define VM32I_AT1 -2.0000000298e-01f	/* 0xbe4ccccd */
#define VM32I_AT2 1.4285714924e-01f	/* 0x3e124925 */
#define VM32I_AT3 -1.1111110449e-01f	/* 0xbde38e38 */
#define VM32I_AT4 9.0908870101e-02f	/* 0x3dba2e6e */
#define VM32I_AT5 -7.6918758452e-02f	/* 0xbd9d8795 */
#define VM32I_AT6 6.6610731184e-02f	/* 0x3d886b35 */
#define VM32I_AT7 -5.8335702866e-02f	/* 0xbd6ef16b */
#define VM32I_AT8 4.9768779427e-02f	/* 0x3d4bda59 */
#define VM32I_AT9 -3.6531571299e-02f	/* 0xbd15a221 */
#define VM32I_AT10 1.6285819933e-02f	/* 0x3c8569d7 */

/* atan of a non-negative binary32 argument with |x| in [2^-29, 10.5]:
   the four fdlibm reductions, branch-free, result carried as (h, l).
   The caller guarantees a finite lane inside the fast range.  */
static inline void
VM32_KFN (invtrig_atan_pos2) (vm32_f_t ax, vm32_f_t *ho, vm32_f_t *lo,
			      size_t vl)
{
  vm32_u_t ix = VM32_VAND_VX (VM32_BITS (ax), 0x7fffffffU, vl);
  /* cumulative "< threshold" masks; the selects below apply them from
     the largest threshold down, so the smaller regions win.  */
  vm32_b_t m0 = VM32_VMSLTU_VX (ix, 0x3ee00000u, vl);	/* < 0.4375 */
  vm32_b_t m1 = VM32_VMSLTU_VX (ix, 0x3f300000u, vl);	/* < 0.6875 */
  vm32_b_t m2 = VM32_VMSLTU_VX (ix, 0x3f980000u, vl);	/* < 1.1875 */
  vm32_b_t m3 = VM32_VMSLTU_VX (ix, 0x401c0000u, vl);	/* < 2.4375 */

  /* c = 0 for |x|<0.4375 (the quotient degenerates to the argument),
     0.5, 1.0, 1.5 for the three middle regions.  */
  vm32_f_t c = VM32_VFMV_V_F (1.5f, vl);
  c = VM32_VFMERGE_VFM (c, 1.0f, m2, vl);
  c = VM32_VFMERGE_VFM (c, 0.5f, m1, vl);
  c = VM32_VFMERGE_VFM (c, 0.0f, m0, vl);
  vm32_f_t r = VM32_VFDIV_VV (
    VM32_VFSUB_VV (ax, c, vl),
    VM32_VFMACC_VV (VM32_VFMV_V_F (1.0f, vl), c, ax, vl), vl);
  /* top region: r = -1/x, which one reciprocal gives directly.  */
  r = VM32_VFMERGE_VVM (r, VM32_VFRDIV_VF (ax, -1.0f, vl),
			VM32_VMNOT_M (m3, vl), vl);

  vm32_f_t z = VM32_VFMUL_VV (r, r, vl);
  vm32_f_t w = VM32_VFMUL_VV (z, z, vl);
  /* odd/even split of the degree-11 polynomial, as in fdlibm.  */
  vm32_f_t o = VM32_VFMV_V_F (VM32I_AT10, vl);
  VM32I_HORNER (o, VM32I_AT8, w, vl);
  VM32I_HORNER (o, VM32I_AT6, w, vl);
  VM32I_HORNER (o, VM32I_AT4, w, vl);
  VM32I_HORNER (o, VM32I_AT2, w, vl);
  VM32I_HORNER (o, VM32I_AT0, w, vl);
  vm32_f_t ev = VM32_VFMV_V_F (VM32I_AT9, vl);
  VM32I_HORNER (ev, VM32I_AT7, w, vl);
  VM32I_HORNER (ev, VM32I_AT5, w, vl);
  VM32I_HORNER (ev, VM32I_AT3, w, vl);
  VM32I_HORNER (ev, VM32I_AT1, w, vl);
  vm32_f_t poly = VM32_VFADD_VV (VM32_VFMUL_VV (z, o, vl),
				 VM32_VFMUL_VV (w, ev, vl), vl);
  vm32_f_t t = VM32_VFMUL_VV (r, poly, vl);

  vm32_f_t hi = VM32_VFMV_V_F (VM32I_ATANHI3, vl);
  hi = VM32_VFMERGE_VFM (hi, VM32I_ATANHI2, m3, vl);
  hi = VM32_VFMERGE_VFM (hi, VM32I_ATANHI1, m2, vl);
  hi = VM32_VFMERGE_VFM (hi, VM32I_ATANHI0, m1, vl);
  hi = VM32_VFMERGE_VFM (hi, 0.0f, m0, vl);
  vm32_f_t lc = VM32_VFMV_V_F (VM32I_ATANLO3, vl);
  lc = VM32_VFMERGE_VFM (lc, VM32I_ATANLO2, m3, vl);
  lc = VM32_VFMERGE_VFM (lc, VM32I_ATANLO1, m2, vl);
  lc = VM32_VFMERGE_VFM (lc, VM32I_ATANLO0, m1, vl);
  lc = VM32_VFMERGE_VFM (lc, 0.0f, m0, vl);
  /* fdlibm's shape: hi - ((x*(s1+s2) - lo) - x).  For the |x|<0.4375
     region hi == lo == 0 and this is exactly x - x*(s1+s2).  */
  vm32_f_t u = VM32_VFSUB_VV (VM32_VFSUB_VV (t, lc, vl), r, vl);
  vm32_f_t h = VM32_VFSUB_VV (hi, u, vl);
  /* Fast2Sum residual of that last subtraction.  On the |x|<0.4375
     lanes hi == 0 and the subtraction is exact there, so the residual
     comes from the r - t side instead.  */
  vm32_f_t el = VM32_VFSUB_VV (VM32_VFSUB_VV (hi, h, vl), u, vl);
  vm32_f_t e0 = VM32_VFSUB_VV (VM32_VFSUB_VV (r, h, vl), t, vl);
  *ho = h;
  *lo = VM32_VFMERGE_VVM (el, e0, m0, vl);
}

/* entry kernels: `bad` lanes are the ones outside the fast range; they
   are sanitized to a benign constant so the fast path cannot raise a
   spurious flag, then the whole strip is also evaluated by the wide
   binary64 path and only the bad lanes take that result.  The good
   lanes always keep the native result, which makes a lane's value
   independent of the strip it is processed in (the length-sweep pass
   requires bitwise identity across call lengths).  */

static inline vm32_f_t
VM32_KFN (atan) (vm32_f_t x, size_t vl)
{
  /* Wide path for |x| >= 10.5 (this also catches every inf/NaN) and for
     |x| < 2^-29, where fdlibm short-circuits to the argument and a
     binary32 x*x would raise a spurious FE_UNDERFLOW.  */
  vm32_u_t ix = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffU, vl);
  vm32_b_t mbad = VM32_VMSGEU_VX (ix, 0x41280000u, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSLTU_VX (ix, 0x31000000u, vl), vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.5f, mbad, vl);

  vm32_f_t h, l;
  VM32_KFN (invtrig_atan_pos2) (VM32_VFABS_V (xs, vl), &h, &l, vl);
  h = VM32_VFSGNJX_VV (h, xs, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t w = VM32_NARROW (VM32_D64 (atan) (VM32_WIDEN (x, vl), vl),
				vl);
      h = VM32_VFMERGE_VVM (h, w, mbad, vl);
    }
  return h;
}

static inline vm32_f_t
VM32_KFN (atanpi) (vm32_f_t x, size_t vl)
{
  vm32_u_t ix = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffU, vl);
  vm32_b_t mbad = VM32_VMSGEU_VX (ix, 0x41280000u, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSLTU_VX (ix, 0x31000000u, vl), vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.5f, mbad, vl);

  vm32_f_t h, l;
  VM32_KFN (invtrig_atan_pos2) (VM32_VFABS_V (xs, vl), &h, &l, vl);
  h = VM32_VFSGNJX_VV (h, xs, vl);
  l = VM32_VFSGNJX_VV (l, xs, vl);
  h = VM32_KFN (invtrig_divpi) (h, l, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t w = VM32_NARROW (VM32_D64 (atanpi) (VM32_WIDEN (x, vl), vl),
				vl);
      h = VM32_VFMERGE_VVM (h, w, mbad, vl);
    }
  return h;
}

/* ==================================================================== */
/* asin / acos: ef_asin.c / ef_acos.c                                   */
/* ==================================================================== */

/* shared pS/qS rational of both kernels (identical coefficient set).  */
#define VM32I_PS0 1.6666667163e-01f	/* 0x3e2aaaab */
#define VM32I_PS1 -3.2556581497e-01f	/* 0xbea6b090 */
#define VM32I_PS2 2.0121252537e-01f	/* 0x3e4e0aa8 */
#define VM32I_PS3 -4.0055535734e-02f	/* 0xbd241146 */
#define VM32I_PS4 7.9153501429e-04f	/* 0x3a4f7f04 */
#define VM32I_PS5 3.4793309169e-05f	/* 0x3811ef08 */
#define VM32I_QS1 -2.4033949375e+00f	/* 0xc019d139 */
#define VM32I_QS2 2.0209457874e+00f	/* 0x4001572d */
#define VM32I_QS3 -6.8828397989e-01f	/* 0xbf303361 */
#define VM32I_QS4 7.7038154006e-02f	/* 0x3d9dc62e */

/* The p/q rational, evaluated at a per-lane argument z (x*x for the
   small branch, (1-|x|)/2 for the sqrt branch): one division covers
   every lane of both kernels.  */
static inline vm32_f_t
VM32_KFN (invtrig_asin_r) (vm32_f_t z, size_t vl)
{
  vm32_f_t p = VM32_VFMV_V_F (VM32I_PS5, vl);
  VM32I_HORNER (p, VM32I_PS4, z, vl);
  vm32_f_t q = VM32_VFMV_V_F (VM32I_QS4, vl);
  VM32I_HORNER (q, VM32I_QS3, z, vl);
  VM32I_HORNER (p, VM32I_PS3, z, vl);
  VM32I_HORNER (q, VM32I_QS2, z, vl);
  VM32I_HORNER (p, VM32I_PS2, z, vl);
  VM32I_HORNER (q, VM32I_QS1, z, vl);
  VM32I_HORNER (p, VM32I_PS1, z, vl);
  VM32I_HORNER (p, VM32I_PS0, z, vl);
  p = VM32_VFMUL_VV (z, p, vl);
  q = VM32_VFMACC_VV (VM32_VFMV_V_F (1.0f, vl), z, q, vl);
  return VM32_VFDIV_VV (p, q, vl);
}

/* asin(|x|) for |x| in [2^-27, 1), carried as (h, l).  Forced inline:
   shared by asin and asinpi, and GCC 15 otherwise emits one out-of-line
   copy whose riscv_vector_cc prologue saves/restores 15 vector
   registers per strip (~30 vector memory ops).  */
static inline __attribute__ ((always_inline)) void
VM32_KFN (invtrig_asin2) (vm32_f_t ax, vm32_u_t ix, vm32_f_t *ho,
			  vm32_f_t *lo, size_t vl)
{
  vm32_b_t mlo = VM32_VMSLTU_VX (ix, 0x3f000000u, vl);	/* |x| < 0.5 */
  vm32_b_t mtop = VM32_VMSGEU_VX (ix, 0x3f79999au, vl);	/* |x| >= 0.975 */
  vm32_f_t t = VM32_VFMUL_VF (VM32_VFRSUB_VF (ax, 1.0f, vl), 0.5f, vl);
  vm32_f_t z = VM32_VFMERGE_VVM (t, VM32_VFMUL_VV (ax, ax, vl), mlo, vl);
  vm32_f_t r = VM32_KFN (invtrig_asin_r) (z, vl);	/* = w in fdlibm */

  /* small branch: |x| + |x|*w.  The sign is applied by the caller.  */
  vm32_f_t hlo, elo;
  VM32I_TWO_SUM (hlo, elo, ax, VM32_VFMUL_VV (ax, r, vl), vl);

  /* sqrt branch: s = sqrt(t), and for |x| <= 0.975 fdlibm recovers the
     root's rounding with c = (t - w*w)/(s + w) for w = s truncated.  */
  vm32_f_t s = VM32_VFSQRT_V (t, vl);
  vm32_f_t wtr = VM32_FROM_BITS (VM32_VAND_VX (VM32_BITS (s), 0xfffff000u,
					       vl));
  vm32_f_t c = VM32_VFDIV_VV (
    VM32_VFNMSAC_VV (t, wtr, wtr, vl),
    VM32_VFADD_VV (s, wtr, vl), vl);
  vm32_f_t sr = VM32_VFMUL_VV (s, r, vl);
  /* |x| <= 0.975:  pio4_hi - (p2 - q2),
       p2 = 2*s*r - (pio2_lo - 2*c),  q2 = pio4_hi - 2*w.  */
  vm32_f_t x2 = VM32_VFSUB_VV (VM32_VFMV_V_F (VM32I_PIO2_LO, vl),
			       VM32_VFMUL_VF (c, 2.0f, vl), vl);
  vm32_f_t p2 = VM32_VFSUB_VV (VM32_VFMUL_VF (sr, 2.0f, vl), x2, vl);
  vm32_f_t q2 = VM32_VFNMSAC_VF (VM32_VFMV_V_F (VM32I_PIO4_HI, vl),
				 2.0f, wtr, vl);
  vm32_f_t hmid, emid;
  VM32I_TWO_SUM (hmid, emid, VM32_VFMV_V_F (VM32I_PIO4_HI, vl),
		 VM32_VFNEG_V (VM32_VFSUB_VV (p2, q2, vl), vl), vl);
  /* |x| > 0.975:  pio2_hi - (2*(s + s*r) - pio2_lo).  */
  vm32_f_t st = VM32_VFMACC_VV (s, s, r, vl);	/* s + s*r */
  vm32_f_t y = VM32_VFSUB_VV (VM32_VFMUL_VF (st, 2.0f, vl),
			      VM32_VFMV_V_F (VM32I_PIO2_LO, vl), vl);
  vm32_f_t htop, etop;
  VM32I_TWO_SUM (htop, etop, VM32_VFMV_V_F (VM32I_PIO2_HI, vl),
		 VM32_VFNEG_V (y, vl), vl);

  vm32_b_t mhi = VM32_VMNOT_M (mlo, vl);
  vm32_f_t h = VM32_VFMERGE_VVM (hmid, htop, mtop, vl);
  vm32_f_t e = VM32_VFMERGE_VVM (emid, etop, mtop, vl);
  *ho = VM32_VFMERGE_VVM (hlo, h, mhi, vl);
  *lo = VM32_VFMERGE_VVM (elo, e, mhi, vl);
}

static inline vm32_f_t
VM32_KFN (asin) (vm32_f_t x, size_t vl)
{
  vm32_u_t ix = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffU, vl);
  /* Wide path: |x| >= 1 (NaN/Inf/domain errors/+-1) and |x| < 2^-27
     (fdlibm's shortcut; a binary32 x*x would raise UNDERFLOW there).  */
  vm32_b_t mbad = VM32_VMSGEU_VX (ix, 0x3f800000u, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSLTU_VX (ix, 0x32000000u, vl), vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.25f, mbad, vl);

  vm32_f_t h, l;
  VM32_KFN (invtrig_asin2) (VM32_VFABS_V (xs, vl), ix, &h, &l, vl);
  h = VM32_VFSGNJX_VV (h, xs, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t w = VM32_NARROW (VM32_D64 (asin) (VM32_WIDEN (x, vl), vl),
				vl);
      h = VM32_VFMERGE_VVM (h, w, mbad, vl);
    }
  return h;
}

static inline vm32_f_t
VM32_KFN (asinpi) (vm32_f_t x, size_t vl)
{
  vm32_u_t ix = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffU, vl);
  vm32_b_t mbad = VM32_VMSGEU_VX (ix, 0x3f800000u, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSLTU_VX (ix, 0x32000000u, vl), vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.25f, mbad, vl);

  /* the sanitized value must also drive the ix-based branch masks.  */
  vm32_u_t ixs = VM32_VAND_VX (VM32_BITS (xs), 0x7fffffffU, vl);
  vm32_f_t h, l;
  VM32_KFN (invtrig_asin2) (VM32_VFABS_V (xs, vl), ixs, &h, &l, vl);
  h = VM32_VFSGNJX_VV (h, xs, vl);
  l = VM32_VFSGNJX_VV (l, xs, vl);
  h = VM32_KFN (invtrig_divpi) (h, l, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t w = VM32_NARROW (VM32_D64 (asinpi) (VM32_WIDEN (x, vl), vl),
				vl);
      h = VM32_VFMERGE_VVM (h, w, mbad, vl);
    }
  return h;
}

/* acos(x) for x in (-1, 1), carried as (h, l).  */
static inline void
VM32_KFN (invtrig_acos2) (vm32_f_t x, vm32_u_t ix, vm32_f_t *ho,
			  vm32_f_t *lo, size_t vl)
{
  vm32_b_t mlo = VM32_VMSLTU_VX (ix, 0x3f000000u, vl);	/* |x| < 0.5 */
  vm32_b_t mneg = VM32_VMSLT_VX (VM32_REINT_U_I (VM32_BITS (x)), 0, vl);
  vm32_f_t ax = VM32_VFABS_V (x, vl);
  vm32_f_t zmid = VM32_VFMUL_VF (VM32_VFRSUB_VF (ax, 1.0f, vl), 0.5f, vl);
  vm32_f_t z = VM32_VFMERGE_VVM (zmid, VM32_VFMUL_VV (x, x, vl), mlo, vl);
  vm32_f_t r = VM32_KFN (invtrig_asin_r) (z, vl);

  /* |x| < 0.5:  pio2_hi - (x - (pio2_lo - x*r)).  */
  vm32_f_t v1 = VM32_VFNMSAC_VV (VM32_VFMV_V_F (VM32I_PIO2_LO, vl), x, r,
				 vl);	/* pio2_lo - x*r */
  vm32_f_t hlo2, elo2;
  VM32I_TWO_SUM (hlo2, elo2, VM32_VFMV_V_F (VM32I_PIO2_HI, vl),
		 VM32_VFNEG_V (VM32_VFSUB_VV (x, v1, vl), vl), vl);

  vm32_f_t s = VM32_VFSQRT_V (zmid, vl);
  vm32_f_t df = VM32_FROM_BITS (VM32_VAND_VX (VM32_BITS (s), 0xfffff000u,
					      vl));
  vm32_f_t c = VM32_VFDIV_VV (
    VM32_VFNMSAC_VV (zmid, df, df, vl),
    VM32_VFADD_VV (s, df, vl), vl);
  vm32_f_t sr = VM32_VFMUL_VV (s, r, vl);
  /* x > 0.5:  2*(df + (r*s + c)).  */
  vm32_f_t w1 = VM32_VFADD_VV (sr, c, vl);
  vm32_f_t hp, ep;
  VM32I_TWO_SUM (hp, ep, df, w1, vl);
  hp = VM32_VFMUL_VF (hp, 2.0f, vl);
  ep = VM32_VFMUL_VF (ep, 2.0f, vl);
  /* x < -0.5:  pi - 2*(s + (r*s - pio2_lo)).  */
  vm32_f_t w2 = VM32_VFMSAC_VV (VM32_VFMV_V_F (VM32I_PIO2_LO, vl), r, s,
				vl);	/* r*s - pio2_lo */
  vm32_f_t hn, en;
  VM32I_TWO_SUM (hn, en, s, w2, vl);
  hn = VM32_VFMUL_VF (hn, 2.0f, vl);
  en = VM32_VFMUL_VF (en, 2.0f, vl);
  vm32_f_t hn2, en2;
  VM32I_TWO_SUM (hn2, en2, VM32_VFMV_V_F (VM32I_PI_HI, vl),
		 VM32_VFNEG_V (hn, vl), vl);
  en2 = VM32_VFADD_VV (VM32_VFSUB_VV (en2, en, vl),
		       VM32_VFMV_V_F (VM32I_PI_LO, vl), vl);

  vm32_f_t h = VM32_VFMERGE_VVM (hp, hn2, mneg, vl);
  vm32_f_t e = VM32_VFMERGE_VVM (ep, en2, mneg, vl);
  vm32_b_t mhi = VM32_VMNOT_M (mlo, vl);
  *ho = VM32_VFMERGE_VVM (hlo2, h, mhi, vl);
  *lo = VM32_VFMERGE_VVM (elo2, e, mhi, vl);
}

static inline vm32_f_t
VM32_KFN (acos) (vm32_f_t x, size_t vl)
{
  vm32_u_t ix = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffU, vl);
  /* Wide path: |x| >= 1 and |x| < 2^-57 (fdlibm's constant shortcut).  */
  vm32_b_t mbad = VM32_VMSGEU_VX (ix, 0x3f800000u, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSLTU_VX (ix, 0x23000000u, vl), vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.25f, mbad, vl);

  vm32_u_t ixs = VM32_VAND_VX (VM32_BITS (xs), 0x7fffffffU, vl);
  vm32_f_t h, l;
  VM32_KFN (invtrig_acos2) (xs, ixs, &h, &l, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t w = VM32_NARROW (VM32_D64 (acos) (VM32_WIDEN (x, vl), vl),
				vl);
      h = VM32_VFMERGE_VVM (h, w, mbad, vl);
    }
  return h;
}

/* NATIVE binary32 fast path.  acospi(x) = acos(x)/pi = 0.5 - asin(x)/pi,
   and the cancelled form is safe here because both halves are evaluated
   where the *pair* is accurate, not where the plain float is:

     |x| < 0.5   asin(x) = x + x*r.  The subtraction from 1/2 would be
                 cancellation-free anyway, and the rational's error sits
                 in the small x*r term, so 0.5 - (x + x*r)/pi is good to
                 2^-45 before the final add.
     |x| >= 0.5  acos(|x|) = 2*(df + (r*s + c)), the acos kernel's sqrt
                 branch, whose error is relative to acos itself and the
                 sqrt's rounding is recovered by c.  x < 0 then uses
                 acos(x) = pi - acos(|x|) as 1 - u: the 1 is exact, so
                 the error stays absolute instead of being re-rounded
                 through a pi-sized value.

   Neither pair is ever re-rounded before the last add: the offset (1/2
   or 0/1) is folded into the same fma that multiplies by the 1/pi or
   2/pi hi word, and the lo word plus the second addend enter the
   remainder, so exactly one rounding is left:

     s = RN(d +- h*K_hi),  e = ((d - s) +- h*K_hi) +- h*K_lo +- l*K_hi
     result = RN(s + e).

   Measured on the FULL blob: max_ulp 0.5719, mean 0.1438, p99.9 0.5097
   (the same run's plain-f64 path scored 0.5000/0.1437/0.4984), against
   1.63 for the asin pair fed straight through 0.5 - u and 1.14 for the
   acos pair over pi -- i.e. the double-float handling is what makes the
   cancelled form viable.

   Covered domain: |x| in [2^-27, 1).  The lower bound is asin's: below
   it a binary32 x*x raises a spurious UNDERFLOW, and acospi(x) is
   exactly 0.5 there anyway, so the plain-f64 path (which returns 0.5
   for every such lane) is used.  |x| >= 1 -- +-1, the domain-error
   lanes and every Inf/NaN -- is likewise recomputed per lane by the
   binary64 kernel, so special values, signed zeros and the flag
   semantics are inherited bit-for-bit.  */
static inline vm32_f_t
VM32_KFN (acospi) (vm32_f_t x, size_t vl)
{
  vm32_u_t ix = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffU, vl);
  vm32_b_t mbad = VM32_VMSGEU_VX (ix, 0x3f800000u, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSLTU_VX (ix, 0x32000000u, vl), vl);
  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.25f, mbad, vl);

  /* the sanitized value must also drive the ix-based branch masks.  */
  vm32_u_t ixs = VM32_VAND_VX (VM32_BITS (xs), 0x7fffffffU, vl);
  vm32_b_t mlo = VM32_VMSLTU_VX (ixs, 0x3f000000u, vl);	/* |x| < 0.5 */
  vm32_b_t mneg = VM32_VMSLT_VX (VM32_REINT_U_I (VM32_BITS (xs)), 0, vl);
  vm32_f_t a = VM32_VFABS_V (xs, vl);
  vm32_f_t t = VM32_VFMUL_VF (VM32_VFRSUB_VF (a, 1.0f, vl), 0.5f, vl);
  vm32_f_t z = VM32_VFMERGE_VVM (t, VM32_VFMUL_VV (a, a, vl), mlo, vl);
  vm32_f_t r = VM32_KFN (invtrig_asin_r) (z, vl);

  /* |x| < 0.5: 0.5 - x/pi - (x*r)/pi.  s1 = RN(0.5 - x*(1/pi)_hi) is the
     only rounded term; z was a*a and x carries the sign, so the two
     remainders are the exact ones.  */
  vm32_f_t xr = VM32_VFMUL_VV (xs, r, vl);
  vm32_f_t s1 = VM32_VFNMSAC_VF (VM32_VFMV_V_F (0.5f, vl), VM32I_INVPI_HI,
				 xs, vl);
  vm32_f_t e1 = VM32_VFNMSAC_VF (
    VM32_VFSUB_VV (VM32_VFMV_V_F (0.5f, vl), s1, vl), VM32I_INVPI_HI, xs, vl);
  e1 = VM32_VFNMSAC_VF (e1, VM32I_INVPI_LO, xs, vl);
  e1 = VM32_VFNMSAC_VF (e1, VM32I_INVPI_HI, xr, vl);
  vm32_f_t us = VM32_VFADD_VV (s1, e1, vl);

  /* |x| >= 0.5: acos(a) = 2*(df + w1); df is the sqrt truncated to
     fdlibm's 12 cleared mantissa bits and c puts its rounding back.  */
  vm32_f_t s = VM32_VFSQRT_V (t, vl);
  vm32_f_t df = VM32_FROM_BITS (VM32_VAND_VX (VM32_BITS (s), 0xfffff000u,
					      vl));
  vm32_f_t c = VM32_VFDIV_VV (
    VM32_VFNMSAC_VV (t, df, df, vl), VM32_VFADD_VV (s, df, vl), vl);
  vm32_f_t w1 = VM32_VFADD_VV (VM32_VFMUL_VV (s, r, vl), c, vl);
  vm32_f_t hb, eb;
  VM32I_TWO_SUM (hb, eb, df, w1, vl);
  /* acos(|x|) = 2*(df + w1), so acospi = (df + w1)*(2/pi): the pair is
     scaled by the 2/pi hi+lo words (the factor two never rounds) and the
     offset d = 1 for x < 0 (1 - acos(|x|)/pi), 0 otherwise is folded
     into the same fma; the sign of x rides on the pair.  */
  hb = VM32_VFSGNJX_VV (hb, xs, vl);
  eb = VM32_VFSGNJX_VV (eb, xs, vl);
  vm32_f_t d = VM32_VFMERGE_VFM (VM32_VFMV_V_F (0.0f, vl), 1.0f, mneg, vl);
  vm32_f_t s2 = VM32_VFMACC_VF (d, VM32I_TWOPI_HI, hb, vl);
  vm32_f_t e2 = VM32_VFMACC_VF (VM32_VFSUB_VV (d, s2, vl), VM32I_TWOPI_HI,
				hb, vl);
  e2 = VM32_VFMACC_VF (e2, VM32I_TWOPI_LO, hb, vl);
  e2 = VM32_VFMACC_VF (e2, VM32I_TWOPI_HI, eb, vl);
  vm32_f_t ub = VM32_VFADD_VV (s2, e2, vl);

  vm32_f_t h = VM32_VFMERGE_VVM (ub, us, mlo, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t w = VM32_NARROW (VM32_D64 (acospi) (VM32_WIDEN (x, vl), vl),
				vl);
      h = VM32_VFMERGE_VVM (h, w, mbad, vl);
    }
  return h;
}

/* ==================================================================== */
/* atan2                                                                */
/* ==================================================================== */

/* NATIVE binary32 fast path, ef_atan2.c's shape -- z = atan(|y/x|) plus
   the fdlibm quadrant fixup -- with the *quotient's* rounding recovered
   from its exact fma remainder (the "divide, then correct with the exact
   remainder" step of the research notes' T7/T8):

     q = RN(|y|/|x|),   rem = |y| - q*|x|        (exact, one fma)
     atan(|y|/|x|) = atan(q) + rem/(|x|(1+q^2)) + O((rem/|x|)^2)

   with |x|(1+q^2) = |x| + |y|*q formed by one fma.  The derivative
   1/(1+q^2) peaks at |q| = 1, so without the correction the ratio's own
   rounding costs up to 1.42 ulp; with it the only error left is
   atan(q)'s fdlibm-float ~0.6-0.8 ulp (the same error atanf carries) and
   the single final hi+lo add.  Replayed over the FULL blob's fast lanes
   this scores max_ulp 0.72 (1.42 without the correction).

   atan(q) is invtrig_atan_pos2, the double-float f32 atan kernel atanf
   already uses: (h, l) with h + l = atan(q) to ~2^-45.

   Covered domain; everything else is evaluated by the per-lane f64 path
   below (never per strip: a lane's value must not depend on its
   neighbours, which the length/alignment sweeps require):
     * both operands finite and inside [2^-29, 10.5)  (excludes +-0,
       subnormals, the poles, +-Inf/NaN and every exact special result);
     * |y/x| inside [2^-18, 2^18], tested as an integer difference of the
       magnitude bit patterns before any division.  The lower bound keeps
       atan_pos2's r^2/r^4 chain normal (no spurious UNDERFLOW); the upper
       bound keeps the pi - atan(|y/x|) cancellation in the x < 0 branch
       far below an ulp of the (tiny) result.  */

/* vfrec7.v -- the 7-bit reciprocal estimate.  Raw intrinsic: f32/compat.h
   does not wrap the estimate instructions.  */
#define VM32I_REC7(a, vl) VM32_XP2 (__riscv_vfrec7_v_f32m, VM32_LMUL) (a, vl)

/* The fast-domain test shared by atan2 and atan2pi: `mbad` marks every
   lane the native ratio/quadrant path cannot take, and (xs, ys) are the
   benign stand-ins the fast path is fed with (2^-1 for both, a finite
   pair whose fast-path evaluation is exact but for the kernel's own
   rounding, so a bad lane cannot raise a flag).
   Covered: both operands finite and inside [2^-29, 10.5) and
   |y/x| inside [2^-18, 2^18], tested as an integer difference of the
   magnitude bit patterns before any division.  The lower bound keeps
   atan_pos2's r^2/r^4 chain normal (no spurious UNDERFLOW); the upper
   bound keeps the pi - atan(|y/x|) cancellation in the x < 0 branch far
   below an ulp of the (tiny) result; the lower bound on the ratio keeps
   the same cancellation bounded for the y-side.  */
#define VM32I_ATAN2_PREP(mbado, xso, yso, y, x, vl) do {		\
  vm32_u_t vap__ix = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffU, (vl));	\
  vm32_u_t vap__iy = VM32_VAND_VX (VM32_BITS (y), 0x7fffffffU, (vl));	\
  vm32_b_t vap__m = VM32_VMSGEU_VX (vap__ix, 0x41280000u, (vl));	\
  vap__m = VM32_VMOR_MM (vap__m,					\
			 VM32_VMSGEU_VX (vap__iy, 0x41280000u, (vl)), (vl)); \
  vap__m = VM32_VMOR_MM (vap__m,					\
			 VM32_VMSLTU_VX (vap__ix, 0x31000000u, (vl)), (vl)); \
  vap__m = VM32_VMOR_MM (vap__m,					\
			 VM32_VMSLTU_VX (vap__iy, 0x31000000u, (vl)), (vl)); \
  vm32_i_t vap__d = VM32_REINT_U_I (					\
    VM32_VSUB_VV (vap__iy, vap__ix, (vl)));				\
  vap__m = VM32_VMOR_MM (vap__m, VM32_VMSGE_VX (vap__d, 0x08800000, (vl)), \
			 (vl));						\
  vap__m = VM32_VMOR_MM (vap__m, VM32_VMSLE_VX (vap__d, -0x08800000, (vl)), \
			 (vl));						\
  (mbado) = vap__m;							\
  (xso) = VM32_VFMERGE_VFM ((x), 0.5f, (mbado), (vl));			\
  (yso) = VM32_VFMERGE_VFM ((y), 0.5f, (mbado), (vl));			\
} while (0)

/* atan2's native fast path for sanitized (ys, xs), as a hi/lo pair:
   z = atan(|y/x|) plus the fdlibm quadrant fixup.  The *quotient's*
   rounding is recovered from its exact fma remainder (the "divide, then
   correct with the exact remainder" step):

     q = RN(|y|/|x|),   rem = |y| - q*|x|        (exact, one fma)
     atan(|y|/|x|) = atan(q) + rem/(|x|(1+q^2)) + O((rem/|x|)^2)

   with |x|(1+q^2) = |x| + |y|*q formed by one fma.  The derivative
   1/(1+q^2) peaks at |q| = 1, so without the correction the ratio's own
   rounding costs up to 1.42 ulp; with it the only error left is
   atan(q)'s fdlibm-float ~0.6-0.8 ulp (the same error atanf carries) and
   the single final hi+lo add.  Replayed over the FULL blob's fast lanes
   this scores max_ulp 0.72 (1.42 without the correction).

   atan(q) is invtrig_atan_pos2, the double-float f32 atan kernel atanf
   already uses: (h, l) with h + l = atan(q) to ~2^-45.  The caller adds
   the two words for atan2, or divides the pair by pi for atan2pi.  */
static inline void
VM32_KFN (invtrig_atan2_pos) (vm32_f_t ys, vm32_f_t xs, vm32_f_t *ho,
			      vm32_f_t *lo, size_t vl)
{
  vm32_f_t ax = VM32_VFABS_V (xs, vl);
  vm32_f_t ay = VM32_VFABS_V (ys, vl);

  /* q = |y|/|x| and its exact remainder; the derivative's denominator.
     The correction is a <= 0.7 ulp term, so its reciprocal only needs a
     few percent of relative accuracy: vfrec7 (the 7-bit estimate, one
     ~40x cheaper instruction than vfdiv.vv) is used directly, which
     costs <= 0.006 ulp of error and saves a second divide per lane.  */
  vm32_f_t q = VM32_VFDIV_VV (ay, ax, vl);
  vm32_f_t rem = VM32_VFNMSAC_VV (ay, q, ax, vl);	/* |y| - q*|x| */
  vm32_f_t der = VM32_VFMACC_VV (ax, ay, q, vl);	/* |x|(1+q^2) */
  vm32_f_t corr = VM32_VFMUL_VV (rem, VM32I_REC7 (der, vl), vl);

  vm32_f_t h, l;
  VM32_KFN (invtrig_atan_pos2) (q, &h, &l, vl);
  l = VM32_VFADD_VV (l, corr, vl);

  /* x < 0:  atan2 = pi - atan(q) == (pi_hi - h) + ((ep - l) + pi_lo).  */
  vm32_f_t hp, ep;
  VM32I_TWO_SUM (hp, ep, VM32_VFMV_V_F (VM32I_PI_HI, vl),
		 VM32_VFNEG_V (h, vl), vl);
  ep = VM32_VFADD_VF (VM32_VFSUB_VV (ep, l, vl), VM32I_PI_LO, vl);

  vm32_b_t msx = VM32_VMSLT_VX (VM32_REINT_U_I (VM32_BITS (xs)), 0, vl);
  vm32_f_t oh = VM32_VFMERGE_VVM (h, hp, msx, vl);
  vm32_f_t ol = VM32_VFMERGE_VVM (l, ep, msx, vl);
  /* sign of y, applied to the hi and lo words alike (vfsgnjx).  */
  *ho = VM32_VFSGNJX_VV (oh, ys, vl);
  *lo = VM32_VFSGNJX_VV (ol, ys, vl);
}

/* The wide fallback as a noinline call: the binary64 kernel (plus its
   double-double atan) dwarfs the fast path, and inlined it would sit
   inside the strip loop -- which *is* the whole function -- costing fetch
   bandwidth on every iteration for a branch that is almost never taken.
   atan2pi keeps its wide path inlined instead (a vector-ABI call costs
   more on the VLEN-128 board than the extra code does).  */
__attribute__ ((noinline)) static vm32_f_t
VM32_KFN (atan2_wide) (vm32_f_t y, vm32_f_t x, size_t vl)
{
  return VM32_NARROW (VM32_D64 (atan2) (VM32_WIDEN (y, vl),
					 VM32_WIDEN (x, vl), vl), vl);
}

/* atan2 keeps the fast-path body spelled out here rather than through
   VM32I_ATAN2_PREP: the same op sequence measured 6% faster on muse-pi
   when it was left exactly as the entry has always compiled (the codegen
   is layout-sensitive to a single instruction).  */
static inline vm32_f_t
VM32_KFN (atan2) (vm32_f_t y, vm32_f_t x, size_t vl)
{
  vm32_u_t ix = VM32_VAND_VX (VM32_BITS (x), 0x7fffffffU, vl);
  vm32_u_t iy = VM32_VAND_VX (VM32_BITS (y), 0x7fffffffU, vl);
  vm32_b_t mbad = VM32_VMSGEU_VX (ix, 0x41280000u, vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSGEU_VX (iy, 0x41280000u, vl), vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSLTU_VX (ix, 0x31000000u, vl), vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSLTU_VX (iy, 0x31000000u, vl), vl);
  vm32_i_t d = VM32_REINT_U_I (VM32_VSUB_VV (iy, ix, vl));
  mbad = VM32_VMOR_MM (mbad, VM32_VMSGE_VX (d, 0x08800000, vl), vl);
  mbad = VM32_VMOR_MM (mbad, VM32_VMSLE_VX (d, -0x08800000, vl), vl);

  vm32_f_t xs = VM32_VFMERGE_VFM (x, 0.5f, mbad, vl);
  vm32_f_t ys = VM32_VFMERGE_VFM (y, 0.5f, mbad, vl);
  vm32_f_t ax = VM32_VFABS_V (xs, vl);
  vm32_f_t ay = VM32_VFABS_V (ys, vl);

  /* q = |y|/|x| and its exact remainder; the derivative's denominator.
     The correction is a <= 0.7 ulp term, so its reciprocal only needs a
     few percent of relative accuracy: vfrec7 (the 7-bit estimate, one
     ~40x cheaper instruction than vfdiv.vv) is used directly, which
     costs <= 0.006 ulp of error and saves a second divide per lane.  */
  vm32_f_t q = VM32_VFDIV_VV (ay, ax, vl);
  vm32_f_t rem = VM32_VFNMSAC_VV (ay, q, ax, vl);	/* |y| - q*|x| */
  vm32_f_t der = VM32_VFMACC_VV (ax, ay, q, vl);	/* |x|(1+q^2) */
  vm32_f_t corr = VM32_VFMUL_VV (rem, VM32I_REC7 (der, vl), vl);

  vm32_f_t h, l;
  VM32_KFN (invtrig_atan_pos2) (q, &h, &l, vl);
  l = VM32_VFADD_VV (l, corr, vl);

  /* x < 0:  atan2 = pi - atan(q) == (pi_hi - h) + ((ep - l) + pi_lo).  */
  vm32_f_t hp, ep;
  VM32I_TWO_SUM (hp, ep, VM32_VFMV_V_F (VM32I_PI_HI, vl),
		 VM32_VFNEG_V (h, vl), vl);
  ep = VM32_VFADD_VF (VM32_VFSUB_VV (ep, l, vl), VM32I_PI_LO, vl);

  vm32_b_t msx = VM32_VMSLT_VX (VM32_REINT_U_I (VM32_BITS (x)), 0, vl);
  vm32_f_t oh = VM32_VFMERGE_VVM (h, hp, msx, vl);
  vm32_f_t ol = VM32_VFMERGE_VVM (l, ep, msx, vl);
  /* sign of y, applied to the hi and lo words alike (vfsgnjx).  */
  oh = VM32_VFSGNJX_VV (oh, ys, vl);
  ol = VM32_VFSGNJX_VV (ol, ys, vl);
  vm32_f_t r = VM32_VFADD_VV (oh, ol, vl);

  if (VM32_VFIRST_M (mbad, vl) >= 0)
    r = VM32_VFMERGE_VVM (r, VM32_KFN (atan2_wide) (y, x, vl), mbad, vl);
  return r;
}

/* atan2pi: the same native f32 fast path as atan2, with the final
   divide by pi folded into the existing (h, l) pair by invtrig_divpi, so
   the scaling costs one rounding on top of the kernel's own (the same
   shape atanpi uses; measured 0.80 ulp over the FULL blob, against 0.72
   for atan2).  The fallback stays inlined (a vector-ABI call is a large
   loss on the VLEN-128 board).  */
static inline vm32_f_t
VM32_KFN (atan2pi) (vm32_f_t y, vm32_f_t x, size_t vl)
{
  vm32_b_t mbad;
  vm32_f_t xs, ys;
  VM32I_ATAN2_PREP (mbad, xs, ys, y, x, vl);
  vm32_f_t oh, ol;
  VM32_KFN (invtrig_atan2_pos) (ys, xs, &oh, &ol, vl);
  vm32_f_t h = VM32_KFN (invtrig_divpi) (oh, ol, vl);
  if (VM32_VFIRST_M (mbad, vl) >= 0)
    {
      vm32_f_t w = VM32_NARROW (VM32_D64 (atan2pi) (VM32_WIDEN (y, vl),
						   VM32_WIDEN (x, vl), vl),
				vl);
      h = VM32_VFMERGE_VVM (h, w, mbad, vl);
    }
  return h;
}
