/* k_hyper.inc -- sinh / cosh / tanh kernels.

   LMUL-generic template: include after vecmath_rvvcompat.h,
   vecmath_dd.h and vecmath_local.h (k_fixup.inc) with VM_LMUL defined.
   Also needs k_exp.inc instantiated at the same LMUL (VM_EXP_DD and
   VM_KFN(exp)).

   Structure: a fast path valid on |x| < 22 plus a per-strip fallback for
   the rare lanes outside it (|x| >= 22, the overflow boundary, +-Inf and
   NaN).  The fallback is computed only when a strip contains such a lane
   (one vfirst), and the result is merged per lane, so the value of a lane
   never depends on the other lanes of its strip: stripmine/VL sweeps stay
   bitwise identical.

   Fast path (always accurate, ~0.5-0.7 ulp):
     e = exp(|x|) as a double-double (VM_EXP_DD, ~2^-60 relative),
     1/e by one division plus the residual,
     sinh = +-0.5*(e - 1/e), cosh = 0.5*(e + 1/e), tanh = (e-1/e)/(e+1/e).
   The dd forms are needed because the plain-double versions amplify the
   0.5 ulp of exp() into 1.5-3 ulp (the dd pair costs only ~15 vector
   instructions over the plain exp).  |x| < 2^-28 (sinh) / < 2^-55 (tanh)
   return x, and the |x| below 0.125 (sinh) / 0.05 (tanh) small range uses
   the Taylor polynomial, because e-1/e loses the leading digits there.

   The |x| >= 22 branch matrix keeps fdlibm's 10.5 ulp-free recipes:
     22 <= |x| < log(DBL_MAX):      0.5*exp(|x|)
     log(DBL_MAX) <= |x| < ofl:     0.5*exp(|x|) via exp(x-1024*ln2)*2^1023
     |x| >= ofl (710.4758600739439): overflow, +-Inf + OVERFLOW
     +-Inf / NaN:                    exact classes, sNaN invalid
   tanh is +-1 (or NaN) there.  */

/* Region edges, as binary64 bit patterns (all inputs are |x| >= 0, and
   positive-double bit patterns order like the values).  */
#define VM_HYP_22	0x4036000000000000ULL	/* 22.0 */
#define VM_HYP_2M28	0x3e30000000000000ULL	/* 2^-28 */
#define VM_HYP_2M300	0x2d30000000000000ULL	/* 2^-300 */
#define VM_HYP_SINH_SM	0x3fc0000000000000ULL	/* 0.125 */
#define VM_HYP_TANH_SM	0x3fa999999999999aULL	/* 0.05 */
#define VM_HYP_LOGE	0x40862e42fefa39efULL	/* log(DBL_MAX) */
#define VM_HYP_OFL	0x408633ce8fb9f87eULL	/* 22+overflow threshold */
#define VM_HYP_INF	0x7ff0000000000000ULL
/* 1024*ln2 split: c1 = 1024*ln2 rounded to double, c2 = the residual.
   c1 is exact (power-of-two scale of the double ln2).  */
#define VM_HYP_LN2K1	709.78271289338397310
#define VM_HYP_LN2K2	2.3747039373786107e-14
#define VM_HYP_2P1023	0x1p1023		/* 2^1023 */

/* (oh, ol) = 1/(eh + el), ~2^-105 relative; |eh| >= 1 so no division by
   zero or overflow.  */
#define VM_HYP_INVDD(oh, ol, eh, el, vl) do {				\
  vm_f_t vhi_r;								\
  VM_RCP_FULL (vhi_r, (eh), (vl));					\
  vm_f_t vhi_e = VM_VFNMSAC_VV (VM_VFMV_V_F (1.0, (vl)), vhi_r, (eh), (vl)); \
  (oh) = vhi_r;								\
  (ol) = VM_VFMUL_VV (							\
    VM_VFNMSAC_VV (vhi_e, vhi_r, (el), (vl)), vhi_r, (vl));		\
} while (0)

/* (qh, ql) = (nh + nl)/(dh + dl), ~2^-105 relative, one division
   (reciprocal of dh refined by the exact residual; dh > 0, no div by
   zero).  */
#define VM_HYP_DIVDD(qh, ql, nh, nl, dh, dl, vl) do {			\
  vm_f_t vhq_r;								\
  VM_RCP_FULL (vhq_r, (dh), (vl));					\
  vm_f_t vhq_e = VM_VFNMSAC_VV (VM_VFMV_V_F (1.0, (vl)), vhq_r, (dh), (vl)); \
  vhq_r = VM_VFMACC_VV (vhq_r, vhq_r, vhq_e, (vl));			\
  (qh) = VM_VFMUL_VV ((nh), vhq_r, (vl));				\
  {									\
    vm_f_t vhq_s = VM_VFNMSAC_VV ((nh), (qh), (dh), (vl));		\
    vhq_s = VM_VFADD_VV (vhq_s, (nl), (vl));				\
    vhq_s = VM_VFNMSAC_VV (vhq_s, (qh), (dl), (vl));			\
    (ql) = VM_VFMUL_VV (vhq_s, vhq_r, (vl));				\
  }									\
} while (0)

/* ---- sinh ------------------------------------------------------- */

static inline vm_f_t
VM_KFN (sinh_special) (vm_f_t x, vm_f_t ax, vm_f_t h, vm_f_t sg, vm_f_t z,
		       vm_b_t m22, size_t vl)
{
  vm_u64_t ab = VM_BITS (ax);
  vm_b_t mov = VM_VMSGEU_VX (ab, VM_HYP_OFL, vl);
  vm_b_t mnf = VM_VMSGEU_VX (ab, VM_HYP_INF, vl);
  vm_b_t mband = VM_VMAND_MM (VM_VMNOT_M (mov, vl),
			      VM_VMSGTU_VX (ab, VM_HYP_LOGE, vl), vl);
  /* 22 <= |x| < log(DBL_MAX): 0.5*exp(|x|).  */
  vm_f_t axa = VM_VFMERGE_VFM (ax, 1.0,
			       VM_VMOR_MM (mband, VM_VMOR_MM (mov, mnf, vl),
					   vl), vl);
  /* Region A: 0.5*exp(|x|) (the h scale, sign included).  */
  vm_f_t zp = VM_VFMUL_VV (h, VM_KFN (exp) (axa, vl, exp), vl);
  /* log(DBL_MAX) <= |x| < overflow: 0.5*exp(x) evaluated as
     exp(x - 1024*ln2) * 2^1023 so that the final scale is exact.  */
  if (VM_VFIRST_M (mband, vl) >= 0)
    {
      vm_f_t b = VM_VFSUB_VF (ax, VM_HYP_LN2K1, vl);
      b = VM_VFSUB_VF (b, VM_HYP_LN2K2, vl);
      b = VM_VFMERGE_VFM (b, 0.0, VM_VMNOT_M (mband, vl), vl);
      vm_f_t zb = VM_VFMUL_VF (VM_KFN (exp) (b, vl, exp), VM_HYP_2P1023, vl);
      zp = VM_VFMERGE_VVM (zp, VM_VFMUL_VV (sg, zb, vl), mband, vl);
    }
  /* overflow: x*shuge raises OVERFLOW on those lanes only.  */
  if (VM_VFIRST_M (mov, vl) >= 0)
    zp = VM_VFMERGE_VVM (
      zp, VM_VFMUL_VV_M (mov, x, VM_VFMV_V_F (1.0e307, vl), vl), mov, vl);
  /* +-Inf -> +-Inf, NaN -> NaN (invalid only for sNaN).  */
  zp = VM_VFMERGE_VVM (zp, VM_VFADD_VV_M (mnf, x, x, vl), mnf, vl);
  return VM_VFMERGE_VVM (z, zp, m22, vl);
}

static inline vm_f_t
VM_KFN (sinh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_b_t m22 = VM_VMSGEU_VX (ix, 0x40360000u, vl);
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3e300000u, vl);	/* |x| < 2^-28 */
  vm_b_t msmall = VM_VMSLTU_VX (ix, 0x3fc00000u, vl);	/* |x| < 0.125 */
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl);
  vm_f_t ax = VM_VFABS_V (x, vl);
  vm_f_t axs = VM_VFMERGE_VFM (ax, 1.0, m22, vl);
  vm_f_t h = VM_VFMERGE_VFM (VM_VFMV_V_F (0.5, vl), -0.5, mneg, vl);
  vm_f_t sg = VM_VFMERGE_VFM (VM_VFMV_V_F (1.0, vl), -1.0, mneg, vl);
  vm_f_t z;
  {
    /* 0.5*(e - 1/e), e = exp(|x|) to ~2^-60 relative.  */
    vm_f_t eh, el, ih, il;
    VM_EXP_DD (eh, el, axs, vl, exp);
    VM_HYP_INVDD (ih, il, eh, el, vl);
    vm_f_t s = VM_VFSUB_VV (eh, ih, vl);
    vm_f_t re = VM_VFSUB_VV (VM_VFSUB_VV (eh, s, vl), ih, vl);
    vm_f_t dl = VM_VFADD_VV (VM_VFSUB_VV (re, il, vl), el, vl);
    z = VM_VFMUL_VV (h, VM_VFADD_VV (s, dl, vl), vl);
  }
  {
    /* |x| < 0.125: x + x*t*P(t), t = x^2.  Only the guard lanes and
       |x| < 2^-300 (whose x^2 would underflow) need a benign input.  */
    vm_b_t mden = VM_VMSLTU_VX (ix, 0x2d300000u, vl);
    vm_f_t axp = VM_VFMERGE_VFM (axs, 1.0,
				 VM_VMOR_MM (mden, m22, vl), vl);
    vm_f_t t2 = VM_VFMUL_VV (axp, axp, vl);
    vm_f_t p = VM_VFADD_VF (
      VM_VFMUL_VF (t2, 1.0 / 362880.0, vl), 1.0 / 5040.0, vl);
    p = VM_VFADD_VF (VM_VFMUL_VV (p, t2, vl), 1.0 / 120.0, vl);
    p = VM_VFADD_VF (VM_VFMUL_VV (p, t2, vl), 1.0 / 6.0, vl);
    vm_f_t rp = VM_VFMACC_VV (axp, VM_VFMUL_VV (t2, p, vl), axp, vl);
    z = VM_VFMERGE_VVM (z, VM_VFMUL_VV (sg, rp, vl), msmall, vl);
  }
  z = VM_VFMERGE_VVM (z, x, mtiny, vl);
  if (VM_VFIRST_M (m22, vl) >= 0)
    z = VM_KFN (sinh_special) (x, ax, h, sg, z, m22, vl);
  return z;
}

/* ---- cosh ------------------------------------------------------- */

static inline vm_f_t
VM_KFN (cosh_special) (vm_f_t ax, vm_f_t z, vm_b_t m22, size_t vl)
{
  vm_u64_t ab = VM_BITS (ax);
  vm_b_t mov = VM_VMSGEU_VX (ab, VM_HYP_OFL, vl);
  vm_b_t mnf = VM_VMSGEU_VX (ab, VM_HYP_INF, vl);
  vm_b_t mband = VM_VMAND_MM (VM_VMNOT_M (mov, vl),
			      VM_VMSGTU_VX (ab, VM_HYP_LOGE, vl), vl);
  vm_f_t axa = VM_VFMERGE_VFM (ax, 1.0,
			       VM_VMOR_MM (mband, VM_VMOR_MM (mov, mnf, vl),
					   vl), vl);
  vm_f_t zp = VM_VFMUL_VF (VM_KFN (exp) (axa, vl, exp), 0.5, vl);
  if (VM_VFIRST_M (mband, vl) >= 0)
    {
      vm_f_t b = VM_VFSUB_VF (ax, VM_HYP_LN2K1, vl);
      b = VM_VFSUB_VF (b, VM_HYP_LN2K2, vl);
      b = VM_VFMERGE_VFM (b, 0.0, VM_VMNOT_M (mband, vl), vl);
      zp = VM_VFMERGE_VVM (
	zp, VM_VFMUL_VF (VM_KFN (exp) (b, vl, exp), VM_HYP_2P1023, vl),
	mband, vl);
    }
  if (VM_VFIRST_M (mov, vl) >= 0)
    zp = VM_VFMERGE_VVM (
      zp, VM_VFMUL_VV_M (mov, ax, VM_VFMV_V_F (1.0e307, vl), vl), mov, vl);
  zp = VM_VFMERGE_VVM (zp, VM_VFADD_VV_M (mnf, ax, ax, vl), mnf, vl);
  return VM_VFMERGE_VVM (z, zp, m22, vl);
}

static inline vm_f_t
VM_KFN (cosh) (vm_f_t x, size_t vl, double (*fixup) (double))
{
  vm_u64_t ix = VM_VAND_VX (
    VM_VSRL_VX (VM_BITS (x), 32, vl), 0x7fffffffULL, vl);
  vm_b_t m22 = VM_VMSGEU_VX (ix, 0x40360000u, vl);
  vm_f_t ax = VM_VFABS_V (x, vl);
  vm_f_t axs = VM_VFMERGE_VFM (ax, 1.0, m22, vl);
  vm_f_t z;
  {
    /* 0.5*(e + 1/e), e = exp(|x|) to ~2^-60 relative; no cancellation.  */
    vm_f_t eh, el, ih, il;
    VM_EXP_DD (eh, el, axs, vl, exp);
    VM_HYP_INVDD (ih, il, eh, el, vl);
    vm_f_t s = VM_VFADD_VV (eh, ih, vl);
    vm_f_t re = VM_VFADD_VV (VM_VFSUB_VV (eh, s, vl), ih, vl);
    vm_f_t dl = VM_VFADD_VV (VM_VFADD_VV (re, el, vl), il, vl);
    z = VM_VFMUL_VF (VM_VFADD_VV (s, dl, vl), 0.5, vl);
  }
  (void) fixup;
  if (VM_VFIRST_M (m22, vl) >= 0)
    z = VM_KFN (cosh_special) (ax, z, m22, vl);
  return z;
}

/* ---- tanh ------------------------------------------------------- */

static inline vm_f_t
VM_KFN (tanh) (vm_f_t x, size_t vl)
{
  vm_i64_t hx = VM_VSRA_VX (VM_REINT_U_I (VM_BITS (x)), 32, vl);
  vm_u64_t ix = VM_VAND_VX (VM_REINT_I_U (hx), 0x7fffffff, vl);
  vm_b_t mneg = VM_VMSLT_VX (hx, 0, vl);
  vm_b_t m22 = VM_VMSGEU_VX (ix, 0x40360000u, vl);
  vm_b_t mtiny = VM_VMSLTU_VX (ix, 0x3c800000u, vl);	/* |x| < 2^-55 */
  vm_b_t msmall = VM_VMSLTU_VX (ix, 0x3fa99999u, vl);	/* |x| < 0.05 */
  vm_f_t ax = VM_VFABS_V (x, vl);
  vm_f_t axs = VM_VFMERGE_VFM (ax, 1.0, m22, vl);
  vm_f_t z;
  {
    /* tanh = (e - 1/e)/(e + 1/e), e = exp(|x|) to ~2^-60 relative.  */
    vm_f_t eh, el, ih, il, qh, ql;
    VM_EXP_DD (eh, el, axs, vl, exp);
    VM_HYP_INVDD (ih, il, eh, el, vl);
    vm_f_t sd = VM_VFSUB_VV (eh, ih, vl);
    vm_f_t rd = VM_VFSUB_VV (VM_VFSUB_VV (eh, sd, vl), ih, vl);
    vm_f_t dl = VM_VFADD_VV (VM_VFSUB_VV (rd, il, vl), el, vl);
    vm_f_t ss = VM_VFADD_VV (eh, ih, vl);
    vm_f_t rs = VM_VFADD_VV (VM_VFSUB_VV (eh, ss, vl), ih, vl);
    vm_f_t sl = VM_VFADD_VV (VM_VFADD_VV (rs, el, vl), il, vl);
    VM_HYP_DIVDD (qh, ql, sd, dl, ss, sl, vl);
    z = VM_VFADD_VV (qh, ql, vl);
  }
  {
    /* |x| < 0.05: x*P(x^2), tanh Taylor.  Guard lanes and |x| < 2^-300
       (whose x^2 would underflow) get a benign input instead.  */
    vm_b_t mden = VM_VMSLTU_VX (ix, 0x2d300000u, vl);
    vm_f_t axp = VM_VFMERGE_VFM (axs, 1.0,
				 VM_VMOR_MM (mden, m22, vl), vl);
    vm_f_t t2 = VM_VFMUL_VV (axp, axp, vl);
    vm_f_t p = VM_VFADD_VF (
      VM_VFMUL_VF (t2, -1382.0 / 155925.0, vl), 62.0 / 2835.0, vl);
    p = VM_VFADD_VF (VM_VFMUL_VV (p, t2, vl), -17.0 / 315.0, vl);
    p = VM_VFADD_VF (VM_VFMUL_VV (p, t2, vl), 2.0 / 15.0, vl);
    p = VM_VFADD_VF (VM_VFMUL_VV (p, t2, vl), -1.0 / 3.0, vl);
    /* x + x*t*p, one fma: no 1+small rounding.  */
    vm_f_t rp = VM_VFMACC_VV (axp, axp, VM_VFMUL_VV (t2, p, vl), vl);
    z = VM_VFMERGE_VVM (z, rp, msmall, vl);
  }
  z = VM_VFMERGE_VVM (z, VM_VFNEG_V (z, vl), mneg, vl);
  z = VM_VFMERGE_VVM (z, x, mtiny, vl);
  if (VM_VFIRST_M (m22, vl) >= 0)
    {
      /* |x| >= 22 -> +-1 (exact); +-Inf -> +-1; NaN -> NaN.  */
      vm_b_t mnf = VM_VMSGEU_VX (ix, 0x7ff00000u, vl);
      vm_b_t minf = VM_VMSEQ_VX (VM_BITS (ax), VM_HYP_INF, vl);
      vm_f_t sg = VM_VFMERGE_VFM (VM_VFMV_V_F (1.0, vl), -1.0, mneg, vl);
      vm_f_t zs = VM_VFMERGE_VVM (sg, VM_VFADD_VV_M (mnf, x, x, vl), mnf,
				  vl);
      zs = VM_VFMERGE_VVM (zs, sg, minf, vl);
      z = VM_VFMERGE_VVM (z, zs, m22, vl);
    }
  return z;
}
