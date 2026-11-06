/* ulp.c -- ULP error evaluation.

   Convention (same as glibc's libm-test-ng): for an exactly-computed
   result E with correctly rounded image RN(E) = hi, the ulp unit is
   U = 2^(floor(log2|hi|) - 52) for normal hi, and U = 2^-1074 (the
   minimum subnormal) when the result lands in the subnormal region.
   The measured error uses the double-double reference (hi + lo), so
   resolution is ~2^-53 ulp -- ample to separate "<=1 ulp" from ">1 ulp".

   Computed entirely with double + fma (no long double dependency).  */

#include <math.h>
#include <float.h>
#include <string.h>
#include "ulp.h"

static inline double
bdouble (uint64_t u)
{
  double d;
  memcpy (&d, &u, 8);
  return d;
}

static inline uint64_t
dbits (double d)
{
  uint64_t u;
  memcpy (&u, &d, 8);
  return u;
}

ulp_verdict_t
ulp_eval (double y, const vmb_rec1_t *r)
{
  return ulp_eval_core (y, r->expect, r->ref_hi, r->ref_lo, r->exact_bits);
}

ulp_verdict_t
ulp_eval2 (double y, const vmb_rec2_t *r, uint32_t expect)
{
  return ulp_eval_core (y, expect, r->ref2_hi, r->ref2_lo, r->exact2_bits);
}

ulp_verdict_t
ulp_eval_core (double y, uint32_t expect, uint64_t ref_hi, uint64_t ref_lo,
	       uint64_t exact_bits)
{
  ulp_verdict_t v = { 0.0, 1, 1 };
  uint64_t yb = dbits (y);
  struct { uint64_t ref_hi, ref_lo, exact_bits; uint32_t expect; } rd;
  rd.expect = expect;
  rd.ref_hi = ref_hi;
  rd.ref_lo = ref_lo;
  rd.exact_bits = exact_bits;
#define r (&rd)
  if (r->expect & VMB_EXPECT_NAN)
    {
      if (!(yb & 0x7ff0000000000000ULL)
	  || !(yb & 0x000fffffffffffffULL))
	v.class_ok = 0;			/* not a NaN */
      else if ((r->expect & VMB_NAN_QUIET)
	       && !(yb & 0x0008000000000000ULL))
	v.class_ok = 0;			/* signaling NaN returned */
      if ((r->expect & VMB_SIGN_STRICT)
	  && ((yb >> 63) != (r->exact_bits >> 63)))
	v.class_ok = 0;
      v.ulp = 0.0;
      return v;
    }

  if (r->expect & VMB_EXPECT_INF)
    {
      v.class_ok = (yb & 0x7fffffffffffffffULL) == 0x7ff0000000000000ULL
		   && (yb >> 63) == (r->exact_bits >> 63);
      v.ulp = 0.0;
      return v;
    }

  if (r->expect & VMB_EXPECT_EXACT)
    {
      v.class_ok = (yb == r->exact_bits);
      v.ulp = 0.0;
      return v;
    }

  /* Normal ulp metric against the dd reference.  */
  if (!isfinite (y))
    {
      v.class_ok = 0;			/* finite expected */
      v.ulp = INFINITY;
      return v;
    }

  double hi = bdouble (r->ref_hi);
  double lo = bdouble (r->ref_lo);
  /* err = y - (hi + lo), computed with the lo correction fused.  */
  double e1 = y - hi;			/* exact when |y-hi| small (Sterbenz) */
  double err = fma (-1.0, lo, e1);
  double ae = fabs (err);

  double ah = fabs (hi);
  double u;
  if (ah >= DBL_MIN)
    {
      int ex;
      frexp (ah, &ex);		/* ah = m*2^ex, m in [0.5,1) */
      u = ldexp (1.0, ex - 53);	/* 2^(floor(log2 ah) - 52) */
    }
  else
    u = DBL_TRUE_MIN;		/* subnormal region: grid is uniform */

  v.ulp = ae / u;
#undef r
  return v;
}

/* ------------------------------------------------------------------ */
/* binary32 metric.  Only the class tests use 32-bit masks; the error
   itself is evaluated in double, where y - (hi + lo) is exact, so the
   measurement has far more than the one ulp of resolution needed.  */

static inline float
bfloat (uint64_t u)
{
  uint32_t w = (uint32_t) u;
  float f;
  memcpy (&f, &w, 4);
  return f;
}

static inline uint32_t
fbits (float f)
{
  uint32_t w;
  memcpy (&w, &f, 4);
  return w;
}

ulp_verdict_t
ulp_eval_f32 (float y, const vmb_rec1_t *r)
{
  uint32_t expect = r->expect;
  uint64_t ref_hi = r->ref_hi, ref_lo = r->ref_lo, exact_bits = r->exact_bits;
  ulp_verdict_t v = { 0.0, 1, 1 };
  uint32_t yb = fbits (y);

  if (expect & VMB_EXPECT_NAN)
    {
      if (!(yb & 0x7f800000u) || !(yb & 0x007fffffu))
	v.class_ok = 0;			/* not a NaN */
      else if ((expect & VMB_NAN_QUIET) && !(yb & 0x00400000u))
	v.class_ok = 0;			/* signaling NaN returned */
      if ((expect & VMB_SIGN_STRICT)
	  && (((yb >> 31) & 1u) != (((uint32_t) exact_bits >> 31) & 1u)))
	v.class_ok = 0;
      v.ulp = 0.0;
      return v;
    }

  if (expect & VMB_EXPECT_INF)
    {
      v.class_ok = (yb & 0x7fffffffu) == 0x7f800000u
		   && ((yb >> 31) & 1u) == (((uint32_t) exact_bits >> 31) & 1u);
      v.ulp = 0.0;
      return v;
    }

  if (expect & VMB_EXPECT_EXACT)
    {
      v.class_ok = (yb == (uint32_t) exact_bits);
      v.ulp = 0.0;
      return v;
    }

  if (!isfinite ((double) y))
    {
      v.class_ok = 0;			/* finite expected */
      v.ulp = INFINITY;
      return v;
    }

  double hi = (double) bfloat (ref_hi);
  double lo = (double) bfloat (ref_lo);
  double err = ((double) y - hi) - lo;
  double ae = fabs (err);

  double ah = fabs (hi);
  double u;
  if (ah >= (double) FLT_MIN)
    {
      int ex;
      frexp (ah, &ex);		/* ah = m*2^ex, m in [0.5,1) */
      u = ldexp (1.0, ex - 24);	/* 2^(floor(log2 ah) - 23) */
    }
  else
    u = ldexp (1.0, -149);	/* subnormal region: uniform grid */

  v.ulp = ae / u;
  return v;
}

ulp_verdict_t
ulp_eval2_f32 (float y2, const vmb_rec2_t *r2, uint32_t expect)
{
  vmb_rec1_t r;
  r.expect = expect;
  r.ref_hi = r2->ref2_hi;
  r.ref_lo = r2->ref2_lo;
  r.exact_bits = r2->exact2_bits;
  return ulp_eval_f32 (y2, &r);
}

double
ulp_distance_f32 (float got, float want)
{
  if (isnan ((double) got) || isnan ((double) want))
    return (isnan ((double) got) && isnan ((double) want)) ? 0.0 : INFINITY;
  if (isinf ((double) got) || isinf ((double) want) || got == want)
    return (got == want) ? 0.0 : INFINITY;
  double ah = fabs ((double) want);
  double u;
  if (ah >= (double) FLT_MIN)
    {
      int ex;
      frexp (ah, &ex);
      u = ldexp (1.0, ex - 24);
    }
  else
    u = ldexp (1.0, -149);
  return fabs ((double) got - (double) want) / u;
}

double
ulp_distance (double got, double want)
{
  if (isnan (got) || isnan (want))
    return (isnan (got) && isnan (want)) ? 0.0 : INFINITY;
  if (isinf (got) || isinf (want) || got == want)
    return (got == want) ? 0.0 : INFINITY;
  double ah = fabs (want);
  double u;
  if (ah >= DBL_MIN)
    {
      int ex;
      frexp (ah, &ex);
      u = ldexp (1.0, ex - 53);
    }
  else
    u = DBL_TRUE_MIN;
  return fabs (got - want) / u;
}
