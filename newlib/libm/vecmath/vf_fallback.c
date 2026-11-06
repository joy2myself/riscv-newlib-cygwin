/* vf_fallback.c -- scalar fallbacks for the binary32 libvecmath array API.

   Compiled only when the translation unit lacks the V extension (a
   non-vector multilib, e.g. the default rv64imafdc build), so that every
   configured libvecmath.a exports the complete binary32 symbol set
   regardless of the -march it was built with.  Accuracy inherits from
   the scalar single-precision libm being called; the greenfield
   SYCL-style functions use portable float+fma references and are not
   covered by the <= 1 ulp contract, which applies to RVV builds.  */

#include <vecmath.h>

#if !defined(__riscv_vector)

#include <math.h>
#include <fenv.h>

/* ---- functions with a direct libm analogue ---- */

void
vsqrtf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sqrtf (x[i]);
}

void
vrsqrtf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      float s = sqrtf (x[i]);	/* invalid for sNaN and x<0; exact else */
      float r0;
      if (s == 0.0f)
	{
	  /* SYCL: rsqrt(+-0) = +Inf, with divbyzero raised.  */
	  r0 = HUGE_VALF;
	  feraiseexcept (FE_DIVBYZERO);
	}
      else
	{
	  r0 = 1.0f / s;
	  if (s == s && s != HUGE_VALF)
	    {
	      /* One Newton step, mirroring the RVV kernel.  Only ==/!=
		 compares: RISC-V ordered compares raise INVALID even for
		 qNaN.  sqrtf never returns a negative, so this branch is
		 exactly "finite positive".  */
	      float r = x[i] * r0;
	      float e = fmaf (-r, r0, 1.0f);
	      r0 = fmaf (r0, 0.5f * e, r0);
	    }
	}
      y[i] = r0;
    }
}

void
vcbrtf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = cbrtf (x[i]);
}

void
vexpf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = expf (x[i]);
}

void
vexpm1f (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = expm1f (x[i]);
}

void
vexp2f (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = exp2f (x[i]);
}

void
vexp10f (float *__restrict y, const float *__restrict x, size_t n)
{
  /* newlib's scalar exp10f is powf(10,x); mirror that.  */
  for (size_t i = 0; i < n; i++)
    y[i] = powf (10.0f, x[i]);
}

void
vsinf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sinf (x[i]);
}

void
vcosf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = cosf (x[i]);
}

void
vtanf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = tanf (x[i]);
}

void
vasinpif (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = asinf (x[i]) / (float) M_PI;
}

void
vacospif (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = acosf (x[i]) / (float) M_PI;
}

void
vasinf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = asinf (x[i]);
}

void
vacosf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = acosf (x[i]);
}

void
vhypotf (float *__restrict y, const float *__restrict ax,
	 const float *__restrict ay, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = hypotf (ax[i], ay[i]);
}

void
vasinhf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = asinhf (x[i]);
}

void
vacoshf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = acoshf (x[i]);
}

void
vatanhf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atanhf (x[i]);
}

void
vatanf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atanf (x[i]);
}

void
vatan2f (float *__restrict y, const float *__restrict ay,
	 const float *__restrict ax, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atan2f (ay[i], ax[i]);
}

void
vatanpif (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atanf (x[i]) / (float) M_PI;
}

void
vatan2pif (float *__restrict y, const float *__restrict ay,
	   const float *__restrict ax, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atan2f (ay[i], ax[i]) / (float) M_PI;
}

void
vsinhf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sinhf (x[i]);
}

void
vcoshf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = coshf (x[i]);
}

void
vtanhf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = tanhf (x[i]);
}

void
verff (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = erff (x[i]);
}

void
verfcf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = erfcf (x[i]);
}

void
vtgammaf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = tgammaf (x[i]);
}

void
vlgammaf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = lgammaf (x[i]);
}

void
vlgamma_rf (float *__restrict y, int *__restrict sign,
	    const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      int sg;
      y[i] = lgammaf_r (x[i], &sg);
      sign[i] = sg;
    }
}

void
vsincosf (float *__restrict sy, float *__restrict cy,
	  const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      sy[i] = sinf (x[i]);
      cy[i] = cosf (x[i]);
    }
}

/* sinpi family: portable scalar references (float).  Reduction:
   k = trunc(x), f = x - k (exact), then f folded into [-1/2, 1/2] with
   s = (-1)^k * (fold sign).  */

#define SC_PIF 3.14159274101257324e+00f

#define SC_PI_REDUCEF(x, f, s) do {					\
  float scpf__k = truncf (x);						\
  (f) = (x) - scpf__k;							\
  if ((f) > 0.5f)							\
    { (f) -= 1.0f; (s) = -1.0f; }					\
  else if ((f) < -0.5f)							\
    { (f) += 1.0f; (s) = -1.0f; }					\
  else									\
    (s) = 1.0f;								\
  if (fmodf (fabsf (scpf__k), 2.0f) == 1.0f)				\
    (s) = -(s);								\
} while (0)

static float
sc_sinpif (float x)
{
  float f, s;
  if (!isfinite (x))
    return x - x;			/* +-Inf -> NaN+invalid; NaN quiet */
  SC_PI_REDUCEF (x, f, s);
  if (f == 0.0f)
    return copysignf (0.0f, x);		/* sinpi(+-n) = +-0 */
  return s * sinf (f * SC_PIF);
}

static float
sc_cospif (float x)
{
  float f, s;
  if (!isfinite (x))
    return x - x;
  SC_PI_REDUCEF (x, f, s);
  if (f == 0.0f)
    return s;				/* cospi(n) = (-1)^n */
  if (fabsf (f) == 0.5f)
    return 0.0f;			/* cospi(n + 1/2) = +0 */
  return s * cosf (f * SC_PIF);
}

static float
sc_tanpif (float x)
{
  float f, s;
  if (!isfinite (x))
    return x - x;
  SC_PI_REDUCEF (x, f, s);
  if (f == 0.0f)
    return copysignf (0.0f, x) * s;	/* tanpi(n) = copysign(0,x)*(-1)^n */
  if (fabsf (f) == 0.5f)
    {
      /* tanpi(n + 1/2) = (-1)^n * +Inf, n = floor(x); for f = -1/2 the
	 trunc-based k is floor(x) + 1, hence the sign flip.  */
      feraiseexcept (FE_DIVBYZERO);
      return ((f > 0.0f) == (s > 0.0f)) ? HUGE_VALF : -HUGE_VALF;
    }
  return s * tanf (f * SC_PIF);
}

void
vsinpif (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sc_sinpif (x[i]);
}

void
vcospif (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sc_cospif (x[i]);
}

void
vtanpif (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sc_tanpif (x[i]);
}

void
vsincospif (float *__restrict sy, float *__restrict cy,
	    const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      sy[i] = sc_sinpif (x[i]);
      cy[i] = sc_cospif (x[i]);
    }
}

void
vpowf (float *__restrict z, const float *__restrict x,
       const float *__restrict y, size_t n)
{
  for (size_t i = 0; i < n; i++)
    z[i] = powf (x[i], y[i]);
}

void
vlogf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = logf (x[i]);
}

void
vlog2f (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = (log2f) (x[i]);
}

void
vlog10f (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = log10f (x[i]);
}

void
vlog1pf (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = log1pf (x[i]);
}

void
vlog2p1f (float *__restrict y, const float *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      float t = log1pf (x[i]);
      /* t * (1/ln2) with the correction folded via fmaf.  */
      y[i] = fmaf (t, 1.44269502162933350e+00f,
		   t * 2.03552737409310330e-17f);
    }
}

#endif /* !__riscv_vector */
