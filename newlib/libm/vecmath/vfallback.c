/* vfallback.c -- scalar fallbacks for the libvecmath array API.

   Compiled only when the translation unit lacks the V extension (a
   non-vector multilib, e.g. the default rv64imafdc build), so that every
   configured libvecmath.a exports the complete symbol set regardless of
   the -march it was built with.  Accuracy inherits from the scalar libm
   being called; the greenfield SYCL-style functions use portable
   double+fma references and are not covered by the <= 1 ulp contract,
   which applies to RVV builds.

   The __riscv_vector source-level gate follows
   newlib/libc/machine/riscv/memcpy.c.  */

#include <vecmath.h>

#if !defined(__riscv_vector)

#include <math.h>
#include <fenv.h>

/* ---- functions with a direct libm analogue ---- */

void
vsqrt (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sqrt (x[i]);
}

void
vrsqrt (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      double s = sqrt (x[i]);	/* invalid for sNaN and x<0; exact else */
      double r0;
      if (s == 0.0)
	{
	  /* SYCL: rsqrt(+-0) = +Inf, with divbyzero raised.  1.0/s would
	     be folded away under GCC's default -fno-trapping-math,
	     dropping the flag.  */
	  r0 = HUGE_VAL;
	  feraiseexcept (FE_DIVBYZERO);
	}
      else
	{
	  r0 = 1.0 / s;
	  if (s == s && s != HUGE_VAL)
	    {
	      /* One Newton step, mirroring the RVV kernel; the raw
		 sqrt+divide composition is up to ~1.25 ulp.  Only ==/!=
		 compares here: RISC-V ordered compares raise INVALID
		 even for qNaN.  sqrt() never returns a negative, so
		 this branch is exactly "finite positive".  */
	      double r = x[i] * r0;
	      double e = fma (-r, r0, 1.0);
	      r0 = fma (r0, 0.5 * e, r0);
	    }
	}
      y[i] = r0;
    }
}

void
vcbrt (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = cbrt (x[i]);
}

void
vexp (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = exp (x[i]);
}

void
vexpm1 (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = expm1 (x[i]);
}

void
vexp2 (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = exp2 (x[i]);
}

void
vexp10 (double *__restrict y, const double *__restrict x, size_t n)
{
  /* newlib's scalar exp10 is pow(10,x); mirror that.  */
  for (size_t i = 0; i < n; i++)
    y[i] = pow (10.0, x[i]);
}

void
vsin (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sin (x[i]);
}

void
vcos (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = cos (x[i]);
}

void
vtan (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = tan (x[i]);
}

void
vasinpi (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = asin (x[i]) / M_PI;
}

void
vacospi (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = acos (x[i]) / M_PI;
}

void
vasin (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = asin (x[i]);
}

void
vacos (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = acos (x[i]);
}

void
vhypot (double *__restrict y, const double *__restrict ax,
	const double *__restrict ay, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = hypot (ax[i], ay[i]);
}

void
vasinh (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = asinh (x[i]);
}

void
vacosh (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = acosh (x[i]);
}

void
vatanh (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atanh (x[i]);
}

void
vatan (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atan (x[i]);
}

void
vatan2 (double *__restrict y, const double *__restrict ay,
	const double *__restrict ax, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atan2 (ay[i], ax[i]);
}

void
vatanpi (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atan (x[i]) / M_PI;
}

void
vatan2pi (double *__restrict y, const double *__restrict ay,
	  const double *__restrict ax, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = atan2 (ay[i], ax[i]) / M_PI;
}

void
vsinh (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sinh (x[i]);
}

void
vcosh (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = cosh (x[i]);
}

void
vtanh (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = tanh (x[i]);
}

void
verf (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = erf (x[i]);
}

void
verfc (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = erfc (x[i]);
}

void
vtgamma (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = tgamma (x[i]);
}

void
vlgamma (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = lgamma (x[i]);
}

void
vlgamma_r (double *__restrict y, int *__restrict sign,
	   const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      int sg;
      y[i] = lgamma_r (x[i], &sg);
      sign[i] = sg;
    }
}

void
vsincos (double *__restrict sy, double *__restrict cy,
	 const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      sy[i] = sin (x[i]);
      cy[i] = cos (x[i]);
    }
}

/* sinpi family: portable scalar references.  Reduction: k = trunc(x),
   f = x - k (exact), then f folded into [-1/2, 1/2] with
   s = (-1)^k * (fold sign).  */

#define SC_PI 3.14159265358979311600e+00

#define SC_PI_REDUCE(x, f, s) do {					\
  double scp__k = trunc (x);						\
  (f) = (x) - scp__k;							\
  if ((f) > 0.5)							\
    { (f) -= 1.0; (s) = -1.0; }						\
  else if ((f) < -0.5)							\
    { (f) += 1.0; (s) = -1.0; }						\
  else									\
    (s) = 1.0;								\
  if (fmod (fabs (scp__k), 2.0) == 1.0)					\
    (s) = -(s);								\
} while (0)

static double
sc_sinpi (double x)
{
  double f, s;
  if (!isfinite (x))
    return x - x;			/* +-Inf -> NaN+invalid; NaN quiet */
  SC_PI_REDUCE (x, f, s);
  if (f == 0.0)
    return copysign (0.0, x);		/* sinpi(+-n) = +-0 */
  return s * sin (f * SC_PI);
}

static double
sc_cospi (double x)
{
  double f, s;
  if (!isfinite (x))
    return x - x;
  SC_PI_REDUCE (x, f, s);
  if (f == 0.0)
    return s;				/* cospi(n) = (-1)^n */
  if (fabs (f) == 0.5)
    return 0.0;				/* cospi(n + 1/2) = +0 */
  return s * cos (f * SC_PI);
}

static double
sc_tanpi (double x)
{
  double f, s;
  if (!isfinite (x))
    return x - x;
  SC_PI_REDUCE (x, f, s);
  if (f == 0.0)
    return copysign (0.0, x) * s;	/* tanpi(n) = copysign(0,x)*(-1)^n */
  if (fabs (f) == 0.5)
    {
      /* tanpi(n + 1/2) = (-1)^n * +Inf, n = floor(x); for f = -1/2 the
	 trunc-based k is floor(x) + 1, hence the sign flip.  */
      feraiseexcept (FE_DIVBYZERO);
      return ((f > 0.0) == (s > 0.0)) ? HUGE_VAL : -HUGE_VAL;
    }
  return s * tan (f * SC_PI);
}

void
vsinpi (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sc_sinpi (x[i]);
}

void
vcospi (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sc_cospi (x[i]);
}

void
vtanpi (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = sc_tanpi (x[i]);
}

void
vsincospi (double *__restrict sy, double *__restrict cy,
	   const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      sy[i] = sc_sinpi (x[i]);
      cy[i] = sc_cospi (x[i]);
    }
}

void
vpow (double *__restrict z, const double *__restrict x,
      const double *__restrict y, size_t n)
{
  for (size_t i = 0; i < n; i++)
    z[i] = pow (x[i], y[i]);
}

void
vlog (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = log (x[i]);
}

void
vlog2 (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = (log2) (x[i]);
}

void
vlog10 (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = log10 (x[i]);
}

void
vlog1p (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    y[i] = log1p (x[i]);
}

void
vlog2p1 (double *__restrict y, const double *__restrict x, size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      double t = log1p (x[i]);
      /* t * (1/ln2) with the dd correction folded via fma.  */
      y[i] = fma (t, 1.44269504088896338700e+00,
		  t * 2.03552737409310331110e-17);
    }
}

#endif /* !__riscv_vector */
