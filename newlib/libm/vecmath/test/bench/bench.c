/* bench.c -- libvecmath throughput benchmark: vector entries vs scalar loops.

   For each function it times

     for (i = 0; i < n; i++) y[i] = sin (x[i]);        (scalar baseline)
     vsin (y, x, n);                                   (libvecmath)

   over the same input array and reports ns/element plus the speedup.
   The scalar baseline is a plain -O2 -static loop, no -ffast-math and no
   -fno-math-errno: that is the loop an untuned caller writes, and -static
   resolves the scalar call directly (no PLT), which is the setting most
   favourable to the baseline.

   All 41 binary64 entries and the binary32 set are covered.

   Build natively on an RVV board:

     make bench                                     # from ../ (test/)
     gcc -O2 -static -march=rv64gcv -mabi=lp64d -DVECMATH_STANDALONE \
         -I../include -I../runner -I../../include -I../../ -I../shim \
         -o bench bench.c ../../v*.c ../../tables/*.c -lm

   Run:  ./bench [--n N] [--reps R] [--func NAME] [--f32|--f64]

   Timing: CLOCK_MONOTONIC, best of R repetitions per pass.  Both passes
   fold their output streams into a printed checksum, so neither can be
   optimized away.  A NaN in the checksum means an input left its function's
   domain and hit the scalar fixup, which would time the wrong path -- the
   per-function domains below are chosen to avoid that.  qemu user-mode
   numbers are functionally meaningful only, not timing.  */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <vecmath.h>

#define VM_PI 3.14159265358979323846
#define VM_LOG2E 1.44269504088896340736

static double
now_ns (void)
{
  struct timespec ts;
  clock_gettime (CLOCK_MONOTONIC, &ts);
  return (double) ts.tv_sec * 1e9 + (double) ts.tv_nsec;
}

static uint64_t rs = 0x2545f4914f6cdd1dULL;
static uint64_t
rnd (void)
{
  rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs;
}
static double
u01 (void)
{
  return (double) (rnd () >> 11) * 0x1p-53;
}

/* Best of `reps` runs of STMT over n elements, in ns/element.  */
#define NS_PER_ELST(reps, n, STMT)                                       \
  ({                                                                     \
    double best_ = 1e300;                                                \
    for (int r_ = 0; r_ < (reps); r_++)                                  \
      {                                                                  \
	double t0_ = now_ns ();                                          \
	STMT;                                                            \
	double t1_ = now_ns ();                                          \
	if (t1_ - t0_ < best_) best_ = t1_ - t0_;                        \
      }                                                                  \
    best_ / (double) (n);                                                \
  })

/* ------------------------------------------------------------------ */
/* Input domains.  Chosen so every sampled argument stays inside the
   function's own domain (no NaN -> no scalar fixup -> the vector kernel's
   main path is what gets timed) and off the over/underflow boundaries:
   a guard lane that has to go through scalar libm measures the fallback,
   not the kernel.  exp[-50,50]; trig/hyper [-10,10]; the [-1,1] family
   clamped to its real domain; logs/powers/roots positive.            */

static double
domain_lo (const char *name)
{
  if (!strcmp (name, "exp") || !strcmp (name, "expf")
      || !strcmp (name, "expm1") || !strcmp (name, "tanh")
      || !strcmp (name, "tanhf") || !strcmp (name, "sinh")
      || !strcmp (name, "cosh") || !strcmp (name, "asinh")
      || !strcmp (name, "atan") || !strcmp (name, "atanpi")
      || !strcmp (name, "erf") || !strcmp (name, "erfc")
      || !strcmp (name, "erff"))
    return -10.0;
  if (!strcmp (name, "exp10"))
    return -20.0;			/* off overflow and off the fixup */
  if (!strcmp (name, "asin") || !strcmp (name, "acos")
      || !strcmp (name, "asinpi") || !strcmp (name, "acospi"))
    return -1.0;			/* domain is exactly [-1,1] */
  if (!strcmp (name, "log1p") || !strcmp (name, "log2p1"))
    return -0.9;			/* x <= -1 gives NaN and the fixup */
  if (!strcmp (name, "atanh"))
    return -0.9;			/* |x| >= 1 gives NaN or +-inf */
  if (!strcmp (name, "acosh"))
    return 1.0000001;			/* domain is [1,inf) */
  if (!strcmp (name, "tgamma") || !strcmp (name, "lgamma")
      || !strcmp (name, "lgamma_r"))
    return 0.5;				/* positive band: no poles, no fixup */
  if (!strcmp (name, "exp2") || !strcmp (name, "log")
      || !strcmp (name, "log2") || !strcmp (name, "logf")
      || !strcmp (name, "log10") || !strcmp (name, "sqrt")
      || !strcmp (name, "sqrtf") || !strcmp (name, "cbrt")
      || !strcmp (name, "pow") || !strcmp (name, "rsqrt"))
    return 0.5;
  return -10.0;				/* sin/cos/tan/sincos/pi family */
}

static double
domain_hi (const char *name)
{
  if (!strcmp (name, "exp10"))
    return 20.0;
  if (!strcmp (name, "asin") || !strcmp (name, "acos")
      || !strcmp (name, "asinpi") || !strcmp (name, "acospi"))
    return 1.0;
  if (!strcmp (name, "atanh"))
    return 0.9;
  if (!strcmp (name, "acosh"))
    return 10.0;
  if (!strcmp (name, "tgamma") || !strcmp (name, "lgamma")
      || !strcmp (name, "lgamma_r"))
    return 5.0;
  if (!strcmp (name, "exp") || !strcmp (name, "expf"))
    return 50.0;
  if (!strcmp (name, "exp2") || !strcmp (name, "log")
      || !strcmp (name, "log2") || !strcmp (name, "logf")
      || !strcmp (name, "log10") || !strcmp (name, "sqrt")
      || !strcmp (name, "sqrtf") || !strcmp (name, "cbrt")
      || !strcmp (name, "pow") || !strcmp (name, "rsqrt"))
    return 1.0;
  return 10.0;				/* log1p, log2p1 keep lo=-0.9, hi=10 */
}

/* ------------------------------------------------------------------ */
/* binary64: one-input entries                                         */
/* ------------------------------------------------------------------ */

#define BENCH64(NAME, SCAL, VEC)                                        \
  static void                                                            \
  b_##NAME (double *y, const double *x, size_t n, int reps,              \
            double *sc, double *vc, double *csum)                        \
  {                                                                      \
    *sc = NS_PER_ELST (reps, n,                                          \
      for (size_t i = 0; i < n; i++) y[i] = SCAL);                       \
    for (size_t i = 0; i < n; i++) *csum += y[i];                        \
    *vc = NS_PER_ELST (reps, n, VEC);                                    \
    for (size_t i = 0; i < n; i++) *csum += y[i];                        \
  }

/* roots / power */
BENCH64 (sqrt,   sqrt (x[i]),   vsqrt (y, x, n))
BENCH64 (rsqrt,  1.0 / sqrt (x[i]), vrsqrt (y, x, n))
BENCH64 (cbrt,   cbrt (x[i]),   vcbrt (y, x, n))

/* exponential / logarithmic */
BENCH64 (exp,    exp (x[i]),    vexp (y, x, n))
BENCH64 (exp2,   exp2 (x[i]),   vexp2 (y, x, n))
BENCH64 (exp10,  exp10 (x[i]),  vexp10 (y, x, n))
BENCH64 (expm1,  expm1 (x[i]),  vexpm1 (y, x, n))
BENCH64 (log,    log (x[i]),    vlog (y, x, n))
BENCH64 (log2,   log2 (x[i]),   vlog2 (y, x, n))
BENCH64 (log10,  log10 (x[i]),  vlog10 (y, x, n))
BENCH64 (log1p,  log1p (x[i]),  vlog1p (y, x, n))
BENCH64 (log2p1, log2 (1.0 + x[i]), vlog2p1 (y, x, n))

/* trigonometric */
BENCH64 (sin,    sin (x[i]),    vsin (y, x, n))
BENCH64 (cos,    cos (x[i]),    vcos (y, x, n))
BENCH64 (tan,    tan (x[i]),    vtan (y, x, n))

/* pi-scaled trigonometric */
BENCH64 (sinpi,  sin (VM_PI * x[i]),  vsinpi (y, x, n))
BENCH64 (cospi,  cos (VM_PI * x[i]),  vcospi (y, x, n))
BENCH64 (tanpi,  tan (VM_PI * x[i]),  vtanpi (y, x, n))

/* inverse trigonometric */
BENCH64 (asin,    asin (x[i]),    vasin (y, x, n))
BENCH64 (acos,    acos (x[i]),    vacos (y, x, n))
BENCH64 (atan,    atan (x[i]),    vatan (y, x, n))
BENCH64 (asinpi,  asin (x[i]) / VM_PI,  vasinpi (y, x, n))
BENCH64 (acospi,  acos (x[i]) / VM_PI,  vacospi (y, x, n))
BENCH64 (atanpi,  atan (x[i]) / VM_PI,  vatanpi (y, x, n))

/* hyperbolic and inverse hyperbolic */
BENCH64 (sinh,   sinh (x[i]),   vsinh (y, x, n))
BENCH64 (cosh,   cosh (x[i]),   vcosh (y, x, n))
BENCH64 (tanh,   tanh (x[i]),   vtanh (y, x, n))
BENCH64 (asinh,  asinh (x[i]),  vasinh (y, x, n))
BENCH64 (acosh,  acosh (x[i]),  vacosh (y, x, n))
BENCH64 (atanh,  atanh (x[i]),  vatanh (y, x, n))

/* special functions */
BENCH64 (erf,    erf (x[i]),    verf (y, x, n))
BENCH64 (erfc,   erfc (x[i]),   verfc (y, x, n))
BENCH64 (tgamma, tgamma (x[i]), vtgamma (y, x, n))
BENCH64 (lgamma, lgamma (x[i]), vlgamma (y, x, n))

/* ------------------------------------------------------------------ */
/* binary64: entries that need more than one buffer                    */
/* ------------------------------------------------------------------ */

/* Two-input entries draw their second operand inside the body, so the
   shared dispatch signature (one x array) stays uniform.  */
#define BENCH64_2(NAME, SCAL, VEC, LO2, HI2)                             \
  static void                                                             \
  b_##NAME (double *y, const double *x, size_t n, int reps,               \
            double *sc, double *vc, double *csum)                         \
  {                                                                       \
    double *z = malloc (n * sizeof (double));                             \
    for (size_t i = 0; i < n; i++) z[i] = (LO2) + u01 () * ((HI2) - (LO2)); \
    *sc = NS_PER_ELST (reps, n,                                           \
      for (size_t i = 0; i < n; i++) y[i] = SCAL);                        \
    for (size_t i = 0; i < n; i++) *csum += y[i];                         \
    *vc = NS_PER_ELST (reps, n, VEC);                                     \
    for (size_t i = 0; i < n; i++) *csum += y[i];                         \
    free (z);                                                             \
  }

/* vpow (y, ax, bx, n): x is the base, z the exponent.  */
BENCH64_2 (pow, pow (x[i], z[i]), vpow (y, x, z, n), 1.0, 2.0)
/* vhypot (y, ax, bx, n).  */
BENCH64_2 (hypot, hypot (x[i], z[i]), vhypot (y, x, z, n), -10.0, 10.0)
/* vatan2 (y, ay, ax, n): the header's argument order is (y, then x).  */
BENCH64_2 (atan2, atan2 (x[i], z[i]), vatan2 (y, x, z, n), -10.0, 10.0)
BENCH64_2 (atan2pi, atan2 (x[i], z[i]) / VM_PI,
	   vatan2pi (y, x, z, n), -10.0, 10.0)

/* sincos: two outputs; the scalar side calls sin and cos separately, which
   is what an untuned caller does (no fused scalar loop to beat).  */
static void
b_sincos (double *y, const double *x, size_t n, int reps, double *sc,
          double *vc, double *csum)
{
  double *y2 = malloc (n * sizeof (double));
  *sc = NS_PER_ELST (reps, n,
    for (size_t i = 0; i < n; i++) { y[i] = sin (x[i]); y2[i] = cos (x[i]); });
  for (size_t i = 0; i < n; i++) *csum += y[i] + y2[i];
  *vc = NS_PER_ELST (reps, n, vsincos (y, y2, x, n));
  for (size_t i = 0; i < n; i++) *csum += y[i] + y2[i];
  free (y2);
}

static void
b_sincospi (double *y, const double *x, size_t n, int reps, double *sc,
            double *vc, double *csum)
{
  double *y2 = malloc (n * sizeof (double));
  *sc = NS_PER_ELST (reps, n,
    for (size_t i = 0; i < n; i++)
      { y[i] = sin (VM_PI * x[i]); y2[i] = cos (VM_PI * x[i]); });
  for (size_t i = 0; i < n; i++) *csum += y[i] + y2[i];
  *vc = NS_PER_ELST (reps, n, vsincospi (y, y2, x, n));
  for (size_t i = 0; i < n; i++) *csum += y[i] + y2[i];
  free (y2);
}

/* lgamma_r: second output is an int sign per lane.  */
static void
b_lgamma_r (double *y, const double *x, size_t n, int reps, double *sc,
             double *vc, double *csum)
{
  int *sg = malloc (n * sizeof (int));
  int *sg2 = malloc (n * sizeof (int));
  *sc = NS_PER_ELST (reps, n,
    for (size_t i = 0; i < n; i++) y[i] = lgamma_r (x[i], &sg2[i]));
  for (size_t i = 0; i < n; i++) { *csum += y[i]; *csum += sg2[i]; }
  *vc = NS_PER_ELST (reps, n, vlgamma_r (y, sg, x, n));
  for (size_t i = 0; i < n; i++) { *csum += y[i]; *csum += sg[i]; }
  free (sg); free (sg2);
}

/* ------------------------------------------------------------------ */
/* binary32                                                            */
/* ------------------------------------------------------------------ */

#define BENCH32(NAME, SCAL, VEC)                                        \
  static void                                                           \
  bf_##NAME (float *y, const float *x, size_t n, int reps,              \
             double *sc, double *vc, double *csum)                      \
  {                                                                     \
    *sc = NS_PER_ELST (reps, n,                                         \
      for (size_t i = 0; i < n; i++) y[i] = SCAL);                      \
    for (size_t i = 0; i < n; i++) *csum += y[i];                        \
    *vc = NS_PER_ELST (reps, n, VEC);                                   \
    for (size_t i = 0; i < n; i++) *csum += y[i];                        \
  }

BENCH32 (sinf,  sinf (x[i]),  vsinf (y, x, n))
BENCH32 (cosf,  cosf (x[i]),  vcosf (y, x, n))
BENCH32 (expf,  expf (x[i]),  vexpf (y, x, n))
BENCH32 (logf,  logf (x[i]),  vlogf (y, x, n))
BENCH32 (tanhf, tanhf (x[i]), vtanhf (y, x, n))
BENCH32 (erff,  erff (x[i]),  verff (y, x, n))
BENCH32 (sqrtf, sqrtf (x[i]), vsqrtf (y, x, n))

/* ------------------------------------------------------------------ */

typedef void (*run64_t) (double *, const double *, size_t, int, double *,
			 double *, double *);
typedef void (*run32_t) (float *, const float *, size_t, int, double *,
			 double *, double *);

typedef struct
{
  const char *name;
  run64_t run64;
  run32_t run32;
} bench_t;

static const bench_t benches[] = {
  /* roots / power */
  { "sqrt",     b_sqrt,     NULL },
  { "rsqrt",    b_rsqrt,    NULL },
  { "cbrt",     b_cbrt,     NULL },
  { "pow",      b_pow,      NULL },
  { "hypot",    b_hypot,    NULL },
  /* exponential */
  { "exp",      b_exp,      NULL },
  { "exp2",     b_exp2,     NULL },
  { "exp10",    b_exp10,    NULL },
  { "expm1",    b_expm1,    NULL },
  /* logarithmic */
  { "log",      b_log,      NULL },
  { "log2",     b_log2,     NULL },
  { "log10",    b_log10,    NULL },
  { "log1p",    b_log1p,    NULL },
  { "log2p1",   b_log2p1,   NULL },
  /* trigonometric */
  { "sin",      b_sin,      NULL },
  { "cos",      b_cos,      NULL },
  { "tan",      b_tan,      NULL },
  { "sincos",   b_sincos,   NULL },
  /* pi-scaled trigonometric */
  { "sinpi",    b_sinpi,    NULL },
  { "cospi",    b_cospi,    NULL },
  { "tanpi",    b_tanpi,    NULL },
  { "sincospi", b_sincospi, NULL },
  /* inverse trigonometric */
  { "asin",     b_asin,     NULL },
  { "acos",     b_acos,     NULL },
  { "atan",     b_atan,     NULL },
  { "atan2",    b_atan2,    NULL },
  { "asinpi",   b_asinpi,   NULL },
  { "acospi",   b_acospi,   NULL },
  { "atanpi",   b_atanpi,   NULL },
  { "atan2pi",  b_atan2pi,  NULL },
  /* hyperbolic */
  { "sinh",     b_sinh,     NULL },
  { "cosh",     b_cosh,     NULL },
  { "tanh",     b_tanh,     NULL },
  { "asinh",    b_asinh,    NULL },
  { "acosh",    b_acosh,    NULL },
  { "atanh",    b_atanh,    NULL },
  /* special */
  { "erf",      b_erf,      NULL },
  { "erfc",     b_erfc,     NULL },
  { "tgamma",   b_tgamma,   NULL },
  { "lgamma",   b_lgamma,   NULL },
  { "lgamma_r", b_lgamma_r, NULL },
  /* binary32 */
  { "sinf",     NULL,       bf_sinf },
  { "cosf",     NULL,       bf_cosf },
  { "expf",     NULL,       bf_expf },
  { "logf",     NULL,       bf_logf },
  { "tanhf",    NULL,       bf_tanhf },
  { "erff",     NULL,       bf_erff },
  { "sqrtf",    NULL,       bf_sqrtf },
};
#define NBENCH ((int) (sizeof benches / sizeof benches[0]))

int
main (int argc, char **argv)
{
  size_t n = 65536;
  int reps = 7;
  const char *only = NULL;
  int want32 = -1;			/* -1 = both, 0 = f64 only, 1 = f32 only */

  for (int i = 1; i < argc; i++)
    {
      if (!strcmp (argv[i], "--n") && i + 1 < argc)
	n = (size_t) strtol (argv[++i], NULL, 0);
      else if (!strcmp (argv[i], "--reps") && i + 1 < argc)
	reps = atoi (argv[++i]);
      else if (!strcmp (argv[i], "--func") && i + 1 < argc)
	only = argv[++i];
      else if (!strcmp (argv[i], "--f32"))
	want32 = 1;
      else if (!strcmp (argv[i], "--f64"))
	want32 = 0;
      else
	{
	  fprintf (stderr,
		   "usage: bench [--n N] [--reps R] [--func NAME] [--f32|--f64]\n");
	  return 2;
	}
    }

  size_t vlenb = 0;
  __asm__ volatile ("csrr %0, vlenb" : "=r" (vlenb));

  printf ("libvecmath bench: n=%zu reps=%d VLEN=%zu bits\n",
	  n, reps, vlenb * 8);
  printf ("%-9s %12s %12s %9s\n", "func", "scalar_ns/e", "vector_ns/e", "speedup");

  double csum = 0.0;
  for (int b = 0; b < NBENCH; b++)
    {
      const int f32 = benches[b].run64 == NULL;
      if (only && strcmp (only, benches[b].name))
	continue;
      if (want32 >= 0 && f32 != want32)
	continue;

      const size_t esz = f32 ? sizeof (float) : sizeof (double);
      void *xb = malloc (n * esz), *yb = malloc (n * esz);
      double sc = 0.0, vc = 0.0;
      const double lo = domain_lo (benches[b].name);
      const double hi = domain_hi (benches[b].name);

      if (f32)
	{
	  float *x = (float *) xb, *y = (float *) yb;
	  for (size_t i = 0; i < n; i++)
	    x[i] = (float) (lo + u01 () * (hi - lo));
	  benches[b].run32 (y, x, n, reps, &sc, &vc, &csum);
	}
      else
	{
	  double *x = (double *) xb, *y = (double *) yb;
	  for (size_t i = 0; i < n; i++)
	    x[i] = lo + u01 () * (hi - lo);
	  benches[b].run64 (y, x, n, reps, &sc, &vc, &csum);
	}

      printf ("%-9s %12.3f %12.3f %8.2fx\n", benches[b].name, sc, vc,
	      vc > 0.0 ? sc / vc : 0.0);
      free (xb); free (yb);
    }
  printf ("checksum %g\n", csum);
  return 0;
}
