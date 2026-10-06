/* vmgen.c -- MPFR-based test-vector generator for libvecmath.

   Host tool.  For every function in vmfuncs.def it produces a binary
   blob of test records: deterministic pseudo-random inputs from several
   distributions plus a special-value battery, each with an exact
   reference value stored as a double-double pair (hi+lo ~= 106 bits of
   the 200-bit MPFR value), an expected special-value class, bitwise
   expectation for exactly-representable results, and the IEEE exception
   flags the call must raise.

   Usage:
     vmgen --outdir DIR [--tier small|full] [--func NAME]... [--seed S]

   Determinism: xoshiro256** seeded per function with seed ^ fnv1a(name);
   the seed is recorded in every blob header.

   Build:  gcc -O2 -o vmgen vmgen.c -lmpfr   (host)  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <errno.h>
#include <mpfr.h>

#include "vmblob.h"

/* ------------------------------------------------------------------ */
/* Function table                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
  const char *name;
  int id, nin, kind;
  double ulp;
} fdesc_t;

/* kind tokens in vmfuncs.def are paste-safe bare words; map to numbers. */
#define VMKN_C(a, b) a##b
#define VMKN(k) VMKN_C(VMKN_, k)
#define VMKN_SINGLE VMB_OUT_SINGLE
#define VMKN_DUAL VMB_OUT_DUAL
#define VMKN_DUAL_INT VMB_OUT_DUAL_INT

static const fdesc_t fdesc[] = {
#define VMFUNC(id, name, nin, kind, ulp, ref) [id] = { #name, id, nin, VMKN(kind), ulp },
#include "vmfuncs.def"
#undef VMFUNC
};
#define NFUNC ((int)(sizeof (fdesc) / sizeof (fdesc[0])))

enum {
#define VMFUNC(id, name, nin, kind, ulp, ref) FID_##name = id,
#include "vmfuncs.def"
#undef VMFUNC
};

/* ------------------------------------------------------------------ */
/* RNG: xoshiro256**                                                   */
/* ------------------------------------------------------------------ */

static uint64_t rng_s[4];

static inline uint64_t
rotl (const uint64_t x, int k)
{
  return (x << k) | (x >> (64 - k));
}

static uint64_t
rng_next (void)
{
  const uint64_t result = rotl (rng_s[1] * 5, 7) * 9;
  const uint64_t t = rng_s[1] << 17;
  rng_s[2] ^= rng_s[0];
  rng_s[3] ^= rng_s[1];
  rng_s[1] ^= rng_s[2];
  rng_s[0] ^= rng_s[3];
  rng_s[2] ^= t;
  rng_s[3] = rotl (rng_s[3], 45);
  return result;
}

static void
rng_seed (uint64_t seed)
{
  /* splitmix64 to fill the state.  */
  for (int i = 0; i < 4; i++)
    {
      seed += 0x9e3779b97f4a7c15ULL;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      rng_s[i] = z ^ (z >> 31);
    }
}

static uint64_t
fnv1a (const char *s)
{
  uint64_t h = 0xcbf29ce484222325ULL;
  while (*s)
    {
      h ^= (uint64_t) (unsigned char) *s++;
      h *= 0x100000001b3ULL;
    }
  return h;
}

/* Uniform double in [0,1).  */
static double
rng_unit (void)
{
  return (double) (rng_next () >> 11) * 0x1p-53;
}

/* ------------------------------------------------------------------ */
/* Bit helpers                                                         */
/* ------------------------------------------------------------------ */

static inline uint64_t
dbits (double d)
{
  uint64_t u;
  memcpy (&u, &d, 8);
  return u;
}

static inline double
bdouble (uint64_t u)
{
  double d;
  memcpy (&d, &u, 8);
  return d;
}

/* ------------------------------------------------------------------ */
/* Special-value battery                                               */
/* ------------------------------------------------------------------ */

/* sign of Gamma(x) for the DUAL_INT record, set by compute_record */
static int32_t rec2i_sign = 1;

static double specials[64];
static int nspecials;

static void
add_special (double d)
{
  if (nspecials < (int)(sizeof specials / sizeof specials[0]))
    specials[nspecials++] = d;
}

static void
init_specials (void)
{
  nspecials = 0;
  double qnan = bdouble (0x7ff8000000000000ULL);
  double snan = bdouble (0x7ff0000000000001ULL);
  double nanp = bdouble (0x7ff8123456789abcULL);	/* qNaN with payload */
  double nanq = bdouble (0xfff8000000000000ULL);	/* negative qNaN */
  add_special (0.0);
  add_special (-0.0);
  add_special (INFINITY);
  add_special (-INFINITY);
  add_special (qnan);
  add_special (-qnan);
  add_special (snan);
  add_special (-snan);
  add_special (nanp);
  add_special (nanq);
  add_special (DBL_MIN);	/* 2^-1022 */
  add_special (-DBL_MIN);
  add_special (DBL_MAX);
  add_special (-DBL_MAX);
  add_special (DBL_TRUE_MIN);	/* 2^-1074 */
  add_special (-DBL_TRUE_MIN);
  add_special (bdouble (0x0008000000000000ULL));	/* 2^-1023 subnormal */
  add_special (bdouble (0x000fffffffffffffULL));	/* largest subnormal */
  add_special (bdouble (0x0010000000000001ULL));	/* min normal + 1ulp */
  add_special (1.0);
  add_special (-1.0);
  add_special (0.5);
  add_special (-0.5);
  add_special (2.0);
  add_special (-2.0);
  add_special (nextafter (1.0, 2.0));		/* 1+2^-52 */
  add_special (nextafter (1.0, 0.0));		/* 1-2^-53 */
  add_special (nextafter (-1.0, -2.0));
  add_special (3.0);
  add_special (-3.0);
  add_special (M_PI);
  add_special (-M_PI);
  add_special (M_PI_2);
  add_special (-M_PI_2);
  add_special (M_PI_4);
  add_special (709.782712893384);	/* exp overflow boundary */
  add_special (-745.133219101941);	/* exp underflow boundary */
  add_special (171.624376956302);	/* tgamma overflow boundary */
  add_special (-171.624376956302);
  add_special (0.4375);			/* atan region edge */
  add_special (2.4375);
  add_special (0.84375);		/* erf region edge */
  add_special (1.25);
  add_special (28.0);
  add_special (22.0);			/* sinh/tanh region edge */
  add_special (710.475860073944);	/* sinh overflow boundary */
  for (int e = -30; e <= 30; e += 6)	/* a spread of powers of two */
    add_special (ldexp (1.0, e));
}

/* ------------------------------------------------------------------ */
/* Input distributions                                                 */
/* ------------------------------------------------------------------ */

/* Draw one "interesting finite" double.  Mixture:
     0: uniform [-1,1]
     1: uniform [-10,10]
     2: log-uniform over the whole exponent range (incl subnormals)
     3: finite uniform bit patterns (any class)
     4: clustered around a random special (±128 ulp)  */
static double
draw_input (void)
{
  switch (rng_next () % 5)
    {
    case 0:
      return 2.0 * rng_unit () - 1.0;
    case 1:
      return 20.0 * rng_unit () - 10.0;
    case 2:
      {
	int e = (int) (rng_next () % 2105) - 1080;	/* -1080..1024 */
	double m = 1.0 + rng_unit ();
	double d = ldexp (m, e);
	return (rng_next () & 1) ? -d : d;
      }
    case 3:
      {
	uint64_t u = rng_next () & 0x7fffffffffffffffULL;
	if ((u & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL)
	  u &= 0x7fefffffffffffffULL;	/* keep finite */
	double d = bdouble (u);
	return (rng_next () & 1) ? -d : d;
      }
    default:
      {
	double base = specials[rng_next () % (uint64_t) nspecials];
	if (!isfinite (base))
	  base = 1.0;
	long k = (long) (rng_next () % 257) - 128;
	double d = base;
	while (k > 0) { d = nextafter (d, INFINITY); k--; }
	while (k < 0) { d = nextafter (d, -INFINITY); k++; }
	return (rng_next () & 1) ? -d : d;
      }
    }
}

/* Second input for binary functions (correlated half the time).  */
static double
draw_input2 (double a)
{
  if (rng_next () & 1)
    {
      switch (rng_next () % 6)
	{
	case 0: return a;
	case 1: return 2.0 * a;
	case 2: return a == 0.0 ? 1.0 : 1.0 / a;
	case 3: return -a;
	case 4: return floor (a);
	default: return a * a;
	}
    }
  return draw_input ();
}

/* ------------------------------------------------------------------ */
/* Per-function breakpoint clusters: +-64 ulp around region edges and
   over/underflow boundaries.  */

static double
ulp_walk (double base, long k)
{
  if (!isfinite (base))
    base = 1.0;
  while (k > 0) { base = nextafter (base, INFINITY); k--; }
  while (k < 0) { base = nextafter (base, -INFINITY); k++; }
  return base;
}

static double
cluster_draw (int id)
{
  static const double c_exp[] = {
    709.78271289338397, -745.13321910194122, 708.0, -744.0,
    512.0, -512.0, 709.0, -745.0,
  };
  static const double c_exp2[] = {
    1024.0, -1075.0, 1023.0, -1074.0, 512.0, -512.0, 928.0, -928.0,
  };
  static const double c_exp10[] = {
    308.25471249634239, -322.3, 308.0, -322.0, 256.0, -256.0, 309.0,
  };
  static const double c_log[] = {
    1.0, 0.9375, 1.064697265625, DBL_MIN, 2.0, 0.5, 1e308, 1e-308,
  };
  static const double c_log2[] = {
    1.0, 0.95758438110351562, 1.04435729980468750, 1024.0, -1074.0,
    DBL_MIN, 2.0, 0.5,
  };
  static const double c_log10[] = {
    1.0, 10.0, 100.0, 0.1, DBL_MIN, 1e300, 1e-300, 2.0,
  };
  static const double c_log1p[] = {
    0.0, -1.0, 0.41421356237, -0.29289321881, 2.0e-29, 1.0e-54,
    9007199254740992.0, -0.99999999999999989, 1.0,
  };
  static const double c_trig[] = {
    /* pi/4, pi/2, pi, 2^20 (medium/large switch), multiples of pi/2 */
    0.78539816339744828, -0.78539816339744828, 1.5707963267948966,
    -1.5707963267948966, 3.1415926535897931, -3.1415926535897931,
    1048576.0, 1048576.5, 6.2831853071795862, 1.0, 2.0, 3.0,
  };
  static const double c_expm1[] = {
    /* e_expm1.c region edges: ln2/2 and 1.5*ln2, 56*ln2, the overflow
       threshold */
    0.34657359027997264, -0.34657359027997264, 0.34657359027997265,
    -0.34657359027997265, 1.0397207708399179, -1.0397207708399179,
    38.816705065180372, -38.816705065180372, 0.69314718055994531,
    1e-20, -1e-20, 709.78271289338397,
  };
  static const double c_hyper[] = {
    /* s_tanh/e_sinh/e_cosh region edges and the overflow boundary */
    0.5, -0.5, 1.0, -1.0, 22.0, -22.0, 0.34657359027997264,
    709.78271289338397, -709.78271289338397, 710.47586007394395,
    710.0, -710.0, 20.0, -20.0, 1e-30, 0.0,
  };
  static const double c_asin[] = {
    /* asin/acos: |x| = 1 and the 0.975 split fdlibm uses near it */
    0.0, 0.5, -0.5, 0.70710678118654752, -0.70710678118654752,
    0.975, -0.975, 0.99999999999999989, -0.99999999999999989,
    1.0, -1.0, 2.0,
  };
  static const double c_hypot[] = {
    /* the dd range ends at 2^500 and 2^-500 */
    0.0, 1.0, -1.0, 3.0, 4.0, 5e-324, 1e-308, 1e308,
    3.2733906078961420e+150, -3.2733906078961420e+150,
    3.0549363634996047e-151, 1e-151, 1e150,
  };
  static const double c_hinv[] = {
    /* s_asinh.c / e_acosh.c / e_atanh.c region edges and domain ends */
    1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 4.0, -4.0, 268435456.0,
    1.0000000000000002, 0.99999999999999989, 3.7252902984619141e-09,
    0.0, 1e-30, 1e300,
  };
  static const double c_gamma[] = {
    /* the kernel's region edges, both signs */
    0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 10.0, -10.0, 1.0, -1.0, 2.0, -2.0,
    /* tgamma's overflow edge and the exp's guard */
    171.0, 171.624376956302, 172.0, 708.5, 709.0, -708.5, -709.0,
    /* lgamma has zeros on the NEGATIVE axis (|Gamma| = 1), where the
       relative ulp metric is delicate; these points would otherwise
       never be sampled.  */
    -1.5615, -2.4, -3.56, -4.6, -5.58, -6.6, -7.58, -8.6, -9.58, -11.6,
    /* poles and near-poles */
    -1.0000000000000002, -0.99999999999999989,
    -2.0000000000000004, -1.9999999999999998, -20.0, -21.0,
    /* subnormal arguments */
    5e-324, -5e-324, 1e-308, -1e-308, 1e-30, -1e-30, 1e300, -1e300,
  };
  static const double c_erf[] = {
    /* the |x| = 1 split, the 0.5 crossover, the rational's cancellation
       point at 20.5, the x = 40 clamp, and the onset of exp(-x^2)
       underflow just past 26.6 */
    0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0,
    5.0, -5.0, 6.0, -6.0, 20.5, -20.5, 26.0, -26.0, 27.0, -27.0,
    0.99999999999999989, -0.99999999999999989, 40.0, -40.0, 41.0, -41.0,
    100.0, -100.0, 0.0, 1e-30, -1e-30,
  };
  static const double c_atan[] = {
    /* region edges of s_atan.c and e_atan2.c fixups */
    0.4375, -0.4375, 0.6875, -0.6875, 1.1875, -1.1875, 2.4375, -2.4375,
    1.0, -1.0, 2.0, -2.0, 1e15, -1e15, 1e-15, 0.0,
  };
  static const double c_pi[] = {
    /* quadrant edges, integers, half-integers, the 2^62 exact cutoff */
    0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.5, -1.5,
    2.0, 3.0, 1048576.5, 4611686018427387904.0,
  };
  const double *c;
  int n;
  switch (id)
    {
    case FID_sin: case FID_cos: case FID_tan: case FID_sincos:
      c = c_trig; n = 12; break;
    case FID_sinpi: case FID_cospi: case FID_tanpi: case FID_sincospi:
      c = c_pi; n = 14; break;
    case FID_asin: case FID_acos:
    case FID_asinpi: case FID_acospi:
      c = c_asin; n = 12; break;
    case FID_erf: case FID_erfc:
      c = c_erf; n = 33; break;
    case FID_tgamma: case FID_lgamma: case FID_lgamma_r:
      c = c_gamma; n = (int) (sizeof c_gamma / sizeof c_gamma[0]); break;
    case FID_hypot:
      c = c_hypot; n = 13; break;
    case FID_asinh: case FID_acosh: case FID_atanh:
      c = c_hinv; n = 15; break;
    case FID_atan: case FID_atan2: case FID_atanpi: case FID_atan2pi:
      c = c_atan; n = 16; break;
    case FID_sinh: case FID_cosh: case FID_tanh:
      c = c_hyper; n = 16; break;
    case FID_expm1:
      c = c_expm1; n = 12; break;
    case FID_exp:  c = c_exp;   n = 8; break;
    case FID_exp2: c = c_exp2;  n = 8; break;
    case FID_exp10: c = c_exp10; n = 7; break;
    case FID_log:  c = c_log;   n = 8; break;
    case FID_log2: c = c_log2;  n = 8; break;
    case FID_log10: c = c_log10; n = 8; break;
    case FID_log1p:
    case FID_log2p1: c = c_log1p; n = 9; break;
    default: return draw_input ();
    }
  double base = c[rng_next () % (uint64_t) n];
  long k = (long) (rng_next () % 129) - 64;
  return ulp_walk (base, k);
}

/* ------------------------------------------------------------------ */
/* dd split of an MPFR value                                           */
/* ------------------------------------------------------------------ */

static void
dd_split (mpfr_srcptr t, uint64_t *hi, uint64_t *lo)
{
  double h = mpfr_get_d (t, MPFR_RNDN);
  mpfr_t r;
  mpfr_init (r);
  mpfr_set_d (r, h, MPFR_RNDN);
  mpfr_sub (r, t, r, MPFR_RNDN);
  double l = mpfr_get_d (r, MPFR_RNDN);
  /* Renormalize so that |l| <= 2^-53 |h| holds structurally.  */
  *hi = dbits (h);
  *lo = dbits (l);
  mpfr_clear (r);
}

/* ------------------------------------------------------------------ */
/* Record computation: exact reference + classification + flags.       */
/* Returns 0 on success, 1 if the function is not implemented yet.     */
/* ------------------------------------------------------------------ */

/* ---- helpers for pow's C99 special-case matrix ---- */

/* Exact half-integer test.  `a - floor(a) == 0.5` is NOT exact -- e.g.
   for a = -(0.5 - 2^-54) the subtraction rounds back to 0.5 -- so compare
   against trunc(a) +- 0.5 (exact comparisons), excluding integers (where
   the addition itself rounds for |a| >= 2^52).  */
static int
is_half_int (double a)
{
  if (a != a || a == floor (a))
    return 0;
  return a == trunc (a) + 0.5 || a == trunc (a) - 0.5;
}

/* 0 = not an integer, 1 = odd integer, 2 = even integer.  */
static int
int_kind (double d)
{
  if (!isfinite (d) || d != floor (d))
    return 0;
  if (fabs (d) >= 0x1p53)
    return 2;
  long long k = (long long) d;
  return (k & 1) ? 1 : 2;
}

#define REF_BITS 200

/* Signaling NaN: exponent all ones, quiet bit clear, payload nonzero.  */
static inline int
is_snan (double d)
{
  uint64_t u = dbits (d);
  return ((u & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL)
	 && (u & 0x000fffffffffffffULL) != 0
	 && !(u & 0x0008000000000000ULL);
}

/* Common NaN-input expectations: qNaN result; invalid raised iff the
   input was signaling.  */
static void
nan_result (vmb_rec1_t *rec, double a)
{
  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
  rec->fpflags = is_snan (a) ? VMB_F_INVALID : 0;
}

static int
compute_record (int id, double a, double b, vmb_rec1_t *rec,
		vmb_rec2_t *rec2)
{
  mpfr_t t, ma, mb;
  mpfr_inits2 (REF_BITS, t, ma, mb, (mpfr_ptr) 0);
  mpfr_set_d (ma, a, MPFR_RNDN);
  mpfr_set_d (mb, b, MPFR_RNDN);

  memset (rec, 0, sizeof *rec);
  if (rec2)
    memset (rec2, 0, sizeof *rec2);
  rec->in0 = dbits (a);
  rec->in1 = dbits (b);
  rec->ref_hi = VMB_POISON;
  rec->ref_lo = 0;
  rec->exact_bits = VMB_POISON;

  int handled = 0;

  /* ---- per-function exact evaluation + special-value semantics ---- */
  switch (id)
    {
    case FID_sqrt:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (signbit (a) && a != 0.0)		/* a < 0 or -inf */
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (a);		/* +-0 preserved */
	}
      else if (isinf (a))			/* +inf */
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (INFINITY);
	}
      else
	mpfr_sqrt (t, ma, MPFR_RNDN);
      break;

    case FID_rsqrt:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* rsqrt(+-0) = +inf */
	  rec->exact_bits = dbits (INFINITY);
	  rec->fpflags = VMB_F_DIVBYZERO;
	}
      else if (signbit (a))			/* a < 0 or -inf */
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else if (isinf (a))
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* rsqrt(+inf) = +0 */
	  rec->exact_bits = dbits (0.0);
	}
      else
	{
	  mpfr_sqrt (t, ma, MPFR_RNDN);
	  mpfr_ui_div (t, 1, t, MPFR_RNDN);
	}
      break;

    case FID_cbrt:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (isinf (a))
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (a);
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (a);
	}
      else
	mpfr_cbrt (t, ma, MPFR_RNDN);
      break;

    case FID_exp:
    case FID_exp2:
    case FID_exp10:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == INFINITY)
	{
	  rec->expect = VMB_EXPECT_INF;	/* exact inf, no overflow flag */
	  rec->exact_bits = dbits (INFINITY);
	}
      else if (a == -INFINITY)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* exact +0, no flags */
	  rec->exact_bits = dbits (0.0);
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* exp(+-0) = 1 exactly */
	  rec->exact_bits = dbits (1.0);
	}
      else if (id == FID_exp2 && a == floor (a) && a >= -1074.0 && a <= -1023.0)
	{
	  /* exp2's specialcase() subnormal branch unconditionally runs
	     force_eval(0x1p-1022 * 0x1p-1022), so it raises UNDERFLOW even
	     when the exact subnormal power-of-two result is representable.  */
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (exp2 (a));
	  rec->fpflags = VMB_F_UNDERFLOW;
	}
      else if (id == FID_exp)
	mpfr_exp (t, ma, MPFR_RNDN);
      else if (id == FID_exp2)
	mpfr_exp2 (t, ma, MPFR_RNDN);
      else
	{
	  mpfr_t ten;
	  mpfr_init2 (ten, 8);
	  mpfr_set_ui (ten, 10, MPFR_RNDN);
	  mpfr_pow (t, ten, ma, MPFR_RNDN);
	  mpfr_clear (ten);
	}
      break;

    case FID_expm1:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == INFINITY)
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (INFINITY);
	}
      else if (a == -INFINITY)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* exactly -1, no flags */
	  rec->exact_bits = dbits (-1.0);
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (a);
	}
      else
	mpfr_expm1 (t, ma, MPFR_RNDN);
      break;

    case FID_acospi:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == 1.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* acos(1)/pi = +0 */
	  rec->exact_bits = dbits (0.0);
	}
      else if (a > 1.0 || a < -1.0)
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else
	{
	  mpfr_t p;
	  mpfr_init2 (p, REF_BITS);
	  mpfr_const_pi (p, MPFR_RNDN);
	  mpfr_acos (t, ma, MPFR_RNDN);
	  mpfr_div (t, t, p, MPFR_RNDN);
	  mpfr_clear (p);
	}
      break;

    case FID_asinpi:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (a);
	}
      else if (a > 1.0 || a < -1.0)
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else
	{
	  mpfr_t p;
	  mpfr_init2 (p, REF_BITS);
	  mpfr_const_pi (p, MPFR_RNDN);
	  mpfr_asin (t, ma, MPFR_RNDN);
	  mpfr_div (t, t, p, MPFR_RNDN);
	  mpfr_clear (p);
	}
      break;

    case FID_asin:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* asin(+-0) = +-0 */
	  rec->exact_bits = dbits (a);
	}
      else if (a > 1.0 || a < -1.0)
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else
	mpfr_asin (t, ma, MPFR_RNDN);
      break;

    case FID_acos:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == 1.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* acos(1) = +0 */
	  rec->exact_bits = dbits (0.0);
	}
      else if (a > 1.0 || a < -1.0)
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else
	mpfr_acos (t, ma, MPFR_RNDN);
      break;

    case FID_erf:
    case FID_erfc:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (isinf (a))
	{
	  /* C99 F.10.5.1/F.10.5.2, all exact and flag-free.  */
	  rec->expect = VMB_EXPECT_EXACT;
	  if (id == FID_erf)
	    rec->exact_bits = dbits (a > 0 ? 1.0 : -1.0);
	  else
	    rec->exact_bits = dbits (a > 0 ? 0.0 : 2.0);
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (id == FID_erf ? a : 1.0);
	}
      else if (id == FID_erf)
	mpfr_erf (t, ma, MPFR_RNDN);
      else
	mpfr_erfc (t, ma, MPFR_RNDN);
      break;

    case FID_tgamma:
      handled = 1;
      /* C99 F.10.4.1.  The expected class is the standard's, not the
	 scalar's (which reaches the poles through exp(lgamma) and
	 returns +-Inf).  */
      if (isnan (a))
	nan_result (rec, a);
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (signbit (a) ? -INFINITY : INFINITY);
	  rec->fpflags = VMB_F_DIVBYZERO;
	}
      else if (isinf (a))
	{
	  if (a > 0)
	    {
	      rec->expect = VMB_EXPECT_INF;
	      rec->exact_bits = dbits (INFINITY);
	    }
	  else
	    {
	      rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	      rec->fpflags = VMB_F_INVALID;
	    }
	}
      else if (a < 0.0 && a == floor (a))
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else
	mpfr_gamma (t, ma, MPFR_RNDN);
      break;

    case FID_lgamma:
    case FID_lgamma_r:
      handled = 1;
      rec2i_sign = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (isinf (a))
	{
	  /* lgamma(+-Inf) = +Inf, exact; *signgamp stays at its initial 1 */
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (INFINITY);
	}
      else if (a == 0.0 || (a < 0.0 && a == floor (a)))
	{
	  /* a pole: +Inf with divide-by-zero.  Only -0 carries the sign
	     into signgam, so it is -1 there and +1 at every other pole.  */
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (INFINITY);
	  rec->fpflags = VMB_F_DIVBYZERO;
	  if (a == 0.0 && signbit (a))
	    rec2i_sign = -1;
	}
      else
	{
	  int sg = 1;
	  mpfr_lgamma (t, &sg, ma, MPFR_RNDN);
	  rec2i_sign = (a > 0.0) ? 1 : sg;
	}
      break;

    case FID_hypot:
      handled = 1;
      /* C99: an infinite argument wins even over a NaN one.  */
      if (isinf (a) || isinf (b))
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (INFINITY);
	}
      else if (isnan (a) || isnan (b))
	nan_result (rec, isnan (a) ? a : b);
      else if (a == 0.0 && b == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (0.0);
	}
      else
	mpfr_hypot (t, ma, mb, MPFR_RNDN);
      break;

    case FID_asinh:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (isinf (a))
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (a);
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (a);
	}
      else
	mpfr_asinh (t, ma, MPFR_RNDN);
      break;

    case FID_acosh:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a < 1.0)			/* covers -Inf as well */
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else if (a == 1.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (0.0);
	}
      else if (a == INFINITY)
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (INFINITY);
	}
      else
	mpfr_acosh (t, ma, MPFR_RNDN);
      break;

    case FID_atanh:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a > 1.0 || a < -1.0)	/* covers +-Inf as well */
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else if (a == 1.0 || a == -1.0)
	{
	  rec->expect = VMB_EXPECT_INF;	/* +-Inf, sign from exact_bits */
	  rec->exact_bits = dbits (a);
	  rec->fpflags = VMB_F_DIVBYZERO;
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (a);
	}
      else
	mpfr_atanh (t, ma, MPFR_RNDN);
      break;

    case FID_log:
    case FID_log2:
    case FID_log10:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == INFINITY)
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (INFINITY);
	}
      else if (a == 0.0)
	{
	  /* log*(+-0) = -inf with divbyzero.  */
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (-INFINITY);
	  rec->fpflags = VMB_F_DIVBYZERO;
	}
      else if (a < 0.0 || a == -INFINITY)
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else if (id == FID_log)
	mpfr_log (t, ma, MPFR_RNDN);
      else if (id == FID_log2)
	mpfr_log2 (t, ma, MPFR_RNDN);
      else
	mpfr_log10 (t, ma, MPFR_RNDN);
      break;

    case FID_log1p:
    case FID_log2p1:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == INFINITY)
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (INFINITY);
	}
      else if (a == -1.0)
	{
	  rec->expect = VMB_EXPECT_INF;	/* -inf with divbyzero */
	  rec->exact_bits = dbits (-INFINITY);
	  rec->fpflags = VMB_F_DIVBYZERO;
	}
      else if (a < -1.0 || a == -INFINITY)
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* +-0 preserved */
	  rec->exact_bits = dbits (a);
	}
      else if (id == FID_log1p)
	mpfr_log1p (t, ma, MPFR_RNDN);
      else
	{
	  /* log2p1 = log1p(x)/ln2 at reference precision (never form
	     1+x: it would lose tiny x).  */
	  mpfr_t ln2v, num;
	  mpfr_init2 (ln2v, REF_BITS);
	  mpfr_init2 (num, REF_BITS);
	  mpfr_log_ui (ln2v, 2, MPFR_RNDN);
	  mpfr_log1p (num, ma, MPFR_RNDN);
	  mpfr_div (t, num, ln2v, MPFR_RNDN);
	  mpfr_clears (ln2v, num, (mpfr_ptr) 0);
	}
      break;

    case FID_pow:
      handled = 1;
      {
	/* C99 F.9.4.4 special cases (bit-exact classes + flags).  */
	int ya = int_kind (b);		/* exponent parity */
	if (isnan (a) || isnan (b))
	  {
	    /* pow(1, NaN) = 1; pow(NaN, 0) = 1; else NaN (invalid only
	       for a signaling NaN operand).  */
	    if (a == 1.0 || b == 0.0)
	      {
		rec->expect = VMB_EXPECT_EXACT;
		rec->exact_bits = dbits (1.0);
	      }
	    else
	      nan_result (rec, isnan (a) ? a : b);
	  }
	else if (b == 0.0)
	  {
	    rec->expect = VMB_EXPECT_EXACT;	/* pow(x, +-0) = 1 */
	    rec->exact_bits = dbits (1.0);
	  }
	else if (a == 1.0)
	  {
	    rec->expect = VMB_EXPECT_EXACT;	/* pow(1, y) = 1 */
	    rec->exact_bits = dbits (1.0);
	  }
	else if (isinf (b))
	  {
	    double ax = fabs (a);
	    if (ax == 1.0)
	      {
		rec->expect = VMB_EXPECT_EXACT;
		rec->exact_bits = dbits (1.0);
	      }
	    else if ((ax > 1.0) == (b > 0.0))
	      {
		rec->expect = VMB_EXPECT_INF;
		rec->exact_bits = dbits (INFINITY);
	      }
	    else
	      {
		rec->expect = VMB_EXPECT_EXACT;	/* exact +0, no flags */
		rec->exact_bits = dbits (0.0);
	      }
	  }
	else if (a == 0.0)
	  {
	    int odd = ya == 1;
	    int neg = signbit (a);
	    if (b > 0.0)
	      {
		rec->expect = VMB_EXPECT_EXACT;
		rec->exact_bits = dbits ((neg && odd) ? -0.0 : 0.0);
	      }
	    else
	      {
		rec->expect = VMB_EXPECT_INF;
		rec->exact_bits = dbits ((neg && odd) ? -INFINITY
						      : INFINITY);
		rec->fpflags = VMB_F_DIVBYZERO;
	      }
	  }
	else if (isinf (a))
	  {
	    int odd = ya == 1;
	    int neg = signbit (a);
	    if (b > 0.0)
	      {
		rec->expect = VMB_EXPECT_INF;
		rec->exact_bits = dbits ((neg && odd) ? -INFINITY
						      : INFINITY);
	      }
	    else
	      {
		rec->expect = VMB_EXPECT_EXACT;
		rec->exact_bits = dbits ((neg && odd) ? -0.0 : 0.0);
	      }
	  }
	else if (a < 0.0 && ya == 0)
	  {
	    rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	    rec->fpflags = VMB_F_INVALID;
	  }
	else
	  {
	    /* Finite non-special: b == 1 is exact, else MPFR.  */
	    if (b == 1.0)
	      {
		rec->expect = VMB_EXPECT_EXACT;
		rec->exact_bits = dbits (a);
		/* subnormal-x lanes still take scalar pow's subnormal
		   specialcase, whose force_eval raises UNDERFLOW even
		   though x^1 == x exactly.  */
		if (fabs (a) < DBL_MIN)
		  rec->fpflags = VMB_F_UNDERFLOW;
	      }
	    else
	      {
		mpfr_t ma2;
		mpfr_init2 (ma2, 64);
		mpfr_set_d (ma2, a, MPFR_RNDN);
		mpfr_pow (t, ma2, mb, MPFR_RNDN);
		mpfr_clear (ma2);
	      }
	  }
      }
      break;

    case FID_sin:
    case FID_cos:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (isinf (a))
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;	/* sin/cos(inf) = NaN + invalid */
	}
      else if (a == 0.0)
	{
	  /* sin(+-0) = +-0 exactly; cos(+-0) = 1.  */
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (id == FID_sin ? a : 1.0);
	}
      else if (id == FID_sin)
	mpfr_sin (t, ma, MPFR_RNDN);
      else
	mpfr_cos (t, ma, MPFR_RNDN);
      break;

    case FID_tan:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (isinf (a))
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* tan(+-0) = +-0 */
	  rec->exact_bits = dbits (a);
	}
      else
	mpfr_tan (t, ma, MPFR_RNDN);
      break;

    case FID_sinh:
    case FID_cosh:
    case FID_tanh:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (isinf (a))
	{
	  /* sinh(+-Inf) = +-Inf, cosh(+-Inf) = +Inf (both exact infinities,
	     no flags), tanh(+-Inf) = +-1 exactly.  */
	  if (id == FID_tanh)
	    {
	      rec->expect = VMB_EXPECT_EXACT;
	      rec->exact_bits = dbits (a > 0 ? 1.0 : -1.0);
	    }
	  else
	    {
	      rec->expect = VMB_EXPECT_INF;
	      rec->exact_bits = dbits (id == FID_cosh ? INFINITY : a);
	    }
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (id == FID_cosh ? 1.0 : a);
	}
      else if (id == FID_sinh)
	mpfr_sinh (t, ma, MPFR_RNDN);
      else if (id == FID_cosh)
	mpfr_cosh (t, ma, MPFR_RNDN);
      else
	mpfr_tanh (t, ma, MPFR_RNDN);
      break;

    case FID_atan:
    case FID_atanpi:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;	/* atan(+-0) = +-0 */
	  rec->exact_bits = dbits (a);
	}
      else if (id == FID_atan)
	mpfr_atan (t, ma, MPFR_RNDN);
      else
	{
	  mpfr_t p;
	  mpfr_init2 (p, REF_BITS);
	  mpfr_const_pi (p, MPFR_RNDN);
	  mpfr_atan (t, ma, MPFR_RNDN);
	  mpfr_div (t, t, p, MPFR_RNDN);
	  mpfr_clear (p);
	}
      break;

    case FID_atan2:
    case FID_atan2pi:
      handled = 1;
      if (isnan (a) || isnan (b))
	nan_result (rec, isnan (a) ? a : b);
      else
	{
	  /* MPFR's atan2 reproduces the C99 F.10.1.4 edge cases, so the
	     generic post-processing classifies +-0/+-pi/+-pi/2/+-3pi/4
	     for us.  */
	  mpfr_atan2 (t, ma, mb, MPFR_RNDN);
	  if (id == FID_atan2pi)
	    {
	      mpfr_t p;
	      mpfr_init2 (p, REF_BITS);
	      mpfr_const_pi (p, MPFR_RNDN);
	      mpfr_div (t, t, p, MPFR_RNDN);
	      mpfr_clear (p);
	    }
	}
      break;

    case FID_sinpi:
    case FID_cospi:
    case FID_tanpi:
      handled = 1;
      if (isnan (a))
	nan_result (rec, a);
      else if (isinf (a))
	{
	  rec->expect = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	  rec->fpflags = VMB_F_INVALID;
	}
      else if (a == 0.0)
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (id == FID_cospi ? 1.0 : a);
	}
      else if (a == floor (a))
	{
	  /* OpenCL/SYCL: sinpi(+-n) = +-0, cospi(n) = (-1)^n,
	     tanpi(n) = copysign(0, x)*(-1)^n.  All exact, no flags.  */
	  int odd = fmod (fabs (a), 2.0) == 1.0;
	  rec->expect = VMB_EXPECT_EXACT;
	  if (id == FID_cospi)
	    rec->exact_bits = dbits (odd ? -1.0 : 1.0);
	  else if (id == FID_sinpi)
	    rec->exact_bits = dbits (copysign (0.0, a));
	  else
	    rec->exact_bits = dbits (copysign (0.0, odd ? -a : a));
	}
      else if (is_half_int (a))
	{
	  if (id == FID_cospi)
	    {
	      rec->expect = VMB_EXPECT_EXACT;	/* cospi(n + 1/2) = +0 */
	      rec->exact_bits = dbits (0.0);
	    }
	  else if (id == FID_tanpi)
	    {
	      /* tanpi(n + 1/2) = (-1)^n * +Inf, n = floor(x), divbyzero.
		 (n + 1/2 must be representable; the fold guarantees it.)  */
	      int odd = fmod (fabs (floor (a)), 2.0) == 1.0;
	      rec->expect = VMB_EXPECT_INF;
	      rec->exact_bits = dbits (odd ? -INFINITY : INFINITY);
	      rec->fpflags = VMB_F_DIVBYZERO;
	    }
	  else
	    mpfr_sinpi (t, ma, MPFR_RNDN);	/* exactly +-1 */
	}
      else if (id == FID_sinpi)
	mpfr_sinpi (t, ma, MPFR_RNDN);
      else if (id == FID_cospi)
	mpfr_cospi (t, ma, MPFR_RNDN);
      else
	mpfr_tanpi (t, ma, MPFR_RNDN);
      break;

    case FID_sincospi:
      handled = 1;
      if (rec2 == NULL)
	{
	  mpfr_clears (t, ma, mb, (mpfr_ptr) 0);
	  return 2;
	}
      {
	uint32_t e1 = 0, f1 = 0, e2 = 0, f2 = 0;
	uint64_t x1 = VMB_POISON, x2 = VMB_POISON;
	int need1 = 0, need2 = 0;
	if (isnan (a))
	  {
	    e1 = e2 = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	    f1 = f2 = is_snan (a) ? VMB_F_INVALID : 0;
	  }
	else if (isinf (a))
	  {
	    e1 = e2 = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	    f1 = f2 = VMB_F_INVALID;
	  }
	else if (a == 0.0)
	  {
	    e1 = e2 = VMB_EXPECT_EXACT;
	    x1 = dbits (a);
	    x2 = dbits (1.0);
	  }
	else if (a == floor (a))
	  {
	    int odd = fmod (fabs (a), 2.0) == 1.0;
	    e1 = e2 = VMB_EXPECT_EXACT;
	    x1 = dbits (copysign (0.0, a));
	    x2 = dbits (odd ? -1.0 : 1.0);
	  }
	else if (is_half_int (a))
	  {
	    e2 = VMB_EXPECT_EXACT;		/* cospi = +0 */
	    x2 = dbits (0.0);
	    need1 = 1;				/* sinpi = +-1 via MPFR */
	  }
	else
	  need1 = need2 = 1;
	rec->expect = e1;
	rec->exact_bits = x1;
	rec->fpflags = f1 | f2;
	rec2->exact2_bits = x2;
	if (need1)
	  {
	    mpfr_sinpi (t, ma, MPFR_RNDN);
	    /* same subnormal rule as the generic post-processing: the sinpi
	       half underflows exactly when it lands below DBL_MIN while
	       being inexact, and the lane carries one flag word.  */
	    if (!mpfr_zero_p (t))
	      {
		double h = mpfr_get_d (t, MPFR_RNDN);
		mpfr_t td;
		mpfr_init2 (td, 64);
		mpfr_set_d (td, h, MPFR_RNDN);
		if (!mpfr_equal_p (t, td) && fabs (h) < DBL_MIN)
		  rec->fpflags |= VMB_F_UNDERFLOW;
		mpfr_clear (td);
	      }
	    dd_split (t, &rec->ref_hi, &rec->ref_lo);
	  }
	if (need2)
	  {
	    mpfr_cospi (t, ma, MPFR_RNDN);
	    dd_split (t, &rec2->ref2_hi, &rec2->ref2_lo);
	  }
	rec->expect |= (e2 & 0x1f) << VMB_EXPECT2_SHIFT;
	rec2->expect = rec->expect;
	rec2->fpflags = rec->fpflags;
	rec2->in0 = rec->in0;
	rec2->in1 = rec->in1;
	rec2->ref_hi = rec->ref_hi;
	rec2->ref_lo = rec->ref_lo;
	rec2->exact_bits = rec->exact_bits;
      }
      break;

    case FID_sincos:
      handled = 1;
      if (rec2 == NULL)
	{
	  mpfr_clears (t, ma, mb, (mpfr_ptr) 0);
	  return 2;		/* internal error: dual needs rec2 */
	}
      {
	uint32_t e1 = 0, f1 = 0, e2 = 0, f2 = 0;
	uint64_t x1 = VMB_POISON, x2 = VMB_POISON;
	if (isnan (a))
	  {
	    e1 = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	    e2 = e1;
	    f1 = is_snan (a) ? VMB_F_INVALID : 0;
	    f2 = f1;
	  }
	else if (isinf (a))
	  {
	    e1 = VMB_EXPECT_NAN | VMB_NAN_QUIET;
	    e2 = e1;
	    f1 = f2 = VMB_F_INVALID;
	  }
	else if (a == 0.0)
	  {
	    e1 = e2 = VMB_EXPECT_EXACT;
	    x1 = dbits (a);		/* sin(+-0) = +-0 */
	    x2 = dbits (1.0);		/* cos(+-0) = 1 */
	  }
	else
	  mpfr_sin (t, ma, MPFR_RNDN);
	rec->expect = e1;
	rec->exact_bits = x1;
	rec->fpflags = f1 | f2;		/* one gate per lane; both agree */
	rec2->exact2_bits = x2;
	if (!(e1 & (VMB_EXPECT_NAN | VMB_EXPECT_EXACT)))
	  dd_split (t, &rec->ref_hi, &rec->ref_lo);
	/* second output: cos */
	if (!(e2 & (VMB_EXPECT_NAN | VMB_EXPECT_EXACT)))
	  {
	    mpfr_cos (t, ma, MPFR_RNDN);
	    dd_split (t, &rec2->ref2_hi, &rec2->ref2_lo);
	  }
	/* Pack the second-output class into rec->expect bits 8..12; the dual
	   record is the vmb_rec2_t image, so mirror the shared prefix into
	   rec2 (whose ref2 fields already hold the second output).  */
	rec->expect |= (e2 & 0x1f) << VMB_EXPECT2_SHIFT;
	rec2->expect = rec->expect;
	rec2->fpflags = rec->fpflags;
	rec2->in0 = rec->in0;
	rec2->in1 = rec->in1;
	rec2->ref_hi = rec->ref_hi;
	rec2->ref_lo = rec->ref_lo;
	rec2->exact_bits = rec->exact_bits;
      }
      break;

    default:
      break;
    }

  if (!handled)
    {
      mpfr_clears (t, ma, mb, (mpfr_ptr) 0);
      return 1;
    }

  /* ---- generic finite-result post-processing (t holds the exact value;
     skipped when the special-value switch already filled the record).
     MPFR notes: (1) mpfr_get_d does NOT reliably set erange on double
     overflow -- compare against DBL_MAX instead; (2) results with true
     exponents below MPFR's emin (~2^-31) underflow to zero INSIDE mpfr
     with erange set -- that always implies double underflow too.  ---- */
  if (rec->expect == VMB_EXPECT_NORMAL)
    {
      mpfr_t td, dmax;
      mpfr_init2 (td, REF_BITS);
      mpfr_init2 (dmax, 64);
      mpfr_set_d (dmax, DBL_MAX, MPFR_RNDN);

      if (mpfr_inf_p (t))
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (mpfr_sgn (t) > 0 ? INFINITY : -INFINITY);
	  rec->fpflags |= VMB_F_OVERFLOW;
	}
      else if (mpfr_zero_p (t))
	{
	  rec->expect = VMB_EXPECT_EXACT;
	  rec->exact_bits = dbits (mpfr_signbit (t) ? -0.0 : 0.0);
	  /* Zero from a finite input: for exp/exp2/exp10 (and any strictly
	     positive-range function) the exact value can never be zero, so
	     this is MPFR-internal underflow (true exponent below MPFR's
	     emin, far below the double subnormal grid): double result is
	     +-0 with UNDERFLOW.  mpfr_get_d/mpfr_exp do NOT set erange for
	     this, so decide by the function's range instead.  Other
	     functions treat a zero result as exact (no flags) unless they
	     opt in.

	     erfc opts in on the same grounds: it is strictly positive on
	     the reals, and its zero is a rounded zero from a nonzero exact
	     value, i.e. tiny and inexact.  */
	  if (id == FID_exp || id == FID_exp2 || id == FID_exp10
	      || id == FID_pow || id == FID_erfc || id == FID_tgamma)
	    rec->fpflags |= VMB_F_UNDERFLOW;
	}
      else if (mpfr_cmpabs (t, dmax) > 0)
	{
	  rec->expect = VMB_EXPECT_INF;
	  rec->exact_bits = dbits (mpfr_sgn (t) > 0 ? INFINITY : -INFINITY);
	  rec->fpflags |= VMB_F_OVERFLOW;
	}
      else
	{
	  double h = mpfr_get_d (t, MPFR_RNDN);
	  if (h == 0.0)
	    {
	      /* Below the subnormal grid: rounds to +-0, UNDERFLOW.  */
	      rec->expect = VMB_EXPECT_EXACT;
	      rec->exact_bits = dbits (mpfr_signbit (t) ? -0.0 : 0.0);
	      rec->fpflags |= VMB_F_UNDERFLOW;
	    }
	  else
	    {
	      mpfr_set_d (td, h, MPFR_RNDN);
	      /* Bitwise-exact gating only for functions whose algorithms
		 are exact at representable points by construction.  Others
		 (log family, exp10, pow, ...) are held to the <=1 ulp
		 metric via the dd reference -- fdlibm/optimized-routines
		 scalars are not bitwise-exact at, e.g., powers of ten
		 either.  */
	      int exact_strict = (id == FID_sqrt || id == FID_rsqrt
				  || id == FID_cbrt || id == FID_exp2);
	      int is_exact = mpfr_equal_p (t, td) != 0;
	      if (exact_strict && is_exact)
		{
		  rec->expect = VMB_EXPECT_EXACT;
		  rec->exact_bits = dbits (h);
		}
	      else
		dd_split (t, &rec->ref_hi, &rec->ref_lo);
	      /* Underflow flag: result tiny (subnormal) AND inexact --
		 decided by true exactness, not by the gating policy.
		 Exception: scalar pow's specialcase() runs the
		 force_eval(0x1p-1022 * 0x1p-1022) trick unconditionally
		 on the subnormal branch, so any subnormal pow result
		 raises UNDERFLOW even when exactly representable.  */
	      if ((!is_exact || id == FID_pow) && fabs (h) < DBL_MIN)
		rec->fpflags |= VMB_F_UNDERFLOW;
	    }
	}
      mpfr_clears (td, dmax, (mpfr_ptr) 0);
    }

  /* lgamma_r is DUAL_INT: the blob's record is a vmb_rec2_t image, the
     value read from ref_hi/ref_lo there and the sign from the extra int.
     Mirror the value record into it.  The class is NOT shifted into the
     upper bits -- the second output is an int, checked separately against
     rec2i_sign, not a value with a class of its own.  */
  if (id == FID_lgamma_r && rec2 != NULL)
    {
      rec2->expect = rec->expect;
      rec2->fpflags = rec->fpflags;
      rec2->in0 = rec->in0;
      rec2->in1 = rec->in1;
      rec2->ref_hi = rec->ref_hi;
      rec2->ref_lo = rec->ref_lo;
      rec2->exact_bits = rec->exact_bits;
    }

  mpfr_clears (t, ma, mb, (mpfr_ptr) 0);
  return 0;
}

/* ------------------------------------------------------------------ */
/* Blob emission                                                       */
/* ------------------------------------------------------------------ */

static uint64_t
tier_count (const char *tier)
{
  return strcmp (tier, "small") == 0 ? 2000 : 100000;
}

static int
gen_function (int id, const char *tier, uint64_t seed, const char *outdir)
{
  const fdesc_t *f = &fdesc[id];
  uint64_t nrand = tier_count (tier);

  rng_seed (seed ^ fnv1a (f->name));

  char path[1024];
  snprintf (path, sizeof path, "%s/%s.bin", outdir, f->name);
  FILE *fp = fopen (path, "wb");
  if (!fp)
    {
      perror (path);
      return -1;
    }

  vmb_hdr_t hdr;
  memset (&hdr, 0, sizeof hdr);
  hdr.magic = VMB_MAGIC;
  hdr.version = VMB_VERSION;
  hdr.func_id = id;
  hdr.rec_size = f->kind == VMB_OUT_SINGLE ? sizeof (vmb_rec1_t)
		 : f->kind == VMB_OUT_DUAL ? sizeof (vmb_rec2_t)
		 : sizeof (vmb_rec2i_t);
  hdr.seed = seed ^ fnv1a (f->name);
  hdr.kind = f->kind;
  hdr.flags_policy = VMB_POLICY_FLAG_PER_REC;
  snprintf (hdr.name, sizeof hdr.name, "%s", f->name);

  /* Placeholder header; nrec patched at the end.  */
  long hdr_off = ftell (fp);
  fwrite (&hdr, sizeof hdr, 1, fp);

  uint64_t nrec = 0;
  vmb_rec1_t rec;
  vmb_rec2_t rec2;
  memset (&rec2, 0, sizeof rec2);

  /* Pass 1: special battery (with a matching second input for binary
     functions).  Pass 2: deterministic random draws.  */
  for (int pass = 0; pass < 2; pass++)
    {
      uint64_t count = (pass == 0) ? (uint64_t) nspecials : nrand;
      for (uint64_t i = 0; i < count; i++)
	{
	  double a = (pass == 0) ? specials[i]
		   : ((i % 4) == 3 ? cluster_draw (id) : draw_input ());
	  double b = (f->nin == 2)
		     ? ((pass == 0) ? specials[(i * 7 + 3) % nspecials]
				    : draw_input2 (a))
		     : bdouble (VMB_POISON);
	  if (compute_record (id, a, b, &rec,
			      f->kind == VMB_OUT_SINGLE ? NULL : &rec2) != 0)
	    {
	      /* Not implemented yet: leave no partial blob behind.  */
	      fclose (fp);
	      remove (path);
	      return 1;
	    }
	  if (f->kind == VMB_OUT_DUAL_INT)
	    {
	      vmb_rec2i_t ri;
	      memcpy (&ri.r2, &rec2, sizeof rec2);
	      ri.expect_sign = rec2i_sign;
	      ri.pad = 0;
	      fwrite (&ri, sizeof ri, 1, fp);
	    }
	  else if (f->kind == VMB_OUT_DUAL)
	    fwrite (&rec2, sizeof rec2, 1, fp);	/* fully built above */
	  else
	    fwrite (&rec, sizeof rec, 1, fp);
	  nrec++;
	}
    }

  hdr.nrec = nrec;
  fseek (fp, hdr_off, SEEK_SET);
  fwrite (&hdr, sizeof hdr, 1, fp);
  fclose (fp);
  printf ("%-10s %-6s %8llu records -> %s\n", f->name, tier,
	  (unsigned long long) nrec, path);
  return 0;
}

int
main (int argc, char **argv)
{
  const char *outdir = "blobs/FULL";
  const char *tier = "full";
  uint64_t seed = 0x5eed20260101ULL;
  const char *want[64];
  int nwant = 0;

  for (int i = 1; i < argc; i++)
    {
      if (!strcmp (argv[i], "--outdir") && i + 1 < argc)
	outdir = argv[++i];
      else if (!strcmp (argv[i], "--tier") && i + 1 < argc)
	tier = argv[++i];
      else if (!strcmp (argv[i], "--seed") && i + 1 < argc)
	seed = strtoull (argv[++i], NULL, 0);
      else if (!strcmp (argv[i], "--func") && i + 1 < argc)
	{
	  if (nwant < 64)
	    want[nwant++] = argv[++i];
	}
      else
	{
	  fprintf (stderr,
		   "usage: vmgen --outdir DIR [--tier small|full]"
		   " [--func NAME]... [--seed S]\n");
	  return 2;
	}
    }

  init_specials ();

  int ngen = 0, nskip = 0;
  for (int id = 1; id < NFUNC; id++)
    {
      if (fdesc[id].name == NULL)
	continue;
      if (nwant)
	{
	  int sel = 0;
	  for (int j = 0; j < nwant; j++)
	    if (!strcmp (want[j], fdesc[id].name))
	      sel = 1;
	  if (!sel)
	    continue;
	}
      int rc = gen_function (id, tier, seed, outdir);
      if (rc == 0)
	ngen++;
      else if (rc == 1)
	{
	  fprintf (stderr, "note: %s not implemented in generator yet\n",
		   fdesc[id].name);
	  nskip++;
	}
      else
	return 1;
    }
  printf ("vmgen: %d generated, %d skipped (%s tier)\n", ngen, nskip, tier);
  return 0;
}
