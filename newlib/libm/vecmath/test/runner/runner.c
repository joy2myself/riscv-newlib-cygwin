/* runner.c -- libvecmath accuracy runner (target program).

   Loads MPFR-generated blobs (see ../gen/vmgen.c), exercises every
   vector entry point through several passes and reports ULP accuracy
   plus special-value/flag conformance:

     pass 1  accuracy    -- one batched call over all records;
                            ulp metric vs the double-double reference,
                            bitwise checks for special classes.
     pass 2  length sweep -- n in {0,1,2,3, L_k-1, L_k, L_k+1, 3*L8+1}
                            (L_k = elements per register group at LMUL k);
                            results must be bitwise identical to pass 1
                            (stripmine/vl consistency), canaries intact.
     pass 3  alignment    -- input/output base offsets 0..7 doubles;
                            bitwise identity to pass 1.
     pass 4  per-record   -- n==1 calls with fenv gating: exception-flag
                            conformance + scalar-libm cross-check
                            (report-only).

   Exit status: 0 iff no FAIL (or --no-fail).

   Build: see ../Makefile (Linux/qemu or bare-metal semihosting).  */

#define _GNU_SOURCE	/* exp10, M_PI on glibc hosts */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fenv.h>

#include <vecmath.h>
#include "vmblob.h"
#include "ulp.h"

/* newlib/glibc RISC-V fenv values must match the blob flag bits.  */
_Static_assert (FE_INVALID == VMB_F_INVALID, "FE_INVALID mismatch");
_Static_assert (FE_DIVBYZERO == VMB_F_DIVBYZERO, "FE_DIVBYZERO mismatch");
_Static_assert (FE_OVERFLOW == VMB_F_OVERFLOW, "FE_OVERFLOW mismatch");
_Static_assert (FE_UNDERFLOW == VMB_F_UNDERFLOW, "FE_UNDERFLOW mismatch");
_Static_assert (FE_INEXACT == VMB_F_INEXACT, "FE_INEXACT mismatch");

/* ------------------------------------------------------------------ */
/* Function table                                                      */
/* ------------------------------------------------------------------ */

typedef struct {
  const char *name;
  int id, nin, kind;
  int prec;			/* 64 (binary64) or 32 (binary32) */
  double ulp;
} fdesc_t;

/* kind tokens in vmfuncs.def are paste-safe bare words; map to numbers. */
#define VMKN_C(a, b) a##b
#define VMKN(k) VMKN_C(VMKN_, k)
#define VMKN_SINGLE VMB_OUT_SINGLE
#define VMKN_DUAL VMB_OUT_DUAL
#define VMKN_DUAL_INT VMB_OUT_DUAL_INT

/* precision tokens likewise: F64/F32 -> 64/32. */
#define VMPR_C(a, b) a##b
#define VMPR(k) VMPR_C(VMPR_, k)
#define VMPR_F64 64
#define VMPR_F32 32

static const fdesc_t fdesc[] = {
#define VMFUNC(id, name, nin, kind, prec, ulp, ref) \
  [id] = { #name, id, nin, VMKN(kind), VMPR(prec), ulp },
#include "vmfuncs.def"
#undef VMFUNC
};
#define NFUNC ((int)(sizeof (fdesc) / sizeof (fdesc[0])))

enum {
#define VMFUNC(id, name, nin, kind, prec, ulp, ref) FID_##name = id,
#include "vmfuncs.def"
#undef VMFUNC
};

/* Weak references to the vector entries: functions not present in the
   library under test resolve to NULL and are reported as SKIP.  T is the
   element type: double for F64 rows, float for F32 rows.  */
#define VM_WU1(ID, NAME, T) \
  extern __attribute__ ((weak)) void v##NAME (T *, const T *, size_t);
#define VM_WB1(ID, NAME, T) \
  extern __attribute__ ((weak)) void v##NAME (T *, const T *, \
					      const T *, size_t);
#define VM_WD1(ID, NAME, T) \
  extern __attribute__ ((weak)) void v##NAME (T *, T *, \
					      const T *, size_t);
#define VM_WDI(ID, NAME, T) \
  extern __attribute__ ((weak)) void v##NAME (T *, int *, \
					      const T *, size_t);
#define VM_W_1_SINGLE_F64(ID, NAME)	 VM_WU1 (ID, NAME, double)
#define VM_W_1_SINGLE_F32(ID, NAME)	 VM_WU1 (ID, NAME, float)
#define VM_W_2_SINGLE_F64(ID, NAME)	 VM_WB1 (ID, NAME, double)
#define VM_W_2_SINGLE_F32(ID, NAME)	 VM_WB1 (ID, NAME, float)
#define VM_W_1_DUAL_F64(ID, NAME)	 VM_WD1 (ID, NAME, double)
#define VM_W_1_DUAL_F32(ID, NAME)	 VM_WD1 (ID, NAME, float)
#define VM_W_1_DUAL_INT_F64(ID, NAME)	 VM_WDI (ID, NAME, double)
#define VM_W_1_DUAL_INT_F32(ID, NAME)	 VM_WDI (ID, NAME, float)

/* nin/kind/prec sit textually adjacent to ## in VMFUNC's replacement list,
   so they are substituted RAW (kind/prec are bare words anyway); VMEXPAND
   then re-expands the pasted selector name so it runs.  */
#define VMEXPAND(f, a, b) f (a, b)

#define VMFUNC(id, name, nin, kind, prec, ulp, ref) \
  VMEXPAND (VM_W_##nin##_##kind##_##prec, id, name)
#include "vmfuncs.def"
#undef VMFUNC

/* Scalar helpers for greenfield functions (cross-check, report-only).  */
static double sc_rsqrt (double a) { return a == 0.0 ? HUGE_VAL : 1.0 / sqrt (a); }
static double sc_exp10 (double a) { return exp10 (a); }
static double sc_log2p1 (double a)
{
  double t = log1p (a);
  return fma (t, 1.44269504088896338700e+00, t * 2.03552737409310331110e-17);
}
static double sc_sinpi (double a) { return sin (M_PI * a); }
static double sc_cospi (double a) { return cos (M_PI * a); }
static double sc_tanpi (double a) { return tan (M_PI * a); }
static double sc_asinpi (double a) { return asin (a) / M_PI; }
static double sc_acospi (double a) { return acos (a) / M_PI; }
static double sc_atanpi (double a) { return atan (a) / M_PI; }
static double sc_atan2pi (double a, double b) { return atan2 (a, b) / M_PI; }

/* binary32 mirrors of the above (float arithmetic and constants).  */
#define VM_PIF 3.14159265358979323846f
static float sc_rsqrtf (float a) { return a == 0.0f ? HUGE_VALF : 1.0f / sqrtf (a); }
static float sc_exp10f (float a) { return exp10f (a); }
static float sc_log2p1f (float a)
{
  float t = log1pf (a);
  return fmaf (t, 1.44269504088896338700e+00f,
	       t * 2.03552737409310331110e-17f);
}
static float sc_sinpif (float a) { return sinf (VM_PIF * a); }
static float sc_cospif (float a) { return cosf (VM_PIF * a); }
static float sc_tanpif (float a) { return tanf (VM_PIF * a); }
static float sc_asinpif (float a) { return asinf (a) / VM_PIF; }
static float sc_acospif (float a) { return acosf (a) / VM_PIF; }
static float sc_atanpif (float a) { return atanf (a) / VM_PIF; }
static float sc_atan2pif (float a, float b) { return atan2f (a, b) / VM_PIF; }

static double
scalar_ref (int id, double a, double b)
{
  switch (id)
    {
#define VMFUNC(id_, name, nin, kind, prec, ulp, ref) case id_: return (ref);
#include "vmfuncs.def"
#undef VMFUNC
    default:
      return NAN;
    }
}

/* Functions whose def entry is VMREF_NONE have no scalar analogue.  */
static int
has_cross (int id)
{
  switch (id)
    {
    case FID_sincos:
    case FID_sincospi:
    case FID_lgamma_r:
    case FID_sincosf:
    case FID_sincospif:
    case FID_lgamma_rf:
      return 0;
    default:
      return 1;
    }
}

/* qsort comparator for doubles.  */
static int
dcmp (const void *p, const void *q)
{
  double a = *(const double *) p, b = *(const double *) q;
  return (a > b) - (a < b);
}

/* Dispatch table over the same X-macro.  Buffers are passed as void *
   and cast to the element type selected by the precision token; the sign
   array stays int.  */
static int
vec_call (int id, void *y, void *y2, int *sgn,
	  const void *x, const void *x2, size_t n)
{
  switch (id)
    {
#define VM_CU1(ID, NAME, T) \
  case ID: if (!v##NAME) return -1; v##NAME ((T *) y, (const T *) x, n); break;
#define VM_CB1(ID, NAME, T) \
  case ID: if (!v##NAME) return -1; \
    v##NAME ((T *) y, (const T *) x, (const T *) x2, n); break;
#define VM_CD1(ID, NAME, T) \
  case ID: if (!v##NAME) return -1; \
    v##NAME ((T *) y, (T *) y2, (const T *) x, n); break;
#define VM_CDI(ID, NAME, T) \
  case ID: if (!v##NAME) return -1; \
    v##NAME ((T *) y, sgn, (const T *) x, n); break;
#define VM_GEN_1_SINGLE_F64(ID, NAME)	 VM_CU1 (ID, NAME, double)
#define VM_GEN_1_SINGLE_F32(ID, NAME)	 VM_CU1 (ID, NAME, float)
#define VM_GEN_2_SINGLE_F64(ID, NAME)	 VM_CB1 (ID, NAME, double)
#define VM_GEN_2_SINGLE_F32(ID, NAME)	 VM_CB1 (ID, NAME, float)
#define VM_GEN_1_DUAL_F64(ID, NAME)	 VM_CD1 (ID, NAME, double)
#define VM_GEN_1_DUAL_F32(ID, NAME)	 VM_CD1 (ID, NAME, float)
#define VM_GEN_1_DUAL_INT_F64(ID, NAME)	 VM_CDI (ID, NAME, double)
#define VM_GEN_1_DUAL_INT_F32(ID, NAME)	 VM_CDI (ID, NAME, float)
#define VMFUNC(id, name, nin, kind, prec, ulp, ref) \
  VMEXPAND (VM_GEN_##nin##_##kind##_##prec, id, name)
#include "vmfuncs.def"
#undef VMFUNC
    default:
      return -2;
    }
  return 0;
}

/* ------------------------------------------------------------------ */
/* VLEN probe                                                          */
/* ------------------------------------------------------------------ */

#if defined(__riscv_vector)
#include <riscv_vector.h>
static size_t
vlen_doubles_m1 (void)
{
  return __riscv_vsetvlmax_e64m1 ();
}

static size_t
vlen_floats_m1 (void)
{
  return __riscv_vsetvlmax_e32m1 ();
}
#else
static size_t
vlen_doubles_m1 (void)
{
  return 2;			/* scalar-fallback build: synthetic small L */
}

static size_t
vlen_floats_m1 (void)
{
  return 4;			/* scalar-fallback build: synthetic small L */
}
#endif

/* ------------------------------------------------------------------ */
/* Blob loading                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
  vmb_hdr_t hdr;
  void *recs;			/* nrec * rec_size bytes */
  void *a, *b;			/* unpacked inputs: double[] or float[] */
  int esz;			/* element size: 8 (F64) or 4 (F32) */
} blob_t;

static int
blob_load (const char *dir, const fdesc_t *f, blob_t *bl)
{
  char path[1024];
  snprintf (path, sizeof path, "%s/%s.bin", dir, f->name);
  FILE *fp = fopen (path, "rb");
  if (!fp)
    return 0;
  if (fread (&bl->hdr, sizeof bl->hdr, 1, fp) != 1
      || bl->hdr.magic != VMB_MAGIC || bl->hdr.version != VMB_VERSION
      || bl->hdr.func_id != (uint32_t) f->id)
    {
      fprintf (stderr, "runner: bad blob header in %s\n", path);
      fclose (fp);
      return -1;
    }
  size_t bytes = (size_t) bl->hdr.nrec * bl->hdr.rec_size;
  bl->recs = malloc (bytes ? bytes : 1);
  if (!bl->recs || (bytes && fread (bl->recs, 1, bytes, fp) != bytes))
    {
      fprintf (stderr, "runner: short read %s\n", path);
      fclose (fp);
      free (bl->recs);
      return -1;
    }
  fclose (fp);

  /* reserved[0] selects the record precision; both paths unpack the low
     bits of in0/in1 into a typed input array (binary32 values live in the
     low 32 bits of each field, see vmblob.h).  */
  bl->esz = bl->hdr.reserved[0] == VMB_PREC_F32 ? 4 : 8;
  size_t asz = (size_t) bl->hdr.nrec * bl->esz + 64;
  bl->a = malloc (asz);
  bl->b = malloc (asz);
  for (uint64_t i = 0; i < bl->hdr.nrec; i++)
    {
      const vmb_rec1_t *r = (const vmb_rec1_t *)
	((const char *) bl->recs + i * bl->hdr.rec_size);
      memcpy ((char *) bl->a + i * bl->esz, &r->in0, bl->esz);
      memcpy ((char *) bl->b + i * bl->esz, &r->in1, bl->esz);
    }
  return 1;
}

static void
blob_free (blob_t *bl)
{
  free (bl->recs);
  free (bl->a);
  free (bl->b);
}

/* ------------------------------------------------------------------ */
/* Worst-N tracker                                                     */
/* ------------------------------------------------------------------ */

#define WORSTN 10
typedef struct {
  double ulp[WORSTN];
  uint64_t idx[WORSTN];
} worst_t;

static void
worst_add (worst_t *w, uint64_t i, double ulp)
{
  for (int k = 0; k < WORSTN; k++)
    {
      if (w->idx[k] == UINT64_MAX || ulp > w->ulp[k])
	{
	  memmove (&w->ulp[k + 1], &w->ulp[k],
		   (WORSTN - k - 1) * sizeof (double));
	  memmove (&w->idx[k + 1], &w->idx[k],
		   (WORSTN - k - 1) * sizeof (uint64_t));
	  w->ulp[k] = ulp;
	  w->idx[k] = i;
	  return;
	}
    }
}

static void
worst_init (worst_t *w)
{
  for (int k = 0; k < WORSTN; k++)
    {
      w->ulp[k] = -1.0;
      w->idx[k] = UINT64_MAX;
    }
}

/* ------------------------------------------------------------------ */
/* Per-function test                                                   */
/* ------------------------------------------------------------------ */

#define CANARY64 0xa55a5aa5a55a5aa5ULL
#define CANARY32 0xa55a5aa5u
#define GUARD 8			/* guard elements around output buffers */

/* Element-typed access.  All buffer indexing goes through these so that
   binary64 (8-byte) and binary32 (4-byte) buffers share one code path;
   memcpy keeps them free of alignment and aliasing assumptions.  */

static void
canary_put (void *base, uint64_t i, int esz)
{
  if (esz == 4)
    {
      uint32_t c = CANARY32;
      memcpy ((char *) base + i * 4, &c, 4);
    }
  else
    {
      uint64_t c = CANARY64;
      memcpy ((char *) base + i * 8, &c, 8);
    }
}

static int
canary_ok (const void *base, uint64_t i, int esz)
{
  if (esz == 4)
    {
      uint32_t c;
      memcpy (&c, (const char *) base + i * 4, 4);
      return c == CANARY32;
    }
  uint64_t c;
  memcpy (&c, (const char *) base + i * 8, 8);
  return c == CANARY64;
}

static uint64_t
elem_bits (const void *base, uint64_t i, int esz)
{
  if (esz == 4)
    {
      uint32_t w;
      memcpy (&w, (const char *) base + i * 4, 4);
      return (uint64_t) w;
    }
  uint64_t w;
  memcpy (&w, (const char *) base + i * 8, 8);
  return w;
}

static double
elem_double (const void *base, uint64_t i, int esz)
{
  if (esz == 4)
    {
      float f;
      memcpy (&f, (const char *) base + i * 4, 4);
      return (double) f;
    }
  double d;
  memcpy (&d, (const char *) base + i * 8, 8);
  return d;
}

/* Accuracy verdict for output 1: the single-output record r, or the first
   output of the dual record r2 (whose expectation lives in exp1).  */
static ulp_verdict_t
eval_out1 (int esz, double yv, const vmb_rec1_t *r, const vmb_rec2_t *r2,
	   uint32_t exp1)
{
  if (esz == 4)
    {
      if (r2)
	{
	  vmb_rec1_t t;
	  t.expect = exp1;
	  t.ref_hi = r2->ref_hi;
	  t.ref_lo = r2->ref_lo;
	  t.exact_bits = r2->exact_bits;
	  return ulp_eval_f32 ((float) yv, &t);
	}
      return ulp_eval_f32 ((float) yv, r);
    }
  return r2 ? ulp_eval_core (yv, exp1, r2->ref_hi, r2->ref_lo, r2->exact_bits)
	    : ulp_eval (yv, r);
}

/* Accuracy verdict for output 2 of a dual record.  */
static ulp_verdict_t
eval_out2 (int esz, double yv2, const vmb_rec2_t *r2, uint32_t exp2)
{
  return esz == 4 ? ulp_eval2_f32 ((float) yv2, r2, exp2)
		  : ulp_eval2 (yv2, r2, exp2);
}

static int
test_function (const fdesc_t *f, blob_t *bl, long cross_max, int no_fail,
	       FILE *rep)
{
  const uint64_t nrec = bl->hdr.nrec;
  const size_t rsz = bl->hdr.rec_size;
  const int esz = bl->esz;		/* 8 (binary64) or 4 (binary32) */
  const int f32 = esz == 4;
  void *ybase = malloc ((nrec + 2 * GUARD) * esz);
  void *y2base = malloc ((nrec + 2 * GUARD) * esz);
  int *sgn = malloc ((nrec + 1) * sizeof (int));
  char *y = (char *) ybase + (size_t) GUARD * esz;
  char *y2 = (char *) y2base + (size_t) GUARD * esz;
  int fails = 0;

  for (int g = 0; g < GUARD; g++)
    {
      canary_put (ybase, g, esz);
      canary_put (ybase, GUARD + nrec + g, esz);
    }

  /* Probe presence.  */
  if (vec_call (f->id, y, y2, sgn, bl->a, bl->b, 0) == -1)
    {
      printf ("%-10s SKIP (not in library)\n", f->name);
      free (ybase); free (y2base); free (sgn);
      return 0;
    }

  /* ---- pass 1: accuracy over the full batch ---- */
  feclearexcept (FE_ALL_EXCEPT);
  vec_call (f->id, y, y2, sgn, bl->a, bl->b, (size_t) nrec);
  int agg_flags = fetestexcept (FE_ALL_EXCEPT) & VMB_F_GATE;

  worst_t worst;
  worst_init (&worst);
  double sum = 0.0, maxulp = 0.0;
  uint64_t n_over = 0, n_class = 0, nulp = 0;
  int agg_expected = 0;
  double *ulps = malloc (nrec * sizeof (double));
  int shown = 0;

  for (uint64_t i = 0; i < nrec; i++)
    {
      const vmb_rec1_t *r = (const vmb_rec1_t *) ((char *) bl->recs + i * rsz);
      /* Dual-output records (sincos/sincospi, lgamma_r) are the vmb_rec2_t
	 image: expect/fpflags live at the rec2 offsets, and the first
	 output's class sits in bits 0..4 of the packed word.  */
      const vmb_rec2_t *r2 = bl->hdr.kind == VMB_OUT_SINGLE
			     ? NULL : (const vmb_rec2_t *) r;
      uint32_t exp1 = r2 ? (r2->expect & 0x1f) : r->expect;
      uint32_t exp2 = r2 ? VMB_EXPECT2 (r2->expect) : 0;
      uint32_t fpf = r2 ? r2->fpflags : r->fpflags;
      const uint32_t CLASSY = VMB_EXPECT_NAN | VMB_EXPECT_INF
			      | VMB_EXPECT_EXACT;
      double yv = elem_double (y, i, esz);
      ulp_verdict_t v = eval_out1 (esz, yv, r, r2, exp1);
      ulp_verdict_t v2 = { 0.0, 1, 1 };
      /* lgamma_r's second output is an int, checked against expect_sign
	 below; there is no second double to evaluate.  */
      if (r2 && bl->hdr.kind == VMB_OUT_DUAL)
	{
	  v2 = eval_out2 (esz, elem_double (y2, i, esz), r2, exp2);
	  v.class_ok = v.class_ok && v2.class_ok;
	}
      agg_expected |= fpf & VMB_F_GATE;
      int bad = !v.class_ok;
      double u = 0.0;
      int have_ulp = 0;
      if (!(exp1 & CLASSY))
	{
	  u = v.ulp;
	  have_ulp = 1;
	}
      if (r2 && bl->hdr.kind == VMB_OUT_DUAL && !(exp2 & CLASSY))
	{
	  if (!have_ulp || v2.ulp > u)
	    u = v2.ulp;
	  have_ulp = 1;
	}
      if (v.class_ok && have_ulp)
	{
	  nulp++;
	  ulps[nulp - 1] = u;
	  sum += u;
	  if (u > maxulp)
	    maxulp = u;
	  worst_add (&worst, i, u);
	  if (u > f->ulp)
	    {
	      n_over++;
	      bad = 1;
	    }
	}
      if (bad)
	{
	  fails++;
	  n_class += !v.class_ok;
	  if (shown < 10)
	    {
	      uint64_t gotb = elem_bits (y, i, esz);
	      double av = elem_double (bl->a, i, esz);
	      char bbuf[48];
	      shown++;
	      /* Binary functions print both arguments: without the second
		 one a FAIL line cannot be reproduced by hand.  */
	      if (f->nin == 2)
		snprintf (bbuf, sizeof bbuf, " b=%.17g (0x%016lx)",
			  elem_double (bl->b, i, esz),
			  (unsigned long) r->in1);
	      else
		bbuf[0] = '\0';
	      printf ("  FAIL %s a=%.17g (0x%016lx)%s -> got %.17g"
		      " (0x%016lx) expect=0x%x exact=0x%016lx ulp=%.4f\n",
		      f->name, av, (unsigned long) r->in0, bbuf, yv,
		      (unsigned long) gotb,
		      exp1 | (exp2 << VMB_EXPECT2_SHIFT),
		      (unsigned long) (r2 ? r2->exact_bits : r->exact_bits), u);
	    }
	}
    }

  /* p99.9 */
  double p999 = 0.0;
  if (nulp)
    {
      qsort (ulps, nulp, sizeof (double), dcmp);
      p999 = ulps[(size_t) ((nulp - 1) * 0.999)];
    }
  free (ulps);

  /* aggregate flag sanity (report-only; per-record pass is the gate) */
  int agg_extra = agg_flags & ~agg_expected;

  /* ---- pass 2: length sweep (stripmine consistency + canaries) ---- */
  size_t L1 = f32 ? vlen_floats_m1 () : vlen_doubles_m1 ();
  size_t lens[32];
  int nlens = 0;
  lens[nlens++] = 0;
  lens[nlens++] = 1; lens[nlens++] = 2; lens[nlens++] = 3;
  for (int k = 0; k < 4; k++)
    {
      size_t L = L1 << k;	/* m1, m2, m4, m8 */
      if (L) lens[nlens++] = L - 1;
      lens[nlens++] = L;
      lens[nlens++] = L + 1;
    }
  lens[nlens++] = 3 * (L1 << 3) + 1;
  int sweep_fail = 0;
  for (int li = 0; li < nlens && li < 32; li++)
    {
      size_t n = lens[li];
      if (n > nrec)
	n = (size_t) nrec;
      void *ys = malloc ((n + 2 * GUARD) * esz + 16);
      char *yp = (char *) ys + (size_t) GUARD * esz;
      for (int g = 0; g < GUARD; g++)
	{
	  canary_put (ys, g, esz);
	  canary_put (ys, GUARD + n + g, esz);
	}
      vec_call (f->id, yp, y2, sgn, bl->a, bl->b, n);
      for (size_t i = 0; i < n; i++)
	if (memcmp (yp + i * esz, y + i * esz, esz) != 0)
	  sweep_fail++;
      for (int g = 0; g < GUARD; g++)
	if (!canary_ok (ys, g, esz) || !canary_ok (ys, GUARD + n + g, esz))
	  sweep_fail += 1000;
      free (ys);
    }
  fails += sweep_fail;

  /* ---- pass 3: alignment sweep ---- */
  int align_fail = 0;
  {
    size_t n = nrec < 128 ? (size_t) nrec : 128;
    void *y2s = malloc ((n + GUARD) * esz);	/* dual 2nd out */
    for (int off = 0; off < 8; off++)
      {
	size_t m = n - (size_t) off;
	char *yo = y2 + (size_t) off * esz;
	const char *xo = (const char *) bl->a + (size_t) off * esz;
	const char *x2o = (const char *) bl->b + (size_t) off * esz;
	vec_call (f->id, yo, y2s, sgn, xo, x2o, m);
	for (size_t i = 0; i < m; i++)
	  if (memcmp (yo + i * esz, y + ((size_t) off + i) * esz, esz) != 0)
	    align_fail++;
      }
    free (y2s);
  }
  fails += align_fail;

  /* ---- pass 4: per-record flags + scalar cross-check ---- */
  int flag_fail = 0, flag_shown = 0;
  uint64_t cross_n = 0, cross_diff = 0;
  double cross_max_ulp = 0.0;
  uint64_t ncross = nrec < (uint64_t) cross_max ? nrec : (uint64_t) cross_max;
  int per_rec = (bl->hdr.flags_policy & VMB_POLICY_FLAG_PER_REC) != 0;
  for (uint64_t i = 0; i < ncross; i++)
    {
      const vmb_rec1_t *r = (const vmb_rec1_t *) ((char *) bl->recs + i * rsz);
      const vmb_rec2_t *r2 = bl->hdr.kind == VMB_OUT_SINGLE
			     ? NULL : (const vmb_rec2_t *) r;
      const char *xio = (const char *) bl->a + i * esz;
      const char *x2io = (const char *) bl->b + i * esz;
      double av = elem_double (bl->a, i, esz);
      double bv = elem_double (bl->b, i, esz);
      if (per_rec)
	{
	  double yo_d;
	  float yo_f;
	  void *yo = f32 ? (void *) &yo_f : (void *) &yo_d;
	  int sgn1 = 0;
	  feclearexcept (FE_ALL_EXCEPT);
	  vec_call (f->id, yo, y2, &sgn1, xio, x2io, 1);
	  int got = fetestexcept (FE_ALL_EXCEPT) & VMB_F_GATE;
	  int want = (r2 ? r2->fpflags : r->fpflags) & VMB_F_GATE;
	  if (bl->hdr.kind == VMB_OUT_DUAL_INT)
	    {
	      const vmb_rec2i_t *ri = (const vmb_rec2i_t *) r;
	      if (sgn1 != ri->expect_sign)
		{
		  flag_fail++;
		  fails++;
		  if (flag_shown < 10)
		    {
		      flag_shown++;
		      printf ("  SIGNFAIL %s a=%.17g got=%d want=%d\n",
			      f->name, av, sgn1, ri->expect_sign);
		    }
		}
	    }
	  if (got != want)
	    {
	      flag_fail++;
	      fails++;
	      if (flag_shown < 10)
		{
		  flag_shown++;
		  printf ("  FLAGFAIL %s a=%.17g (0x%016lx) b=%.17g got=0x%x"
			  " want=0x%x\n",
			  f->name, av,
			  (unsigned long) r->in0, bv, got, want);
		}
	    }
	}
      if (has_cross (f->id))
	{
	  double sr = scalar_ref (f->id, av, bv);
	  cross_n++;
	  if (f32)
	    {
	      float srf = (float) sr;
	      float yvf = (float) elem_double (y, i, esz);
	      if (memcmp (&srf, &yvf, 4) != 0)
		{
		  cross_diff++;
		  double d = ulp_distance_f32 (yvf, srf);
		  if (d > cross_max_ulp)
		    cross_max_ulp = d;
		}
	    }
	  else
	    {
	      double yvd = elem_double (y, i, esz);
	      if (memcmp (&sr, &yvd, 8) != 0)
		{
		  cross_diff++;
		  double d = ulp_distance (yvd, sr);
		  if (d > cross_max_ulp)
		    cross_max_ulp = d;
		}
	    }
	}
    }

  /* ---- verdict ---- */
  const char *status = (fails == 0) ? "PASS" : (no_fail ? "FAIL(nofail)" : "FAIL");
  printf ("%-10s n=%-7lu max_ulp=%-8.4f mean=%-8.4f p99.9=%-8.4f"
	  " >%g ulp: %-5lu class_fail: %-4lu sweep_fail: %-4d flag_fail: %-4d"
	  " cross_diff: %lu/%lu (max %.3f ulp) agg_extra_flags=0x%x  %s\n",
	  f->name, (unsigned long) nrec, maxulp,
	  nulp ? sum / (double) nulp : 0.0, p999, f->ulp,
	  (unsigned long) n_over, (unsigned long) n_class,
	  sweep_fail, flag_fail,
	  (unsigned long) cross_diff, (unsigned long) cross_n,
	  cross_max_ulp, agg_extra, status);

  if (worst.idx[0] != UINT64_MAX && maxulp > 0.5)
    {
      printf ("  worst:");
      for (int k = 0; k < WORSTN && worst.idx[k] != UINT64_MAX; k++)
	printf (" [%.17g:%.3f]", elem_double (bl->a, worst.idx[k], esz),
		worst.ulp[k]);
      printf ("\n");
    }

  if (rep)
    fprintf (rep, "| %s | %lu | %.4f | %.4f | %.4f | %lu | %d | %d | %s |\n",
	     f->name, (unsigned long) nrec, maxulp,
	     nulp ? sum / (double) nulp : 0.0, p999,
	     (unsigned long) (n_over + n_class),
	     sweep_fail, flag_fail, fails == 0 ? "PASS" : "FAIL");

  free (ybase); free (y2base); free (sgn);
  return (fails == 0 || no_fail) ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* dd self-test (--dd-selftest): validates the double-double layer of
   the public inline header vecmath_rvv.h (all-LMUL instantiation path)
   via exact fma identities on 10k deterministic random pairs:
     two_prod: p == RN(a*b) and e == fma(a,b,-p)   (exact residual)
     two_sum : s == RN(a+b) and e == Knuth complement
     dd_mul  : hi/lo of a*b matches two_prod for lo-free inputs
     dd_sqrt : (sqrt(x))^2 - x == 0 (hi) with |lo| <= 2^-95 x
     dd_div  : (a/b)*b - a within 2^-100 relative                      */

#if defined(__riscv_vector)
/* Compile-check of the public all-LMUL inline header (self-cleans its
   macros), then a private m2 instantiation of the compat + dd layers
   for the identity checks below.  */
#include <vecmath_rvv.h>
#define VM_LMUL 2
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>

#define DDST_N 10000
static uint64_t ddst_s = 0x12345678ULL;
static uint64_t
ddst_rand (void)
{
  ddst_s ^= ddst_s << 13;
  ddst_s ^= ddst_s >> 7;
  ddst_s ^= ddst_s << 17;
  return ddst_s;
}
static double
ddst_dbl (void)
{
  int e = (int) (ddst_rand () % 200) - 100;
  double m = 1.0 + (double) (ddst_rand () >> 11) * 0x1p-53;
  return ldexp (ddst_rand () & 1 ? m : -m, e);
}

static int
dd_selftest (void)
{
  static double a[DDST_N], b[DDST_N];
  static double p[DDST_N], e[DDST_N], dhi[DDST_N], dlo[DDST_N], s[DDST_N];
  int fails = 0;

  ddst_s = 0x12345678ULL;
  for (int i = 0; i < DDST_N; i++)
    {
      a[i] = ddst_dbl ();
      b[i] = ddst_dbl ();
    }

  /* two_prod + dd_mul on lo-free inputs (must coincide).  */
  for (size_t i = 0; i < DDST_N;)
    {
      size_t vl = __riscv_vsetvl_e64m2 (DDST_N - i);
      vfloat64m2_t va = __riscv_vle64_v_f64m2 (&a[i], vl);
      vfloat64m2_t vb = __riscv_vle64_v_f64m2 (&b[i], vl);
      vfloat64m2_t z = __riscv_vfmv_v_f_f64m2 (0.0, vl);
      vfloat64m2_t vp, ve, mh, ml;
      VM_TWO_PROD (vp, ve, va, vb, vl);
      VM_DD_MUL (mh, ml, va, z, vb, z, vl);
      __riscv_vse64_v_f64m2 (&p[i], vp, vl);
      __riscv_vse64_v_f64m2 (&e[i], ve, vl);
      __riscv_vse64_v_f64m2 (&dhi[i], mh, vl);
      __riscv_vse64_v_f64m2 (&dlo[i], ml, vl);
      i += vl;
    }
  int f1 = 0, f2 = 0;
  for (int i = 0; i < DDST_N; i++)
    {
      double want_p = a[i] * b[i];
      double want_e = fma (a[i], b[i], -want_p);
      if (p[i] != want_p || e[i] != want_e)
	f1++;
      if (dhi[i] != p[i] || dlo[i] != e[i])
	f2++;
    }
  printf ("dd_selftest two_prod: %s (%d / %d)\n", f1 ? "FAIL" : "ok", f1, DDST_N);
  printf ("dd_selftest dd_mul:   %s (%d / %d)\n", f2 ? "FAIL" : "ok", f2, DDST_N);
  fails += f1 + f2;

  /* dd_sqrt and dd_div round trips.  */
  for (int i = 0; i < DDST_N; i++)
    b[i] = fabs (b[i]);
  for (size_t i = 0; i < DDST_N;)
    {
      size_t vl = __riscv_vsetvl_e64m2 (DDST_N - i);
      vfloat64m2_t va = __riscv_vle64_v_f64m2 (&a[i], vl);
      vfloat64m2_t vb = __riscv_vle64_v_f64m2 (&b[i], vl);
      vfloat64m2_t z = __riscv_vfmv_v_f_f64m2 (0.0, vl);
      vfloat64m2_t sh, sl, qh, ql, rh, rl;
      VM_DD_SQRT (sh, sl, vb, z, vl);
      VM_DD_MUL (qh, ql, sh, sl, sh, sl, vl);
      VM_DD_SUB (rh, rl, qh, ql, vb, z, vl);
      __riscv_vse64_v_f64m2 (&dhi[i], rh, vl);
      __riscv_vse64_v_f64m2 (&dlo[i], rl, vl);
      VM_DD_DIV (qh, ql, va, z, vb, z, vl);
      VM_DD_MUL (rh, rl, qh, ql, vb, z, vl);
      VM_DD_SUB (rh, rl, rh, rl, va, z, vl);
      __riscv_vse64_v_f64m2 (&s[i], rh, vl);
      __riscv_vse64_v_f64m2 (&p[i], rl, vl);
      i += vl;
    }
  int f3 = 0, f4 = 0;
  for (int i = 0; i < DDST_N; i++)
    {
      if (fabs (dhi[i]) > ldexp (b[i], -98)
	  || fabs (dlo[i]) > ldexp (b[i], -140))
	f3++;
      if (fabs (s[i]) > ldexp (fabs (a[i]), -100)
	  || fabs (p[i]) > ldexp (fabs (a[i]), -99))
	f4++;
    }
  printf ("dd_selftest dd_sqrt:  %s (%d / %d)\n", f3 ? "FAIL" : "ok", f3, DDST_N);
  printf ("dd_selftest dd_div:   %s (%d / %d)\n", f4 ? "FAIL" : "ok", f4, DDST_N);
  fails += f3 + f4;
  return fails == 0 ? 0 : 1;
}
#endif /* __riscv_vector */

int
main (int argc, char **argv)
{
  const char *dir = "blobs/SMALL";
  const char *report = NULL;
  const char *only = NULL;
  long cross_max = 20000;
  int no_fail = 0;

  for (int i = 1; i < argc; i++)
    {
      if (!strcmp (argv[i], "--blobs") && i + 1 < argc)
	dir = argv[++i];
      else if (!strcmp (argv[i], "--report") && i + 1 < argc)
	report = argv[++i];
      else if (!strcmp (argv[i], "--func") && i + 1 < argc)
	only = argv[++i];
      else if (!strcmp (argv[i], "--cross") && i + 1 < argc)
	cross_max = strtol (argv[++i], NULL, 0);
      else if (!strcmp (argv[i], "--no-fail"))
	no_fail = 1;
#if defined(__riscv_vector)
      else if (!strcmp (argv[i], "--dd-selftest"))
	return dd_selftest ();
#endif
      else
	{
	  fprintf (stderr,
		   "usage: runner --blobs DIR [--func NAME] [--report F.md]"
		   " [--cross N] [--no-fail]\n");
	  return 2;
	}
    }

  /* Line-buffered progress output (bare-metal semihost consoles are not
     ttys; without this nothing appears until exit/flush).  */
  setvbuf (stdout, NULL, _IOLBF, 0);

  size_t L1 = vlen_doubles_m1 ();
  printf ("libvecmath runner: blobs=%s VLEN=%lu bits (%lu doubles/LMUL1)\n",
	  dir, (unsigned long) (L1 * 64), (unsigned long) L1);

  FILE *rep = NULL;
  if (report)
    {
      rep = fopen (report, "w");
      if (rep)
	fprintf (rep,
		 "| function | n | max_ulp | mean_ulp | p99.9 | fails |"
		 " sweep | flags | status |\n"
		 "|---|---|---|---|---|---|---|---|---|\n");
    }

  int total_fail = 0, tested = 0;
  for (int id = 1; id < NFUNC; id++)
    {
      if (fdesc[id].name == NULL)
	continue;
      if (only && strcmp (only, fdesc[id].name))
	continue;
      blob_t bl;
      int rc = blob_load (dir, &fdesc[id], &bl);
      if (rc <= 0)
	{
	  if (rc < 0)
	    total_fail++;
	  continue;
	}
      tested++;
      total_fail += test_function (&fdesc[id], &bl, cross_max, no_fail, rep);
      blob_free (&bl);
    }

  if (rep)
    fclose (rep);
  printf ("runner: %d functions tested, %d failed\n", tested, total_fail);
  return total_fail == 0 ? 0 : 1;
}
