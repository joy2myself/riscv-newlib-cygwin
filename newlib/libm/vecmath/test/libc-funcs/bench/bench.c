/* Throughput harness: RVV (newlib) vs glibc for the hot string/memory
   routines.

   Build: see ../build.sh (the same objcopy --redefine-sym trick as the
   correctness harness renames the newlib entry points to rvv_*).
   Run:   ./bench [-c] [--csv FILE]

   "GB/s" is  payload bytes per call / elapsed time, i.e. one direction only
   (a memcpy of n bytes counts n, not 2n).  Each measurement is the best of
   several timed rounds after a warm-up round; calls go through volatile
   function pointers so nothing is optimised away.

   Portable C + POSIX clock_gettime only: it also runs natively on the
   RISC-V boards.  */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- implementations under test (renamed by objcopy) ---- */
extern void *rvv_memcpy (void *, const void *, size_t);
extern void *rvv_memset (void *, int, size_t);
extern void *rvv_memmove (void *, const void *, size_t);
extern int rvv_memcmp (const void *, const void *, size_t);
extern size_t rvv_strlen (const char *);
extern int rvv_strcmp (const char *, const char *);
extern char *rvv_strchr (const char *, int);
extern char *rvv_strstr (const char *, const char *);

static volatile uintptr_t sink;
static unsigned char *src, *dst, *dst2, *str1, *str2, *ndl;
static size_t N;
static int chr_c;

/* ---- the operations, one tiny wrapper per (function, implementation) --- */
static void op_memcpy_r (void) { sink += (uintptr_t) rvv_memcpy (dst, src, N); }
static void op_memcpy_l (void) { sink += (uintptr_t) memcpy (dst, src, N); }
static void op_memset_r (void) { sink += (uintptr_t) rvv_memset (dst, 0xa5, N); }
static void op_memset_l (void) { sink += (uintptr_t) memset (dst, 0xa5, N); }
static void op_memmove_r (void) { sink += (uintptr_t) rvv_memmove (dst2, dst, N); }
static void op_memmove_l (void) { sink += (uintptr_t) memmove (dst2, dst, N); }
static void op_memcmp_r (void) { sink += (uintptr_t) rvv_memcmp (src, dst, N); }
static void op_memcmp_l (void) { sink += (uintptr_t) memcmp (src, dst, N); }
static void op_strlen_r (void) { sink += (uintptr_t) rvv_strlen ((char *) str1); }
static void op_strlen_l (void) { sink += (uintptr_t) strlen ((char *) str1); }
static void op_strcmp_r (void) { sink += (uintptr_t) rvv_strcmp ((char *) str1, (char *) str2); }
static void op_strcmp_l (void) { sink += (uintptr_t) strcmp ((char *) str1, (char *) str2); }
static void op_strchr_r (void) { sink += (uintptr_t) rvv_strchr ((char *) str1, chr_c); }
static void op_strchr_l (void) { sink += (uintptr_t) strchr ((char *) str1, chr_c); }
static void op_strstr_r (void) { sink += (uintptr_t) rvv_strstr ((char *) str1, (char *) ndl); }
static void op_strstr_l (void) { sink += (uintptr_t) strstr ((char *) str1, (char *) ndl); }

enum
{
  K_MEMCPY, K_MEMSET, K_MEMMOVE, K_MEMCMP,
  K_STRLEN, K_STRCMP, K_STRCHR, K_STRSTR, K_STRSTR_CAND, K_COUNT
};

/* volatile: the compiler must not devirtualise the table and hoist a
   "pure" call (strlen/strchr/strstr) out of the measurement loop.  */
static void (*volatile const ops[2][K_COUNT]) (void) = {
  { op_memcpy_r, op_memset_r, op_memmove_r, op_memcmp_r,
    op_strlen_r, op_strcmp_r, op_strchr_r, op_strstr_r, op_strstr_r },
  { op_memcpy_l, op_memset_l, op_memmove_l, op_memcmp_l,
    op_strlen_l, op_strcmp_l, op_strchr_l, op_strstr_l, op_strstr_l }
};

static const char *const names[K_COUNT] = {
  "memcpy", "memset", "memmove", "memcmp",
  "strlen", "strcmp", "strchr", "strstr", "strstr-cand"
};

static const size_t sizes[] =
  { 8, 16, 32, 64, 128, 256, 1024, 4096, 65536 };
#define NSIZES ((int) (sizeof sizes / sizeof *sizes))

#define TARGET_BYTES (8u << 20)
#define ROUNDS 3
#define MAX_REPS (1L << 26)

static double
now_sec (void)
{
  struct timespec ts;
  clock_gettime (CLOCK_MONOTONIC, &ts);
  return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static long
reps_for (size_t n)
{
  long r = (long) (TARGET_BYTES / (n ? n : 1));
  if (r < 64)
    r = 64;
  if (r > MAX_REPS)
    r = MAX_REPS;
  return r;
}

static double
measure (int impl, int kind, long reps)
{
  double best = 1e30;
  int round;
  long i;

  for (round = -1; round < ROUNDS; round++)
    {
      double t0, t1;
      t0 = now_sec ();
      for (i = 0; i < reps; i++)
	{
	  ops[impl][kind] ();
	  __asm__ volatile ("" ::: "memory");
	}
      t1 = now_sec ();
      if (round < 0)
	continue;		/* warm-up */
      if (t1 - t0 < best)
	best = t1 - t0;
    }
  return best;
}

static unsigned long
vlen_bits (void)
{
#ifdef __riscv_vector
  unsigned long x;
  __asm__ volatile ("csrr %0, vlenb" : "=r" (x));
  return x * 8;
#else
  return 0;
#endif
}

/* String data: bytes 1..255, never NUL, so the scan length really is n.  */
static void
fill_str (unsigned char *p, size_t n, unsigned seed)
{
  size_t i;
  uint32_t x = seed * 2654435761u + 1;
  for (i = 0; i < n; i++)
    {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      p[i] = (unsigned char) (1 + ((x >> 16) % 255));
    }
}

static void
fill (unsigned char *p, size_t n, unsigned seed)
{
  size_t i;
  uint32_t x = seed * 2654435761u + 1;
  for (i = 0; i < n; i++)
    {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      p[i] = (unsigned char) (x >> 16);
    }
}

int
main (int argc, char **argv)
{
  const char *csvname = NULL;
  FILE *csv = NULL;
  int i, k;
  double ratio[K_COUNT][NSIZES];
  for (k = 0; k < K_COUNT; k++)
    for (i = 0; i < NSIZES; i++)
      ratio[k][i] = -1.0;	/* not measured */
  unsigned long vlen = vlen_bits ();

  for (i = 1; i < argc; i++)
    {
      if (!strcmp (argv[i], "--csv") && i + 1 < argc)
	csvname = argv[++i];
      else if (!strcmp (argv[i], "-h") || !strcmp (argv[i], "--help"))
	{
	  printf ("usage: %s [--csv FILE]\n", argv[0]);
	  return 0;
	}
      else
	{
	  fprintf (stderr, "unknown option %s\n", argv[i]);
	  return 2;
	}
    }

  src = (unsigned char *) malloc (2 * 65536 + 256);
  dst = (unsigned char *) malloc (2 * 65536 + 256);
  dst2 = (unsigned char *) malloc (2 * 65536 + 256);
  str1 = (unsigned char *) malloc (65536 + 256);
  str2 = (unsigned char *) malloc (65536 + 256);
  ndl = (unsigned char *) malloc (256);
  if (!src || !dst || !dst2 || !str1 || !str2 || !ndl)
    {
      fprintf (stderr, "out of memory\n");
      return 2;
    }

  printf ("# RVV libc function throughput");
  if (vlen)
    printf (" (VLEN = %lu bits)", vlen);
  printf ("\n");
  printf ("# GB/s = bytes per call / time (one direction only); "
	  "best of %d rounds after warm-up\n", ROUNDS);
  printf ("%-12s %8s %12s %12s %10s %10s %8s\n",
	  "function", "size", "rvv ns/call", "glibc ns/call",
	  "rvv GB/s", "glibc GB/s", "rvv/libc");

  if (csvname)
    {
      csv = fopen (csvname, "w");
      if (!csv)
	{
	  perror (csvname);
	  return 2;
	}
      fprintf (csv, "vlen,function,size,rvv_ns,glibc_ns,rvv_gbs,glibc_gbs,"
	       "ratio\n");
    }

  for (k = 0; k < K_COUNT; k++)
    for (i = 0; i < NSIZES; i++)
      {
	size_t n = sizes[i];
	long reps;
	if (k == K_STRSTR_CAND && n > 4096)
	  continue;
	reps = reps_for (n);
	if (k == K_STRSTR_CAND && reps > 256)
	  reps = 256;
	double tr, tl, nr, nl, gr, gl;

	N = n;
	/* buffers for the copy/fill/compare group */
	fill (src, n > 0 ? n : 1, 1);
	fill (dst, n > 0 ? n : 1, 2);
	fill (dst2, n > 0 ? n : 1, 3);
	if (k == K_MEMCMP)
	  {
	    /* equal buffers: the whole range is compared */
	    fill (src, n, 7);
	    memcpy (dst, src, n);
	  }
	if (k == K_MEMMOVE)
	  {
	    /* non-overlapping: dst and dst2 are adjacent halves */
	    fill (dst, n > 0 ? n : 1, 5);
	  }
	/* string group: str1/str2 of length n, NUL at [n] */
	fill_str (str1, n, 11);
	str1[n] = 0;
	fill_str (str2, n, 11);
	str2[n] = 0;
	if (k == K_STRCHR)
	  {
	    /* Place the target at the very end and make sure it does not
	       occur earlier, so the whole string is scanned.  */
	    size_t j;
	    chr_c = 'q';
	    for (j = 0; j < n; j++)
	      if (str1[j] == 'q')
		str1[j] = 'r';
	    if (n > 0)
	      str1[n - 1] = 'q';
	    else
	      str1[0] = 0;
	  }
	if (k == K_STRSTR || k == K_STRSTR_CAND)
	  {
	    if (k == K_STRSTR)
	      strcpy ((char *) ndl, "zqzq");	/* absent: full scan */
	    else
	      {
		/* every position is a candidate: all 'a' haystack,
		   needle "aab" (absent) */
		memset (str1, 'a', n);
		str1[n] = 0;
		strcpy ((char *) ndl, "aab");
	      }
	  }

	tr = measure (0, k, reps);
	tl = measure (1, k, reps);
	nr = tr / (double) reps * 1e9;
	nl = tl / (double) reps * 1e9;
	gr = (double) n * (double) reps / tr / 1e9;
	gl = (double) n * (double) reps / tl / 1e9;
	ratio[k][i] = gl > 0 ? gr / gl : 0.0;
	printf ("%-12s %8zu %12.3f %12.3f %10.3f %10.3f %8.2f\n",
		names[k], n, nr, nl, gr, gl, ratio[k][i]);
	fflush (stdout);
	if (csv)
	  {
	    fprintf (csv, "%lu,%s,%zu,%.3f,%.3f,%.3f,%.3f,%.3f\n",
		     vlen, names[k], n, nr, nl, gr, gl, gl > 0 ? gr / gl : 0.0);
	    fflush (csv);
	  }
      }

  if (csv)
    fclose (csv);
  printf ("\n# rvv slower than glibc (ratio < 1.00):");
  {
    int any = 0;
    for (k = 0; k < K_COUNT; k++)
      for (i = 0; i < NSIZES; i++)
	if (ratio[k][i] >= 0.0 && ratio[k][i] < 1.0)
	  {
	    printf (" %s/%zu(%.2f)", names[k], sizes[i], ratio[k][i]);
	    any = 1;
	  }
    if (!any)
      printf (" none");
  }
  printf ("\n");
  return 0;
}
