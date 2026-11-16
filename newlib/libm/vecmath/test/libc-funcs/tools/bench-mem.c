/* bench9.c -- throughput harness for ALL nine RVV memory routines.
 *
 * The shipped harness/bench/bench.c only measures memcpy, memset, memmove
 * and memcmp.  This companion measures those four plus mempcpy, memchr,
 * memrchr, rawmemchr and memccpy, using exactly the same methodology
 * (volatile function-pointer table, asm barrier, warm-up round, best of 3
 * rounds, ~8 MiB moved per round, GB/s = payload bytes per call / time).
 *
 * The rvv_* entry points are the objcopy-renamed newlib routines produced
 * by test/libc-funcs/build.sh, so this binary compares a candidate against
 * the platform (glibc) routine of the same name in one process.
 *
 * Build on a board (after ./build.sh in test/libc-funcs with the candidate
 * installed as the real .S):
 *
 *   gcc -O2 -static -march=rv64gcv -mabi=lp64d -o bench9 bench9.c \
 *       build/obj/rvv_mem*.o build/obj/rvv_rawmemchr.o
 *
 * Run:
 *   ./bench9 [--csv file.csv]
 *
 * Plain POSIX C + clock_gettime; no dependency on the harness headers.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void *rvv_memcpy (void *, const void *, size_t);
extern void *rvv_mempcpy (void *, const void *, size_t);
extern void *rvv_memset (void *, int, size_t);
extern void *rvv_memmove (void *, const void *, size_t);
extern int rvv_memcmp (const void *, const void *, size_t);
extern void *rvv_memchr (const void *, int, size_t);
extern void *rvv_memrchr (const void *, int, size_t);
extern void *rvv_rawmemchr (const void *, int);
extern void *rvv_memccpy (void *, const void *, int, size_t);

static volatile uintptr_t sink;
static unsigned char *src, *dst, *dst2, *str_;
static size_t N;
static int chr_c = 'q';

/* ---- one wrapper per (function, implementation) --------------------- */
static void op_memcpy_r (void) { sink += (uintptr_t) rvv_memcpy (dst, src, N); }
static void op_memcpy_l (void) { sink += (uintptr_t) memcpy (dst, src, N); }
static void op_mempcpy_r (void) { sink += (uintptr_t) rvv_mempcpy (dst, src, N); }
static void op_mempcpy_l (void) { sink += (uintptr_t) mempcpy (dst, src, N); }
static void op_memset_r (void) { sink += (uintptr_t) rvv_memset (dst, 0xa5, N); }
static void op_memset_l (void) { sink += (uintptr_t) memset (dst, 0xa5, N); }
static void op_memmove_r (void) { sink += (uintptr_t) rvv_memmove (dst2, dst, N); }
static void op_memmove_l (void) { sink += (uintptr_t) memmove (dst2, dst, N); }
static void op_memcmp_r (void) { sink += (uintptr_t) rvv_memcmp (src, dst, N); }
static void op_memcmp_l (void) { sink += (uintptr_t) memcmp (src, dst, N); }
static void op_memchr_r (void) { sink += (uintptr_t) rvv_memchr (str_, chr_c, N); }
static void op_memchr_l (void) { sink += (uintptr_t) memchr (str_, chr_c, N); }
static void op_memrchr_r (void) { sink += (uintptr_t) rvv_memrchr (str_, chr_c, N); }
static void op_memrchr_l (void) { sink += (uintptr_t) memrchr (str_, chr_c, N); }
static void op_rawmemchr_r (void) { sink += (uintptr_t) rvv_rawmemchr (str_, chr_c); }
static void op_rawmemchr_l (void) { sink += (uintptr_t) rawmemchr (str_, chr_c); }
static void op_memccpy_r (void) { sink += (uintptr_t) rvv_memccpy (dst, str_, chr_c, N); }
static void op_null (void) { }
static void op_memccpy_l (void) { sink += (uintptr_t) memccpy (dst, str_, chr_c, N); }

enum
{
  K_MEMCPY, K_MEMSET, K_MEMMOVE, K_MEMCMP, K_MEMPCPY,
  K_MEMCHR, K_MEMRCHR, K_RAWMEMCHR, K_MEMCCPY, K_COUNT
};

static void (*volatile const ops[2][K_COUNT]) (void) = {
  { op_memcpy_r, op_memset_r, op_memmove_r, op_memcmp_r, op_mempcpy_r,
    op_memchr_r, op_memrchr_r, op_rawmemchr_r, op_memccpy_r },
  { op_memcpy_l, op_memset_l, op_memmove_l, op_memcmp_l, op_mempcpy_l,
    op_memchr_l, op_memrchr_l, op_rawmemchr_l, op_memccpy_l }
};

static const char *const names[K_COUNT] = {
  "memcpy", "memset", "memmove", "memcmp", "mempcpy",
  "memchr", "memrchr", "rawmemchr", "memccpy"
};

static const size_t sizes[] =
  { 8, 16, 32, 64, 128, 256, 1024, 4096, 65536 };
#define NSIZES ((int) (sizeof sizes / sizeof *sizes))

#define TARGET_BYTES (8u << 20)
#define ROUNDS 3

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
      double t0 = now_sec (), t1;
      for (i = 0; i < reps; i++)
	{
	  ops[impl][kind] ();
	  __asm__ volatile ("" ::: "memory");
	}
      t1 = now_sec ();
      if (round < 0)
	continue;
      if (t1 - t0 < best)
	best = t1 - t0;
    }
  return best;
}

/* Same loop shape as measure(), but for an empty function: what one call
   through the volatile pointer table costs by itself.  Subtract it from both
   ns/call columns before comparing 8/16/32 B rows.  */
static double
measure_null (long reps)
{
  void (*volatile p) (void) = op_null;
  double best = 1e30;
  int round;
  long i;
  for (round = -1; round < ROUNDS; round++)
    {
      double t0 = now_sec (), t1;
      for (i = 0; i < reps; i++)
	{
	  p ();
	  __asm__ volatile ("" ::: "memory");
	}
      t1 = now_sec ();
      if (round < 0)
	continue;
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

/* Bytes 1..255, so the target character never occurs by accident.  */
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
      p[i] = (unsigned char) (1 + ((x >> 16) % 255));
    }
}

int
main (int argc, char **argv)
{
  const char *csvname = NULL;
  FILE *csv = NULL;
  int mrchr_first = 0;		/* --memrchr-first: needle at src[0] */
  int i, k;
  double ratio[K_COUNT][NSIZES];
  unsigned long vlen = vlen_bits ();

  for (k = 0; k < K_COUNT; k++)
    for (i = 0; i < NSIZES; i++)
      ratio[k][i] = -1.0;

  for (i = 1; i < argc; i++)
    {
      if (!strcmp (argv[i], "--csv") && i + 1 < argc)
	csvname = argv[++i];
      else if (!strcmp (argv[i], "--memrchr-first"))
	mrchr_first = 1;
      else
	{
	  fprintf (stderr, "usage: %s [--csv FILE] [--memrchr-first]\n", argv[0]);
	  return 2;
	}
    }

  src = (unsigned char *) malloc (2 * 65536 + 256);
  dst = (unsigned char *) malloc (2 * 65536 + 256);
  dst2 = (unsigned char *) malloc (2 * 65536 + 256);
  str_ = (unsigned char *) malloc (65536 + 256);
  if (!src || !dst || !dst2 || !str_)
    {
      fprintf (stderr, "out of memory\n");
      return 2;
    }

  printf ("# RVV mem* throughput, all nine routines");
  if (vlen)
    printf (" (VLEN = %lu bits)", vlen);
  printf ("\n# GB/s = bytes per call / time (one direction only); "
	  "best of %d rounds after warm-up\n", ROUNDS);
  if (mrchr_first)
    printf ("# memrchr: needle at src[0] (full backward scan)\n");
  printf ("%-11s %8s %12s %12s %10s %10s %8s\n",
	  "function", "size", "rvv ns/call", "libc ns/call",
	  "rvv GB/s", "libc GB/s", "rvv/libc");
  {
    long nr = reps_for (8);
    printf ("# null-call overhead: %.3f ns/call -- subtract it from both "
	    "ns/call columns\n#   before comparing the 8/16/32 B rows\n",
	    measure_null (nr) / (double) nr * 1e9);
  }

  if (csvname)
    {
      csv = fopen (csvname, "w");
      if (!csv)
	{
	  perror (csvname);
	  return 2;
	}
      fprintf (csv, "vlen,function,size,rvv_ns,libc_ns,rvv_gbs,libc_gbs,"
	       "ratio\n");
    }

  for (k = 0; k < K_COUNT; k++)
    for (i = 0; i < NSIZES; i++)
      {
	size_t n = sizes[i];
	long reps = reps_for (n);
	double tr, tl, nr, nl, gr, gl;

	N = n;
	fill (src, n > 0 ? n : 1, 1);
	fill (dst, n > 0 ? n : 1, 2);
	fill (dst2, n > 0 ? n : 1, 3);
	if (k == K_MEMCMP || k == K_MEMCHR || k == K_MEMRCHR
	    || k == K_RAWMEMCHR || k == K_MEMCCPY)
	  {
	    /* identical and target-free content; the target sits at the very
	       end so the whole range is scanned / copied */
	    fill (str_, n, 7);
	    memcpy (dst, str_, n);
	    chr_c = 'q';
	    if (n > 0)
	      /* --memrchr-first: for memrchr only, put the needle at the very
	         start, so the backward scan has to read all n bytes (the
	         default layout is found by memrchr in its first strip, which
	         measures per-call latency rather than scanning throughput).  */
	      str_[(mrchr_first && k == K_MEMRCHR) ? 0 : n - 1]
		= (unsigned char) chr_c;
	    else
	      str_[0] = (unsigned char) chr_c;
	    if (k == K_RAWMEMCHR)
	      str_[n] = 0;		/* rawmemchr has no length */
	  }

	tr = measure (0, k, reps);
	tl = measure (1, k, reps);
	nr = tr / (double) reps * 1e9;
	nl = tl / (double) reps * 1e9;
	gr = (double) n * (double) reps / tr / 1e9;
	gl = (double) n * (double) reps / tl / 1e9;
	ratio[k][i] = gl > 0 ? gr / gl : 0.0;
	printf ("%-11s %8zu %12.3f %12.3f %10.3f %10.3f %8.2f\n",
		names[k], n, nr, nl, gr, gl, ratio[k][i]);
	fflush (stdout);
	if (csv)
	  {
	    fprintf (csv, "%lu,%s,%zu,%.3f,%.3f,%.3f,%.3f,%.3f\n",
		     vlen, names[k], n, nr, nl, gr, gl,
		     gl > 0 ? gr / gl : 0.0);
	    fflush (csv);
	  }
      }

  if (csv)
    fclose (csv);
  printf ("\n# rvv slower than libc (ratio < 1.00):");
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
