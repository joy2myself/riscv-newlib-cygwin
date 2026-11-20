/* speedup = glibc_ns / rvv_ns (>1: the vector routine is faster).
   net_speedup subtracts the empty-call overhead, which matters for
   the 8-64 B rows; ns is the only raw datum.  */
/* bench8.c -- throughput of the eight RVV string routines vs the platform
 * (glibc) routines, in the same style as the libc-funcs bench harness:
 * best of ROUNDS timed rounds after a warm-up, calls through a volatile
 * function-pointer table with an asm barrier so nothing is hoisted.
 *
 * Build (from candidates/, after ./mkvariant.sh all-final ...):
 *   gcc -O2 -static -march=rv64gcv -mabi=lp64d -o bench8 bench8.c \
 *       build/all-final/build/obj/rvv_str{cpy,ncpy,cat,ncat,spn,cspn,pbrk,str}.o
 * Run on a board:   QEMU=native ./bench8            (or just ./bench8)
 * Run under qemu:   qemu-riscv64 -cpu rv64,v=true,vlen=256 ./bench8
 *
 * "GB/s" is payload bytes per call / time (one direction).
 */
#define _GNU_SOURCE 1
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern char *rvv_strcpy (char *, const char *);
extern char *rvv_strncpy (char *, const char *, size_t);
extern char *rvv_strcat (char *, const char *);
extern char *rvv_strncat (char *, const char *, size_t);
extern size_t rvv_strspn (const char *, const char *);
extern size_t rvv_strcspn (const char *, const char *);
extern char *rvv_strpbrk (const char *, const char *);
extern char *rvv_strstr (const char *, const char *);

static volatile uintptr_t sink;
static unsigned char *src, *dst, *hay, *ndl;
static size_t N;
static const char *set;

#define NOP_LEN 65536

static void op_strcpy_r (void) { sink += (uintptr_t) rvv_strcpy ((char *) dst, (char *) src); }
static void op_strcpy_l (void) { sink += (uintptr_t) strcpy ((char *) dst, (char *) src); }
static void op_strncpy_r (void) { sink += (uintptr_t) rvv_strncpy ((char *) dst, (char *) src, N); }
static void op_strncpy_l (void) { sink += (uintptr_t) strncpy ((char *) dst, (char *) src, N); }
static void op_strcat_r (void) { dst[8] = 0; sink += (uintptr_t) rvv_strcat ((char *) dst, (char *) src); }
static void op_strcat_l (void) { dst[8] = 0; sink += (uintptr_t) strcat ((char *) dst, (char *) src); }
static void op_strncat_r (void) { dst[8] = 0; sink += (uintptr_t) rvv_strncat ((char *) dst, (char *) src, N); }
static void op_strncat_l (void) { dst[8] = 0; sink += (uintptr_t) strncat ((char *) dst, (char *) src, N); }
static void op_strspn_r (void) { sink += rvv_strspn ((char *) hay, set); }
static void op_strspn_l (void) { sink += strspn ((char *) hay, set); }
static void op_strcspn_r (void) { sink += rvv_strcspn ((char *) hay, set); }
static void op_strcspn_l (void) { sink += strcspn ((char *) hay, set); }
static void op_strpbrk_r (void) { sink += (uintptr_t) rvv_strpbrk ((char *) hay, set); }
static void op_strpbrk_l (void) { sink += (uintptr_t) strpbrk ((char *) hay, set); }
static void op_strstr_r (void) { sink += (uintptr_t) rvv_strstr ((char *) hay, (char *) ndl); }
static void op_strstr_l (void) { sink += (uintptr_t) strstr ((char *) hay, (char *) ndl); }

enum
{
  K_STRCPY, K_STRNCPY, K_STRCAT, K_STRNCAT,
  K_STRSPN, K_STRCSPN, K_STRPBRK, K_STRSTR,
  K_STRSPN26, K_STRCSPN26, K_STRSTR_CAND, K_COUNT
};

static void (*volatile const ops[2][K_COUNT]) (void) = {
  { op_strcpy_r, op_strncpy_r, op_strcat_r, op_strncat_r,
    op_strspn_r, op_strcspn_r, op_strpbrk_r, op_strstr_r,
    op_strspn_r, op_strcspn_r, op_strstr_r },
  { op_strcpy_l, op_strncpy_l, op_strcat_l, op_strncat_l,
    op_strspn_l, op_strcspn_l, op_strpbrk_l, op_strstr_l,
    op_strspn_l, op_strcspn_l, op_strstr_l }
};

static const char *const names[K_COUNT] = {
  "strcpy", "strncpy", "strcat", "strncat",
  "strspn", "strcspn", "strpbrk", "strstr",
  "strspn26", "strcspn26", "strstr-cand"
};

static const size_t sizes[] =
  { 8, 16, 32, 64, 128, 256, 1024, 4096, 65536 };
#define NSIZES ((int) (sizeof sizes / sizeof *sizes))

#define TARGET_BYTES (4u << 20)
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
  if (r < 64) r = 64;
  if (r > MAX_REPS) r = MAX_REPS;
  return r;
}

static double
measure (int impl, int kind, long reps)
{
  double best = 1e30;
  for (int round = -1; round < ROUNDS; round++)
    {
      double t0 = now_sec ();
      for (long i = 0; i < reps; i++)
	{
	  ops[impl][kind] ();
	  __asm__ volatile ("" ::: "memory");
	}
      double t1 = now_sec ();
      if (round < 0) continue;
      if (t1 - t0 < best) best = t1 - t0;
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

/* bytes 1..255, no NUL, so strlen really is n */
static void
fill_str (unsigned char *p, size_t n, unsigned seed)
{
  uint32_t x = seed * 2654435761u + 1;
  for (size_t i = 0; i < n; i++)
    {
      x ^= x << 13; x ^= x >> 17; x ^= x << 5;
      p[i] = (unsigned char) (1 + ((x >> 16) % 255));
    }
}

int
main (int argc, char **argv)
{
  const char *csvname = NULL;
  double ratio[K_COUNT][NSIZES];
  unsigned long vlen = vlen_bits ();
  static const char set26[] = "abcdefghijklmnopqrstuvwxyz";

  for (int k = 0; k < K_COUNT; k++)
    for (int i = 0; i < NSIZES; i++) ratio[k][i] = -1.0;
  for (int i = 1; i < argc; i++)
    if (!strcmp (argv[i], "--csv") && i + 1 < argc) csvname = argv[++i];

  src = malloc (NOP_LEN + 256);
  dst = malloc (NOP_LEN + 256);
  hay = malloc (NOP_LEN + 256);
  ndl = malloc (256);
  if (!src || !dst || !hay || !ndl) { fprintf (stderr, "oom\n"); return 2; }

  FILE *csv = csvname ? fopen (csvname, "w") : NULL;
  printf ("# RVV string throughput (VLEN = %lu bits), best of %d rounds\n",
	  vlen, ROUNDS);
  printf ("%-12s %8s %12s %12s %8s\n",
	  "function", "size", "rvv ns/call", "glibc ns/call", "rvv/libc");
  if (csv)
    fprintf (csv, "vlen,function,size,rvv_ns,glibc_ns,ratio\n");

  for (int k = 0; k < K_COUNT; k++)
    for (int i = 0; i < NSIZES; i++)
      {
	size_t n = sizes[i];
	long reps = reps_for (n);
	if (k == K_STRSTR_CAND && n > 4096) continue;
	if (k == K_STRSTR_CAND && reps > 256) reps = 256;
	N = n;

	fill_str (src, n > 0 ? n : 1, 11);
	src[n] = 0;
	/* destination: a fixed 8-byte prefix so strcat/strncat have work */
	memcpy (dst, "prefixxx", 8);
	dst[8] = 0;
	fill_str (hay, n > 0 ? n : 1, 7);
	hay[n] = 0;

	if (k == K_STRSPN || k == K_STRPBRK || k == K_STRCSPN)
	  set = "ab";
	else
	  set = set26;

	if (k == K_STRSPN26 || k == K_STRCSPN26)
	  set = set26;

	switch (k)
	  {
	  case K_STRCPY: case K_STRNCPY: case K_STRCAT: case K_STRNCAT:
	    break;
	  case K_STRSPN: case K_STRCSPN: case K_STRPBRK:
	  case K_STRSPN26: case K_STRCSPN26:
	    /* haystack of 'a's: full scan for every member of every set */
	    memset (hay, 'a', n);
	    hay[n] = 0;
	    break;
	  default:
	    strcpy ((char *) ndl, "zqzq");	/* absent: full scan */
	    break;
	  }
	if (k == K_STRSTR_CAND)
	  {
	    memset (hay, 'a', n);
	    hay[n] = 0;
	    strcpy ((char *) ndl, "aab");
	  }

	double tr = measure (0, k, reps), tl = measure (1, k, reps);
	double nr = tr / (double) reps * 1e9, nl = tl / (double) reps * 1e9;
	ratio[k][i] = nl > 0 ? nl / nr : 0.0;
	printf ("%-12s %8zu %12.3f %12.3f %8.2f\n", names[k], n, nr, nl,
		ratio[k][i]);
	fflush (stdout);
	if (csv)
	  {
	    fprintf (csv, "%lu,%s,%zu,%.3f,%.3f,%.3f\n", vlen, names[k], n,
		     nr, nl, ratio[k][i]);
	    fflush (csv);
	  }
      }

  printf ("\n# rvv slower than glibc (ratio < 1.00):");
  int any = 0;
  for (int k = 0; k < K_COUNT; k++)
    for (int i = 0; i < NSIZES; i++)
      if (ratio[k][i] >= 0.0 && ratio[k][i] < 1.0)
	{
	  printf (" %s/%zu(%.2f)", names[k], sizes[i], ratio[k][i]);
	  any = 1;
	}
  if (!any) printf (" none");
  printf ("\n");
  if (csv) fclose (csv);
  return 0;
}
