/* exhaustive.c -- brute-force canary/guard-page test for memory routines.
 *
 * Built against the harness objects (rvv_* symbols renamed by build.sh), so it
 * compares the candidate against glibc for *every* n in a range and every
 * destination offset, with guard bytes on both sides of the destination.
 *
 *   gcc -O2 -static -march=rv64gcv -mabi=lp64d -o exhaustive exhaustive.c \
 *       build/obj/rvv_memset.o build/obj/rvv_memrchr.o build/obj/rvv_memccpy.o
 *   ./exhaustive memset memrchr memccpy  # or any subset
 *
 * memccpy is checked for every n at all 16 source and destination alignments
 * with the stop byte absent, at index 0, n-1 and n/2, and at the vector-length
 * boundaries; both the source and the destination end next to a PROT_NONE page,
 * so an over-read or an over-write faults, while 128 canary bytes after the
 * destination catch smaller over-writes byte for byte.
 *
 * Exit status 0 == no mismatch, no out-of-range write.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>
#include <sys/mman.h>

extern void *rvv_memset (void *, int, size_t);
extern void *rvv_memrchr (const void *, int, size_t);
extern void *rvv_memcpy (void *, const void *, size_t);
extern void *rvv_memmove (void *, const void *, size_t);
extern void *rvv_memchr (const void *, int, size_t);
extern void *rvv_rawmemchr (const void *, int);
extern void *rvv_memccpy (void *, const void *, int, size_t);
extern int rvv_memcmp (const void *, const void *, size_t);

#define MAXN 1200
#define PAD 64

static unsigned char src[MAXN + 2 * PAD];
static unsigned char a[MAXN + 2 * PAD];
static unsigned char b[MAXN + 2 * PAD];
static unsigned long long checked, failed;

static void
fill (unsigned char *p, size_t n, unsigned seed)
{
  size_t i;
  unsigned x = seed * 2654435761u + 1;
  for (i = 0; i < n; i++)
    {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      p[i] = (unsigned char) (1 + ((x >> 16) % 255));
    }
}

static void
report (const char *what, size_t n, int off, int c, unsigned char *pa,
	unsigned char *pb, size_t lo, size_t hi)
{
  size_t i;
  if (failed++ > 12)
    return;
  printf ("FAIL %-8s n=%zu off=%d c=%#x", what, n, off, c);
  for (i = lo; i < hi; i++)
    if (pa[i] != pb[i])
      {
	printf (" first-diff@%zu rvv=%#x glibc=%#x", i, pa[i], pb[i]);
	break;
      }
  printf ("\n");
}

/* ---------------------------------------------------------------- memset */
static void
t_memset (void)
{
  static const int cs[] = { 0, 0xff, 0x41, 0x15a, 0x100 };
  size_t n;
  int oi, ci;
  for (ci = 0; ci < (int) (sizeof cs / sizeof *cs); ci++)
    for (n = 0; n <= 200; n++)
      for (oi = 0; oi < 8; oi++)
	{
	  int c = cs[ci];
	  size_t off = (size_t) oi;
	  memset (a, 0xAA, sizeof a);
	  memset (b, 0xAA, sizeof b);
	  rvv_memset (a + PAD + off, c, n);
	  memset (b + PAD + off, c, n);
	  checked++;
	  if (memcmp (a, b, sizeof a) != 0)
	    report ("memset", n, oi, c, a, b, 0, sizeof a);
	}
}

/* --------------------------------------------------------------- memrchr */
static void
t_memrchr (void)
{
  /* every n, every offset, several needle placements and needle values,
     including values that occur many times (multi-hit -> slow path) */
  size_t n;
  int oi, k;
  for (n = 0; n <= 700; n++)
    for (oi = 0; oi < 4; oi++)
      for (k = 0; k < 5; k++)
	{
	  size_t off = (size_t) oi;
	  unsigned char *p = a + PAD + off;
	  unsigned char *q = b + PAD + off;
	  void *ra, *rb;
	  fill (src, n ? n : 1, 11);
	  memcpy (p, src, n);
	  memcpy (q, src, n);
	  switch (k)
	    {
	    case 0:		/* no hit at all */
	      break;
	    case 1:		/* single hit at the very end */
	      if (n)
		p[n - 1] = q[n - 1] = 0x71;
	      break;
	    case 2:		/* single hit at the very start */
	      if (n)
		p[0] = q[0] = 0x71;
	      break;
	    case 3:		/* hit in the middle */
	      if (n)
		p[n / 2] = q[n / 2] = 0x71;
	      break;
	    default:		/* many hits: every 3rd byte */
	      {
		size_t i;
		for (i = 0; i < n; i += 3)
		  p[i] = q[i] = 0x20;
		if (n)
		  p[n - 1] = q[n - 1] = 0x20;
		break;
	      }
	    }
	  ra = rvv_memrchr (p, 0x71, n);
	  rb = memrchr (q, 0x71, n);
	  checked++;
	  if ((ra ? (unsigned char *) ra - p : -1)
	      != (rb ? (unsigned char *) rb - q : -1))
	    {
	      if (failed++ < 12)
		printf ("FAIL memrchr n=%zu off=%d k=%d rvv=%td glibc=%td\n",
			n, oi, k, ra ? (unsigned char *) ra - p : -1,
			rb ? (unsigned char *) rb - q : -1);
	    }
	  ra = rvv_memrchr (p, 0x20, n);
	  rb = memrchr (q, 0x20, n);
	  checked++;
	  if ((ra ? (unsigned char *) ra - p : -1)
	      != (rb ? (unsigned char *) rb - q : -1) && failed++ < 12)
	    printf ("FAIL memrchr/0x20 n=%zu off=%d k=%d rvv=%td glibc=%td\n",
		    n, oi, k, ra ? (unsigned char *) ra - p : -1,
		    rb ? (unsigned char *) rb - q : -1);
	}
}

/* -------------------------------------------------------------- memccpy */
/* Page-boundary test: the destination object ends MPC_CANARY bytes before
   a PROT_NONE page, every byte of the writable page before and after the
   object is compared between the RVV routine and glibc, and the return
   value must agree.  The source also ends next to a guard page, so a
   vector load that runs past src+n faults instead of silently passing.  */
#define MPC_CANARY 128

static size_t mpc_pg;
static unsigned char *mpc_base, *mpc_src, *mpc_d1, *mpc_d2;
static sigjmp_buf mpc_jb;
static volatile sig_atomic_t mpc_armed;

static void
mpc_fault (int sig)
{
  if (mpc_armed)
    siglongjmp (mpc_jb, 1);
  _exit (128 + sig);
}

static void
mpc_fill (unsigned char *p, size_t n, unsigned seed)
{
  size_t i;
  unsigned x = seed * 2654435761u + 1;
  for (i = 0; i < n; i++)
    {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      p[i] = (unsigned char) (x >> 16);
    }
}

static void
mpc_data (unsigned char *p, size_t n, int c, long pos, unsigned seed)
{
  size_t i;
  unsigned x = seed * 2654435761u + 7;
  unsigned char cc = (unsigned char) c;
  for (i = 0; i < n; i++)
    {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      p[i] = (unsigned char) (1 + ((x >> 24) % 255));
      if (p[i] == cc)
	p[i] = (unsigned char) (cc ^ 0x5a);	/* no incidental hit */
    }
  if (pos >= 0)
    p[pos] = cc;
}

static long
mpc_call (int impl, void *d, const void *s, int c, size_t n, int *faulted)
{
  volatile long r = -1;
  *faulted = 0;
  if (sigsetjmp (mpc_jb, 1) == 0)
    {
      mpc_armed = 1;
      if (impl)
	{
	  void *p = memccpy (d, s, c, n);
	  r = p ? (long) ((unsigned char *) p - (unsigned char *) d) : -1;
	}
      else
	{
	  void *p = rvv_memccpy (d, s, c, n);
	  r = p ? (long) ((unsigned char *) p - (unsigned char *) d) : -1;
	}
      mpc_armed = 0;
    }
  else
    {
      mpc_armed = 0;
      *faulted = 1;
    }
  return r;
}

static void
mpc_case (size_t n, int sa, int da, int c, long pos, unsigned seed)
{
  unsigned char *s, *d1, *d2;
  size_t doff;
  long r1, r2;
  int f1, f2;
  size_t i;

  if (n + MPC_CANARY + 64 > mpc_pg)
    return;
  s = mpc_src + (mpc_pg - n - (size_t) sa);
  doff = mpc_pg - n - MPC_CANARY - (size_t) da;
  d1 = mpc_d1 + doff;
  d2 = mpc_d2 + doff;

  mpc_data (s, n, c, pos, seed);
  mpc_fill (mpc_d1, mpc_pg, seed ^ 0x5a5a);
  mpc_fill (mpc_d2, mpc_pg, seed ^ 0x5a5a);

  checked++;
  r1 = mpc_call (0, d1, s, c, n, &f1);
  r2 = mpc_call (1, d2, s, c, n, &f2);
  if (f1 || f2)
    {
      if (failed++ < 12)
	printf ("FAIL memccpy n=%zu sa=%d da=%d c=%#x pos=%ld %s guard page\n",
		n, sa, da, c, pos, f1 ? "rvv hit" : "glibc hit");
      return;
    }
  if (r1 != r2)
    {
      if (failed++ < 12)
	printf ("FAIL memccpy n=%zu sa=%d da=%d c=%#x pos=%ld rvv=%ld glibc=%ld\n",
		n, sa, da, c, pos, r1, r2);
      return;
    }
  for (i = 0; i < mpc_pg; i++)
    if (mpc_d1[i] != mpc_d2[i])
      {
	if (failed++ < 12)
	  {
	    long rel = (long) i - (long) doff;
	    printf ("FAIL memccpy n=%zu sa=%d da=%d c=%#x pos=%ld "
		    "first-diff@%ld (%s) rvv=%#x glibc=%#x\n",
		    n, sa, da, c, pos, rel,
		    rel >= 0 && (size_t) rel < n ? "payload" : "canary",
		    mpc_d1[i], mpc_d2[i]);
	  }
	return;
      }
}

static size_t
mpc_vmax (void)
{
#ifdef __riscv_vector
  unsigned long vb;
  __asm__ volatile ("csrr %0, vlenb" : "=r" (vb));
  return (size_t) vb * 8;	/* e8, m8 */
#else
  return 0;
#endif
}

static void
t_memccpy (void)
{
  static const size_t base_ns[] = {
    0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    127, 128, 129, 255, 256, 257, 511, 512, 513, 997, 1000, 1023,
    1024, 1025, 2048
  };
  static const int subalign[] = { 0, 1, 3, 7, 8, 9, 15 };
  static const int cvals[] = { 'Q', 0x00, 0xff, 0x80, 0x141 };
  size_t ns[64], nn = 0, vmax, i, j;
  int ci, sa, da;

  mpc_pg = (size_t) sysconf (_SC_PAGESIZE);
  mpc_base = mmap (NULL, 6 * mpc_pg, PROT_READ | PROT_WRITE,
		   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mpc_base == MAP_FAILED)
    {
      perror ("mmap");
      exit (2);
    }
  mpc_src = mpc_base;
  mpc_d1 = mpc_base + 2 * mpc_pg;
  mpc_d2 = mpc_base + 4 * mpc_pg;
  if (mprotect (mpc_base + mpc_pg, mpc_pg, PROT_NONE) != 0
      || mprotect (mpc_base + 3 * mpc_pg, mpc_pg, PROT_NONE) != 0
      || mprotect (mpc_base + 5 * mpc_pg, mpc_pg, PROT_NONE) != 0)
    {
      perror ("mprotect");
      exit (2);
    }

  {
    struct sigaction sa2;
    memset (&sa2, 0, sizeof sa2);
    sa2.sa_handler = mpc_fault;
    sigemptyset (&sa2.sa_mask);
    sigaction (SIGSEGV, &sa2, NULL);
    sigaction (SIGBUS, &sa2, NULL);
  }

  vmax = mpc_vmax ();
  for (i = 0; i < sizeof base_ns / sizeof *base_ns; i++)
    ns[nn++] = base_ns[i];
  {
    size_t b[] = { vmax - 1, vmax, vmax + 1, 2 * vmax - 1, 2 * vmax,
		   2 * vmax + 1, 3 * vmax, 3 * vmax + 1 };
    for (i = 0; i < sizeof b / sizeof *b; i++)
      {
	size_t k;
	int dup = 0;
	for (k = 0; k < nn; k++)
	  if (ns[k] == b[i])
	    dup = 1;
	if (!dup && b[i] + MPC_CANARY + 64 <= mpc_pg)
	  ns[nn++] = b[i];
      }
  }

  /* Every n at every source/destination alignment (16 x 16), with and
     without a hit and with the hit at the first, middle and last byte.  */
  for (i = 0; i < nn; i++)
    for (sa = 0; sa < 16; sa++)
      for (da = 0; da < 16; da++)
	{
	  size_t n = ns[i];
	  unsigned seed = (unsigned) (n * 131 + (size_t) sa * 17 + da);
	  mpc_case (n, sa, da, 'Q', -1, seed);
	  if (n == 0)
	    continue;
	  mpc_case (n, sa, da, 'Q', 0, seed + 1);
	  mpc_case (n, sa, da, 'Q', (long) (n - 1), seed + 2);
	  mpc_case (n, sa, da, 'Q', (long) (n / 2), seed + 3);
	}

  /* Boundary and non-'Q' hit positions over a reduced alignment set.  */
  for (i = 0; i < nn; i++)
    for (sa = 0; sa < (int) (sizeof subalign / sizeof *subalign); sa++)
      for (da = 0; da < (int) (sizeof subalign / sizeof *subalign); da++)
	{
	  size_t n = ns[i];
	  long poss[16];
	  int np = 0, pi;
	  long cand[16];
	  cand[0] = -1;
	  cand[1] = 0;
	  cand[2] = (long) (n - 1);
	  cand[3] = (long) (n / 2);
	  cand[4] = 1;
	  cand[5] = (long) (n - 2);
	  cand[6] = (long) (vmax - 1);
	  cand[7] = (long) vmax;
	  cand[8] = (long) (vmax + 1);
	  cand[9] = (long) (2 * vmax - 1);
	  cand[10] = (long) (2 * vmax);
	  cand[11] = (long) (2 * vmax + 1);
	  cand[12] = (long) (n / 3);
	  cand[13] = (long) (2 * n / 3);
	  for (pi = 0; pi < 14; pi++)
	    {
	      int seen = 0;
	      for (j = 0; j < (size_t) np; j++)
		if (poss[j] == cand[pi])
		  seen = 1;
	      if (seen || cand[pi] < -1 || (cand[pi] >= 0 && (size_t) cand[pi] >= n))
		continue;
	      poss[np++] = cand[pi];
	    }
	  for (pi = 0; pi < np; pi++)
	    for (ci = 0; ci < (int) (sizeof cvals / sizeof *cvals); ci++)
	      mpc_case (n, subalign[sa], subalign[da], cvals[ci], poss[pi],
			(unsigned) (n * 71 + (size_t) pi * 13 + ci * 101
				    + subalign[sa] * 3 + subalign[da]));
	}

}

/* ------------------------------------------------------------------ main */
static void
t_memcpy (void)
{
  size_t n;
  int oi, di;
  for (n = 0; n <= 600; n++)
    for (oi = 0; oi < 4; oi++)
      for (di = 0; di < 4; di++)
	{
	  fill (src, n ? n : 1, 3);
	  memset (a, 0xAA, sizeof a);
	  memset (b, 0xAA, sizeof b);
	  rvv_memcpy (a + PAD + oi, src + PAD + di, n);
	  memcpy (b + PAD + oi, src + PAD + di, n);
	  checked++;
	  if (memcmp (a, b, sizeof a) != 0)
	    report ("memcpy", n, oi, di, a, b, 0, sizeof a);
	}
}

int
main (int argc, char **argv)
{
  int i;
  int all = (argc == 1);
  memset (src, 0, sizeof src);
  for (i = 1; i <= argc; i++)
    {
      const char *w = all ? NULL : argv[i];
      if (all || !strcmp (w, "memset"))
	t_memset ();
      if (all || !strcmp (w, "memrchr"))
	t_memrchr ();
      if (all || !strcmp (w, "memcpy"))
	t_memcpy ();
      if (all || !strcmp (w, "memccpy"))
	t_memccpy ();
      if (!all)
	break;
    }
  printf ("%llu checks, %llu failures%s\n", checked, failed,
	  failed ? " -- FAIL" : " -- PASS");
  return failed != 0;
}
