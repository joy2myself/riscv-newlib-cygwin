/* memcpy, mempcpy, memmove, memccpy.  */

#include "harness.h"
#include "tests.h"
#include <string.h>

#define MAX_SMALL 200

static int
sub_align (int a)
{
  return a == 0 || a == 1 || a == 3 || a == 7 || a == 8 || a == 9 || a == 15;
}

static void
prep_dest (unsigned char *region, size_t align, size_t touch, unsigned seed)
{
  canary_fill (region + WIN_OFF, WIN_LEN (touch) + align, seed);
}

static void
run_dest (int fk, const char *label, unsigned char *r1, unsigned char *r2,
	  unsigned char *d1, unsigned char *d2, const unsigned char *src,
	  size_t n, size_t align, size_t touch)
{
  long a = fcall (IMPL_RVV, fk, d1, src, n, 0);
  long b = fcall (IMPL_LIBC, fk, d2, src, n, 0);
  chk_long (label, a, b);
  chk_win (label, r1 + WIN_OFF, r2 + WIN_OFF, touch, align);
}

/* ------------------------------------------------------------------ */
/* memcpy / mempcpy                                                   */
/* ------------------------------------------------------------------ */
static void
copy_sweep (int fk)
{
  unsigned char *ro = ro_base (), *r1 = d1_base (), *r2 = d2_base ();
  size_t n;
  int sa, da, v;

  for (n = 0; n <= MAX_SMALL; n++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      for (da = 0; da < ALIGN_COUNT; da++)
	for (v = 0; v < 3; v++)
	  {
	    unsigned char *s = win (ro, sa);
	    unsigned char *d1 = win (r1, da), *d2 = win (r2, da);
	    if (v >= 1 && !(sub_align (sa) && sub_align (da)))
	      continue;
	    make_data (s, n, v == 2 ? 7 : v);
	    prep_dest (r1, da, n, 21);
	    prep_dest (r2, da, n, 21);
	    t_case ("n=%zu sa=%d da=%d pat=%d", n, sa, da, v);
	    run_dest (fk, fk_name (fk), r1, r2, d1, d2, s, n, da, n);
	  }

  {
    static const size_t big[] = { 997, 4096, 100003 };
    int i;
    for (i = 0; i < (int) (sizeof big / sizeof *big); i++)
      for (sa = 0; sa < ALIGN_COUNT; sa++)
	{
	  size_t k = big[i], da = (size_t) ((sa * 5) % 16);
	  unsigned char *s = win (ro, sa);
	  unsigned char *d1 = win (r1, da), *d2 = win (r2, da);
	  make_data (s, k, 0);
	  prep_dest (r1, da, k, 21);
	  prep_dest (r2, da, k, 21);
	  t_case ("n=%zu sa=%d da=%zu", k, sa, da);
	  run_dest (fk, fk_name (fk), r1, r2, d1, d2, s, k, da, k);
	}
  }
}

void
test_memcpy (void)
{
  t_begin ("memcpy");
  copy_sweep (FK_memcpy);
  edge_memcpy ();
  t_end ();
}

void
test_mempcpy (void)
{
  t_begin ("mempcpy");
  copy_sweep (FK_mempcpy);
  edge_mempcpy ();
  t_end ();
}

/* ------------------------------------------------------------------ */
/* memmove                                                            */
/* ------------------------------------------------------------------ */
/* One overlapping (or not) move inside a region.  Both regions get
   identical content, so the whole window can be compared afterwards.  */
static void
move_one (unsigned char *r1, unsigned char *r2, size_t n, int sa, long d,
	  int dir, unsigned pat)
{
  size_t base = WIN_OFF;
  size_t touch = n + 512;
  size_t wlen = WIN_LEN (touch);
  unsigned char *s1 = r1 + base + 256 + sa;
  unsigned char *s2 = r2 + base + 256 + sa;
  unsigned char *d1, *d2;
  long a, b;

  canary_fill (r1 + base, wlen, 31);
  canary_fill (r2 + base, wlen, 31);
  make_data (s1, n, pat);
  make_data (s2, n, pat);

  d1 = s1 + dir * d;
  d2 = s2 + dir * d;

  t_case ("n=%zu sa=%d d=%ld dir=%s", n, sa, d, dir < 0 ? "back" : "fwd");
  a = fcall (IMPL_RVV, FK_memmove, d1, s1, n, 0);
  b = fcall (IMPL_LIBC, FK_memmove, d2, s2, n, 0);
  chk_long ("return", a, b);
  chk_win ("overlap", r1 + base, r2 + base, touch, 0);
}

static void
move_sweep (void)
{
  unsigned char *r1 = m1_base (), *r2 = m2_base ();
  size_t n;
  int sa;
  long d;

  /* every small length, every source alignment, distances 1..40 and a few
     larger ones, both overlap directions plus dst == src.  */
  for (n = 0; n <= MAX_SMALL; n++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      {
	move_one (r1, r2, n, sa, 0, 1, 0);	/* dst == src */
	for (d = 1; d <= 40; d++)
	  {
	    move_one (r1, r2, n, sa, d, 1, 0);	/* dst > src */
	    move_one (r1, r2, n, sa, d, -1, 0);	/* dst < src */
	  }
	if (!sub_align (sa))
	  continue;
	move_one (r1, r2, n, sa, (long) n, 1, 2);	/* adjacent, no overlap */
	move_one (r1, r2, n, sa, (long) n, -1, 2);
	move_one (r1, r2, n, sa, 1024, 1, 1);
	move_one (r1, r2, n, sa, 1024, -1, 1);
      }

  {
    static const size_t big[] = { 997, 4096, 100003 };
    static const long dist[] = { 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40,
				 -1, 0, 1, 2 };	/* -1/0/1/2 are offsets
							   relative to n, see
							   below */
    int i, j;
    for (i = 0; i < (int) (sizeof big / sizeof *big); i++)
      for (sa = 0; sa < ALIGN_COUNT; sa++)
	for (j = 0; j < (int) (sizeof dist / sizeof *dist); j++)
	  {
	    long d = dist[j];
	    if (j >= 13)	/* n-1, n, n+1, n+2 */
	      d = (long) big[i] + (dist[j] == -1 ? -1 : dist[j]);
	    if (big[i] < 100000 || (sa % 4) == 0)
	      {
		move_one (r1, r2, big[i], sa, d, 1, 0);
		move_one (r1, r2, big[i], sa, d, -1, 0);
	      }
	  }
  }

}

void
test_memmove (void)
{
  t_begin ("memmove");
  move_sweep ();
  edge_memmove ();
  t_end ();
}

/* ------------------------------------------------------------------ */
/* memccpy                                                            */
/* ------------------------------------------------------------------ */
void
test_memccpy (void)
{
  unsigned char *ro = ro_base (), *r1 = d1_base (), *r2 = d2_base ();
  static const int cs[] = { 'a', 0x00, 0xff, 0x55 };
  size_t n;
  int sa, da, ci, pi;

  t_begin ("memccpy");
  for (n = 0; n <= MAX_SMALL; n++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      for (da = 0; da < ALIGN_COUNT; da++)
	{
	  unsigned char *s = win (ro, sa);
	  for (ci = 0; ci < (int) (sizeof cs / sizeof *cs); ci++)
	    for (pi = 0; pi < 4; pi++)
	      {
		int c = cs[ci];
		long pos;
		unsigned char *d1, *d2;
		long a, b;

		if (pi > 0 && n == 0)
		  continue;
		if ((ci > 0 || pi > 0) && !(sub_align (sa) && sub_align (da)))
		  continue;
		switch (pi)
		  {
		  case 0: pos = -1; break;			/* absent */
		  case 1: pos = 0; break;
		  case 2: pos = (long) (n / 2); break;
		  default: pos = (long) (n - 1); break;
		  }
		/* Build source data without any incidental occurrence of c,
		   then place c exactly where wanted.  */
		make_data (s, n, ci & 1);
		{
		  size_t i;
		  for (i = 0; i < n; i++)
		    if (s[i] == (unsigned char) c)
		      s[i] = (unsigned char) ~c;
		}
		if (pos >= 0)
		  s[pos] = (unsigned char) c;

		d1 = win (r1, da);
		d2 = win (r2, da);
		prep_dest (r1, da, n, 23);
		prep_dest (r2, da, n, 23);
		t_case ("n=%zu sa=%d da=%d c=0x%02x pos=%ld", n, sa, da,
			(unsigned) (c & 0xff), pos);
		a = fcall (IMPL_RVV, FK_memccpy, d1, s, n, c);
		b = fcall (IMPL_LIBC, FK_memccpy, d2, s, n, c);
		chk_long ("memccpy", a, b);
		chk_win ("memccpy", r1 + WIN_OFF, r2 + WIN_OFF, n, (size_t) da);
	      }
	}

  {
    static const size_t big[] = { 997, 4096, 100003 };
    int i;
    for (i = 0; i < (int) (sizeof big / sizeof *big); i++)
      for (sa = 0; sa < ALIGN_COUNT; sa++)
	{
	  size_t k = big[i];
	  size_t da = (size_t) ((sa * 7) % 16);
	  unsigned char *s = win (ro, sa);
	  unsigned char *d1 = win (r1, da), *d2 = win (r2, da);
	  long pos = (long) (k / 3);
	  long a, b;
	  make_data (s, k, 0);
	  s[pos] = 'Q';
	  prep_dest (r1, da, k, 23);
	  prep_dest (r2, da, k, 23);
	  t_case ("n=%zu sa=%d da=%zu c='Q' pos=%ld", k, sa, da, pos);
	  a = fcall (IMPL_RVV, FK_memccpy, d1, s, k, 'Q');
	  b = fcall (IMPL_LIBC, FK_memccpy, d2, s, k, 'Q');
	  chk_long ("memccpy", a, b);
	  chk_win ("memccpy", r1 + WIN_OFF, r2 + WIN_OFF, k, da);
	}
  }
  edge_memccpy ();
  t_end ();
}
