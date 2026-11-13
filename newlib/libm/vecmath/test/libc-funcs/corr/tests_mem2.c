/* memchr, memrchr, rawmemchr, memcmp, memset.  */

#include "harness.h"
#include "tests.h"
#include <string.h>

#define MAX_SMALL 200

static int
sub_align (int a)
{
  return a == 0 || a == 1 || a == 3 || a == 7 || a == 8 || a == 9 || a == 15;
}

/* ------------------------------------------------------------------ */
/* memchr / memrchr / rawmemchr                                       */
/* ------------------------------------------------------------------ */
static void
chr_one (int fk, const unsigned char *s, size_t n, int sa, int c, long pos,
	 int pi)
{
  char label[48];
  t_case ("n=%zu sa=%d c=0x%02x pos=%ld", n, sa, (unsigned) (c & 0xff), pos);
  snprintf (label, sizeof label, "%s(c=0x%02x,pos=%ld)", fk_name (fk),
	    (unsigned) (c & 0xff), pos);
  if (pi < 0 && fk == FK_rawmemchr)
    return;			/* rawmemchr needs an occurrence */
  chk_long (label, fcall (IMPL_RVV, fk, (void *) s, NULL, n, c),
	    fcall (IMPL_LIBC, fk, (void *) s, NULL, n, c));
}

static void
chr_sweep (int fk)
{
  unsigned char *ro = ro_base ();
  static const int cs[] = { 0x00, 'a', 0xff, 0x141, 0x55 };
  size_t n;
  int sa, ci, pi;

  for (n = 0; n <= MAX_SMALL; n++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      {
	unsigned char *s = win (ro, sa);
	for (ci = 0; ci < (int) (sizeof cs / sizeof *cs); ci++)
	  for (pi = 0; pi < 4; pi++)
	    {
	      int c = cs[ci];
	      long pos;
	      size_t i;
	      if (pi > 0 && n == 0)
		continue;
	      if (ci > 0 && pi > 0 && !sub_align (sa))
		continue;
	      switch (pi)
		{
		case 0: pos = -1; break;
		case 1: pos = 0; break;
		case 2: pos = (long) (n / 2); break;
		default: pos = (long) (n - 1); break;
		}
	      make_data (s, n, ci & 1);
	      for (i = 0; i < n; i++)
		if (s[i] == (unsigned char) c)
		  s[i] = (unsigned char) ~c;
	      if (pos >= 0)
		s[pos] = (unsigned char) c;
	      if (pi == 0)
		{
		  /* make sure of "absent" */
		  for (i = 0; i < n; i++)
		    if (s[i] == (unsigned char) c)
		      s[i] = (unsigned char) ~c;
		}
	      chr_one (fk, s, n, sa, c, pos, pi);
	    }
      }

  {
    static const size_t big[] = { 997, 4096, 100003 };
    int i;
    for (i = 0; i < (int) (sizeof big / sizeof *big); i++)
      for (sa = 0; sa < ALIGN_COUNT; sa++)
	{
	  size_t k = big[i];
	  unsigned char *s = win (ro, sa);
	  make_data (s, k, 0);
	  s[k / 3] = 'Q';
	  chr_one (fk, s, k, sa, 'Q', (long) (k / 3), 2);
	  chr_one (fk, s, k, sa, 0x00, -1, 0);
	  chr_one (fk, s, k, sa, 0xff, -1, 0);
	}
  }
}

void
test_memchr (void)
{
  t_begin ("memchr");
  chr_sweep (FK_memchr);
  edge_memchr ();
  t_end ();
}

void
test_memrchr (void)
{
  t_begin ("memrchr");
  chr_sweep (FK_memrchr);
  edge_memrchr ();
  t_end ();
}

void
test_rawmemchr (void)
{
  t_begin ("rawmemchr");
  chr_sweep (FK_rawmemchr);
  edge_rawmemchr ();
  t_end ();
}

/* ------------------------------------------------------------------ */
/* memcmp                                                             */
/* ------------------------------------------------------------------ */
enum { MC_EQUAL, MC_D0, MC_DMID, MC_DLAST, MC_HIGH, MC_COUNT };

static void
cmp_pair (unsigned char *a, unsigned char *b, size_t n, int variant, int pat)
{
  size_t i;
  make_data (a, n, pat);
  memcpy (b, a, n);
  switch (variant)
    {
    case MC_EQUAL: break;
    case MC_D0: b[0] = (unsigned char) (b[0] ^ 0x5a); break;
    case MC_DMID: i = n / 2; b[i] = (unsigned char) (b[i] ^ 0x5a); break;
    case MC_DLAST: i = n - 1; b[i] = (unsigned char) (b[i] ^ 0x5a); break;
    case MC_HIGH: a[0] = 0x80; b[0] = 0x01; break;
    }
}

void
test_memcmp (void)
{
  unsigned char *ra = ro_base (), *rb = s1_base ();
  size_t n;
  int sa, sb, v, pat;

  t_begin ("memcmp");
  for (n = 0; n <= MAX_SMALL; n++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      for (sb = 0; sb < ALIGN_COUNT; sb++)
	for (v = 0; v < MC_COUNT; v++)
	  for (pat = 0; pat < 2; pat++)
	    {
	      unsigned char *a = win (ra, sa), *b = win (rb, sb);
	      int wide = (v == MC_EQUAL || v == MC_D0);
	      if (n == 0 && (v == MC_DMID || v == MC_DLAST || v == MC_HIGH))
		continue;
	      if (!wide && (!sub_align (sa) || !sub_align (sb)))
		continue;
	      if (pat == 1 && !(wide && sub_align (sa)))
		continue;
	      cmp_pair (a, b, n, v, pat);
	      t_case ("n=%zu sa=%d sb=%d diff=%d pat=%d", n, sa, sb, v, pat);
	      chk_long ("memcmp",
			fcall (IMPL_RVV, FK_memcmp, a, b, n, 0),
			fcall (IMPL_LIBC, FK_memcmp, a, b, n, 0));
	    }

  {
    static const size_t big[] = { 997, 4096, 100003 };
    static const int bigv[] = { MC_EQUAL, MC_DMID, MC_DLAST };
    int i, j;
    for (i = 0; i < (int) (sizeof big / sizeof *big); i++)
      for (j = 0; j < (int) (sizeof bigv / sizeof *bigv); j++)
	for (sa = 0; sa < ALIGN_COUNT; sa++)
	  {
	    unsigned char *a = win (ra, sa);
	    unsigned char *b = win (rb, (sa + 3) % 16);
	    size_t k = big[i];
	    if (k > 5000 && sa % 5)
	      continue;
	    cmp_pair (a, b, k, bigv[j], 0);
	    t_case ("n=%zu sa=%d sb=%d diff=%d", k, sa, (sa + 3) % 16, bigv[j]);
	    chk_long ("memcmp", fcall (IMPL_RVV, FK_memcmp, a, b, k, 0),
		      fcall (IMPL_LIBC, FK_memcmp, a, b, k, 0));
	  }
  }
  edge_memcmp ();
  t_end ();
}

/* ------------------------------------------------------------------ */
/* memset                                                             */
/* ------------------------------------------------------------------ */
void
test_memset (void)
{
  unsigned char *r1 = d1_base (), *r2 = d2_base ();
  static const int cs[] = { 0x00, 'a', 0xff, 0x141, -1 };
  size_t n;
  int da, ci;

  t_begin ("memset");
  for (n = 0; n <= MAX_SMALL; n++)
    for (da = 0; da < ALIGN_COUNT; da++)
      for (ci = 0; ci < (int) (sizeof cs / sizeof *cs); ci++)
	{
	  unsigned char *d1 = win (r1, da), *d2 = win (r2, da);
	  long a, b;
	  size_t i;

	  if (ci > 0 && !sub_align (da))
	    continue;
	  canary_fill (r1 + WIN_OFF, WIN_LEN (n) + da, 41);
	  canary_fill (r2 + WIN_OFF, WIN_LEN (n) + da, 41);
	  t_case ("n=%zu da=%d c=0x%02x", n, da, (unsigned) (cs[ci] & 0xff));
	  a = fcall (IMPL_RVV, FK_memset, d1, NULL, n, cs[ci]);
	  b = fcall (IMPL_LIBC, FK_memset, d2, NULL, n, cs[ci]);
	  chk_long ("return", a, b);
	  chk_win ("memset", r1 + WIN_OFF, r2 + WIN_OFF, n, (size_t) da);
	  for (i = 0; i < n; i++)
	    if (d1[i] != (unsigned char) cs[ci])
	      {
		chk_byte ("fill", d1[i], (unsigned char) cs[ci]);
		break;
	      }
	}

  {
    static const size_t big[] = { 997, 4096, 100003 };
    int i;
    for (i = 0; i < (int) (sizeof big / sizeof *big); i++)
      for (da = 0; da < ALIGN_COUNT; da++)
	{
	  size_t k = big[i];
	  unsigned char *d1 = win (r1, da), *d2 = win (r2, da);
	  long a, b;
	  canary_fill (r1 + WIN_OFF, WIN_LEN (k) + da, 41);
	  canary_fill (r2 + WIN_OFF, WIN_LEN (k) + da, 41);
	  t_case ("n=%zu da=%d c=0xa5", k, da);
	  a = fcall (IMPL_RVV, FK_memset, d1, NULL, k, 0xa5);
	  b = fcall (IMPL_LIBC, FK_memset, d2, NULL, k, 0xa5);
	  chk_long ("return", a, b);
	  chk_win ("memset", r1 + WIN_OFF, r2 + WIN_OFF, k, (size_t) da);
	}
  }
  edge_memset ();
  t_end ();
}
