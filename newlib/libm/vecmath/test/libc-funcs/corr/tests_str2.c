/* strcmp, strncmp, strcasecmp, strncasecmp.  */

#include "harness.h"
#include "tests.h"
#include <string.h>

#define MAX_SMALL 200

enum
{
  PV_EQUAL,			/* identical                                   */
  PV_D0,			/* differ at byte 0                            */
  PV_DMID,			/* differ at len/2                             */
  PV_DLAST,			/* differ at len-1                             */
  PV_PREFIX_B,			/* b = a + one char (a is a prefix of b)       */
  PV_PREFIX_A,			/* a = b + one char                            */
  PV_HIGH,			/* 0x80 vs 0x7f: unsigned comparison           */
  PV_CASE,			/* case-only difference                        */
  PV_COUNT
};

static void
make_pair (unsigned char *a, unsigned char *b, size_t len, int variant)
{
  size_t i;

  make_str (a, len, 0);
  memcpy (b, a, len + 1);
  switch (variant)
    {
    case PV_EQUAL:
      break;
    case PV_D0:
      b[0] = (unsigned char) (b[0] ^ 0x55);
      break;
    case PV_DMID:
      i = len / 2;
      b[i] = (unsigned char) (b[i] ^ 0x55);
      break;
    case PV_DLAST:
      i = len - 1;
      b[i] = (unsigned char) (b[i] ^ 0x55);
      break;
    case PV_PREFIX_B:
      b[len] = 'x';
      b[len + 1] = 0;
      break;
    case PV_PREFIX_A:
      a[len] = 'x';
      a[len + 1] = 0;
      break;
    case PV_HIGH:
      a[0] = 0x80;
      b[0] = 0x7f;
      break;
    case PV_CASE:
      for (i = 0; i <= len; i++)
	{
	  if (b[i] >= 'a' && b[i] <= 'z')
	    b[i] = (unsigned char) (b[i] - 32);
	  else if (b[i] >= 'A' && b[i] <= 'Z')
	    b[i] = (unsigned char) (b[i] + 32);
	}
      break;
    }
}

static int
variant_needs_len (int v)
{
  return v == PV_D0 || v == PV_DMID || v == PV_DLAST || v == PV_HIGH;
}

/* Alignment coverage: full 16x16 for the three variants that exercise the
   plain scan/compare loop, a reduced set for the rest (the loop shape is
   identical; the reduced set still covers both sides of every 4/8/16/32
   byte boundary).  */
static int
sub_align (int a)
{
  return a == 0 || a == 1 || a == 3 || a == 7 || a == 8 || a == 9 || a == 15;
}

static void
cmp_one (int fk, int is_n, const unsigned char *a, const unsigned char *b,
	 size_t len, int sa, int sb, int variant)
{
  size_t ns[5];
  int nn = 0, i;
  char w[64];

  t_case ("len=%zu sa=%d sb=%d pair=%d", len, sa, sb, variant);
  chk_long ("cmp", fcall (IMPL_RVV, fk, (void *) a, b, 0, 0),
	    fcall (IMPL_LIBC, fk, (void *) a, b, 0, 0));
  if (!is_n)
    return;

  ns[nn++] = 0;
  ns[nn++] = len / 2;
  ns[nn++] = len;
  ns[nn++] = len + 1;
  ns[nn++] = (size_t) -1;	/* "no limit" */
  for (i = 0; i < nn; i++)
    {
      snprintf (w, sizeof w, "cmp(n=%zu)", ns[i]);
      chk_long (w, fcall (IMPL_RVV, fk, (void *) a, b, ns[i], 0),
		fcall (IMPL_LIBC, fk, (void *) a, b, ns[i], 0));
    }
}

static void
cmp_sweep (int fk, int is_n)
{
  unsigned char *ra = ro_base (), *rb = s1_base ();
  size_t len;
  int sa, sb, v;

  for (len = 0; len <= MAX_SMALL; len++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      for (sb = 0; sb < ALIGN_COUNT; sb++)
	for (v = 0; v < PV_COUNT; v++)
	  {
	    unsigned char *a = win (ra, sa);
	    unsigned char *b = win (rb, sb);
	    int wide = (v == PV_EQUAL || v == PV_D0 || v == PV_PREFIX_B);

	    if (len == 0 && variant_needs_len (v))
	      continue;
	    /* Reduced coverage for the secondary pair variants, and a
	       slightly smaller one again when the function also takes an
	       n (its n sweep already supplies extra alignment coverage).  */
	    if (!wide && !sub_align (sb))
	      continue;
	    if (!wide && is_n && !sub_align (sa))
	      continue;
	    make_pair (a, b, len, v);
	    cmp_one (fk, is_n, a, b, len, sa, sb, v);
	  }

  /* a few large inputs, all source alignments */
  {
    static const size_t big[] = { 997, 4096, 100003 };
    static const int bigv[] = { PV_EQUAL, PV_DMID, PV_PREFIX_B };
    int i, j;
    for (i = 0; i < (int) (sizeof big / sizeof *big); i++)
      for (j = 0; j < (int) (sizeof bigv / sizeof *bigv); j++)
	for (sa = 0; sa < ALIGN_COUNT; sa++)
	  {
	    unsigned char *a = win (ra, sa);
	    unsigned char *b = win (rb, (sa + 5) % 16);
	    make_pair (a, b, big[i], bigv[j]);
	    cmp_one (fk, is_n, a, b, big[i], sa, (sa + 5) % 16, bigv[j]);
	  }
  }
}

void
test_strcmp (void)
{
  t_begin ("strcmp");
  cmp_sweep (FK_strcmp, 0);
  exh_cmp (FK_strcmp, 0);
  edge_strcmp ();
  t_end ();
}

void
test_strncmp (void)
{
  t_begin ("strncmp");
  cmp_sweep (FK_strncmp, 1);
  exh_cmp (FK_strncmp, 1);
  edge_strncmp ();
  t_end ();
}

void
test_strcasecmp (void)
{
  t_begin ("strcasecmp");
  cmp_sweep (FK_strcasecmp, 0);
  exh_cmp (FK_strcasecmp, 0);
  edge_strcasecmp ();
  t_end ();
}

void
test_strncasecmp (void)
{
  t_begin ("strncasecmp");
  cmp_sweep (FK_strncasecmp, 1);
  exh_cmp (FK_strncasecmp, 1);
  edge_strncasecmp ();
  t_end ();
}
