/* strlen, strnlen, strchr, strchrnul, strrchr.  */

#include "harness.h"
#include "tests.h"
#include <string.h>

#define MAX_SMALL 200

/* 0 mixed, 1 all-'a', 2 "abab..", 4 all of 0x80..0xff */
static const int scan_variants[] = { 0, 1, 2, 4 };
static const int scan_chars[] = { 0, 'a', 'b', 'z', 0x80, 0xff };

/* ------------------------------------------------------------------ */
/* strlen / strnlen                                                   */
/* ------------------------------------------------------------------ */
static void
strlen_one (const unsigned char *s, size_t len, int sa)
{
  t_case ("len=%zu sa=%d", len, sa);
  chk_size ("strlen", rvv_strlen ((const char *) s),
	    strlen ((const char *) s));
}

static void
strnlen_one (const unsigned char *s, size_t len, int sa)
{
  char w[48];
  size_t lims[7];
  int i;

  t_case ("len=%zu sa=%d", len, sa);
  lims[0] = 0;
  lims[1] = 1;
  lims[2] = len ? len - 1 : 0;
  lims[3] = len;
  lims[4] = len + 1;
  lims[5] = len + 8;
  lims[6] = (size_t) -1;	/* SIZE_MAX: "no limit" */
  for (i = 0; i < 7; i++)
    {
      snprintf (w, sizeof w, "strnlen(n=%zu)", lims[i]);
      chk_size (w, rvv_strnlen ((const char *) s, lims[i]),
		strnlen ((const char *) s, lims[i]));
    }
}

static void
len_sweep (void (*one) (const unsigned char *, size_t, int))
{
  unsigned char *ro = ro_base ();
  size_t len;
  int sa, i;

  for (len = 0; len <= MAX_SMALL; len++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      {
	unsigned char *s = win (ro, sa);
	make_str (s, len, 0);
	one (s, len, sa);
      }

  for (i = 0; i < n_big_lens; i++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      {
	size_t n = big_lens[i];
	unsigned char *s = win (ro, sa);
	make_str (s, n, (i & 1) ? 1 : 0);
	one (s, n, sa);
      }
}

void
test_strlen (void)
{
  t_begin ("strlen");
  len_sweep (strlen_one);
  edge_strlen ();
  t_end ();
}

void
test_strnlen (void)
{
  t_begin ("strnlen");
  len_sweep (strnlen_one);
  edge_strnlen ();
  t_end ();
}

/* ------------------------------------------------------------------ */
/* strchr / strchrnul / strrchr                                       */
/* ------------------------------------------------------------------ */
static void
scan_one (int fk, const unsigned char *s, size_t len, int sa, int cv,
	  int variant)
{
  char w[64];
  long a, b;

  t_case ("len=%zu sa=%d pat=%d c=0x%02x", len, sa, variant,
	  (unsigned) (cv & 0xff));
  snprintf (w, sizeof w, "%s(c=0x%02x)", fk_name (fk), (unsigned) (cv & 0xff));
  a = fcall (IMPL_RVV, fk, (void *) s, NULL, 0, cv);
  b = fcall (IMPL_LIBC, fk, (void *) s, NULL, 0, cv);
  chk_long (w, a, b);
}

static void
scan_sweep (int fk, size_t maxl)
{
  unsigned char *ro = ro_base ();
  size_t len;
  int sa, v, ci;

  for (len = 0; len <= maxl; len++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      for (v = 0; v < (int) (sizeof scan_variants / sizeof *scan_variants); v++)
	{
	  unsigned char *s = win (ro, sa);
	  make_str (s, len, scan_variants[v]);
	  for (ci = 0; ci < (int) (sizeof scan_chars / sizeof *scan_chars);
	       ci++)
	    scan_one (fk, s, len, sa, scan_chars[ci], scan_variants[v]);
	}
}

static void
scan_big (int fk)
{
  unsigned char *ro = ro_base ();
  static const int cs[] = { 0, 'a', 'z', 0xff };
  int i, sa, ci;

  for (i = 0; i < n_big_lens; i++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      {
	unsigned char *s = win (ro, sa);
	size_t len = big_lens[i];
	make_str (s, len, (i & 1) ? 1 : 0);
	for (ci = 0; ci < (int) (sizeof cs / sizeof *cs); ci++)
	  scan_one (fk, s, len, sa, cs[ci], (i & 1) ? 1 : 0);
      }
}

void
test_strchr (void)
{
  t_begin ("strchr");
  scan_sweep (FK_strchr, MAX_SMALL);
  scan_big (FK_strchr);
  exh_chr (FK_strchr);
  edge_strchr ();
  t_end ();
}

void
test_strchrnul (void)
{
  t_begin ("strchrnul");
  scan_sweep (FK_strchrnul, MAX_SMALL);
  scan_big (FK_strchrnul);
  exh_chr (FK_strchrnul);
  edge_strchrnul ();
  t_end ();
}

void
test_strrchr (void)
{
  t_begin ("strrchr");
  scan_sweep (FK_strrchr, MAX_SMALL);
  scan_big (FK_strrchr);
  exh_chr (FK_strrchr);
  edge_strrchr ();
  t_end ();
}
