/* strspn, strcspn, strpbrk, strstr.  */

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
/* strspn / strcspn / strpbrk                                         */
/* ------------------------------------------------------------------ */
static const char *const sets[] = {
  "",
  "a",
  "b",
  "ab",
  "z",
  "aZbY0_~!%\x80\xa5\xfe\xff",
  " ",				/* not present in any pattern */
};

static void
set_one (int fk, const unsigned char *s, const char *set, size_t len, int sa,
	 int sb, int si)
{
  char label[48];
  t_case ("len=%zu sa=%d sb=%d set=%d", len, sa, sb, si);
  snprintf (label, sizeof label, "%s(set=%d)", fk_name (fk), si);
  chk_long (label, fcall (IMPL_RVV, fk, (void *) s, set, 0, 0),
	    fcall (IMPL_LIBC, fk, (void *) s, set, 0, 0));
}

static void
span_sweep (int fk)
{
  unsigned char *ro = ro_base (), *rs = s2_base ();
  unsigned char *bigset;
  size_t len, i;
  int sa, sb, si;

  /* A set that contains every non-zero byte.  */
  bigset = rs + 8192;
  for (i = 0; i < 255; i++)
    bigset[i] = (unsigned char) (i + 1);
  bigset[255] = 0;

  for (len = 0; len <= MAX_SMALL; len++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      {
	unsigned char *s = win (ro, sa);
	make_str (s, len, 2);
	for (sb = 0; sb < ALIGN_COUNT; sb++)
	  for (si = 0; si < (int) (sizeof sets / sizeof *sets); si++)
	    {
	      unsigned char *set = win (rs, sb);
	      int wide = si < 3;
	      if (!wide && !sub_align (sb))
		continue;
	      strcpy ((char *) set, sets[si]);
	      set_one (fk, s, (const char *) set, len, sa, sb, si);
	    }
      }

  /* the 255-byte set, a few alignments */
  for (len = 0; len <= MAX_SMALL; len += 7)
    for (sa = 0; sa < ALIGN_COUNT; sa += 5)
      {
	unsigned char *s = win (ro, sa);
	make_str (s, len, 2);
	for (sb = 0; sb < ALIGN_COUNT; sb += 5)
	  {
	    unsigned char *set = win (rs, sb);
	    memcpy (set, bigset, 256);
	    set_one (fk, s, (const char *) set, len, sa, sb, 7);
	  }
      }

  /* big haystacks */
  {
    static const size_t big[] = { 997, 4096, 100003 };
    int bi;
    for (bi = 0; bi < (int) (sizeof big / sizeof *big); bi++)
      for (sa = 0; sa < ALIGN_COUNT; sa += 3)
	{
	  unsigned char *s = win (ro, sa);
	  unsigned char *set = win (rs, 0);
	  make_str (s, big[bi], 2);
	  strcpy ((char *) set, "ab");
	  set_one (fk, s, (const char *) set, big[bi], sa, 0, 3);
	  strcpy ((char *) set, "");
	  set_one (fk, s, (const char *) set, big[bi], sa, 0, 0);
	}
  }
}

void
test_strspn (void)
{
  t_begin ("strspn");
  span_sweep (FK_strspn);
  exh_span (FK_strspn);
  edge_strspn ();
  t_end ();
}

void
test_strcspn (void)
{
  t_begin ("strcspn");
  span_sweep (FK_strcspn);
  exh_span (FK_strcspn);
  edge_strcspn ();
  t_end ();
}

void
test_strpbrk (void)
{
  t_begin ("strpbrk");
  span_sweep (FK_strpbrk);
  exh_span (FK_strpbrk);
  edge_strpbrk ();
  t_end ();
}

/* ------------------------------------------------------------------ */
/* strstr                                                             */
/* ------------------------------------------------------------------ */
static void
strstr_one (const unsigned char *hay, const unsigned char *needle, size_t len,
	    int sa, int sb, int kind)
{
  char label[48];
  t_case ("len=%zu sa=%d sb=%d needle=%d", len, sa, sb, kind);
  snprintf (label, sizeof label, "strstr(needle=%d)", kind);
  chk_long (label, fcall (IMPL_RVV, FK_strstr, (void *) hay, needle, 0, 0),
	    fcall (IMPL_LIBC, FK_strstr, (void *) hay, needle, 0, 0));
}

static void
strstr_sweep (void)
{
  unsigned char *ro = ro_base (), *rn = s2_base ();
  size_t len;
  int sa, sb, k;

  for (len = 0; len <= MAX_SMALL; len++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      {
	unsigned char *hay = win (ro, sa);
	make_str (hay, len, 0);
	for (sb = 0; sb < ALIGN_COUNT; sb++)
	  {
	    unsigned char *nd = win (rn, sb);
	    if (!sub_align (sb) && !sub_align (sa))
	      continue;
	    for (k = 0; k < 8; k++)
	      {
		switch (k)
		  {
		  case 0:	/* empty needle -> haystack */
		    nd[0] = 0;
		    break;
		  case 1:	/* absent needle */
		    strcpy ((char *) nd, "zzz");
		    break;
		  case 2:	/* whole haystack */
		    memcpy (nd, hay, len);
		    nd[len] = 0;
		    break;
		  case 3:	/* first three bytes */
		    if (len >= 3)
		      {
			memcpy (nd, hay, 3);
			nd[3] = 0;
		      }
		    break;
		  case 4:	/* last three bytes */
		    if (len >= 3)
		      {
			memcpy (nd, hay + len - 3, 3);
			nd[3] = 0;
		      }
		    break;
		  case 5:	/* middle three bytes */
		    if (len >= 3)
		      {
			memcpy (nd, hay + len / 2, 3);
			nd[3] = 0;
		      }
		    break;
		  case 6:	/* haystack plus one: cannot match */
		    memcpy (nd, hay, len);
		    nd[len] = 'x';
		    nd[len + 1] = 0;
		    break;
		  case 7:	/* single first byte */
		    if (len)
		      {
			nd[0] = hay[0];
			nd[1] = 0;
		      }
		    break;
		  }
		if (k == 3 || k == 4 || k == 5 || k == 7)
		  {
		    if (len < 3)
		      continue;
		    if (k == 7 && len < 1)
		      continue;
		  }
		if (k == 0)
		  strcpy ((char *) nd, "");
		strstr_one (hay, nd, len, sa, sb, k);
	      }
	  }
      }

  /* big haystacks */
  {
    static const size_t big[] = { 997, 4096, 100003 };
    int bi;
    for (bi = 0; bi < (int) (sizeof big / sizeof *big); bi++)
      for (sa = 0; sa < ALIGN_COUNT; sa += 3)
	{
	  size_t n = big[bi];
	  unsigned char *hay = win (ro, sa);
	  unsigned char *nd = win (rn, 0);
	  make_str (hay, n, 0);
	  strcpy ((char *) nd, "zzz");
	  strstr_one (hay, nd, n, sa, 0, 1);
	  memcpy (nd, hay + n - 3, 3);
	  nd[3] = 0;
	  strstr_one (hay, nd, n, sa, 0, 4);
	  memcpy (nd, hay, n);
	  nd[n] = 'x';
	  nd[n + 1] = 0;
	  strstr_one (hay, nd, n, sa, 0, 6);
	}
  }
}

/* Deliberately pathological haystacks/needles: every position is a
   candidate for the naive verify loop.  */
static void
strstr_candidates (void)
{
  unsigned char *ro = ro_base (), *rn = s2_base ();
  int sa, sb;
  size_t i;

  for (sa = 0; sa < ALIGN_COUNT; sa++)
    for (sb = 0; sb < ALIGN_COUNT; sb++)
      {
	unsigned char *hay = win (ro, sa);
	unsigned char *nd = win (rn, sb);

	/* 200 x 'a', needle "aaab": absent, every position a candidate */
	memset (hay, 'a', 200);
	hay[200] = 0;
	strcpy ((char *) nd, "aaab");
	strstr_one (hay, nd, 200, sa, sb, 100);

	/* 200 x 'a' + "b": "aab" found at 198 */
	hay[200] = 'b';
	hay[201] = 0;
	strcpy ((char *) nd, "aab");
	strstr_one (hay, nd, 200, sa, sb, 101);

	/* "ababab...": needle "abab" found at 0 */
	for (i = 0; i < 200; i++)
	  hay[i] = (i & 1) ? 'b' : 'a';
	hay[200] = 0;
	strcpy ((char *) nd, "abab");
	strstr_one (hay, nd, 200, sa, sb, 102);
	strcpy ((char *) nd, "bab");
	strstr_one (hay, nd, 200, sa, sb, 103);
	strcpy ((char *) nd, "abaab");
	strstr_one (hay, nd, 200, sa, sb, 104);

	/* needle == haystack */
	make_str (hay, 64, 0);
	memcpy (nd, hay, 64);
	nd[64] = 0;
	strstr_one (hay, nd, 64, sa, sb, 105);
      }
}

void
test_strstr (void)
{
  t_begin ("strstr");
  strstr_sweep ();
  strstr_candidates ();
  exh_strstr ();
  edge_strstr ();
  t_end ();
}
