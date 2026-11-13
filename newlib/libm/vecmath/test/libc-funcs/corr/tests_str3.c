/* strcpy, strncpy, strcat, strncat.  */

#include "harness.h"
#include "tests.h"
#include <string.h>

#define MAX_SMALL 200

static int
sub_align (int a)
{
  return a == 0 || a == 1 || a == 3 || a == 7 || a == 8 || a == 9 || a == 15;
}

/* Prepare both destination windows with identical canary content.  */
static void
prep_dest (unsigned char *region, size_t align, size_t touch, unsigned seed)
{
  canary_fill (region + WIN_OFF, WIN_LEN (touch) + align, seed);
}

/* Run one destination-writing call on both implementations and compare the
   return value plus the whole destination window.  */
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
void
test_strcpy (void)
{
  unsigned char *ro = ro_base (), *r1 = d1_base (), *r2 = d2_base ();
  size_t len;
  int sa, da;

  t_begin ("strcpy");
  for (len = 0; len <= MAX_SMALL; len++)
    {
      unsigned char *s = win (ro, 0);
      make_str (s, len, 0);
      for (sa = 0; sa < ALIGN_COUNT; sa++)
	{
	  s = win (ro, sa);
	  make_str (s, len, 0);
	  for (da = 0; da < ALIGN_COUNT; da++)
	    {
	      unsigned char *d1 = win (r1, da), *d2 = win (r2, da);
	      prep_dest (r1, da, len + 1, 7);
	      prep_dest (r2, da, len + 1, 7);
	      t_case ("len=%zu sa=%d da=%d", len, sa, da);
	      run_dest (FK_strcpy, "strcpy", r1, r2, d1, d2, s, 0, da,
			len + 1);
	    }
	}
    }
  /* big */
  {
    static const size_t big[] = { 997, 4096, 100003 };
    int i;
    for (i = 0; i < (int) (sizeof big / sizeof *big); i++)
      for (sa = 0; sa < ALIGN_COUNT; sa++)
	{
	  size_t n = big[i];
	  unsigned char *s = win (ro, sa);
	  unsigned char *d1 = win (r1, (sa * 3) % 16);
	  unsigned char *d2 = win (r2, (sa * 3) % 16);
	  make_str (s, n, 0);
	  prep_dest (r1, (sa * 3) % 16, n + 1, 7);
	  prep_dest (r2, (sa * 3) % 16, n + 1, 7);
	  t_case ("len=%zu sa=%d da=%d", n, sa, (sa * 3) % 16);
	  run_dest (FK_strcpy, "strcpy", r1, r2, d1, d2, s, 0,
		    (sa * 3) % 16, n + 1);
	}
  }
  edge_strcpy ();
  t_end ();
}

/* ------------------------------------------------------------------ */
void
test_strncpy (void)
{
  unsigned char *ro = ro_base (), *r1 = d1_base (), *r2 = d2_base ();
  size_t len;
  int sa, da, k;

  t_begin ("strncpy");
  for (len = 0; len <= MAX_SMALL; len++)
    {
      size_t ns[6];
      int nn = 0;
      ns[nn++] = 0;
      ns[nn++] = len ? len - 1 : 0;
      ns[nn++] = len;
      ns[nn++] = len + 1;
      ns[nn++] = len + 8;
      ns[nn++] = 2 * len + 3;

      for (sa = 0; sa < ALIGN_COUNT; sa++)
	{
	  unsigned char *s = win (ro, sa);
	  make_str (s, len, 0);
	  for (da = 0; da < ALIGN_COUNT; da++)
	    for (k = 0; k < nn; k++)
	      {
		unsigned char *d1, *d2;
		size_t n = ns[k], touch;
		char label[32];
		if (k != 2 && !(sub_align (sa) && sub_align (da)))
		  continue;
		d1 = win (r1, da);
		d2 = win (r2, da);
		touch = n > len + 1 ? n : len + 1;
		prep_dest (r1, da, touch, 11);
		prep_dest (r2, da, touch, 11);
		snprintf (label, sizeof label, "strncpy(n=%zu)", n);
		t_case ("len=%zu sa=%d da=%d n=%zu", len, sa, da, n);
		run_dest (FK_strncpy, label, r1, r2, d1, d2, s, n, da, touch);
	      }
	}
    }
  /* big */
  {
    static const size_t big[] = { 997, 4096, 100003 };
    int i;
    for (i = 0; i < (int) (sizeof big / sizeof *big); i++)
      for (sa = 0; sa < ALIGN_COUNT; sa++)
	{
	  size_t n = big[i];
	  size_t ns[3];
	  int j;
	  unsigned char *s = win (ro, sa);
	  make_str (s, n, 0);
	  ns[0] = n; ns[1] = n / 2; ns[2] = n + 8;
	  for (j = 0; j < 3; j++)
	    {
	      size_t da = (size_t) ((sa * 3) % 16);
	      unsigned char *d1 = win (r1, da), *d2 = win (r2, da);
	      char label[32];
	      prep_dest (r1, da, ns[j], 11);
	      prep_dest (r2, da, ns[j], 11);
	      snprintf (label, sizeof label, "strncpy(n=%zu)", ns[j]);
	      t_case ("len=%zu sa=%d da=%zu n=%zu", n, sa, da, ns[j]);
	      run_dest (FK_strncpy, label, r1, r2, d1, d2, s, ns[j], da,
			ns[j]);
	    }
	}
  }
  edge_strncpy ();
  t_end ();
}

/* ------------------------------------------------------------------ */
static void
cat_sweep (int fk, int is_n, int full_prefixes)
{
  unsigned char *ro = ro_base (), *r1 = d1_base (), *r2 = d2_base ();
  static const char *prefixes[] = { "", "x", "pre", "0123456789abcdef" };
  size_t len;
  int sa, da, p, k;

  for (len = 0; len <= MAX_SMALL; len++)
    for (sa = 0; sa < ALIGN_COUNT; sa++)
      {
	unsigned char *s = win (ro, sa);
	make_str (s, len, 0);
	for (da = 0; da < ALIGN_COUNT; da++)
	  for (p = 0; p < 4; p++)
	    {
	      size_t plen = strlen (prefixes[p]);
	      size_t ns[6];
	      int nn = 0;
	      if (p >= 2 && !(sub_align (sa) && sub_align (da)))
		continue;
	      if (p == 3 && !full_prefixes)
		continue;
	      ns[nn++] = 0;
	      if (is_n)
		{
		  ns[nn++] = 1;
		  ns[nn++] = len ? len - 1 : 0;
		  ns[nn++] = len;
		  ns[nn++] = len + 1;
		  ns[nn++] = len + 8;
		  ns[nn++] = (size_t) -1;	/* "no limit" */
		}
	      for (k = 0; k < nn; k++)
		{
		  size_t n = is_n ? ns[k] : 0, touch;
		  unsigned char *d1 = win (r1, da), *d2 = win (r2, da);
		  char label[32];
		  size_t used;
		  if (is_n && k != 3 && !(sub_align (sa) && sub_align (da)))
		    continue;
		  used = (is_n && n < len) ? n : len;
		  touch = plen + used + 1;
		  prep_dest (r1, da, touch, 13);
		  prep_dest (r2, da, touch, 13);
		  /* Always write a valid C string, even for the empty
		     prefix: the routines take strlen(dst).  */
		  memcpy (d1, prefixes[p], plen + 1);
		  memcpy (d2, prefixes[p], plen + 1);
		  if (is_n)
		    snprintf (label, sizeof label, "%s(n=%zu)", fk_name (fk), n);
		  else
		    snprintf (label, sizeof label, "%s", fk_name (fk));
		  t_case ("len=%zu sa=%d da=%d pfx=\"%s\"%s", len, sa, da,
			  prefixes[p], is_n ? "" : "");
		  run_dest (fk, label, r1, r2, d1, d2, s, n, da, touch);
		}
	    }
      }
}

void
test_strcat (void)
{
  t_begin ("strcat");
  cat_sweep (FK_strcat, 0, 1);
  edge_strcat ();
  t_end ();
}

void
test_strncat (void)
{
  t_begin ("strncat");
  cat_sweep (FK_strncat, 1, 1);
  edge_strncat ();
  t_end ();
}
