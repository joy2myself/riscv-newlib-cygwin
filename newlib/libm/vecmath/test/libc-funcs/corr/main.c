/* Driver for the RVV string/memory correctness harness.  */

#include "harness.h"
#include "tests.h"

#include <stdio.h>
#include <string.h>

const size_t big_lens[] = { 255, 256, 257, 511, 512, 513, 997, 4096, 100003 };
const int n_big_lens = (int) (sizeof big_lens / sizeof *big_lens);

/* ------------------------------------------------------------------ */
/* Optional cross-check of the scalar (non-vector) newlib fallbacks.  */
/* Only available for the sources that build with a glibc cross       */
/* compiler; the rest include newlib-private headers.                 */
/* ------------------------------------------------------------------ */
static int
sgn (long v)
{
  return (v > 0) - (v < 0);
}

static long
poff (const void *p, const void *base)
{
  return p ? (long) ((const char *) p - (const char *) base) : -1;
}

static void
scalar_selfcheck (void)
{
  unsigned char *ro = ro_base (), *r1 = d1_base (), *r2 = d2_base ();
  size_t len;
  int sa, avail = 0;

  t_begin ("scalar(bonus)");
#define HAVE(name) (scalar_##name != NULL)
  if (HAVE (strlen)) avail++;
  if (HAVE (strcmp)) avail++;
  if (HAVE (strcpy)) avail++;
  if (HAVE (strspn)) avail++;
  if (HAVE (strcspn)) avail++;
  if (HAVE (strpbrk)) avail++;
  if (HAVE (strrchr)) avail++;
  if (HAVE (strchrnul)) avail++;
  if (HAVE (strcasecmp)) avail++;
  if (HAVE (strncasecmp)) avail++;
  if (HAVE (memchr)) avail++;
  if (HAVE (memrchr)) avail++;

  for (len = 0; len <= 96; len++)
    for (sa = 0; sa < 16; sa += 5)
      {
	unsigned char *s = win (ro, sa);
	unsigned char *t = win (ro, (sa + 7) % 16);
	unsigned char *d1 = win (r1, sa), *d2 = win (r2, sa);
	size_t p = 3, n = len / 2;

	make_str (s, len, 0);
	make_str (t, len, 1);
	t_case ("len=%zu sa=%d", len, sa);

	if (HAVE (strlen))
	  chk_size ("strlen", scalar_strlen ((char *) s),
		    strlen ((char *) s));
	if (HAVE (strcmp))
	  /* The C standard only requires the sign; the newlib scalar
	     strcmp returns a byte difference scaled by 256 when the first
	     differing byte is the second byte of a 16-bit pair.  Compare
	     the sign here and note the magnitude difference once.  */
	  {
	    long a = scalar_strcmp ((char *) s, (char *) t);
	    long b = strcmp ((char *) s, (char *) t);
	    if (sgn (a) != sgn (b))
	      chk_long ("strcmp(sign)", sgn (a), sgn (b));
	    else if (a != b && len == 1)
	      t_warn ("scalar strcmp returns %ld where glibc returns %ld "
		      "(same sign; legal but not bit-identical)", a, b);
	    else
	      chk_long ("strcmp(sign)", sgn (a), sgn (b));
	  }
	if (HAVE (strcasecmp))
	  chk_long ("strcasecmp", scalar_strcasecmp ((char *) s, (char *) t),
		    strcasecmp ((char *) s, (char *) t));
	if (HAVE (strncasecmp))
	  {
	    chk_long ("strncasecmp(n)",
		      scalar_strncasecmp ((char *) s, (char *) t, n),
		      strncasecmp ((char *) s, (char *) t, n));
	    chk_long ("strncasecmp(n+5)",
		      scalar_strncasecmp ((char *) s, (char *) t, n + 5),
		      strncasecmp ((char *) s, (char *) t, n + 5));
	  }
	if (HAVE (strrchr))
	  chk_long ("strrchr('a')", poff (scalar_strrchr ((char *) s, 'a'), s),
		    poff (strrchr ((char *) s, 'a'), s));
	if (HAVE (strchrnul))
	  chk_long ("strchrnul('a')",
		    poff (scalar_strchrnul ((char *) s, 'a'), s),
		    poff (strchrnul ((char *) s, 'a'), s));
	if (HAVE (strspn))
	  chk_long ("strspn", (long) scalar_strspn ((char *) s, "ab"),
		    (long) strspn ((char *) s, "ab"));
	if (HAVE (strcspn))
	  chk_long ("strcspn", (long) scalar_strcspn ((char *) s, "ab"),
		    (long) strcspn ((char *) s, "ab"));
	if (HAVE (strpbrk))
	  chk_long ("strpbrk", poff (scalar_strpbrk ((char *) s, "ab"), s),
		    poff (strpbrk ((char *) s, "ab"), s));
	if (HAVE (strcpy))
	  {
	    canary_fill (d1 - WIN_PAD, WIN_LEN (len), 3);
	    canary_fill (d2 - WIN_PAD, WIN_LEN (len), 3);
	    scalar_strcpy ((char *) d1, (char *) s);
	    strcpy ((char *) d2, (char *) s);
	    chk_buf ("strcpy", d1, d2, len + 2);
	  }
	if (HAVE (memchr))
	  chk_long ("memchr('a')", poff (scalar_memchr (s, 'a', len), s),
		    poff (memchr (s, 'a', len), s));
	if (HAVE (memrchr))
	  chk_long ("memrchr('a')", poff (scalar_memrchr (s, 'a', len), s),
		    poff (memrchr (s, 'a', len), s));
	(void) p;
      }

  if (avail == 0)
    t_warn ("no scalar fallback could be built with the glibc toolchain");
  t_end ();
#undef HAVE
}

/* ------------------------------------------------------------------ */
static const struct
{
  const char *name;
  void (*fn) (void);
} tests[] = {
  { "strlen", test_strlen },
  { "strnlen", test_strnlen },
  { "strchr", test_strchr },
  { "strchrnul", test_strchrnul },
  { "strrchr", test_strrchr },
  { "strcmp", test_strcmp },
  { "strncmp", test_strncmp },
  { "strcasecmp", test_strcasecmp },
  { "strncasecmp", test_strncasecmp },
  { "strcpy", test_strcpy },
  { "strncpy", test_strncpy },
  { "strcat", test_strcat },
  { "strncat", test_strncat },
  { "strspn", test_strspn },
  { "strcspn", test_strcspn },
  { "strpbrk", test_strpbrk },
  { "strstr", test_strstr },
  { "memcpy", test_memcpy },
  { "mempcpy", test_mempcpy },
  { "memmove", test_memmove },
  { "memccpy", test_memccpy },
  { "memchr", test_memchr },
  { "memrchr", test_memrchr },
  { "rawmemchr", test_rawmemchr },
  { "memcmp", test_memcmp },
  { "memset", test_memset },
};

int
main (int argc, char **argv)
{
  size_t i;
  int arg, ran = 0;

  t_banner ();
  if (argc > 1 && (!strcmp (argv[1], "-h") || !strcmp (argv[1], "--help")))
    {
      printf ("usage: %s [function ...]\n", argv[0]);
      printf ("functions:");
      for (i = 0; i < sizeof tests / sizeof *tests; i++)
	printf (" %s", tests[i].name);
      printf ("\n");
      return 0;
    }

  for (i = 0; i < sizeof tests / sizeof *tests; i++)
    {
      int want = (argc == 1);
      for (arg = 1; arg < argc; arg++)
	if (!strcmp (argv[arg], tests[i].name))
	  want = 1;
      if (!want)
	continue;
      ran++;
      tests[i].fn ();
    }

  if (argc == 1)
    scalar_selfcheck ();

  printf ("\n## VLEN %lu bits: %ld checks, %ld failures -- %s\n",
	  t_vlen_bits (), t_total_checks (), t_total_fails (),
	  t_total_fails () ? "FAIL" : "PASS");
  return t_exit_code ();
}
