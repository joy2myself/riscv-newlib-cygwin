/* Exhaustive small-alphabet tests.

   Random-looking patterns can hide logic errors (a wrong comparison, an
   advance-by-one off-by-one).  These tests enumerate *every* string up to a
   few bytes over a small alphabet (including high bytes and NUL-adjacent
   case pairs) and compare the two implementations for every combination.
   They are cheap and catch the classic "works for my pattern" bugs.  */

#include "harness.h"
#include "tests.h"
#include <string.h>

#define MAXLEN 5

/* Enumerate all strings of length 0..maxlen over the given alphabet,
   calling fn(ctx, s, len, idx) for each.  */
typedef void (*strfn) (void *ctx, const unsigned char *s, size_t len, int idx);

static void
enum_strings (const unsigned char *alpha, int k, int maxlen, strfn fn,
	      void *ctx)
{
  unsigned char buf[MAXLEN + 1];
  int len, i, idx = 0;

  for (len = 0; len <= maxlen; len++)
    {
      int ncomb = 1;
      for (i = 0; i < len; i++)
	ncomb *= k;
      for (i = 0; i < ncomb; i++)
	{
	  int v = i, j;
	  for (j = len - 1; j >= 0; j--)
	    {
	      buf[j] = alpha[v % k];
	      v /= k;
	    }
	  buf[len] = 0;
	  fn (ctx, buf, (size_t) len, idx++);
	}
    }
}

/* ------------------------------------------------------------------ */
/* strstr / strspn / strcspn / strpbrk                                */
/* ------------------------------------------------------------------ */
struct pairctx
{
  int fk;
  unsigned char *hay;
  unsigned char *nd;
  const unsigned char *needle;
  size_t nlen;
};

static void
strstr_cb (void *vctx, const unsigned char *s, size_t len, int idx)
{
  struct pairctx *c = vctx;
  char w[64];
  memcpy (c->hay, s, len + 1);
  snprintf (w, sizeof w, "%s(hay=\"%s\",nd=%zu)", fk_name (c->fk),
	    (const char *) c->hay, c->nlen);
  t_case ("hay=%s len=%zu ndlen=%zu", (const char *) c->hay, len, c->nlen);
  chk_long (w, fcall (IMPL_RVV, c->fk, c->hay, c->needle, 0, 0),
	    fcall (IMPL_LIBC, c->fk, c->hay, c->needle, 0, 0));
  (void) idx;
}

static void
strstr_needle_cb (void *vctx, const unsigned char *s, size_t len, int idx)
{
  struct pairctx *c = vctx;
  memcpy (c->nd, s, len + 1);
  c->needle = c->nd;
  c->nlen = len;
  enum_strings ((const unsigned char *) "ab", 2, MAXLEN, strstr_cb, c);
  (void) idx;
}

void
exh_strstr (void)
{
  struct pairctx c;
  unsigned char *a = s1_base (), *b = s2_base ();
  int ai;

  memset (&c, 0, sizeof c);
  c.fk = FK_strstr;
  for (ai = 0; ai < 2; ai++)
    {
      c.hay = win (a, (size_t) (ai * 7));
      c.nd = win (b, (size_t) (ai * 3 + 1));
      enum_strings ((const unsigned char *) "ab", 2, MAXLEN,
		    strstr_needle_cb, &c);
    }
}

struct spctx
{
  int fk;
  unsigned char *hay;
  const char *set;
};

static void
span_cb (void *vctx, const unsigned char *s, size_t len, int idx)
{
  struct spctx *c = vctx;
  char w[64];
  memcpy (c->hay, s, len + 1);
  snprintf (w, sizeof w, "%s(set=\"%s\")", fk_name (c->fk), c->set);
  t_case ("hay=\"%s\" set=\"%s\"", (const char *) c->hay, c->set);
  chk_long (w, fcall (IMPL_RVV, c->fk, c->hay, c->set, 0, 0),
	    fcall (IMPL_LIBC, c->fk, c->hay, c->set, 0, 0));
  (void) idx;
}

void
exh_span (int fk)
{
  static const char *const sets[] =
    { "", "a", "b", "c", "ab", "bc", "ac", "abc" };
  struct spctx c;
  unsigned char *a = s1_base ();
  size_t si;
  int ai;

  c.fk = fk;
  for (ai = 0; ai < 2; ai++)
    {
      c.hay = win (a, (size_t) ai);
      for (si = 0; si < sizeof sets / sizeof *sets; si++)
	{
	  c.set = sets[si];
	  enum_strings ((const unsigned char *) "abc", 3, 4, span_cb, &c);
	}
    }
}

/* ------------------------------------------------------------------ */
/* strcmp / strncmp / strcasecmp / strncasecmp                        */
/* ------------------------------------------------------------------ */
struct cmpctx
{
  int fk, is_n;
  unsigned char *sa;
  unsigned char *sb;
  const unsigned char *b;
  size_t blen;
  const char *bs;
};

static void
cmp_cb (void *vctx, const unsigned char *s, size_t len, int idx)
{
  struct cmpctx *c = vctx;
  static const size_t extra[] = { 0, 1, 2, 3, 4, (size_t) -1 };
  char w[64];
  size_t k;

  memcpy (c->sa, s, len + 1);
  t_case ("a=\"%s\" b=\"%s\"", (const char *) c->sa, c->bs);
  chk_long ("cmp", fcall (IMPL_RVV, c->fk, c->sa, c->b, 0, 0),
	    fcall (IMPL_LIBC, c->fk, c->sa, c->b, 0, 0));
  if (c->is_n)
    for (k = 0; k < sizeof extra / sizeof *extra; k++)
      {
	snprintf (w, sizeof w, "cmp(n=%zu)", extra[k]);
	chk_long (w, fcall (IMPL_RVV, c->fk, c->sa, c->b, extra[k], 0),
		  fcall (IMPL_LIBC, c->fk, c->sa, c->b, extra[k], 0));
      }
  (void) idx;
}

static void
cmp_b_cb (void *vctx, const unsigned char *s, size_t len, int idx)
{
  struct cmpctx *c = vctx;
  memcpy (c->sb, s, len + 1);
  c->b = c->sb;
  c->blen = len;
  c->bs = (const char *) c->sb;
  enum_strings ((const unsigned char *) "aA\x80", 3, 3, cmp_cb, c);
  (void) idx;
}

void
exh_cmp (int fk, int is_n)
{
  struct cmpctx c;
  unsigned char *ra = ro_base (), *rb = s1_base ();
  int ai;

  memset (&c, 0, sizeof c);
  c.fk = fk;
  c.is_n = is_n;
  for (ai = 0; ai < 2; ai++)
    {
      c.sa = win (ra, (size_t) ai);
      c.sb = win (rb, (size_t) (ai * 3 + 1));
      enum_strings ((const unsigned char *) "aA\x80", 3, 3, cmp_b_cb, &c);
    }
}

/* ------------------------------------------------------------------ */
/* strchr / strchrnul / strrchr                                       */
/* ------------------------------------------------------------------ */
struct chrctx
{
  int fk;
  unsigned char *s;
  int c;
};

static void
chr_cb (void *vctx, const unsigned char *s, size_t len, int idx)
{
  struct chrctx *c = vctx;
  char w[64];
  memcpy (c->s, s, len + 1);
  snprintf (w, sizeof w, "%s(c=0x%02x)", fk_name (c->fk),
	    (unsigned) (c->c & 0xff));
  t_case ("s=\"%s\" c=0x%02x", (const char *) c->s,
	  (unsigned) (c->c & 0xff));
  chk_long (w, fcall (IMPL_RVV, c->fk, c->s, NULL, 0, c->c),
	    fcall (IMPL_LIBC, c->fk, c->s, NULL, 0, c->c));
  (void) idx;
}

void
exh_chr (int fk)
{
  static const int chars[] = { 0, 'a', 'b', 'c', 0x80, 0xff, 0x161 };
  struct chrctx c;
  unsigned char *a = s1_base ();
  size_t i;
  int ai;

  c.fk = fk;
  for (ai = 0; ai < 2; ai++)
    {
      c.s = win (a, (size_t) ai);
      for (i = 0; i < sizeof chars / sizeof *chars; i++)
	{
	  c.c = chars[i];
	  enum_strings ((const unsigned char *) "ab", 2, MAXLEN, chr_cb, &c);
	}
    }
}
