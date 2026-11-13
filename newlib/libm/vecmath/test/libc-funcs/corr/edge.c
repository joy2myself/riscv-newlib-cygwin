/* Page-boundary tests.

   Every case places the data so that the last byte the routine is allowed
   to touch is the last byte of a mapped page, immediately followed by a
   PROT_NONE page.  A routine that reads (or writes) one byte too far --
   for example a vector load that is not fault-only-first, or a byte loop
   that dereferences one element past the NUL -- takes SIGSEGV; the handler
   longjmps back and the case is reported as a failure.  The mirror image
   (guard page *before* the buffer) catches backward over-reads.

   These are also the tests for the RVV boundary conditions: the NUL (or the
   end of the range) is at every offset 0..15 relative to a page boundary,
   which is at least as strong as testing every offset relative to a
   vector-length boundary for VLEN 128/256/512.

   All buffers are prepared identically on both sides, so the complete
   destination page can be compared afterwards (this catches over-writes
   that stay inside the page as well).  */

#include "harness.h"
#include "tests.h"
#include <string.h>

static size_t PG;

/* ------------------------------------------------------------------ */
static void
edge_pair (int fk, const char *desc, void *p1r, const void *p2r, size_t nr,
	   int cr, void *p1g, const void *p2g, size_t ng, int cg)
{
  int f1 = 0, f2 = 0;
  long a, b;

  a = guard_call (IMPL_RVV, fk, p1r, p2r, nr, cr, &f1);
  if (f1)
    {
      t_fail ("%s: rvv faulted on the guard page (over-read or over-write)",
	      desc);
      return;
    }
  b = guard_call (IMPL_LIBC, fk, p1g, p2g, ng, cg, &f2);
  if (f2)
    {
      t_fail ("%s: glibc faulted -- invalid test setup", desc);
      return;
    }
  chk_long (desc, a, b);
}

static void
edge_pair_dst (int fk, const char *desc, unsigned char *d1, unsigned char *d2,
	       const void *src, size_t n, int c, unsigned char *page1,
	       unsigned char *page2)
{
  int f1 = 0, f2 = 0;
  long a, b;

  a = guard_call (IMPL_RVV, fk, d1, src, n, c, &f1);
  if (f1)
    {
      t_fail ("%s: rvv faulted on the destination guard page "
	      "(over-write past the page)", desc);
      return;
    }
  b = guard_call (IMPL_LIBC, fk, d2, src, n, c, &f2);
  if (f2)
    {
      t_fail ("%s: glibc faulted -- invalid test setup", desc);
      return;
    }
  chk_long (desc, a, b);
  chk_buf (desc, page1, page2, PG);
}

/* ------------------------------------------------------------------ */
/* single-string scans                                                */
/* ------------------------------------------------------------------ */
static const size_t scan_lens[] =
  { 0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    200, 255, 256, 257 };
static const int scan_shifts[] = { 0, 1, 3, 7, 8, 15 };

static void
run_scan (int fk)
{
  static const int chars[] = { 0, 'a', 0xff };
  size_t li, si;
  int ci, lo;

  for (lo = 0; lo < 2; lo++)
    for (li = 0; li < sizeof scan_lens / sizeof *scan_lens; li++)
      for (ci = 0; ci < (int) (sizeof chars / sizeof *chars); ci++)
	for (si = 0; si < (sizeof scan_shifts / sizeof *scan_shifts)
	     && (si == 0 || !lo); si++)
	  {
	    size_t L = scan_lens[li], shift = scan_shifts[si];
	    unsigned char *pg = lo ? guard_lo () : guard_hi ();
	    unsigned char *s = lo ? pg : pg - L - 1 - shift;
	    char label[80];

	    make_str (s, L, 0);
	    if (L && ci)
	      s[L - 1] = (unsigned char) chars[ci];
	    snprintf (label, sizeof label, "%s len=%zu %s%s c=0x%02x",
		      fk_name (fk), L, lo ? "lo-guard" : "hi-guard",
		      shift ? "+shift" : "", (unsigned) (chars[ci] & 0xff));
	    t_case ("len=%zu %s shift=%zu c=0x%02x", L,
		    lo ? "lo" : "hi", shift, (unsigned) (chars[ci] & 0xff));
	    if (fk == FK_strnlen)
	      {
		edge_pair (fk, label, s, NULL, L, 0, s, NULL, L, 0);
		edge_pair (fk, label, s, NULL, L + 1, 0, s, NULL, L + 1, 0);
		edge_pair (fk, label, s, NULL, L + 16, 0, s, NULL, L + 16, 0);
		edge_pair (fk, label, s, NULL, (size_t) -1, 0, s, NULL,
			   (size_t) -1, 0);
	      }
	    else
	      edge_pair (fk, label, s, NULL, 0, chars[ci], s, NULL, 0,
			 chars[ci]);
	    guard_free (pg);
	  }
}

void
edge_strlen (void) { run_scan (FK_strlen); }
void
edge_strnlen (void) { run_scan (FK_strnlen); }
void
edge_strchr (void) { run_scan (FK_strchr); }
void
edge_strchrnul (void) { run_scan (FK_strchrnul); }
void
edge_strrchr (void) { run_scan (FK_strrchr); }

/* ------------------------------------------------------------------ */
/* two-string comparisons                                             */
/* ------------------------------------------------------------------ */
enum { EV_EQUAL, EV_LASTDIFF, EV_PREFIX, EV_COUNT };

static void
run_cmp (int fk, int is_n)
{
  size_t li, si;
  int v;

  for (li = 0; li < sizeof scan_lens / sizeof *scan_lens; li++)
    for (v = 0; v < EV_COUNT; v++)
      for (si = 0; si < sizeof scan_shifts / sizeof *scan_shifts; si++)
	{
	  size_t L = scan_lens[li], shift = scan_shifts[si];
	  size_t la = (v == EV_PREFIX) ? L + 1 : L;
	  unsigned char *pa, *pb, *a, *b, *pagea, *pageb;
	  size_t n = 0;
	  char label[80];

	  pa = guard_hi ();
	  pb = guard_hi ();
	  pagea = pa - PG;
	  pageb = pb - PG;
	  a = pa - la - 1 - shift;
	  b = pb - L - 1 - shift;
	  make_str (a, la, 0);
	  make_str (b, L, 0);
	  memcpy (b, a, L);	/* b is a prefix of a (or equal) */
	  if (v == EV_LASTDIFF && L)
	    b[L - 1] = (unsigned char) (b[L - 1] ^ 0x5a);
	  if (v == EV_EQUAL)
	    {
	      la = L;
	      a = pa - L - 1 - shift;
	      make_str (a, L, 0);
	      memcpy (b, a, L);
	    }
	  if (is_n)
	    n = L + 1;
	  snprintf (label, sizeof label, "%s len=%zu %s shift=%zu",
		    fk_name (fk), L, v == EV_EQUAL ? "equal" :
		    v == EV_LASTDIFF ? "lastdiff" : "prefix", shift);
	  t_case ("len=%zu shift=%zu variant=%d", L, shift, v);
	  edge_pair (fk, label, a, b, n, 0, a, b, n, 0);
	  if (is_n)
	    {
	      edge_pair (fk, label, a, b, (size_t) -1, 0, a, b,
			 (size_t) -1, 0);
	      edge_pair (fk, label, a, b, L, 0, a, b, L, 0);
	      edge_pair (fk, label, a, b, L + 1, 0, a, b, L + 1, 0);
	    }
	  guard_free (pa);
	  guard_free (pb);
	  (void) pagea;
	  (void) pageb;
	}
}

void
edge_strcmp (void) { run_cmp (FK_strcmp, 0); }
void
edge_strncmp (void) { run_cmp (FK_strncmp, 1); }
void
edge_strcasecmp (void) { run_cmp (FK_strcasecmp, 0); }
void
edge_strncasecmp (void) { run_cmp (FK_strncasecmp, 1); }

/* ------------------------------------------------------------------ */
/* destination-writing string routines                                */
/* ------------------------------------------------------------------ */
static const char *const edge_prefixes[] = { "", "pre", "0123456789abcdef" };

static void
run_strcpy_like (int fk)
{
  static const int ns[] = { 0, 1, 15, 16, 17, 33, 64, 65 };
  size_t li, pi, ni;
  int lo;

  for (lo = 0; lo < 2; lo++)
    for (li = 0; li < sizeof scan_lens / sizeof *scan_lens; li++)
      for (pi = 0; pi < sizeof edge_prefixes / sizeof *edge_prefixes; pi++)
	for (ni = 0; ni < sizeof ns / sizeof *ns; ni++)
	  {
	    size_t L = scan_lens[li], plen = strlen (edge_prefixes[pi]);
	    long n = 0, app, total;
	    unsigned char *ps, *pd1, *pd2, *src, *d1, *d2, *b1, *b2;
	    char label[96];

	    /* only the first prefix gets the full length sweep */
	    if (pi > 0 && (L > 40 || (li & 1)))
	      continue;
	    switch (fk)
	      {
	      case FK_strcpy:
		n = 0;
		app = (long) L;
		break;
	      case FK_strcat:
		n = 0;
		app = (long) L;
		break;
	      case FK_strncpy:
		n = ns[ni];
		app = n;
		break;
	      default:		/* strncat */
		n = ns[ni];
		app = n < (long) L ? n : (long) L;
		break;
	      }
	    if (fk == FK_strcpy || fk == FK_strcat)
	      {
		if (ni)
		  continue;	/* no n parameter */
	      }
	    total = (long) plen + app + 1;
	    if (total <= 0 || (size_t) total > PG)
	      continue;
	    if (fk == FK_strncpy && n == 0)
	      continue;		/* n == 0 is a no-op, covered by sweep */

	    ps = guard_hi ();
	    pd1 = guard_hi ();
	    pd2 = guard_hi ();
	    src = ps - L - 1;
	    d1 = pd1 - total;
	    d2 = pd2 - total;
	    b1 = pd1 - PG;
	    b2 = pd2 - PG;
	    make_str (src, L, 0);
	    canary_fill (b1, PG, 51);
	    canary_fill (b2, PG, 51);
	    memcpy (d1, edge_prefixes[pi], plen + 1);
	    memcpy (d2, edge_prefixes[pi], plen + 1);

	    snprintf (label, sizeof label, "%s(%s) len=%zu pfx=%zu n=%ld",
		      fk_name (fk), lo ? "lo-src" : "hi-src", L, pi, n);
	    t_case ("len=%zu pfx=%zu n=%ld %s", L, pi, n, lo ? "src-lo" : "");
	    edge_pair_dst (fk, label, d1, d2, src, n, 0, b1, b2);
	    guard_free (ps);
	    guard_free (pd1);
	    guard_free (pd2);
	  }
}

void
edge_strcpy (void) { run_strcpy_like (FK_strcpy); }
void
edge_strncpy (void) { run_strcpy_like (FK_strncpy); }
void
edge_strcat (void) { run_strcpy_like (FK_strcat); }
void
edge_strncat (void) { run_strcpy_like (FK_strncat); }

/* ------------------------------------------------------------------ */
/* strspn / strcspn / strpbrk                                          */
/* ------------------------------------------------------------------ */
static void
run_span (int fk)
{
  static const char *const sets[] = { "", "a", "ab", "z", "x", "aZbY0_~!%" };
  size_t li, si;
  int lo;

  for (lo = 0; lo < 2; lo++)
    for (li = 0; li < sizeof scan_lens / sizeof *scan_lens; li++)
      for (si = 0; si < sizeof sets / sizeof *sets; si++)
	{
	  size_t L = scan_lens[li];
	  unsigned char *pg = lo ? guard_lo () : guard_hi ();
	  unsigned char *s = lo ? pg : pg - L - 1;
	  char label[80];

	  make_str (s, L, 2);
	  if (L && si == 1)
	    s[L - 1] = 'a';
	  if (L && si == 4)
	    s[L - 1] = 'x';
	  snprintf (label, sizeof label, "%s len=%zu %s set=%zu",
		    fk_name (fk), L, lo ? "lo-guard" : "hi-guard", si);
	  t_case ("len=%zu %s set=%zu", L, lo ? "lo" : "hi", si);
	  edge_pair (fk, label, s, sets[si], 0, 0, s, sets[si], 0, 0);
	  guard_free (pg);
	}
}

void
edge_strspn (void) { run_span (FK_strspn); }
void
edge_strcspn (void) { run_span (FK_strcspn); }
void
edge_strpbrk (void) { run_span (FK_strpbrk); }

/* ------------------------------------------------------------------ */
/* strstr                                                             */
/* ------------------------------------------------------------------ */
static void
run_strstr (void)
{
  size_t li, si;
  int k, lo;

  for (lo = 0; lo < 2; lo++)
    for (li = 0; li < sizeof scan_lens / sizeof *scan_lens; li++)
      for (si = 0; si < sizeof scan_shifts / sizeof *scan_shifts; si++)
	for (k = 0; k < 3; k++)
	  {
	    size_t L = scan_lens[li], shift = scan_shifts[si];
	    unsigned char *ph = lo ? guard_lo () : guard_hi ();
	    unsigned char *pn = guard_hi ();
	    unsigned char *hay = lo ? ph : ph - L - 1 - shift;
	    unsigned char *nd = pn - 8;
	    const char *label = k == 0 ? "at-end" : k == 1 ? "absent" : "prefix";
	    char desc[96];

	    make_str (hay, L, 0);
	    make_str (nd, 3, 0);
	    if (k == 0)
	      {
		if (L >= 3)
		  memcpy (nd, hay + L - 3, 3);
		nd[3] = 0;
	      }
	    else if (k == 1)
	      {
		strcpy ((char *) nd, "zzz");
		if (L)
		  hay[L - 1] = 'q';
	      }
	    else
	      {
		if (L >= 3)
		  memcpy (nd, hay, 3);
		nd[3] = 0;
	      }
	    snprintf (desc, sizeof desc, "strstr len=%zu %s needle=%s",
		      L, lo ? "lo-guard" : "hi-guard", label);
	    t_case ("len=%zu %s needle=%s", L, lo ? "lo" : "hi", label);
	    edge_pair (FK_strstr, desc, hay, nd, 0, 0, hay, nd, 0, 0);
	    guard_free (ph);
	    guard_free (pn);
	  }
}

void
edge_strstr (void) { run_strstr (); }

/* ------------------------------------------------------------------ */
/* memcpy / mempcpy                                                   */
/* ------------------------------------------------------------------ */
static const size_t mem_lens[] =
  { 0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    200, 255, 256, 257, 1000, 2048 };

static void
run_memcpy_like (int fk)
{
  size_t li, si;
  int lo;

  for (lo = 0; lo < 2; lo++)
    for (li = 0; li < sizeof mem_lens / sizeof *mem_lens; li++)
      for (si = 0; si < sizeof scan_shifts / sizeof *scan_shifts; si++)
	{
	  size_t n = mem_lens[li], shift = scan_shifts[si];
	  unsigned char *ps = lo ? guard_lo () : guard_hi ();
	  unsigned char *pd1 = guard_hi (), *pd2 = guard_hi ();
	  unsigned char *src = lo ? ps : ps - n - shift;
	  unsigned char *d1 = pd1 - n, *d2 = pd2 - n;
	  unsigned char *b1 = pd1 - PG, *b2 = pd2 - PG;
	  char label[80];

	  make_data (src, n, 0);
	  canary_fill (b1, PG, 61);
	  canary_fill (b2, PG, 61);
	  snprintf (label, sizeof label, "%s n=%zu %s", fk_name (fk), n,
		    lo ? "src-lo-guard" : "src-hi-guard");
	  t_case ("n=%zu %s shift=%zu", n, lo ? "lo" : "hi", shift);
	  edge_pair_dst (fk, label, d1, d2, src, n, 0, b1, b2);
	  guard_free (ps);
	  guard_free (pd1);
	  guard_free (pd2);
	}
}

void
edge_memcpy (void) { run_memcpy_like (FK_memcpy); }
void
edge_mempcpy (void) { run_memcpy_like (FK_mempcpy); }

/* ------------------------------------------------------------------ */
/* Overlapping moves at a page boundary.

   dir=1 (dst > src): the write range ends exactly at the page end, so a
                      write one byte past dst+n faults (the classic
                      backward-copy overrun).
   dir=0 (dst < src): the read range ends exactly at the page end, so a
                      read one byte past src+n faults (the classic
                      forward-copy overrun).  */
static void
run_memmove (void)
{
  static const long dists[] = { 1, 7, 8, 9, 16, 32, 40 };
  size_t li, di, si;
  int dir;

  for (li = 0; li < sizeof mem_lens / sizeof *mem_lens; li++)
    for (di = 0; di < sizeof dists / sizeof *dists; di++)
      for (si = 0; si < sizeof scan_shifts / sizeof *scan_shifts; si++)
	for (dir = 0; dir < 2; dir++)
	  {
	    size_t n = mem_lens[li];
	    long d = dists[di];
	    size_t shift = scan_shifts[si];
	    unsigned char *p1 = guard_hi (), *p2 = guard_hi ();
	    unsigned char *b1 = p1 - PG, *b2 = p2 - PG;
	    unsigned char *src1, *src2, *d1, *d2;
	    char label[96];

	    if (n == 0)
	      continue;
	    if (n + (size_t) d + shift + 128 > PG)
	      continue;
	    canary_fill (b1, PG, 71);
	    canary_fill (b2, PG, 71);
	    if (dir)
	      {
		src1 = p1 - n - d - shift;
		src2 = p2 - n - d - shift;
		d1 = p1 - n - shift;
		d2 = p2 - n - shift;
	      }
	    else
	      {
		src1 = p1 - n - shift;
		src2 = p2 - n - shift;
		d1 = p1 - n - d - shift;
		d2 = p2 - n - d - shift;
	      }
	    make_data (src1, n, 0);
	    make_data (src2, n, 0);
	    snprintf (label, sizeof label,
		      "memmove n=%zu d=%ld dir=%s shift=%zu", n, d,
		      dir ? "dst>src" : "dst<src", shift);
	    t_case ("n=%zu d=%ld dir=%s", n, d, dir ? "fwd" : "bwd");
	    edge_pair_dst (FK_memmove, label, d1, d2, src1, n, 0, b1, b2);
	    guard_free (p1);
	    guard_free (p2);
	  }
}

void
edge_memmove (void) { run_memmove (); }

/* ------------------------------------------------------------------ */
/* memchr / memrchr / rawmemchr                                       */
/* ------------------------------------------------------------------ */
static void
run_memchr (int fk)
{
  static const int chars[] = { 'a', 0x00, 0xff };
  size_t li, si;
  int ci, lo;

  for (lo = 0; lo < 2; lo++)
    for (li = 0; li < sizeof mem_lens / sizeof *mem_lens; li++)
      for (ci = 0; ci < (int) (sizeof chars / sizeof *chars); ci++)
	for (si = 0; si < sizeof scan_shifts / sizeof *scan_shifts
	     && (si == 0 || !lo); si++)
	  {
	    size_t n = mem_lens[li], shift = scan_shifts[si];
	    unsigned char *pg = lo ? guard_lo () : guard_hi ();
	    unsigned char *s = lo ? pg : pg - n - shift;
	    char label[80];
	    size_t k;

	    if (fk == FK_rawmemchr && n == 0)
	      continue;
	    if (fk == FK_rawmemchr && lo)
	      continue;
	    make_data (s, n, 0);
	    for (k = 0; k < n; k++)
	      if (s[k] == (unsigned char) chars[ci])
		s[k] = (unsigned char) ~chars[ci];
	    if (n)
	      s[n - 1] = (unsigned char) chars[ci];
	    snprintf (label, sizeof label, "%s n=%zu %s c=0x%02x",
		      fk_name (fk), n, lo ? "lo-guard" : "hi-guard",
		      (unsigned) (chars[ci] & 0xff));
	    t_case ("n=%zu %s c=0x%02x", n, lo ? "lo" : "hi",
		    (unsigned) (chars[ci] & 0xff));
	    edge_pair (fk, label, s, NULL, n, chars[ci], s, NULL, n,
		       chars[ci]);
	    guard_free (pg);
	  }
}

void
edge_memchr (void) { run_memchr (FK_memchr); }
void
edge_memrchr (void) { run_memchr (FK_memrchr); }
void
edge_rawmemchr (void) { run_memchr (FK_rawmemchr); }

/* ------------------------------------------------------------------ */
static void
run_memcmp (void)
{
  size_t li, si;
  int v, lo;

  for (lo = 0; lo < 2; lo++)
    for (li = 0; li < sizeof mem_lens / sizeof *mem_lens; li++)
      for (si = 0; si < sizeof scan_shifts / sizeof *scan_shifts; si++)
	for (v = 0; v < 3; v++)
	  {
	    size_t n = mem_lens[li], shift = scan_shifts[si];
	    unsigned char *pa = lo ? guard_lo () : guard_hi ();
	    unsigned char *pb = guard_hi ();
	    unsigned char *a = lo ? pa : pa - n - shift;
	    unsigned char *b = pb - n - shift;
	    char label[80];

	    if (n == 0 && v == 1)
	      {
		guard_free (pa); guard_free (pb);
		continue;
	      }
	    make_data (a, n, 0);
	    memcpy (b, a, n);
	    if (v == 1 && n)
	      b[n - 1] = (unsigned char) ~b[n - 1];
	    if (v == 2 && n)
	      {
		a[0] = 0x80;
		b[0] = 0x01;
	      }
	    snprintf (label, sizeof label, "memcmp n=%zu %s variant=%d",
		      n, lo ? "lo-guard" : "hi-guard", v);
	    t_case ("n=%zu %s variant=%d", n, lo ? "lo" : "hi", v);
	    edge_pair (FK_memcmp, label, a, b, n, 0, a, b, n, 0);
	    guard_free (pa);
	    guard_free (pb);
	  }
}

void
edge_memcmp (void) { run_memcmp (); }

/* ------------------------------------------------------------------ */
static void
run_memset (void)
{
  size_t li, si;

  for (li = 0; li < sizeof mem_lens / sizeof *mem_lens; li++)
    for (si = 0; si < sizeof scan_shifts / sizeof *scan_shifts; si++)
      {
	size_t n = mem_lens[li], shift = scan_shifts[si];
	unsigned char *pd1 = guard_hi (), *pd2 = guard_hi ();
	unsigned char *d1 = pd1 - n - shift, *d2 = pd2 - n - shift;
	unsigned char *b1 = pd1 - PG, *b2 = pd2 - PG;
	char label[80];
	canary_fill (b1, PG, 81);
	canary_fill (b2, PG, 81);
	snprintf (label, sizeof label, "memset n=%zu shift=%zu", n, shift);
	t_case ("n=%zu shift=%zu", n, shift);
	edge_pair_dst (FK_memset, label, d1, d2, NULL, n, 0xa5, b1, b2);
	guard_free (pd1);
	guard_free (pd2);
      }
}

void
edge_memset (void) { run_memset (); }

/* ------------------------------------------------------------------ */
/* memccpy: source range ending at the page end, the stop byte at the
   last position (must copy exactly to it), in the middle, or absent
   (must copy the whole range).  */
static void
run_memccpy (void)
{
  size_t li, si;
  int v;

  for (li = 0; li < sizeof mem_lens / sizeof *mem_lens; li++)
    for (si = 0; si < sizeof scan_shifts / sizeof *scan_shifts; si++)
      for (v = 0; v < 3; v++)
	{
	  size_t n = mem_lens[li], shift = scan_shifts[si];
	  unsigned char *ps = guard_hi (), *pd1 = guard_hi (), *pd2 = guard_hi ();
	  unsigned char *b1 = pd1 - PG, *b2 = pd2 - PG;
	  unsigned char *src = ps - n - shift;
	  unsigned char *d1 = pd1 - n, *d2 = pd2 - n;
	  long pos = (n == 0) ? -1
	    : v == 0 ? -1 : v == 1 ? (long) (n - 1) : (long) (n / 2);
	  size_t k;
	  char label[96];

	  make_data (src, n, 0);
	  for (k = 0; k < n; k++)
	    if (src[k] == 'Q')
	      src[k] = 'q';
	  if (pos >= 0)
	    src[pos] = 'Q';
	  canary_fill (b1, PG, 91);
	  canary_fill (b2, PG, 91);
	  snprintf (label, sizeof label, "memccpy n=%zu shift=%zu c=%s",
		    n, shift, v == 0 ? "absent" : v == 1 ? "last" : "mid");
	  t_case ("n=%zu shift=%zu variant=%d", n, shift, v);
	  edge_pair_dst (FK_memccpy, label, d1, d2, src, n, 'Q', b1, b2);
	  guard_free (ps);
	  guard_free (pd1);
	  guard_free (pd2);
	}
}

void
edge_memccpy (void) { run_memccpy (); }
