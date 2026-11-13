/* Shared infrastructure for the RVV string/memory correctness harness.  */

#include "harness.h"

#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

/* ------------------------------------------------------------------ */
/* Reporting                                                          */
/* ------------------------------------------------------------------ */
static const char *cur_fn = "?";
static char cur_case[512];
static long n_checks, n_fails, n_warns;
static long tot_checks, tot_fails;
static int printed, print_max = 12;

static void
vreport (const char *tag, const char *fmt, va_list ap)
{
  printf ("%-4s %-12s %-52s : ", tag, cur_fn, cur_case);
  vprintf (fmt, ap);
  putchar ('\n');
}

void
t_begin (const char *fn)
{
  cur_fn = fn;
  cur_case[0] = 0;
  n_checks = n_fails = n_warns = 0;
  printed = 0;
}

void
t_case (const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  vsnprintf (cur_case, sizeof cur_case, fmt, ap);
  va_end (ap);
}

void
t_fail (const char *fmt, ...)
{
  va_list ap;
  n_fails++;
  if (printed < print_max)
    {
      va_start (ap, fmt);
      vreport ("FAIL", fmt, ap);
      va_end (ap);
      printed++;
      if (printed == print_max)
	printf ("     %-12s (further failures for this function elided)\n",
		cur_fn);
    }
}

void
t_warn (const char *fmt, ...)
{
  va_list ap;
  n_warns++;
  va_start (ap, fmt);
  vreport ("WARN", fmt, ap);
  va_end (ap);
}

void
t_end (void)
{
  tot_checks += n_checks;
  tot_fails += n_fails;
  printf ("[%s] %-14s %8ld checks  %6ld fail  %4ld warn\n",
	  n_fails ? "FAIL" : "PASS", cur_fn, n_checks, n_fails, n_warns);
  fflush (stdout);
}

long t_checks (void) { return n_checks; }
long t_fails (void) { return n_fails; }
long t_total_checks (void) { return tot_checks; }
long t_total_fails (void) { return tot_fails; }
int t_exit_code (void) { return tot_fails ? 1 : 0; }

static unsigned long
read_vlenb (void)
{
#ifdef __riscv_vector
  unsigned long x;
  __asm__ volatile ("csrr %0, vlenb" : "=r" (x));
  return x;
#else
  return 0;
#endif
}

unsigned long
t_vlen_bits (void)
{
  return read_vlenb () * 8;
}

void
t_banner (void)
{
  unsigned long vb = read_vlenb ();
  printf ("# RVV string/memory correctness harness\n");
  if (vb)
    printf ("# VLEN = %lu bits (vlenb=%lu), page size = %zu\n",
	    vb * 8, vb, page_size ());
  else
    printf ("# (not built with the V extension)\n");
}

/* ------------------------------------------------------------------ */
/* Buffers                                                            */
/* ------------------------------------------------------------------ */
#define REGION_SIZE (4u << 20)

static unsigned char *
new_region (void)
{
  void *p = mmap (NULL, REGION_SIZE, PROT_READ | PROT_WRITE,
		  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED)
    {
      perror ("mmap");
      exit (2);
    }
  return p;
}

static unsigned char *R_ro, *R_d1, *R_d2, *R_m1, *R_m2, *R_s1, *R_s2;

static void
regions_init (void)
{
  if (R_ro)
    return;
  R_ro = new_region ();
  R_d1 = new_region ();
  R_d2 = new_region ();
  R_m1 = new_region ();
  R_m2 = new_region ();
  R_s1 = new_region ();
  R_s2 = new_region ();
}

unsigned char *ro_base (void) { regions_init (); return R_ro; }
unsigned char *d1_base (void) { regions_init (); return R_d1; }
unsigned char *d2_base (void) { regions_init (); return R_d2; }
unsigned char *m1_base (void) { regions_init (); return R_m1; }
unsigned char *m2_base (void) { regions_init (); return R_m2; }
unsigned char *s1_base (void) { regions_init (); return R_s1; }
unsigned char *s2_base (void) { regions_init (); return R_s2; }

size_t
page_size (void)
{
  static size_t ps;
  if (!ps)
    ps = (size_t) sysconf (_SC_PAGESIZE);
  return ps;
}

/* ------------------------------------------------------------------ */
/* Canary pattern                                                     */
/* ------------------------------------------------------------------ */
static uint32_t
mix32 (uint32_t x)
{
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

#define TAB_SZ 4096
static unsigned char canary_tab[TAB_SZ];
static unsigned char str_tab[8][TAB_SZ];
static unsigned char data_tab[8][TAB_SZ];
static int tabs_ready;

static unsigned char str_raw (int variant, size_t i);
static unsigned char data_raw (int variant, size_t i);

static void
tabs_init (void)
{
  uint32_t x = 0x12345678u;
  size_t i;
  int v;

  if (tabs_ready)
    return;
  tabs_ready = 1;
  for (i = 0; i < TAB_SZ; i++)
    {
      x = mix32 (x + 0x9e3779b9u);
      canary_tab[i] = (unsigned char) (x >> 17);
    }
  for (v = 0; v < 8; v++)
    for (i = 0; i < TAB_SZ; i++)
      {
	str_tab[v][i] = str_raw (v, i);
	data_tab[v][i] = data_raw (v, i);
      }
}

void
canary_fill (unsigned char *window, size_t len, unsigned seed)
{
  size_t off = 0, start = (size_t) (seed * 64u) % TAB_SZ;
  tabs_init ();
  while (off < len)
    {
      size_t c = TAB_SZ - start;
      if (c > len - off)
	c = len - off;
      memcpy (window + off, canary_tab + start, c);
      off += c;
      start = 0;
    }
}

void
chk_win (const char *what, const unsigned char *w1, const unsigned char *w2,
	 size_t n, size_t align)
{
  size_t len = WIN_LEN (n), i;
  for (i = 0; i < len; i++)
    if (w1[i] != w2[i])
      {
	long rel = (long) i - (long) (WIN_PAD + align);
	if (rel >= 0 && (size_t) rel < n)
	  t_fail ("%s: payload byte %ld differs: rvv=0x%02x glibc=0x%02x",
		  what, rel, w1[i], w2[i]);
	else
	  t_fail ("%s: %s guard byte at payload offset %ld differs: "
		  "rvv=0x%02x glibc=0x%02x", what,
		  rel < 0 ? "leading" : "trailing", rel, w1[i], w2[i]);
	return;
      }
}

/* ------------------------------------------------------------------ */
/* Guard pages                                                        */
/* ------------------------------------------------------------------ */
static unsigned char *
guard_new (int hi)
{
  size_t pg = page_size ();
  unsigned char *b = mmap (NULL, 2 * pg, PROT_READ | PROT_WRITE,
			   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (b == MAP_FAILED)
    {
      perror ("mmap guard");
      exit (2);
    }
  if (mprotect (hi ? b + pg : b, pg, PROT_NONE) != 0)
    {
      perror ("mprotect");
      exit (2);
    }
  return b + pg;
}

unsigned char *guard_hi (void) { return guard_new (1); }
unsigned char *guard_lo (void) { return guard_new (0); }

void
guard_free (unsigned char *p)
{
  size_t pg = page_size ();
  munmap (p - pg, 2 * pg);
}

static sigjmp_buf jb;
static volatile sig_atomic_t armed;

static void
fault_handler (int sig)
{
  if (armed)
    siglongjmp (jb, 1);
  /* Not inside guard_call: a harness bug or a wild pointer.  */
  fprintf (stderr, "fatal: signal %d outside a guarded call (fn=%s case=%s)\n",
	   sig, cur_fn, cur_case);
  _exit (128 + sig);
}

static void
install_handler (void)
{
  static int done;
  struct sigaction sa;
  if (done)
    return;
  done = 1;
  memset (&sa, 0, sizeof sa);
  sa.sa_handler = fault_handler;
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction (SIGSEGV, &sa, NULL);
  sigaction (SIGBUS, &sa, NULL);
}

long
guard_call (int impl, int fk, void *p1, const void *p2, size_t n, int c,
	    int *faulted)
{
  volatile long r = 0;
  install_handler ();
  *faulted = 0;
  if (sigsetjmp (jb, 1) == 0)
    {
      armed = 1;
      r = fcall (impl, fk, p1, p2, n, c);
      armed = 0;
    }
  else
    {
      armed = 0;
      *faulted = 1;
    }
  return r;
}

/* ------------------------------------------------------------------ */
/* Data patterns                                                      */
/* ------------------------------------------------------------------ */
/* Eight 4096-byte, aperiodic-within-the-table patterns per class.  The
   tables are copied in 4096-byte chunks, so the effective period is 4096
   bytes: a routine that shifts data by one vector chunk cannot hide behind
   a short repeating pattern.  String patterns never contain a NUL.  */
static unsigned char
str_raw (int variant, size_t i)
{
  static const unsigned char v0[] = "aZbY0_~!%\x80\xa5\xfe\xff";
  static const unsigned char v3[] = "AaBbCcDdEeFfGgHh";
  static const unsigned char v5[] = "@AZ[`az{|}~";
  static const unsigned char v6[] = "abcABC0123"; 
  switch (variant & 7)
    {
    case 0: return v0[i % (sizeof v0 - 1)];
    case 1: return 'a';
    case 2: return (i & 1) ? 'b' : 'a';
    case 3: return v3[i % (sizeof v3 - 1)];
    case 4: return (unsigned char) (0x80 + (i % 128));
    case 5: return v5[i % (sizeof v5 - 1)];
    case 6: return v6[i % (sizeof v6 - 1)];
    default: return (unsigned char) ('A' + (i % 26));
    }
}

static unsigned char
data_raw (int variant, size_t i)
{
  switch (variant & 7)
    {
    case 0:
      if (i % 7 == 0) return 0x00;
      if (i % 11 == 0) return 0xff;
      if (i % 5 == 0) return 0x80;
      return (unsigned char) (i * 37 + (i >> 8) * 13 + 11);
    case 1: return 0x00;
    case 2: return (unsigned char) (i * 37 + (i >> 8) * 13 + 11);
    case 3: return 0xff;
    case 4: return (i & 1) ? 0x00 : 0xff;
    case 5: return (unsigned char) (i * 7 + i / 13 + 3);
    case 6: return (unsigned char) (0x40 + (i % 60));
    default: return (unsigned char) (i ^ (i >> 3));
    }
}

unsigned char
datach (int variant, size_t i) { tabs_init (); return data_tab[variant & 7][i % TAB_SZ]; }
unsigned char
strch (int variant, size_t i) { tabs_init (); return str_tab[variant & 7][i % TAB_SZ]; }

static void
copy_pattern (unsigned char *p, size_t len, const unsigned char *tab)
{
  size_t off = 0;
  while (off < len)
    {
      size_t c = len - off;
      if (c > TAB_SZ)
	c = TAB_SZ;
      memcpy (p + off, tab, c);
      off += c;
    }
}

void
make_str (unsigned char *p, size_t len, int variant)
{
  tabs_init ();
  copy_pattern (p, len, str_tab[variant & 7]);
  p[len] = 0;
}

void
make_data (unsigned char *p, size_t len, int variant)
{
  tabs_init ();
  copy_pattern (p, len, data_tab[variant & 7]);
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                           */
/* ------------------------------------------------------------------ */
static long
ptr_off (const void *p, const void *base)
{
  return p ? (long) ((const char *) p - (const char *) base) : LONG_MIN;
}

void
fmt_off (char *buf, size_t buflen, long off)
{
  if (off == LONG_MIN)
    snprintf (buf, buflen, "NULL");
  else
    snprintf (buf, buflen, "offset %ld", off);
}

long
fcall (int impl, int fk, void *p1, const void *p2, size_t n, int c)
{
  const char *s1 = (const char *) p1;
  const char *s2 = (const char *) p2;
  switch (fk)
    {
    case FK_strlen:
      return (long) (impl ? strlen (s1) : rvv_strlen (s1));
    case FK_strnlen:
      return (long) (impl ? strnlen (s1, n) : rvv_strnlen (s1, n));
    case FK_strchr:
      return ptr_off (impl ? (void *) strchr (s1, c) : rvv_strchr (s1, c), p1);
    case FK_strchrnul:
      return ptr_off (impl ? (void *) strchrnul (s1, c) : rvv_strchrnul (s1, c),
		      p1);
    case FK_strrchr:
      return ptr_off (impl ? (void *) strrchr (s1, c) : rvv_strrchr (s1, c), p1);
    case FK_strcmp:
      return (long) (impl ? strcmp (s1, s2) : rvv_strcmp (s1, s2));
    case FK_strncmp:
      return (long) (impl ? strncmp (s1, s2, n) : rvv_strncmp (s1, s2, n));
    case FK_strcpy:
      return ptr_off (impl ? (void *) strcpy (p1, s2) : rvv_strcpy (p1, s2), p1);
    case FK_strncpy:
      return ptr_off (impl ? (void *) strncpy (p1, s2, n)
		      : rvv_strncpy (p1, s2, n), p1);
    case FK_strcat:
      return ptr_off (impl ? (void *) strcat (p1, s2) : rvv_strcat (p1, s2), p1);
    case FK_strncat:
      return ptr_off (impl ? (void *) strncat (p1, s2, n)
		      : rvv_strncat (p1, s2, n), p1);
    case FK_strspn:
      return (long) (impl ? strspn (s1, s2) : rvv_strspn (s1, s2));
    case FK_strcspn:
      return (long) (impl ? strcspn (s1, s2) : rvv_strcspn (s1, s2));
    case FK_strpbrk:
      return ptr_off (impl ? (void *) strpbrk (s1, s2) : rvv_strpbrk (s1, s2),
		      p1);
    case FK_strstr:
      return ptr_off (impl ? (void *) strstr (s1, s2) : rvv_strstr (s1, s2),
		      p1);
    case FK_strcasecmp:
      return (long) (impl ? strcasecmp (s1, s2) : rvv_strcasecmp (s1, s2));
    case FK_strncasecmp:
      return (long) (impl ? strncasecmp (s1, s2, n)
		      : rvv_strncasecmp (s1, s2, n));
    case FK_memcpy:
      return ptr_off (impl ? memcpy (p1, p2, n) : rvv_memcpy (p1, p2, n), p1);
    case FK_memmove:
      return ptr_off (impl ? memmove (p1, p2, n) : rvv_memmove (p1, p2, n),
		      p1);
    case FK_memccpy:
      return ptr_off (impl ? memccpy (p1, p2, c, n)
		      : rvv_memccpy (p1, p2, c, n), p1);
    case FK_mempcpy:
      return ptr_off (impl ? mempcpy (p1, p2, n) : rvv_mempcpy (p1, p2, n),
		      p1);
    case FK_memchr:
      return ptr_off (impl ? memchr (p1, c, n) : rvv_memchr (p1, c, n), p1);
    case FK_memrchr:
      return ptr_off (impl ? memrchr (p1, c, n) : rvv_memrchr (p1, c, n), p1);
    case FK_memcmp:
      return (long) (impl ? memcmp (p1, p2, n) : rvv_memcmp (p1, p2, n));
    case FK_rawmemchr:
      return ptr_off (impl ? rawmemchr (p1, c) : rvv_rawmemchr (p1, c), p1);
    case FK_memset:
      return ptr_off (impl ? memset (p1, c, n) : rvv_memset (p1, c, n), p1);
    default:
      return 0;
    }
}

const char *
fk_name (int fk)
{
  static const char *names[] = {
    "strlen", "strnlen", "strchr", "strchrnul", "strrchr",
    "strcmp", "strncmp", "strcpy", "strncpy", "strcat", "strncat",
    "strspn", "strcspn", "strpbrk", "strstr",
    "strcasecmp", "strncasecmp",
    "memcpy", "memmove", "memccpy", "mempcpy",
    "memchr", "memrchr", "memcmp", "rawmemchr", "memset"
  };
  return (fk >= 0 && fk < FK_COUNT) ? names[fk] : "?";
}

/* ------------------------------------------------------------------ */
/* Comparisons                                                        */
/* ------------------------------------------------------------------ */
void
chk_long (const char *what, long ours, long ref)
{
  n_checks++;
  if (ours != ref)
    t_fail ("%s: rvv=%ld glibc=%ld", what, ours, ref);
}

void
chk_size (const char *what, size_t ours, size_t ref)
{
  n_checks++;
  if (ours != ref)
    t_fail ("%s: rvv=%zu glibc=%zu", what, ours, ref);
}

void
chk_byte (const char *what, unsigned ours, unsigned ref)
{
  n_checks++;
  if (ours != ref)
    t_fail ("%s: rvv=0x%02x glibc=0x%02x", what, ours, ref);
}

void
chk_buf (const char *what, const unsigned char *ours, const unsigned char *ref,
	 size_t n)
{
  size_t i;
  n_checks++;
  for (i = 0; i < n; i++)
    if (ours[i] != ref[i])
      {
	t_fail ("%s: first differing byte at index %zu: "
		"rvv=0x%02x glibc=0x%02x", what, i, ours[i], ref[i]);
	return;
      }
}

void
chk_ptr (const char *what, const void *ours, const void *ourbase,
	 const void *ref, const void *refbase)
{
  long a = ptr_off (ours, ourbase);
  long b = ptr_off (ref, refbase);
  char ba[32], bb[32];
  n_checks++;
  if (a == b)
    return;
  fmt_off (ba, sizeof ba, a);
  fmt_off (bb, sizeof bb, b);
  t_fail ("%s: rvv=%s glibc=%s", what, ba, bb);
}
