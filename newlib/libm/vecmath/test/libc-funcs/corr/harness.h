/* Shared infrastructure for the RVV-vs-glibc string/memory correctness
   harness.

   The vector routines from newlib/libc/machine/riscv/<name>-asm.S are
   assembled
   and their entry symbols renamed with
        objcopy --redefine-sym strlen=rvv_strlen
   so that the unmodified platform (glibc) routines remain callable under
   their normal names.  Every test therefore runs both implementations on
   identical inputs and compares the results and the complete destination
   window (including canary bytes) byte for byte.

   Everything here is plain POSIX C so the harness also builds and runs on a
   real RISC-V Linux board.  */

#ifndef HARNESS_H
#define HARNESS_H 1

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Implementations under test (built from newlib/libc/machine/riscv).  */
/* ------------------------------------------------------------------ */
extern size_t rvv_strlen (const char *);
extern size_t rvv_strnlen (const char *, size_t);
extern char *rvv_strchr (const char *, int);
extern char *rvv_strchrnul (const char *, int);
extern char *rvv_strrchr (const char *, int);
extern int rvv_strcmp (const char *, const char *);
extern int rvv_strncmp (const char *, const char *, size_t);
extern char *rvv_strcpy (char *, const char *);
extern char *rvv_strncpy (char *, const char *, size_t);
extern char *rvv_strcat (char *, const char *);
extern char *rvv_strncat (char *, const char *, size_t);
extern size_t rvv_strspn (const char *, const char *);
extern size_t rvv_strcspn (const char *, const char *);
extern char *rvv_strpbrk (const char *, const char *);
extern char *rvv_strstr (const char *, const char *);
extern int rvv_strcasecmp (const char *, const char *);
extern int rvv_strncasecmp (const char *, const char *, size_t);
extern void *rvv_memcpy (void *, const void *, size_t);
extern void *rvv_memmove (void *, const void *, size_t);
extern void *rvv_memccpy (void *, const void *, int, size_t);
extern void *rvv_mempcpy (void *, const void *, size_t);
extern void *rvv_memchr (const void *, int, size_t);
extern void *rvv_memrchr (const void *, int, size_t);
extern int rvv_memcmp (const void *, const void *, size_t);
extern void *rvv_rawmemchr (const void *, int);
extern void *rvv_memset (void *, int, size_t);

/* Scalar (non-vector) fallbacks from the same directory.  They are linked
   in on a best-effort basis (several of the newlib scalar sources include
   newlib-private headers and cannot be built with a glibc cross compiler),
   hence weak references plus a NULL check.  Used only by the small
   self-check section, never as the oracle.  */
extern size_t scalar_strlen (const char *) __attribute__ ((weak));
extern int scalar_strcmp (const char *, const char *) __attribute__ ((weak));
extern char *scalar_strcpy (char *, const char *) __attribute__ ((weak));
extern size_t scalar_strspn (const char *, const char *) __attribute__ ((weak));
extern size_t scalar_strcspn (const char *, const char *) __attribute__ ((weak));
extern char *scalar_strpbrk (const char *, const char *) __attribute__ ((weak));
extern char *scalar_strrchr (const char *, int) __attribute__ ((weak));
extern char *scalar_strchrnul (const char *, int) __attribute__ ((weak));
extern size_t scalar_strnlen (const char *, size_t) __attribute__ ((weak));
extern char *scalar_strcat (char *, const char *) __attribute__ ((weak));
extern char *scalar_strncat (char *, const char *, size_t) __attribute__ ((weak));
extern char *scalar_strncpy (char *, const char *, size_t) __attribute__ ((weak));
extern char *scalar_strstr (const char *, const char *) __attribute__ ((weak));
extern int scalar_strcasecmp (const char *, const char *) __attribute__ ((weak));
extern int scalar_strncasecmp (const char *, const char *, size_t) __attribute__ ((weak));
extern void *scalar_memchr (const void *, int, size_t) __attribute__ ((weak));
extern void *scalar_memrchr (const void *, int, size_t) __attribute__ ((weak));
extern void *scalar_memset (void *, int, size_t) __attribute__ ((weak));

/* ------------------------------------------------------------------ */
/* Reported results.                                                   */
/* ------------------------------------------------------------------ */
void t_begin (const char *fn);
void t_end (void);
void t_case (const char *fmt, ...) __attribute__ ((format (printf, 1, 2)));
void t_fail (const char *fmt, ...) __attribute__ ((format (printf, 1, 2)));
void t_warn (const char *fmt, ...) __attribute__ ((format (printf, 1, 2)));

long t_checks (void);
long t_fails (void);
long t_total_checks (void);
long t_total_fails (void);
void t_banner (void);
unsigned long t_vlen_bits (void);
int t_exit_code (void);

/* ------------------------------------------------------------------ */
/* Buffers.  Each region is a separate page-aligned mapping so that the  */
/* same offset inside two regions has the same alignment.              */
/* ------------------------------------------------------------------ */
unsigned char *ro_base (void);       /* shared read-only source data     */
unsigned char *d1_base (void);       /* destination for the RVV routine  */
unsigned char *d2_base (void);       /* destination for the glibc routine*/
unsigned char *m1_base (void);       /* memmove/overlap, RVV             */
unsigned char *m2_base (void);       /* memmove/overlap, glibc           */
unsigned char *s1_base (void);       /* spare string area 1              */
unsigned char *s2_base (void);       /* spare string area 2              */
size_t page_size (void);

#define WIN_OFF 4096                 /* payload area starts here         */
#define WIN_PAD 128                  /* canary bytes before the payload  */

/* Pointer to a payload of alignment (0..15) inside a region.  */
static inline unsigned char *
win (unsigned char *region, size_t align)
{
  return region + WIN_OFF + WIN_PAD + align;
}

/* Length of the window to compare: everything from region+WIN_OFF up to
   WIN_PAD bytes past the end of the payload.  */
#define WIN_LEN(n) (WIN_PAD + 16 + (n) + WIN_PAD)

void canary_fill (unsigned char *window, size_t len, unsigned seed);

/* Compare two destination windows (canary + payload + canary) and report the
   first differing byte relative to the payload start.  */
void chk_win (const char *what, const unsigned char *w1, const unsigned char *w2,
	      size_t n, size_t align);

/* ------------------------------------------------------------------ */
/* Guard pages: catch a routine that reads or writes one byte too far. */
/* ------------------------------------------------------------------ */
unsigned char *guard_hi (void);   /* last usable byte is the page end   */
unsigned char *guard_lo (void);   /* first usable byte is a page start  */
void guard_free (unsigned char *p);

/* Run one call while watching for SIGSEGV/SIGBUS.  Returns the normalised
   result and sets *faulted when the call touched the guard page.  */
long guard_call (int impl, int fk, void *p1, const void *p2, size_t n, int c,
		 int *faulted);

/* ------------------------------------------------------------------ */
/* Dispatch: a single call site for every routine under test.          */
/* ------------------------------------------------------------------ */
enum fk
{
  FK_strlen, FK_strnlen, FK_strchr, FK_strchrnul, FK_strrchr,
  FK_strcmp, FK_strncmp, FK_strcpy, FK_strncpy, FK_strcat, FK_strncat,
  FK_strspn, FK_strcspn, FK_strpbrk, FK_strstr,
  FK_strcasecmp, FK_strncasecmp,
  FK_memcpy, FK_memmove, FK_memccpy, FK_mempcpy,
  FK_memchr, FK_memrchr, FK_memcmp, FK_rawmemchr, FK_memset,
  FK_COUNT
};

#define IMPL_RVV 0
#define IMPL_LIBC 1

/* Argument convention: p1 is the pointer the result is an offset into
   (destination or haystack), p2 the second pointer, n the length, c the
   character argument.  Pointer results are normalised to an offset from p1;
   NULL becomes LONG_MIN so it compares unequal to any real pointer.  */
long fcall (int impl, int fk, void *p1, const void *p2, size_t n, int c);
const char *fk_name (int fk);

/* ------------------------------------------------------------------ */
/* Data patterns.                                                      */
/* ------------------------------------------------------------------ */
unsigned char datach (int variant, size_t i);
unsigned char strch (int variant, size_t i);

/* Fill p[0..len) then a NUL.  */
void make_str (unsigned char *p, size_t len, int variant);
/* Fill p[0..len) (no terminator).  */
void make_data (unsigned char *p, size_t len, int variant);

/* ------------------------------------------------------------------ */
/* Comparisons: ours vs the platform (glibc) reference.               */
/* ------------------------------------------------------------------ */
void chk_long (const char *what, long ours, long ref);
void chk_size (const char *what, size_t ours, size_t ref);
void chk_byte (const char *what, unsigned ours, unsigned ref);
void chk_buf (const char *what, const unsigned char *ours,
	      const unsigned char *ref, size_t n);
void chk_ptr (const char *what, const void *ours, const void *ourbase,
	      const void *ref, const void *refbase);

/* Formatting helper for pointers/offsets in failure messages.  */
void fmt_off (char *buf, size_t buflen, long off);

#define ALIGN_COUNT 16

/* Large lengths used in addition to 0..200.  */
extern const size_t big_lens[];
extern const int n_big_lens;

#endif /* HARNESS_H */
