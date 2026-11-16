/* Negative control #1: a memmove that always copies forward, so overlapping
   dst > src is corrupted.  Used only to prove the oracle still detects bugs. */
#include <stddef.h>
void *
rvv_memmove (void *dst, const void *src, size_t n)
{
  unsigned char *d = (unsigned char *) dst;
  const unsigned char *s = (const unsigned char *) src;
  size_t i;
  for (i = 0; i < n; i++)
    d[i] = s[i];
  return dst;
}
