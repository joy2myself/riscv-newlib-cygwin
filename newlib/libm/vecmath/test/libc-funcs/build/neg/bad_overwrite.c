/* Negative control #2: correct copy but one byte overwritten past the end. */
#include <stddef.h>
void *
rvv_memmove (void *dst, const void *src, size_t n)
{
  unsigned char *d = (unsigned char *) dst;
  const unsigned char *s = (const unsigned char *) src;
  size_t i;
  if (d > s)
    for (i = n; i > 0; i--)
      d[i - 1] = s[i - 1];
  else
    for (i = 0; i < n; i++)
      d[i] = s[i];
  if (n)
    d[n] = 0x5a;                 /* over-write one byte past the object */
  return dst;
}
