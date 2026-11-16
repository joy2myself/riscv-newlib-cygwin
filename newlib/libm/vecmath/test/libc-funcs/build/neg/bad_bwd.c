/* Negative control #3: always copies backward, so overlapping dst < src is
   corrupted.  Specifically exercises the large-n backward cases in move_one
   and the dir=dst<src page-boundary cases in edge_memmove. */
#include <stddef.h>
void *
rvv_memmove (void *dst, const void *src, size_t n)
{
  unsigned char *d = (unsigned char *) dst;
  const unsigned char *s = (const unsigned char *) src;
  size_t i;
  for (i = n; i > 0; i--)
    d[i - 1] = s[i - 1];
  return dst;
}
