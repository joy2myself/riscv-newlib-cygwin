/* vf_log.c -- binary32 array API entries for the logarithm family.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  The native binary32 kernels keep
   their own per-lane scalar epilogue, so this translation unit no longer
   needs the binary64 layer.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM32_LMUL 2
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_log.inc>

void
vlogf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (log) (VM32_VLE (x, vl), vl, log), vl);
    }
}

void
vlog2f (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (log2) (VM32_VLE (x, vl), vl, log2), vl);
    }
}

void
vlog10f (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (log10) (VM32_VLE (x, vl), vl, log10), vl);
    }
}

void
vlog1pf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (log1p) (VM32_VLE (x, vl), vl, log1p), vl);
    }
}

void
vlog2p1f (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (log2p1) (VM32_VLE (x, vl), vl, log1p), vl);
    }
}

#endif /* __riscv_vector */
