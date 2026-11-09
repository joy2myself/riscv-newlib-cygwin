/* vf_hyperinv.c -- binary32 array API entries for asinh / acosh / atanh.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  The native binary32 kernels form
   their log arguments in f32 double-float and keep a per-lane scalar
   epilogue, so this translation unit no longer needs the binary64 layer.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM32_LMUL 2
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_log.inc>
#include <f32/k_hyperinv.inc>

void
vasinhf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (asinh) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vacoshf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (acosh) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vatanhf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (atanh) (VM32_VLE (x, vl), vl), vl);
    }
}

#endif /* __riscv_vector */
