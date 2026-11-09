/* vf_hyper.c -- binary32 array API entries for sinh / cosh / tanh.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 4
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_misc.inc>
#include <k_exp.inc>
#include <k_expm1.inc>
#include <k_hyper.inc>

#define VM32_LMUL 2
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_exp.inc>
#include <f32/k_hyper.inc>

void
vsinhf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (sinh) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vcoshf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (cosh) (VM32_VLE (x, vl), vl, cosh), vl);
    }
}

void
vtanhf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (tanh) (VM32_VLE (x, vl), vl), vl);
    }
}

#endif /* __riscv_vector */
