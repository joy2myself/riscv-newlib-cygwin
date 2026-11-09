/* vf_exp.c -- binary32 array API entries for exp / exp2 / exp10 / expm1.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 4	/* 2 * VM32_LMUL: this is the binary32 layer.  */
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_exp.inc>

#define VM32_LMUL 2
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_exp.inc>

static double
exp10_scalar (double x)
{
  return pow (10.0, x);
}

void
vexpf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (exp) (VM32_VLE (x, vl), vl, exp), vl);
    }
}

void
vexp2f (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (exp2) (VM32_VLE (x, vl), vl, exp2), vl);
    }
}

void
vexp10f (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (exp10) (VM32_VLE (x, vl), vl, exp10_scalar), vl);
    }
}

void
vexpm1f (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (expm1) (VM32_VLE (x, vl), vl), vl);
    }
}

#endif /* __riscv_vector */
