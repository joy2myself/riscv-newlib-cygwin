/* vexp.c -- array API entries for exp / exp2 / exp10.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  Guard lanes (overflow, underflow,
   inf/NaN, subnormal double-rounding) are computed by the scalar libm
   functions, so they are bit- and flag-identical to scalar exp/exp2.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 2	/* native LMUL for the table-driven exp family.  */
#include <vecmath_rvvcompat.h>
#include "vecmath_local.h"
#include <k_exp.inc>

static double
exp10_scalar (double x)
{
  return pow (10.0, x);
}

void
vexp (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (exp) (VM_VLE64 (x, vl), vl, exp), vl);
    }
}

void
vexp2 (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (exp2) (VM_VLE64 (x, vl), vl, exp2), vl);
    }
}

void
vexp10 (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (exp10) (VM_VLE64 (x, vl), vl, exp10_scalar), vl);
    }
}

#endif /* __riscv_vector */
