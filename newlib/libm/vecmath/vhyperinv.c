/* vhyperinv.c -- array API entries for asinh / acosh / atanh.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 2	/* native LMUL for the log kernels.  */
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_log.inc>
#include <k_hyperinv.inc>

void
vasinh (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (asinh) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vacosh (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (acosh) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vatanh (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (atanh) (VM_VLE64 (x, vl), vl), vl);
    }
}

#endif /* __riscv_vector */
