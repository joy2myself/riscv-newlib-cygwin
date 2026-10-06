/* vexpm1.c -- array API entry for expm1.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 2	/* native LMUL for the expm1 kernel.  */
#include <vecmath_rvvcompat.h>
#include "vecmath_local.h"
#include <k_expm1.inc>

void
vexpm1 (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (expm1) (VM_VLE64 (x, vl), vl), vl);
    }
}

#endif /* __riscv_vector */
