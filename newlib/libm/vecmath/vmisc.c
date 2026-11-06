/* vmisc.c -- array API entries for sqrt / rsqrt / cbrt.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  Without the V extension this file
   is empty and vfallback.c provides the same symbols instead.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>

#define VM_LMUL 4	/* native LMUL for the simple root kernels.  */
#include <vecmath_rvvcompat.h>
#include "vecmath_local.h"
#include <k_misc.inc>

void
vsqrt (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL(avl);
      VM_VSE64(y, VM_KFN(sqrt)(VM_VLE64(x, vl), vl), vl);
    }
}

void
vrsqrt (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL(avl);
      VM_VSE64(y, VM_KFN(rsqrt)(VM_VLE64(x, vl), vl), vl);
    }
}

void
vcbrt (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL(avl);
      VM_VSE64(y, VM_KFN(cbrt)(VM_VLE64(x, vl), vl), vl);
    }
}

#endif /* __riscv_vector */
