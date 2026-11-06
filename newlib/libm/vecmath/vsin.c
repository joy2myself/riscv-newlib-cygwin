/* vsin.c -- array API entries for sin / cos / tan / sincos.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 2
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_trig.inc>

void
vsin (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (sin) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vcos (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (cos) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vtan (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (tan) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vsincos (double *__restrict sy, double *__restrict cy,
	 const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, sy += vl, cy += vl)
    {
      vm_f_t vx, vs, vc;
      vl = VM_VSETVL (avl);
      vx = VM_VLE64 (x, vl);
      VM_SINCOS (vs, vc, vx, vl);
      VM_VSE64 (sy, vs, vl);
      VM_VSE64 (cy, vc, vl);
    }
}

#endif /* __riscv_vector */
