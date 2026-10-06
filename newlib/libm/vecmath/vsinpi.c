/* vsinpi.c -- array API entries for sinpi / cospi / tanpi / sincospi.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract and the SYCL/C23 special-value
   semantics these follow.  */

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
vsinpi (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (sinpi) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vcospi (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (cospi) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vtanpi (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (tanpi) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vsincospi (double *__restrict sy, double *__restrict cy,
	   const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, sy += vl, cy += vl)
    {
      vm_f_t vx, vs, vc;
      vl = VM_VSETVL (avl);
      vx = VM_VLE64 (x, vl);
      VM_SINCOSPI (vs, vc, vx, VM_PI_ALL (vl), vl);
      VM_VSE64 (sy, vs, vl);
      VM_VSE64 (cy, vc, vl);
    }
}

#endif /* __riscv_vector */
