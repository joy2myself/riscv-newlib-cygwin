/* vf_pow.c -- binary32 array API entry for pow.

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
#include <k_pow.inc>

#define VM32_LMUL 2
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_pow.inc>

void
vpowf (float *__restrict y, const float *__restrict ax,
       const float *__restrict bx, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, ax += vl, bx += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (pow) (VM32_VLE (ax, vl), VM32_VLE (bx, vl), vl,
				   pow), vl);
    }
}

#endif /* __riscv_vector */
