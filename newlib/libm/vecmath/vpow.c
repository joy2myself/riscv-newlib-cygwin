/* vpow.c -- array API entry for pow.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  Every special case (x subnormal /
   zero / inf / NaN / negative, |y| outside the (2^-65, 2^63) window,
   and the overflow/underflow tails of y*log(x)) comes from the scalar
   libm pow, so the C99 special-value matrix and FP flags match.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 2	/* native LMUL for the pow kernel.  */
#include <vecmath_rvvcompat.h>
#include "vecmath_local.h"
#include <k_pow.inc>

void
vpow (double *__restrict z, const double *__restrict x,
      const double *__restrict y, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl, z += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (z, VM_KFN (pow_k) (VM_VLE64 (x, vl), VM_VLE64 (y, vl), vl,
				  pow), vl);
    }
}

#endif /* __riscv_vector */
