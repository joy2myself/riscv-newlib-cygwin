/* vhypot.c -- array API entry for hypot.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  Exponent extremes (|max| >= 2^500,
   < 2^-500, zero, subnormal and every non-finite input) are computed by
   the scalar hypot, so their results and flags match scalar libm.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 2
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_hypot.inc>

void
vhypot (double *__restrict y, const double *__restrict ax,
	const double *__restrict ay, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, ax += vl, ay += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (hypot2) (VM_VLE64 (ax, vl), VM_VLE64 (ay, vl), vl,
				   hypot), vl);
    }
}

#endif /* __riscv_vector */
