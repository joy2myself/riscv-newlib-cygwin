/* vgamma.c -- array API entries for tgamma / lgamma / lgamma_r.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 2	/* native LMUL for the gamma kernels.  */
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_misc.inc>
#include <k_log.inc>
#include <k_exp.inc>
#include <k_trig.inc>
#include <k_gamma.inc>

void
vlgamma (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      vm_f_t h, l, vres;
      VM_GAM_LGAMMA (h, l, VM_VLE64 (x, vl), vl);
      VM_DD_COLLAPSE (vres, h, l, vl);
      VM_VSE64 (y, vres, vl);
    }
}

void
vlgamma_r (double *__restrict y, int *__restrict sign,
	   const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl, sign += vl)
    {
      vl = VM_VSETVL (avl);
      vm_f_t h, l, s, vres;
      vm_i64_t si;
      VM_GAM_LGAMMA_R (h, l, s, VM_VLE64 (x, vl), vl);
      VM_DD_COLLAPSE (vres, h, l, vl);
      VM_VSE64 (y, vres, vl);
      si = VM_VFCVT_X_F_V_RM (s, __RISCV_FRM_RTZ, vl);
      VM_VSE32 (sign, VM_VNCVT_X_X_W (si, vl), vl);
    }
}

void
vtgamma (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      vm_f_t h, l, vres;
      VM_GAM_TGAMMA (h, l, VM_VLE64 (x, vl), vl);
      VM_DD_COLLAPSE (vres, h, l, vl);
      VM_VSE64 (y, vres, vl);
    }
}

#endif /* __riscv_vector */
