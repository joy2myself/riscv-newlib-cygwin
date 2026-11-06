/* vf_gamma.c -- binary32 array API entries for tgamma / lgamma / lgamma_r.

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
#include <k_misc.inc>
#include <k_log.inc>
#include <k_exp.inc>
#include <k_trig.inc>
#include <k_gamma.inc>

#define VM32_LMUL 2
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_gamma.inc>

void
vlgammaf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (lgamma) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vlgamma_rf (float *__restrict y, int *__restrict sign,
	    const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl, sign += vl)
    {
      vm32_f_t vres;
      vm32_i_t si;
      vl = VM32_VSETVL (avl);
      VM32_GAM_LGAMMA_R (vres, si, VM32_VLE (x, vl), vl);
      VM32_VSE (y, vres, vl);
      VM32_VSE_I (sign, si, vl);
    }
}

void
vtgammaf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (tgamma) (VM32_VLE (x, vl), vl), vl);
    }
}

#endif /* __riscv_vector */
