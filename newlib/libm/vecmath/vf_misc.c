/* vf_misc.c -- binary32 array API entries for sqrt / rsqrt / cbrt / hypot.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.

   The binary32 entries evaluate through the binary64 kernels (see
   f32/wrap.h): sqrt alone is native, since vfsqrt.v on binary32 is
   already correctly rounded.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

/* ---- binary64 kernels this layer builds on (VM_LMUL = 2 * VM32_LMUL) */
#define VM_LMUL 4
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_misc.inc>

/* ---- binary32 layer ---- */
#define VM32_LMUL 2
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_misc.inc>

void
vsqrtf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (sqrt) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vrsqrtf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (rsqrt) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vcbrtf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (cbrt) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vhypotf (float *__restrict y, const float *__restrict ax,
	 const float *__restrict ay, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, ax += vl, ay += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (hypot) (VM32_VLE (ax, vl), VM32_VLE (ay, vl), vl,
				     hypot), vl);
    }
}

#endif /* __riscv_vector */
