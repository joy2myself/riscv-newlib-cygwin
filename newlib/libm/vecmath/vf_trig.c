/* vf_trig.c -- binary32 array API entries for the trigonometric and
   pi-scaled trigonometric families.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 8
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_trig.inc>

#define VM32_LMUL 4
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_trig.inc>

void
vsinf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (sin) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vcosf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (cos) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vtanf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (tan) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vsincosf (float *__restrict sy, float *__restrict cy,
	  const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, sy += vl, cy += vl)
    {
      vm32_f_t vx, vs, vc;
      vl = VM32_VSETVL (avl);
      vx = VM32_VLE (x, vl);
      VM32_SINCOS (vs, vc, vx, vl);
      VM32_VSE (sy, vs, vl);
      VM32_VSE (cy, vc, vl);
    }
}

void
vsinpif (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (sinpi) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vcospif (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (cospi) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vtanpif (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (tanpi) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vsincospif (float *__restrict sy, float *__restrict cy,
	    const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, sy += vl, cy += vl)
    {
      vm32_f_t vx, vs, vc;
      vl = VM32_VSETVL (avl);
      vx = VM32_VLE (x, vl);
      VM32_SINCOSPI (vs, vc, vx, vl);
      VM32_VSE (sy, vs, vl);
      VM32_VSE (cy, vc, vl);
    }
}

#endif /* __riscv_vector */
