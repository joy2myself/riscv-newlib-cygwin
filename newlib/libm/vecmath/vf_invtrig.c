/* vf_invtrig.c -- binary32 array API entries for the inverse
   trigonometric and pi-scaled inverse trigonometric families.

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
#include <k_trig.inc>	/* VM_INVPI_HI/LO, used by k_invtrig.inc */
#include <k_invtrig.inc>

#define VM32_LMUL 4
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_invtrig.inc>

void
vasinf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (asin) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vacosf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (acos) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vatanf (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (atan) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vatan2f (float *__restrict y, const float *__restrict ay,
	 const float *__restrict ax, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, ay += vl, ax += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (atan2) (VM32_VLE (ay, vl), VM32_VLE (ax, vl), vl),
		vl);
    }
}

void
vasinpif (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (asinpi) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vacospif (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (acospi) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vatanpif (float *__restrict y, const float *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (atanpi) (VM32_VLE (x, vl), vl), vl);
    }
}

void
vatan2pif (float *__restrict y, const float *__restrict ay,
	   const float *__restrict ax, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, ay += vl, ax += vl, y += vl)
    {
      vl = VM32_VSETVL (avl);
      VM32_VSE (y, VM32_KFN (atan2pi) (VM32_VLE (ay, vl), VM32_VLE (ax, vl),
				       vl), vl);
    }
}

#endif /* __riscv_vector */
