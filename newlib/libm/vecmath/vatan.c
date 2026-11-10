/* vatan.c -- array API entries for atan / atan2 / atanpi and the
   asin/acos family.  atan2pi lives in its own translation unit because it
   wants LMUL 2 (its dd /pi tail spills at LMUL 1) while the rest of the
   family is faster at LMUL 1 on the VLEN 128 board.  

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 1
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_trig.inc>
#include <k_invtrig.inc>

void
vatan (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (atan) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vatan2 (double *__restrict y, const double *__restrict ay,
	const double *__restrict ax, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, ay += vl, ax += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (atan2) (VM_VLE64 (ay, vl), VM_VLE64 (ax, vl), vl),
		vl);
    }
}

void
vatanpi (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (atanpi) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vasin (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (asin) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vacos (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (acos) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vasinpi (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (asinpi) (VM_VLE64 (x, vl), vl), vl);
    }
}

void
vacospi (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (acospi) (VM_VLE64 (x, vl), vl), vl);
    }
}

#endif /* __riscv_vector */
