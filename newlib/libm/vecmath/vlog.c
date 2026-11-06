/* vlog.c -- array API entries for log / log2 / log10 / log1p / log2p1.

   Part of the newlib RISC-V vector math library (libvecmath); see
   <vecmath.h> for the API contract.  Fixup lanes (x <= 0, subnormal,
   inf/NaN for log/log2; x <= -1 for log1p/log2p1) are computed by the
   scalar libm functions.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

#define VM_LMUL 2	/* native LMUL for the table-driven log family.  */
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_log.inc>

void
vlog (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (log_k) (VM_VLE64 (x, vl), vl, log), vl);
    }
}

void
vlog2 (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      /* (log2): newlib's math.h defines log2 as a macro.  */
      VM_VSE64 (y, VM_KFN (log2_k) (VM_VLE64 (x, vl), vl, (log2)), vl);
    }
}

void
vlog10 (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      /* fixup = scalar log: log10's special cases mirror log's.  */
      VM_VSE64 (y, VM_KFN (log10) (VM_VLE64 (x, vl), vl, log), vl);
    }
}

void
vlog1p (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      VM_VSE64 (y, VM_KFN (log1p_k) (VM_VLE64 (x, vl), vl, log1p), vl);
    }
}

void
vlog2p1 (double *__restrict y, const double *__restrict x, size_t n)
{
  size_t vl;
  for (size_t avl = n; avl > 0; avl -= vl, x += vl, y += vl)
    {
      vl = VM_VSETVL (avl);
      /* fixup = scalar log1p: log2p1 dd-scales every lane by 1/ln2.  */
      VM_VSE64 (y, VM_KFN (log2p1) (VM_VLE64 (x, vl), vl, log1p), vl);
    }
}

#endif /* __riscv_vector */
