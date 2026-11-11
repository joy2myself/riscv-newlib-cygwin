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
  size_t avl = n;
  size_t vmax = VM_VSETVLMAX ();	/* e64m2: also sets vl = vmax */

  /* Fast loop.  Every iteration is one whole strip with a whole strip still
     behind it, so lgamma_r's sign output -- +1 on every lane, because the
     fast domain is x > 0 -- leaves as one e64 store of the packed constant
     in VM_GAM_SIGNPAIR_D (two int32 +1 per element, i.e. this strip's signs
     and the next strip's): vl never changes, the vtype is set once before
     the loop, and the loop carries no vsetvli, no e32 SEW change and no
     narrowing convert.  A strip with any lane outside the fast domain, or
     an int array that is only 4-byte aligned, is left to the general loop
     below (which is all a short call has too).  */
  if (((uintptr_t) sign & 7) == 0)
    {
      /* The fast path's whole sign output is this one constant vector.  */
      vm_f_t vpair = VM_VFMV_V_F (VM_GAM_SIGNPAIR_D, vmax);
      while (avl >= 2 * vmax)
	{
	  vm_f_t h, l, vres, vx = VM_VLE64 (x, vmax);
	  if (!VM_GAM_FASTALL (vx, vmax))
	    break;
	  VM_GAM_LFAST (h, l, vx, vmax);
	  VM_DD_COLLAPSE (vres, h, l, vmax);
	  VM_VSE64 (y, vres, vmax);
	  VM_VSE64 ((double *) (void *) sign, vpair, vmax);
	  avl -= vmax, x += vmax, y += vmax, sign += vmax;
	}
    }

  /* General loop: at most two strips when every lane was fast, the rest of
     the call after a slow strip.  */
  for (size_t vl; avl > 0; avl -= vl, x += vl, y += vl, sign += vl)
    {
      vl = VM_VSETVL (avl);
      vm_f_t h, l, vres;
      vm_f_t vx = VM_VLE64 (x, vl);
      if (VM_GAM_FASTALL (vx, vl))
	{
	  VM_GAM_LFAST (h, l, vx, vl);
	  VM_DD_COLLAPSE (vres, h, l, vl);
	  VM_VSE64 (y, vres, vl);
	  VM_VSE32 (sign, VM_GAM_SIGN_ONES32 (vl), vl);
	}
      else
	{
	  vm_f_t s;
	  VM_GAM_LGAMMA_R_SLOW (h, l, s, vx, vl);
	  VM_DD_COLLAPSE (vres, h, l, vl);
	  VM_VSE64 (y, vres, vl);
	  /* s is exactly +-1.0, so the sign fits straight out of its sign
	     bit: 0x3ff0... >> 63 is 0 and 0xbff0... >> 63 is -1, and ORing 1
	     gives +1 / -1 without touching the rounding mode (a vfcvt with an
	     explicit frm costs an fsrm/fsrmi pair around the strip).  */
	  VM_VSE32 (sign, VM_VNCVT_X_X_W (VM_GAM_SIGNBITS (s, vl), vl), vl);
	}
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
