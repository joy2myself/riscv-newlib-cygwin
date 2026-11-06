/* vecmath_rvv.h -- inline per-lane RVV kernels for libvecmath.

   Installed header.  Include with VECMATH_WANT_KERNELS (via <vecmath.h>)
   or directly, from a translation unit compiled with the V extension.

   Exposes, for every kernel <name> and LMUL K in {1,2,4,8}:

     static inline vfloat64mK_t __vecmath_<name>_f64mK(vfloat64mK_t x,
                                                       size_t vl);

   plus the double-double type and helpers per LMUL:

     vm_dd_t_K { vfloat64mK_t hi, lo; }   (aliased vm_dd_t while active)
     __vecmath_dd_add_f64mK, __vecmath_dd_mul_f64mK, ...

   Kernels are strictly vl-respecting (no cross-lane assumptions, no
   tail-lane writes) and VLEN-agnostic.  The library's array API entries
   and its VFABI aliases (_ZGV...) are thin wrappers around these same
   kernel bodies, so all three API layers are bitwise identical.

   This header instantiates the kernels at all four LMULs; it leaves the
   VM_* compat macro namespace clean after inclusion (all helper macros
   are undefined again), so it can coexist with user code.  */

#ifndef _VECMATH_RVV_H_
#define _VECMATH_RVV_H_

#if !defined(__riscv_vector)
#error "vecmath_rvv.h requires a V-extension compilation (-march=...v)"
#endif

#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

/* vecmath_local.h constants are needed by the kernel templates; it is not
   installed next to this header in the sysroot, so replicate the small
   constant set here under the same names when building out-of-tree.  */
#if defined(VECMATH_IN_TREE)
#include "vecmath_local.h"
#else
#ifndef VM_PINF_U64
#define VM_SIGNBIT_U64	0x8000000000000000ULL
#define VM_ABSMASK_U64	0x7fffffffffffffffULL
#define VM_HIGHWORD_U64	0xffffffff00000000ULL
#define VM_PINF_U64	0x7ff0000000000000ULL
#define VM_NINF_U64	0xfff0000000000000ULL
#define VM_CLS_NEGINF	(1ULL << 0)
#define VM_CLS_NEGNORM	(1ULL << 1)
#define VM_CLS_NEGSUB	(1ULL << 2)
#define VM_CLS_NEGZERO	(1ULL << 3)
#define VM_CLS_POSZERO	(1ULL << 4)
#define VM_CLS_POSSUB	(1ULL << 5)
#define VM_CLS_POSNORM	(1ULL << 6)
#define VM_CLS_POSINF	(1ULL << 7)
#define VM_CLS_SNAN	(1ULL << 8)
#define VM_CLS_QNAN	(1ULL << 9)
#define VM_CLS_NAN	(VM_CLS_SNAN | VM_CLS_QNAN)
#define VM_CLS_INF	(VM_CLS_NEGINF | VM_CLS_POSINF)
#define VM_CLS_ZERO	(VM_CLS_NEGZERO | VM_CLS_POSZERO)
#define VM_CLS_SUB	(VM_CLS_NEGSUB | VM_CLS_POSSUB)
#endif
#ifndef VM_VINF
#define VM_VINF(vl)	VM_FROM_BITS(VM_VMV_V_X(VM_PINF_U64, vl))
#endif
#endif

#define VM_LMUL 1
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include <k_misc.inc>
#undef VM_LMUL

#define VM_LMUL 2
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include <k_misc.inc>
#undef VM_LMUL

#define VM_LMUL 4
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include <k_misc.inc>
#undef VM_LMUL

#define VM_LMUL 8
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include <k_misc.inc>
#undef VM_LMUL

/* Final cleanup pass: leave no VM_* macros defined.  */
#define VM_COMPAT_UNDEF_ONLY 1
#include <vecmath_rvvcompat.h>

#endif /* _VECMATH_RVV_H_ */
