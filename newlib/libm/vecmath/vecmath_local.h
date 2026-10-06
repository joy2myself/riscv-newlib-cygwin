/* vecmath_local.h -- internal helpers for the libvecmath RVV kernels.

   Part of the newlib RISC-V vector math library.  NOT an installed header.

   Requires vecmath_rvvcompat.h to be included first (VM_LMUL active).  */

#ifndef VECMATH_LOCAL_H
#define VECMATH_LOCAL_H

#include <stdint.h>
#include <stddef.h>

/* IEEE-754 binary64 bit constants.  */
#define VM_SIGNBIT_U64	0x8000000000000000ULL
#define VM_ABSMASK_U64	0x7fffffffffffffffULL
#define VM_HIGHWORD_U64	0xffffffff00000000ULL
#define VM_PINF_U64	0x7ff0000000000000ULL
#define VM_NINF_U64	0xfff0000000000000ULL
#define VM_EXP_SHIFT	52

/* vfclass.v result bits (RVV 1.0 spec Table "vfclass layout"):
   0:-inf 1:-normal 2:-subnormal 3:-0 4:+0 5:+subnormal 6:+normal
   7:+inf 8:sNaN 9:qNaN.  */
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
#define VM_CLS_POSFINITE (VM_CLS_POSNORM | VM_CLS_POSSUB)
#define VM_CLS_NEGFINITE (VM_CLS_NEGNORM | VM_CLS_NEGSUB)
#define VM_CLS_FINITE	(VM_CLS_POSFINITE | VM_CLS_NEGFINITE | VM_CLS_ZERO)

/* NaN-safe classification: RVV ordered float compares (vmflt/vmfle/
   vmfgt/vmfge) are signaling -- they raise FE_INVALID even for quiet NaN
   operands, per the IEEE signalingComparison semantics.  Only vmfeq/
   vmfne are quiet for qNaN (they still raise for sNaN).  Kernels must
   therefore classify lanes that may contain NaN via vfclass.v (which
   never raises), and reserve ordered compares for lanes already known
   non-NaN, or where the invalid flag is the expected outcome.  */

/* Constant vectors from bit patterns (avoids <math.h> in kernels).  */
#define VM_VINF(vl)	VM_FROM_BITS(VM_VMV_V_X(VM_PINF_U64, vl))
#define VM_VPINF_CONST(vl) VM_VINF(vl)

#ifdef VM_COMPAT_ACTIVE
/* Masked scalar-fixup epilogue (LMUL-generic template, name scoped by
   VM_KFN).  In-tree TUs get one instantiation at their LMUL.  */
#include <k_fixup.inc>
#endif /* VM_COMPAT_ACTIVE */

#endif /* VECMATH_LOCAL_H */
