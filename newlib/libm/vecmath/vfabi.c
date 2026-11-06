/* vfabi.c -- RISC-V Vector Function ABI aliases for libvecmath.

   Exports the mangled entry points of the RISC-V Vector Function ABI
   (riscv-elf-psabi-doc, "Name Mangling for Vector Function"):
   _ZGV<lmul>Nx<params>_<name> for LMUL 1/2/4/8, VLA variant ("x"), no
   mask variant ("N"), parameter shape "v" (vector) / "vv" (two vectors).

   GCC has no -mveclibabi support for RISC-V, so nothing consumes these
   automatically; they are for future compilers and for hand-written SIMD
   code.  Each wrapper calls the same LMUL-generic kernel as the array
   API with vl = VLMAX, so results are bitwise identical to vsin()/vexp()
   and to the inline kernels of vecmath_rvv.h.

   Not exported (the mangling cannot express them): vsincos, vsincospi,
   vlgamma_r.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

/* Local paste helpers: the compat layer undefines its own on every
   re-inclusion.  */
#define VF_C2_(a, b) a##b
#define VF_C2(a, b) VF_C2_(a, b)
#define VF_C3_(a, b, c) a##b##c
#define VF_C3(a, b, c) VF_C3_(a, b, c)

static double
vfabi_exp10_scalar (double x)
{
  return pow (10.0, x);
}

/* Wrapper generators: ABI is the exported (mangled) name, KERN is the
   kernel identifier, FN the scalar fixup for special lanes.  */
#define VFABI_U(K, ABI, KERN)						\
  vfloat64m##K##_t VF_C2(VF_C3(_ZGV, K, Nxv_), ABI)			\
    (vfloat64m##K##_t x)						\
  {									\
    return VF_C2(VF_C3(__vecmath_, KERN, f64m), K)			\
      (x, VF_C2(__riscv_vsetvlmax_e64m, K) ());				\
  }
#define VFABI_UF(K, ABI, KERN, FN)					\
  vfloat64m##K##_t VF_C2(VF_C3(_ZGV, K, Nxv_), ABI)			\
    (vfloat64m##K##_t x)						\
  {									\
    return VF_C2(VF_C3(__vecmath_, KERN, f64m), K)			\
      (x, VF_C2(__riscv_vsetvlmax_e64m, K) (), FN);			\
  }
#define VFABI_B(K, ABI, KERN, FN)					\
  vfloat64m##K##_t VF_C2(VF_C3(_ZGV, K, Nxvv_), ABI)			\
    (vfloat64m##K##_t x, vfloat64m##K##_t y)				\
  {									\
    return VF_C2(VF_C3(__vecmath_, KERN, f64m), K)			\
      (x, y, VF_C2(__riscv_vsetvlmax_e64m, K) (), FN);			\
  }

/* One instantiation block per LMUL.  */
#define VM_FABI_ALL(K)							\
  VFABI_U (K, sqrt, sqrt)						\
  VFABI_U (K, rsqrt, rsqrt)						\
  VFABI_U (K, cbrt, cbrt)						\
  VFABI_UF (K, exp, exp, exp)						\
  VFABI_UF (K, exp2, exp2, exp2)					\
  VFABI_UF (K, exp10, exp10, vfabi_exp10_scalar)			\
  VFABI_U (K, expm1, expm1)						\
  VFABI_UF (K, log, log_k, log)						\
  VFABI_UF (K, log2, log2_k, log2)					\
  VFABI_UF (K, log10, log10, log)					\
  VFABI_UF (K, log1p, log1p_k, log1p)					\
  VFABI_UF (K, log2p1, log2p1, log1p)					\
  VFABI_B (K, pow, pow_k, pow)						\
  VFABI_U (K, sin, sin)							\
  VFABI_U (K, cos, cos)							\
  VFABI_U (K, tan, tan)							\
  VFABI_U (K, sinpi, sinpi)						\
  VFABI_U (K, cospi, cospi)						\
  VFABI_U (K, tanpi, tanpi)						\
  VFABI_U (K, sinh, sinh)						\
  VFABI_UF (K, cosh, cosh, cosh)					\
  VFABI_U (K, tanh, tanh)						\
  VFABI_U (K, erf, erf)							\
  VFABI_U (K, erfc, erfc)						\
  VFABI_U (K, tgamma, tgamma)						\
  VFABI_U (K, lgamma, lgamma)

#define VM_LMUL 1
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_misc.inc>
#include <k_exp.inc>
#include <k_expm1.inc>
#include <k_log.inc>
#include <k_pow.inc>
#include <k_trig.inc>
#include <k_hyper.inc>
#include <k_erf.inc>
#include <k_gamma.inc>
VM_FABI_ALL (1)

#define VM_LMUL 2
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include <k_fixup.inc>
#include <k_misc.inc>
#include <k_exp.inc>
#include <k_expm1.inc>
#include <k_log.inc>
#include <k_pow.inc>
#include <k_trig.inc>
#include <k_hyper.inc>
#include <k_erf.inc>
#include <k_gamma.inc>
VM_FABI_ALL (2)

#define VM_LMUL 4
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include <k_fixup.inc>
#include <k_misc.inc>
#include <k_exp.inc>
#include <k_expm1.inc>
#include <k_log.inc>
#include <k_pow.inc>
#include <k_trig.inc>
#include <k_hyper.inc>
#include <k_erf.inc>
#include <k_gamma.inc>
VM_FABI_ALL (4)

#define VM_LMUL 8
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include <k_fixup.inc>
#include <k_misc.inc>
#include <k_exp.inc>
#include <k_expm1.inc>
#include <k_log.inc>
#include <k_pow.inc>
#include <k_trig.inc>
#include <k_hyper.inc>
#include <k_erf.inc>
#include <k_gamma.inc>
VM_FABI_ALL (8)

#endif /* __riscv_vector */
