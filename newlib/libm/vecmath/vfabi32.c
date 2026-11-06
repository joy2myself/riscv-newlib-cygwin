/* vfabi32.c -- RISC-V Vector Function ABI aliases for the binary32
   libvecmath entry points.

   Part of the newlib RISC-V vector math library.  Mirrors vfabi.c: the
   same 26 single-output functions, the same mangling scheme
   _ZGV<lmul>Nx<params>_<name>, with the binary32 function names (the
   "f" suffix distinguishes them).

   Each wrapper calls the same binary32 kernel as the array API and is
   bitwise identical to vsinf()/vexpf() and friends.

   LMUL 8 needs 32 binary64 register groups to widen into, one past the
   architectural maximum of 8, so the LMUL 8 wrappers split their operand
   into two LMUL 4 halves.  The ABI uses vl = VLMAX, so the split is
   exact.

   Not exported (the mangling cannot express them): vsincosf,
   vsincospif, vlgamma_rf.  */

#include <vecmath.h>

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <math.h>

/* Local paste helpers: the compat layers undefine their own on every
   re-inclusion.  */
#define VF32_C2_(a, b) a##b
#define VF32_C2(a, b) VF32_C2_(a, b)
#define VF32_C3_(a, b, c) a##b##c
#define VF32_C3(a, b, c) VF32_C3_(a, b, c)

/* Name of a binary32 kernel at LMUL K: VF32_KN(4, sin) ->
   __vecmath_sinf32m4.  */
#define VF32_KN(K, n) VF32_C2(VF32_C3(__vecmath_, n, f32m), K)

static double
vfabi32_exp10_scalar (double x)
{
  return pow (10.0, x);
}

/* ---- binary64 kernels the wrappers build on: LMUL 2, 4, 8 ---- */
#define VM_LMUL 2
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include "vecmath_local.h"
#include <k_misc.inc>
#include <k_hypot.inc>
#include <k_exp.inc>
#include <k_expm1.inc>
#include <k_log.inc>
#include <k_pow.inc>
#include <k_trig.inc>
#include <k_hyper.inc>
#include <k_erf.inc>
#include <k_gamma.inc>

#define VM_LMUL 4
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include <k_fixup.inc>
#include <k_misc.inc>
#include <k_hypot.inc>
#include <k_exp.inc>
#include <k_expm1.inc>
#include <k_log.inc>
#include <k_pow.inc>
#include <k_trig.inc>
#include <k_hyper.inc>
#include <k_erf.inc>
#include <k_gamma.inc>

#define VM_LMUL 8
#include <vecmath_rvvcompat.h>
#include <vecmath_dd.h>
#include <k_fixup.inc>
#include <k_misc.inc>
#include <k_hypot.inc>
#include <k_exp.inc>
#include <k_expm1.inc>
#include <k_log.inc>
#include <k_pow.inc>
#include <k_trig.inc>
#include <k_hyper.inc>
#include <k_erf.inc>
#include <k_gamma.inc>

/* ---- binary32 kernels: LMUL 1, 2, 4 ---- */
#define VM32_LMUL 1
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_misc.inc>
#include <f32/k_exp.inc>
#include <f32/k_log.inc>
#include <f32/k_pow.inc>
#include <f32/k_trig.inc>
#include <f32/k_hyper.inc>
#include <f32/k_erf.inc>
#include <f32/k_gamma.inc>
#undef VM32_LMUL
#define VM32_COMPAT_UNDEF_ONLY 1
#include <f32/compat.h>

#define VM32_LMUL 2
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_misc.inc>
#include <f32/k_exp.inc>
#include <f32/k_log.inc>
#include <f32/k_pow.inc>
#include <f32/k_trig.inc>
#include <f32/k_hyper.inc>
#include <f32/k_erf.inc>
#include <f32/k_gamma.inc>
#undef VM32_LMUL
#define VM32_COMPAT_UNDEF_ONLY 1
#include <f32/compat.h>

#define VM32_LMUL 4
#include <f32/compat.h>
#include <f32/wrap.h>
#include <f32/k_misc.inc>
#include <f32/k_exp.inc>
#include <f32/k_log.inc>
#include <f32/k_pow.inc>
#include <f32/k_trig.inc>
#include <f32/k_hyper.inc>
#include <f32/k_erf.inc>
#include <f32/k_gamma.inc>
#undef VM32_LMUL
#define VM32_COMPAT_UNDEF_ONLY 1
#include <f32/compat.h>

/* ---- wrapper generators ------------------------------------------ */

#define VF32_U(K, ABI, KERN)						\
  vfloat32m##K##_t VF32_C2(VF32_C3(_ZGV, K, Nxv_), ABI)			\
    (vfloat32m##K##_t x)						\
  {									\
    return VF32_KN (K, KERN) (x, VF32_C2(__riscv_vsetvlmax_e32m, K) ()); \
  }
#define VF32_UF(K, ABI, KERN, FN)					\
  vfloat32m##K##_t VF32_C2(VF32_C3(_ZGV, K, Nxv_), ABI)			\
    (vfloat32m##K##_t x)						\
  {									\
    return VF32_KN (K, KERN) (x, VF32_C2(__riscv_vsetvlmax_e32m, K) (), FN); \
  }
#define VF32_B(K, ABI, KERN, FN)					\
  vfloat32m##K##_t VF32_C2(VF32_C3(_ZGV, K, Nxvv_), ABI)		\
    (vfloat32m##K##_t x, vfloat32m##K##_t y)				\
  {									\
    return VF32_KN (K, KERN) (x, y, VF32_C2(__riscv_vsetvlmax_e32m, K) (), \
			      FN);					\
  }

/* LMUL 8 shapes: split the operand, run the LMUL 4 kernels on both
   halves (each at its own VLMAX, so the two halves exactly tile the
   LMUL 8 group), and rejoin.  */
#define VF32_U8(ABI, KERN)						\
  vfloat32m8_t VF32_C2(_ZGV8Nxv_, ABI) (vfloat32m8_t x)			\
  {									\
    size_t vl = __riscv_vsetvlmax_e32m4 ();				\
    vfloat32m4_t lo = VF32_KN (4, KERN)					\
      (__riscv_vget_v_f32m8_f32m4 (x, 0), vl);				\
    vfloat32m4_t hi = VF32_KN (4, KERN)					\
      (__riscv_vget_v_f32m8_f32m4 (x, 1), vl);				\
    return __riscv_vset_v_f32m4_f32m8 (				\
      __riscv_vset_v_f32m4_f32m8 (x, 0, lo), 1, hi);			\
  }
#define VF32_UF8(ABI, KERN, FN)						\
  vfloat32m8_t VF32_C2(_ZGV8Nxv_, ABI) (vfloat32m8_t x)			\
  {									\
    size_t vl = __riscv_vsetvlmax_e32m4 ();				\
    vfloat32m4_t lo = VF32_KN (4, KERN)					\
      (__riscv_vget_v_f32m8_f32m4 (x, 0), vl, FN);			\
    vfloat32m4_t hi = VF32_KN (4, KERN)					\
      (__riscv_vget_v_f32m8_f32m4 (x, 1), vl, FN);			\
    return __riscv_vset_v_f32m4_f32m8 (				\
      __riscv_vset_v_f32m4_f32m8 (x, 0, lo), 1, hi);			\
  }
#define VF32_B8(ABI, KERN, FN)						\
  vfloat32m8_t VF32_C2(_ZGV8Nxvv_, ABI)					\
    (vfloat32m8_t x, vfloat32m8_t y)					\
  {									\
    size_t vl = __riscv_vsetvlmax_e32m4 ();				\
    vfloat32m4_t xl = __riscv_vget_v_f32m8_f32m4 (x, 0);		\
    vfloat32m4_t xh = __riscv_vget_v_f32m8_f32m4 (x, 1);		\
    vfloat32m4_t yl = __riscv_vget_v_f32m8_f32m4 (y, 0);		\
    vfloat32m4_t yh = __riscv_vget_v_f32m8_f32m4 (y, 1);		\
    vfloat32m4_t rl = VF32_KN (4, KERN) (xl, yl, vl, FN);		\
    vfloat32m4_t rh = VF32_KN (4, KERN) (xh, yh, vl, FN);		\
    return __riscv_vset_v_f32m4_f32m8 (				\
      __riscv_vset_v_f32m4_f32m8 (x, 0, rl), 1, rh);			\
  }

/* One instantiation block per LMUL.  */
#define VM_FABI32_ALL(K)						\
  VF32_U (K, sqrtf, sqrt)						\
  VF32_U (K, rsqrtf, rsqrt)						\
  VF32_U (K, cbrtf, cbrt)						\
  VF32_UF (K, expf, exp, exp)						\
  VF32_UF (K, exp2f, exp2, exp2)					\
  VF32_UF (K, exp10f, exp10, vfabi32_exp10_scalar)			\
  VF32_U (K, expm1f, expm1)						\
  VF32_UF (K, logf, log, log)						\
  VF32_UF (K, log2f, log2, log2)					\
  VF32_UF (K, log10f, log10, log)					\
  VF32_UF (K, log1pf, log1p, log1p)					\
  VF32_UF (K, log2p1f, log2p1, log1p)					\
  VF32_B (K, powf, pow, pow)						\
  VF32_U (K, sinf, sin)							\
  VF32_U (K, cosf, cos)							\
  VF32_U (K, tanf, tan)							\
  VF32_U (K, sinpif, sinpi)						\
  VF32_U (K, cospif, cospi)						\
  VF32_U (K, tanpif, tanpi)						\
  VF32_U (K, sinhf, sinh)						\
  VF32_UF (K, coshf, cosh, cosh)					\
  VF32_U (K, tanhf, tanh)						\
  VF32_U (K, erff, erf)							\
  VF32_U (K, erfcf, erfc)						\
  VF32_U (K, tgammaf, tgamma)						\
  VF32_U (K, lgammaf, lgamma)

VM_FABI32_ALL (1)
VM_FABI32_ALL (2)
VM_FABI32_ALL (4)

VF32_U8 (sqrtf, sqrt)
VF32_U8 (rsqrtf, rsqrt)
VF32_U8 (cbrtf, cbrt)
VF32_UF8 (expf, exp, exp)
VF32_UF8 (exp2f, exp2, exp2)
VF32_UF8 (exp10f, exp10, vfabi32_exp10_scalar)
VF32_U8 (expm1f, expm1)
VF32_UF8 (logf, log, log)
VF32_UF8 (log2f, log2, log2)
VF32_UF8 (log10f, log10, log)
VF32_UF8 (log1pf, log1p, log1p)
VF32_UF8 (log2p1f, log2p1, log1p)
VF32_B8 (powf, pow, pow)
VF32_U8 (sinf, sin)
VF32_U8 (cosf, cos)
VF32_U8 (tanf, tan)
VF32_U8 (sinpif, sinpi)
VF32_U8 (cospif, cospi)
VF32_U8 (tanpif, tanpi)
VF32_U8 (sinhf, sinh)
VF32_UF8 (coshf, cosh, cosh)
VF32_U8 (tanhf, tanh)
VF32_U8 (erff, erf)
VF32_U8 (erfcf, erfc)
VF32_U8 (tgammaf, tgamma)
VF32_U8 (lgammaf, lgamma)

#endif /* __riscv_vector */
