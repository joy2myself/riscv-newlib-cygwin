/* vecmath.h -- RISC-V RVV 1.0 vector math library (libvecmath) public API.

   Part of newlib.  Enabled by configuring with --enable-vecmath, which
   defines _WANT_VECMATH in newlib.h, when building for a RISC-V target.

   API model (array-style, VLA): every entry processes n elements,
   stripmined over the hardware VLEN internally, so one binary runs on
   any RVV 1.0 implementation (VLEN 128/256/512/1024...).

   Contract:
     - y (and second outputs) must not overlap the input arrays.
     - Pointers may have any alignment; n == 0 is legal.
     - errno is NEVER set (libmvec/VFABI semantics).  Floating-point
       exception flags (invalid, div-by-zero, overflow, underflow) are
       raised in the fcsr exactly as the scalar C99 functions would, per
       lane; FE_INEXACT accumulates from internal double-double
       arithmetic and is not meaningful.
     - Results are correctly rounded, or within 1 ulp of the infinitely
       precise result, for all inputs, except where C99/SYCL allow
       implementation-defined NaN payloads.  Signed zeros, infinities and
       NaN propagation follow IEEE-754 / C99 / SYCL conventions.
     - Bare-metal users: the RISC-V vector unit must be enabled
       (sstatus.VS != Off) before calling these functions; Linux user
       space does this automatically.

   Link with:  -lvecmath -lm -lc

   Greenfield functions with no scalar C counterpart follow SYCL
   semantics; for the pi-scaled ones these are the OpenCL C 3.0 edge
   cases (n is an integer, so n + 1/2 is exactly representable):
     sinpi(x)  = sin(pi*x)   (exact argument reduction for all finite x;
     cospi, tanpi, sincospi likewise; *pi(+-inf) = NaN with FE_INVALID;
     sinpi(+-n) = +-0, cospi(n) = (-1)^n, cospi(n + 1/2) = +0,
     tanpi(n) = copysign(0, x)*(-1)^n, tanpi(n + 1/2) = (-1)^n * +Inf
     with FE_DIVBYZERO)
     asinpi(x) = asin(x)/pi  (acospi, atanpi, atan2pi likewise)
     rsqrt(x)  = 1/sqrt(x)   (rsqrt(+-0) = +inf, rsqrt(+inf) = +0)
     log2p1(x) = log2(1+x)   (accurate as x -> 0; never forms 1+x)
     exp10(x)  = 10^x

   If VECMATH_WANT_KERNELS is defined and the RVV intrinsics are
   available, <vecmath_rvv.h> is pulled in, exposing the per-lane inline
   kernels __vecmath_<name>_f64m{1,2,4,8}(vfloat64mK_t, vl) for direct use
   in SIMD code, plus the double-double helpers.  */

#ifndef _VECMATH_H_
#define _VECMATH_H_

#include <stddef.h>

/* Pull in newlib.h when available so that the _WANT_VECMATH gate works;
   hosted (non-newlib) builds declare the API unconditionally.  */
#if defined(__has_include)
# if __has_include(<newlib.h>)
#  include <newlib.h>
# endif
#endif

#if defined(_WANT_VECMATH) || defined(VECMATH_STANDALONE) \
    || !defined(__NEWLIB_H__)

#ifdef __cplusplus
extern "C" {
#endif

/* ---- trigonometric ---- */
void vsin (double *__restrict y, const double *__restrict x, size_t n);
void vcos (double *__restrict y, const double *__restrict x, size_t n);
void vtan (double *__restrict y, const double *__restrict x, size_t n);
void vsincos (double *__restrict sy, double *__restrict cy,
	      const double *__restrict x, size_t n);

/* ---- pi-scaled trigonometric (SYCL) ---- */
void vsinpi (double *__restrict y, const double *__restrict x, size_t n);
void vcospi (double *__restrict y, const double *__restrict x, size_t n);
void vtanpi (double *__restrict y, const double *__restrict x, size_t n);
void vsincospi (double *__restrict sy, double *__restrict cy,
		const double *__restrict x, size_t n);

/* ---- inverse trigonometric ---- */
void vasin (double *__restrict y, const double *__restrict x, size_t n);
void vacos (double *__restrict y, const double *__restrict x, size_t n);
void vatan (double *__restrict y, const double *__restrict x, size_t n);
void vatan2 (double *__restrict y, const double *__restrict ay,
	     const double *__restrict ax, size_t n);

/* ---- pi-scaled inverse trigonometric (SYCL) ---- */
void vasinpi (double *__restrict y, const double *__restrict x, size_t n);
void vacospi (double *__restrict y, const double *__restrict x, size_t n);
void vatanpi (double *__restrict y, const double *__restrict x, size_t n);
void vatan2pi (double *__restrict y, const double *__restrict ay,
	       const double *__restrict ax, size_t n);

/* ---- hyperbolic ---- */
void vsinh (double *__restrict y, const double *__restrict x, size_t n);
void vcosh (double *__restrict y, const double *__restrict x, size_t n);
void vtanh (double *__restrict y, const double *__restrict x, size_t n);
void vasinh (double *__restrict y, const double *__restrict x, size_t n);
void vacosh (double *__restrict y, const double *__restrict x, size_t n);
void vatanh (double *__restrict y, const double *__restrict x, size_t n);

/* ---- exponential / logarithmic ---- */
void vexp (double *__restrict y, const double *__restrict x, size_t n);
void vexp2 (double *__restrict y, const double *__restrict x, size_t n);
void vexp10 (double *__restrict y, const double *__restrict x, size_t n);
void vexpm1 (double *__restrict y, const double *__restrict x, size_t n);
void vlog (double *__restrict y, const double *__restrict x, size_t n);
void vlog2 (double *__restrict y, const double *__restrict x, size_t n);
void vlog10 (double *__restrict y, const double *__restrict x, size_t n);
void vlog1p (double *__restrict y, const double *__restrict x, size_t n);
void vlog2p1 (double *__restrict y, const double *__restrict x, size_t n);

/* ---- power / root ---- */
void vpow (double *__restrict y, const double *__restrict ax,
	   const double *__restrict bx, size_t n);
void vsqrt (double *__restrict y, const double *__restrict x, size_t n);
void vrsqrt (double *__restrict y, const double *__restrict x, size_t n);
void vcbrt (double *__restrict y, const double *__restrict x, size_t n);
void vhypot (double *__restrict y, const double *__restrict ax,
	     const double *__restrict bx, size_t n);

/* ---- special functions (double-double based, <= 1 ulp) ---- */
void verf (double *__restrict y, const double *__restrict x, size_t n);
void verfc (double *__restrict y, const double *__restrict x, size_t n);
void vtgamma (double *__restrict y, const double *__restrict x, size_t n);
void vlgamma (double *__restrict y, const double *__restrict x, size_t n);
/* vlgamma_r: like vlgamma, additionally writing sign(gamma(x)) per lane
   (+1/-1) into sign[].  No signgam global is ever touched.  */
void vlgamma_r (double *__restrict y, int *__restrict sign,
		const double *__restrict x, size_t n);

#ifdef __cplusplus
}
#endif

#if defined(VECMATH_WANT_KERNELS) && defined(__riscv_vector)
#include <vecmath_rvv.h>
#endif

#endif /* _WANT_VECMATH || VECMATH_STANDALONE || !__NEWLIB_H__ */

#endif /* _VECMATH_H_ */
