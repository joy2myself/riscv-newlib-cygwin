/* f32/wrap.h -- binary32 kernels evaluated through binary64.

   Part of the newlib RISC-V vector math library (libvecmath).

   Every binary32 kernel in this layer is the same three-step shape:

     widen the f32mK operand group to f64m(2K)   (exact -- every binary32
     value is a binary64 value),
     evaluate the matching, already-verified binary64 kernel on it,
     narrow the f64m(2K) result back to f32mK  (one correctly-rounded
     conversion that also raises FE_OVERFLOW / FE_UNDERFLOW / FE_INEXACT
     exactly where the binary32 result's own rounding requires).

   This is the same computation newlib's scalar single-precision
   functions perform: the Arm optimized-routines ports that back expf,
   exp2f, logf, log2f, powf, sinf, cosf and sincosf all evaluate in
   `double_t` (= double) and round once at the end.  Evaluating in
   binary64 makes every binary32 entry correctly rounded to within
   0.5 ulp + 2^-29 ulp -- strictly inside the <= 1 ulp contract -- and
   inherits the binary64 layer's special-case and exception-flag
   semantics verbatim.

   LMUL: an f32mK group is 2K binary64 register groups, so K is capped
   at 4 (f64m8, the architectural maximum).  K = 1, 2, 4 are
   instantiated.

   Requires f32/compat.h included first (VM32_LMUL active), and, in the
   including translation unit, the binary64 kernels active at
   VM_LMUL == 2*VM32_LMUL.  */

/* RE-INCLUDABLE: like the compat layer, every macro defined here is
   #undef'ed again on re-inclusion, so the wrappers can be instantiated
   once per VM32_LMUL from a single source of truth.  */

#if !defined(VM32_COMPAT_ACTIVE)
#error "include f32/compat.h before f32/wrap.h"
#endif

#undef VM32_WD_T
#undef VM32_WIDEN
#undef VM32_NARROW
#undef VM32_D64M
#undef VM32_D64
#undef VM32_W1
#undef VM32_W1F
#undef VM32_W2
#undef VM32_W2F
#undef VM32_RAISE_UF_SUBNORMAL

#if VM32_LMUL == 1
# define VM32_WD_T	  vfloat64m2_t
# define VM32_WIDEN(v, vl)  __riscv_vfwcvt_f_f_v_f64m2 (v, vl)
# define VM32_NARROW(v, vl) __riscv_vfncvt_f_f_w_f32m1 (v, vl)
# define VM32_D64M	  2
#elif VM32_LMUL == 2
# define VM32_WD_T	  vfloat64m4_t
# define VM32_WIDEN(v, vl)  __riscv_vfwcvt_f_f_v_f64m4 (v, vl)
# define VM32_NARROW(v, vl) __riscv_vfncvt_f_f_w_f32m2 (v, vl)
# define VM32_D64M	  4
#elif VM32_LMUL == 4
# define VM32_WD_T	  vfloat64m8_t
# define VM32_WIDEN(v, vl)  __riscv_vfwcvt_f_f_v_f64m8 (v, vl)
# define VM32_NARROW(v, vl) __riscv_vfncvt_f_f_w_f32m4 (v, vl)
# define VM32_D64M	  8
#else
# error "binary32 kernels are instantiated at VM32_LMUL 1, 2 or 4"
#endif

/* Name of the binary64 kernel at the matching LMUL, e.g.
   VM32_D64 (exp) -> __vecmath_expf64m4 when VM32_LMUL == 2.  */
#define VM32_D64(n) VM32_XP3 (__vecmath_, n, VM32_XP2 (f64m, VM32_D64M))

/* ---- wrapper shapes ---------------------------------------------- */

/* one input, no scalar fixup */
#define VM32_W1(name, kern)						\
  static inline vm32_f_t						\
  VM32_KFN (name) (vm32_f_t x, size_t vl)				\
  {									\
    return VM32_NARROW (VM32_D64 (kern) (VM32_WIDEN (x, vl), vl), vl);	\
  }

/* one input, scalar fixup for the guard lanes */
#define VM32_W1F(name, kern)						\
  static inline vm32_f_t						\
  VM32_KFN (name) (vm32_f_t x, size_t vl, double (*fixup) (double))	\
  {									\
    return VM32_NARROW (VM32_D64 (kern) (VM32_WIDEN (x, vl), vl, fixup), \
			vl);						\
  }

/* two inputs, no scalar fixup */
#define VM32_W2(name, kern)						\
  static inline vm32_f_t						\
  VM32_KFN (name) (vm32_f_t x, vm32_f_t y, size_t vl)			\
  {									\
    return VM32_NARROW (VM32_D64 (kern) (VM32_WIDEN (x, vl),		\
					 VM32_WIDEN (y, vl), vl), vl);	\
  }

/* two inputs, scalar fixup */
#define VM32_W2F(name, kern)						\
  static inline vm32_f_t						\
  VM32_KFN (name) (vm32_f_t x, vm32_f_t y, size_t vl,			\
		   double (*fixup) (double, double))			\
  {									\
    return VM32_NARROW (VM32_D64 (kern) (VM32_WIDEN (x, vl),		\
					 VM32_WIDEN (y, vl), vl, fixup), \
			vl);						\
  }

/* ---- masked underflow carrier ------------------------------------ */

/* Raise FE_UNDERFLOW (with FE_INEXACT) on the lanes whose binary32
   result is a nonzero subnormal.

   A binary64 evaluation can land EXACTLY on the binary32 subnormal grid,
   in which case the narrowing conversion is exact and raises nothing --
   while the binary32 operation itself would have been both tiny and
   inexact.  For expm1 and log1p that is the normal case for every
   subnormal input: their binary64 kernels short-circuit |x| below the
   binary64 tiny threshold and return x verbatim, which for a binary32
   argument is exact, where the binary32 evaluation would have had to
   round x + x^2/2 onto the much coarser binary32 grid.  The values
   themselves are already correct; only the flag is missing.

   The masked multiply of 2^-126 by 2^-126 flushes to zero with
   UNDERFLOW + INEXACT, and the masked add of that zero leaves the result
   untouched (its inactive lanes merge the old value).  This mirrors the
   explicit underflow raiser the binary64 log2p1 kernel carries.  */
#define VM32_RAISE_UF_SUBNORMAL(yo, vl) do {				\
  vm32_u_t vmuf__b = VM32_VAND_VX (VM32_BITS (yo), 0x7fffffffu, (vl));	\
  vm32_b_t vmuf__m = VM32_VMSLTU_VX (					\
    VM32_VSUB_VX (vmuf__b, 1, (vl)), 0x007fffffu, (vl));		\
  vm32_f_t vmuf__z = VM32_VFMUL_VF_MU (vmuf__m,				\
				       VM32_VFMV_V_F (0.0f, (vl)),	\
				       VM32_VFMV_V_F (0x1p-126f, (vl)),	\
				       0x1p-126f, (vl));			\
  (yo) = VM32_VFADD_VV_MU (vmuf__m, (yo), (yo), vmuf__z, (vl));		\
} while (0)
