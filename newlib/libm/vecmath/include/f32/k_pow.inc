/* f32/k_pow.inc -- binary32 pow kernel.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_pow.inc,
   and k_log.inc / k_exp.inc which it uses) active at
   VM_LMUL == 2*VM32_LMUL.  */

/* pow raises FE_UNDERFLOW for a subnormal result -- not only for an
   inexact one.  The binary64 kernel does the same for its own subnormal
   results (delegating to scalar pow), and the binary32 evaluation can
   land exactly on the binary32 subnormal grid, where the narrowing
   conversion would stay silent.  */
static inline vm32_f_t
VM32_KFN (pow) (vm32_f_t x, vm32_f_t y, size_t vl,
		double (*fixup) (double, double))
{
  vm32_f_t r = VM32_NARROW (VM32_D64 (pow_k) (VM32_WIDEN (x, vl),
					      VM32_WIDEN (y, vl), vl, fixup),
			    vl);
  VM32_RAISE_UF_SUBNORMAL (r, vl);
  return r;
}
