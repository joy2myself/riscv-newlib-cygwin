/* f32/k_misc.inc -- binary32 sqrt / rsqrt / cbrt / hypot kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_misc.inc,
   k_hypot.inc) active at VM_LMUL == 2*VM32_LMUL.  */

/* sqrt is a native binary32 kernel: vfsqrt.v is correctly rounded, so
   widening would only cost a pair of conversions.  */
static inline vm32_f_t
VM32_KFN (sqrt) (vm32_f_t x, size_t vl)
{
  return VM32_VFSQRT_V (x, vl);
}

VM32_W1 (rsqrt, rsqrt)
VM32_W1 (cbrt, cbrt)
VM32_W2F (hypot, hypot2)
