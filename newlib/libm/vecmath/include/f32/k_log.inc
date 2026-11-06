/* f32/k_log.inc -- binary32 log / log2 / log10 / log1p / log2p1 kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_log.inc)
   active at VM_LMUL == 2*VM32_LMUL.  */

VM32_W1F (log, log_k)
VM32_W1F (log2, log2_k)
VM32_W1F (log10, log10)
VM32_W1F (log2p1, log2p1)

/* log1p has the same tiny-input shortcut as expm1 (it returns x for
   |x| < 2^-54), so subnormal binary32 results need the explicit
   underflow flag.  */
static inline vm32_f_t
VM32_KFN (log1p) (vm32_f_t x, size_t vl, double (*fixup) (double))
{
  vm32_f_t y = VM32_NARROW (VM32_D64 (log1p_k) (VM32_WIDEN (x, vl), vl,
						fixup), vl);
  VM32_RAISE_UF_SUBNORMAL (y, vl);
  return y;
}
