/* f32/k_exp.inc -- binary32 exp / exp2 / exp10 / expm1 kernels.

   LMUL-generic template: include with VM32_LMUL active after
   f32/compat.h and f32/wrap.h, with the binary64 kernels (k_exp.inc,
   k_expm1.inc) active at VM_LMUL == 2*VM32_LMUL.  */

VM32_W1F (exp, exp)
VM32_W1F (exp2, exp2)
VM32_W1F (exp10, exp10)

/* expm1 returns x verbatim below the binary64 tiny threshold, so every
   subnormal binary32 result is exact in binary64 and needs the explicit
   underflow flag (see VM32_RAISE_UF_SUBNORMAL).  */
static inline vm32_f_t
VM32_KFN (expm1) (vm32_f_t x, size_t vl)
{
  vm32_f_t y = VM32_NARROW (VM32_D64 (expm1) (VM32_WIDEN (x, vl), vl), vl);
  VM32_RAISE_UF_SUBNORMAL (y, vl);
  return y;
}
